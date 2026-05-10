import argparse
import itertools
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from src.data_loader.dataset import BioMedDataset
from src.models.fusion import AttentionFusion, GatedFusion
from src.models.graph_encoders import HGTEncoder, RGCNEncoder
from src.models.multi_view_model import MultiViewModel
from src.models.projection_head import ProjectionHead
from src.models.text_encoders import EncoderConfig, TransformerTextEncoder
from src.training.trainer import MultiViewTrainer


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _collate_identity(batch):
    return batch


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    module.train(requires_grad)
    for param in module.parameters():
        param.requires_grad = requires_grad


def _prepare_batch(batch: Sequence[Dict], device: torch.device) -> Dict[str, Any]:
    textual_batches = [item["text_view"]["textual_content"] for item in batch]
    sequence_batches = [item["text_view"]["sequence_content"] for item in batch]
    graph_list = [item["graph_view"] for item in batch]
    graph_batch = Batch.from_data_list(graph_list).to(device)
    return {
        "textual_batches": textual_batches,
        "sequence_batches": sequence_batches,
        "graph_batch": graph_batch,
    }


def _parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    if not raw:
        return values
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def _make_train_val_subsets(
    dataset: torch.utils.data.Dataset,
    fraction: float,
    seed: int,
) -> Tuple[torch.utils.data.Dataset, Optional[Subset]]:
    if fraction <= 0:
        return dataset, None
    total = len(dataset)
    if total == 0:
        return dataset, None
    val_size = max(1, int(total * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        train_indices = val_indices
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    return train_subset, val_subset


def _evaluate_retrieval(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
    ks: List[int],
) -> Dict[str, float]:
    model.eval()
    text_embeddings: List[torch.Tensor] = []
    graph_embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader, start=1):
            if max_batches > 0 and idx > max_batches:
                break
            prepared = _prepare_batch(batch, device)
            outputs = model(
                prepared["textual_batches"],
                prepared["sequence_batches"],
                prepared["graph_batch"],
            )
            text_embeddings.append(outputs["z_text"].detach().cpu())
            graph_embeddings.append(outputs["z_graph"].detach().cpu())
    if not text_embeddings:
        return {}

    z_text = torch.cat(text_embeddings, dim=0)
    z_graph = torch.cat(graph_embeddings, dim=0)
    sims = z_text @ z_graph.t()
    metrics: Dict[str, float] = {}

    if ks:
        for k in ks:
            k = min(k, sims.size(1))
            topk = sims.topk(k, dim=1).indices
            target = torch.arange(sims.size(0)).unsqueeze(1)
            hits = (topk == target).any(dim=1).float().mean().item()
            metrics[f"recall@{k}"] = hits

    sorted_idx = sims.argsort(dim=1, descending=True)
    target = torch.arange(sims.size(0)).unsqueeze(1)
    matches = sorted_idx == target
    ranks = matches.float().argmax(dim=1) + 1
    metrics["mrr"] = (1.0 / ranks.float()).mean().item()
    return metrics


def _save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_loss: Optional[float],
    config: Dict[str, Any],
    args: Dict[str, Any],
) -> None:
    _ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_loss": best_loss,
            "config": config,
            "args": args,
        },
        path,
    )


def _load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, int, Optional[float]]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    epoch = int(payload.get("epoch", 0))
    step = int(payload.get("step", 0))
    best_loss = payload.get("best_loss")
    return epoch, step, best_loss


def _parse_grid(value: str) -> Dict[str, List[str]]:
    grid: Dict[str, List[str]] = {}
    if not value:
        return grid
    for group in value.split(";"):
        group = group.strip()
        if not group:
            continue
        if "=" not in group:
            raise ValueError(f"Invalid grid segment: {group}")
        key, raw_values = group.split("=", 1)
        values = [v.strip() for v in raw_values.split(",") if v.strip()]
        if not values:
            raise ValueError(f"Grid key '{key}' has no values")
        grid[key.strip()] = values
    return grid


def _expand_grid(grid: Dict[str, List[str]]) -> List[Dict[str, str]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = itertools.product(*(grid[key] for key in keys))
    return [dict(zip(keys, combo)) for combo in combos]


def _set_nested(config: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _convert_value(raw: str) -> Any:
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    return raw


def _apply_overrides(
    config: Dict[str, Any],
    run_args: Dict[str, Any],
    overrides: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = {**config}
    updated_args = {**run_args}

    for key, raw_value in overrides.items():
        value = _convert_value(raw_value)
        if "." in key:
            _set_nested(config, key, value)
        else:
            updated_args[key] = value
    return config, updated_args


def build_model(config: Dict[str, Any], graph_metadata: Optional[Tuple[List[str], List[Tuple[str, str, str]]]] = None) -> MultiViewModel:
    textual_cfg = EncoderConfig(
        model_name=_get(config, "encoders", "textual", "model_name", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
        pooling=_get(config, "encoders", "textual", "pooling", default="cls"),
        max_length=int(_get(config, "encoders", "textual", "max_length", default=256)),
        batch_size=int(_get(config, "encoders", "textual", "batch_size", default=4)),
    )
    sequence_cfg = EncoderConfig(
        model_name=_get(config, "encoders", "sequence", "model_name", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"),
        pooling=_get(config, "encoders", "sequence", "pooling", default="cls"),
        max_length=int(_get(config, "encoders", "sequence", "max_length", default=64)),
        batch_size=int(_get(config, "encoders", "sequence", "batch_size", default=8)),
    )

    textual_encoder = TransformerTextEncoder(textual_cfg)
    sequence_encoder = TransformerTextEncoder(sequence_cfg)

    graph_type = _get(config, "encoders", "graph", "type", default="hgt")
    hidden_dim = int(_get(config, "encoders", "graph", "hidden_dim", default=256))
    num_layers = int(_get(config, "encoders", "graph", "num_layers", default=2))
    num_heads = int(_get(config, "encoders", "graph", "num_heads", default=4))
    dropout = float(_get(config, "encoders", "graph", "dropout", default=0.1))

    if graph_type == "rgcn":
        graph_encoder = RGCNEncoder(
            in_dim=None, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout
        )
    else:
        graph_encoder = HGTEncoder(
            in_dim=None,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            fixed_metadata=graph_metadata,
        )

    fusion_type = _get(config, "fusion", "type", default="gated")
    if fusion_type == "attention":
        fusion = AttentionFusion(dim=textual_encoder.hidden_size)
    else:
        fusion = GatedFusion(dim=textual_encoder.hidden_size)

    proj_hidden = int(_get(config, "projection", "hidden_dim", default=256))
    proj_out = int(_get(config, "projection", "out_dim", default=128))

    text_projection = ProjectionHead(
        in_dim=textual_encoder.hidden_size,
        hidden_dim=proj_hidden,
        out_dim=proj_out,
    )
    graph_projection = ProjectionHead(
        in_dim=hidden_dim,
        hidden_dim=proj_hidden,
        out_dim=proj_out,
    )

    return MultiViewModel(
        textual_encoder=textual_encoder,
        sequence_encoder=sequence_encoder,
        graph_encoder=graph_encoder,
        fusion=fusion,
        text_projection=text_projection,
        graph_projection=graph_projection,
    )


def _run_training(
    config: Dict[str, Any],
    args: Dict[str, Any],
    run_name: str,
    use_wandb: bool,
    wandb_meta: Dict[str, Any],
) -> None:
    device_name = args["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = BioMedDataset(
        root=os.getcwd(),
        uri=args["uri"],
        user=args["user"],
        password=args["password"],
        database=args.get("database"),
        split=args["split"],
        neighbor_hops=int(_get(config, "data", "neighbor_hops", default=2)),
        neighbor_rel_types=_get(config, "data", "neighbor_rel_types", default=None),
        include_summary=bool(_get(config, "data", "include_summary", default=True)),
        include_relation_texts=bool(
            _get(config, "data", "include_relation_texts", default=False)
        ),
        max_texts=int(_get(config, "data", "max_texts", default=200)),
    )

    val_fraction = float(args.get("val_fraction", 0.0))
    val_seed = int(args.get("val_seed", 42))
    val_max_batches = int(args.get("val_max_batches", 0))
    val_k = _parse_int_list(args.get("val_k", "1,5,10"))
    train_dataset, val_dataset = _make_train_val_subsets(dataset, val_fraction, val_seed)

    dataloader = DataLoader(
        train_dataset,
        batch_size=int(args["batch_size"]),
        shuffle=True,
        collate_fn=_collate_identity,
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=int(args["batch_size"]),
            shuffle=False,
            collate_fn=_collate_identity,
        )

    graph_metadata = None
    if _get(config, "encoders", "graph", "type", default="hgt") == "hgt":
        graph_metadata = dataset.loader.fetch_hgt_metadata()
    model = build_model(config, graph_metadata=graph_metadata)
    optimizer = AdamW(
        model.parameters(),
        lr=float(args["learning_rate"]),
        weight_decay=float(args["weight_decay"]),
    )

    temperature = float(_get(config, "loss", "temperature", default=0.07))
    memory_bank_size = int(args.get("memory_bank_size", 0))
    hard_neg_k = int(args.get("hard_neg_k", 0))
    learnable_temperature = bool(args.get("learnable_temperature", False))
    temperature_min = float(args.get("temperature_min", 0.01))
    temperature_max = float(args.get("temperature_max", 0.5))
    trainer = MultiViewTrainer(
        model=model,
        optimizer=optimizer,
        temperature=temperature,
        memory_bank_size=memory_bank_size,
        hard_neg_k=hard_neg_k,
        learnable_temperature=learnable_temperature,
        temperature_min=temperature_min,
        temperature_max=temperature_max,
        device=device,
    )
    use_amp = bool(args.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum_steps = int(args.get("grad_accum_steps", 1))

    wandb_run = None
    if use_wandb:
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("wandb is not installed") from exc
        wandb_run = wandb.init(
            project=wandb_meta.get("project"),
            entity=wandb_meta.get("entity"),
            name=run_name,
            group=wandb_meta.get("group"),
            tags=wandb_meta.get("tags"),
            config={"config": config, "args": args},
        )

    checkpoint_dir = args.get("checkpoint_dir") or os.path.join("checkpoints", run_name)
    _ensure_dir(checkpoint_dir)
    checkpoint_last = os.path.join(checkpoint_dir, "last.pt")
    checkpoint_best = os.path.join(checkpoint_dir, "best.pt")

    start_epoch = 1
    global_step = 0
    best_loss = None
    if args.get("resume"):
        resume_path = args["resume"]
        if os.path.exists(resume_path):
            resume_epoch, resume_step, best_loss = _load_checkpoint(
                resume_path, model, optimizer, device
            )
            start_epoch = resume_epoch + 1
            global_step = resume_step
            print(f"Resumed from {resume_path} (epoch={resume_epoch}, step={resume_step})")
        else:
            print(f"Resume checkpoint not found: {resume_path}")

    log_interval = int(args.get("log_interval", 20))
    save_every = int(args.get("save_every", 0))
    patience = int(args.get("early_stop_patience", 0))
    min_delta = float(args.get("early_stop_delta", 0.0))
    patience_counter = 0

    freeze_text_epochs = int(args.get("freeze_text_epochs", 0))
    text_frozen = None
    for epoch in range(start_epoch, int(args["epochs"]) + 1):
        if freeze_text_epochs > 0:
            should_freeze = epoch <= freeze_text_epochs
            if text_frozen != should_freeze:
                _set_requires_grad(model.textual_encoder.model, not should_freeze)
                _set_requires_grad(model.sequence_encoder.model, not should_freeze)
                text_frozen = should_freeze
                state = "frozen" if should_freeze else "unfrozen"
                print(f"Run {run_name} | Epoch {epoch}: text encoders {state}")
        total_loss = 0.0
        steps = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader, start=1):
            loss = trainer.train_step_scaled(
                batch,
                scaler=scaler if use_amp else None,
                grad_accum_steps=grad_accum_steps,
            )
            total_loss += loss * max(grad_accum_steps, 1)
            steps += 1

            if batch_idx % grad_accum_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if log_interval > 0 and global_step % log_interval == 0:
                print(
                    f"Run {run_name} | Epoch {epoch} | Step {global_step}: loss={loss:.4f}"
                )
            if wandb_run is not None:
                wandb_run.log({"step": global_step, "loss_step": loss})

            if save_every > 0 and global_step % save_every == 0:
                _save_checkpoint(
                    checkpoint_last,
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    best_loss,
                    config,
                    args,
                )
        avg_loss = total_loss / max(steps, 1)
        print(f"Run {run_name} | Epoch {epoch}: loss={avg_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "loss": avg_loss})

        if val_dataloader is not None:
            metrics = _evaluate_retrieval(
                model,
                val_dataloader,
                device,
                max_batches=val_max_batches,
                ks=val_k,
            )
            if metrics:
                metrics_line = " | ".join(
                    f"{name}={value:.4f}" for name, value in metrics.items()
                )
                print(f"Run {run_name} | Epoch {epoch}: {metrics_line}")
                if wandb_run is not None:
                    wandb_run.log({"epoch": epoch, **metrics})

        _save_checkpoint(
            checkpoint_last,
            model,
            optimizer,
            epoch,
            global_step,
            best_loss,
            config,
            args,
        )

        is_best = best_loss is None or avg_loss < (best_loss - min_delta)
        if is_best:
            best_loss = avg_loss
            patience_counter = 0
            _save_checkpoint(
                checkpoint_best,
                model,
                optimizer,
                epoch,
                global_step,
                best_loss,
                config,
                args,
            )
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch} (best_loss={best_loss:.4f})"
                )
                break

    if wandb_run is not None:
        wandb_run.finish()
    dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-view contrastive model")
    parser.add_argument("--config", default="src/configs/config.yaml")
    parser.add_argument("--settings", default="src/configs/settings.yaml")
    parser.add_argument("--uri", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default=None)
    parser.add_argument("--split", default="top")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid", default="", help="Grid search like key=a,b;other=1,2")
    parser.add_argument("--run-name", default="run", help="Base run name")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-delta", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--memory-bank-size", type=int, default=0)
    parser.add_argument("--hard-neg-k", type=int, default=0)
    parser.add_argument("--learnable-temperature", action="store_true")
    parser.add_argument("--temperature-min", type=float, default=0.01)
    parser.add_argument("--temperature-max", type=float, default=0.5)
    parser.add_argument("--freeze-text-epochs", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--val-seed", type=int, default=42)
    parser.add_argument("--val-max-batches", type=int, default=0)
    parser.add_argument("--val-k", default="1,5,10")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    settings = _load_yaml(args.settings)

    base_args = {
        "uri": args.uri,
        "user": args.user,
        "password": args.password,
        "database": args.database,
        "split": args.split,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "checkpoint_dir": args.checkpoint_dir,
        "save_every": args.save_every,
        "resume": args.resume,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_delta": args.early_stop_delta,
        "log_interval": args.log_interval,
        "amp": args.amp,
        "grad_accum_steps": args.grad_accum_steps,
        "memory_bank_size": args.memory_bank_size,
        "hard_neg_k": args.hard_neg_k,
        "learnable_temperature": args.learnable_temperature,
        "temperature_min": args.temperature_min,
        "temperature_max": args.temperature_max,
        "freeze_text_epochs": args.freeze_text_epochs,
        "val_fraction": args.val_fraction,
        "val_seed": args.val_seed,
        "val_max_batches": args.val_max_batches,
        "val_k": args.val_k,
    }

    grid = _parse_grid(args.grid)
    runs = _expand_grid(grid)
    tags = [tag for tag in args.wandb_tags.split(",") if tag] if args.wandb_tags else None
    wandb_meta = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "tags": tags,
    }

    for idx, overrides in enumerate(runs, start=1):
        run_config, run_args = _apply_overrides(config, base_args, overrides)
        run_name = f"{args.run_name}-{idx}" if len(runs) > 1 else args.run_name
        _run_training(
            run_config,
            run_args,
            run_name=run_name,
            use_wandb=args.use_wandb,
            wandb_meta=wandb_meta,
        )


if __name__ == "__main__":
    main()
