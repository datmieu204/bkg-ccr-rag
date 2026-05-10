# BKG-CCR RAG: Biomedical Knowledge Graph with Contextual Contrastive Reasoning RAG

## Abstract

BKG-CCR RAG proposes a multimodal alignment paradigm for biomedical knowledge graphs that couples structured relational evidence with textual and sequence-derived semantics. The framework constructs graph-centric subviews anchored on target entities, encodes heterogeneous topology with relation-aware graph encoders, and fuses language and biological sequence signals into a unified representation. A contrastive objective aligns graph and text/sequence embeddings in a shared latent space, enabling robust retrieval and reasoning under incomplete or noisy knowledge graphs. This approach emphasizes cross-view consistency, hard negative discrimination, and scalable training to improve downstream RAG fidelity.

## Core Idea

The system learns cross-modal consistency between (i) a graph view derived from heterogeneous relations and (ii) a text/sequence view derived from entity summaries and biological sequences. Each view is projected into a common embedding space and optimized with a symmetric contrastive loss. Memory bank sampling and hard negative mining can further sharpen decision boundaries, yielding more discriminative alignments across modalities.

## Project Architecture

```
BKG-CCR RAG
|-- Data Layer
|   |-- Knowledge graph in Neo4j
|   |-- Subgraph extraction (neighbor hops, relation filtering)
|   `-- Anchor node selection
|-- Representation Layer
|   |-- Text and sequence encoding
|   |-- Heterogeneous graph encoding
|   `-- Modality fusion (gated or attention)
|-- Alignment Layer
|   |-- Projection to shared space
|   |-- Normalization
|   `-- Contrastive objective
`-- Training and Evaluation
		|-- Memory bank and hard negatives (optional)
		`-- Retrieval metrics (Recall@K, MRR)
```

## Graph Construction Workflow

1) Load the biomedical knowledge graph into Neo4j.

2) For each target entity, extract a subgraph by expanding neighbors up to a configurable hop count and restricting to relevant relation types.

3) Mark anchor nodes belonging to the target entity's original subgraph to prioritize them during pooling.

4) Construct two semantic views:
    - Text view from summaries and entity descriptions.
    - Sequence view from biological sequences grouped by entity type.

## Alignment and Training Workflow

1) Encode the text/sequence view with transformer encoders and aggregate into a single textual embedding per subgraph.

2) Encode the graph view with a heterogeneous graph encoder (HGT or RGCN) and pool node representations with anchor-aware pooling.

3) Fuse textual and sequence embeddings (gated or attention fusion).

4) Project text and graph embeddings into a shared latent space and apply normalization.

5) Optimize a symmetric InfoNCE objective. Optional memory banks and hard negatives improve discrimination under small batch sizes.

## How to Run (Docker)

1) Start Neo4j

```bash
docker compose up -d neo4j
```

2) Build the image

```bash
docker build -t bkg-ccr-rag .
```

3) Train

```bash
docker run --rm --network host bkg-ccr-rag \
	--uri bolt://localhost:25505 \
	--user neo4j \
	--password example_password \
	--config src/configs/config.yaml \
	--settings src/configs/settings.yaml \
	--database neo4j \
	--split top \
	--batch-size 8 \
	--epochs 10 \
	--learning-rate 3.0e-5 \
	--weight-decay 1.0e-4 \
	--device cuda \
	--grid "encoders.graph.type=hgt,rgcn;fusion.type=gated,attention" \
	--run-name run \
	--use-wandb \
	--wandb-project bkg-ccr-rag \
	--wandb-entity your-entity \
	--wandb-group exp-1 \
	--wandb-tags "contrastive,kg" \
	--checkpoint-dir checkpoints/run \
	--save-every 200 \
	--resume checkpoints/run/last.pt \
	--early-stop-patience 3 \
	--early-stop-delta 0.0 \
	--log-interval 20 \
	--amp \
	--grad-accum-steps 1 \
	--memory-bank-size 2048 \
	--hard-neg-k 64 \
	--learnable-temperature \
	--temperature-min 0.01 \
	--temperature-max 0.5 \
	--freeze-text-epochs 1 \
	--val-fraction 0.1 \
	--val-seed 42 \
	--val-max-batches 0 \
	--val-k 10
```

## How to Run (Local)

1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
python - <<'PY'
import pathlib
import subprocess
import sys
import tomllib

data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
deps = data.get("project", {}).get("dependencies", [])
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
PY
```

3) Train

```bash
python -m src.train \
	--uri bolt://localhost:25505 \
	--user neo4j \
	--password example_password \
	--config src/configs/config.yaml \
	--settings src/configs/settings.yaml \
	--database neo4j \
	--split top \
	--batch-size 8 \
	--epochs 10 \
	--learning-rate 3.0e-5 \
	--weight-decay 1.0e-4 \
	--device cuda \
	--grid "encoders.graph.type=hgt,rgcn;fusion.type=gated,attention" \
	--run-name run \
	--use-wandb \
	--wandb-project bkg-ccr-rag \
	--wandb-entity your-entity \
	--wandb-group exp-1 \
	--wandb-tags "contrastive,kg" \
	--checkpoint-dir checkpoints/run \
	--save-every 200 \
	--resume checkpoints/run/last.pt \
	--early-stop-patience 3 \
	--early-stop-delta 0.0 \
	--log-interval 20 \
	--amp \
	--grad-accum-steps 1 \
	--memory-bank-size 2048 \
	--hard-neg-k 64 \
	--learnable-temperature \
	--temperature-min 0.01 \
	--temperature-max 0.5 \
	--freeze-text-epochs 1 \
	--val-fraction 0.1 \
	--val-seed 42 \
	--val-max-batches 0 \
	--val-k 10
```

## Notes

- Neo4j must be populated before training.
- Update connection credentials and ports to match your environment.

## Author

Nguyen Tuan Dat

## License

MIT