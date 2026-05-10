from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.optim import Optimizer
from torch_geometric.data import Batch

from .losses import LearnableTemperature, MemoryBank, info_nce_loss, info_nce_loss_with_memory


class MultiViewTrainer:
	def __init__(
		self,
		model: nn.Module,
		optimizer: Optimizer,
		temperature: float = 0.07,
		memory_bank_size: int = 0,
		hard_neg_k: int = 0,
		learnable_temperature: bool = False,
		temperature_min: float = 0.01,
		temperature_max: float = 0.5,
		device: torch.device = torch.device("cpu"),
	) -> None:
		self.model = model.to(device)
		self.optimizer = optimizer
		self.temperature = temperature
		self.device = device
		self.memory_bank_size = int(memory_bank_size)
		self.hard_neg_k = int(hard_neg_k)
		self.text_bank: Optional[MemoryBank] = None
		self.graph_bank: Optional[MemoryBank] = None
		self.temperature_module: Optional[LearnableTemperature] = None
		if learnable_temperature:
			self.temperature_module = LearnableTemperature(
				init_temperature=temperature,
				min_temperature=temperature_min,
				max_temperature=temperature_max,
			).to(device)
			self.optimizer.add_param_group(
				{"params": self.temperature_module.parameters()}
			)

	def _prepare_batch(self, batch: Sequence[Dict]) -> Dict:
		textual_batches = [item["text_view"]["textual_content"] for item in batch]
		sequence_batches = [item["text_view"]["sequence_content"] for item in batch]
		graph_list = [item["graph_view"] for item in batch]
		graph_batch = Batch.from_data_list(graph_list).to(self.device)

		return {
			"textual_batches": textual_batches,
			"sequence_batches": sequence_batches,
			"graph_batch": graph_batch,
		}

	def train_step(self, batch: Sequence[Dict]) -> float:
		return self.train_step_scaled(batch, scaler=None, grad_accum_steps=1)

	def train_step_scaled(
		self,
		batch: Sequence[Dict],
		scaler: Optional[torch.cuda.amp.GradScaler],
		grad_accum_steps: int = 1,
	) -> float:
		self.model.train()
		prepared = self._prepare_batch(batch)
		use_amp = scaler is not None

		if use_amp:
			with torch.autocast(device_type=self.device.type, enabled=True):
				outputs = self.model(
					prepared["textual_batches"],
					prepared["sequence_batches"],
					prepared["graph_batch"],
				)
		else:
			outputs = self.model(
				prepared["textual_batches"],
				prepared["sequence_batches"],
				prepared["graph_batch"],
			)

		if self.memory_bank_size > 0 and self.text_bank is None:
			self.text_bank = MemoryBank(
				size=self.memory_bank_size,
				dim=outputs["z_text"].size(1),
				device=outputs["z_text"].device,
			)
			self.graph_bank = MemoryBank(
				size=self.memory_bank_size,
				dim=outputs["z_graph"].size(1),
				device=outputs["z_graph"].device,
			)

		logit_scale = None
		if self.temperature_module is not None:
			logit_scale = self.temperature_module()

		if self.text_bank is None:
			loss = info_nce_loss(
				outputs["z_text"],
				outputs["z_graph"],
				temperature=self.temperature,
				symmetric=True,
				logit_scale=logit_scale,
			)
		else:
			loss = info_nce_loss_with_memory(
				outputs["z_text"],
				outputs["z_graph"],
				temperature=self.temperature,
				symmetric=True,
				graph_bank=self.graph_bank,
				text_bank=self.text_bank,
				hard_neg_k=self.hard_neg_k,
				logit_scale=logit_scale,
			)

		loss = loss / max(grad_accum_steps, 1)
		if use_amp:
			scaler.scale(loss).backward()
		else:
			loss.backward()

		if self.text_bank is not None:
			self.text_bank.enqueue(outputs["z_text"])
			self.graph_bank.enqueue(outputs["z_graph"])
		return float(loss.detach().cpu().item())
