from typing import Dict, Sequence

import torch
from torch import nn
from torch_geometric.data import Batch

from .losses import info_nce_loss


class MultiViewEvaluator:
	def __init__(
		self,
		model: nn.Module,
		temperature: float = 0.07,
		device: torch.device = torch.device("cpu"),
	) -> None:
		self.model = model.to(device)
		self.temperature = temperature
		self.device = device

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

	def evaluate_step(self, batch: Sequence[Dict]) -> float:
		self.model.eval()
		with torch.no_grad():
			prepared = self._prepare_batch(batch)
			outputs = self.model(
				prepared["textual_batches"],
				prepared["sequence_batches"],
				prepared["graph_batch"],
			)
			loss = info_nce_loss(
				outputs["z_text"],
				outputs["z_graph"],
				temperature=self.temperature,
				symmetric=True,
			)
		return float(loss.detach().cpu().item())
