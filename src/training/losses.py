import torch
from torch import nn
from torch.nn import functional as F


class MemoryBank:
	def __init__(self, size: int, dim: int, device: torch.device) -> None:
		self.size = int(size)
		self.dim = int(dim)
		self.device = device
		self.bank = torch.zeros(self.size, self.dim, device=self.device)
		self.ptr = 0
		self.full = False

	def enqueue(self, x: torch.Tensor) -> None:
		if x is None:
			return
		x = x.detach()
		if x.dim() == 1:
			x = x.unsqueeze(0)
		for row in x:
			self.bank[self.ptr] = row
			self.ptr = (self.ptr + 1) % self.size
			if self.ptr == 0:
				self.full = True

	def get(self) -> torch.Tensor | None:
		if self.full:
			return self.bank
		if self.ptr == 0:
			return None
		return self.bank[: self.ptr]


class LearnableTemperature(nn.Module):
	def __init__(
		self,
		init_temperature: float = 0.07,
		min_temperature: float = 0.01,
		max_temperature: float = 0.5,
	) -> None:
		super().__init__()
		init_scale = 1.0 / max(init_temperature, 1.0e-6)
		self.logit_scale = nn.Parameter(torch.tensor(init_scale).log())
		self.min_scale = 1.0 / max(max_temperature, 1.0e-6)
		self.max_scale = 1.0 / max(min_temperature, 1.0e-6)

	def forward(self) -> torch.Tensor:
		scale = self.logit_scale.exp()
		return scale.clamp(self.min_scale, self.max_scale)


def _apply_temperature(
	logits: torch.Tensor,
	temperature: float,
	logit_scale: torch.Tensor | None,
) -> torch.Tensor:
	if logit_scale is not None:
		return logits * logit_scale
	return logits / temperature


def _select_hard_negatives(
	neg_logits: torch.Tensor,
	hard_neg_k: int,
) -> torch.Tensor:
	if hard_neg_k <= 0 or hard_neg_k >= neg_logits.size(1):
		return neg_logits
	return neg_logits.topk(hard_neg_k, dim=1).values


def info_nce_loss(
	z_text: torch.Tensor,
	z_graph: torch.Tensor,
	temperature: float = 0.07,
	symmetric: bool = True,
	logit_scale: torch.Tensor | None = None,
) -> torch.Tensor:
	if z_text.size(0) != z_graph.size(0):
		raise ValueError("Batch size mismatch between text and graph embeddings")
	logits = _apply_temperature(z_text @ z_graph.t(), temperature, logit_scale)
	labels = torch.arange(z_text.size(0), device=z_text.device)
	loss_text = F.cross_entropy(logits, labels)
	if not symmetric:
		return loss_text
	loss_graph = F.cross_entropy(logits.t(), labels)
	return 0.5 * (loss_text + loss_graph)


def info_nce_loss_with_memory(
	z_text: torch.Tensor,
	z_graph: torch.Tensor,
	temperature: float = 0.07,
	symmetric: bool = True,
	graph_bank: MemoryBank | None = None,
	text_bank: MemoryBank | None = None,
	hard_neg_k: int = 0,
	logit_scale: torch.Tensor | None = None,
) -> torch.Tensor:
	if z_text.size(0) != z_graph.size(0):
		raise ValueError("Batch size mismatch between text and graph embeddings")
	pos_logits = z_text @ z_graph.t()
	logits = pos_logits
	if graph_bank is not None:
		neg_graph = graph_bank.get()
		if neg_graph is not None:
			neg_logits = z_text @ neg_graph.t()
			neg_logits = _select_hard_negatives(neg_logits, hard_neg_k)
			logits = torch.cat([logits, neg_logits], dim=1)
	logits = _apply_temperature(logits, temperature, logit_scale)
	labels = torch.arange(z_text.size(0), device=z_text.device)
	loss_text = F.cross_entropy(logits, labels)

	if not symmetric:
		return loss_text

	logits_t = z_graph @ z_text.t()
	if text_bank is not None:
		neg_text = text_bank.get()
		if neg_text is not None:
			neg_logits = z_graph @ neg_text.t()
			neg_logits = _select_hard_negatives(neg_logits, hard_neg_k)
			logits_t = torch.cat([logits_t, neg_logits], dim=1)
	logits_t = _apply_temperature(logits_t, temperature, logit_scale)
	labels_t = torch.arange(z_graph.size(0), device=z_graph.device)
	loss_graph = F.cross_entropy(logits_t, labels_t)
	return 0.5 * (loss_text + loss_graph)