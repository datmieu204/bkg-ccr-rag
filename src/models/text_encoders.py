
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
	mask = attention_mask.unsqueeze(-1).float()
	masked = last_hidden * mask
	summed = masked.sum(dim=1)
	denom = mask.sum(dim=1).clamp(min=1.0)
	return summed / denom


@dataclass
class EncoderConfig:
	model_name: str
	pooling: str = "cls"
	max_length: int = 256
	batch_size: int = 8


class TransformerTextEncoder(nn.Module):
	def __init__(self, config: EncoderConfig) -> None:
		super().__init__()
		self.config = config
		self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
		self.model = AutoModel.from_pretrained(config.model_name)

	@property
	def hidden_size(self) -> int:
		return int(self.model.config.hidden_size)

	def device(self) -> torch.device:
		return next(self.model.parameters()).device

	def forward(self, texts: Sequence[str]) -> torch.Tensor:
		if not texts:
			return torch.zeros(0, self.hidden_size, device=self.device())

		batch_size = max(1, int(self.config.batch_size))
		all_outputs = []
		for start in range(0, len(texts), batch_size):
			chunk = list(texts[start : start + batch_size])
			tokens = self.tokenizer(
				chunk,
				padding=True,
				truncation=True,
				max_length=self.config.max_length,
				return_tensors="pt",
			)
			tokens = {key: value.to(self.device()) for key, value in tokens.items()}
			outputs = self.model(**tokens)
			if self.config.pooling == "mean":
				pooled = _mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
			else:
				pooled = outputs.last_hidden_state[:, 0]
			all_outputs.append(pooled)

		return torch.cat(all_outputs, dim=0)

	def encode_grouped(self, grouped_texts: Sequence[Sequence[str]]) -> torch.Tensor:
		if not grouped_texts:
			return torch.zeros(0, self.hidden_size, device=self.device())

		flat_texts: List[str] = []
		ranges: List[Tuple[int, int]] = []
		for group in grouped_texts:
			start = len(flat_texts)
			for text in group:
				if text:
					flat_texts.append(text)
			end = len(flat_texts)
			ranges.append((start, end))

		if not flat_texts:
			return torch.zeros(len(grouped_texts), self.hidden_size, device=self.device())

		# Mini-batching forward pass to avoid OOM with large flat_texts
		chunk_size = 32
		embedding_chunks = []
		for i in range(0, len(flat_texts), chunk_size):
			chunk_texts = flat_texts[i:i + chunk_size]
			chunk_emb = self.forward(chunk_texts)
			embedding_chunks.append(chunk_emb)
		embeddings = torch.cat(embedding_chunks, dim=0)

		pooled: List[torch.Tensor] = []
		for start, end in ranges:
			if start == end:
				pooled.append(torch.zeros(self.hidden_size, device=self.device()))
			else:
				pooled.append(embeddings[start:end].mean(dim=0))

		return torch.stack(pooled, dim=0)