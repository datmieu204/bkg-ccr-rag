from typing import Optional

import torch
from torch import nn


class GatedFusion(nn.Module):
	def __init__(self, dim: int) -> None:
		super().__init__()
		self.gate = nn.Linear(dim * 2, dim)

	def forward(self, textual: torch.Tensor, sequence: Optional[torch.Tensor]) -> torch.Tensor:
		if sequence is None:
			return textual
		if textual.shape != sequence.shape:
			raise ValueError("Textual and sequence embeddings must share the same shape")
		gate = torch.sigmoid(self.gate(torch.cat([textual, sequence], dim=-1)))
		return gate * textual + (1.0 - gate) * sequence


class AttentionFusion(nn.Module):
	def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
		super().__init__()
		self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

	def forward(self, textual: torch.Tensor, sequence: Optional[torch.Tensor]) -> torch.Tensor:
		if sequence is None:
			return textual
		query = textual.unsqueeze(1)
		key = sequence.unsqueeze(1)
		value = sequence.unsqueeze(1)
		attended, _ = self.attn(query=query, key=key, value=value)
		return attended.squeeze(1)
