from typing import Optional

import torch
from torch import nn


class AttributeFusion(nn.Module):
	def __init__(self, text_dim: int, graph_dim: int, out_dim: int) -> None:
		super().__init__()
		self.proj = nn.Linear(text_dim + graph_dim, out_dim)

	def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
		return self.proj(torch.cat([text_emb, graph_emb], dim=-1))


class CrossAttentionAlignment(nn.Module):
	def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
		super().__init__()
		self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

	def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
		query = text_emb.unsqueeze(1)
		key = graph_emb.unsqueeze(1)
		value = graph_emb.unsqueeze(1)
		attended, _ = self.attn(query=query, key=key, value=value)
		return attended.squeeze(1)
