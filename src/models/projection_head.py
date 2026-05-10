import torch
from torch import nn


class ProjectionHead(nn.Module):
	def __init__(
		self,
		in_dim: int,
		hidden_dim: int,
		out_dim: int,
		dropout: float = 0.1,
	) -> None:
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, out_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.mlp(x)
