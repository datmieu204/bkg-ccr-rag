
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, RGCNConv, global_mean_pool


def _pool_by_type(
	data: HeteroData,
	x_dict: Dict[str, torch.Tensor],
	num_graphs: int,
) -> torch.Tensor:
	pooled = []
	for node_type, x in x_dict.items():
		if x.numel() == 0:
			continue
		mask = data[node_type].get("is_anchor")
		if mask is not None:
			mask = mask.bool()
			if mask.any():
				x = x[mask]

		if hasattr(data[node_type], "batch"):
			batch = data[node_type].batch
			if not torch.is_tensor(batch):
				batch = torch.tensor(batch, device=x.device)
			if batch.numel() == x.size(0):
				actual_size = max(num_graphs, int(batch.max().item()) + 1)
				p = global_mean_pool(x, batch, size=actual_size)
				pooled.append(p[:num_graphs])

				# pooled.append(global_mean_pool(x, batch, size=num_graphs))
			else:
				pooled.append(x.mean(dim=0, keepdim=True).repeat(num_graphs, 1))
		else:
			pooled.append(x.mean(dim=0, keepdim=True).repeat(num_graphs, 1))

	if not pooled:
		if x_dict:
			any_x = next(iter(x_dict.values()))
			return torch.zeros(num_graphs, any_x.size(-1), device=any_x.device)
		return torch.zeros(num_graphs, 1)

	if pooled[0].dim() == 1:
		pooled = [p.unsqueeze(0) for p in pooled]
	stacked = torch.stack(pooled, dim=0)
	return stacked.mean(dim=0)


class HGTEncoder(nn.Module):
	def __init__(
		self,
		in_dim: Optional[int] = None,
		hidden_dim: int = 256,
		num_layers: int = 2,
		num_heads: int = 4,
		dropout: float = 0.1,
		fixed_metadata: Optional[Tuple[List[str], List[Tuple[str, str, str]]]] = None,
	) -> None:
		super().__init__()
		self.in_dim = in_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.dropout = dropout
		self._fixed_metadata = fixed_metadata
		self._metadata = fixed_metadata
		self.convs = nn.ModuleList()
		self.type_embeddings = nn.ParameterDict()
		self.input_proj = nn.ModuleDict()
		self.output_proj = None

	def _project_output(self, emb: torch.Tensor, device: torch.device) -> torch.Tensor:
		if emb.size(-1) == self.hidden_dim:
			return emb
		if self.output_proj is None:
			self.output_proj = nn.Linear(emb.size(-1), self.hidden_dim).to(device)
		return self.output_proj(emb)

	def _ensure_type_embeddings(self, node_types, device: torch.device) -> None:
		for node_type in node_types:
			if node_type not in self.type_embeddings:
				self.type_embeddings[node_type] = nn.Parameter(
					torch.zeros(self.in_dim, dtype=torch.float, device=device)
				)

	def _build_layers(self, metadata, device: torch.device) -> None:
		if self.convs:
			return
		self._metadata = metadata
		if self.in_dim is None:
			self.in_dim = self.hidden_dim
		for layer_idx in range(self.num_layers):
			in_channels = self.in_dim if layer_idx == 0 else self.hidden_dim
			self.convs.append(
				HGTConv(
					in_channels=in_channels,
					out_channels=self.hidden_dim,
					metadata=metadata,
					heads=self.num_heads,
				)
			)
		self.convs.to(device)

	def _reset_layers(self) -> None:
		self._metadata = self._fixed_metadata
		self.convs = nn.ModuleList()

	def _prepare_inputs(self, data: HeteroData, node_types: Optional[list] = None) -> Dict[str, torch.Tensor]:
		x_dict: Dict[str, torch.Tensor] = {}
		device = None
		for node_type in data.node_types:
			if hasattr(data[node_type], "x"):
				device = data[node_type].x.device
				break
		if device is None:
			device = torch.device("cpu")
		if self.in_dim is None:
			for node_type in data.node_types:
				if hasattr(data[node_type], "x"):
					self.in_dim = int(data[node_type].x.size(-1))
					break
			if self.in_dim is None:
				self.in_dim = self.hidden_dim

		node_types = node_types or list(data.node_types)
		self._ensure_type_embeddings(node_types, device)

		for node_type in node_types:
			if node_type not in data.node_types:
				x_dict[node_type] = torch.zeros((0, self.in_dim), device=device)
				continue
			if hasattr(data[node_type], "x"):
				x = data[node_type].x
				if x.size(-1) != self.in_dim:
					if node_type not in self.input_proj:
						self.input_proj[node_type] = nn.Linear(x.size(-1), self.in_dim)
						self.input_proj[node_type].to(device)
					x = self.input_proj[node_type](x)
			else:
				base = self.type_embeddings[node_type]
				num_nodes = data[node_type].num_nodes
				x = base.unsqueeze(0).repeat(num_nodes, 1)
			x_dict[node_type] = x
		return x_dict

	def forward(self, data: HeteroData) -> torch.Tensor:
		metadata = self._fixed_metadata or data.metadata()
		if self._fixed_metadata is None and self._metadata is not None and self._metadata != metadata:
			self._reset_layers()
		try:
			edge_index_dict_raw = data.edge_index_dict
		except KeyError:
			edge_index_dict_raw = {}
		if metadata is not None:
			node_types = list(metadata[0])
		else:
			node_types = set(data.node_types)
			for src, _, dst in edge_index_dict_raw.keys():
				node_types.add(src)
				node_types.add(dst)
			node_types = list(node_types)
		x_dict = self._prepare_inputs(data, node_types=node_types)
		if not self.convs:
			any_x = next(iter(x_dict.values()))
			self._build_layers(metadata, any_x.device)
		if metadata is not None:
			any_x = next(iter(x_dict.values()))
			for node_type in metadata[0]:
				if node_type not in x_dict:
					x_dict[node_type] = torch.zeros(
						(0, self.in_dim), device=any_x.device
					)
		num_graphs = int(getattr(data, "num_graphs", 1))
		for conv in self.convs:
			current_edge_index_dict = {
				key: edge_index
				for key, edge_index in edge_index_dict_raw.items()
				if key[0] in x_dict and key[2] in x_dict and x_dict[key[0]].size(0) > 0 and x_dict[key[2]].size(0) > 0
			}
			if metadata is not None:
				allowed_edge_types = set(metadata[1])
				current_edge_index_dict = {
					key: edge_index
					for key, edge_index in current_edge_index_dict.items()
					if key in allowed_edge_types
				}
			if not current_edge_index_dict:
				break
			x_dict = conv(x_dict, current_edge_index_dict)
			x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
		pooled = _pool_by_type(data, x_dict, num_graphs)
		return self._project_output(pooled, pooled.device)


class RGCNEncoder(nn.Module):
	def __init__(
		self,
		in_dim: Optional[int] = None,
		hidden_dim: int = 256,
		num_layers: int = 2,
		dropout: float = 0.1,
	) -> None:
		super().__init__()
		self.in_dim = in_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.dropout = dropout
		self.convs = nn.ModuleList()
		self.output_proj = None

	def _project_output(self, emb: torch.Tensor, device: torch.device) -> torch.Tensor:
		if emb.size(-1) == self.hidden_dim:
			return emb
		if self.output_proj is None:
			self.output_proj = nn.Linear(emb.size(-1), self.hidden_dim).to(device)
		return self.output_proj(emb)

	def forward(self, data: HeteroData) -> torch.Tensor:
		homo = data.to_homogeneous()
		x = homo.x
		device = x.device
		edge_index = getattr(homo, "edge_index", None)
		num_graphs = int(getattr(data, "num_graphs", 1))

		if edge_index is None or edge_index.numel() == 0:
			if hasattr(homo, "batch") and homo.batch is not None and homo.batch.numel() > 0:
				actual_size = max(num_graphs, int(homo.batch.max().item()) + 1)
				pooled = global_mean_pool(x, homo.batch, size=actual_size)[:num_graphs]
				# pooled = global_mean_pool(x, homo.batch, size=num_graphs)
			else:
				pooled = x.mean(dim=0, keepdim=True).repeat(num_graphs, 1)
			return self._project_output(pooled, device)
		if self.in_dim is None:
			self.in_dim = int(x.size(-1))
		if not self.convs:
			for layer_idx in range(self.num_layers):
				in_channels = self.in_dim if layer_idx == 0 else self.hidden_dim
				self.convs.append(
					RGCNConv(in_channels, self.hidden_dim, num_relations=1)
				)
			self.convs.to(device)
		if not hasattr(homo, "edge_type") or homo.edge_type.numel() == 0:
			num_relations = 1
		else:
			num_relations = int(homo.edge_type.max().item() + 1)
		if self.convs[0].num_relations != num_relations:
			self.convs = nn.ModuleList(
				[
					RGCNConv(
						self.in_dim if i == 0 else self.hidden_dim,
						self.hidden_dim,
						num_relations=num_relations,
					)
					for i in range(self.num_layers)
				]
			)
			self.convs.to(device)

		for conv in self.convs:
			x = conv(x, homo.edge_index, homo.edge_type)
			x = torch.relu(x)

		if hasattr(homo, "batch") and homo.batch is not None and homo.batch.numel() > 0:
			actual_size = max(num_graphs, int(homo.batch.max().item()) + 1)
			pooled = global_mean_pool(x, homo.batch, size=actual_size)[:num_graphs]
			# pooled = global_mean_pool(x, homo.batch, size=num_graphs)
		else:
			pooled = x.mean(dim=0, keepdim=True).repeat(num_graphs, 1)
		return self._project_output(pooled, device)