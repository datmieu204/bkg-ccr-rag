from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, HeteroData

from .fusion import AttentionFusion, GatedFusion
from .graph_encoders import HGTEncoder, RGCNEncoder
from .projection_head import ProjectionHead
from .text_encoders import EncoderConfig, TransformerTextEncoder


class MultiViewModel(nn.Module):
	def __init__(
		self,
		textual_encoder: TransformerTextEncoder,
		sequence_encoder: TransformerTextEncoder,
		graph_encoder: nn.Module,
		fusion: nn.Module,
		text_projection: ProjectionHead,
		graph_projection: ProjectionHead,
	) -> None:
		super().__init__()
		self.textual_encoder = textual_encoder
		self.sequence_encoder = sequence_encoder
		self.graph_encoder = graph_encoder
		self.fusion = fusion
		self.text_projection = text_projection
		self.graph_projection = graph_projection

	def encode_textual(self, textual_batches: Sequence[Sequence[str]]) -> torch.Tensor:
		return self.textual_encoder.encode_grouped(textual_batches)

	def encode_sequence(self, sequence_batches: Sequence[Dict[str, List[str]]]) -> Optional[torch.Tensor]:
		if not sequence_batches:
			return None
		grouped_terms: List[List[str]] = []
		for sequence_map in sequence_batches:
			terms: List[str] = []
			for bucket_terms in sequence_map.values():
				terms.extend(bucket_terms)
			grouped_terms.append(terms)

		if all(len(group) == 0 for group in grouped_terms):
			return None
		return self.sequence_encoder.encode_grouped(grouped_terms)

	def encode_graph(self, graph_batch: HeteroData) -> torch.Tensor:
		return self.graph_encoder(graph_batch)

	def forward(
		self,
		textual_batches: Sequence[Sequence[str]],
		sequence_batches: Sequence[Dict[str, List[str]]],
		graph_batch: HeteroData,
	) -> Dict[str, torch.Tensor]:
		text_emb = self.encode_textual(textual_batches)
		seq_emb = self.encode_sequence(sequence_batches)
		fused_text = self.fusion(text_emb, seq_emb)
		graph_emb = self.encode_graph(graph_batch)

		z_text = self.text_projection(fused_text)
		z_graph = self.graph_projection(graph_emb)

		z_text = F.normalize(z_text, dim=-1)
		z_graph = F.normalize(z_graph, dim=-1)

		return {
			"text": fused_text,
			"graph": graph_emb,
			"z_text": z_text,
			"z_graph": z_graph,
		}


def build_default_model(
	textual_model_name: str,
	sequence_model_name: str,
	graph_type: str = "hgt",
	hidden_dim: int = 256,
	projection_dim: int = 128,
) -> MultiViewModel:
	textual_encoder = TransformerTextEncoder(EncoderConfig(model_name=textual_model_name))
	sequence_encoder = TransformerTextEncoder(EncoderConfig(model_name=sequence_model_name))

	if graph_type == "rgcn":
		graph_encoder = RGCNEncoder(in_dim=None, hidden_dim=hidden_dim)
	else:
		graph_encoder = HGTEncoder(in_dim=None, hidden_dim=hidden_dim)

	fusion = GatedFusion(dim=textual_encoder.hidden_size)
	text_projection = ProjectionHead(
		in_dim=textual_encoder.hidden_size,
		hidden_dim=hidden_dim,
		out_dim=projection_dim,
	)
	graph_projection = ProjectionHead(
		in_dim=hidden_dim,
		hidden_dim=hidden_dim,
		out_dim=projection_dim,
	)

	return MultiViewModel(
		textual_encoder=textual_encoder,
		sequence_encoder=sequence_encoder,
		graph_encoder=graph_encoder,
		fusion=fusion,
		text_projection=text_projection,
		graph_projection=graph_projection,
	)
