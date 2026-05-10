# ./src/data_loader/dataset.py

import os
from typing import Dict, List, Optional, Sequence

from torch.utils.data import Dataset as TorchDataset

from .cache_manager import CacheManager
from .feature_builder import build_sequence_content, build_textual_content
from .kg_loader import DEFAULT_NEIGHBOR_REL_TYPES, KGLoader
from .preprocessing import truncate_texts


class BioMedDataset(TorchDataset):
    """
    Multi-view dataset:
    - text_view: textual content + optional sequence content
    - graph_view: PyG HeteroData with optional neighbors across layers
    """

    def __init__(
        self,
        root: str,
        uri: str,
        user: str,
        password: str,
        database: Optional[str] = None,
        split: str = "top",
        gids: Optional[Sequence[str]] = None,
        neighbor_hops: int = 2,
        neighbor_rel_types: Optional[Sequence[str]] = None,
        include_summary: bool = True,
        include_relation_texts: bool = False,
        max_texts: Optional[int] = 200,
        max_text_chars: Optional[int] = 1024,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        return_raw_graph: bool = False,
    ) -> None:
        self.root = root
        self.loader = KGLoader(uri, user, password, database=database)
        self.layer_gids = self.loader.list_layer_gids()
        self.split = split

        if gids is not None:
            self.gids = list(gids)
        else:
            if split not in self.layer_gids:
                raise ValueError(f"Unknown split '{split}', available: {list(self.layer_gids.keys())}")
            self.gids = list(self.layer_gids[split])

        self.neighbor_hops = neighbor_hops
        self.neighbor_rel_types = list(neighbor_rel_types or DEFAULT_NEIGHBOR_REL_TYPES)
        self.include_summary = include_summary
        self.include_relation_texts = include_relation_texts
        self.max_texts = max_texts
        self.max_text_chars = max_text_chars
        self.return_raw_graph = return_raw_graph

        cache_dir = cache_dir or os.path.join(self.root, "cache")
        self.cache = CacheManager(cache_dir, enabled=use_cache)

    def __len__(self) -> int:
        return len(self.gids)

    def _cache_key(self, gid: str) -> str:
        rel_key = "-".join(sorted(self.neighbor_rel_types))
        return f"{self.split}_{gid}_h{self.neighbor_hops}_{rel_key}"

    def __getitem__(self, index: int) -> Dict:
        gid = self.gids[index]
        cache_key = self._cache_key(gid)

        if self.cache.has(cache_key):
            return self.cache.load(cache_key)

        graph_view = self.loader.load_graph_view(
            gid,
            neighbor_hops=self.neighbor_hops,
            neighbor_rel_types=self.neighbor_rel_types,
        )

        summary = self.loader.fetch_summary(gid) if self.include_summary else None

        textual_content = build_textual_content(
            graph_view.nodes,
            summary=summary,
            relations=graph_view.edges if self.include_relation_texts else None,
            max_texts=None,
        )
        textual_content = truncate_texts(
            textual_content,
            max_items=self.max_texts,
            max_chars=self.max_text_chars,
        )
        sequence_content = build_sequence_content(graph_view.nodes)

        sample = {
            "gid": gid,
            "layer": self.split,
            "text_view": {
                "summary": summary,
                "textual_content": textual_content,
                "sequence_content": sequence_content,
            },
            "graph_view": graph_view.hetero,
        }

        if self.return_raw_graph:
            sample["graph_nodes"] = graph_view.nodes
            sample["graph_edges"] = graph_view.edges

        self.cache.save(cache_key, sample)
        return sample

    def close(self) -> None:
        self.loader.close()

