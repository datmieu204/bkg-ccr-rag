# ./src/data_loader/kg_loader.py

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData

DEFAULT_NEIGHBOR_REL_TYPES = ("REFERENCE", "IS_REFERENCE_OF", "IS_REFERENCED_BY")

LABEL_PRIORITY = [
    "DISEASE",
    "DRUG",
    "GENE",
    "PROTEIN",
    "PHENOTYPE",
    "PROCEDURE",
    "ANATOMY",
    "ORGANISM",
    "CONCEPT",
]


@dataclass
class GraphView:
    nodes: List[Dict]
    edges: List[Dict]
    hetero: HeteroData
    anchor_node_ids: List[int]


def _pick_primary_label(labels: Sequence[str]) -> str:
    if not labels:
        return "ENTITY"
    label_set = {label.upper() for label in labels}
    for label in LABEL_PRIORITY:
        if label in label_set:
            return label
    for label in labels:
        if label.upper() != "SUMMARY":
            return label.upper()
    return "ENTITY"


def _infer_embedding_dim(nodes: Iterable[Dict]) -> int:
    for node in nodes:
        emb = node.get("embedding")
        if isinstance(emb, list) and emb:
            return len(emb)
    return 0


def _edge_feature(rel_props: Dict) -> float:
    for key in ("similarity", "strength"):
        value = rel_props.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def to_heterodata(
    nodes: List[Dict],
    edges: List[Dict],
    anchor_neo_ids: Optional[Iterable[int]] = None,
    embedding_dim: Optional[int] = None,
) -> HeteroData:
    data = HeteroData()
    anchor_set = set(anchor_neo_ids or [])

    nodes_by_type: Dict[str, List[Dict]] = {}
    node_index: Dict[int, Tuple[str, int]] = {}
    for node in nodes:
        node_type = _pick_primary_label(node.get("labels") or [])
        bucket = nodes_by_type.setdefault(node_type, [])
        node_index[node["neo_id"]] = (node_type, len(bucket))
        bucket.append(node)

    emb_dim = embedding_dim if embedding_dim is not None else _infer_embedding_dim(nodes)

    for node_type, bucket in nodes_by_type.items():
        data[node_type].num_nodes = len(bucket)
        if emb_dim > 0:
            embeddings = []
            for node in bucket:
                emb = node.get("embedding")
                if isinstance(emb, list) and len(emb) == emb_dim:
                    embeddings.append(emb)
                else:
                    embeddings.append([0.0] * emb_dim)
            data[node_type].x = torch.tensor(embeddings, dtype=torch.float)

        data[node_type].node_id = [node.get("id") or node.get("name") for node in bucket]
        data[node_type].name = [node.get("name") for node in bucket]
        data[node_type].gid = [node.get("gid") for node in bucket]
        data[node_type].text = [
            node.get("description") or node.get("name") or node.get("id") or ""
            for node in bucket
        ]

        if anchor_set:
            data[node_type].is_anchor = torch.tensor(
                [1 if node["neo_id"] in anchor_set else 0 for node in bucket],
                dtype=torch.uint8,
            )

    edges_by_key: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}
    edge_attr_by_key: Dict[Tuple[str, str, str], List[float]] = {}
    for edge in edges:
        src_info = node_index.get(edge["src"])
        dst_info = node_index.get(edge["dst"])
        if not src_info or not dst_info:
            continue
        src_type, src_idx = src_info
        dst_type, dst_idx = dst_info
        rel_type = edge.get("rel_type", "RELATED_TO")
        key = (src_type, rel_type, dst_type)
        edges_by_key.setdefault(key, []).append((src_idx, dst_idx))
        edge_attr_by_key.setdefault(key, []).append(
            _edge_feature(edge.get("rel_props") or {})
        )

    for key, pairs in edges_by_key.items():
        if not pairs:
            continue
        edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        data[key].edge_index = edge_index
        edge_attrs = edge_attr_by_key.get(key)
        if edge_attrs is not None:
            data[key].edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(-1)

    return data


class KGLoader:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: Optional[str] = None,
        encrypted: bool = False,
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password), encrypted=encrypted)
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def _run(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        return self._run(query, params)

    def _fetch_gids(self, query: str) -> List[str]:
        rows = self._run(query)
        return sorted({row.get("gid") for row in rows if row.get("gid")})

    def list_layer_gids(self) -> Dict[str, List[str]]:
        all_gids = self._fetch_gids(
            """
            MATCH (n)
            WHERE n.gid IS NOT NULL AND NOT n:Summary
            RETURN DISTINCT n.gid AS gid
            """
        )

        bottom_gids = self._fetch_gids(
            """
            MATCH (n)
            WHERE n.gid IS NOT NULL AND NOT n:Summary
              AND (n.source = 'UMLS' OR n.data_type = 'structured')
            RETURN DISTINCT n.gid AS gid
            """
        )

        middle_gids = self._fetch_gids(
            """
            MATCH (n)-[:IS_REFERENCE_OF]->()
            WHERE n.gid IS NOT NULL AND NOT n:Summary
            RETURN DISTINCT n.gid AS gid
            """
        )

        top_gids = sorted(set(all_gids) - set(bottom_gids) - set(middle_gids))
        return {
            "all": all_gids,
            "bottom": bottom_gids,
            "middle": middle_gids,
            "top": top_gids,
        }

    def fetch_summary(self, gid: str) -> Optional[str]:
        rows = self._run(
            """
            MATCH (s:Summary {gid: $gid})
            RETURN s.content AS content
            LIMIT 1
            """,
            {"gid": gid},
        )
        if not rows:
            return None
        return rows[0].get("content")

    def fetch_nodes_by_gids(
        self,
        gids: List[str],
        include_summary: bool = False,
    ) -> List[Dict]:
        rows = self._run(
            """
            MATCH (n)
            WHERE n.gid IN $gids
              AND ($include_summary OR NOT n:Summary)
            RETURN elementId(n) AS neo_id,
                   labels(n) AS labels,
                   n.gid AS gid,
                   n.id AS id,
                   n.name AS name,
                   n.description AS description,
                   n.embedding AS embedding,
                   properties(n) AS props
            """,
            {"gids": gids, "include_summary": include_summary},
        )
        return rows

    def fetch_edges_by_gids(
        self,
        gids: List[str],
        include_summary: bool = False,
    ) -> List[Dict]:
        rows = self._run(
            """
            MATCH (a)-[r]->(b)
            WHERE a.gid IN $gids AND b.gid IN $gids
              AND ($include_summary OR (NOT a:Summary AND NOT b:Summary))
            RETURN elementId(a) AS src,
                   elementId(b) AS dst,
                   type(r) AS rel_type,
                   properties(r) AS rel_props
            """,
            {"gids": gids, "include_summary": include_summary},
        )
        return rows

    def fetch_hgt_metadata(self) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        node_types: set = set()
        edge_types: set = set()

        node_rows = self._run(
            """
            MATCH (n)
            WHERE NOT n:Summary
            RETURN DISTINCT labels(n) AS labels
            """
        )
        for row in node_rows:
            labels = row.get("labels") or []
            node_types.add(_pick_primary_label(labels))

        edge_rows = self._run(
            """
            MATCH (a)-[r]->(b)
            WHERE NOT a:Summary AND NOT b:Summary
            RETURN DISTINCT labels(a) AS a_labels, type(r) AS rel, labels(b) AS b_labels
            """
        )
        for row in edge_rows:
            src = _pick_primary_label(row.get("a_labels") or [])
            dst = _pick_primary_label(row.get("b_labels") or [])
            rel = row.get("rel") or "RELATED_TO"
            node_types.add(src)
            node_types.add(dst)
            edge_types.add((src, rel, dst))

        return sorted(node_types), sorted(edge_types)

    def fetch_neighbor_nodes(
        self,
        gid: str,
        hops: int,
        rel_types: Sequence[str],
    ) -> List[Dict]:
        hop_count = max(1, int(hops))
        query = f"""
            MATCH (n)
            WHERE n.gid = $gid AND NOT n:Summary
            MATCH p=(n)-[r*1..{hop_count}]-(m)
            WHERE NOT m:Summary
              AND ALL(rel IN r WHERE type(rel) IN $rel_types)
            WITH DISTINCT m
            ORDER BY rand()
            LIMIT 300
            RETURN elementId(m) AS neo_id,
                            labels(m) AS labels,
                            m.gid AS gid,
                            m.id AS id,
                            m.name AS name,
                            m.description AS description,
                            m.embedding AS embedding,
                            properties(m) AS props
            """
        rows = self._run(
            query,
            {"gid": gid, "rel_types": list(rel_types)},
        )
        return rows

    def fetch_neighbor_edges(
        self,
        gid: str,
        hops: int,
        rel_types: Sequence[str],
    ) -> List[Dict]:
        hop_count = max(1, int(hops))
        query = f"""
            MATCH (n)
            WHERE n.gid = $gid AND NOT n:Summary
            MATCH p=(n)-[r*1..{hop_count}]-(m)
            WHERE NOT m:Summary
              AND ALL(rel IN r WHERE type(rel) IN $rel_types)
            UNWIND r AS rel
            WITH DISTINCT rel
            ORDER BY rand()
            LIMIT 1000
            RETURN elementId(startNode(rel)) AS src,
                            elementId(endNode(rel)) AS dst,
                            type(rel) AS rel_type,
                            properties(rel) AS rel_props
            """
        rows = self._run(
            query,
            {"gid": gid, "rel_types": list(rel_types)},
        )
        return rows

    def load_graph_view(
        self,
        gid: str,
        neighbor_hops: int = 1,
        neighbor_rel_types: Optional[Sequence[str]] = None,
    ) -> GraphView:
        neighbor_rel_types = neighbor_rel_types or DEFAULT_NEIGHBOR_REL_TYPES

        base_nodes = self.fetch_nodes_by_gids([gid], include_summary=False)
        base_edges = self.fetch_edges_by_gids([gid], include_summary=False)
        anchor_ids = [node["neo_id"] for node in base_nodes]

        neighbor_nodes: List[Dict] = []
        neighbor_edges: List[Dict] = []
        if neighbor_hops and neighbor_hops > 0:
            neighbor_nodes = self.fetch_neighbor_nodes(gid, neighbor_hops, neighbor_rel_types)
            neighbor_edges = self.fetch_neighbor_edges(gid, neighbor_hops, neighbor_rel_types)

        node_map: Dict[int, Dict] = {node["neo_id"]: node for node in base_nodes}
        for node in neighbor_nodes:
            node_map.setdefault(node["neo_id"], node)

        edge_map: Dict[Tuple[int, int, str], Dict] = {}
        for edge in base_edges + neighbor_edges:
            key = (edge["src"], edge["dst"], edge.get("rel_type", "RELATED_TO"))
            edge_map.setdefault(key, edge)

        nodes = list(node_map.values())
        edges = list(edge_map.values())

        hetero = to_heterodata(nodes, edges, anchor_neo_ids=anchor_ids)
        return GraphView(nodes=nodes, edges=edges, hetero=hetero, anchor_node_ids=anchor_ids)