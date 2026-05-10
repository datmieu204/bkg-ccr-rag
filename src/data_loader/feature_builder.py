# ./src/data_loader/feature_builder.py

from typing import Dict, Iterable, List, Optional, Sequence

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

SEQUENCE_LABELS = {
	"gene_protein": {"GENE", "PROTEIN"},
	"drug": {"DRUG"},
	"disease": {"DISEASE"},
}


def pick_primary_label(labels: Sequence[str]) -> str:
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


def _node_text(node: Dict) -> str:
	description = (node.get("description") or "").strip()
	if description:
		return description
	name = (node.get("name") or "").strip()
	if name:
		return name
	node_id = (node.get("id") or "").strip()
	return node_id


def _normalize_summary(summary: Optional[object]) -> str:
	if summary is None:
		return ""
	if isinstance(summary, list):
		return "\n".join(str(item) for item in summary if item)
	return str(summary)


def build_textual_content(
	nodes: Iterable[Dict],
	summary: Optional[object] = None,
	relations: Optional[Iterable[Dict]] = None,
	max_texts: Optional[int] = None,
) -> List[str]:
	texts: List[str] = []
	summary_text = _normalize_summary(summary)
	if summary_text:
		summary_clean = summary_text.strip()
		if summary_clean:
			texts.append(summary_clean)

	for node in nodes:
		text = _node_text(node)
		if text:
			texts.append(text)

	if relations:
		node_lookup = {node.get("neo_id"): node for node in nodes}
		for rel in relations:
			src = node_lookup.get(rel.get("src"))
			dst = node_lookup.get(rel.get("dst"))
			if not src or not dst:
				continue
			src_text = _node_text(src)
			dst_text = _node_text(dst)
			rel_type = rel.get("rel_type", "RELATED_TO")
			if src_text and dst_text:
				texts.append(f"{src_text} {rel_type} {dst_text}")

	if max_texts is not None:
		texts = texts[:max_texts]

	return texts


def build_sequence_content(
	nodes: Iterable[Dict],
	label_map: Optional[Dict[str, set]] = None,
) -> Dict[str, List[str]]:
	label_map = label_map or SEQUENCE_LABELS
	sequences: Dict[str, List[str]] = {key: [] for key in label_map.keys()}

	for node in nodes:
		labels = node.get("labels") or []
		primary_label = pick_primary_label(labels)
		node_value = (node.get("sequence") or node.get("name") or node.get("id") or "").strip()
		if not node_value:
			continue

		for bucket, label_set in label_map.items():
			if primary_label in label_set:
				sequences[bucket].append(node_value)

	for bucket in sequences:
		sequences[bucket] = sorted(set(sequences[bucket]))

	return sequences
