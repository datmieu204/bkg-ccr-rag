import argparse
import os
from typing import Dict, List, Optional

from camel.storages import Neo4jGraph
from helpers.logger import get_logger
from kgs_builder.core.linking import smart_ref_link
from kgs_builder.utils import ref_link

logger = get_logger("link_layers", log_file="logs/link_layers.log")


def _connect_neo4j(
	url: Optional[str],
	username: Optional[str],
	password: Optional[str],
) -> Neo4jGraph:
	neo4j_url = url or os.getenv("NEO4J_URL_CGX") or os.getenv("NEO4J_URL") or "bolt://localhost:25515"
	neo4j_username = username or os.getenv("NEO4J_USERNAME_CGX") or os.getenv("NEO4J_USERNAME") or "neo4j"
	neo4j_password = password or os.getenv("NEO4J_PASSWORD_CGX") or os.getenv("NEO4J_PASSWORD") or "datmieu2004cgx"

	if not neo4j_password:
		raise SystemExit("NEO4J_PASSWORD_CGX or NEO4J_PASSWORD is required")

	try:
		return Neo4jGraph(
			url=neo4j_url,
			username=neo4j_username,
			password=neo4j_password,
			encrypted=False,
		)
	except TypeError:
		return Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)


def _fetch_gids(n4j: Neo4jGraph, query: str) -> List[str]:
	results = n4j.query(query)
	return sorted({row.get("gid") for row in results if row.get("gid")})


def _collect_gids(n4j: Neo4jGraph) -> Dict[str, List[str]]:
	all_gids = _fetch_gids(
		n4j,
		"""
		MATCH (n)
		WHERE n.gid IS NOT NULL AND NOT n:Summary
		RETURN DISTINCT n.gid AS gid
		""",
	)

	bottom_gids = _fetch_gids(
		n4j,
		"""
		MATCH (n)
		WHERE n.gid IS NOT NULL AND NOT n:Summary
		  AND (n.source = 'UMLS' OR n.data_type = 'structured')
		RETURN DISTINCT n.gid AS gid
		""",
	)

	middle_gids = _fetch_gids(
		n4j,
		"""
		MATCH (n)-[:IS_REFERENCE_OF]->()
		WHERE n.gid IS NOT NULL AND NOT n:Summary
		RETURN DISTINCT n.gid AS gid
		""",
	)

	top_gids = sorted(set(all_gids) - set(bottom_gids) - set(middle_gids))

	return {
		"all": all_gids,
		"bottom": bottom_gids,
		"middle": middle_gids,
		"top": top_gids,
	}


def _link_smart(
	n4j: Neo4jGraph,
	top_gids: List[str],
	middle_gids: List[str],
	top_k: int,
	threshold: float,
) -> Dict[str, int]:
	stats = {
		"entities_extracted": 0,
		"middle_chunks_found": 0,
		"links_created": 0,
	}

	if not top_gids:
		return stats

	for idx, top_gid in enumerate(top_gids, start=1):
		logger.info(f"[Smart Linking] {idx}/{len(top_gids)} GID: {top_gid[:8]}...")
		result = smart_ref_link(
			n4j,
			top_gid,
			middle_layer_gids=middle_gids,
			top_k=top_k,
			similarity_threshold=threshold,
		)
		stats["entities_extracted"] += int(result.get("entities_extracted", 0))
		stats["middle_chunks_found"] += int(result.get("middle_chunks_found", 0))
		stats["links_created"] += int(result.get("links_created", 0))

	return stats


def _link_direct(
	n4j: Neo4jGraph,
	top_gids: List[str],
	middle_gids: List[str],
) -> Dict[str, int]:
	stats = {"links_created": 0}

	if not top_gids or not middle_gids:
		return stats

	for i, top_gid in enumerate(top_gids, start=1):
		logger.info(f"[Direct Cosine] {i}/{len(top_gids)} GID: {top_gid[:8]}...")
		for middle_gid in middle_gids:
			result = ref_link(n4j, top_gid, middle_gid)
			if result:
				stats["links_created"] += len(result)

	return stats


def _link_middle_to_bottom(
	n4j: Neo4jGraph,
	middle_gids: List[str],
	batch_size: int,
) -> int:
	if not middle_gids:
		return 0

	total_created = 0
	for i in range(0, len(middle_gids), batch_size):
		batch = middle_gids[i : i + batch_size]
		logger.info(f"[Middle->Bottom] Processing batch {i + 1}-{i + len(batch)}")
		result = n4j.query(
			"""
			UNWIND $gids AS gid
			MATCH (m {gid: gid})
			WHERE NOT m:Summary AND m.id IS NOT NULL
			WITH m, toUpper(m.id) AS entity_name
			MATCH (b)
			WHERE UPPER(b.name) = entity_name
			  AND (b.source = 'UMLS'
				   OR b.data_type = 'structured'
				   OR b:DISEASE OR b:DRUG OR b:PHENOTYPE
				   OR b:PROCEDURE OR b:ANATOMY OR b:CONCEPT)
			MERGE (m)-[r:IS_REFERENCE_OF]->(b)
			ON CREATE SET r.created_at = datetime(), r.method = 'relink'
			RETURN count(DISTINCT r) AS created
			""",
			{"gids": batch},
		)
		created = result[0]["created"] if result else 0
		total_created += created

	return total_created


def _link_bottom_to_middle(n4j: Neo4jGraph) -> int:
	result = n4j.query(
		"""
		MATCH (m)-[:IS_REFERENCE_OF]->(b)
		MERGE (b)-[r:IS_REFERENCED_BY]->(m)
		ON CREATE SET r.created_at = datetime(), r.method = 'relink'
		RETURN count(DISTINCT r) AS created
		"""
	)
	return result[0]["created"] if result else 0


def main() -> None:
	parser = argparse.ArgumentParser(description="Link layers across existing subgraphs")
	parser.add_argument("--neo4j-url", type=str, default=None)
	parser.add_argument("--neo4j-username", type=str, default=None)
	parser.add_argument("--neo4j-password", type=str, default=None)
	parser.add_argument("--top-k", type=int, default=50)
	parser.add_argument("--threshold", type=float, default=0.6)
	parser.add_argument("--limit-top", type=int, default=None, help="Limit number of top gids")
	parser.add_argument("--direct-cosine", action="store_true", help="Use direct cosine linking")
	parser.add_argument("--link-top-middle", action="store_true", help="Link Top -> Middle")
	parser.add_argument("--link-middle-bottom", action="store_true", help="Link Middle -> Bottom")
	parser.add_argument("--link-bottom-middle", action="store_true", help="Create reverse links Bottom -> Middle")
	parser.add_argument("--all-links", action="store_true", help="Run all linking steps")
	parser.add_argument("--batch-gids", type=int, default=50, help="Batch size for middle->bottom relinking")
	args = parser.parse_args()

	n4j = _connect_neo4j(args.neo4j_url, args.neo4j_username, args.neo4j_password)

	gids = _collect_gids(n4j)
	top_gids = gids["top"]
	middle_gids = gids["middle"]

	if args.limit_top is not None:
		top_gids = top_gids[: args.limit_top]

	logger.info(
		"GID counts - all=%d, bottom=%d, middle=%d, top=%d",
		len(gids["all"]),
		len(gids["bottom"]),
		len(gids["middle"]),
		len(top_gids),
	)

	requested = (
		args.link_top_middle
		or args.link_middle_bottom
		or args.link_bottom_middle
		or args.all_links
	)
	run_top_middle = args.link_top_middle or args.all_links or not requested
	run_middle_bottom = args.link_middle_bottom or args.all_links or not requested
	run_bottom_middle = args.link_bottom_middle or args.all_links or not requested

	if run_middle_bottom:
		created = _link_middle_to_bottom(n4j, middle_gids, args.batch_gids)
		logger.info(f"Middle->Bottom links created: {created}")

	if run_top_middle:
		if args.direct_cosine:
			stats = _link_direct(n4j, top_gids, middle_gids)
			logger.info(f"Direct cosine links created: {stats['links_created']}")
		else:
			stats = _link_smart(n4j, top_gids, middle_gids, args.top_k, args.threshold)
			logger.info(
				"Smart linking summary: entities=%d, middle_chunks=%d, links=%d",
				stats["entities_extracted"],
				stats["middle_chunks_found"],
				stats["links_created"],
			)

	if run_bottom_middle:
		created = _link_bottom_to_middle(n4j)
		logger.info(f"Bottom->Middle reverse links created: {created}")


if __name__ == "__main__":
	main()
