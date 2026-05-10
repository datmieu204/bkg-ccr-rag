import argparse
import os
from typing import Dict, List, Optional

from camel.storages import Neo4jGraph
from helpers.logger import get_logger
from kgs_builder.core.summerize import process_chunks

logger = get_logger("regenerate_summaries", log_file="logs/regenerate_summaries.log")

_EMPTY_CONTENT_CLAUSE = (
    "s.content IS NULL\n"
    "OR s.content = ''\n"
    "OR s.content = []\n"
    "OR size([x IN s.content WHERE x IS NOT NULL AND trim(toString(x)) <> '']) = 0"
)


def _normalize_content(content) -> str:
    if isinstance(content, list):
        return " ".join(str(item) for item in content if item)
    if content is None:
        return ""
    return str(content)


def _fetch_all_gids(n4j: Neo4jGraph) -> List[str]:
    rows = n4j.query(
        """
        MATCH (n)
        WHERE n.gid IS NOT NULL AND NOT n:Summary
        RETURN DISTINCT n.gid AS gid
        """
    )
    return sorted({row.get("gid") for row in rows if row.get("gid")})


def _fetch_summary_map(n4j: Neo4jGraph) -> Dict[str, List[object]]:
    rows = n4j.query(
        """
        MATCH (s:Summary)
        RETURN s.gid AS gid, s.content AS content
        """
    )
    summary_map: Dict[str, List[object]] = {}
    for row in rows:
        gid = row.get("gid")
        if not gid:
            continue
        summary_map.setdefault(gid, []).append(row.get("content"))
    return summary_map


def _count_empty_summaries(n4j: Neo4jGraph) -> int:
    result = n4j.query(
        f"""
        MATCH (s:Summary)
        WHERE {_EMPTY_CONTENT_CLAUSE}
        RETURN count(s) AS empty_count
        """
    )
    return int(result[0]["empty_count"]) if result else 0


def _count_orphan_summaries(n4j: Neo4jGraph, only_empty: bool) -> int:
    query = """
        MATCH (s:Summary)
        WHERE NOT EXISTS {
            MATCH (n) WHERE n.gid = s.gid AND NOT n:Summary
        }
    """
    if only_empty:
        query += f"\nAND ({_EMPTY_CONTENT_CLAUSE})"
    query += "\nRETURN count(s) AS orphan_count"
    result = n4j.query(query)
    return int(result[0]["orphan_count"]) if result else 0


def _delete_orphan_summaries(n4j: Neo4jGraph, only_empty: bool) -> int:
    query = """
        MATCH (s:Summary)
        WHERE NOT EXISTS {
            MATCH (n) WHERE n.gid = s.gid AND NOT n:Summary
        }
    """
    if only_empty:
        query += f"\nAND ({_EMPTY_CONTENT_CLAUSE})"
    query += "\nDETACH DELETE s RETURN count(*) AS deleted"
    result = n4j.query(query)
    return int(result[0]["deleted"]) if result else 0


def _is_empty_summary_content(content: object) -> bool:
    normalized = _normalize_content(content)
    return not normalized.strip()


def _fetch_source_text(n4j: Neo4jGraph, gid: str, max_nodes: int) -> str:
    rows = n4j.query(
        """
        MATCH (n {gid: $gid})
        WHERE NOT n:Summary
        RETURN n.id AS id, n.description AS description
        LIMIT $limit
        """,
        {"gid": gid, "limit": max_nodes},
    )

    if not rows:
        return ""

    parts = []
    for row in rows:
        node_id = row.get("id")
        description = row.get("description")
        if node_id and description:
            parts.append(f"{node_id}. {description}")
        elif node_id:
            parts.append(str(node_id))
        elif description:
            parts.append(str(description))

    return "\n".join(parts)


def _upsert_summary(n4j: Neo4jGraph, gid: str, summary_content: object) -> None:
    updated = n4j.query(
        f"""
        MATCH (s:Summary {{gid: $gid}})
        WHERE {_EMPTY_CONTENT_CLAUSE}
        SET s.content = $content
        RETURN count(s) AS updated
        """,
        {"gid": gid, "content": summary_content},
    )
    if not updated or updated[0].get("updated", 0) == 0:
        n4j.query(
            """
            CREATE (s:Summary {gid: $gid, content: $content})
            RETURN s
            """,
            {"gid": gid, "content": summary_content},
        )

    n4j.query(
        """
        MATCH (s:Summary {gid: $gid}), (n)
        WHERE n.gid = $gid AND NOT n:Summary
        MERGE (s)-[:SUMMARIZES]->(n)
        """,
        {"gid": gid},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate missing/empty summaries")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of GIDs to process")
    parser.add_argument("--force", action="store_true", help="Regenerate for all GIDs, not only empty")
    parser.add_argument("--max-nodes", type=int, default=200, help="Max nodes to build source text")
    parser.add_argument("--delete-orphans", action="store_true", help="Delete Summary nodes with no matching GID")
    parser.add_argument(
        "--delete-orphans-empty-only",
        action="store_true",
        help="Delete only orphan Summary nodes that are empty",
    )
    args = parser.parse_args()

    n4j = Neo4jGraph(
        url=os.getenv("NEO4J_URL_CGX", "bolt://localhost:25515"),
        username=os.getenv("NEO4J_USERNAME_CGX", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD_CGX", "datmieu2004cgx"),
    )

    gids = _fetch_all_gids(n4j)
    summary_map = _fetch_summary_map(n4j)

    empty_count = _count_empty_summaries(n4j)
    orphan_count = _count_orphan_summaries(
        n4j,
        only_empty=args.delete_orphans_empty_only,
    )

    targets: List[str] = []
    for gid in gids:
        contents = summary_map.get(gid, [])
        if args.force:
            targets.append(gid)
            continue
        if not contents:
            targets.append(gid)
            continue
        if any(_is_empty_summary_content(content) for content in contents):
            targets.append(gid)

    if args.limit and args.limit > 0:
        targets = targets[: args.limit]

    logger.info("Total GIDs: %d", len(gids))
    logger.info("Empty Summary nodes: %d", empty_count)
    logger.info("Orphan Summary nodes: %d", orphan_count)
    logger.info("Targets: %d", len(targets))

    if args.delete_orphans:
        deleted = _delete_orphan_summaries(
            n4j,
            only_empty=args.delete_orphans_empty_only,
        )
        logger.info("Deleted orphan Summary nodes: %d", deleted)

    if not targets:
        logger.info("No summaries to regenerate")
        return

    success = 0
    skipped = 0

    for idx, gid in enumerate(targets, start=1):
        logger.info("[%d/%d] Regenerating summary for %s", idx, len(targets), gid[:8])
        source_text = _fetch_source_text(n4j, gid, args.max_nodes)
        if not source_text.strip():
            logger.warning("No source text for %s, skipping", gid[:8])
            skipped += 1
            continue

        summary_content = process_chunks(source_text)
        if not _normalize_content(summary_content).strip():
            logger.warning("Empty summary generated for %s, skipping", gid[:8])
            skipped += 1
            continue

        _upsert_summary(n4j, gid, summary_content)
        success += 1

    logger.info("Completed: success=%d, skipped=%d", success, skipped)


if __name__ == "__main__":
    main()
