import argparse
import os

from camel.storages import Neo4jGraph
from helpers.logger import get_logger

logger = get_logger("diagnose_gid_context", log_file="logs/diagnose_gid_context.log")

def _connect_neo4j() -> Neo4jGraph:
    url = "bolt://localhost:25515"
    username = "neo4j"
    password = "datmieu2004cgx"

    try:
        return Neo4jGraph(
            url=url, 
            username=username, 
            password=password,
            encrypted=False
        )
    except TypeError:
        return Neo4jGraph(url=url, username=username, password=password)


def _fetch_count(n4j: Neo4jGraph, query: str, gid: str) -> int:
    result = n4j.query(query, {"gid": gid})
    return int(result[0]["count"]) if result else 0


def diagnose_gid(n4j: Neo4jGraph, gid: str, sample: int) -> None:
    logger.info("=" * 60)
    logger.info(f"Diagnose GID: {gid}")
    logger.info("=" * 60)

    node_count = _fetch_count(
        n4j,
        """
        MATCH (n {gid: $gid})
        RETURN count(n) AS count
        """,
        gid,
    )
    summary_count = _fetch_count(
        n4j,
        """
        MATCH (s:Summary {gid: $gid})
        RETURN count(s) AS count
        """,
        gid,
    )
    rel_count = _fetch_count(
        n4j,
        """
        MATCH (n)-[r]-(m)
        WHERE n.gid = $gid AND m.gid = $gid
          AND NOT n:Summary AND NOT m:Summary
        RETURN count(r) AS count
        """,
        gid,
    )
    reference_count = _fetch_count(
        n4j,
        """
        MATCH (n {gid: $gid})-[r:REFERENCE]->(m)
        RETURN count(r) AS count
        """,
        gid,
    )
    is_reference_of_count = _fetch_count(
        n4j,
        """
        MATCH (n {gid: $gid})-[r:IS_REFERENCE_OF]->(m)
        RETURN count(r) AS count
        """,
        gid,
    )
    link_context_count = _fetch_count(
        n4j,
        """
        MATCH (n {gid: $gid})-[r:REFERENCE]->(m)
        WHERE NOT m:Summary
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND type(s) <> 'REFERENCE'
        RETURN count(s) AS count
        """,
        gid,
    )

    logger.info(f"Nodes: {node_count}")
    logger.info(f"Summary nodes: {summary_count}")
    logger.info(f"Intra-subgraph relationships: {rel_count}")
    logger.info(f"REFERENCE relationships: {reference_count}")
    logger.info(f"IS_REFERENCE_OF relationships: {is_reference_of_count}")
    logger.info(f"Link-context triples: {link_context_count}")

    if sample > 0:
        samples = n4j.query(
            """
            MATCH (n {gid: $gid})
            WHERE NOT n:Summary
            RETURN labels(n) AS labels, n.id AS id, n.name AS name
            LIMIT $limit
            """,
            {"gid": gid, "limit": sample},
        )
        logger.info(f"Sample nodes ({len(samples)}): {samples}")


def diagnose_all_gids(n4j: Neo4jGraph, sample: int, limit: int | None) -> None:
    query = """
        MATCH (n)
        WHERE n.gid IS NOT NULL
        RETURN DISTINCT n.gid AS gid
        ORDER BY gid
    """
    gids = n4j.query(query)
    gid_list = [row.get("gid") for row in gids if row.get("gid")]

    if limit is not None:
        gid_list = gid_list[:limit]

    logger.info(f"Total GIDs to diagnose: {len(gid_list)}")
    for gid in gid_list:
        diagnose_gid(n4j, gid, sample)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose subgraph context by GID")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gid", help="Target graph id (gid)")
    group.add_argument("--all", action="store_true", help="Diagnose all gids in the graph")
    parser.add_argument("--sample", type=int, default=5, help="Sample nodes to print")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of gids when using --all")
    args = parser.parse_args()

    n4j = _connect_neo4j()
    if args.all:
        diagnose_all_gids(n4j, args.sample, args.limit)
    else:
        diagnose_gid(n4j, args.gid, args.sample)


if __name__ == "__main__":
    main()
