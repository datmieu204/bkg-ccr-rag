# ./kgs_builder/run.py

import argparse
import os

from camel.storages import Neo4jGraph

from helpers.logger import get_logger
from kgs_builder.core.retrieve import get_improved_response, hybrid_retrieve
from kgs_builder.data_processing.data_loader import load_high
from kgs_builder.graph.create_graph_with_description import creat_metagraph_with_description
from kgs_builder.utils import merge_similar_nodes, ref_link, str_uuid

logger = get_logger("run", log_file="logs/run.log")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the KGS Builder pipeline")
    parser.add_argument("-simple", action="store_true")
    parser.add_argument("-construct_graph", action="store_true")
    parser.add_argument("-inference", action="store_true")
    parser.add_argument("-improved_inference", action="store_true")
    parser.add_argument("-grained_chunk", action="store_true")
    parser.add_argument("-bottom_filter", action="store_true")
    parser.add_argument("-min_overlap", type=int, default=3)
    parser.add_argument("-trinity", action="store_true")
    parser.add_argument("-trinity_gid1", type=str)
    parser.add_argument("-trinity_gid2", type=str)
    parser.add_argument("-ingraphmerge", action="store_true")
    parser.add_argument("-crossgraphmerge", action="store_true")
    parser.add_argument("-dataset", type=str, default="mimic_ex")
    parser.add_argument("-data_path", type=str, default="./dataset_test")
    parser.add_argument("-test_data_path", type=str, default="./dataset_ex/report_0.txt")
    return parser


def _create_neo4j() -> Neo4jGraph:
    url = os.getenv("NEO4J_URL")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    missing = [k for k, v in {
        "NEO4J_URL": url,
        "NEO4J_USERNAME": username,
        "NEO4J_PASSWORD": password,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing Neo4j environment variables: {', '.join(missing)}")

    return Neo4jGraph(url=url, username=username, password=password)


def improved_inference_pipeline(n4j, question: str, use_multi_subgraph: bool = False):
    """Improved inference pipeline using hybrid retrieval."""
    logger.info("IMPROVED INFERENCE PIPELINE (Hybrid U-Retrieval)")
    logger.info(f"Question: {question[:200]}...")
    logger.info(f"Multi-subgraph mode: {use_multi_subgraph}")

    logger.info("PHASE 1: HYBRID RETRIEVAL")
    logger.info("[1/4] Vector Search - Pre-filtering candidates...")
    top_k = 3 if use_multi_subgraph else 1
    gids = hybrid_retrieve(n4j, question, top_k=top_k)

    if not gids:
        logger.error("No relevant subgraphs found!")
        return None

    logger.info(f"Selected {len(gids)} subgraph(s): {[g[:8] + '...' for g in gids]}")
    logger.info("PHASE 2: RESPONSE GENERATION")

    answer, primary_gid = get_improved_response(
        n4j,
        question,
        use_multi_subgraph=use_multi_subgraph,
        top_k_subgraphs=top_k,
    )

    if not answer:
        logger.error("Failed to generate answer")
        return None

    logger.info("INFERENCE COMPLETE")
    logger.info(f"Primary GID: {primary_gid[:16]}...")
    logger.info(f"Answer length: {len(answer)} characters")
    logger.info(f"Preview: {answer[:300]}...")

    return answer


def main() -> None:
    args = build_parser().parse_args()

    if args.simple:
        from kgs_builder.nano_graphrag import GraphRAG, QueryParam

        graph_func = GraphRAG(working_dir="./nanotest")
        with open("./dataset_ex/report_0.txt", encoding="utf-8") as f:
            graph_func.insert(f.read())

        print(graph_func.query("What is the main symptom of the patient?", param=QueryParam(mode="local")))
        return

    if not args.construct_graph and not args.improved_inference:
        logger.info("No action selected. Use -construct_graph and/or -improved_inference.")
        return

    n4j = _create_neo4j()

    if args.construct_graph:
        if args.dataset != "mimic_ex":
            logger.warning(f"Unsupported dataset option: {args.dataset}")
        else:
            if not os.path.isdir(args.data_path):
                raise FileNotFoundError(f"Data path not found: {args.data_path}")

            files = [
                file for file in os.listdir(args.data_path)
                if os.path.isfile(os.path.join(args.data_path, file))
            ]

            for file_name in files:
                file_path = os.path.join(args.data_path, file_name)
                content = load_high(file_path)
                gid = str_uuid()
                n4j = creat_metagraph_with_description(args, content, gid, n4j)

                if args.trinity:
                    if not args.trinity_gid1 or not args.trinity_gid2:
                        logger.warning("Trinity mode requires both -trinity_gid1 and -trinity_gid2")
                    else:
                        ref_link(n4j, args.trinity_gid1, args.trinity_gid2)

            if args.crossgraphmerge:
                merge_similar_nodes(n4j, None)

    if args.improved_inference:
        prompt_path = "./prompt.txt"
        if os.path.exists(prompt_path):
            question = load_high(prompt_path)
            logger.info("Loaded question from prompt.txt")
        else:
            question = load_high(args.test_data_path)
            logger.warning(f"prompt.txt not found, fallback to: {args.test_data_path}")

        answer = improved_inference_pipeline(n4j, question, use_multi_subgraph=False)

        if answer:
            logger.info("FINAL ANSWER")
            print(answer)
        else:
            logger.error("Cannot perform inference - no valid results found.")
            logger.error("Please ensure the knowledge graph is built and contains relevant data.")


if __name__ == "__main__":
    main()
