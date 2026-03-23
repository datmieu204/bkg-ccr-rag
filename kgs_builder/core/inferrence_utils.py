# ./kgs_builder/core/inferrence_utils.py

from .retrieve import get_improved_response, hybrid_retrieve
from helpers.logger import get_logger

logger = get_logger("inferrence_utils", log_file="logs/inferrence_utils.log")

def infer(n4j, question: str, use_multi_subgraph: bool=False):
    """
    Main inference function to get answer for a question.
    
    Args:
        n4j: Neo4j connection
        question: The input question string
        use_multi_subgraph: Whether to use multi-subgraph retrieval or not (default False)
    Returns:
        answer: The generated answer string
    """
    logger.info("INFERENCE")
    logger.info(f"Question: {question[:200]}...")
    logger.info(f"Multi-subgraph mode: {use_multi_subgraph}")
    logger.info("[1/4] Vector Search - Pre-filtering candidates...")
    top_k = 3 if use_multi_subgraph else 1
    gids = hybrid_retrieve(n4j, question, top_k=top_k)

    if not gids:
        logger.warning("No relevant subgraphs found. Returning fallback answer.")
        return None

    logger.info(f"Retrieved GIDs: {gids}")
    logger.info("[2/4] Generating answer with retrieved context...")

    answer, primary_gid = get_improved_response(
        n4j,
        question,
        use_multi_subgraph=use_multi_subgraph,
        top_k_subgraphs=top_k,
    )

    if not answer:
        logger.warning("Failed to generate an answer. Returning fallback response.")
        return None
    
    logger.info("INFERENCE COMPLETE")
    logger.info(f"Primary GID: {primary_gid[:16]}...")
    logger.info(f"Answer length: {len(answer)} characters")
    logger.info(f"Preview: {answer[:300]}...")
    
    return answer