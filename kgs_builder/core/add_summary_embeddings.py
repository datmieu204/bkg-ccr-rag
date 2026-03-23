# ./kgs_builder/core/add_summary_embeddings.py

import os
import argparse

from tqdm import tqdm
from camel.storages import Neo4jGraph
from kgs_builder.utils import get_bge_m3_embedding
from helpers.logger import get_logger

logger = get_logger("add_summary_embeddings", log_file="logs/add_summary_embeddings.log")

def get_summaries_without_embeddings(n4j) -> list:
    """Get all Summary nodes that don't have embeddings yet"""
    query = """
        MATCH (s:Summary)
        WHERE s.embedding IS NULL
        RETURN s.gid AS gid, s.content AS content
    """
    return n4j.query(query)

def get_all_summaries(n4j) -> list:
    """Get all Summary nodes"""
    query = """
        MATCH (s:Summary)
        RETURN s.gid AS gid, s.content AS content, 
               s.embedding IS NOT NULL AS has_embedding
    """
    return n4j.query(query)

def add_embedding_to_summary(n4j, gid: str, embedding: list):
    """Store embedding in Summary node"""
    query = """
        MATCH (s:Summary {gid: $gid})
        SET s.embedding = $embedding
        RETURN s.gid
    """
    return n4j.query(query, {'gid': gid, 'embedding': embedding})

def process_summaries(n4j, batch_size: int = 50, force: bool = False):
    """
    Add embeddings to all Summary nodes
    
    Args:
        n4j: Neo4j connection
        batch_size: Number of summaries to process before logging progress
        force: If True, recompute all embeddings even if they exist
    """
    logger.info("="*60)
    logger.info("Adding Embeddings to Summary Nodes")
    logger.info("="*60)
    
    # Get summaries to process
    if force:
        summaries = get_all_summaries(n4j)
        summaries = [s for s in summaries]  # Process all
        logger.info(f"Force mode: will recompute all {len(summaries)} embeddings")
    else:
        summaries = get_summaries_without_embeddings(n4j)
        logger.info(f"Found {len(summaries)} summaries without embeddings")
    
    if not summaries:
        logger.info("All summaries already have embeddings!")
        return
    
    # Process each summary
    success_count = 0
    error_count = 0
    
    for i, summary in enumerate(tqdm(summaries, desc="Adding embeddings")):
        try:
            gid = summary['gid']
            content = summary.get('content', '')
            
            if isinstance(content, list):
                content = " ".join(content) if content else ""
            elif content is None:
                content = ""
            elif not isinstance(content, str):
                content = str(content)
            
            if len(content) > 2000:
                content = content[:2000]
            
            if not content.strip():
                logger.warning(f"Skipping empty summary: {gid[:8]}...")
                error_count += 1
                continue
            
            embedding = get_bge_m3_embedding(content)
            add_embedding_to_summary(n4j, gid, embedding)
            success_count += 1
            
            if (i + 1) % batch_size == 0:
                logger.info(f"Processed {i+1}/{len(summaries)} summaries")
                
        except Exception as e:
            logger.error(f"Error processing summary {summary.get('gid', 'unknown')}: {e}")
            error_count += 1
            continue
    
    logger.info("="*60)
    logger.info("COMPLETE")
    logger.info("="*60)
    logger.info(f"Successfully added embeddings: {success_count}")
    logger.info(f"Errors: {error_count}")

    verify_query = """
        MATCH (s:Summary)
        RETURN 
            count(*) AS total,
            sum(CASE WHEN s.embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_embedding
    """
    result = n4j.query(verify_query)
    if result:
        logger.info(f"Verification: {result[0]['with_embedding']}/{result[0]['total']} summaries have embeddings")

def main():
    parser = argparse.ArgumentParser(description='Add embeddings to Summary nodes')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for progress logging')
    parser.add_argument('--force', action='store_true', help='Recompute all embeddings')
    
    args = parser.parse_args()
    
    # Connect to Neo4j
    n4j = Neo4jGraph(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    logger.info("Connected to Neo4j")
    
    process_summaries(n4j, batch_size=args.batch_size, force=args.force)

if __name__ == "__main__":
    main()