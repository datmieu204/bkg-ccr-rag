# ./kgs_builder/core/linking.py

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from helpers.logger import get_logger

logger = get_logger("linking", log_file="logs/linking.log")

def link_middle_to_bottom_incremental(n4j, entities: List[Dict], middle_gid: str) -> int:
    """
    Incremental linking: Create IS_REFERENCE_OF links immediately after extraction
    Called right after uploading graph_elements to Neo4j
    
    Flow:
    1. Extract entity names from graph_elements
    2. Find matching Bottom layer entities
    3. Create IS_REFERENCE_OF: (Middle)-[:IS_REFERENCE_OF]->(Bottom)
    
    Args:
        n4j: Neo4j connection
        entities: List of entity dicts [{'entity_name': ..., 'entity_type': ...}, ...]
        middle_gid: Middle layer GID (current document being processed)
    
    Returns:
        int: Number of links created
    """
    if not entities:
        logger.debug(f"  [Incremental Linking] No entities for GID {middle_gid[:8]}...")
        return 0
    
    logger.info(f"  [Incremental Linking] Linking Middle→Bottom for GID {middle_gid[:8]}...")
    
    entity_names = []
    for e in entities:
        name = e.get('entity_name', '').strip()
        if name and len(name) >= 2:
            entity_names.append(name.upper())
    
    if not entity_names:
        logger.debug("    No valid entity names")
        return 0
    
    logger.info(f"    Processing {len(entity_names)} entities...")
    logger.debug(f"    Sample: {entity_names[:5]}")
    
    batch_link_query = """
    UNWIND $entity_names AS entity_name
    // Find Bottom entity
    MATCH (b)
    WHERE UPPER(b.name) = entity_name
      AND (b.source = 'UMLS'
                     OR b:DISEASE OR b:DRUG OR b:PHENOTYPE
           OR b:PROCEDURE OR b:ANATOMY OR b:CONCEPT)
    
    // Find Middle entity with same name in current GID
    MATCH (m {gid: $middle_gid})
    WHERE UPPER(m.id) = entity_name
    
    // Create IS_REFERENCE_OF link
    MERGE (m)-[r:IS_REFERENCE_OF]->(b)
    ON CREATE SET r.created_at = datetime(), r.method = 'incremental'
    
    RETURN count(DISTINCT r) AS links_created
    """
    
    try:
        result = n4j.query(batch_link_query, {
            'entity_names': entity_names,
            'middle_gid': middle_gid
        })
        
        links_created = result[0]['links_created'] if result else 0
        
        if links_created > 0:
            logger.info(f"Created {links_created} IS_REFERENCE_OF links")
        else:
            logger.debug(f"No Bottom entities matched")
        
        return links_created
        
    except Exception as e:
        logger.warning(f"Incremental linking failed: {e}")
        return 0
    
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    """
    if vec1 is None or vec2 is None:
        return 0.0
    
    vec1 = np.asarray(vec1, dtype=float)
    vec2 = np.asarray(vec2, dtype=float)
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def extract_entities_from_top_layer(n4j, top_gid: str, ner_extractor=None) -> Set[str]:
    """
    Extract entities from Top layer nodes using NER
    
    Args:
        n4j: Neo4j connection
        top_gid: Top layer graph ID
        ner_extractor: RareDiseaseExtractor instance (lazy loaded if None)
    
    Returns:
        Set of entity names (uppercase)
    """
    logger.info(f"\n[Step 1] Extracting entities from Top layer (GID: {top_gid[:8]}...)")
    
    # Get all content from Top layer nodes (except Summary)
    query = """
    MATCH (n)
    WHERE n.gid = $gid AND NOT n:Summary
    RETURN n.id as node_id, n.description as description
    """
    
    nodes = n4j.query(query, {'gid': top_gid})
    
    if not nodes:
        logger.warning(f"No nodes found for GID: {top_gid[:8]}...")
        return set()
    
    if ner_extractor is None:
        try:
            from kgs_builder.ner.raredisease_extractor import RareDiseaseExtractor
            ner_extractor = RareDiseaseExtractor()
            logger.info("NER model loaded")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            return set()
    
    all_entities = set()
    
    for node in nodes:
        text = f"{node['node_id']} {node.get('description', '')}"
        
        try:
            extracted = ner_extractor.extract_entities(text)
            
            for entity_class, entities in extracted.items():
                for entity in entities:
                    entity_clean = entity.strip().upper()
                    if len(entity_clean) >= 2:  # Minimum length
                        all_entities.add(entity_clean)
        
        except Exception as e:
            logger.warning(f"NER extraction failed for node {node['node_id']}: {e}")
            continue
    
    logger.info(f"Extracted {len(all_entities)} unique entities from Top layer")
    logger.info(f"     Sample entities: {list(all_entities)[:5]}")
    
    return all_entities

def find_middle_chunks_via_bottom(
    n4j,
    top_entities: Set[str],
    top_k: int = 50,
    middle_layer_gids: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Find Middle layer chunks that reference the same Bottom layer entities
    
    Args:
        n4j: Neo4j connection
        top_entities: Set of entity names from Top layer
        top_k: Maximum number of Middle chunks to return
    
    Returns:
        List of Middle layer nodes with their metadata
    """
    logger.info(f"\n[Step 2] Finding Middle layer chunks via Bottom layer")
    logger.info(f"Searching for {len(top_entities)} entities in Bottom layer...")
    
    entity_list = list(top_entities)
    
    # Find Middle chunks that reference Bottom entities matching our Top entities
    query = """
    // Step 1: Find Bottom entities matching Top entities
    UNWIND $entity_names AS entity_name
    MATCH (b)
    WHERE UPPER(b.name) = entity_name
      AND (b.source = 'UMLS'
                     OR b:DISEASE OR b:DRUG OR b:PHENOTYPE
           OR b:PROCEDURE OR b:ANATOMY OR b:CONCEPT)
    
    // Step 2: Find Middle nodes that have IS_REFERENCE_OF to these Bottom entities
    WITH collect(DISTINCT b) AS bottom_entities
    UNWIND bottom_entities AS b
    MATCH (m)-[:IS_REFERENCE_OF]->(b)
        WHERE NOT m:Summary
            AND m.embedding IS NOT NULL
            AND ($middle_gids IS NULL OR m.gid IN $middle_gids)
    
    // Step 3: Count how many Bottom entities each Middle node references
    WITH m, count(DISTINCT b) AS entity_count
    
    // Step 4: Return Middle nodes sorted by relevance
    RETURN m.gid AS middle_gid,
           m.id AS middle_id,
           m.embedding AS embedding,
           m.description AS description,
           entity_count
    ORDER BY entity_count DESC
    LIMIT $top_k
    """
    
    try:
        results = n4j.query(query, {
            'entity_names': entity_list,
            'top_k': top_k,
            'middle_gids': middle_layer_gids if middle_layer_gids else None,
        })
        
        middle_chunks = []
        for r in results:
            middle_chunks.append({
                'gid': r['middle_gid'],
                'id': r['middle_id'],
                'embedding': r['embedding'],
                'description': r.get('description', ''),
                'entity_count': r['entity_count']
            })
        
        logger.info(f"Found {len(middle_chunks)} Middle layer chunks")
        if middle_chunks:
            logger.info(f"     Top chunk has {middle_chunks[0]['entity_count']} overlapping entities")
            logger.info(f"     Sample: {middle_chunks[0]['id'][:50]}...")
        
        return middle_chunks
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []
    
def filter_by_cosine_similarity(
    n4j, 
    top_gid: str, 
    middle_chunks: List[Dict], 
    threshold: float = 0.6
) -> List[Tuple[str, str, str, float]]:
    """
    Filter Middle chunks by cosine similarity with Top layer embedding
    
    Args:
        n4j: Neo4j connection
        top_gid: Top layer graph ID
        middle_chunks: List of Middle layer candidates
        threshold: Minimum similarity threshold
    
    Returns:
        List of (top_node_id, middle_gid, middle_id, similarity) tuples above threshold
    """
    logger.info(f"\n[Step 3] Filtering by cosine similarity (threshold={threshold})")
    
    # Get Top layer embeddings
    top_query = """
    MATCH (t)
    WHERE t.gid = $gid AND NOT t:Summary AND t.embedding IS NOT NULL
    RETURN t.id AS top_id, t.embedding AS embedding
    """
    
    top_nodes = n4j.query(top_query, {'gid': top_gid})
    
    if not top_nodes:
        logger.warning(f"No Top layer nodes with embeddings found")
        return []
    
    links = []
    
    for top_node in top_nodes:
        top_emb = top_node['embedding']
        top_id = top_node['top_id']
        
        for middle_chunk in middle_chunks:
            middle_emb = middle_chunk['embedding']
            middle_gid = middle_chunk['gid']
            
            # Calculate cosine similarity
            sim = cosine_similarity(top_emb, middle_emb)
            
            if sim >= threshold:
                links.append((top_id, middle_gid, middle_chunk.get('id', ''), sim))
    
    links.sort(key=lambda x: x[3], reverse=True)
    
    logger.info(f"Found {len(links)} valid links above threshold")
    if links:
        logger.info(f"     Best similarity: {links[0][3]:.3f}")
        logger.info(f"     Worst similarity: {links[-1][3]:.3f}")
    
    return links


def create_reference_relationships(
    n4j, 
    top_gid: str, 
    links: List[Tuple[str, str, str, float]]
) -> int:
    """
    Create REFERENCE relationships in Neo4j
    
    Args:
        n4j: Neo4j connection
        top_gid: Top layer graph ID
        links: List of (top_node_id, middle_gid, middle_id, similarity) tuples
    
    Returns:
        Number of relationships created
    """
    logger.info(f"\n[Step 4] Creating REFERENCE relationships")
    
    if not links:
        logger.warning("No links to create")
        return 0
    
    # Batch create relationships
    create_query = """
    UNWIND $links AS link
    MATCH (t {id: link.top_id, gid: $top_gid})
    MATCH (m {gid: link.middle_gid})
        WHERE NOT m:Summary
            AND (link.middle_id = '' OR m.id = link.middle_id)
    MERGE (t)-[r:REFERENCE]->(m)
    ON CREATE SET r.similarity = link.similarity, r.method = 'smart_linking'
    ON MATCH SET r.similarity = link.similarity, r.method = 'smart_linking'
        RETURN count(DISTINCT r) AS created
    """
    
    links_data = [
        {'top_id': top_id, 'middle_gid': middle_gid, 'middle_id': middle_id or '', 'similarity': sim}
        for top_id, middle_gid, middle_id, sim in links
    ]
    
    try:
        result = n4j.query(create_query, {
            'top_gid': top_gid,
            'links': links_data
        })
        
        created = result[0]['created'] if result else 0
        logger.info(f"Created {created} REFERENCE relationships")
        
        return created
    
    except Exception as e:
        logger.error(f"Failed to create relationships: {e}")
        return 0


def smart_ref_link(
    n4j, 
    top_gid: str, 
    middle_layer_gids: List[str] = None,
    top_k: int = 50,
    similarity_threshold: float = 0.6,
    ner_extractor=None
) -> Dict:
    """
    Smart reference linking using entity-based filtering and cosine similarity
    
    This is the main function that implements the improved linking strategy:
    1. Extract entities from Top layer using NER
    2. Find Middle chunks via Bottom layer entity overlap
    3. Filter by cosine similarity
    4. Create optimized REFERENCE relationships
    
    Args:
        n4j: Neo4j connection
        top_gid: Top layer graph ID
        middle_layer_gids: Optional list of Middle layer GIDs to search
                          If None, searches all Middle layer nodes
        top_k: Maximum Middle chunks to consider (default: 50)
        similarity_threshold: Minimum cosine similarity (default: 0.6)
        ner_extractor: Optional RareDiseaseExtractor instance (lazy loaded if None)
    
    Returns:
        Dict with statistics: {
            'entities_extracted': int,
            'middle_chunks_found': int,
            'links_created': int,
            'best_similarity': float
        }
    """
    logger.info(f"[Smart Linking] Top GID: {top_gid[:8]}...")
    
    top_entities = extract_entities_from_top_layer(n4j, top_gid, ner_extractor)
    
    if not top_entities:
        logger.warning("No entities extracted, aborting")
        return {
            'entities_extracted': 0,
            'middle_chunks_found': 0,
            'links_created': 0,
            'best_similarity': 0.0
        }
    
    middle_chunks = find_middle_chunks_via_bottom(
        n4j,
        top_entities,
        top_k=top_k,
        middle_layer_gids=middle_layer_gids,
    )
    
    if not middle_chunks:
        logger.warning("No Middle chunks found, aborting")
        return {
            'entities_extracted': len(top_entities),
            'middle_chunks_found': 0,
            'links_created': 0,
            'best_similarity': 0.0
        }
    
    links = filter_by_cosine_similarity(n4j, top_gid, middle_chunks, similarity_threshold)
    
    if not links:
        logger.warning("No links above similarity threshold")
        return {
            'entities_extracted': len(top_entities),
            'middle_chunks_found': len(middle_chunks),
            'links_created': 0,
            'best_similarity': 0.0
        }
    
    links_created = create_reference_relationships(n4j, top_gid, links)
    
    best_sim = links[0][3] if links else 0.0
    
    stats = {
        'entities_extracted': len(top_entities),
        'middle_chunks_found': len(middle_chunks),
        'links_created': links_created,
        'best_similarity': best_sim
    }
    
    logger.info(f"[Summary] Smart Linking Statistics:")
    logger.info(f"  Entities extracted: {stats['entities_extracted']}")
    logger.info(f"  Middle chunks found: {stats['middle_chunks_found']}")
    logger.info(f"  Links created: {stats['links_created']}")
    logger.info(f"  Best similarity: {stats['best_similarity']:.3f}")
    
    return stats