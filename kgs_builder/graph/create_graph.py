# ./kgs_builder/graph/create_graph.py

from http import client
import os
import re
import argparse

from getpass import getpass
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from kgs_builder.data_processing.data_loader import load_high

from kgs_builder.chunking.semantic_chunker import chunk_document
from kgs_builder.utils import *
from helpers.logger import get_logger

logger = get_logger("create_graph", log_file="logs/create_graph.log")

def is_valid_entity(entity_text: str) -> bool:
    """
    Fliter out invalid entities based on certain heuristics.

    Args:
        entity_text (str): The text of the entity to validate.
    Returns:
        bool: True if the entity is valid, False otherwise.
    """

    entity_text = entity_text.strip()

    alpha_count = sum(c.isalpha() for c in entity_text)

    if alpha_count < 2:
        return False
    
    noise_patterns = [
        r'^\[+$',
        r'^\{+$',
        r'^\(+$',
        r'^[\d\s\.,;:]+$',
        r'^[^\w\s]+$',
    ]

    for pattern in noise_patterns:
        if re.match(pattern, entity_text):
            return False

    if len(entity_text) == 1 and not entity_text.isalnum():
        return False
    
    return True

def check_entities_in_bottom_layer(n4j, extracted_entities: dict, gid: str, min_overlap: int = 3) -> tuple:
    """
    Check how many extracted entities exist in the Bottom layer (UMLS/foundational knowledge)
    
    Args:
        n4j: Neo4j connection
        extracted_entities: Dict from NER {entity_class: [entity1, entity2, ...]}
        gid: Current graph ID
        min_overlap: Minimum overlapping entities to consider chunk relevant
    
    Returns:
        (relevant_count, matched_entities, total_entities)
    """
    # Flatten all entities from all classes
    all_entities = []
    for entity_class, entities in extracted_entities.items():
        for entity in entities:
            if is_valid_entity(entity):
                all_entities.append(entity.upper().strip())
    
    if not all_entities:
        return 0, [], 0
    
    total_entities = len(all_entities)
    
        # Query to check which entities exist in Bottom layer (UMLS medical knowledge)
        # Bottom layer = entities with source='UMLS' or canonical rare-disease labels
    # Use UPPER() for case-insensitive matching on n.name
    query = """
    MATCH (n)
    WHERE UPPER(n.name) IN $entity_ids 
      AND (n.source = 'UMLS'
            OR n:DISEASE OR n:DRUG OR n:PHENOTYPE
           OR n:PROCEDURE OR n:ANATOMY OR n:CONCEPT)
    RETURN DISTINCT n.name as matched_entity
    LIMIT 100
    """
    
    try:
        results = n4j.query(query, {'entity_ids': all_entities})
        matched_entities = [r['matched_entity'] for r in results]
        relevant_count = len(matched_entities)
        
        logger.info(f"[NER Filter] {relevant_count}/{total_entities} entities found in Bottom layer")
        if matched_entities:
            logger.debug(f"Sample matches: {matched_entities[:3]}")
        
        return relevant_count, matched_entities, total_entities
    
    except Exception as e:
        logger.warning(f"[NER Filter] Error checking Bottom layer: {e}")
        return 0, [], total_entities

def creat_metagraph(args, content, gid, n4j):
    """
    Create a KG from the given content and add it to Neo4j, with optional NER-based filtering before LLM processing.
    """
    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent()
    whole_chunk = content
        
    logger.info("[NER] Initializing RareDiseaseExtractor model...")
    try:
        from kgs_builder.ner.raredisease_extractor import RareDiseaseExtractor
        ner_extractor = RareDiseaseExtractor()
        logger.info("NER model loaded successfully")
    except Exception as e:
        logger.warning(f"NER model failed to load: {e}")
        logger.warning("Falling back to no-filter mode")
        ner_extractor = None

    # Chunking
    if getattr(args, 'grained_chunk', False):
        logger.info("[Chunking] Using semantic chunking...")
        content = chunk_document(
            content,
            threshold=0.85,
            max_chunk_sentences=15,
            max_chunk_tokens=512,
            log_stats=True
        )
    else:
        logger.info("[Chunking] Using full content (no chunking)...")
        content = [content]
    
    logger.info(f"[Processing] Total chunks to process: {len(content)}")
    
    total_chunks = len(content)
    processed_chunks = 0
    skipped_chunks = 0
    
    for chunk_idx, cont in enumerate(content, 1):
        logger.info(f"[Chunk {chunk_idx}/{total_chunks}] Processing...")
        
        if ner_extractor is not None:
            logger.info(f"[NER] Extracting entities from chunk...")
            try:
                extracted_entities = ner_extractor.extract_entities(cont)
                
                total_extracted = sum(len(v) for v in extracted_entities.values())
                logger.info(f"Extracted {total_extracted} entities across {len(extracted_entities)} classes")
                
                top_classes = sorted(extracted_entities.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                for entity_class, entities in top_classes:
                    logger.info(f"    - {entity_class}: {len(entities)} entities")
                
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")
                extracted_entities = {}
        else:
            extracted_entities = {}
        
        if extracted_entities and getattr(args, 'bottom_filter', False):
            relevant_count, matched_entities, total_entities = check_entities_in_bottom_layer(
                n4j, 
                extracted_entities, 
                gid,
                min_overlap=args.min_overlap if hasattr(args, 'min_overlap') else 3
            )
            
            min_threshold = getattr(args, 'min_overlap', 3)

            # If NER fails to extract entities, do not block processing.
            if total_entities == 0:
                logger.info("PROCESSING: NER extracted 0 entities, fallback to no-filter mode for this chunk")
            else:
                # Avoid impossible gate (e.g., total_entities=2 but min_overlap=5).
                effective_threshold = min(min_threshold, total_entities)
            
                if relevant_count < effective_threshold:
                    logger.info(
                        f"SKIPPING: Only {relevant_count}/{total_entities} entities match Bottom layer "
                        f"(< {effective_threshold}, configured={min_threshold})"
                    )
                    logger.info(f"Matched entities: {matched_entities[:10]}...")
                    skipped_chunks += 1
                    continue
                else:
                    logger.info(
                        f"PROCESSING: {relevant_count}/{total_entities} entities match Bottom layer "
                        f"(threshold={effective_threshold}, configured={min_threshold})"
                    )
                    logger.info(f"Sample matches: {matched_entities[:5]}...")
        
        logger.info(f"[LLM] Calling KnowledgeGraphAgent...")
        
        try:
            element_example = uio.create_element_from_text(text=cont)
            
            # First pass: check if graph extraction makes sense
            ans_str = kg_agent.run(element_example, parse_graph_elements=False)
            
            # Second pass: parse graph elements
            graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
            
            graph_elements = add_ge_emb(graph_elements)
            graph_elements = add_gid(graph_elements, gid)
            
            n4j.add_graph_elements(graph_elements=[graph_elements])
            
            processed_chunks += 1
            logger.info(f"Chunk processed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process chunk: {e}")
            skipped_chunks += 1
            continue
    
    logger.info(f"[Summary] Chunk Processing Statistics:")
    logger.info(f"  Total chunks: {total_chunks}")
    logger.info(f"  Processed: {processed_chunks} ({processed_chunks*100//total_chunks if total_chunks > 0 else 0}%)")
    logger.info(f"  Skipped: {skipped_chunks} ({skipped_chunks*100//total_chunks if total_chunks > 0 else 0}%)")
    logger.info(f"  LLM calls saved: {skipped_chunks * 2}")
    
    if getattr(args, 'ingraphmerge', False) and processed_chunks > 0:
        logger.info("[Post-processing] Merging similar nodes...")
        merge_similar_nodes(n4j, gid)
    
    logger.info("[Post-processing] Creating summary node...")
    add_sum(n4j, whole_chunk, gid)
    
    logger.info(f"\nGraph creation completed for GID: {gid[:8]}...\n")
    
    return n4j