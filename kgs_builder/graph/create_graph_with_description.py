# ./kgs_builder/graph/create_graph_with_description.py

import time
import os
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

from typing import List, Dict
from collections import defaultdict

from kgs_builder.nano_graphrag.prompt import PROMPTS
from kgs_builder.nano_graphrag._utils import compute_mdhash_id
from kgs_builder.nano_graphrag._llm import gemini_complete_if_cache, _get_openrouter_model

from camel.loaders import UnstructuredIO
from kgs_builder.utils import get_embedding, str_uuid, add_sum
from helpers.logger import get_logger

logger = get_logger("create_graph_with_description", log_file="logs/create_graph_with_description.log")


def _run_coro_sync(coro):
    """Run async coroutine safely from sync context (supports nested event-loop environments)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def _normalize_entity_type(entity_type: str) -> str:
    """Map free-form LLM entity types to canonical rare-disease KG labels."""
    t = (entity_type or "").upper().replace("-", "_").replace(" ", "_")

    if any(k in t for k in ["DISEASE", "CONDITION", "SYNDROME", "DISORDER"]):
        return "DISEASE"
    if any(k in t for k in ["SYMPTOM", "PHENOTYPIC", "PHENOTYPE", "SIGN"]):
        return "PHENOTYPE"
    if any(k in t for k in ["GENE", "PROTEIN", "DNA", "RNA", "VARIANT"]):
        return "GENE"
    if any(k in t for k in ["DRUG", "MEDICATION", "CHEMICAL"]):
        return "DRUG"
    if any(k in t for k in ["PROCEDURE", "TEST", "DIAGNOSTIC", "THERAPEUTIC", "LAB"]):
        return "PROCEDURE"
    if any(k in t for k in ["ANATOM", "ORGAN", "TISSUE", "CELL"]):
        return "ANATOMY"
    if any(k in t for k in ["ORGANISM", "TAXON", "PATHOGEN", "VIRUS", "BACTERIA"]):
        return "ORGANISM"

    return "CONCEPT"

def clean_str(input_str: str) -> str:
    """
    Clean the input string by removing unwanted characters and normalizing whitespace.
    
    Args:
        input_str (str): The string to be cleaned.
    Returns:
        str: The cleaned string.
    """
    if not input_str:
        return ""
    
    input_str = input_str.strip().strip('"').strip("'")
    return input_str

async def extract_entities_with_description(content: str, entity_types=None):
    """
    Extract entities along with their descriptions from the input content using LLM.
    
    Args:
        content (str): text content to extract entities from
        entity_types (List[str], optional): List of entity types to extract. If None, extract all types. Defaults to None.
    Returns:
        (entities, relationships)
        entities: [{'entity_name': ..., 'entity_type': ..., 'description': ...}, ...]
        relationships: [{'src': ..., 'tgt': ..., 'description': ..., 'strength': ...}, ...]
    """

    if entity_types is None:
        entity_types = [
            "Disease", "Symptom", "Treatment", "Medication", "Test", 
            "Anatomy", "Procedure", "Condition", "Measurement", "Hormone",
            "Diagnostic_Criteria", "Clinical_Guideline", "Patient", "Doctor"
        ]

    prompt_template = PROMPTS["entity_extraction"]

    entity_type_str = ", ".join(entity_types)
    tuple_delimiter = "<|>"
    record_delimiter = "##"
    completion_delimiter = "<|COMPLETE|>"

    # Build the prompt for LLM
    prompt = prompt_template.format(
        # content=content,
        input_text=content[:3000], # limit
        entity_types=entity_type_str,
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        completion_delimiter=completion_delimiter
    )

    logger.info(f"  [Entity Extraction] Extracting entities and relationships...")
    provider = os.getenv("LLM_PROVIDER") or "openrouter"
    model = _get_openrouter_model()
    response = await gemini_complete_if_cache(
        model=model,
        prompt=prompt,
        provider=provider,
        system_prompt="You are a helpful assistant that extracts entities and relationships from medical texts."
    )

    entities = []
    relationships = []

    if not response:
        return entities, relationships
    
    records = response.split(record_delimiter)

    for record in records:
        record = record.strip()
        if not record or completion_delimiter in record:
            continue

        match = re.search(r'\((.*?)\)', record)
        if not match:
            continue

        record_content = match.group(1)
        attributes = record_content.split(tuple_delimiter)
        
        if len(attributes) < 2:
            continue

        record_type = clean_str(attributes[0])

        if record_type == "entity" and len(attributes) >= 4:
            # Entity record: ("entity"<|>entity_name<|>entity_type<|>entity_description)
            entity = {
                'entity_name': clean_str(attributes[1]).upper(),
                'entity_type': clean_str(attributes[2]).upper(),
                'description': clean_str(attributes[3])
            }
            if entity['entity_name']:
                entities.append(entity)

        elif record_type == "relationship" and len(attributes) >= 5:
            # Relationship record: ("relationship"<|>src_entity<|>tgt_entity<|>relationship_description<|>relationship_strength)
            relationship = {
                'src': clean_str(attributes[1]).upper(),
                'tgt': clean_str(attributes[2]).upper(),
                'description': clean_str(attributes[3]),
                'strength': clean_str(attributes[4])
            }
            if relationship['src'] and relationship['tgt']:
                relationships.append(relationship)

    logger.info(f"  [Entity Extraction] Extracted {len(entities)} entities and {len(relationships)} relationships.")
    return entities, relationships

def create_neo4j_nodes_and_relationships(n4j, entities: List[Dict], relationships: List[Dict], gid: str):
    """
    Create Neo4j nodes and relationships based on the extracted entities and relationships.
    
    Args:
        n4j: Neo4j connection
        entities: List of entity dicts with keys 'entity_name', 'entity_type', 'description'
        relationships: List of relationship dicts with keys 'src', 'tgt', 'description', 'strength'
        gid: Current graph ID
    """
    logger.info(f"  [Neo4j Creation] Creating nodes and relationships in Neo4j...")

    for entity in entities:
        entity_name = entity['entity_name']
        entity_type = _normalize_entity_type(entity.get('entity_type', ''))
        description = entity['description']

        embedding_text = f"{entity_name}): {description}" if description else entity_name
        embedding = get_embedding(embedding_text)

        create_node_query = """
        MERGE (n:`%s` {id: $id, gid: $gid})
        ON CREATE SET 
            n.description = $description,
            n.embedding = $embedding,
            n.source = 'nano_graphrag'
        ON MATCH SET
            n.description = CASE WHEN n.description IS NULL OR n.description = '' 
                                    THEN $description 
                                    ELSE n.description END,
            n.embedding = CASE WHEN n.embedding IS NULL 
                                THEN $embedding 
                                ELSE n.embedding END
        RETURN n
        """ % entity_type

        try:
            n4j.query(create_node_query, {
                'id': entity_name,
                'gid': gid,
                'description': description,
                'embedding': embedding
            })
        except Exception as e:
            logger.warning(f"    [Neo4j Creation] Error creating node for entity '{entity_name}': {e}")
        
    logger.info(f"  [Neo4j Creation] Created/updated {len(entities)} nodes.")

    for rel in relationships:
        src = rel['src']
        tgt = rel['tgt']
        rel_type = "RELATED_TO"

        description = rel.get('description', '')
        desc_lower = description.lower()

        if 'treat' in desc_lower or 'cure' in desc_lower:
            rel_type = "TREATS"
        elif 'cause' in desc_lower or 'lead' in desc_lower:
            rel_type = "CAUSES"
        elif 'diagnose' in desc_lower or 'indicate' in desc_lower:
            rel_type = "INDICATES"
        elif 'symptom' in desc_lower or 'manifest' in desc_lower:
            rel_type = "HAS_MANIFESTATION"
        
        create_rel_query = """
        MATCH (a {id: $src, gid: $gid})
        MATCH (b {id: $tgt, gid: $gid})
        MERGE (a)-[r:`%s`]->(b)
        ON CREATE SET r.description = $description, r.strength = $strength
        RETURN r
        """ % rel_type

        try:
            n4j.query(create_rel_query, {
                'src': src,
                'tgt': tgt,
                'gid': gid,
                'description': description,
                'strength': rel.get('strength', '')
            })
        except Exception as e:
            logger.warning(f"    [Neo4j Creation] Error creating relationship from '{src}' to '{tgt}': {e}")

    logger.info(f"  [Neo4j Creation] Created/updated {len(relationships)} relationships.")

def creat_metagraph_with_description(args, content: str, gid: str, n4j):
    """
    Create knowledge graph using nano_graphrag's extraction logic (with description)
    But writes to Neo4j and supports three-layer architecture (gid)
    
    IMPROVEMENTS:
    - Semantic chunking (embedding-based)
    - NER-based filtering (skip chunks with low Bottom layer overlap)
    - Incremental Middle→Bottom linking
    
    Args:
        args: Arguments
        content: Text content
        gid: Graph ID (for three-layer architecture)
        n4j: Neo4j connection
    """

    logger.info(f"[Graph Construction] Starting knowledge graph construction (GID: {gid[:8]}...)")

    ner_extractor = None
    if hasattr(args, 'bottom_filter') and args.bottom_filter:
        logger.info("[NER] Initializing RareDiseaseExtractor for filtering...")
        try:
            from kgs_builder.ner.raredisease_extractor import RareDiseaseExtractor
            ner_extractor = RareDiseaseExtractor()
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.warning(f"NER model failed to load: {e}")
            logger.warning("  Falling back to no-filter mode")
    
    if getattr(args, 'grained_chunk', False):
        from kgs_builder.chunking.semantic_chunker import chunk_document

        logger.info("[Chunking] Using semantic chunking (embedding-based)...")
        content_chunks = chunk_document(
            content,
            threshold=0.85,
            max_chunk_sentences=15,
            max_chunk_tokens=512,
            log_stats=True
        )
    else:
        logger.info("[Chunking] Using full content...")
        content_chunks = [content]
    
    total_chunks = len(content_chunks)
    processed_chunks = 0
    skipped_chunks = 0
    
    all_entities = []
    all_relationships = []
    
    for idx, chunk in enumerate(content_chunks, 1):
        logger.info(f"\n[Chunk {idx}/{total_chunks}] Processing...")
        
        if ner_extractor is not None:
            logger.info(f"  [NER Filter] Checking Bottom layer overlap...")
            try:
                from .create_graph import check_entities_in_bottom_layer
                
                extracted_ner = ner_extractor.extract_entities(chunk)
                
                min_threshold = getattr(args, 'min_overlap', 3)
                relevant_count, matched_entities, total_entities = check_entities_in_bottom_layer(
                    n4j, extracted_ner, gid, min_overlap=min_threshold
                )

                # If NER fails to extract entities, do not block processing.
                if total_entities == 0:
                    logger.info("PROCESSING: NER extracted 0 entities, fallback to no-filter mode for this chunk")
                    continue

                # Avoid impossible gate.
                effective_threshold = min(min_threshold, total_entities)
                
                if relevant_count < effective_threshold:
                    logger.info(
                        f"SKIPPING: Only {relevant_count}/{total_entities} entities match Bottom layer "
                        f"(< {effective_threshold}, configured={min_threshold})"
                    )
                    logger.info(f"LLM calls saved: 2 (entity extraction + relationship extraction)")
                    skipped_chunks += 1
                    continue
                else:
                    logger.info(
                        f"PROCESSING: {relevant_count}/{total_entities} entities match Bottom layer "
                        f"(threshold={effective_threshold}, configured={min_threshold})"
                    )
            
            except Exception as e:
                logger.warning(f"NER filtering failed: {e}")
 
        entities, relationships = _run_coro_sync(
            extract_entities_with_description(chunk)
        )
        
        all_entities.extend(entities)
        all_relationships.extend(relationships)
        processed_chunks += 1
    
    logger.info(f"\n[Extraction Summary] Total extracted:")
    logger.info(f"  - Entities: {len(all_entities)}")
    logger.info(f"  - Relationships: {len(all_relationships)}")

    # Merge duplicate entities
    logger.info(f"\n[Merging] Merging duplicate entities...")
    entity_dict = {}
    for entity in all_entities:
        name = entity['entity_name']
        if name in entity_dict:
            # Merge descriptions
            existing_desc = entity_dict[name]['description']
            new_desc = entity['description']
            if new_desc and new_desc not in existing_desc:
                entity_dict[name]['description'] = f"{existing_desc}; {new_desc}"
        else:
            entity_dict[name] = entity
    
    merged_entities = list(entity_dict.values())
    logger.info(f"After merging: {len(merged_entities)} entities")
    logger.info(f"\n[Writing to Neo4j] Starting...")
    create_neo4j_nodes_and_relationships(n4j, merged_entities, all_relationships, gid)
    
    logger.info(f"\n[Incremental Linking] Creating Middle→Bottom references...")
    from kgs_builder.core.linking import link_middle_to_bottom_incremental
    links_created = link_middle_to_bottom_incremental(n4j, merged_entities, gid)
    logger.info(f"{links_created} IS_REFERENCE_OF links created")
    
    if getattr(args, 'ingraphmerge', False):
        logger.info(f"\n[In-Graph Merge] Merging similar nodes...")
        from kgs_builder.utils import merge_similar_nodes
        merge_similar_nodes(n4j, gid)
    
    logger.info(f"\n[Summary] Creating summary node...")
    add_sum(n4j, content, gid)

    logger.info(f"[Graph Construction] Completed! (GID: {gid[:8]}...)")
    return n4j