# ./bkg-ccr-rag/kgs_builder/data_processing/data_import_umls.py

import re
import csv

from typing import List, Dict, Tuple
from kgs_builder.utils import get_embedding
from collections import Counter

from helpers.logger import get_logger
logger = get_logger("import_umls", log_file="logs/import_umls.log")


def _infer_entity_type(name: str, definition: str) -> str:
    """
    Infering entity type based on keywords in the name and definition.
    """
    text = f"{name or ''} {definition or ''}".lower()

    def _has_any(words: List[str]) -> bool:
        return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)

    keyword_groups = {
        "GENE": [
            "gene", "genetic", "genotype", "allele", "mutation", "variant",
            "protein", "enzyme", "transcript", "mrna", "rna", "dna",
        ],
        "DISEASE": [
            "disease", "disorder", "syndrome", "infection", "deficiency",
            "failure", "cancer", "carcinoma", "neoplasm", "tumor", "tumour",
            "hepatitis", "anemia", "anaemia", "pathology",
        ],
        "PHENOTYPE": [
            "symptom", "sign", "manifestation", "phenotype", "clinical feature",
            "finding", "abnormality", "impairment", "pain", "fever",
        ],
        "DRUG": [
            "drug", "medication", "medicine", "compound", "chemical", "agent",
            "therapy", "therapeutic", "treatment", "antibiotic", "vaccine",
        ],
        "PROCEDURE": [
            "procedure", "diagnosis", "diagnostic", "screening", "test",
            "assay", "imaging", "surgery", "surgical", "transplant", "biopsy",
        ],
        "ANATOMY": [
            "organ", "tissue", "cell", "anatomy", "anatomical", "artery", "vein",
            "liver", "heart", "brain", "lung", "kidney", "bone", "muscle",
            "epithelium",
        ],
        "ORGANISM": [
            "virus", "bacteria", "bacterium", "fungus", "parasite", "pathogen",
            "species", "organism", "microbe",
        ],
    }

    # Strong disease suffix patterns for biomedical terms in rare-disease corpora.
    if re.search(r"\b\w+(itis|osis|emia|opathy|plasia)\b", text):
        return "DISEASE"

    # Use priority to keep disease-centric entities dominant for rare-disease KG.
    for entity_type in ["DISEASE", "GENE", "PHENOTYPE", "DRUG", "PROCEDURE", "ANATOMY", "ORGANISM"]:
        if _has_any(keyword_groups[entity_type]):
            return entity_type

    return "CONCEPT"

def _map_relationship_type(rel: str, rela: str) -> str:
    """
    Mapping UMLS REL and RELA to a more standardized relationship type.
    """

    def _normalize_text(value: str) -> str:
        text = (value or "").strip().lower()
        if text in {"", "nan", "none", "null", "na"}:
            return ""
        return text

    def _to_rel_label(value: str) -> str:
        cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", value.upper()).strip("_")
        return cleaned if cleaned else "RELATED_TO"

    rel_code = _normalize_text(rel).upper()
    rela_name = _normalize_text(rela)

    # Skip deleted relationships by source assertion.
    if rel_code == "DEL":
        return "SKIP"

    # RELA is more specific than REL, so prefer it when available.
    if rela_name:
        rela_exact_map = {
            "clinically_associated_with": "ASSOCIATED_WITH",
            "associated_with": "ASSOCIATED_WITH",
            "member_of": "PART_OF",
            "has_member": "HAS_PART",
            "classified_as": "IS_A",
            "isa": "IS_A",
            "is_a": "IS_A",
            "has_translation": "HAS_TRANSLATION",
            "translation_of": "TRANSLATION_OF",
            "mapped_to": "SAME_AS",
            "same_as": "SAME_AS",
            "synonym_of": "SAME_AS",
        }
        if rela_name in rela_exact_map:
            return rela_exact_map[rela_name]

        if "associat" in rela_name or "related" in rela_name:
            return "ASSOCIATED_WITH"
        if "caus" in rela_name or "etiolog" in rela_name:
            return "CAUSES"
        if any(k in rela_name for k in ["treat", "therapy", "prevent", "manage"]):
            return "TREATS"
        if any(k in rela_name for k in ["diagnos", "screen", "test", "assay"]):
            return "DIAGNOSES"
        if any(k in rela_name for k in ["symptom", "sign", "manifest", "phenotype", "presentation"]):
            return "HAS_MANIFESTATION"
        if any(k in rela_name for k in ["site", "location", "anatom", "region", "organ"]):
            return "LOCATED_IN"
        if any(k in rela_name for k in ["part", "member", "component", "structure"]):
            if rela_name.startswith("has_"):
                return "HAS_PART"
            return "PART_OF"

        return _to_rel_label(rela_name)

    rel_map = {
        "RO": "RELATED_TO",
        "RQ": "ASSOCIATED_WITH",
        "RU": "RELATED_TO",
        "SY": "SAME_AS",
        "PAR": "HAS_PARENT",
        "CHD": "HAS_CHILD",
        "RN": "NARROWER_THAN",
        "RB": "BROADER_THAN",
        "AQ": "QUALIFIED_BY",
        "QB": "CAN_QUALIFY",
        "RL": "SIMILAR_TO",
        "XR": "NOT_RELATED",
    }

    return rel_map.get(rel_code, "RELATED_TO")

def parse_umls_csv(file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse the UMLS CSV file and extract concepts and relationships.
    Args:
        file_path (str): Path to the UMLS CSV file.
    Returns:
        (entities, relationships):
        entities [{'id': CUI, 'name': name, 'type': type, 'description': def}, ...]
        relationships [{'src': CUI1, 'tgt': CUI2, 'type': RELA, 'description': desc}, ...]
    """

    logger.info(f"Parsing UMLS CSV file: {file_path}")

    entities_dict = {}
    relationships = []
    relation_seen = set()
    relation_type_counter = Counter()
    rela_fallback_counter = Counter()
    skipped_relationships = 0

    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for idx, row in enumerate(reader, start=1):
                try: 
                    # extract entity 1
                    cui1 = row.get('CUI1', '').strip()
                    name1 = row.get('name_1', '').strip()
                    def1 = row.get('def_1', '').strip()
                    # type1 = row.get('type_1', '').strip()

                    cui2 = row.get('CUI2', '').strip()
                    name2 = row.get('name_2', '').strip()
                    def2 = row.get('def_2', '').strip()
                    # type2 = row.get('type_2', '').strip()

                    rel = row.get('REL', '').strip()
                    rela = row.get('RELA', '').strip()
                    sab = row.get('SAB', '').strip()

                    if not cui1 or not cui2:
                        continue

                    if cui1 not in entities_dict:
                        entity_type = _infer_entity_type(name1, def1)
                        entities_dict[cui1] = {
                            'id': cui1,
                            'name': name1.upper(),
                            'type': entity_type,
                            'description': def1
                        }

                    if cui2 not in entities_dict:
                        entity_type = _infer_entity_type(name2, def2)
                        entities_dict[cui2] = {
                            'id': cui2,
                            'name': name2.upper(),
                            'type': entity_type,
                            'description': def2
                        }

                    rel_type = _map_relationship_type(rel, rela)
                    if rel_type == "SKIP":
                        skipped_relationships += 1
                        continue

                    rel_desc = f"{rel} ({rela}) from {sab}" if rela else rel

                    dedupe_key = (cui1, cui2, rel_type, sab)
                    if dedupe_key in relation_seen:
                        continue
                    relation_seen.add(dedupe_key)

                    relation_type_counter[rel_type] += 1
                    if rela and rel_type == re.sub(r"[^0-9a-zA-Z]+", "_", rela.upper()).strip("_"):
                        rela_fallback_counter[rela] += 1

                    relationships.append({
                        'src': cui1,
                        'tgt': cui2,
                        'type': rel_type,
                        'description': rel_desc,
                        'source': sab,
                        'raw_rel': rel,
                        'raw_rela': rela,
                    })
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {e}")
                    continue

        entities = list(entities_dict.values())
        logger.info(f"Finished parsing UMLS CSV. Total entities: {len(entities)}, Total relationships: {len(relationships)}")
        logger.info(f"Skipped relationships (DEL): {skipped_relationships}")
        logger.info(f"Top mapped relation types: {relation_type_counter.most_common(15)}")
        if rela_fallback_counter:
            logger.info(f"Top RELA fallback labels (unmapped exact rules): {rela_fallback_counter.most_common(15)}")
        return entities, relationships
                        
    except Exception as e:
        logger.error(f"Error processing row {idx}: {e}")
        return [], []

def create_umls_nodes_and_relationships(n4j, entities: List[Dict], relationships: List[Dict], gid: str):
    """
    Create nodes and relationships in Neo4j based on the parsed UMLS data.
    Args:
        n4j: Neo4j connection/session object.
        entities: List of entity dictionaries.
        relationships: List of relationship dictionaries.
        gid: Graph ID for partitioning.
    """

    import gc

    BATCH_SIZE = 1000

    logger.info(f"Creating nodes and relationships in Neo4j with batch size {BATCH_SIZE}")
    logger.info(f"Total entities: {len(entities)}, Total relationships: {len(relationships)}")

    created_nodes = 0

    # 1. Create nodes in batches
    total_batches = (len(entities) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(entities), BATCH_SIZE):
        batch_entities = entities[batch_idx: batch_idx + BATCH_SIZE]
        batch_num = (batch_idx // BATCH_SIZE) + 1

        logger.info(f"Processing batch {batch_num}/{total_batches} for nodes...")

        for entity in batch_entities:
            entity_id = entity['id'] # CUI
            entity_name = entity['name']
            entity_type = entity['type']
            description = entity['description']

            # Embedding
            embedding_text = f"{entity_name}: {description}" if description else entity_name
            embedding = get_embedding(embedding_text)

            # Cypher query to create node
            create_node_query = """
            MERGE (n:`%s` {id: $id, gid: $gid})
            ON CREATE SET 
                n.cui = $cui,
                n.name = $name,
                n.description = $description,
                n.embedding = $embedding,
                n.source = 'UMLS',
                n.data_type = 'structured'
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
                result = n4j.query(create_node_query, {
                    'id': entity_name,
                    'cui': entity_id,
                    'name': entity_name,
                    'description': description,
                    'embedding': embedding,
                    'gid': gid
                })
                if result:
                    created_nodes += 1
            except Exception as e:
                logger.error(f"Error creating node for entity {entity_id} ({entity_name}): {e}")

        gc.collect()
        logger.info(f"Finished batch {batch_num}/{total_batches}. Created/updated nodes so far: {created_nodes}")

    logger.info(f"Finished creating nodes. Total created/updated nodes: {created_nodes}/{len(entities)}")

    # 2. Create relationships in batches
    logger.info(f"Creating relationships in Neo4j with batch size {BATCH_SIZE}")
    
    REALTION_BATCH__SIZE = 1000

    created_rels = 0
    total_rel_batches = (len(relationships) + REALTION_BATCH__SIZE - 1) // REALTION_BATCH__SIZE

    for batch_idx in range(0, len(relationships), REALTION_BATCH__SIZE):
        batch_rels = relationships[batch_idx: batch_idx + REALTION_BATCH__SIZE]
        batch_num = (batch_idx // REALTION_BATCH__SIZE) + 1

        logger.info(f"Processing batch {batch_num}/{total_rel_batches} for relationships...")

        for rel in batch_rels:
            src_id = rel['src']
            tgt_id = rel['tgt']
            rel_type = rel['type']
            rel_desc = rel['description']

            # Cypher query to create relationship
            create_rel_query = """
            MATCH (a {cui: $src_cui, gid: $gid})
            MATCH (b {cui: $tgt_cui, gid: $gid})
            MERGE (a)-[r:`%s`]->(b)
            ON CREATE SET 
                r.description = $description,
                r.source = $source
            RETURN r
            """ % rel_type

            try:
                result = n4j.query(create_rel_query, {
                    'src_cui': src_id,
                    'tgt_cui': tgt_id,
                    'description': rel_desc,
                    'source': rel.get('source', 'UMLS'),
                    'gid': gid
                })
                if result:
                    created_rels += 1
            except Exception as e:
                logger.error(f"Error creating relationship from {src_id} to {tgt_id}: {e}")

        gc.collect()
        logger.info(f"Finished batch {batch_num}/{total_rel_batches}. Created/updated relationships so far: {created_rels}")
    logger.info(f"Finished creating relationships. Total created/updated relationships: {created_rels}/{len(relationships)}")

def import_umls_csv_to_neo4j(file_path: str, gid: str, n4j) -> bool:
    """
    Import UMLS CSV to Neo4j
    Args:
        file_path: CSV file path
        gid: Graph ID for partitioning
        n4j: Neo4j connection/session object
    Returns:
        success (bool): True if import is successful, False otherwise
    """
    logger.info(f"Starting import of UMLS CSV to Neo4j: {file_path}")
    logger.info(f"Graph ID: {gid[:8]}...")

    try:
        # 1. CSV
        entities, relationships = parse_umls_csv(file_path)
        if not entities:
            logger.warning(f"No entities found in the CSV file: {file_path}")
            return False
        
        # 2. Neo4j
        create_umls_nodes_and_relationships(n4j, entities, relationships, gid)
        logger.info(f"Successfully imported UMLS CSV to Neo4j: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error importing UMLS CSV to Neo4j: {e}")
        import traceback
        traceback.print_exc()
        return False