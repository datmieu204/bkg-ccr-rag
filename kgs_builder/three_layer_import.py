# ./kgs_builder/three_layer_import.py

import os
import sys
import gc
import argparse
import traceback

from pathlib import Path
from camel.storages import Neo4jGraph
from kgs_builder.data_processing.data_loader import load_high
from kgs_builder.graph.create_graph_with_description import creat_metagraph_with_description
from kgs_builder.data_processing.data_import_umls import import_umls_csv_to_neo4j
from kgs_builder.core.linking import smart_ref_link
from kgs_builder.utils import str_uuid, ref_link, add_sum
from helpers.logger import get_logger

logger = get_logger("three_layer_import", log_file="logs/three_layer_import.log")

class ThreeLayerImporter:
    def __init__(self, neo4j_url, neo4j_username, neo4j_password, neo4j_database=None):
        logger.info("Three-layer architecture knowledge graph importer")
        logger.info("\n[Connecting to Neo4j]...")
        
        self.n4j = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
        )
        if neo4j_database:
            logger.info(f"Target database: {neo4j_database} (Neo4jGraph uses default connection context)")
        logger.info("Neo4j connection successful")
        
        # Store GID for each layer
        self.layer_gids = {
            'bottom': [],
            'middle': [],
            'top': []
        }

    def clear_database(self):
        """
        Clear database (automatically clear, no confirmation needed)
        """
        logger.info("\n[Clearing database]...")
        result = self.n4j.query("MATCH (n) RETURN count(n) as count")
        count = result[0]['count'] if result else 0
        logger.info(f"Current number of nodes: {count}")
        
        if count > 0:
            logger.info("Automatically clearing database in batches...")
            batch_size = 1000
            deleted_total = 0
            
            while True:
                # Delete in batches to avoid memory overflow
                result = self.n4j.query(f"""
                    MATCH (n)
                    WITH n LIMIT {batch_size}
                    DETACH DELETE n
                    RETURN count(n) as deleted
                """)
                deleted = result[0]['deleted'] if result else 0
                deleted_total += deleted
                
                if deleted == 0:
                    break
                    
                logger.info(f"  Deleted {deleted_total}/{count} nodes...")
            
            logger.info(f" Database cleared ({deleted_total} nodes deleted)")
        else:
            logger.info("Database is already empty")

    def import_layer(self, layer_name: str, data_path: str, args):
        """
        Import one layer's data
        
        Args:
            layer_name: Layer name (bottom/middle/top)
            data_path: Data path
            args: Other parameters
        """
        if layer_name not in self.layer_gids:
            logger.error(f"Invalid layer name: {layer_name}")
            return

        logger.info("\n" + "="*80)
        logger.info(f"[{layer_name.upper()} layer] Starting import")
        logger.info(f"Data path: {data_path}")
        logger.info("="*80)

        data_path = Path(data_path)
        if not data_path.exists():
            logger.error(f"Data path does not exist: {data_path}")
            return
        
        if data_path.is_file():
            files = [data_path]
        else:
            txt_files = list(data_path.rglob("*.txt"))
            csv_files = list(data_path.rglob("*.csv"))
            files = txt_files + csv_files

        files = sorted(set(files))

        logger.info(f"\nFound {len(files)} files")

        txt_count = sum(1 for f in files if f.suffix.lower() == '.txt')
        csv_count = sum(1 for f in files if f.suffix.lower() == '.csv')
        logger.info(f"- TXT files: {txt_count} (free text)")
        logger.info(f"- CSV files: {csv_count} (structured data)")

        for idx, file_path in enumerate(files, 1):
            logger.info(f"[File {idx}/{len(files)}] {file_path.name}")

            # checkpoint check
            done_flag = file_path.with_suffix(file_path.suffix + ".done")
            if done_flag.exists():
                logger.info(f"Skipping (completed): {file_path.name}")
                continue

            try:
                gid = str_uuid()
                self.layer_gids[layer_name].append(gid)

                suffix = file_path.suffix.lower()

                if suffix == '.csv':
                    logger.info(f"[Type] Structured data (CSV)")

                    success = import_umls_csv_to_neo4j(str(file_path), gid, self.n4j)
                    if success:
                        summary_text = f"UMLS knowledge from {file_path.name}"
                        add_sum(self.n4j, summary_text, gid)
                    else:
                        logger.warning(f"Processing failed: {file_path.name}")
                        self.layer_gids[layer_name].remove(gid)
                        continue

                elif suffix == '.txt':
                    logger.info(f"[Type] Free text (TXT)")
                    content = load_high(str(file_path))
                    if not content or len(content.strip()) < 50:
                        logger.warning(f"Skip: Content too short")
                        self.layer_gids[layer_name].remove(gid)
                        continue

                    self.n4j = creat_metagraph_with_description(
                        args, content, gid, self.n4j
                    )

                else:
                    logger.warning(f"Skipping unsupported file type: {file_path.name}")
                    self.layer_gids[layer_name].remove(gid)
                    continue

                done_flag.touch()
                logger.info(f" Completed and recorded checkpoint: {done_flag.name}")

                gc.collect()

            except Exception as e:
                logger.error(f"Error: {file_path.name} - {e}")
                traceback.print_exc()
                try:
                    self.layer_gids[layer_name].remove(gid)
                except:
                    pass
                continue


        logger.info(f"\n{'='*80}")
        logger.info(f"[{layer_name.upper()} layer] Import completed")
        logger.info(f"Imported {len(self.layer_gids[layer_name])} subgraphs")
        logger.info(f"{'='*80}")

    def create_top_to_middle_links(self, use_smart_linking: bool = True, 
                                   top_k: int = 50, similarity_threshold: float = 0.6):
        """
        Create Top→Middle REFERENCE relationships using smart entity-based linking
        
        Note: Bottom→Middle links are created incrementally during import (IS_REFERENCE_OF)
        This function only handles Top→Middle linking
        
        Args:
            use_smart_linking: Use smart entity-based linking (recommended)
            top_k: Number of top Middle chunks to consider (default: 50)
            similarity_threshold: Minimum cosine similarity (default: 0.6)
        """
        logger.info("[Top→Middle Linking] Creating smart references")
        logger.info(f"Method: {'Smart Entity-based' if use_smart_linking else 'Direct Cosine Similarity'}")
        logger.info(f"Parameters: top_k={top_k}, similarity_threshold={similarity_threshold}")
        
        if not self.layer_gids['top']:
            logger.warning("No Top layer GIDs to link")
            return
        
        if not self.layer_gids['middle']:
            logger.warning("No Middle layer GIDs to link to")
            return
        
        total_stats = {
            'entities_extracted': 0,
            'middle_chunks_found': 0,
            'links_created': 0
        }
        
        ner_extractor = None
        if use_smart_linking:
            try:
                from kgs_builder.ner.raredisease_extractor import RareDiseaseExtractor
                logger.info("\n[Initialization] Loading NER model...")
                ner_extractor = RareDiseaseExtractor()
                logger.info("NER model ready")
            except Exception as e:
                logger.warning(f"Failed to load NER model: {e}")
                logger.warning("Falling back to direct cosine similarity")
                use_smart_linking = False
        
        if use_smart_linking:
            logger.info(f"\n[Processing] {len(self.layer_gids['top'])} Top layer documents...")
            
            for idx, top_gid in enumerate(self.layer_gids['top'], 1):
                logger.info(f"\n[{idx}/{len(self.layer_gids['top'])}] Processing Top GID: {top_gid[:8]}...")
                
                try:
                    stats = smart_ref_link(
                        self.n4j,
                        top_gid,
                        middle_layer_gids=self.layer_gids['middle'],
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        ner_extractor=ner_extractor
                    )
                    
                    total_stats['entities_extracted'] += stats['entities_extracted']
                    total_stats['middle_chunks_found'] += stats['middle_chunks_found']
                    total_stats['links_created'] += stats['links_created']
                    
                    if stats['links_created'] > 0:
                        logger.info(f"   Created {stats['links_created']} links (best sim: {stats['best_similarity']:.3f})")
                    else:
                        logger.warning(f"No links created")
                
                except Exception as e:
                    logger.error(f"Error: {e}")
                    logger.debug(traceback.format_exc())
        
        else:
            # Fallback: Direct cosine similarity
            logger.info("\n[Fallback] Using direct cosine similarity...")
            
            for top_gid in self.layer_gids['top']:
                for middle_gid in self.layer_gids['middle']:
                    try:
                        result = ref_link(self.n4j, top_gid, middle_gid)
                        if result:
                            count = len(result)
                            total_stats['links_created'] += count
                            if count > 0:
                                logger.info(f"   {top_gid[:8]}... → {middle_gid[:8]}...: {count} links")
                    except Exception as e:
                        logger.warning(f"Error: {e}")
        
        logger.info(f"[Summary] Top→Middle Linking Complete")
        logger.info(f"  Total Top documents: {len(self.layer_gids['top'])}")
        logger.info(f"  Entities extracted: {total_stats['entities_extracted']}")
        logger.info(f"  Middle chunks found: {total_stats['middle_chunks_found']}")
        logger.info(f"  Links created: {total_stats['links_created']}")

    def print_statistics(self):
        logger.info("[Statistics]")
        
        result = self.n4j.query("MATCH (n) WHERE NOT n:Summary RETURN count(n) as count")
        node_count = result[0]['count'] if result else 0
        
        result = self.n4j.query("MATCH (s:Summary) RETURN count(s) as count")
        summary_count = result[0]['count'] if result else 0
        
        result = self.n4j.query("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result[0]['count'] if result else 0
        
        result = self.n4j.query("MATCH ()-[r:REFERENCE]->() RETURN count(r) as count")
        ref_count = result[0]['count'] if result else 0
        
        result = self.n4j.query("""
            MATCH (n)
            WHERE NOT n:Summary
            RETURN labels(n)[0] as type, count(n) as count
            ORDER BY count DESC
            LIMIT 10
        """)

        logger.info(f"\nOverall statistics:")
        logger.info(f"  - Entity nodes: {node_count}")
        logger.info(f"  - Summary nodes: {summary_count}")
        logger.info(f"  - Total relationships: {rel_count}")
        logger.info(f"  - REFERENCE relationships: {ref_count}")

        logger.info(f"\nLayer statistics:")
        logger.info(f"  - Bottom layer: {len(self.layer_gids['bottom'])} subgraphs")
        logger.info(f"  - Middle layer: {len(self.layer_gids['middle'])} subgraphs")
        logger.info(f"  - Top layer: {len(self.layer_gids['top'])} subgraphs")

        logger.info(f"\nEntity types (top 10):")
        for item in result:
            logger.info(f"  - {item['type']}: {item['count']}")

def main():
    parser = argparse.ArgumentParser(description='Three-layer architecture knowledge graph import')
    
    # Data paths
    parser.add_argument('--bottom', type=str, help='Bottom layer data path (medical dictionary)')
    parser.add_argument('--middle', type=str, help='Middle layer data path (diagnostic and treatment guidelines)')
    parser.add_argument('--top', type=str, help='Top layer data path (cases)')
    
    # Function switches
    parser.add_argument('--clear', action='store_true', help='Clear database')
    parser.add_argument('--trinity', action='store_true', help='Create Trinity relationships')
    parser.add_argument('--grained_chunk', action='store_true', help='Use fine-grained chunking')
    parser.add_argument('--ingraphmerge', action='store_true', help='Merge similar nodes within graph')
    
    # NER filtering options
    parser.add_argument('--bottom_filter', action='store_true', 
                       help='Enable NER-based filtering (skip chunks with low Bottom layer overlap)')
    parser.add_argument('--min_overlap', type=int, default=5,
                       help='Minimum number of overlapping Bottom entities to process chunk (default: 5)')
    
    # Neo4j configuration
    parser.add_argument('--neo4j-url', type=str, 
                       default=os.getenv('NEO4J_URL') or os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    parser.add_argument('--neo4j-username', type=str, 
                       default=os.getenv('NEO4J_USERNAME', 'neo4j'))
    parser.add_argument('--neo4j-password', type=str, 
                       default=os.getenv('NEO4J_PASSWORD'))
    parser.add_argument('--neo4j-database', type=str,
                       default=os.getenv('NEO4J_DATABASE', 'neo4j'),
                       help='Neo4j database name (default: neo4j)')
    
    args = parser.parse_args()
    
    # Check Neo4j password
    if not args.neo4j_password:
        logger.error("Error: No Neo4j password provided")
        logger.info("Please set environment variable NEO4J_PASSWORD or use --neo4j-password parameter")
        sys.exit(1)
    
    # Initialize importer
    importer = ThreeLayerImporter(
        args.neo4j_url,
        args.neo4j_username,
        args.neo4j_password,
        args.neo4j_database
    )
    
    # Clear database
    if args.clear:
        importer.clear_database()
    
    # Import each layer
    if args.bottom:
        importer.import_layer('bottom', args.bottom, args)
    
    if args.middle:
        importer.import_layer('middle', args.middle, args)
    
    if args.top:
        importer.import_layer('top', args.top, args)
    
    # Create Top→Middle relationships (Bottom→Middle is created incrementally)
    if args.trinity:
        importer.create_top_to_middle_links(
            use_smart_linking=True,
            top_k=50,
            similarity_threshold=0.6
        )
    
    importer.print_statistics()

    logger.info("\nAll tasks completed!")


if __name__ == '__main__':
    main()