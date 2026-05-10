# ./kgs_builder/core/retrieve.py

import os
import asyncio
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple
from kgs_builder.utils import get_bge_m3_embedding
from kgs_builder.nano_graphrag._llm import gemini_complete_if_cache

from helpers.logger import get_logger
logger = get_logger("retrieve", log_file="logs/retrieve.log")

async def safe_gemini_complete_with_retry(retries=5, **kwargs) -> str:
    for attempt in range(retries):
        try:
            return await gemini_complete_if_cache(**kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            # Nếu là lỗi 429 hoặc Quota
            if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
                if attempt == retries - 1:
                    logger.error(f"[LLM] Bỏ cuộc sau {retries} lần thử. Lỗi: {e}")
                    return ""
                
                wait_time = 10.0 * (2 ** attempt) 
                
                match = re.search(r"retry in ([\d\.]+)s", error_msg)
                if match:
                    try:
                        wait_time = float(match.group(1)) + 1.0
                    except ValueError:
                        pass
                
                logger.warning(f"[LLM 429 Quota] Đang bị giới hạn. Ngủ {wait_time:.2f}s trước khi thử lại... (Lần {attempt+1}/{retries})")
                await asyncio.sleep(wait_time)
            
            else:
                logger.error(f"[LLM Lỗi Không Xác Định] {e}")
                if attempt == retries - 1:
                    return ""
                await asyncio.sleep(5.0)
    return ""

def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()

def vector_search_summaries(n4j, query: str, top_n: int = 20) -> List[Dict]:
    """
    Fast vector-based pre-filtering using BGE embeddings
    
    Args:
        n4j: Neo4j connection
        query: User query
        top_n: Number of candidates to retrieve
    
    Returns:
        List of {gid, content, similarity} sorted by similarity
    """

    query_embedding = get_bge_m3_embedding(query)
    check_query = """
        MATCH (s:Summary)
        RETURN 
            count(*) AS total,
            sum(CASE WHEN s.embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_emb
    """
    check_result = n4j.query(check_query)

    if check_result:
        total = check_result[0]["total"]
        with_emb = check_result[0]["with_emb"]
        logger.info(f"Total summaries: {total}, with embeddings: {with_emb}")

    if check_result and check_result[0]["with_emb"] > 0:
        sum_query = """
            MATCH (s:Summary)
            WHERE s.embedding IS NOT NULL
            RETURN s.content AS content, s.gid AS gid, s.embedding AS embedding
        """
        results = n4j.query(sum_query)
        use_precomputed = True
    else:
        logger.warning("No pre-computed embeddings! Run: python add_summary_embeddings.py")
        sum_query = """
            MATCH (s:Summary)
            RETURN s.content AS content, s.gid AS gid
        """
        results = n4j.query(sum_query)
        use_precomputed = False
    
    if not results:
        logger.error("No Summary nodes found in database")
        return []
    
    logger.info(f"[Vector Search] Processing {len(results)} summaries...")

    candidates = []
    for r in results:
        try:
            if use_precomputed and 'embedding' in r and r['embedding'] is not None:
                emb = r['embedding']
            else:
                content = r['content']
                if isinstance(content, list):
                    content = content[0] if content else ""
                content = content[:1000] if len(content) > 1000 else content
                emb = get_bge_m3_embedding(content)
            
            # Cosine similarity
            dot_product = np.dot(query_embedding, emb)
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            similarity = dot_product / (norm_product + 1e-8)
            
            content_str = r['content']
            if isinstance(content_str, list):
                content_str = content_str[0] if content_str else ""
            
            candidates.append({
                'gid': r['gid'],
                'content': content_str,
                'similarity': float(similarity)
            })
        except Exception as e:
            logger.debug(f"Skip candidate: {e}")
            continue
    
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    logger.info(f"[Vector Search] Found {len(candidates)} candidates")
    if candidates:
        logger.info(f"  Top similarity: {candidates[0]['similarity']:.3f}")
        if len(candidates) >= top_n:
            logger.info(f"  #{top_n} similarity: {candidates[top_n-1]['similarity']:.3f}")
    
    return candidates[:top_n]

async def llm_rerank(candidates: List[Dict], query: str, top_k: int = 5) -> List[str]:
    """
    Rerank candidates using LLM for better relevance
    
    Args:
        candidates: List of {gid, content, similarity} from vector search
        query: User query
        top_k: Number of top candidates to return after reranking
    Returns:
        List of gids of top_k candidates after reranking
    """

    if len(candidates) <= top_k:
        return [c['gid'] for c in candidates]
    
    logger.info(f"[LLM Rerank] Reranking top {len(candidates)} candidates with LLM...")

    candidate_texts = []
    for i, c in enumerate(candidates[:min(10, len(candidates))]):
        summary_preview = c['content'][:200].replace("\n", " ")
        candidate_texts.append(f"{i+1}. {summary_preview}")

    rerank_prompt = f"""Given the query and candidates summaries, rank them by relevance.

    Query: {query}

    Candidates:
    {chr(10).join(candidate_texts)}

    Return ONLY the numbers in order of relevance (most relevant first), separated by commas.
    Example: 3,1,5,2,4

    Your ranking:"""

    try:
        provider = os.getenv("LLM_PROVIDER") or "openrouter"
        model = os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL") or "google/gemini-2.0-flash-lite-001"
        
        # SỬ DỤNG HÀM CÓ RETRY Ở ĐÂY
        response_text = await safe_gemini_complete_with_retry(
            model=model,
            prompt=rerank_prompt,
            provider=provider,
        )

        if not response_text:
            logger.warning("[LLM Rerank] Empty response (maybe out of retries), returning default ranking.")
            return [c['gid'] for c in candidates[:top_k]]

        numbers = [int(s.strip()) for s in response_text.split(",") if s.strip().isdigit()]

        ranked_gids = []
        for idx in numbers:
            if 1 <= idx <= len(candidates):
                gid = candidates[idx-1]['gid']
                if gid not in ranked_gids:
                    ranked_gids.append(gid)

        for c in candidates:
            if c['gid'] not in ranked_gids:
                ranked_gids.append(c['gid'])

        logger.info(f"[LLM Rerank] Reranked successfully")
        return ranked_gids[:top_k]
    
    except Exception as e:
        logger.warning(f"[LLM Rerank] Reranking failed: {e}")
        return [c['gid'] for c in candidates[:top_k]]
    
def hybrid_retrieve(
    n4j,
    query: str,
    top_k: int = 3,
    vector_candidates: int = 20,
    use_rerank: bool = True,
) -> List[str]:
    """
    Hybrid retrieval combining vector search and LLM reranking
    
    Args:
        n4j: Neo4j connection
        query: User query
        top_k: Number of top candidates to return after reranking
        vector_candidates: Number of candidates to retrieve from vector search before reranking
    Returns:
        List of gids of top_k candidates after hybrid retrieval
    """
    candidates = vector_search_summaries(n4j, query, top_n=vector_candidates)
    
    if not candidates:
        logger.info("No candidates found from vector search")
        return []
    
    if len(candidates) <= top_k:
        return [c['gid'] for c in candidates]

    if not use_rerank:
        logger.info("Hybrid retrieval rerank disabled; using vector-only top_k=%d", top_k)
        return [c['gid'] for c in candidates[:top_k]]

    ranked_gids = _run_async(llm_rerank(candidates, query, top_k=top_k))

    logger.info(f"Hybrid retrieval completed. Top {top_k} GIDs: {ranked_gids}")
    return ranked_gids

def get_ranked_context(n4j, gid: str, query: str, max_items: int=50) -> List[str]:
    """
    Get context sentences from the Summary node, ranked by relevance to the query
    
    Args:
        n4j: Neo4j connection
        gid: Graph ID of the Summary node
        query: User query
        max_items: Maximum number of context sentences to return
    Returns:
        List of context sentences sorted by relevance
    """
    query_embedding = get_bge_m3_embedding(query)

    # 1. Fetch Sentences from Summary Node
    context_query = """
    MATCH (s:Summary {gid: $gid})-[:HAS_SENTENCE]->(sent)
    WHERE sent.embedding IS NOT NULL
    RETURN sent.text AS text, sent.embedding AS embedding
    """
    logger.info(f"Retrieving context sentences for GID: {gid[:8]}...")
    
    sentence_results = n4j.query(context_query, {'gid': gid})
    scored_sentences = []
    
    for r in sentence_results:
        sim = cosine_similarity(query_embedding, r['embedding'])
        scored_sentences.append((r['text'], sim))
    
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # 2. Fetch Triples from neighborhood
    MAX_TRIPLETS = 500
    ret_query = """
        MATCH (n)-[r]-(m)
        WHERE n.gid = $gid AND NOT n:Summary AND NOT m:Summary
          AND elementId(n) < elementId(m)
        RETURN n.id AS n_id, TYPE(r) AS rel_type, m.id AS m_id
        LIMIT $max_triples
    """
    triple_results = n4j.query(ret_query, {'gid': gid, 'max_triples': MAX_TRIPLETS})
    scored_triples = []
    
    if triple_results:
        query_terms = set(query.lower().split())
        for r in triple_results:
            triple_str = f"{r['n_id']} {r['rel_type']} {r['m_id']}"
            triple_lower = triple_str.lower()
            matches = sum(1 for term in query_terms if term in triple_lower)
            relevance = matches / (len(query_terms) + 1)
            scored_triples.append((triple_str, relevance))
        scored_triples.sort(key=lambda x: x[1], reverse=True)

    # Combine: Sentences first, then triples
    combined_context = [s[0] for s in scored_sentences[:max_items]]
    remaining_slots = max_items - len(combined_context)
    if remaining_slots > 0:
        combined_context.extend([t[0] for t in scored_triples[:remaining_slots]])

    logger.info(f"[Ranked Context] GID: {gid[:8]} - Retrieved {len(scored_sentences)} sentences, {len(scored_triples)} triples.")
    if scored_sentences:
        logger.info(f"  Top sentence similarity: {scored_sentences[0][1]:.3f}")
    
    return combined_context

def get_ranked_link_context(n4j, gid: str, query: str, max_items: int=50) -> List[str]:
    """
    Get link context (references) ranked by relevance to query
    
    Args:
        n4j: Neo4j connection
        gid: Graph ID
        query: User query
        max_items: Maximum context items to return
    
    Returns:
        List of reference strings, sorted by relevance
    """
    logger.info(f"[Ranked Link Context] GID: {gid[:8]}")

    MAX_REFS = 500

    retrieve_query = """
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary
        MATCH (n)-[r:REFERENCE]->(m)
        WHERE NOT m:Summary
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'REFERENCE'
        RETURN n.id AS n_id, m.id AS m_id, TYPE(s) AS rel_type, o.id AS o_id
        LIMIT $max_refs
    """

    results = n4j.query(retrieve_query, {'gid': gid, 'max_refs': MAX_REFS})
    link_type = "REFERENCE"

    if not results:
        logger.warning(f"No reference context found for GID: {gid[:8]}, falling back to IS_REFERENCE_OF")
        retrieve_query = """
            MATCH (n)
            WHERE n.gid = $gid AND NOT n:Summary
            MATCH (n)-[r:IS_REFERENCE_OF]-(m)
            WHERE NOT m:Summary
            MATCH (m)-[s]-(o)
            WHERE NOT o:Summary AND TYPE(s) <> 'IS_REFERENCE_OF'
            RETURN n.id AS n_id, m.id AS m_id, TYPE(s) AS rel_type, o.id AS o_id
            LIMIT $max_refs
        """
        results = n4j.query(retrieve_query, {'gid': gid, 'max_refs': MAX_REFS})
        link_type = "IS_REFERENCE_OF"
        if not results:
            logger.warning(f"No IS_REFERENCE_OF context found for GID: {gid[:8]}")
            return []

    logger.info(
        f"[Ranked Link Context] Retrieved {len(results)} {link_type} triples for GID: {gid[:8]}"
    )

    query_lower = query.lower()
    query_terms = set(query_lower.split())

    scored_refs = []
    label = "Reference" if link_type == "REFERENCE" else "Reference(IS_REFERENCE_OF)"
    for r in results:
        ref_str = f"{label}: {r['n_id']} has reference that {r['m_id']} {r['rel_type']} {r['o_id']}"
        ref_lower = ref_str.lower()

        matches = sum(1 for term in query_terms if term in ref_lower)
        relevance = matches / max(len(query_terms), 1)

        scored_refs.append((ref_str, relevance))

    scored_refs.sort(key=lambda x: x[1], reverse=True)

    seen = set()
    unique_refs = []

    for ref, score in scored_refs:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)
            if len(unique_refs) >= max_items:
                break

    logger.info(f"[Ranked Link Context] Scored {len(unique_refs)} unique references for GID: {gid[:8]}")
    return unique_refs

def aggregate_multi_subgraph_context(
    n4j, gids: List[str], 
    query: str, 
    max_items: int = 100
)-> Tuple[List[str], List[str]]:
    """
    Aggregate context from multiple subgraphs
    
    Args:
        n4j: Neo4j connection
        gids: List of Graph IDs
        query: User query
        max_items: Maximum total context items
    
    Returns:
        Tuple of (self_context, link_context)
    """

    logger.info(f"[Multi-Subgraph] Aggregating from {len(gids)} subgraphs...")
    
    all_self_context = []
    all_link_context = []
    
    items_per_subgraph = max_items // len(gids) if gids else max_items

    for gid in gids:
        self_ctx = get_ranked_context(n4j, gid, query, max_items=items_per_subgraph)
        link_ctx = get_ranked_link_context(n4j, gid, query, max_items=items_per_subgraph)

        all_self_context.extend(self_ctx)
        all_link_context.extend(link_ctx)
    
    all_self_context = list(dict.fromkeys(all_self_context))
    all_link_context = list(dict.fromkeys(all_link_context))

    logger.info(f"[Multi-Subgraph] Aggregated {len(all_self_context)} self-context items and {len(all_link_context)} link-context items")

    return all_self_context[:max_items], all_link_context[:max_items]

def get_improved_response(n4j, query: str, client=None, 
                          use_multi_subgraph: bool = False,
                          top_k_subgraphs: int = 1) -> Tuple[str, str]:
    """
    Generate response using improved retrieval pipeline
        
    Args:
        n4j: Neo4j connection
        query: User query
        client: Optional DedicatedKeyClient
        use_multi_subgraph: Whether to aggregate from multiple subgraphs (slower)
        top_k_subgraphs: Number of subgraphs to use (default 1 for speed)
    
    Returns:
        Tuple of (answer, primary_gid)
    """
    return _run_async(
        _get_improved_response_async(
            n4j,
            query,
            use_multi_subgraph=use_multi_subgraph,
            top_k_subgraphs=top_k_subgraphs,
        )
    )


async def _get_improved_response_async(
    n4j,
    query: str,
    use_multi_subgraph: bool = False,
    top_k_subgraphs: int = 1,
) -> Tuple[str, str]:
    logger.info(f"\n{'='*80}")
    logger.info("[Improved Response] Starting pipeline...")
    logger.info(f"{'='*80}")

    gids = hybrid_retrieve(n4j, query, top_k=top_k_subgraphs)

    if not gids:
        logger.warning("No relevant subgraphs found. Returning empty response.")
        return "", ""

    primary_gid = gids[0]

    if use_multi_subgraph and len(gids) > 1:
        self_context, link_context = aggregate_multi_subgraph_context(n4j, gids, query, max_items=100)
    else:
        self_context = get_ranked_context(n4j, primary_gid, query, max_items=50)
        link_context = get_ranked_link_context(n4j, primary_gid, query, max_items=50)

    sys_prompt_one = """
Please answer the question using insights supported by provided graph-based data relevant to medical information.
"""
    
    sys_prompt_two = """
Modify the response to the question using the provided references. Include precise citations relevant to your answer. You may use multiple citations simultaneously, denoting each with the reference index number. For example, cite the first and third documents as [1][3]. If the references do not pertain to the response, simply provide a concise answer to the original question.
"""

    MAX_CONTEXT_CHARS = 4000

    selfcont_str = "\n".join(self_context)
    linkcont_str = "\n".join(link_context)
    
    if len(selfcont_str) > MAX_CONTEXT_CHARS:
        selfcont_str = selfcont_str[:MAX_CONTEXT_CHARS] + "...(truncated)"
    
    if len(linkcont_str) > MAX_CONTEXT_CHARS:
        linkcont_str = linkcont_str[:MAX_CONTEXT_CHARS] + "...(truncated)"

    logger.info(f"[Improved Response] Context: {len(selfcont_str)} self, {len(linkcont_str)} link chars")
    
    user_one = f"the question is: {query}\n\nthe provided information is:\n{selfcont_str}"
    provider = os.getenv("LLM_PROVIDER") or "openrouter"
    model = os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL") or "google/gemini-2.0-flash-lite-001"
    
    res = await safe_gemini_complete_with_retry(
        model=model,
        prompt=user_one,
        system_prompt=sys_prompt_one,
        provider=provider,
    )
    
    user_two = f"the question is: {query}\n\nthe last response of it is:\n{res}\n\nthe references are:\n{linkcont_str}"
    
    final_answer = await safe_gemini_complete_with_retry(
        model=model,
        prompt=user_two,
        system_prompt=sys_prompt_two,
        provider=provider,
    )
    
    logger.info(f"[Improved Response] Generated answer ({len(final_answer)} chars)")
    
    return final_answer, primary_gid

def improved_seq_ret(n4j, sumq):
    """
    Drop-in replacement for seq_ret with improved retrieval
    
    Args:
        n4j: Neo4j connection
        sumq: Query summary (list or string)
        client: Optional DedicatedKeyClient
    
    Returns:
        Best matching GID (single)
    """
    query = sumq[0] if isinstance(sumq, list) else sumq
    gids = hybrid_retrieve(n4j, query, top_k=1)
    
    return gids[0] if gids else None

if __name__ == "__main__":
    import os
    from camel.storages import Neo4jGraph
    
    n4j = Neo4jGraph(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    test_query = "What are the treatment options for rare diseases with reduced ejection fraction?"