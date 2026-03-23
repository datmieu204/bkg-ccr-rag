# ./kgs_builder/core/retrieve.py

import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple
from kgs_builder.utils import get_bge_m3_embedding
from kgs_builder.nano_graphrag._llm import _get_openrouter_client

from helpers.logger import get_logger
logger = get_logger("retrieve", log_file="logs/retrieve.log")


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Running under an existing event loop (e.g. notebooks): execute in worker thread.
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
        client = _get_openrouter_client()
        response = await client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001",
            messages=[{"role": "user", "content": rerank_prompt}],
        )

        numbers = [int(s.strip()) for s in response.choices[0].message.content.split(",") if s.strip().isdigit()]

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
    
def hybrid_retrieve(n4j, query: str, top_k: int=3, vector_candidates: int=20) -> List[str]:
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

    context_query = """
    MATCH (s:Summary {gid: $gid})-[:HAS_SENTENCE]->(sent)
    WHERE sent.embedding IS NOT NULL
    RETURN sent.text AS text, sent.embedding AS embedding
    """
    logger.info(f"Retrieving context sentences for GID: {gid[:8]}...")
    
    MAX_TRIPLETS = 1000

    ret_query = """
        MATCH (n)-[r]-(m)
        WHERE n.gid = $gid AND NOT n:Summary AND NOT m:Summary
          AND id(n) < id(m)
        RETURN n.id AS n_id, TYPE(r) AS rel_type, m.id AS m_id
        LIMIT $max_triples
    """

    results = n4j.query(ret_query, {'gid': gid, 'max_triples': MAX_TRIPLETS})

    if not results:
        logger.warning(f"No sentences with embeddings found for GID: {gid[:8]}")
        return []
    
    logger.info(f"[Ranked Context] Retrieved {len(results)} sentences for GID: {gid[:8]}")

    query_terms = set(query.lower().split())

    scored_triples = []

    for r in results:
        triple_str = f"{r['n_id']} {r['rel_type']} {r['m_id']}"
        triple_lower = triple_str.lower()

        matches = sum(1 for term in query_terms if term in triple_lower)
        relevance = matches / (len(query_terms) + 1)

        scored_triples.append((triple_str, relevance))

    scored_triples.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"[Ranked Context] Scored {len(scored_triples)} triples for GID: {gid[:8]}")

    if scored_triples and scored_triples[0][1] > 0:
        logger.info(f"  Top triple relevance: {scored_triples[0][1]:.3f}")
        if len(scored_triples) >= max_items:
            logger.info(f"  #{max_items} triple relevance: {scored_triples[max_items-1][1]:.3f}")

    return [t[0] for t in scored_triples[:max_items]]

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

    if not results:
        logger.warning(f"No reference context found for GID: {gid[:8]}")
        return []

    logger.info(f"[Ranked Link Context] Retrieved {len(results)} reference triples for GID: {gid[:8]}")

    query_lower = query.lower()
    query_terms = set(query_lower.split())

    scored_refs = []
    for r in results:
        ref_str = f"Reference: {r['n_id']} has reference that {r['m_id']} {r['rel_type']} {r['o_id']}"
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
    client = _get_openrouter_client()

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
    full_prompt_one = f"{sys_prompt_one}\n\n{user_one}"
    first_response = await client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[{"role": "user", "content": full_prompt_one}],
    )
    res = first_response.choices[0].message.content or ""
    user_two = f"the question is: {query}\n\nthe last response of it is:\n{res}\n\nthe references are:\n{linkcont_str}"
    full_prompt_two = f"{sys_prompt_two}\n\n{user_two}"
    second_response = await client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[{"role": "user", "content": full_prompt_two}],
    )
    final_answer = second_response.choices[0].message.content or ""
    
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