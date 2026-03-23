# ./kgs_builder/utils.py

import os
import uuid
import torch
import numpy as np

from neo4j import GraphDatabase
from camel.storages import Neo4jGraph
from transformers import AutoTokenizer, AutoModel

from kgs_builder.nano_graphrag._llm import _get_openrouter_client
from kgs_builder.core.summerize import process_chunks
from helpers.logger import get_logger
logger = get_logger("utils", log_file="logs/utils.log")

sys_prompt_one = """
Please answer the question using insights supported by provided graph-based data relevant to medical information.
"""

sys_prompt_two = """
Modify the response to the question using the provided references. Include precise citations relevant to your answer. You may use multiple citations simultaneously, denoting each with the reference index number. For example, cite the first and third documents as [1][3]. If the references do not pertain to the response, simply provide a concise answer to the original question.
"""

# Initialize HuggingFace bge-m3 model for embeddings
_embedding_tokenizer = None
_embedding_model = None

def get_bge_m3_embedding(text):
    """
    Get embeddings using HuggingFace bge-m3 model
    """
    global _embedding_tokenizer, _embedding_model
    
    if _embedding_tokenizer is None or _embedding_model is None:
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        _embedding_tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-m3", 
            token=hf_token
        )
        _embedding_model = AutoModel.from_pretrained(
            "BAAI/bge-m3",
            token=hf_token
        )
        _embedding_model.eval()
    
    # Tokenize and get embeddings
    inputs = _embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = _embedding_model(**inputs)
        # Use mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings[0].cpu().numpy().tolist()

def get_llm_embedding(text):
    pass

def get_embedding(text, mod="bge-m3"):
    if mod == "bge-m3":
        return get_bge_m3_embedding(text)
    elif mod == "llm":
        return get_llm_embedding(text)
    else:
        raise ValueError(f"Unknown embedding model: {mod}")

def fetch_texts(n4j):
    # Fetch the text for each node
    query = "MATCH (n) RETURN n.id AS id"
    return n4j.query(query)

def add_embeddings(n4j, node_id, embedding):
    # Upload embeddings to Neo4j
    query = "MATCH (n) WHERE n.id = $node_id SET n.embedding = $embedding"
    n4j.query(query, params = {"node_id":node_id, "embedding":embedding})

def add_nodes_emb(n4j):
    nodes = fetch_texts(n4j)

    for node in nodes:
        # Calculate embedding for each node's text
        if node['id']:
            embedding = get_embedding(node['id'])
            add_embeddings(n4j, node['id'], embedding)

def add_ge_emb(graph_element):
    for node in graph_element.nodes:
        emb = get_embedding(node.id)
        node.properties['embedding'] = emb
    return graph_element

def add_gid(graph_element, gid):
    for node in graph_element.nodes:
        node.properties['gid'] = gid
    for rel in graph_element.relationships:
        rel.properties['gid'] = gid
    return graph_element

def add_sum(n4j, content, gid):
    """
    Create summary node in Neo4j with optional dedicated client
    
    Args:
        n4j: Neo4j connection
        content: Text to summarize
        gid: Graph ID
    """
    
    sum = process_chunks(content)

    creat_sum_query = """
        CREATE (s:Summary {content: $sum, gid: $gid})
        RETURN s
        """
    s = n4j.query(creat_sum_query, {'sum': sum, 'gid': gid})
    
    link_sum_query = """
        MATCH (s:Summary {gid: $gid}), (n)
        WHERE n.gid = s.gid AND NOT n:Summary
        CREATE (s)-[:SUMMARIZES]->(n)
        RETURN s, n
        """
    n4j.query(link_sum_query, {'gid': gid})

    return s

async def call_llm(sys, user):
    """
    Calling Gemini 2.0 Flash with OpenRouter client.
    """
    client = _get_openrouter_client()

    response = await client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ],
        max_tokens=500,
        temperature=0.2,
        n=1,
        stop=None,
    )

    return response.choices[0].message.content or ""

def find_index_of_largest(nums):
    if not nums:
        print("Warning: No ratings found. Database may be empty.")
        return -1
    
    sorted_with_index = sorted((num, index) for index, num in enumerate(nums))
    
    largest_original_index = sorted_with_index[-1][1]
    
    return largest_original_index

async def get_response(n4j, gid, query):
    """Generate response using knowledge graph context
    
    Args:
        n4j: Neo4j connection
        gid: Graph ID
        query: User query
    """
    
    selfcont = ret_context(n4j, gid)
    linkcont = link_context(n4j, gid)
    
    logger.info(f"Self context: {len(selfcont)} items, ~{sum(len(s) for s in selfcont)} chars")
    logger.info(f"Link context: {len(linkcont)} items, ~{sum(len(s) for s in linkcont)} chars")
    
    MAX_CONTEXT_ITEMS = 50
    MAX_CONTEXT_CHARS = 3000
    
    if len(selfcont) > MAX_CONTEXT_ITEMS:
        logger.warning(f"Truncating self context from {len(selfcont)} to {MAX_CONTEXT_ITEMS} items")
        selfcont = selfcont[:MAX_CONTEXT_ITEMS]
    
    if len(linkcont) > MAX_CONTEXT_ITEMS:
        logger.warning(f"Truncating link context from {len(linkcont)} to {MAX_CONTEXT_ITEMS} items")
        linkcont = linkcont[:MAX_CONTEXT_ITEMS]
    
    selfcont_str = "\n".join(selfcont)
    linkcont_str = "\n".join(linkcont)
    
    if len(selfcont_str) > MAX_CONTEXT_CHARS:
        logger.warning(f"Truncating self context string from {len(selfcont_str)} to {MAX_CONTEXT_CHARS} chars")
        selfcont_str = selfcont_str[:MAX_CONTEXT_CHARS] + "...(truncated)"
    
    if len(linkcont_str) > MAX_CONTEXT_CHARS:
        logger.warning(f"Truncating link context string from {len(linkcont_str)} to {MAX_CONTEXT_CHARS} chars")
        linkcont_str = linkcont_str[:MAX_CONTEXT_CHARS] + "...(truncated)"
    
    logger.info(f"Final context lengths - Self: {len(selfcont_str)} chars, Link: {len(linkcont_str)} chars")
    
    user_one = f"the question is: {query}\n\nthe provided information is:\n{selfcont_str}"
    logger.info(f"Calling LLM (step 1) with prompt length: {len(user_one)} chars")
    res = await call_llm(sys_prompt_one, user_one)
    
    user_two = f"the question is: {query}\n\nthe last response of it is:\n{res}\n\nthe references are:\n{linkcont_str}"
    logger.info(f"Calling LLM (step 2) with prompt length: {len(user_two)} chars")
    res = await call_llm(sys_prompt_two, user_two)
    return res

def link_context(n4j, gid):
    cont = []
    retrieve_query = """
        // Match all 'n' nodes with a specific gid but not of the "Summary" type
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary

        // Find all 'm' nodes where 'm' is a reference of 'n' via a 'REFERENCES' relationship
        MATCH (n)-[r:REFERENCE]->(m)
        WHERE NOT m:Summary

        // Find all 'o' nodes connected to each 'm', and include the relationship type,
        // while excluding 'Summary' type nodes and 'REFERENCE' relationship
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'REFERENCE'

        // Collect and return details in a structured format
        RETURN n.id AS NodeId1, 
            m.id AS Mid, 
            TYPE(r) AS ReferenceType, 
            collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    res = n4j.query(retrieve_query, {'gid': gid})
    for r in res:
        # Expand each set of connections into separate entries with n and m
        for ind, connection in enumerate(r["Connections"]):
            cont.append("Reference " + str(ind) + ": " + r["NodeId1"] + "has the reference that" + r['Mid'] + connection['RelationType'] + connection['Oid'])
    return cont

def ret_context(n4j, gid):
    cont = []
    ret_query = """
    // Match all nodes with a specific gid but not of type "Summary" and collect them
    MATCH (n)
    WHERE n.gid = $gid AND NOT n:Summary
    WITH collect(n) AS nodes

    // Unwind the nodes to a pairs and match relationships between them
    UNWIND nodes AS n
    UNWIND nodes AS m
    MATCH (n)-[r]-(m)
    WHERE n.gid = m.gid AND id(n) < id(m) AND NOT n:Summary AND NOT m:Summary
    WITH n, m, TYPE(r) AS relType

    // Return node IDs and relationship types in structured format
    RETURN n.id AS NodeId1, relType, m.id AS NodeId2
    """
    res = n4j.query(ret_query, {'gid': gid})
    for r in res:
        cont.append(r['NodeId1'] + r['relType'] + r['NodeId2'])
    return cont

def merge_similar_nodes(n4j, gid):
    """
    Merge similar nodes based on embedding similarity using vector.similarity.cosine
    or manual cosine calculation if that's not available
    """
    if gid:
        # using vector.similarity.cosine (Neo4j 5.x+)
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary 
                AND n.gid = m.gid AND n.gid = $gid 
                AND n<>m 
                AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
                AND n.embedding IS NOT NULL AND m.embedding IS NOT NULL
            WITH n, m, threshold,
                // Manual cosine similarity calculation
                reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | 
                    dot + n.embedding[i] * m.embedding[i]) / 
                (sqrt(reduce(norm1 = 0.0, x IN n.embedding | norm1 + x * x)) * 
                 sqrt(reduce(norm2 = 0.0, y IN m.embedding | norm2 + y * y))) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*) as merged_count
        """
        result = n4j.query(merge_query, {'gid': gid})
    else:
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary 
                AND n<>m 
                AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
                AND n.embedding IS NOT NULL AND m.embedding IS NOT NULL
            WITH n, m, threshold,
                // Manual cosine similarity calculation
                reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | 
                    dot + n.embedding[i] * m.embedding[i]) / 
                (sqrt(reduce(norm1 = 0.0, x IN n.embedding | norm1 + x * x)) * 
                 sqrt(reduce(norm2 = 0.0, y IN m.embedding | norm2 + y * y))) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*) as merged_count
        """
        result = n4j.query(merge_query)
    return result

def ref_link(n4j, gid1, gid2):
    """
    Create reference links between similar nodes from different graphs
    using manual cosine similarity calculation
    """
    trinity_query = """
        // Match nodes from Graph A
        MATCH (a)
        WHERE a.gid = $gid1 AND NOT a:Summary AND a.embedding IS NOT NULL
        WITH collect(a) AS GraphA

        // Match nodes from Graph B
        MATCH (b)
        WHERE b.gid = $gid2 AND NOT b:Summary AND b.embedding IS NOT NULL
        WITH GraphA, collect(b) AS GraphB

        // Unwind the nodes to compare each against each
        UNWIND GraphA AS n
        UNWIND GraphB AS m

        // Set the threshold for cosine similarity
        WITH n, m, 0.6 AS threshold

        // Compute cosine similarity and apply the threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold,
            // Manual cosine similarity calculation
            reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | 
                dot + n.embedding[i] * m.embedding[i]) / 
            (sqrt(reduce(norm1 = 0.0, x IN n.embedding | norm1 + x * x)) * 
             sqrt(reduce(norm2 = 0.0, y IN m.embedding | norm2 + y * y))) AS similarity
        WHERE similarity > threshold

        // Create a relationship based on the condition
        MERGE (m)-[:REFERENCE]->(n)

        // Return results
        RETURN n, m, similarity
"""
    result = n4j.query(trinity_query, {'gid1': gid1, 'gid2': gid2})
    return result

def str_uuid():
    # Generate a random UUID
    generated_uuid = uuid.uuid4()

    # Convert UUID to a string
    return str(generated_uuid)