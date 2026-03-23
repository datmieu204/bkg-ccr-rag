# ./kgs_builder/chunking/semantic_chunker.py


import numpy as np

from typing import List, Optional
from nltk.tokenize import sent_tokenize
from kgs_builder.utils import get_embedding
from helpers.logger import get_logger

logger = get_logger("semantic_chunker", log_file="logs/chunking/semantic_chunker.log")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
    
    Returns:
        Similarity score between -1 and 1 (typically 0-1 for embeddings)
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)

def chunk_document(
    text: str,
    threshold: float = 0.85,
    max_chunk_sentences: int = 15,
    min_chunk_sentences: int = 2,
    max_chunk_tokens: Optional[int] = 512,
    log_stats: bool = True
) -> List[str]:
    """
    Chunk document based on semantic similarity between sentences
    
    Algorithm:
    1. Split text into sentences using NLTK
    2. Embed each sentence using bge-small-en-v1.5
    3. Group consecutive sentences with high cosine similarity (>= threshold)
    4. Respect max/min constraints to avoid too large/small chunks
    
    Args:
        text: Full document text to chunk
        threshold: Cosine similarity threshold (0.75-0.95 recommended)
                  Higher = stricter semantic grouping
        max_chunk_sentences: Maximum sentences per chunk (soft limit)
        min_chunk_sentences: Minimum sentences before allowing split (1-3 recommended)
        max_chunk_tokens: Maximum tokens per chunk (None = no limit)
        log_stats: Whether to log chunking statistics
    
    Returns:
        List of text chunks (strings)
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to chunk_document")
        return []
    
    logger.info(f"Chunking document ({len(text)} chars)...")
    sentences = sent_tokenize(text)
    
    if len(sentences) == 0:
        logger.warning("No sentences found after tokenization")
        return []
    
    if len(sentences) == 1:
        logger.info("Only 1 sentence - returning as single chunk")
        return [text.strip()]
    
    logger.info(f"Tokenized into {len(sentences)} sentences")
    logger.info(f"Generating embeddings for {len(sentences)} sentences...")
    embeddings = []
    for i, sent in enumerate(sentences):
        if not sent.strip():
            embeddings.append(np.zeros(384))  # bge-small-en-v1.5 dim
            continue
        
        try:
            emb = get_embedding(sent.strip())
            embeddings.append(np.array(emb))
        except Exception as e:
            logger.warning(f"Failed to embed sentence {i}: {e}")
            embeddings.append(np.zeros(384))
    
    logger.info(f"Embeddings generated")
    
    all_chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0
    last_embedding = None
    
    for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
        if not sent.strip():
            continue
        
        if len(current_chunk_sentences) == 0:
            current_chunk_sentences.append(sent)
            current_chunk_tokens = len(sent.split())
            last_embedding = emb
            continue
        
        similarity = cosine_similarity(last_embedding, emb)
        
        sent_tokens = len(sent.split())
        would_exceed_tokens = (max_chunk_tokens and 
                              current_chunk_tokens + sent_tokens > max_chunk_tokens)
        would_exceed_sentences = len(current_chunk_sentences) >= max_chunk_sentences
        
        should_split = False
        
        if would_exceed_tokens or would_exceed_sentences:
            should_split = True
            reason = "token limit" if would_exceed_tokens else "sentence limit"
            logger.debug(f"    Sentence {i}: Forcing split (exceeded {reason})")
        
        elif (similarity < threshold and 
              len(current_chunk_sentences) >= min_chunk_sentences):
            should_split = True
            logger.debug(f"Sentence {i}: Similarity {similarity:.3f} < {threshold} - splitting")
        
        else:
            logger.debug(f"Sentence {i}: Similarity {similarity:.3f} - continuing chunk")
        
        if should_split:
            chunk_text = " ".join(current_chunk_sentences).strip()
            if chunk_text:
                all_chunks.append(chunk_text)
            
            current_chunk_sentences = [sent]
            current_chunk_tokens = sent_tokens
            last_embedding = emb
        else:
            current_chunk_sentences.append(sent)
            current_chunk_tokens += sent_tokens
            last_embedding = emb
    
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences).strip()
        if chunk_text:
            all_chunks.append(chunk_text)
    
    if log_stats and all_chunks:
        chunk_lengths = [len(c.split()) for c in all_chunks]
        chunk_sent_counts = [len(sent_tokenize(c)) for c in all_chunks]
        
        logger.info(f"\nChunking Statistics:")
        logger.info(f"Total chunks: {len(all_chunks)}")
        logger.info(f"Avg chunk length: {np.mean(chunk_lengths):.1f} tokens")
        logger.info(f"Avg sentences/chunk: {np.mean(chunk_sent_counts):.1f}")
        logger.info(f"Min/Max tokens: {min(chunk_lengths)}/{max(chunk_lengths)}")
        logger.info(f"Min/Max sentences: {min(chunk_sent_counts)}/{max(chunk_sent_counts)}")
        logger.info(f"Threshold used: {threshold}")
    
    return all_chunks

def chunk_document_with_overlap(
    text: str,
    threshold: float = 0.85,
    overlap_sentences: int = 1,
    **kwargs
) -> List[str]:
    """
    Chunk document with overlapping sentences between chunks
    Useful for maintaining context across chunk boundaries
    
    Args:
        text: Full document text
        threshold: Similarity threshold
        overlap_sentences: Number of sentences to overlap between chunks
        **kwargs: Additional args passed to chunk_document()
    
    Returns:
        List of overlapping chunks
    """
    # Get base chunks
    chunks = chunk_document(text, threshold=threshold, **kwargs)
    
    if len(chunks) <= 1 or overlap_sentences == 0:
        return chunks
    
    # Add overlap
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped_chunks.append(chunk)
        else:
            prev_chunk = chunks[i-1]
            prev_sentences = sent_tokenize(prev_chunk)
            overlap = prev_sentences[-overlap_sentences:]
            
            curr_sentences = sent_tokenize(chunk)
            combined = overlap + curr_sentences
            overlapped_chunks.append(" ".join(combined))
    
    logger.info(f"Added {overlap_sentences}-sentence overlap between chunks")
    
    return overlapped_chunks

def get_optimal_threshold(
    sample_texts: List[str],
    thresholds: List[float] = [0.75, 0.80, 0.85, 0.90, 0.95],
    target_chunk_count: Optional[int] = None
) -> float:
    """
    Find optimal threshold by testing on sample documents
    
    Args:
        sample_texts: List of sample documents
        thresholds: List of thresholds to test
        target_chunk_count: Target average chunks per document (optional)
    
    Returns:
        Optimal threshold value
    """
    logger.info(f"\nFinding optimal threshold on {len(sample_texts)} samples...")
    
    results = {}
    for t in thresholds:
        chunk_counts = []
        for text in sample_texts:
            chunks = chunk_document(text, threshold=t, log_stats=False)
            chunk_counts.append(len(chunks))
        
        avg_chunks = np.mean(chunk_counts)
        std_chunks = np.std(chunk_counts)
        results[t] = {
            'avg': avg_chunks,
            'std': std_chunks,
            'min': min(chunk_counts),
            'max': max(chunk_counts)
        }
        
        logger.info(f"Threshold {t:.2f}: avg={avg_chunks:.1f}±{std_chunks:.1f} chunks")
    
    if target_chunk_count:
        best_t = min(results.keys(), 
                    key=lambda t: abs(results[t]['avg'] - target_chunk_count))
        logger.info(f"\nOptimal threshold: {best_t:.2f} (closest to target {target_chunk_count})")
    else:
        best_t = 0.85
        logger.info(f"\nUsing default threshold: {best_t:.2f}")
    
    return best_t