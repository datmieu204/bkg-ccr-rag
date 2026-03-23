import os
import torch
import numpy as np

from openai import AsyncOpenAI
from transformers import AutoTokenizer, AutoModel

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage


def _get_openrouter_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENROUTER_API_BASE_URL") or "https://openrouter.ai/api/v1"
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

async def gemini_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    """
    Use Gemini API to generate content, with caching to avoid redundant calls for the same inputs.
    """
    history_messages = history_messages or []
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    args_hash = None
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages, kwargs)
        cache_item = await hashing_kv.get_by_id(args_hash)
        if cache_item is not None:
            # Backward compatible with both dict and plain-string cached payloads.
            if isinstance(cache_item, dict):
                return cache_item.get("return", "")
            if isinstance(cache_item, str):
                return cache_item

    client = _get_openrouter_client()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )

    generated_text = response.choices[0].message.content or ""

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": generated_text, "model": model}}
        )
    
    return generated_text


async def gemini_2_5_flash_complete(
    prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    return await gemini_complete_if_cache(
        "google/gemini-2.0-flash-lite-001",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gemini_2_5_flash_lite_complete(
    prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    return await gemini_complete_if_cache(
        "google/gemini-2.0-flash-lite-001",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    

# Initialize embedding model globally
_bge_tokenizer = None
_bge_model = None

def get_bge_model():
    """Initialize and return bge-m3 model"""
    global _bge_tokenizer, _bge_model
    
    if _bge_tokenizer is None or _bge_model is None:
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        _bge_tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-m3", 
            token=hf_token
        )
        _bge_model = AutoModel.from_pretrained(
            "BAAI/bge-m3",
            token=hf_token
        )
        _bge_model.eval()
    
    return _bge_tokenizer, _bge_model


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=512)
async def bge_m3_embedding(texts: list[str]) -> np.ndarray:
    """Use HuggingFace bge-m3 for embeddings"""
    tokenizer, model = get_bge_model()
    
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1)
        
        embeddings.append(embedding[0].cpu().numpy())
    
    return np.array(embeddings)