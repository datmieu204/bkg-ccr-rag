import os
import asyncio
import time
from collections import deque
import torch
import numpy as np

from openai import AsyncOpenAI
from transformers import AutoTokenizer, AutoModel

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage


def _get_openrouter_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENROUTER_API_BASE_URL") or "https://openrouter.ai/api/v1"
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY or OPENAI_API_KEY is required")
    return AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)


def _get_openrouter_model() -> str:
    return (
        os.getenv("OPENROUTER_MODEL")
        or os.getenv("LLM_MODEL")
        or "google/gemma-3-27b-it:free"
    )


class RateLimiter:
    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self.max_calls = max_calls
        self.period = period_seconds
        self._lock = asyncio.Lock()
        self._calls = deque()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                sleep_time = self.period - (now - self._calls[0])

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)


_openrouter_rate_limiter = RateLimiter(max_calls=15, period_seconds=60.0)


def _model_supports_system(model: str | None) -> bool:
    if os.getenv("OPENROUTER_DISABLE_SYSTEM", "false").lower() == "true":
        return False
    if not model:
        return True
    model_lower = model.lower()
    return "gemma" not in model_lower

async def gemini_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=None, provider=None, **kwargs
) -> str:
    """
    Use OpenRouter (OpenAI-compatible) API to generate content with caching.
    """
    history_messages = history_messages or []
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    messages = []
    allow_system = _model_supports_system(model)
    if system_prompt and allow_system:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    if system_prompt and not allow_system:
        prompt = f"System instruction:\n{system_prompt}\n\n{prompt}"
    messages.append({"role": "user", "content": prompt})

    if provider is None or provider != "openrouter":
        provider = "openrouter"

    if not model:
        model = _get_openrouter_model()

    cache_messages = messages
    cache_kwargs = kwargs

    args_hash = None
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, cache_messages, cache_kwargs)
        cache_item = await hashing_kv.get_by_id(args_hash)
        if cache_item is not None:
            # Backward compatible with both dict and plain-string cached payloads.
            if isinstance(cache_item, dict):
                return cache_item.get("return", "")
            if isinstance(cache_item, str):
                return cache_item

    if provider != "openrouter":
        raise ValueError(f"Unsupported provider: {provider}")

    client = _get_openrouter_client()
    try:
        await _openrouter_rate_limiter.acquire()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        generated_text = response.choices[0].message.content or ""
    finally:
        try:
            await client.close()
        except Exception:
            pass

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": generated_text, "model": model}}
        )
    
    return generated_text


async def gemini_2_5_flash_complete(
    prompt, system_prompt=None, history_messages=None, provider=None, **kwargs
) -> str:
    return await gemini_complete_if_cache(
        model=_get_openrouter_model(),
        prompt=prompt,
        provider=provider,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gemini_2_5_flash_lite_complete(
    prompt, system_prompt=None, history_messages=None, provider=None, **kwargs
) -> str:
    return await gemini_complete_if_cache(
        model=_get_openrouter_model(),
        prompt=prompt,
        provider=provider,
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