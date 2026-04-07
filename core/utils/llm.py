"""
core/utils/llm.py — Single source of truth for all LLM calls.

Provides:
    llm()        — async, returns full response string
    llm_stream() — async generator, yields tokens as they arrive
"""

import os
from typing import AsyncGenerator

from openai import AsyncOpenAI

_client: AsyncOpenAI | None = None
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def llm(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Non-streaming call. Returns the full response string."""
    client = _get_client()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


async def llm_stream(
    prompt: str,
    system: str = "You are a helpful assistant.",
) -> AsyncGenerator[str, None]:
    """Streaming call. Yields tokens as they arrive from the API."""
    client = _get_client()
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
