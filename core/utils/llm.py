"""
core/utils/llm.py — Single source of truth for all LLM calls.

Provides:
    llm()        — async, returns full response string
    llm_stream() — async generator, yields tokens as they arrive

Backed by LiteLLM, which gives a single interface to 100+ providers.
Switch any agent's model with a one-line env-var change — no code edits.

Per-agent routing (env overrides):
    PLANNER_MODEL   default: gpt-4o
    CODER_MODEL     default: deepseek/deepseek-chat
    TESTER_MODEL    default: groq/llama-3.3-70b-versatile
    DEBUGGER_MODEL  default: anthropic/claude-sonnet-4-5
    CRITIC_MODEL    default: anthropic/claude-sonnet-4-5
    DEFAULT_MODEL   fallback when agent is unrecognised
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

import litellm

# Silence LiteLLM's verbose success logs; keep warnings/errors.
litellm.set_verbose = False

# ── Model routing ──────────────────────────────────────────────────────────

_ROUTING: dict[str, str] = {
    "planner":  "PLANNER_MODEL",
    "coder":    "CODER_MODEL",
    "tester":   "TESTER_MODEL",
    "debugger": "DEBUGGER_MODEL",
    "critic":   "CRITIC_MODEL",
}

_DEFAULTS: dict[str, str] = {
    "planner":  "gpt-4o",
    "coder":    "deepseek/deepseek-chat",
    "tester":   "groq/llama-3.3-70b-versatile",
    "debugger": "anthropic/claude-sonnet-4-5",
    "critic":   "anthropic/claude-sonnet-4-5",
}


def _resolve_model(agent: str) -> str:
    """Return the model string for *agent*, respecting env-var overrides."""
    agent = agent.lower()
    env_key = _ROUTING.get(agent)
    if env_key:
        return os.getenv(env_key, _DEFAULTS[agent])
    return os.getenv("DEFAULT_MODEL", "gpt-4o")


# ── Public API ─────────────────────────────────────────────────────────────

async def llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    agent: str = "",
) -> str:
    """Non-streaming call. Returns the full response string.

    Args:
        prompt: The user message.
        system: System prompt. Defaults to a generic helpful-assistant prompt.
        agent:  Optional agent name for per-agent model routing
                ("planner", "coder", "tester", "debugger", "critic").
                If omitted the DEFAULT_MODEL env var (or gpt-4o) is used.
    """
    model = _resolve_model(agent)
    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        seed=42,
    )
    return response.choices[0].message.content or ""


async def llm_stream(
    prompt: str,
    system: str = "You are a helpful assistant.",
    agent: str = "",
) -> AsyncGenerator[str, None]:
    """Streaming call. Yields tokens as they arrive from the API.

    Args:
        prompt: The user message.
        system: System prompt.
        agent:  Optional agent name for per-agent model routing.
    """
    model = _resolve_model(agent)
    stream = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        seed=42,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
