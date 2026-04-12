"""Shared provider utilities for text and vision calls.

The pipeline currently uses OpenAI-compatible models behind a small wrapper so the
rest of the codebase does not need to instantiate provider SDK clients directly.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

from openai import OpenAI

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower() or "openai"
DEFAULT_TEXT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
DEFAULT_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", DEFAULT_TEXT_MODEL)


class UnsupportedProviderError(RuntimeError):
    """Raised when the configured provider is not available."""


def get_text_provider_name() -> str:
    """Return the configured provider name."""
    return LLM_PROVIDER


def get_text_model_name() -> str:
    """Return the configured text model name."""
    return DEFAULT_TEXT_MODEL


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)


def _get_client() -> OpenAI:
    if LLM_PROVIDER != "openai":
        raise UnsupportedProviderError(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Only 'openai' is currently implemented."
        )
    return _get_openai_client()


def chat_text(
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    *,
    model: str = "",
    max_tokens: int = 4096,
    temperature: float | None = None,
) -> str:
    """Run a chat completion and return text content."""
    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": model or DEFAULT_TEXT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def chat_text_messages(
    messages: list[dict[str, Any]],
    *,
    model: str = "",
    max_tokens: int = 4096,
    temperature: float | None = None,
) -> str:
    """Run a chat completion from a prebuilt message list."""
    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": model or DEFAULT_TEXT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def chat_json(
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    *,
    model: str = "",
    max_tokens: int = 4096,
    temperature: float | None = None,
) -> dict[str, Any] | list[Any]:
    """Run a chat completion and parse a JSON object or array from the response."""
    raw = chat_text(
        system_prompt,
        user_content,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return parse_json_response(raw)


def vision_text(
    system_prompt: str,
    user_text: str,
    image_bytes: bytes,
    mime_type: str,
    *,
    model: str = "",
    max_tokens: int = 1024,
) -> str:
    """Run a multimodal text response against an inline image."""
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return chat_text(
        system_prompt,
        [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
            },
        ],
        model=model or DEFAULT_VISION_MODEL,
        max_tokens=max_tokens,
    )


def parse_json_response(raw: str) -> dict[str, Any] | list[Any]:
    """Parse JSON from model responses, tolerating markdown fences and wrappers."""
    cleaned = (raw or "").strip()
    if not cleaned:
        raise ValueError("Empty response from model")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse JSON from response: {cleaned[:200]}...")
