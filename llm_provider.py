"""
Modular LLM Provider — switchable backend for OpenAI and Google Gemini.

Usage:
    provider = get_provider()           # uses settings.llm_provider
    response = provider.chat(system_msg, user_msg, model, temperature)

To add a new provider:
    1. Subclass ``BaseLLMProvider``
    2. Register it in ``_REGISTRY``
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from settings import get_settings

logger = logging.getLogger(__name__)


# ── Abstract Provider ───────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    """Interface that every LLM backend must implement."""

    @abstractmethod
    def chat(
        self,
        system_message: str,
        user_message: str,
        *,
        model: str,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> str:
        """Send a chat request and return the raw text response."""
        ...

    @abstractmethod
    def resolve_model(self, role: str) -> str:
        """Return the concrete model name for a given role ('cheap' or 'smart')."""
        ...


# ── OpenAI Provider ─────────────────────────────────────────────────────

class OpenAIProvider(BaseLLMProvider):
    """OpenAI ChatCompletion backend."""

    def __init__(self) -> None:
        from openai import OpenAI

        cfg = get_settings()
        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._models = {
            "cheap": cfg.cheap_model_name,
            "smart": cfg.smart_model_name,
        }

    def chat(
        self,
        system_message: str,
        user_message: str,
        *,
        model: str,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or "{}"

    def resolve_model(self, role: str) -> str:
        return self._models.get(role, self._models["cheap"])


# ── Google Gemini Provider ──────────────────────────────────────────────

class GeminiProvider(BaseLLMProvider):
    """Google Generative AI (Gemini) backend."""

    def __init__(self) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Install google-genai: pip install google-genai"
            ) from exc

        cfg = get_settings()
        self._client = genai.Client(api_key=cfg.google_api_key)
        self._models = {
            "cheap": cfg.gemini_cheap_model_name,
            "smart": cfg.gemini_smart_model_name,
        }

    def chat(
        self,
        system_message: str,
        user_message: str,
        *,
        model: str,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> str:
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "system_instruction": system_message,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        response = self._client.models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text or "{}"

    def resolve_model(self, role: str) -> str:
        return self._models.get(role, self._models["cheap"])


# ── Registry & Factory ─────────────────────────────────────────────────

_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}


def register_provider(name: str, cls: type[BaseLLMProvider]) -> None:
    """Register a custom LLM provider at runtime."""
    _REGISTRY[name] = cls


@lru_cache(maxsize=1)
def get_provider() -> BaseLLMProvider:
    """Return the configured LLM provider singleton."""
    cfg = get_settings()
    provider_name = cfg.llm_provider.lower()
    if provider_name not in _REGISTRY:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    logger.info("LLM provider: %s", provider_name)
    return _REGISTRY[provider_name]()
