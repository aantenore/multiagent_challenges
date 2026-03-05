"""
Modular LLM Provider — switchable backend for OpenAI and Google Gemini.
Uses LangChain wrappers to enable native Langfuse CallbackHandler integration.

Usage:
    provider = get_provider()           # uses settings.llm_provider
    response = provider.chat(system_msg, user_msg, model, temperature, callbacks=[handler])

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


def _extract_text(content: Any) -> str:
    """Normalize LangChain AIMessage.content to a plain string.

    LangChain's Gemini wrapper can return content as:
    - str: plain text (normal case)
    - list[str]: multiple text chunks
    - list[dict]: content blocks like [{'type':'text','text':'...'}]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


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
        callbacks: list[Any] | None = None,
    ) -> str:
        """Send a chat request and return the raw text response."""
        ...

    @abstractmethod
    def resolve_model(self, role: str) -> str:
        """Return the concrete model name for a given role ('cheap' or 'smart')."""
        ...


# ── OpenAI Provider (via LangChain) ─────────────────────────────────────

class OpenAIProvider(BaseLLMProvider):
    """OpenAI ChatCompletion backend via LangChain ChatOpenAI."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._api_key = cfg.openai_api_key
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
        callbacks: list[Any] | None = None,
    ) -> str:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        kwargs: dict[str, Any] = {
            "api_key": self._api_key,
            "model": model,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        llm = ChatOpenAI(**kwargs)
        
        from langfuse_utils import run_llm_call, get_current_session_id
        session_id = get_current_session_id()
        response_content = run_llm_call(session_id, llm, system_message, user_message)

        return _extract_text(response_content) or "{}"

    def resolve_model(self, role: str) -> str:
        return self._models.get(role, self._models["cheap"])


# ── Google Gemini Provider (via LangChain) ──────────────────────────────

class GeminiProvider(BaseLLMProvider):
    """Google Generative AI (Gemini) backend via LangChain ChatGoogleGenerativeAI."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._api_key = cfg.google_api_key
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
        callbacks: list[Any] | None = None,
    ) -> str:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGoogleGenerativeAI(
            google_api_key=self._api_key,
            model=model,
            temperature=temperature,
        )
        
        from langfuse_utils import run_llm_call, get_current_session_id
        session_id = get_current_session_id()
        response_content = run_llm_call(session_id, llm, system_message, user_message)

        return _extract_text(response_content) or "{}"

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
    logger.info("LLM provider: %s (via LangChain)", provider_name)
    return _REGISTRY[provider_name]()
