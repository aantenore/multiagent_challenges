"""
Base agent interface for all LLM-powered agents (Layer 1 & 2).
Handles retries, structured JSON parsing, and Langfuse observation.
Uses the modular LLM provider (OpenAI / Gemini / custom) via LangChain.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from llm_provider import get_provider
from models import AgentVerdict, EntityDossier
from prompt_loader import load_prompt
from settings import get_settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for LLM agents.

    Every concrete agent must implement :meth:`_build_prompt`.
    """

    def __init__(
        self,
        name: str,
        model_name: str | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        cfg = get_settings()
        self.name = name
        self._provider = get_provider()
        # If no explicit model_name, resolve from provider
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self._provider.resolve_model("cheap")
        self.temperature = temperature if temperature is not None else cfg.model_temperature
        self.max_retries = max_retries or cfg.max_agent_retries

    # ── Abstract ────────────────────────────────────────────────────────

    @abstractmethod
    def _build_prompt(
        self,
        dossier: EntityDossier,
        rag_examples: list[dict],
        **kwargs,
    ) -> str:
        """Build the full user-side prompt for the LLM.

        Must return a prompt string that instructs the model to respond
        with JSON matching ``AgentVerdict`` (minus ``agent_name``).
        """
        ...

    # ── Public ──────────────────────────────────────────────────────────

    def analyze(
        self,
        dossier: EntityDossier,
        rag_examples: list[dict] | None = None,
        **kwargs,
    ) -> AgentVerdict:
        """Call the LLM and return a validated AgentVerdict."""
        rag_examples = rag_examples or []
        prompt = self._build_prompt(dossier, rag_examples, **kwargs)

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._call_llm(prompt)
                verdict = self._parse_verdict(raw)
                return verdict
            except Exception as exc:
                logger.warning(
                    "[%s] attempt %d/%d failed: %s",
                    self.name, attempt, self.max_retries, exc,
                )
                if attempt == self.max_retries:
                    logger.error(
                        "[%s] all retries exhausted — raising final exception",
                        self.name,
                    )
                    raise
        # unreachable, but keeps mypy happy
        raise RuntimeError("Unreachable")  # pragma: no cover

    # ── LLM call ────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, callbacks: list[Any] | None = None) -> str:
        """Call the configured LLM provider with optional Langfuse callbacks."""
        system_text = load_prompt("system")
        domain_text = load_prompt("domain")
        system_msg = f"{system_text}\n\n{domain_text}".strip()

        return self._provider.chat(
            system_message=system_msg,
            user_message=prompt,
            model=self.model_name,
            temperature=self.temperature,
            json_mode=True,
            callbacks=callbacks,
        )

    # ── Parsing ─────────────────────────────────────────────────────────

    def _parse_verdict(self, raw_json: str) -> AgentVerdict:
        """Parse raw LLM output into a validated AgentVerdict."""
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_json).strip().rstrip("`")
        data = json.loads(cleaned)
        return AgentVerdict(
            agent_name=self.name,
            prediction=int(data["prediction"]),
            confidence=float(data["confidence"]),
            reasoning=str(data.get("reasoning", "")),
        )
