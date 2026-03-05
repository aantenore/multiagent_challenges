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

        # Disable json_mode to allow the new expert text-based format
        return self._provider.chat(
            system_message=system_msg,
            user_message=prompt,
            model=self.model_name,
            temperature=self.temperature,
            json_mode=False,
            callbacks=callbacks,
        )

    # ── Parsing ─────────────────────────────────────────────────────────

    def _parse_verdict(self, raw_output: str) -> AgentVerdict:
        """Parse raw LLM output into a validated AgentVerdict.
        
        Supports both traditional JSON and the new KV Expert format:
        REASONING: ...
        CONFIDENCE: ...
        PREDICTION: ...
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_output).strip().rstrip("`")

        # 1. Try JSON first
        try:
            data = json.loads(cleaned)
            return AgentVerdict(
                agent_name=self.name,
                prediction=int(data["prediction"]),
                confidence=float(data["confidence"]),
                reasoning=str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, KeyError):
            # 2. Fallback to Expert KV Format (REASONING: ... CONFIDENCE: ... PREDICTION: ...)
            try:
                # Regex to extract blocks. reasoning can be multi-line.
                # Format: 
                # REASONING: <text> 
                # CONFIDENCE: <float>
                # PREDICTION: <int>
                reasoning = re.search(r"REASONING:\s*(.*?)\s*(?=CONFIDENCE:|$)", cleaned, re.DOTALL | re.IGNORECASE)
                confidence = re.search(r"CONFIDENCE:\s*([\d.]+)", cleaned, re.IGNORECASE)
                prediction = re.search(r"PREDICTION:\s*([01])", cleaned, re.IGNORECASE)

                if confidence and prediction:
                    return AgentVerdict(
                        agent_name=self.name,
                        prediction=int(prediction.group(1)),
                        confidence=float(confidence.group(1)),
                        reasoning=reasoning.group(1).strip() if reasoning else "",
                    )
            except Exception as e:
                logger.error("Failed to parse Expert KV format: %s", e)

        # 3. Final Fallback: if it's garbage but we need to proceed
        logger.error("All parsing failed for output: %s", cleaned)
        raise ValueError(f"Could not parse agent output: {cleaned}")
