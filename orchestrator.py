"""
Layer 2 — Global Orchestrator (Economic Decider).
Uses a smart model to synthesise swarm verdicts, RAG context,
and economic trade-offs into a final classification.
"""

from __future__ import annotations

import json
import logging

from langfuse.decorators import observe

from agent_base import BaseAgent
from models import AgentVerdict, EntityDossier
from prompt_loader import load_prompt
from settings import get_settings

logger = logging.getLogger(__name__)


class GlobalOrchestrator(BaseAgent):
    """Smart-model agent that weighs swarm verdicts economically."""

    def __init__(self) -> None:
        cfg = get_settings()
        super().__init__(
            name="L2_GlobalOrchestrator",
            model_name=cfg.smart_model_name,
            temperature=cfg.model_temperature,
        )
        self._fp_cost = cfg.fp_cost
        self._fn_cost = cfg.fn_cost

    @observe(name="L2_orchestrator_decide")
    def decide(
        self,
        dossier: EntityDossier,
        swarm_verdicts: list[AgentVerdict],
        rag_examples: list[dict] | None = None,
    ) -> AgentVerdict:
        """Synthesise swarm votes into a final verdict."""
        return self.analyze(dossier, rag_examples, _swarm_verdicts=swarm_verdicts)

    def analyze(  # type: ignore[override]
        self,
        dossier: EntityDossier,
        rag_examples: list[dict] | None = None,
        *,
        _swarm_verdicts: list[AgentVerdict] | None = None,
    ) -> AgentVerdict:
        """Override to inject swarm verdicts into the prompt."""
        self._pending_swarm = _swarm_verdicts or []
        return super().analyze(dossier, rag_examples)

    def _build_prompt(
        self,
        dossier: EntityDossier,
        rag_examples: list[dict],
    ) -> str:
        swarm = self._pending_swarm

        verdict_lines = "\n".join(
            f"- **{v.agent_name}**: prediction={v.prediction}, "
            f"confidence={v.confidence:.2f}, reasoning={v.reasoning}"
            for v in swarm
        )

        rag_section = ""
        if rag_examples:
            cases = "\n".join(
                f"  Case: true={ex.get('true_label')}, predicted={ex.get('predicted_label')} — {ex.get('summary', '')}"
                for ex in rag_examples
            )
            rag_section = f"### Past Error Cases\n{cases}\n\n"

        # Load external template; use inline fallback if missing
        template = load_prompt("orchestrator")
        if template:
            return template.format(
                entity_id=dossier.entity_id,
                verdict_lines=verdict_lines,
                profile_json=json.dumps(dossier.profile_data, default=str, indent=2),
                context=dossier.context_data[:2000] if dossier.context_data else "N/A",
                features_json=json.dumps(dossier.features, indent=2),
                rag_section=rag_section,
                fp_cost=self._fp_cost,
                fn_cost=self._fn_cost,
                fn_ratio=self._fn_cost / self._fp_cost,
            )

        # Inline fallback
        return (
            f"## Global Decision for Entity '{dossier.entity_id}'\n\n"
            f"### Swarm Agent Verdicts\n{verdict_lines}\n\n"
            f"### Entity Profile\n"
            f"```json\n{json.dumps(dossier.profile_data, default=str, indent=2)}\n```\n\n"
            f"### Context\n{dossier.context_data[:2000] if dossier.context_data else 'N/A'}\n\n"
            f"### Engineered Features\n"
            f"```json\n{json.dumps(dossier.features, indent=2)}\n```\n\n"
            f"{rag_section}"
            f"### Economic Framework\n"
            f"- FP cost: {self._fp_cost}, FN cost: {self._fn_cost}\n\n"
            f"Respond with JSON: "
            f'{{"prediction": 0 or 1, "confidence": 0.0-1.0, "reasoning": "..."}}'
        )
