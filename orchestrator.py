"""
Layer 2 — Global Orchestrator (Economic Decider).
Uses a smart model to synthesise swarm consensus verdicts, RAG context,
and economic trade-offs into a final classification.
"""

from __future__ import annotations

import json
import logging

from agent_base import BaseAgent
from llm_provider import get_provider
from models import AgentVerdict, EntityDossier, SwarmConsensus
from prompt_loader import load_prompt
from settings import get_settings

logger = logging.getLogger(__name__)


class GlobalOrchestrator(BaseAgent):
    """Smart-model agent that weighs swarm consensus verdicts economically."""

    def __init__(self) -> None:
        cfg = get_settings()
        provider = get_provider()
        super().__init__(
            name="L2_GlobalOrchestrator",
            model_name=provider.resolve_model("smart"),
            temperature=cfg.model_temperature,
        )
        self._fp_cost = cfg.fp_cost
        self._fn_cost = cfg.fn_cost

    def decide(
        self,
        dossier: EntityDossier,
        swarm_verdicts: list[SwarmConsensus] | list[AgentVerdict],
        rag_examples: list[dict] | None = None,
    ) -> AgentVerdict:
        """Synthesise swarm consensus into a final verdict with conservative fallback."""
        verdict = self.analyze(dossier, rag_examples, _swarm_verdicts=swarm_verdicts)
        
        # Optimization: Consistent with "FN >> FP", if confidence is low, we assume risk (1)
        cfg = get_settings()
        if verdict.confidence < cfg.l2_confidence_threshold and verdict.prediction == 0:
            logger.warning(
                "L2 confidence (%.2f) below threshold (%.2f). Overriding pred=0 to pred=1 (Conservative Fallback)",
                verdict.confidence, cfg.l2_confidence_threshold
            )
            verdict.prediction = 1
            verdict.reasoning = f"[LOW_CONFIDENCE_OVERRIDE] {verdict.reasoning}"
            
        return verdict

    def analyze(  # type: ignore[override]
        self,
        dossier: EntityDossier,
        rag_examples: list[dict] | None = None,
        *,
        _swarm_verdicts: list[SwarmConsensus] | list[AgentVerdict] | None = None,
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

        # Format verdicts — use consensus format if available
        verdict_lines = self._format_verdicts(swarm)

        rag_section = ""
        if rag_examples:
            cases = "\n".join(
                f"  Case: predicted={ex.get('predicted_label')} — {ex.get('summary', '')}"
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
            f"### Swarm Consensus Verdicts\n{verdict_lines}\n\n"
            f"### Entity Profile\n"
            f"```json\n{json.dumps(dossier.profile_data, default=str, indent=2)}\n```\n\n"
            f"### Context\n{dossier.context_data[:2000] if dossier.context_data else 'N/A'}\n\n"
            f"### Engineered Features\n"
            f"```json\n{json.dumps(dossier.features, indent=2)}\n```\n\n"
            f"{rag_section}"
            f"### Economic Framework\n"
            f"- FP cost: {self._fp_cost}, FN cost: {self._fn_cost}\n\n"
            f"Respond with JSON: "
            f'{{{{"prediction": 0 or 1, "confidence": 0.0-1.0, "reasoning": "..."}}}}'
        )

    @staticmethod
    def _format_verdicts(verdicts: list) -> str:
        """Format verdicts with consensus info when available."""
        lines = []
        for v in verdicts:
            if isinstance(v, SwarmConsensus):
                lines.append(
                    f"- **{v.role.title()} Consensus** (N={v.n_agents}): "
                    f"prediction={v.prediction}, agreement={v.agreement_ratio:.0%}, "
                    f"confidence={v.confidence:.2f}, complexity={v.complexity_score:.2f}\n"
                    f"  Reasoning: {v.reasoning}"
                )
            else:
                lines.append(
                    f"- **{v.agent_name}**: prediction={v.prediction}, "
                    f"confidence={v.confidence:.2f}, reasoning={v.reasoning}"
                )
        return "\n".join(lines)
