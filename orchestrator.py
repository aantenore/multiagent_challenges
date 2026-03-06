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
        """
        Synthesize swarm consensus, historical RAG memory, and raw profile data 
        into a final economic decision.
        """
        logger.info("  [L2] Orchestrating final decision for entity %s...", dossier.entity_id)
        return self.analyze(dossier, rag_examples, _swarm_verdicts=swarm_verdicts)

    def analyze(  # type: ignore[override]
        self,
        dossier: EntityDossier,
        rag_examples: list[dict] | None = None,
        *,
        _swarm_verdicts: list[SwarmConsensus] | list[AgentVerdict] | None = None,
    ) -> AgentVerdict:
        """
        Specialized analysis for L2: inject swarm results into the prompt state
        before calling the LLM provider.
        """
        self._pending_swarm = _swarm_verdicts or []
        return super().analyze(dossier, rag_examples)

    def _build_prompt(
        self,
        dossier: EntityDossier,
        rag_examples: list[dict],
    ) -> str:
        """
        Construct the Global Orchestrator prompt.
        Includes an economic framework (FP vs FN costs) to bridge the gap 
        between AI prediction and business risk.
        """
        swarm = self._pending_swarm

        # Format verdicts — inclusion of role consensus metrics (agreement/complexity)
        verdict_lines = self._format_verdicts(swarm)

        rag_section = ""
        if rag_examples:
            cases = "\n".join(
                f"  Case: predicted={ex.get('predicted_label')} — {ex.get('summary', '')}"
                for ex in rag_examples
            )
            rag_section = f"### Historical Memory — Similar Scenarios\n{cases}\n\n"

        # Load external template for standardized reasoning
        template = load_prompt("orchestrator")
        if template:
            return template.format(
                entity_id=dossier.entity_id,
                verdict_lines=verdict_lines,
                profile_summary=dossier.get_compact_profile(),
                context=dossier.context_data[:1500] if dossier.context_data else "N/A",
                features_summary=json.dumps(dossier.get_filtered_features(top_n=12), indent=2),
                rag_section=rag_section,
                fp_cost=self._fp_cost,
                fn_cost=self._fn_cost,
                fn_ratio=self._fn_cost / self._fp_cost,
            )

        # Inline fallback — Chief Economic Orchestrator
        return (
            f"## Case Triage — Global Orchestration for '{dossier.entity_id}'\n\n"
            f"You are the senior decision maker. Your goal is to review findings from "
            f"multiple domain-specific swarms and issue a final 'Preventive Support' verdict.\n\n"
            f"### Domain Expert Verdicts\n{verdict_lines}\n\n"
            f"### Entity Profile & Identity\n"
            f"```json\n{json.dumps(dossier.profile_data, default=str, indent=2)}\n```\n\n"
            f"### Biographic Context\n{dossier.context_data[:2000] if dossier.context_data else 'N/A'}\n\n"
            f"### Statistical Indicators (L0 Extraction)\n"
            f"```json\n{json.dumps(dossier.features, indent=2)}\n```\n\n"
            f"{rag_section}"
            f"### Decision Framework — Economic Trade-offs\n"
            f"- False Positive Cost (FP): {self._fp_cost}\n"
            f"- False Negative Cost (FN): {self._fn_cost}\n"
            f"- Risk Bias: Missing a case (FN) is {self._fn_cost / self._fp_cost}x more impactful.\n\n"
            f"Respond with JSON format: "
            f'{{{{"prediction": 0|1, "confidence": float, "reasoning": "string"}}}}'
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
