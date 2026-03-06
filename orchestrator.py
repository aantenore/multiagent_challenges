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
    """Pillar 4 — Chronic vs. Acute Judge (Project Antigravity).

    Directive: Evaluate "Recovery". 
    - If sliding window shows return to normal → 0 (Acute/Transient).
    - If sustained or accelerating → 1 (Chronic/Intervention).
    """

    def __init__(self) -> None:
        cfg = get_settings()
        provider = get_provider()
        super().__init__(
            name="GlobalOrchestrator",
            model_name=provider.resolve_model("smart"),
            temperature=cfg.model_temperature,
        )
        self._cfg = cfg

    def decide(
        self,
        dossier: EntityDossier,
        swarm_verdicts: list[SwarmConsensus],
        rag_examples: list[dict] | None = None,
    ) -> AgentVerdict:
        """Synthesize squad findings with a focus on recovery trends."""
        if not self._cfg.pillar_orchestrator_enabled:
            logger.info("  [Pillar 4] Orchestrator disabled. Emitting weighted consensus.")
            return self._fallback_consensus(swarm_verdicts)
            
        logger.info(
            "  [Pillar 4] Analyzing %d squad verdicts vs. %d memory precedents for final adjudication...",
            len(swarm_verdicts), len(rag_examples or [])
        )
        verdict = self.analyze(dossier, rag_examples, _swarm_verdicts=swarm_verdicts)
        
        logger.info(
            "  [Decision] Final Verdict: %d (Confidence: %.2f) | Reasoning: %s",
            verdict.prediction, verdict.confidence, verdict.reasoning
        )
        return verdict

    def _fallback_consensus(self, verdicts: list[SwarmConsensus]) -> AgentVerdict:
        """Simple mathematical fallback if Orchestrator LLM is disabled."""
        if not verdicts: return AgentVerdict(agent_name="Orchestrator_Fallback", prediction=0, confidence=0.5, reasoning="No squads.")
        avg_v = sum(v.prediction * v.confidence for v in verdicts) / (sum(v.confidence for v in verdicts) + 1e-9)
        pred = 1 if avg_v >= 0.5 else 0
        return AgentVerdict(agent_name="Orchestrator_Fallback", prediction=pred, confidence=0.7, reasoning="Weighted squad consensus.")

    def analyze(self, dossier: EntityDossier, rag_examples: list[dict] | None = None, *, _swarm_verdicts: list[SwarmConsensus] | None = None) -> AgentVerdict:
        """Execute the orchestrator's analysis logic over the provided dossier and underlying swarm verdicts."""
        self._pending_swarm = _swarm_verdicts or []
        return super().analyze(dossier, rag_examples)

    def _build_prompt(self, dossier: EntityDossier, rag_examples: list[dict]) -> str:
        """Build the orchestrator's specific prompt by synthesizing swarm verdicts and data trends."""
        verdict_lines = self._format_verdicts(self._pending_swarm)
        
        # Recovery context: highlight the most recent trend in features
        trend_summary = self._extract_trend_context(dossier)
        
        template = load_prompt("orchestrator")
        
        return template.format(
            entity_id=dossier.entity_id,
            verdict_lines=verdict_lines,
            trend_summary=trend_summary,
            profile_summary=dossier.get_compact_profile(),
            context=dossier.context_data[:1000] if dossier.context_data else "N/A",
            rag_section=self._format_rag(rag_examples),
            fp_cost=self._cfg.fp_cost,
            fn_cost=self._cfg.fn_cost,
        )

    def _extract_trend_context(self, dossier: EntityDossier) -> str:
        """Heuristic to describe if data is recovering or declining based on filter pillars."""
        # Use primary filter column if available for trend analysis
        target_col = self._cfg.filter_primary_col
        
        # Find which domain data list contains this column
        series_data = []
        for role_list in dossier.domain_data.values():
            if role_list and target_col in role_list[0]:
                series_data = role_list
                break
        
        if len(series_data) < 5: return "Stable (Insufficient history)"
        
        vals = [float(row.get(target_col, 0)) for row in series_data if target_col in row]
        if len(vals) < 5: return "Stable (No index found)"
        
        recent = sum(vals[-2:]) / 2
        older = sum(vals[-5:-2]) / 3
        
        if recent > older * 1.1: return "RECOVERING (Trending upwards)"
        if recent < older * 0.8: return "DECLINING (Sustained drop)"
        return "STABLE / TRANSIENT"

    @staticmethod
    def _format_rag(examples: list[dict]) -> str:
        """Format contextual memory examples into a markdown string specifically for the Orchestrator."""
        if not examples: return ""
        lines = ["### Contextual Memory (RAG)\n"]
        for ex in examples:
            scope = ex.get("scope", "global").upper()
            lines.append(f"- [{scope}] Case ID: {ex.get('entity_id')} | Label: {ex.get('label')} | Summary: {ex.get('summary')}")
        return "\n".join(lines)

    @staticmethod
    def _format_verdicts(verdicts: list[SwarmConsensus]) -> str:
        """Format the swarm consensus verdicts into a markdown bulleted list."""
        lines = []
        for v in verdicts:
            lines.append(
                f"- **{v.squad.replace('_', ' ').title()} Squad**: "
                f"prediction={v.prediction}, agreement={v.agreement_ratio:.0%}, "
                f"confidence={v.confidence:.2f}\n"
                f"  Reasoning: {v.reasoning}"
            )
        return "\n".join(lines)
