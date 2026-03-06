"""
Layer 1 — Domain Swarms with Dynamic RoleCoordinators.

Each data role (temporal, spatial, profile, context) gets a RoleCoordinator
that:
  1. Assesses complexity for its domain slice.
  2. Scales the swarm linearly from MIN to MAX agents based on complexity.
  3. Runs N agents with varied temperatures for opinion diversity.
  4. Aggregates votes into a single SwarmConsensus (weighted voting).
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agent_base import BaseAgent
from models import AgentVerdict, EntityDossier, ManifestEntry, SwarmConsensus
from prompt_loader import load_prompt
from settings import get_settings

logger = logging.getLogger(__name__)


# ── Concrete Domain Agent ───────────────────────────────────────────────

class DomainAgent(BaseAgent):
    """Agent specialising in a single data role (temporal, spatial, etc.)."""

    def __init__(self, role: str, semantic_metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(name=f"L1_{role}_agent", **kwargs)
        self.role = role
        self._semantic = semantic_metadata or {}

    def _build_prompt(
        self,
        dossier: EntityDossier,
        rag_examples: list[dict],
        l0_report: str = "",
    ) -> str:
        # Select the data slice relevant to this agent's role
        data_slice = self._select_slice(dossier)
        rag_section = self._format_rag(rag_examples)

        # Build semantic context from manifest metadata
        sem_section = ""
        if self._semantic:
            desc = self._semantic.get("description", "")
            cols = self._semantic.get("columns", {})
            if desc or cols:
                sem_section = f"### Dataset Description\n{desc}\n\n"
            if cols:
                col_lines = "\n".join(f"- **{k}**: {v}" for k, v in cols.items())
                sem_section += f"### Column Definitions\n{col_lines}\n\n"

        # L0 anomaly report section
        l0_section = ""
        if l0_report:
            l0_section = (
                f"### L0 Anomaly Report (Mathematical Detection)\n"
                f"{l0_report}\n\n"
            )

        # Load external template; use inline fallback if missing
        template = load_prompt("domain_agent")
        if template:
            return template.format(
                role=self.role,
                entity_id=dossier.entity_id,
                semantic_section=sem_section,
                role_title=self.role.title(),
                data_slice=data_slice,
                profile_summary=dossier.get_compact_profile(),
                context=dossier.context_data[:1500] if dossier.context_data else "N/A",
                rag_section=rag_section,
                l0_section=l0_section,
            )

        # Inline fallback — Anti-False-Positive Filter
        return (
            f"## Task — Anti-False-Positive Filter\n"
            f"The Layer 0 One-Class engine (SVM + IsolationForest) has "
            f"detected a **mathematical anomaly** for entity '{dossier.entity_id}'.\n\n"
            f"Your job is NOT to discover anomalies in numbers (L0 already did that). "
            f"Your job is to act as a **contextual false-positive filter**:\n"
            f"- Read the context (persona, profile, lifestyle) and the {self.role} data.\n"
            f"- Decide if the mathematical deviation is **justified** by the person's "
            f"life context (e.g. seasonal change, known condition, lifestyle choice).\n"
            f"- If justified → output 0 (standard monitoring, false positive from L0).\n"
            f"- If NOT justified → output 1 (confirm preventive support).\n\n"
            f"{l0_section}"
            f"{sem_section}"
            f"### {self.role.title()} Data\n"
            f"```json\n{data_slice}\n```\n\n"
            f"### Profile Summary\n"
            f"```json\n{json.dumps(dossier.profile_data, default=str, indent=2)}\n```\n\n"
            f"### Context\n{dossier.context_data[:2000] if dossier.context_data else 'N/A'}\n\n"
            f"{rag_section}"
            f"### Instructions\n"
            f"- The L0 engine flagged this entity as outlier. Check if it is truly abnormal.\n"
            f"- False Negatives are WORSE than False Positives.\n"
            f"- But don't blindly confirm: if life context explains the deviation, output 0.\n\n"
            f"Respond with JSON: "
            f'{{"prediction": 0 or 1, "confidence": 0.0-1.0, "reasoning": "..."}}'
        )

    def _select_slice(self, dossier: EntityDossier) -> str:
        """Return the JSON-serialised data slice for this role."""
        match self.role:
            case "temporal":
                # Reduced from 20 to 10 events for token efficiency
                data = dossier.temporal_data[-10:]  
            case "spatial":
                # Reduced from 30 to 15 locations for token efficiency
                data = dossier.spatial_data[-15:]  
            case "profile":
                data = dossier.profile_data
            case "context":
                # Reduced from 3000 to 1500 chars for token efficiency
                return dossier.context_data[:1500]
            case _:
                data = {}
        return json.dumps(data, default=str, indent=1)

    @staticmethod
    def _format_rag(examples: list[dict]) -> str:
        if not examples:
            return ""
        lines = ["### Historical Similar Cases (few-shot)\n"]
        for i, ex in enumerate(examples, 1):
            lines.append(
                f"**Case {i}** (label={ex.get('label')}): "
                f"{ex.get('summary', 'N/A')}\n"
            )
        lines.append("")
        return "\n".join(lines)


# ── Role Coordinator ────────────────────────────────────────────────────

class RoleCoordinator:
    """Coordinator for a single data role.

    Decides how many agents to spawn based on complexity,
    runs them, and aggregates votes into a SwarmConsensus.
    """

    def __init__(
        self,
        role: str,
        semantic_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.role = role
        self._semantic = semantic_metadata or {}
        self._cfg = get_settings()

    def assess_complexity(self, dossier: EntityDossier, l0_complexity: float = 1.0) -> float:
        """Assess complexity for this role's data slice.

        Combines the L0 ML complexity score with a data-variance heuristic
        specific to this role. Returns a value in [0.0, 1.0].
        """
        # Data-richness heuristic: more data → potentially more complex
        role_complexity = self._data_variance_heuristic(dossier)
        # Blend L0 complexity (global) with role-specific signal
        blended = 0.6 * l0_complexity + 0.4 * role_complexity
        return min(1.0, max(0.0, blended))

    def _data_variance_heuristic(self, dossier: EntityDossier) -> float:
        """Quick data-variance heuristic per role. Returns 0-1."""
        match self.role:
            case "temporal":
                data = dossier.temporal_data
                if len(data) < 3:
                    return 0.2
                # Variance in numeric fields → higher complexity
                numeric_vals = []
                for row in data[-10:]:
                    for v in row.values():
                        try:
                            f_val = float(v)
                            if not math.isnan(f_val):
                                numeric_vals.append(f_val)
                        except (ValueError, TypeError):
                            pass
                if numeric_vals:
                    mean = sum(numeric_vals) / len(numeric_vals)
                    var = sum((x - mean) ** 2 for x in numeric_vals) / len(numeric_vals)
                    # Normalise: high variance → high complexity
                    return min(1.0, math.sqrt(var) / (mean + 1e-6))
                return 0.5
            case "spatial":
                data = dossier.spatial_data
                if len(data) < 2:
                    return 0.2
                # Number of distinct cities → more spatial diversity → more complex
                cities = {str(p.get("city", "")) for p in data if isinstance(p, dict)}
                return min(1.0, len(cities) / 5.0)
            case "context":
                if not dossier.context_data:
                    return 0.1
                # Longer context → more nuance → more complex
                return min(1.0, len(dossier.context_data) / 2000.0)
            case _:
                return 0.3

    def _decide_n_agents(self, complexity: float) -> int:
        """Map complexity linearly to agent count.

        Below threshold → min agents.
        At 1.0 → max agents.
        Linear interpolation in between.
        """
        threshold = self._cfg.swarm_complexity_threshold
        min_n = self._cfg.swarm_min_agents
        max_n = self._cfg.swarm_max_agents

        if complexity <= threshold:
            return min_n

        # Linear scale: threshold → min_n, 1.0 → max_n
        ratio = (complexity - threshold) / (1.0 - threshold + 1e-9)
        n = min_n + ratio * (max_n - min_n)
        return max(min_n, min(max_n, round(n)))

    def run(
        self,
        dossier: EntityDossier,
        rag_examples: list[dict] | None = None,
        l0_complexity: float = 1.0,
        l0_report: str = "",
    ) -> SwarmConsensus:
        """Spawn N agents, collect verdicts, aggregate into consensus."""
        rag_examples = rag_examples or []
        complexity = self.assess_complexity(dossier, l0_complexity)
        n_agents = self._decide_n_agents(complexity)

        logger.info(
            "  RoleCoordinator[%s] complexity=%.2f → N=%d agents",
            self.role, complexity, n_agents,
        )

        # Spawn agents with varied temperatures
        base_temp = self._cfg.model_temperature
        spread = self._cfg.swarm_temp_spread
        agents = self._spawn_agents(n_agents, base_temp, spread)

        # Run all agents IN PARALLEL
        verdicts: list[AgentVerdict] = []

        def _run_agent(agent: DomainAgent) -> AgentVerdict:
            return agent.analyze(dossier, rag_examples, l0_report=l0_report)

        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            futures = {pool.submit(_run_agent, agent): agent for agent in agents}
            for future in as_completed(futures):
                agent = futures[future]
                verdict = future.result()
                verdicts.append(verdict)
                logger.info(
                    "    [%s] → pred=%d  conf=%.2f",
                    agent.name, verdict.prediction, verdict.confidence,
                )
                logger.debug(
                    "    [%s] Reasoning: %s",
                    agent.name, verdict.reasoning,
                )

        # Aggregate
        return self._aggregate(verdicts, complexity)

    def _spawn_agents(
        self,
        n: int,
        base_temp: float,
        spread: float,
    ) -> list[DomainAgent]:
        """Create N DomainAgents with staggered temperatures."""
        cfg = self._cfg
        from llm_provider import get_provider
        provider = get_provider()
        model_name = provider.resolve_model("cheap")

        agents: list[DomainAgent] = []
        for i in range(n):
            if n == 1:
                temp = base_temp
            else:
                # Spread temperatures evenly: base - spread/2 .. base + spread/2
                t = base_temp + spread * (i / (n - 1) - 0.5)
                temp = max(0.0, min(1.0, t))
            agent = DomainAgent(
                role=self.role,
                semantic_metadata=self._semantic,
                model_name=model_name,
                temperature=temp,
            )
            if n > 1:
                agent.name = f"L1_{self.role}_agent_{i + 1}/{n}"
            agents.append(agent)
        return agents

    def _aggregate(
        self,
        verdicts: list[AgentVerdict],
        complexity: float,
    ) -> SwarmConsensus:
        """Aggregate N verdicts into a weighted SwarmConsensus."""
        if not verdicts:
            return SwarmConsensus(
                agent_name=f"L1_{self.role}_coordinator",
                role=self.role,
                prediction=1,
                confidence=0.3,
                reasoning="No verdicts received — conservative fallback.",
                n_agents=0,
                agreement_ratio=0.0,
                complexity_score=complexity,
            )

        n = len(verdicts)

        # Weighted voting: each verdict's vote is weighted by its confidence
        weighted_yes = sum(v.confidence for v in verdicts if v.prediction == 1)
        weighted_no = sum(v.confidence for v in verdicts if v.prediction == 0)
        total_weight = weighted_yes + weighted_no

        if total_weight == 0:
            majority_pred = 1  # conservative
            agreement = 0.5
        else:
            majority_pred = 1 if weighted_yes >= weighted_no else 0
            winning_weight = weighted_yes if majority_pred == 1 else weighted_no
            agreement = winning_weight / total_weight

        # Average confidence of agents that agree with majority
        agreeing = [v for v in verdicts if v.prediction == majority_pred]
        avg_conf = sum(v.confidence for v in agreeing) / len(agreeing) if agreeing else 0.5

        # Final confidence = agreement * average_confidence
        final_conf = min(1.0, agreement * avg_conf)

        # Collect reasoning
        reasoning_parts = [
            f"{v.agent_name}: pred={v.prediction} conf={v.confidence:.2f}"
            for v in verdicts
        ]
        summary = (
            f"Consensus {majority_pred} ({sum(1 for v in verdicts if v.prediction == majority_pred)}/{n} agree, "
            f"agreement={agreement:.0%}, avg_conf={avg_conf:.2f}). "
            f"Votes: [{'; '.join(reasoning_parts)}]"
        )

        return SwarmConsensus(
            agent_name=f"L1_{self.role}_coordinator",
            role=self.role,
            prediction=majority_pred,
            confidence=final_conf,
            reasoning=summary,
            n_agents=n,
            agreement_ratio=agreement,
            complexity_score=complexity,
            individual_verdicts=verdicts,
        )


# ── Swarm Factory ───────────────────────────────────────────────────────

class SwarmFactory:
    """Creates RoleCoordinators for each role found in the manifest."""

    @staticmethod
    def create_coordinators(
        roles: set[str],
        manifest_entries: list[ManifestEntry] | None = None,
    ) -> list[RoleCoordinator]:
        """Instantiate one RoleCoordinator per role with semantic metadata."""
        # Build role → semantic metadata mapping from manifest entries
        role_meta: dict[str, dict[str, Any]] = {}
        if manifest_entries:
            for entry in manifest_entries:
                role_meta.setdefault(entry.role, {
                    "description": entry.description,
                    "columns": entry.columns,
                })

        coordinators = [
            RoleCoordinator(
                role=role,
                semantic_metadata=role_meta.get(role),
            )
            for role in sorted(roles)
        ]
        logger.info("Coordinators created: %s", [c.role for c in coordinators])
        return coordinators

    @staticmethod
    def run_coordinators(
        coordinators: list[RoleCoordinator],
        dossier: EntityDossier,
        rag_examples: list[dict] | None = None,
        l0_complexity: float = 1.0,
        detection_metadata: Any = None,
    ) -> list[SwarmConsensus]:
        """Run all coordinators IN PARALLEL and collect consensus verdicts."""
        l0_report = ""
        if detection_metadata is not None:
            l0_report = getattr(detection_metadata, 'report', '')

        def _run_coord(coord: RoleCoordinator) -> SwarmConsensus:
            return coord.run(dossier, rag_examples, l0_complexity, l0_report=l0_report)

        results: list[SwarmConsensus] = []
        with ThreadPoolExecutor(max_workers=len(coordinators)) as pool:
            futures = {pool.submit(_run_coord, coord): coord for coord in coordinators}
            for future in as_completed(futures):
                coord = futures[future]
                consensus = future.result()
                results.append(consensus)
                logger.info(
                    "  [%s] → pred=%d  agreement=%.0f%%  N=%d",
                    coord.role, consensus.prediction,
                    consensus.agreement_ratio * 100, consensus.n_agents,
                )
        return results

    # ── Legacy compatibility ────────────────────────────────────────────
