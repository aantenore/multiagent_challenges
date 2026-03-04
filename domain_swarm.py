"""
Layer 1 — Domain Swarms (Specialized Agents).
Dynamically instantiates one agent per manifest role to analyse
role-specific data slices using a cheap LLM model.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langfuse.decorators import observe

from agent_base import BaseAgent
from models import AgentVerdict, EntityDossier, ManifestEntry
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

        # Load external template; use inline fallback if missing
        template = load_prompt("domain_agent")
        if template:
            return template.format(
                role=self.role,
                entity_id=dossier.entity_id,
                semantic_section=sem_section,
                role_title=self.role.title(),
                data_slice=data_slice,
                profile_json=json.dumps(dossier.profile_data, default=str, indent=2),
                context=dossier.context_data[:2000] if dossier.context_data else "N/A",
                rag_section=rag_section,
            )

        # Inline fallback
        return (
            f"## Task\n"
            f"Analyse the following **{self.role}** data for entity "
            f"'{dossier.entity_id}' and decide whether the trajectory "
            f"suggests *preventive support* (1) or *standard monitoring* (0).\n\n"
            f"{sem_section}"
            f"### {self.role.title()} Data\n"
            f"```json\n{data_slice}\n```\n\n"
            f"### Profile Summary\n"
            f"```json\n{json.dumps(dossier.profile_data, default=str, indent=2)}\n```\n\n"
            f"### Context\n{dossier.context_data[:2000] if dossier.context_data else 'N/A'}\n\n"
            f"{rag_section}"
            f"### Instructions\n"
            f"- Look for deteriorating trends, anomalies, or warning signals.\n"
            f"- False Negatives are MUCH WORSE than False Positives.\n\n"
            f"Respond with JSON: "
            f'{{"prediction": 0 or 1, "confidence": 0.0-1.0, "reasoning": "..."}}'
        )

    def _select_slice(self, dossier: EntityDossier) -> str:
        """Return the JSON-serialised data slice for this role."""
        match self.role:
            case "temporal":
                data = dossier.temporal_data[-20:]  # last 20 events
            case "spatial":
                data = dossier.spatial_data[-30:]  # last 30 pings
            case "profile":
                data = dossier.profile_data
            case "context":
                return dossier.context_data[:3000]
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
                f"**Case {i}** (true_label={ex.get('true_label')}, "
                f"predicted={ex.get('predicted_label')}): "
                f"{ex.get('summary', 'N/A')}\n"
            )
        lines.append("")
        return "\n".join(lines)


# ── Swarm Factory ───────────────────────────────────────────────────────

class SwarmFactory:
    """Creates one DomainAgent per role found in the manifest."""

    @staticmethod
    def create_swarm(
        roles: set[str],
        manifest_entries: list[ManifestEntry] | None = None,
    ) -> list[DomainAgent]:
        """Instantiate agents for each role with semantic metadata."""
        cfg = get_settings()

        # Build role → semantic metadata mapping from manifest entries
        role_meta: dict[str, dict[str, Any]] = {}
        if manifest_entries:
            for entry in manifest_entries:
                role_meta.setdefault(entry.role, {
                    "description": entry.description,
                    "columns": entry.columns,
                })

        agents = [
            DomainAgent(
                role=role,
                semantic_metadata=role_meta.get(role),
                model_name=cfg.cheap_model_name,
                temperature=cfg.model_temperature,
            )
            for role in sorted(roles)
        ]
        logger.info("Swarm created: %s", [a.name for a in agents])
        return agents

    @staticmethod
    @observe(name="L1_swarm_vote")
    def run_swarm(
        agents: list[DomainAgent],
        dossier: EntityDossier,
        rag_examples: list[dict] | None = None,
    ) -> list[AgentVerdict]:
        """Run all swarm agents on a single entity and collect verdicts."""
        verdicts: list[AgentVerdict] = []
        for agent in agents:
            verdict = agent.analyze(dossier, rag_examples)
            verdicts.append(verdict)
            logger.info(
                "  [%s] → pred=%d  conf=%.2f",
                agent.name, verdict.prediction, verdict.confidence,
            )
        return verdicts
