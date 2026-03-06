"""
Pillar 3 — Analytical Squads (Project Antigravity).

The Pillar 3 Analytical Swarm is divided into N specialized Squads,
each dynamically instantiated from the ``squad_configs`` dictionary
in ``settings.py``. No domain-specific squads are hardcoded.
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agent_base import BaseAgent
from models import AgentVerdict, EntityDossier, ManifestEntry, SwarmConsensus
from langfuse_utils import get_current_session_id, set_current_session_id
from prompt_loader import load_prompt
from settings import get_settings
from token_compressor import AcronymMapper, compress_domain_data, compress_persona

logger = logging.getLogger(__name__)

class SpecializedAgent(BaseAgent):
    """Agent specializing in a specific Antigravity Squad."""

    def __init__(self, squad: str, **kwargs: Any) -> None:
        super().__init__(name=f"{squad}_agent", **kwargs)
        self.squad = squad

    def _build_prompt(
        self,
        dossier: EntityDossier,
        rag_examples: list[dict],
        filter_report: str = "",
    ) -> str:
        """Construct a squad-specific prompt using the unified config template."""
        cfg = get_settings()
        squad_cfg = cfg.squad_configs.get(self.squad, {})

        data_slice, legend = self._select_squad_data(dossier, squad_cfg)
        rag_section = self._format_rag(rag_examples)

        prompt_name = squad_cfg.get("prompt", f"squad_{self.squad}")
        template = load_prompt(prompt_name) or load_prompt("domain_agent")

        # Compress persona text if enabled
        context_text = dossier.context_data or ""
        if cfg.token_compression_enabled:
            context_text = compress_persona(context_text)
        else:
            context_text = context_text[:1500] if context_text else "N/A"

        # Inject legend into the data slice header
        if legend:
            data_slice = f"{legend}\n\n{data_slice}"

        return template.format(
            squad_name=self.squad.replace("_", " ").title(),
            entity_id=dossier.entity_id,
            data_slice=data_slice,
            profile_summary=dossier.get_compact_profile(),
            context=context_text,
            rag_section=rag_section,
            filter_section=f"### Mathematical Filter Assessment\n{filter_report}\n\n" if filter_report else ""
        )

    def _select_squad_data(self, dossier: EntityDossier, squad_cfg: dict[str, Any]) -> tuple[str, str]:
        """Filter dossier data relevant to this squad's domain and compress it.

        Returns:
            A tuple of (compressed_data_string, legend_string).
        """
        cfg = get_settings()
        relevant_roles = squad_cfg.get("roles", [])

        mapper = AcronymMapper(cfg.acronym_map) if cfg.token_compression_enabled else None

        parts: list[str] = []
        for role in relevant_roles:
            if role == cfg.profile_role:
                # Profile is handled separately by get_compact_profile()
                continue
            elif role == cfg.context_role:
                # Context is handled in _build_prompt via compress_persona
                continue
            else:
                data = dossier.domain_data.get(role, [])
                if data:
                    slice_data = data[-20:]  # Efficiency slice
                    if cfg.token_compression_enabled and mapper:
                        parts.append(f"**{role}**\n{compress_domain_data(slice_data, mapper, date_key=cfg.timestamp_col)}")
                    else:
                        import json
                        parts.append(json.dumps(slice_data, default=str, indent=1))

        legend = mapper.get_legend() if mapper else ""
        return "\n\n".join(parts) if parts else "_No relevant data._", legend

    @staticmethod
    def _format_rag(examples: list[dict]) -> str:
        """Format contextual memory examples into a markdown string."""
        if not examples: return ""
        lines = ["### Contextual Memory (RAG)\n"]
        for i, ex in enumerate(examples, 1):
            lines.append(f"**Case {i}** (label={ex.get('label')}): {ex.get('summary', 'N/A')}\n")
        return "\n".join(lines)


class SquadCoordinator:
    """Manages a single specialized squad, scaling agents dynamically."""

    def __init__(self, squad: str) -> None:
        self.squad = squad
        self._cfg = get_settings()

    def _decide_n_agents(self, dataset_size: int = 1) -> int:
        """Scale swarm based exclusively on dataset size."""
        squad_cfg = self._cfg.squad_configs.get(self.squad, {})
        
        base = squad_cfg.get("base_agents", self._cfg.squad_base_agents)
        factor = self._cfg.squad_dataset_scaling_factor
        
        # Scaling logic: base + floor(dataset_size * factor)
        n = base + math.floor(dataset_size * factor)
        
        # Ensure we stay within architectural limits
        return max(self._cfg.swarm_min_agents, min(self._cfg.swarm_max_agents, n))

    def run(
        self,
        dossier: EntityDossier,
        rag_examples: list[dict] | None = None,
        filter_report: str = "",
        dataset_size: int = 1,
    ) -> SwarmConsensus:
        """Execute squad analysis with a dynamic swarm."""
        n_agents = self._decide_n_agents(dataset_size)
        logger.info(f"Squad {self.squad} running with {n_agents} agents (dataset_size={dataset_size})")

        from llm_provider import get_provider
        provider = get_provider()
        model_name = provider.resolve_model("cheap")
        
        agents = [
            SpecializedAgent(
                squad=self.squad,
                model_name=model_name,
                temperature=self._cfg.model_temperature + (0.1 * i)
            ) for i in range(n_agents)
        ]

        verdicts: list[AgentVerdict] = []
        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            futures = {pool.submit(a.analyze, dossier, rag_examples or [], filter_report=filter_report): a for a in agents}
            for future in as_completed(futures):
                verdicts.append(future.result())

        return self._aggregate(verdicts)

    def _aggregate(self, verdicts: list[AgentVerdict]) -> SwarmConsensus:
        """Weighted aggregation of squad verdicts."""
        n = len(verdicts)
        if n == 0: return SwarmConsensus(agent_name=f"{self.squad}_coord", squad=self.squad, prediction=1, confidence=0.0)

        votes_1 = sum(v.confidence for v in verdicts if v.prediction == 1)
        votes_0 = sum(v.confidence for v in verdicts if v.prediction == 0)
        
        prediction = 1 if votes_1 >= votes_0 else 0
        agreement = (votes_1 if prediction == 1 else votes_0) / (votes_1 + votes_0 + 1e-9)
        
        avg_conf = sum(v.confidence for v in verdicts if v.prediction == prediction) / (sum(1 for v in verdicts if v.prediction == prediction) or 1)
        
        return SwarmConsensus(
            agent_name=f"{self.squad}_coord",
            squad=self.squad,
            prediction=prediction,
            confidence=min(1.0, agreement * avg_conf),
            reasoning=f"Squad {self.squad} consensus: {prediction} with {agreement:.1%} agreement.",
            n_agents=n,
            agreement_ratio=agreement,
            individual_verdicts=verdicts
        )

class AntigravitySwarmFactory:
    """Factory for Project Antigravity squads (Pillar 3)."""
    
    @staticmethod
    def create_squads() -> list[SquadCoordinator]:
        """Instantiate all configured analytical squads based on project settings."""
        cfg = get_settings()
        return [SquadCoordinator(s) for s in cfg.squad_configs.keys()]

    @staticmethod
    def run_all(
        squads: list[SquadCoordinator],
        dossier: EntityDossier,
        rag_examples: list[dict] | None = None,
        filter_report: str = "",
        dataset_size: int = 1
    ) -> list[SwarmConsensus]:
        """Execute all given analytical squads in parallel and collect their consensus verdicts."""
        cfg = get_settings()
        if not cfg.pillar_squads_enabled:
            logger.info("  [Pillar 3] Analytical Squads disabled in settings. Skipping.")
            return []
            
        results = []
        with ThreadPoolExecutor(max_workers=len(squads)) as pool:
            futures = [pool.submit(s.run, dossier, rag_examples, filter_report, dataset_size) for s in squads]
            for f in as_completed(futures):
                results.append(f.result())
        return results
