"""
Pydantic data models shared across every pillar of the pipeline.
Every inter-pillar exchange is validated through these schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Detection Metadata (Mathematical Filter → Analytical Squads) ───────

class DetectionMetadata(BaseModel):
    """Structured explanation from the Mathematical Filter (Pillar 1).

    Passed to Analytical Squads so they know WHY the entity was flagged
    (mathematical anomaly details). Squads act as verification/contextual filter.
    """

    is_anomalous: bool = Field(
        default=False, description="Whether the entity was flagged as anomalous"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Overall anomaly confidence"
    )
    detection_type: str = Field(
        default="none",
        description="'mathematical_trigger' or 'none'"
    )
    deviating_features: dict[str, float] = Field(
        default_factory=dict,
        description="Map of feature_name → z-score for features that exceeded threshold"
    )
    incident_start: str | None = Field(
        default=None, description="Start timestamp of the incident window"
    )
    incident_end: str | None = Field(
        default=None, description="End timestamp of the incident window"
    )
    report: str = Field(
        default="", description="Human-readable report for subsequent analysis"
    )
    max_z_score: float = Field(
        default=0.0, description="Maximum Z-Score observed across the sliding window"
    )


# ── Manifest Schema ─────────────────────────────────────────────────────

class ManifestEntry(BaseModel):
    """Single data-source descriptor inside manifest.json."""

    path: str = Field(..., description="Relative path to the data file")
    role: str = Field(
        ..., description="Logical role of this data source"
    )
    id_column: str = Field(..., description="Name of the entity-ID column")
    format: Literal["csv", "json", "md"] = Field(
        ..., description="File format"
    )
    description: str = Field(
        default="",
        description="Semantic description of this dataset's purpose and content",
    )
    columns: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of column names to their semantic descriptions",
    )


class Stage(BaseModel):
    """One train → eval stage in the pipeline.

    The manifest can contain N stages. During execution the pipeline
    accumulates all training data from stage 0..i before evaluating stage i.
    """

    name: str = Field(..., description="Human-readable stage name, e.g. 'stage_1'")
    training_sources: list[ManifestEntry] = Field(
        default_factory=list, description="Sources used to train on this stage"
    )
    evaluation_sources: list[ManifestEntry] = Field(
        default_factory=list, description="Sources used to predict/evaluate on this stage"
    )
    output_file: str = Field(
        default="",
        description="Output predictions filename for this stage, e.g. 'predictions_stage1.txt'",
    )


class Manifest(BaseModel):
    """Top-level manifest with N configurable stages.

    Supports two layouts:
    - **Staged** (recommended): ``{"stages": [...]}``.
    - **Legacy flat**: ``{"sources": [...]}``, treated as a single unnamed stage.
    """

    stages: list[Stage] = Field(
        default_factory=list, description="Ordered list of train/eval stages"
    )
    sources: list[ManifestEntry] = Field(
        default_factory=list, description="Legacy flat source list (single-stage)"
    )

    def get_stages(self) -> list[Stage]:
        """Return the stage list. Wraps legacy ``sources`` as a single stage."""
        if self.stages:
            return self.stages
        if self.sources:
            return [
                Stage(
                    name="default",
                    training_sources=self.sources,
                    evaluation_sources=self.sources,
                    output_file="predictions.txt",
                )
            ]
        return []


# ── Entity Dossier ──────────────────────────────────────────────────────

class EntityDossier(BaseModel):
    """Unified dossier for a single entity, assembled from all sources."""

    entity_id: str = Field(..., description="Unique entity identifier")
    domain_data: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="Dynamic bucket for all domain-specific data as defined in the manifest.",
    )
    profile_data: dict[str, Any] = Field(
        default_factory=dict, description="Static user profile"
    )
    context_data: str = Field(
        default="", description="Free-text context (personas / notes)"
    )
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Engineered numeric features (populated by Pillar 1)",
    )

    def get_compact_profile(self) -> str:
        """Return a compact, PII-free string representation of the profile."""
        if not self.profile_data:
            return "N/A"
        
        pii_keys = {"first_name", "last_name", "email", "phone", "phone_number", "address", "ssn"}
        clean = {k: v for k, v in self.profile_data.items() if k not in pii_keys}
        
        # Format as a concise comma-separated list
        parts = []
        for k, v in sorted(clean.items()):
            if isinstance(v, dict):
                # Flatten small nested dicts like 'residence'
                v_str = ", ".join(f"{sk}={sv}" for sk, sv in v.items() if sk not in pii_keys)
                parts.append(f"{k}:({v_str})")
            else:
                parts.append(f"{k}={v}")
        
        return ", ".join(parts) if parts else "Empty Profile"

    def get_filtered_features(self, top_n: int = 15, threshold: float = 0.1) -> dict[str, float]:
        """Return only the most significant features (by magnitude)."""
        if not self.features:
            return {}
            
        # Sort by absolute value descending
        sorted_feats = sorted(
            self.features.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Filter by threshold and take top N
        filtered = {
            k: v for k, v in sorted_feats[:top_n]
            if abs(v) >= threshold
        }
        return filtered


# ── Agent Verdict ───────────────────────────────────────────────────────

class AgentVerdict(BaseModel):
    """Structured output from any pillar (Pillar 1 / 3 / 4)."""

    agent_name: str = Field(..., description="Name of the emitting agent")
    prediction: int = Field(
        ..., ge=0, le=1, description="0 = standard, 1 = preventive support"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Self-assessed confidence"
    )
    reasoning: str = Field(
        default="", description="Natural-language justification"
    )


# ── Swarm Consensus ─────────────────────────────────────────────────────

class SwarmConsensus(AgentVerdict):
    """Aggregated verdict from a RoleCoordinator's swarm.

    Extends AgentVerdict with voting metadata so the Orchestrator
    can weigh the consensus strength of each role's swarm.
    """

    squad: str = Field(..., description="The functional squad this consensus covers")
    window_start: str | None = None
    window_end: str | None = None
    n_agents: int = Field(
        default=1, ge=1, description="Number of agents that voted"
    )
    agreement_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Fraction of agents that agree with the majority prediction",
    )
    complexity_score: float = Field(
        default=0.0, ge=0.0,
        description="Assessed complexity for this role (scales with anomaly magnitude)",
    )
    individual_verdicts: list[AgentVerdict] = Field(
        default_factory=list,
        description="Raw verdicts from each agent in the swarm",
    )


# ── Pipeline Result ─────────────────────────────────────────────────────

class PipelineResult(BaseModel):
    """Final output record for one entity after the full pipeline."""

    entity_id: str
    session_id: str | None = Field(
        default=None, description="Langfuse session ID for this result"
    )
    final_prediction: int = Field(
        ..., ge=0, le=1, description="Final binary label"
    )
    component_decided: str = Field(
        ..., description="Which system component emitted the final decision"
    )
    verdicts: list[AgentVerdict] = Field(
        default_factory=list,
        description="Ordered trail of per-agent verdicts"
    )
