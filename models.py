"""
Pydantic data models shared across every layer of the pipeline.
Every inter-agent exchange is validated through these schemas.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Detection Metadata (Layer 0 → Layer 1 explainability) ──────────────

class DetectionMetadata(BaseModel):
    """Structured explanation from the L0 One-Class Anomaly Engine.

    Passed to L1 agents so they know WHY L0 escalated this entity
    (mathematical anomaly details). L1 acts as Anti-False-Positive filter.
    """

    is_anomalous: bool = Field(
        default=False, description="Whether the entity was flagged as anomalous"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Overall anomaly confidence"
    )
    forest_flagged: bool = Field(
        default=False, description="IsolationForest flagged this entity as outlier"
    )
    detection_type: str = Field(
        default="none",
        description="'behavioural' or 'none'"
    )
    deviating_features: dict[str, float] = Field(
        default_factory=dict,
        description="Map of feature_name → z-score for features that exceeded threshold"
    )
    forest_score: float = Field(
        default=0.0, description="IsolationForest anomaly decision score"
    )
    report: str = Field(
        default="", description="Human-readable report for L1 agents"
    )


# ── Manifest Schema ─────────────────────────────────────────────────────

class ManifestEntry(BaseModel):
    """Single data-source descriptor inside manifest.json."""

    path: str = Field(..., description="Relative path to the data file")
    role: Literal["temporal", "spatial", "profile", "context"] = Field(
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

    name: str = Field(..., description="Human-readable stage name, e.g. 'level_1'")
    training_sources: list[ManifestEntry] = Field(
        default_factory=list, description="Sources used to train on this stage"
    )
    evaluation_sources: list[ManifestEntry] = Field(
        default_factory=list, description="Sources used to predict/evaluate on this stage"
    )
    output_file: str = Field(
        default="",
        description="Output predictions filename for this stage, e.g. 'predictions_lev1.txt'",
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
    temporal_data: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Time-series events (status / check-ups)",
    )
    spatial_data: list[dict[str, Any]] = Field(
        default_factory=list, description="GPS / location pings"
    )
    profile_data: dict[str, Any] = Field(
        default_factory=dict, description="Static user profile"
    )
    context_data: str = Field(
        default="", description="Free-text context (personas / notes)"
    )
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Engineered numeric features (populated by Layer 0)",
    )


# ── Agent Verdict ───────────────────────────────────────────────────────

class AgentVerdict(BaseModel):
    """Structured output from any agent (Layer 0 / 1 / 2)."""

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

    role: str = Field(..., description="The data role this consensus covers")
    n_agents: int = Field(
        default=1, ge=1, description="Number of agents that voted"
    )
    agreement_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Fraction of agents that agree with the majority prediction",
    )
    complexity_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Assessed complexity for this role (0=trivial, 1=maximally ambiguous)",
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
    layer_decided: str = Field(
        ..., description="Which layer emitted the final decision"
    )
    verdicts: list[AgentVerdict] = Field(
        default_factory=list,
        description="Ordered trail of per-agent verdicts",
    )
