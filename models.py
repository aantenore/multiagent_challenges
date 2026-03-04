"""
Pydantic data models shared across every layer of the pipeline.
Every inter-agent exchange is validated through these schemas.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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


class Manifest(BaseModel):
    """Top-level manifest wrapping a list of data sources."""

    sources: list[ManifestEntry] = Field(
        default_factory=list, description="Data-source entries"
    )


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


# ── Pipeline Result ─────────────────────────────────────────────────────

class PipelineResult(BaseModel):
    """Final output record for one entity after the full pipeline."""

    entity_id: str
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
