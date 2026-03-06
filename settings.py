"""
Centralized configuration via Pydantic Settings.
Reads from .env file with sensible defaults for all parameters.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider ────────────────────────────────────────────────────
    llm_provider: str = "gemini"  # "openai" | "gemini"

    # ── LLM Provider Keys ───────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # ── Model Configuration ─────────────────────────────────────────────
    nano_model_name: str = "gpt-5-nano"
    cheap_model_name: str = "gpt-5-mini"
    smart_model_name: str = "gpt-5.4"
    
    gemini_nano_model_name: str = "gemini-3.1-flash-lite-preview"
    gemini_cheap_model_name: str = "gemini-3-flash-preview"
    gemini_smart_model_name: str = "gemini-3.1-pro-preview"
    
    model_temperature: float = 0.1
    max_agent_retries: int = 3

    # ── Langfuse Observability ──────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    team_name: str = "A(CC)I-Tua"

    # ── Pipeline Hyperparameters ────────────────────────────────────────
    l0_lower_threshold: float = 0.15
    l0_upper_threshold: float = 0.85
    fp_cost: float = 3.0
    fn_cost: float = 5.0

    # ── Layer 0 Engine ──────────────────────────────────────────────────
    # "isolation" (IsolationForest, zero-cost) 
    # "llm" (Nano model, very cheap)
    l0_engine: str = "llm"
    bypass_l0: bool = False

    # ── Role Names (Configurable Abstractions) ──────────────────────────
    profile_role: str = "profile"
    context_role: str = "context"

    # ── Feature Engineering Configuration ───────────────────────────────
    feature_ignore_columns: list[str] = [
        "CitizenID", "EventID", "user_id", "_user_id", 
        "Timestamp", "id", "ID", "entity_id"
    ]
    default_window_ma3: str = "3D"
    default_window_ma7: str = "7D"

    # ── Swarm Configuration ──────────────────────────────────────────────
    swarm_min_agents: int = 1
    swarm_max_agents: int = 5
    swarm_complexity_threshold: float = 0.2
    swarm_temp_spread: float = 0.5

    # ── RAG Configuration ───────────────────────────────────────────────
    rag_collection_name: str = "case_memory"
    top_k_rag: int = 3
    rag_db_path: str = "./chroma_db"

    # ── Validators ──────────────────────────────────────────────────────
    @field_validator("l0_upper_threshold")
    @classmethod
    def _upper_gt_zero(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("l0_upper_threshold must be in (0, 1)")
        return v

    @field_validator("l0_lower_threshold")
    @classmethod
    def _lower_gt_zero(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("l0_lower_threshold must be in (0, 1)")
        return v

    def validate_thresholds(self) -> None:
        """Cross-field validation: lower must be < upper."""
        if self.l0_lower_threshold >= self.l0_upper_threshold:
            raise ValueError(
                f"l0_lower_threshold ({self.l0_lower_threshold}) "
                f"must be < l0_upper_threshold ({self.l0_upper_threshold})"
            )

    # ── Derived helpers ─────────────────────────────────────────────────
    @property
    def rag_db_dir(self) -> Path:
        return Path(self.rag_db_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor for Settings."""
    settings = Settings()
    settings.validate_thresholds()
    return settings
