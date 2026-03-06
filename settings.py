from functools import lru_cache
from pathlib import Path
from typing import Any, Annotated, Dict, List

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
    pipeline_max_workers: int = 4

    # ── Langfuse Observability ──────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    team_name: str = "A(CC)I-Tua"

    # ── Mathematical Filter (Pillar 1) ──────────────────────────────────
    filter_upper_threshold: float = 0.0  # Set to 0.0 to disable 'Grey Area' (all stable entities skip LLMs)
    filter_skip_enabled: bool = True
    filter_z_score_threshold: float = 2.0  # Z-Score deviation to trigger anomaly
    fp_cost: float = 3.0
    fn_cost: float = 5.0

    # ── Project Antigravity Pillars ─────────────────────────────────────
    pillar_filter_enabled: bool = True       # Pillar 1
    pillar_memory_enabled: bool = True       # Pillar 2
    pillar_squads_enabled: bool = True       # Pillar 3
    pillar_orchestrator_enabled: bool = True # Pillar 4
    
    # ── Mathematical Filter Settings ────────────────────────────────────
    filter_primary_col: str = "BehavioralIndex"
    filter_secondary_col: str = "TrendIndex"
    filter_baseline_min_days: int = 14
    filter_incident_lookback_days: int = 3

    # ── Role Names (Configurable Abstractions) ──────────────────────────
    profile_role: str = "profile"
    context_role: str = "context"
    timestamp_col: str = "Timestamp"

    # ── Token Compression ────────────────────────────────────────────────
    token_compression_enabled: bool = True
    acronym_map: Dict[str, str] = {}  # Explicit column→acronym overrides
    spatial_coordinate_cols: List[str] = []  # e.g. ["lat", "lng"] — set via manifest
    persona_dense_prefixes: List[str] = [  # Lines starting with these are kept
        "Mobility", "Daily routine", "Work pattern",
        "Risk factor", "Living situation", "Financial",
    ]

    # ── Feature Engineering Configuration ───────────────────────────────
    feature_ignore_columns: List[str] = [
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
    swarm_skip_threshold: float = 0.9  # Min confidence to skip Pillar 4 if unanimous
    swarm_skip_enabled: bool = True

    # ── Squad Configuration ──────────────────────────────────────────────
    squad_base_agents: int = 1
    squad_dataset_scaling_factor: float = 0.005  # n_agents += int(dataset_size * factor)
    autocorr_max_lag: int = 30
    
    # Unified Squad Registry: Rules (roles), Prompts, and Scaling
    squad_configs: Dict[str, Dict[str, Any]] = {
        "profile_context": {
            "roles": ["profile", "context"],
            "prompt": "squad_profile_context",
            "base_agents": 1,
        },
        "spatial_patterns": {
            "roles": ["spatial"],
            "prompt": "squad_spatial_patterns",
            "base_agents": 1,
        },
        "temporal_routines": {
            "roles": ["temporal"],
            "prompt": "squad_temporal_routines",
            "base_agents": 1,
        },
        "health_behavioral": {
            "roles": ["activity", "health", "sleep", "pharmacy", "hospital"],
            "prompt": "squad_health_behavioral",
            "base_agents": 1,
        }
    }
    
    # ── RAG Hierarchy ───────────────────────────────────────────────────
    rag_individual_collection: str = "individual_memory"
    rag_geo_collection: str = "geo_local_memory"
    rag_global_collection: str = "global_architectural_memory"
    top_k_rag: int = 3
    rag_db_path: str = "./chroma_db"

    # ── Validators ──────────────────────────────────────────────────────
    @field_validator("filter_upper_threshold")
    @classmethod
    def _upper_gt_zero(cls, v: float) -> float:
        """Validate that the upper mathematical filter threshold is between 0 and 1."""
        if not (0.0 < v < 1.0):
            raise ValueError("filter_upper_threshold must be in (0, 1)")
        return v

    # ── Derived helpers ─────────────────────────────────────────────────
    @property
    def rag_db_dir(self) -> Path:
        """Return the configured directory path for the Chroma vector database as a Path object."""
        return Path(self.rag_db_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor for Settings."""
    settings = Settings()
    return settings
