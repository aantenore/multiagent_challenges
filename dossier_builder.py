"""
DossierBuilder — assembles a Unified Entity Dossier per entity
by loading all manifest sources and joining on id_column per role.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_loader import load_file
from models import EntityDossier, ManifestEntry
from settings import get_settings

logger = logging.getLogger(__name__)


class DossierBuilder:
    """Build one :class:`EntityDossier` per unique entity ID."""

    def __init__(self, entries: list[ManifestEntry], base_dir: Path) -> None:
        self._entries = entries
        self._base_dir = base_dir

    # ── Factory ─────────────────────────────────────────────────────────

    @classmethod
    def from_entries(
        cls, entries: list[ManifestEntry], base_dir: Path
    ) -> DossierBuilder:
        """Create a builder from a raw list of ManifestEntry objects."""
        return cls(entries, base_dir)

    # ── Public ──────────────────────────────────────────────────────────

    def build_all(self) -> dict[str, EntityDossier]:
        """Load every source, merge by entity ID, and return dossiers."""
        # role_buckets: role_name -> list of (entry, dataframe)
        role_buckets: dict[str, list[tuple[ManifestEntry, pd.DataFrame]]] = {}

        for entry in self._entries:
            df = load_file(entry, self._base_dir)
            if entry.role not in role_buckets:
                role_buckets[entry.role] = []
            role_buckets[entry.role].append((entry, df))

        # Discover all unique entity IDs across every source
        all_ids: set[str] = set()
        for role, frames in role_buckets.items():
            for entry, df in frames:
                all_ids.update(df[entry.id_column].astype(str).unique())

        logger.info("Building dossiers for %d entities", len(all_ids))

        dossiers: dict[str, EntityDossier] = {}
        for eid in sorted(all_ids):
            # Dynamic domain data for all roles except special descriptors
            cfg = get_settings()
            domain_data: dict[str, list[dict[str, Any]]] = {}
            profile_data: dict[str, Any] = {}
            context_data: str = ""

            for role, frames in role_buckets.items():
                if role == cfg.profile_role:
                    profile_data = self._profile_for(eid, frames)
                elif role == cfg.context_role:
                    context_data = self._context_for(eid, frames)
                else:
                    domain_data[role] = self._rows_for(eid, frames)

            dossiers[eid] = EntityDossier(
                entity_id=eid,
                domain_data=domain_data,
                profile_data=profile_data,
                context_data=context_data,
            )

        return dossiers

    # ── Private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _iter_all(
        buckets: dict[str, list[tuple[ManifestEntry, pd.DataFrame]]],
    ):
        for role, frames in buckets.items():
            for pair in frames:
                yield role, pair

    @staticmethod
    def _rows_for(
        entity_id: str,
        frames: list[tuple[ManifestEntry, pd.DataFrame]],
    ) -> list[dict]:
        """Collect all rows matching *entity_id* across multiple files."""
        out: list[dict] = []
        for entry, df in frames:
            mask = df[entry.id_column].astype(str) == entity_id
            rows = df.loc[mask]
            out.extend(rows.to_dict(orient="records"))
        return out

    @staticmethod
    def _profile_for(
        entity_id: str,
        frames: list[tuple[ManifestEntry, pd.DataFrame]],
    ) -> dict:
        """Return the first matching profile row as a flat dict."""
        for entry, df in frames:
            mask = df[entry.id_column].astype(str) == entity_id
            rows = df.loc[mask]
            if not rows.empty:
                return rows.iloc[0].to_dict()
        return {}

    @staticmethod
    def _context_for(
        entity_id: str,
        frames: list[tuple[ManifestEntry, pd.DataFrame]],
    ) -> str:
        """Concatenate all matching context text for the entity."""
        parts: list[str] = []
        for entry, df in frames:
            id_col = entry.id_column
            mask = df[id_col].astype(str) == entity_id
            rows = df.loc[mask]
            for _, row in rows.iterrows():
                if "context_text" in row.index:
                    parts.append(str(row["context_text"]))
                else:
                    text_parts = [
                        str(v)
                        for k, v in row.items()
                        if k != id_col and isinstance(v, str)
                    ]
                    if text_parts:
                        parts.append(" | ".join(text_parts))
        return "\n\n".join(parts)
