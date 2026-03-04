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
        temporal_frames: list[tuple[ManifestEntry, pd.DataFrame]] = []
        spatial_frames: list[tuple[ManifestEntry, pd.DataFrame]] = []
        profile_frames: list[tuple[ManifestEntry, pd.DataFrame]] = []
        context_frames: list[tuple[ManifestEntry, pd.DataFrame]] = []

        role_buckets = {
            "temporal": temporal_frames,
            "spatial": spatial_frames,
            "profile": profile_frames,
            "context": context_frames,
        }

        for entry in self._entries:
            df = load_file(entry, self._base_dir)
            bucket = role_buckets.get(entry.role)
            if bucket is not None:
                bucket.append((entry, df))
            else:
                logger.warning("Unknown role %r — skipping", entry.role)

        # Discover all unique entity IDs across every source
        all_ids: set[str] = set()
        for _, (entry, df) in self._iter_all(role_buckets):
            all_ids.update(df[entry.id_column].astype(str).unique())

        logger.info("Building dossiers for %d entities", len(all_ids))

        dossiers: dict[str, EntityDossier] = {}
        for eid in sorted(all_ids):
            dossiers[eid] = EntityDossier(
                entity_id=eid,
                temporal_data=self._rows_for(eid, temporal_frames),
                spatial_data=self._rows_for(eid, spatial_frames),
                profile_data=self._profile_for(eid, profile_frames),
                context_data=self._context_for(eid, context_frames),
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
