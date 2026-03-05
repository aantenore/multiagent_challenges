"""
ManifestManager — loads and validates the manifest.json file.
Supports N-stage train/eval pipelines with per-level training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from models import Manifest, ManifestEntry, Stage

logger = logging.getLogger(__name__)


class ManifestManager:
    """Loads manifest.json and provides stage-aware access to entries."""

    def __init__(self, manifest_path: str | Path) -> None:
        self._path = Path(manifest_path)
        self._manifest: Manifest | None = None

    # ── Public API ──────────────────────────────────────────────────────

    def load(self) -> Manifest:
        """Parse and validate the manifest file."""
        logger.info("Loading manifest from %s", self._path)
        if not self._path.exists():
            raise FileNotFoundError(f"Manifest not found: {self._path}")

        raw = json.loads(self._path.read_text(encoding="utf-8"))

        # Accept both {"stages": [...]} / {"sources": [...]} / bare [...]
        if isinstance(raw, list):
            entries = [ManifestEntry(**item) for item in raw]
            self._manifest = Manifest(sources=entries)
        elif isinstance(raw, dict):
            self._manifest = Manifest(**raw)
        else:
            raise ValueError(
                "manifest.json must be a list or an object with 'stages' or 'sources'"
            )

        stages = self._manifest.get_stages()
        logger.info(
            "Manifest loaded: %d stages — %s",
            len(stages),
            [s.name for s in stages],
        )
        return self._manifest

    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:
            raise RuntimeError("Call .load() before accessing .manifest")
        return self._manifest

    @property
    def stages(self) -> list[Stage]:
        """Ordered list of stages."""
        return self.manifest.get_stages()

    def cumulative_training_sources(self, up_to_stage: int) -> list[ManifestEntry]:
        """Return training sources accumulated from stage 0..up_to_stage (inclusive)."""
        all_sources: list[ManifestEntry] = []
        for stage in self.stages[: up_to_stage + 1]:
            all_sources.extend(stage.training_sources)
        return all_sources

    def cumulative_training_roles(self, up_to_stage: int) -> set[str]:
        """Distinct roles across all cumulative training sources."""
        return {e.role for e in self.cumulative_training_sources(up_to_stage)}

    @property
    def base_dir(self) -> Path:
        """Directory containing the manifest file (used for relative paths)."""
        return self._path.parent
