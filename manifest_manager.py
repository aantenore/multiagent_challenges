"""
ManifestManager — loads and validates the manifest.json file
that describes every data source used by the pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from models import Manifest, ManifestEntry

logger = logging.getLogger(__name__)


class ManifestManager:
    """Loads manifest.json, validates it, and groups entries by role."""

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

        # Accept both {"sources": [...]} and bare [...]
        if isinstance(raw, list):
            entries = [ManifestEntry(**item) for item in raw]
            self._manifest = Manifest(sources=entries)
        elif isinstance(raw, dict) and "sources" in raw:
            self._manifest = Manifest(**raw)
        else:
            raise ValueError(
                "manifest.json must be a list or an object with a 'sources' key"
            )

        logger.info(
            "Manifest loaded: %d sources across roles %s",
            len(self._manifest.sources),
            sorted({e.role for e in self._manifest.sources}),
        )
        return self._manifest

    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:
            raise RuntimeError("Call .load() before accessing .manifest")
        return self._manifest

    def entries_by_role(
        self, role: str
    ) -> list[ManifestEntry]:
        """Return all manifest entries matching a given role."""
        return [e for e in self.manifest.sources if e.role == role]

    @property
    def roles(self) -> set[str]:
        """Distinct roles present in the manifest."""
        return {e.role for e in self.manifest.sources}

    @property
    def base_dir(self) -> Path:
        """Directory containing the manifest file (used for relative paths)."""
        return self._path.parent
