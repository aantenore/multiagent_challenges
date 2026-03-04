"""
Feature engineering with configurable sliding windows.
Extracts statistical trend features from temporal data.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from models import EntityDossier
from settings import get_settings

logger = logging.getLogger(__name__)


class SlidingWindowExtractor:
    """Extract trend features over configurable sliding windows.

    For each numeric column in the temporal data it computes:
    - mean, std, min, max
    - delta  (last - first)
    - velocity (delta / window_span_days)
    """

    def __init__(self, window_size: int | None = None) -> None:
        self._ws = window_size or get_settings().window_size

    # ── Public API ──────────────────────────────────────────────────────

    def extract(self, dossier: EntityDossier) -> dict[str, float]:
        """Return a flat dict of numeric features for one entity."""
        features: dict[str, float] = {}

        # --- Temporal features ---
        temporal = dossier.temporal_data
        if temporal:
            features.update(self._temporal_features(temporal))

        # --- Spatial features ---
        spatial = dossier.spatial_data
        if spatial:
            features.update(self._spatial_features(spatial))

        # --- Profile features ---
        if dossier.profile_data:
            features.update(self._profile_features(dossier.profile_data))

        return features

    # ── Temporal ────────────────────────────────────────────────────────

    _NUMERIC_COLS = {
        "PhysicalActivityIndex",
        "SleepQualityIndex",
        "EnvironmentalExposureLevel",
    }

    _EVENT_RISK_MAP: dict[str, int] = {
        "routine check-up": 0,
        "lifestyle coaching session": 0,
        "preventive screening": 1,
        "follow-up assessment": 2,
        "emergency visit": 3,
        "specialist consultation": 3,
    }

    def _temporal_features(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        """Compute per-column window stats + event-type features."""
        feats: dict[str, float] = {}

        # Sort by timestamp
        sorted_rows = sorted(rows, key=lambda r: str(r.get("Timestamp", "")))
        n = len(sorted_rows)
        feats["n_events"] = float(n)

        # Per-numeric-column sliding window
        for col in self._NUMERIC_COLS:
            values = [
                float(r[col]) for r in sorted_rows
                if col in r and r[col] is not None
            ]
            if not values:
                continue
            window = values[-self._ws:] if len(values) >= self._ws else values
            arr = np.array(window, dtype=float)

            feats[f"{col}_mean"] = float(np.mean(arr))
            feats[f"{col}_std"] = float(np.std(arr))
            feats[f"{col}_min"] = float(np.min(arr))
            feats[f"{col}_max"] = float(np.max(arr))
            feats[f"{col}_delta"] = float(arr[-1] - arr[0])

            # Full-series first-to-last delta
            all_arr = np.array(values, dtype=float)
            feats[f"{col}_total_delta"] = float(all_arr[-1] - all_arr[0])

            # Velocity: delta over time span (or over count as proxy)
            if len(values) > 1:
                feats[f"{col}_velocity"] = feats[f"{col}_delta"] / max(len(window) - 1, 1)
            else:
                feats[f"{col}_velocity"] = 0.0

        # Event-type risk encoding
        event_types = [r.get("EventType", "") for r in sorted_rows]
        risk_scores = [self._EVENT_RISK_MAP.get(e, 0) for e in event_types]
        feats["max_event_risk"] = float(max(risk_scores)) if risk_scores else 0.0
        feats["mean_event_risk"] = float(np.mean(risk_scores)) if risk_scores else 0.0
        feats["has_emergency"] = 1.0 if any(s >= 3 for s in risk_scores) else 0.0
        feats["has_specialist"] = 1.0 if "specialist consultation" in event_types else 0.0
        feats["n_preventive_screenings"] = float(
            sum(1 for e in event_types if e == "preventive screening")
        )

        return feats

    # ── Spatial ─────────────────────────────────────────────────────────

    def _spatial_features(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        """Compute mobility radius, location entropy, outlier travel count."""
        feats: dict[str, float] = {}
        lats = [float(r["lat"]) for r in rows if "lat" in r]
        lngs = [float(r["lng"]) for r in rows if "lng" in r]

        if not lats:
            return feats

        feats["n_locations"] = float(len(lats))
        feats["lat_std"] = float(np.std(lats))
        feats["lng_std"] = float(np.std(lngs))

        # Approx mobility radius (std of distance from centroid in km)
        clat, clng = np.mean(lats), np.mean(lngs)
        dists = [
            self._haversine(clat, clng, la, lo)
            for la, lo in zip(lats, lngs)
        ]
        feats["mobility_radius_km"] = float(np.std(dists))
        feats["max_dist_from_home_km"] = float(max(dists)) if dists else 0.0

        # Distinct cities
        cities = {r.get("city", "") for r in rows if r.get("city")}
        feats["n_distinct_cities"] = float(len(cities))

        return feats

    # ── Profile ─────────────────────────────────────────────────────────

    def _profile_features(self, profile: dict[str, Any]) -> dict[str, float]:
        feats: dict[str, float] = {}
        if "birth_year" in profile:
            try:
                feats["birth_year"] = float(profile["birth_year"])
            except (ValueError, TypeError):
                pass
        if "_travel_per_year" in profile:
            try:
                feats["travel_per_year"] = float(profile["_travel_per_year"])
            except (ValueError, TypeError):
                pass
        return feats

    # ── Utils ───────────────────────────────────────────────────────────

    @staticmethod
    def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Haversine distance in km."""
        R = 6371.0
        rlat1, rlng1, rlat2, rlng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        dlat = rlat2 - rlat1
        dlng = rlng2 - rlng1
        a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlng / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))
