"""
Feature engineering with configurable sliding windows.
Extracts statistical trend features from temporal data.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from models import EntityDossier
from settings import get_settings

logger = logging.getLogger(__name__)


class SlidingWindowExtractor:
    """Extract trend features over configurable sliding windows.

    For each numeric column in the domain data it computes statistical metrics.
    """

    def __init__(self, window_size: int | None = None) -> None:
        self._ws = window_size or 3

    # ── Public API ──────────────────────────────────────────────────────

    def extract(self, dossier: EntityDossier) -> dict[str, float]:
        """Return a flat dict of numeric features for one entity."""
        features: dict[str, float] = {}

        # --- Dynamic Domain features ---
        for role, rows in dossier.domain_data.items():
            if rows:
                features.update(self._extract_generic_features(role, rows))

        # --- Profile features ---
        if dossier.profile_data:
            features.update(self._profile_features(dossier.profile_data))

        return features

    # ── Temporal ────────────────────────────────────────────────────────

    def get_optimal_windows(self, dossier: EntityDossier) -> dict[str, int]:
        """Determine optimal windows for the specialized analytical squads."""
        cfg = get_settings()
        ts_col = cfg.timestamp_col
        all_numeric_series = []
        
        for rows in dossier.domain_data.values():
            if not rows: continue
            df = pd.DataFrame(rows)
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce").dropna()
                df = df.sort_values(ts_col)
                numeric_df = df.select_dtypes(include=[np.number])
                for col in numeric_df.columns:
                    if col not in cfg.feature_ignore_columns:
                        all_numeric_series.append(df[col])

        # Default fallback windows (mapped to squad config keys)
        windows = {
            "profile_context": 3,
            "spatial_patterns": 7,
            "temporal_routines": 30,
            "health_behavioral": 90
        }

        if not all_numeric_series:
            return windows

        try:
            consensus_lags = []
            for s in all_numeric_series:
                if len(s) < 10: continue
                # Resample to daily if possible
                s_daily = s.resample('1D').mean().ffill() if hasattr(s.index, 'freq') else s
                lags = min(cfg.autocorr_max_lag, len(s_daily) - 1)
                if lags < 2: continue
                acf_vals = acf(s_daily, nlags=lags, fft=True)
                if len(acf_vals) > 1:
                    peak = int(np.argmax(acf_vals[1:])) + 1
                    consensus_lags.append(peak)
            
            if consensus_lags:
                median_lag = int(np.median(consensus_lags))
                windows["profile_context"] = max(1, median_lag // 2)
                windows["spatial_patterns"] = max(7, median_lag)
                windows["temporal_routines"] = max(30, median_lag * 2)
                windows["health_behavioral"] = max(2, median_lag // 4)
        except Exception as e:
            logger.warning(f"Consensus ACF failed: {e}. Using fallback windows.")

        return windows

    def extract_window(self, dossier: EntityDossier, squad: str, window_days: int, end_time: pd.Timestamp) -> dict[str, float]:
        """Extract features for a specific squad within a given time window."""
        start_time = end_time - pd.Timedelta(days=window_days)
        sliced_dossier = self._slice_dossier(dossier, start_time, end_time)
        
        feats = self.extract(sliced_dossier)
        return {f"{squad}_{k}": v for k, v in feats.items()}

    def _slice_dossier(self, dossier: EntityDossier, start: pd.Timestamp, end: pd.Timestamp) -> EntityDossier:
        """Return a slice of the dossier between start and end timestamps."""
        cfg = get_settings()
        ts_col = cfg.timestamp_col
        new_domain_data = {}
        for role, rows in dossier.domain_data.items():
            if not rows:
                new_domain_data[role] = []
                continue
            df = pd.DataFrame(rows)
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                mask = (df[ts_col] >= start) & (df[ts_col] <= end)
                new_domain_data[role] = df.loc[mask].to_dict(orient="records")
            else:
                new_domain_data[role] = rows
        
        return EntityDossier(
            entity_id=dossier.entity_id,
            domain_data=new_domain_data,
            profile_data=dossier.profile_data,
            context_data=dossier.context_data
        )

    def _extract_generic_features(self, role: str, rows: list[dict[str, Any]]) -> dict[str, float]:
        """Compute basic stats for any numeric columns in any domain role."""
        df = pd.DataFrame(rows)
        numeric_df = df.select_dtypes(include=[np.number])
        feats: dict[str, float] = {}

        for col in numeric_df.columns:
            if col in get_settings().feature_ignore_columns: continue
            
            series = numeric_df[col].dropna()
            if series.empty: continue

            prefix = f"{role}_{col}"
            feats[f"{prefix}_mean"] = float(series.mean())
            feats[f"{prefix}_std"] = float(series.std()) if len(series) > 1 else 0.0
            feats[f"{prefix}_max"] = float(series.max())
            feats[f"{prefix}_delta"] = float(series.iloc[-1] - series.iloc[0])

        # Spatial features: only if columns are mapped in settings
        cfg = get_settings()
        coord_cols = cfg.spatial_coordinate_cols
        if len(coord_cols) == 2:
            lat_col, lng_col = coord_cols
            if lat_col in df.columns and lng_col in df.columns:
                feats.update(self._spatial_features(rows, lat_col, lng_col))

        return feats

    def _spatial_features(self, rows: list[dict[str, Any]], lat_col: str = "lat", lng_col: str = "lng") -> dict[str, float]:
        """Compute generic mobility metrics from configurable coordinate columns."""
        feats: dict[str, float] = {}
        lats = [float(r.get(lat_col, 0)) for r in rows if lat_col in r]
        lngs = [float(r.get(lng_col, 0)) for r in rows if lng_col in r]

        if not lats: return feats

        feats["n_locations"] = float(len(lats))
        feats["lat_std"] = float(np.std(lats))
        feats["lng_std"] = float(np.std(lngs))

        # Approx mobility radius
        clat, clng = np.mean(lats), np.mean(lngs)
        dists = [self._haversine(clat, clng, la, lo) for la, lo in zip(lats, lngs)]
        feats["mobility_radius_km"] = float(np.std(dists)) if dists else 0.0

        return feats

    def _profile_features(self, profile: dict[str, Any], prefix: str = "") -> dict[str, float]:
        """Recursively extract all numeric features from any profile structure."""
        feats: dict[str, float] = {}
        for k, v in profile.items():
            key = f"{prefix}_{k}".lstrip("_")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                feats[key] = float(v)
            elif isinstance(v, dict):
                feats.update(self._profile_features(v, prefix=key))
        return feats

    @staticmethod
    def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Haversine distance in km."""
        R = 6371.0
        rlat1, rlng1, rlat2, rlng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        dlat = rlat2 - rlat1
        dlng = rlng2 - rlng1
        a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlng / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))
