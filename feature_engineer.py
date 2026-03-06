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

    For each numeric column in the domain data it computes:
    - mean, std, min, max
    - delta  (last - first)
    - velocity (delta / window_span_days)
    """

    def __init__(self, window_size: int | None = None) -> None:
        self._ws = window_size or 3

    # ── Public API ──────────────────────────────────────────────────────

    def extract(self, dossier: EntityDossier) -> dict[str, float]:
        """Return a flat dict of numeric features for one entity."""
        features: dict[str, float] = {}

        # --- Dynamic Domain features ---
        # Automatically process every role in domain_data (temporal, spatial, or custom)
        for role, rows in dossier.domain_data.items():
            if rows:
                features.update(self._extract_generic_features(role, rows))

        # --- Profile features ---
        if dossier.profile_data:
            features.update(self._profile_features(dossier.profile_data))

        return features

    # ── Temporal ────────────────────────────────────────────────────────

    # --- Abstract Analytics ---

    def _extract_generic_features(self, role: str, rows: list[dict[str, Any]]) -> dict[str, float]:
        """Compute per-column window stats + trend features for ANY role."""
        feats: dict[str, float] = {}
        if not rows:
            return feats

        # Convert to DataFrame
        df = pd.DataFrame(rows)
        feats["n_events"] = float(len(df))
        
        # Ensure Timestamp is properly converted and set as index
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp"])
            df = df.sort_values(by="Timestamp").set_index("Timestamp")
        else:
            return feats

        # Auto-detect numeric columns (excluding IDs and ignored columns)
        cfg = get_settings()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_process = [c for c in numeric_cols if c not in cfg.feature_ignore_columns]

        # Per-numeric-column sliding window
        for col in cols_to_process:
            series = df[col].dropna().astype(float)
            if series.empty:
                continue

            feats[f"{role}_{col}_mean"] = float(series.mean())
            feats[f"{role}_{col}_std"] = float(series.std()) if len(series) > 1 else 0.0
            feats[f"{role}_{col}_min"] = float(series.min())
            feats[f"{role}_{col}_max"] = float(series.max())
            feats[f"{role}_{col}_delta"] = float(series.iloc[-1] - series.iloc[0])

            # Full-series first-to-last delta
            feats[f"{role}_{col}_total_delta"] = feats[f"{role}_{col}_delta"]

            # Velocity: delta over time span (in days)
            timespan_days = (series.index[-1] - series.index[0]).total_seconds() / 86400.0
            if timespan_days > 0:
                feats[f"{role}_{col}_velocity"] = feats[f"{role}_{col}_delta"] / timespan_days
            else:
                feats[f"{role}_{col}_velocity"] = 0.0

            # ── Moving Averages (True time-based windows) ──
            if not series.empty:
                # Primary window
                roll3 = series.rolling(cfg.default_window_ma3).mean()
                ma3 = float(roll3.iloc[-1])
                feats[f"{role}_{col}_ma3"] = ma3
                feats[f"{role}_{col}_ma3_deviation"] = float(series.iloc[-1] - ma3)
                
                # Secondary window
                roll7 = series.rolling(cfg.default_window_ma7).mean()
                ma7 = float(roll7.iloc[-1])
                feats[f"{role}_{col}_ma7"] = ma7
                feats[f"{role}_{col}_ma7_deviation"] = float(series.iloc[-1] - ma7)

            # ── Linear Slope / Trend ──
                x = (series.index - series.index[0]).total_seconds() / 86400.0
                slope = float(np.polyfit(x, series.values, 1)[0])
                feats[f"{role}_{col}_slope"] = slope

            # ── Dynamic Sizing via ACF (Autocorrelation) ──
            # Re-sample smoothly to daily frequency to calculate ACF reliably if we have enough span
            if len(series) >= 5 and timespan_days > 2:
                # Resample to daily frequency (mean) and forward-fill missing
                daily_series = series.resample('1D').mean().ffill()
                if len(daily_series) >= 5:
                    # nlags defaults to min(10, len(daily_series) - 1)
                    nlags = min(15, len(daily_series) - 1)
                    try:
                        acf_vals = acf(daily_series, nlags=nlags, fft=True)
                        # Find highest peak ignoring lag 0
                        if len(acf_vals) > 1:
                            best_lag = int(np.argmax(acf_vals[1:])) + 1
                            optimal_days = best_lag
                            logger.debug(f"[{col}] ACF found optimal lag: {optimal_days} days (from {len(daily_series)} daily samples)")
                        else:
                            optimal_days = self._ws
                            logger.debug(f"[{col}] ACF failed to find peak, fallback to default window: {self._ws} days")
                    except Exception as e:
                        logger.warning(f"ACF failed for {col}: {e}")
                        optimal_days = self._ws
                else:
                    optimal_days = self._ws
                    logger.debug(f"[{col}] Not enough daily samples for ACF ({len(daily_series)}), fallback to default: {self._ws} days")
            else:
                optimal_days = self._ws
                logger.debug(f"[{col}] Series too short or narrow for ACF (len={len(series)}, span={timespan_days:.1f}d), fallback: {self._ws} days")
                
            # Compute dynamic features based on optimal lag
            feats[f"{col}_optimal_lag_days"] = float(optimal_days)
            roll_dynamic = series.rolling(f"{int(optimal_days)}D").mean()
            dyn_mean = float(roll_dynamic.iloc[-1])
            feats[f"{col}_dynamic_mean"] = dyn_mean
            feats[f"{col}_dynamic_deviation"] = float(series.iloc[-1] - dyn_mean)

        return feats

    # ── Specialized Geometric Extras ────────────────────────────────────

    def _spatial_features(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        """Compute generic mobility metrics if lat/lng are present."""
        feats: dict[str, float] = {}
        lats = [float(r.get("lat", 0)) for r in rows if "lat" in r]
        lngs = [float(r.get("lng", 0)) for r in rows if "lng" in r]

        if not lats: return feats

        feats["n_locations"] = float(len(lats))
        feats["lat_std"] = float(np.std(lats))
        feats["lng_std"] = float(np.std(lngs))

        # Approx mobility radius (std of distance from centroid in km)
        clat, clng = np.mean(lats), np.mean(lngs)
        dists = [self._haversine(clat, clng, la, lo) for la, lo in zip(lats, lngs)]
        feats["mobility_radius_km"] = float(np.std(dists))
        feats["max_dist_from_home_km"] = float(max(dists)) if dists else 0.0

        return feats

    # ── Profile ─────────────────────────────────────────────────────────

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
