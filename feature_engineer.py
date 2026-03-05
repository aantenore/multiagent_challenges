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

logger = logging.getLogger(__name__)


class SlidingWindowExtractor:
    """Extract trend features over configurable sliding windows.

    For each numeric column in the temporal data it computes:
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
        """Compute per-column window stats + event-type features.

        Uses pandas to handle true time-based rolling windows (3D, 7D),
        properly accounting for irregular sampling intervals.
        Additionally computes Autocorrelation (ACF) to discover natural
        cycles (dynamic sizing) and computes features based on optimal lag.
        """
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

        # Per-numeric-column sliding window
        for col in self._NUMERIC_COLS:
            if col not in df.columns:
                continue
                
            series = df[col].dropna().astype(float)
            if series.empty:
                continue

            feats[f"{col}_mean"] = float(series.mean())
            feats[f"{col}_std"] = float(series.std()) if len(series) > 1 else 0.0
            feats[f"{col}_min"] = float(series.min())
            feats[f"{col}_max"] = float(series.max())
            feats[f"{col}_delta"] = float(series.iloc[-1] - series.iloc[0])

            # Full-series first-to-last delta (same as above since we process all available series)
            feats[f"{col}_total_delta"] = feats[f"{col}_delta"]

            # Velocity: delta over time span (in days)
            timespan_days = (series.index[-1] - series.index[0]).total_seconds() / 86400.0
            if timespan_days > 0:
                feats[f"{col}_velocity"] = feats[f"{col}_delta"] / timespan_days
            else:
                feats[f"{col}_velocity"] = 0.0

            # ── Moving Averages (True time-based 3D and 7D windows) ──
            if not series.empty:
                # 3 days
                roll3 = series.rolling("3D").mean()
                ma3 = float(roll3.iloc[-1])
                feats[f"{col}_ma3"] = ma3
                feats[f"{col}_ma3_deviation"] = float(series.iloc[-1] - ma3)
                
                # 7 days
                roll7 = series.rolling("7D").mean()
                ma7 = float(roll7.iloc[-1])
                feats[f"{col}_ma7"] = ma7
                feats[f"{col}_ma7_deviation"] = float(series.iloc[-1] - ma7)

            # ── Linear Slope / Trend ──
            if len(series) >= 3:
                # Use days since first event as X axis for proper slope
                x = (series.index - series.index[0]).total_seconds() / 86400.0
                slope = float(np.polyfit(x, series.values, 1)[0])
                feats[f"{col}_slope"] = slope

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

        # Event-type risk encoding
        if "EventType" in df.columns:
            event_types = df["EventType"].dropna().astype(str).tolist()
            risk_scores = [self._EVENT_RISK_MAP.get(e, 0) for e in event_types]
            feats["max_event_risk"] = float(max(risk_scores)) if risk_scores else 0.0
            feats["mean_event_risk"] = float(np.mean(risk_scores)) if risk_scores else 0.0
            feats["has_emergency"] = 1.0 if any(s >= 3 for s in risk_scores) else 0.0
            feats["has_specialist"] = 1.0 if "specialist consultation" in event_types else 0.0
            feats["n_preventive_screenings"] = float(
                sum(1 for e in event_types if e == "preventive screening")
            )
        else:
            feats["max_event_risk"] = 0.0
            feats["mean_event_risk"] = 0.0
            feats["has_emergency"] = 0.0
            feats["has_specialist"] = 0.0
            feats["n_preventive_screenings"] = 0.0

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
