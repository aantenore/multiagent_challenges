"""
Pillar 1 — Mathematical Streaming Filter (Project Antigravity).

Architecture:
- **Autocorrelation (ACF):** Detects the natural behavioral frequency per entity.
- **Incident Trigger:** Detects deviations > 2 Std Dev in sliding windows.
- **Incident Cropping:** Identifies the relevant temporal window for LLM Swarm analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from models import DetectionMetadata, EntityDossier, AgentVerdict
from settings import get_settings

logger = logging.getLogger(__name__)


class MathematicalFilter:
    """The Mathematical Streaming Filter for Project Antigravity (Pillar 1)."""

    def __init__(self) -> None:
        self._cfg = get_settings()
        logger.info("Antigravity Mathematical Filter Initialized (Pillar 1).")

    def analyze_incident(self, entity_id: str, dossier: EntityDossier) -> tuple[AgentVerdict, float, DetectionMetadata]:
        """Analyze data to find and crop an 'Anomaly Incident'.
        
        Returns:
            (AgentVerdict, complexity_score, DetectionMetadata)
        """
        if not self._cfg.pillar_filter_enabled:
            logger.info("  [Pillar 1] Filter disabled via config.")
            meta = DetectionMetadata(is_anomalous=False, report="Filter Pillar disabled.")
            verdict = AgentVerdict(agent_name="Pillar1_Filter", prediction=0, confidence=0.0, reasoning="Disabled")
            return verdict, 1.0, meta
        
        # 1. Assemble Time Series
        df = self._assemble_series(dossier)
        if df is None or len(df) < self._cfg.filter_baseline_min_days:
            meta = DetectionMetadata(
                is_anomalous=False, 
                report=f"Insufficient data for Pillar 1 analysis (need {self._cfg.filter_baseline_min_days} days)."
            )
            verdict = AgentVerdict(agent_name="Pillar1_Filter", prediction=0, confidence=0.0, reasoning="Low data")
            return verdict, 1.0, meta

        # 2. Determine Natural Frequency via ACF
        target_col = self._get_target_col(df)
        if not target_col:
            meta = DetectionMetadata(is_anomalous=False, report="No suitable numeric target column found.")
            verdict = AgentVerdict(agent_name="Pillar1_Filter", prediction=0, confidence=0.0, reasoning="No target col")
            return verdict, 1.0, meta
            
        natural_frequency = self._detect_frequency(df[target_col])
        logger.debug(f"Entity {entity_id} detected frequency: {natural_frequency} days")

        # 3. Slide Window & Trigger
        incident_idx, max_z = self._find_incident_trigger(df, target_col, natural_frequency)
        
        if incident_idx is None:
            confidence = min(1.0, max(0.5, 1.0 - (max_z / self._cfg.filter_z_score_threshold)))
            meta = DetectionMetadata(
                is_anomalous=False, max_z_score=max_z,
                report=f"No significant deviation detected. Max Z-Score={max_z:.2f} (threshold={self._cfg.filter_z_score_threshold})."
            )
            verdict = AgentVerdict(agent_name="Pillar1_Filter", prediction=0, confidence=confidence, reasoning=f"Sub-threshold: max Z={max_z:.2f}")
            return verdict, 1.0, meta

        # 4. Crop Incident Window
        trigger_time = df.index[incident_idx]
        start_time = trigger_time - pd.Timedelta(days=self._cfg.filter_incident_lookback_days)
        end_time = df.index[-1] 

        report = f"Anomaly detected in {target_col}. Triggered at {trigger_time.date()} with Z-Score={max_z:.2f}"
        
        meta = DetectionMetadata(
            is_anomalous=True,
            confidence=min(1.0, max_z / 5.0),
            max_z_score=max_z,
            detection_type="mathematical_trigger",
            incident_start=str(start_time),
            incident_end=str(end_time),
            report=report
        )
        
        verdict = AgentVerdict(
            agent_name="Pillar1_Filter",
            prediction=1,
            confidence=meta.confidence,
            reasoning=report
        )
        
        # Complexity scales with the magnitude of the deviation (Z-Score)
        # We cap it at 1.0 for the Swarm Model but keep the raw score for auditing
        complexity = min(1.0, max(0.1, max_z / 5.0))
        
        return verdict, complexity, meta

    def _assemble_series(self, dossier: EntityDossier) -> pd.DataFrame | None:
        """Merge all domain data into a daily sampled DataFrame."""
        all_rows = []
        for role_data in dossier.domain_data.values():
            all_rows.extend(role_data)
        
        if not all_rows: return None
        
        df = pd.DataFrame(all_rows)
        ts_col = self._cfg.timestamp_col
        if ts_col not in df.columns: return None
        
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.sort_values(ts_col).set_index(ts_col)
        
        # Resample to daily frequency
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty: return None
        return numeric_df.resample('1D').mean().ffill()

    def _get_target_col(self, df: pd.DataFrame) -> str:
        """Find the best column for behavioral analysis."""
        if self._cfg.filter_primary_col in df.columns:
            return self._cfg.filter_primary_col
        if self._cfg.filter_secondary_col in df.columns:
            return self._cfg.filter_secondary_col
        
        # Fallback to the first numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns
        return num_cols[0] if len(num_cols) > 0 else ""

    def _detect_frequency(self, series: pd.Series) -> int:
        """Find the peak autocorrelation lag and log the narrative window choice."""
        try:
            max_lags = min(30, len(series) // 2)
            if max_lags < 2: 
                logger.info("  [Pillar 1] Series too short for frequency detection. Using default 7-day window.")
                return 7
            
            acf_vals = acf(series, nlags=max_lags, fft=True)
            if len(acf_vals) <= 1: 
                logger.info("  [Pillar 1] No clear autocorrelation found. Falling back to weekly cycle (7 days).")
                return 7
            
            peak_lag = int(np.argmax(acf_vals[1:])) + 1
            logger.info("  [Pillar 1] Detected natural rhythm: %d days. Analytical window set to %d days (2x cycles).", peak_lag, peak_lag*2)
            return peak_lag
        except Exception as e:
            logger.debug("ACF detection failed: %s", e)
            return 7

    def _find_incident_trigger(self, df: pd.DataFrame, col: str, window_size: int) -> tuple[int | None, float]:
        """Slide window and return the index where deviation > threshold."""
        series = df[col]
        
        rolling_mean = series.rolling(window=window_size*6, min_periods=window_size).mean()
        rolling_std = series.rolling(window=window_size*6, min_periods=window_size).std()
        
        max_z = 0.0
        trigger_idx = None
        all_z_scores = []
        
        for i in range(window_size, len(series)):
            baseline_mean = rolling_mean.iloc[i-1]
            baseline_std = rolling_std.iloc[i-1]
            current_val = series.iloc[i]
            
            if pd.isna(baseline_mean) or pd.isna(baseline_std) or baseline_std <= 1e-6:
                continue
            
            z_score = abs(current_val - baseline_mean) / baseline_std
            all_z_scores.append((z_score, i, current_val, baseline_mean, baseline_std))
            
            if z_score > max_z:
                max_z = z_score
            
            if z_score > self._cfg.filter_z_score_threshold and trigger_idx is None:
                trigger_idx = i
                logger.info(
                    "  [Pillar 1] TRIGGER! Value (%.2f) deviates from baseline (Mean:%.2f, Std:%.2f) with Z-Score: %.2f",
                    current_val, baseline_mean, baseline_std, z_score
                )
                
        if trigger_idx is None:
            # Narrative reasoning for non-triggering: show near misses
            sorted_z = sorted(all_z_scores, key=lambda x: x[0], reverse=True)[:3]
            reasoning = "Top near-misses: " + ", ".join([f"Z={z:.2f} (at idx {idx})" for z, idx, _, _, _ in sorted_z])
            logger.info("  [Pillar 1] Analysis complete: No behavioral deviations > 2.0 found. %s", reasoning)
            
        return trigger_idx, max_z
