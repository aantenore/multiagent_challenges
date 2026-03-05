"""
Layer 0 — Hybrid Ensemble Anomaly Meta-Router (Zero-Label Efficiency Engine).

Runs Z-Score (univariate) AND IsolationForest (multivariate) simultaneously.
Uses weighted ensemble to produce a DetectionMetadata object that explains
*why* a case was escalated, enabling L1 agents to focus on flagged signals.

Routing logic ("The Switch"):
- Both silent  → verdict=0 (standard monitoring, 0 LLM tokens, 100% saving)
- Z-Score only → "Statistical anomaly on [SENSOR]"  → escalate to L1
- Forest only  → "Multi-factor behavioural deviation"  → escalate to L1
- Both fire    → combined report → escalate to L1
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from models import AgentVerdict, DetectionMetadata, EntityDossier
from settings import get_settings

logger = logging.getLogger(__name__)


# ── Feature columns used for baseline computation ───────────────────────
_BASELINE_COLS = [
    "PhysicalActivityIndex",
    "SleepQualityIndex",
    "EnvironmentalExposureLevel",
]


class AnomalyRouter:
    """Hybrid Ensemble Meta-Router: Z-Score + IsolationForest.

    Workflow
    --------
    1. ``build_baselines(train_dossiers)``
       - Compute per-entity μ/σ from temporal data (Z-Score baselines).
       - Train population-level IsolationForest on feature matrix.
    2. ``to_verdict(entity_id, dossier)``
       - Run BOTH detectors simultaneously.
       - Ensemble score = sigma_weight × z_score + forest_weight × forest_score.
       - Emit verdict + DetectionMetadata for L1 explainability.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._threshold_sigma: float = cfg.anomaly_threshold
        self._threshold_forest: float = cfg.anomaly_threshold_forest
        self._min_samples: int = cfg.min_historical_samples
        self._sigma_weight: float = cfg.sigma_weight
        self._forest_weight: float = cfg.forest_weight

        # Per-entity baselines: {entity_id: {col: (mean, std)}}
        self._baselines: dict[str, dict[str, tuple[float, float]]] = {}

        # Population-level IsolationForest
        self._iso_forest: IsolationForest | None = None
        self._iso_feature_names: list[str] = []

        self._is_fitted = False
        logger.info(
            "AnomalyRouter initialised: ensemble mode, σ_threshold=%.2f, "
            "forest_threshold=%.2f, σ_weight=%.2f, forest_weight=%.2f",
            self._threshold_sigma,
            self._threshold_forest, self._sigma_weight, self._forest_weight,
        )

    # ── Baseline Building ───────────────────────────────────────────────

    def build_baselines(self, train_dossiers: dict[str, EntityDossier]) -> None:
        """Build per-entity μ/σ baselines AND train IsolationForest."""
        n_built = 0
        population_features: list[list[float]] = []

        for eid, dossier in train_dossiers.items():
            temporal = dossier.temporal_data
            if not temporal or len(temporal) < self._min_samples:
                continue

            sorted_rows = sorted(temporal, key=lambda r: str(r.get("Timestamp", "")))
            entity_baseline: dict[str, tuple[float, float]] = {}

            for col in _BASELINE_COLS:
                values = [
                    float(r[col]) for r in sorted_rows
                    if col in r and r[col] is not None
                ]
                if len(values) >= self._min_samples:
                    mu = float(np.mean(values))
                    sigma = float(np.std(values))
                    entity_baseline[col] = (mu, max(sigma, 1e-6))

            if entity_baseline:
                self._baselines[eid] = entity_baseline
                n_built += 1

            # Collect population feature vector
            if dossier.features:
                population_features.append(
                    [dossier.features.get(f"{c}_mean", 0.0) for c in _BASELINE_COLS]
                )

        # Always train IsolationForest when hybrid mode is on
        if len(population_features) >= 4:
            X = np.array(population_features, dtype=float)
            X = np.nan_to_num(X, nan=0.0)
            self._iso_forest = IsolationForest(
                n_estimators=100,
                contamination="auto",
                random_state=42,
            )
            self._iso_forest.fit(X)
            self._iso_feature_names = [f"{c}_mean" for c in _BASELINE_COLS]
            logger.info("IsolationForest trained on %d population samples", len(X))

        self._is_fitted = n_built > 0 or self._iso_forest is not None
        logger.info(
            "L0 baselines built for %d/%d entities (ensemble mode)",
            n_built, len(train_dossiers),
        )

    # ── Z-Score Detector (Univariate) ───────────────────────────────────

    def _zscore_detect(
        self, entity_id: str, dossier: EntityDossier,
    ) -> tuple[bool, float, dict[str, float]]:
        """Univariate Z-Score per feature.

        Returns (is_flagged, max_z_score, {feature: z_score}).
        """
        baseline = self._baselines.get(entity_id)

        if baseline is None:
            # Build live baseline from the entity's own data
            temporal = dossier.temporal_data
            if temporal and len(temporal) >= self._min_samples:
                baseline = {}
                sorted_rows = sorted(temporal, key=lambda r: str(r.get("Timestamp", "")))
                for col in _BASELINE_COLS:
                    values = [
                        float(r[col]) for r in sorted_rows
                        if col in r and r[col] is not None
                    ]
                    if len(values) >= self._min_samples:
                        mu = float(np.mean(values[:-1]))
                        sigma = float(np.std(values[:-1]))
                        baseline[col] = (mu, max(sigma, 1e-6))

        if not baseline:
            return False, 0.0, {}

        temporal = dossier.temporal_data
        if not temporal:
            return False, 0.0, {}

        sorted_rows = sorted(temporal, key=lambda r: str(r.get("Timestamp", "")))
        latest = sorted_rows[-1]

        z_scores: dict[str, float] = {}
        for col, (mu, sigma) in baseline.items():
            raw = latest.get(col)
            if raw is None:
                continue
            z = abs(float(raw) - mu) / sigma
            z_scores[col] = round(z, 3)

        if not z_scores:
            return False, 0.0, {}

        max_z = max(z_scores.values())
        # Flag features exceeding threshold
        deviating = {k: v for k, v in z_scores.items() if v > self._threshold_sigma}
        is_flagged = len(deviating) > 0

        return is_flagged, max_z, deviating

    # ── IsolationForest Detector (Multivariate) ─────────────────────────

    def _forest_detect(
        self, dossier: EntityDossier,
    ) -> tuple[bool, float]:
        """Multivariate IsolationForest anomaly detection.

        Returns (is_outlier, decision_score).
        """
        if self._iso_forest is None or not self._iso_feature_names:
            return False, 0.0

        x = np.array(
            [[dossier.features.get(f, 0.0) for f in self._iso_feature_names]],
            dtype=float,
        )
        x = np.nan_to_num(x, nan=0.0)

        score = float(self._iso_forest.decision_function(x)[0])
        prediction = int(self._iso_forest.predict(x)[0])  # 1=normal, -1=outlier

        is_outlier = prediction == -1
        return is_outlier, score

    # ── Hybrid Ensemble ─────────────────────────────────────────────────

    def _build_detection_metadata(
        self,
        entity_id: str,
        dossier: EntityDossier,
    ) -> DetectionMetadata:
        """Run both detectors and produce a structured DetectionMetadata."""
        # Z-Score (univariate)
        z_flagged, max_z, deviating = self._zscore_detect(entity_id, dossier)

        # IsolationForest (multivariate)
        f_flagged, f_score = self._forest_detect(dossier)

        # Determine detection type
        if z_flagged and f_flagged:
            det_type = "both"
        elif z_flagged:
            det_type = "statistical"
        elif f_flagged:
            det_type = "behavioural"
        else:
            det_type = "none"

        is_anomalous = z_flagged or f_flagged

        # Ensemble confidence
        z_conf = min(max_z / (2 * self._threshold_sigma), 1.0) if max_z > 0 else 0.0
        f_conf = min(abs(f_score) * 3, 1.0) if f_flagged else 0.0

        # Ensemble confidence (always weighted blend)
        if z_flagged or f_flagged:
            total_weight = self._sigma_weight + self._forest_weight
            confidence = (
                self._sigma_weight * z_conf + self._forest_weight * f_conf
            ) / total_weight
        else:
            confidence = 1.0 - max(z_conf, f_conf)  # confidence in normality

        confidence = min(1.0, max(0.0, confidence))

        # Build report
        report_parts: list[str] = []
        if z_flagged and deviating:
            sensors = ", ".join(f"{k} (z={v:.2f}σ)" for k, v in deviating.items())
            report_parts.append(f"Anomalia statistica rilevata nei sensori: {sensors}")
        if f_flagged:
            report_parts.append(
                f"Rilevata deviazione comportamentale multi-fattore "
                f"(isolation_score={f_score:.4f})"
            )
        if not report_parts:
            report_parts.append("Nessuna anomalia rilevata — profilo nella norma.")

        return DetectionMetadata(
            is_anomalous=is_anomalous,
            confidence=confidence,
            zscore_flagged=z_flagged,
            forest_flagged=f_flagged,
            detection_type=det_type,
            deviating_features=deviating,
            forest_score=f_score,
            report=" | ".join(report_parts),
        )

    # ── Public API ──────────────────────────────────────────────────────

    def to_verdict(
        self,
        entity_id: str,
        dossier: EntityDossier,
    ) -> tuple[AgentVerdict | None, float, DetectionMetadata]:
        """Return (AgentVerdict | None, complexity_score, DetectionMetadata).

        The Switch:
        - Both silent  → verdict=0 at zero LLM cost.
        - Any flagged  → None (escalate) with DetectionMetadata for L1.
        """
        if not self._is_fitted:
            return None, 1.0, DetectionMetadata()

        meta = self._build_detection_metadata(entity_id, dossier)

        if not meta.is_anomalous:
            # Both detectors silent → return 0 instantly (100% token saving)
            return AgentVerdict(
                agent_name="L0_HybridRouter",
                prediction=0,
                confidence=meta.confidence,
                reasoning=meta.report,
            ), 0.0, meta

        # Anomaly detected → escalate to L1 with explainability metadata
        complexity = min(1.0, meta.confidence)
        logger.info(
            "  L0[%s] ESCALATING %s: type=%s, confidence=%.2f, report=%s",
            entity_id, meta.detection_type, meta.detection_type,
            meta.confidence, meta.report[:120],
        )
        return None, complexity, meta

    def get_complexity(self, entity_id: str, dossier: EntityDossier) -> float:
        """Return a complexity score in [0.0, 1.0]."""
        if not self._is_fitted:
            return 1.0
        _, complexity, _ = self.to_verdict(entity_id, dossier)
        return complexity
