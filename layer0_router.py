"""
Layer 0 — One-Class Anomaly Engine (IsolationForest).

Architecture:
- **Fit phase:** Trains ONLY on class-0 (well-being) training data,
  learning the decision boundary of "normal behaviour".
- **Predict phase:** New entities are classified as:
  - Inlier (inside boundary) → emit pred=0 at zero LLM cost.
  - Outlier (outside boundary) → escalate to L1 with math details.

L1 agents act as Anti-False-Positive Filters: they read the context
(personas, profile, users) and decide whether the L0 anomaly is
justified by life context. If justified → 0. If not → confirm 1.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from models import AgentVerdict, DetectionMetadata, EntityDossier
from settings import get_settings

logger = logging.getLogger(__name__)


class OneClassRouter:
    """One-Class Anomaly Engine: IsolationForest.

    Workflow
    --------
    1. ``build_baselines(train_dossiers)``
       - Extract feature vectors from ALL training entities (class 0).
       - Fit StandardScaler → IsolationForest.
    2. ``to_verdict(entity_id, dossier)``
       - Extract same features for the eval entity.
       - IsolationForest: inlier → emit 0; outlier → escalate with details.
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._iso: IsolationForest | None = None
        self._feature_names: list[str] = []
        self._is_fitted = False

        # Population stats for explainability
        self._population_means: dict[str, float] = {}
        self._population_stds: dict[str, float] = {}

        logger.info("OneClassRouter initialised (IsolationForest engine)")

    # ── Feature Extraction ──────────────────────────────────────────────

    @staticmethod
    def _dossier_to_feature_vector(dossier: EntityDossier) -> dict[str, float]:
        """Extract the feature dict for one-class fitting/prediction."""
        return dossier.features or {}

    # ── Fit (Training = Class 0 only) ───────────────────────────────────

    def build_baselines(self, train_dossiers: dict[str, EntityDossier]) -> None:
        """Fit IsolationForest on class-0 training data."""
        feature_dicts: list[dict[str, float]] = []
        for dossier in train_dossiers.values():
            fv = self._dossier_to_feature_vector(dossier)
            if fv:
                feature_dicts.append(fv)

        if len(feature_dicts) < 4:
            logger.warning(
                "L0: only %d training samples — too few for IsolationForest fit",
                len(feature_dicts),
            )
            return

        # Build consistent feature matrix
        all_keys = sorted(set().union(*(d.keys() for d in feature_dicts)))
        self._feature_names = all_keys

        X = np.array(
            [[d.get(k, 0.0) for k in all_keys] for d in feature_dicts],
            dtype=float,
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Store population statistics for explainability
        for i, name in enumerate(all_keys):
            col = X[:, i]
            self._population_means[name] = float(np.mean(col))
            self._population_stds[name] = float(np.std(col))

        # Scale
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Fit IsolationForest
        self._iso = IsolationForest(
            n_estimators=100,
            contamination=get_settings().l0_contamination,
            random_state=42,
        )
        self._iso.fit(X_scaled)

        self._is_fitted = True
        logger.info(
            "L0 IsolationForest fitted on %d class-0 samples (%d features)",
            len(X), len(all_keys),
        )

    # ── Predict ─────────────────────────────────────────────────────────

    def _predict(self, dossier: EntityDossier) -> tuple[bool, float, dict[str, float]]:
        """Run IsolationForest on a single entity.

        Returns (is_outlier, if_score, deviating_features).
        """
        fv = self._dossier_to_feature_vector(dossier)
        x = np.array(
            [[fv.get(k, 0.0) for k in self._feature_names]], dtype=float,
        )
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_scaled = self._scaler.transform(x)

        # IsolationForest: +1 = inlier, -1 = outlier
        iso_pred = int(self._iso.predict(x_scaled)[0])
        iso_score = float(self._iso.decision_function(x_scaled)[0])
        is_outlier = iso_pred == -1

        # Find which features deviate most from population
        deviating: dict[str, float] = {}
        for i, name in enumerate(self._feature_names):
            pop_std = self._population_stds.get(name, 1e-6)
            if pop_std < 1e-6:
                continue
            z = abs(fv.get(name, 0.0) - self._population_means.get(name, 0.0)) / pop_std
            if z > 2.0:
                deviating[name] = round(z, 2)

        return is_outlier, iso_score, deviating

    # ── Build DetectionMetadata ─────────────────────────────────────────

    def _build_metadata(
        self, entity_id: str, dossier: EntityDossier,
    ) -> DetectionMetadata:
        """Run IsolationForest and produce structured metadata."""
        is_outlier, iso_score, deviating = self._predict(dossier)

        confidence = min(abs(iso_score) * 3, 1.0)

        # Build human-readable report for L1
        report_parts: list[str] = []
        if is_outlier:
            if deviating:
                top_devs = sorted(deviating.items(), key=lambda x: -x[1])[:5]
                sensors = ", ".join(f"{k} (z={v:.1f}σ)" for k, v in top_devs)
                report_parts.append(f"Anomalia rilevata su: {sensors}")
            report_parts.append(f"IF_score={iso_score:.4f}")
        else:
            report_parts.append("Nessuna anomalia — profilo nella norma.")

        return DetectionMetadata(
            is_anomalous=is_outlier,
            confidence=confidence if is_outlier else 1.0 - confidence,
            forest_flagged=is_outlier,
            detection_type="behavioural" if is_outlier else "none",
            deviating_features=deviating,
            forest_score=iso_score,
            report=" | ".join(report_parts),
        )

    # ── Public API ──────────────────────────────────────────────────────

    def to_verdict(
        self, entity_id: str, dossier: EntityDossier,
    ) -> tuple[AgentVerdict | None, float, DetectionMetadata]:
        """Return (AgentVerdict | None, complexity, DetectionMetadata).

        - Inlier  → verdict=0 at zero LLM cost.
        - Outlier → None (escalate to L1) with math details.
        """
        if not self._is_fitted:
            return None, 1.0, DetectionMetadata()

        meta = self._build_metadata(entity_id, dossier)

        if not meta.is_anomalous:
            return AgentVerdict(
                agent_name="L0_OneClassRouter",
                prediction=0,
                confidence=meta.confidence,
                reasoning=meta.report,
            ), 0.0, meta

        complexity = min(1.0, meta.confidence)
        logger.info(
            "  L0[%s] OUTLIER → escalating to L1: conf=%.2f, %s",
            entity_id, meta.confidence, meta.report[:150],
        )
        return None, complexity, meta

    def get_complexity(self, entity_id: str, dossier: EntityDossier) -> float:
        """Return a complexity score in [0.0, 1.0]."""
        if not self._is_fitted:
            return 1.0
        _, complexity, _ = self.to_verdict(entity_id, dossier)
        return complexity
