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
        self._fitted = False

        # Population stats for explainability
        self._population_means: dict[str, float] = {}
        self._population_stds: dict[str, float] = {}

        logger.info("OneClassRouter initialised (IsolationForest engine)")

    @property
    def is_fitted(self) -> bool:
        """Returns True if the model has been fitted."""
        return self._fitted

    # ── Feature Extraction ──────────────────────────────────────────────

    @staticmethod
    def _dossier_to_feature_vector(dossier: EntityDossier) -> dict[str, float]:
        """Extract the feature dict for one-class fitting/prediction."""
        return dossier.features or {}

    # ── Fit (Training = Class 0 only) ───────────────────────────────────

    def reset(self) -> None:
        """Resets the internal state of the router.
        
        Clears the fitted IsolationForest model, the StandardScaler, and 
        all population statistics. This is used during multi-pass 
        distillation to ensure the second pass starts from a clean state.
        """
        self._scaler = None
        self._iso = None
        self._feature_names = []
        self._fitted = False
        self._population_means = {}
        self._population_stds = {}
        logger.info("OneClassRouter internal state reset.")

    def build_baselines(self, train_dossiers: dict[str, EntityDossier]) -> None:
        """Fits the IsolationForest engine on normal well-being clusters.

        This method extracts feature vectors from the provided dossiers, 
        calculates population statistics for interpretability, and fits 
        an IsolationForest to learn the boundary of 'normal' behaviour.

        Args:
            train_dossiers (dict[str, EntityDossier]): The training set of 
                dossiers assumed to represent normal Class 0 behavior.
        """
        assert not self._fitted, "Layer 0 already fitted for this stage! Reset or re-instantiate requested."
        
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
            contamination="auto",
            random_state=42,
        )
        self._iso.fit(X_scaled)

        self._fitted = True
        logger.info(
            "INFO: Layer 0 successfully FITTED on Training Set (%d samples, %d features)",
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

        - Secure Inlier (0) → verdict=0 at zero LLM cost.
        - Outlier or Uncertain → None (escalate to L1) with math details.
        """
        assert self._fitted, "Layer 0 MUST be fitted before prediction!"

        meta = self._build_metadata(entity_id, dossier)
        cfg = get_settings()

        # Rule: Anomalies ALWAYS escalate to L1/L2.
        # Only Non-Anomalous (0) cases with high confidence can skip.
        if not meta.is_anomalous and meta.confidence > cfg.l0_upper_threshold:
            return AgentVerdict(
                agent_name="L0_OneClassRouter",
                prediction=0,
                confidence=meta.confidence,
                reasoning=meta.report,
            ), 0.0, meta

        # Otherwise, force escalation (Outlier OR uncertain Inlier)
        complexity = min(1.0, meta.confidence) if meta.is_anomalous else 0.5
        logger.info(
            "  L0[%s] %s (conf=%.2f) → escalating to L1",
            entity_id, "OUTLIER" if meta.is_anomalous else "UNCERTAIN_INLIER",
            meta.confidence
        )
        return None, complexity, meta

    def get_complexity(self, entity_id: str, dossier: EntityDossier) -> float:
        """Return a complexity score in [0.0, 1.0]."""
        assert self._fitted, "Layer 0 MUST be fitted before complexity assessment!"
        _, complexity, _ = self.to_verdict(entity_id, dossier)
        return complexity
