"""
Layer 0 — Deterministic Router (Efficiency Engine).
Trains an XGBoost or RandomForest model and routes entities based on
confidence thresholds, emitting verdicts at zero LLM cost when
the model is confident enough.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models import AgentVerdict
from settings import get_settings

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier  # type: ignore[import-untyped]

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False
    logger.info("XGBoost not installed — will fall back to RandomForest")


class DeterministicRouter:
    """ML-based router that emits fast verdicts for high-confidence cases.

    Workflow
    --------
    1. ``fit(X, y)`` trains the model on labelled data.
    2. ``predict_one(features)`` returns ``(verdict | None, confidence)``.
       *None* means "escalate to Layer 1".
    """

    def __init__(
        self,
        lower_threshold: float | None = None,
        upper_threshold: float | None = None,
    ) -> None:
        cfg = get_settings()
        self._lower = lower_threshold or cfg.l0_lower_threshold
        self._upper = upper_threshold or cfg.l0_upper_threshold
        self._model = self._build_model()
        self._is_fitted = False
        self._feature_names: list[str] = []

    # ── Training ────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray | list[dict[str, float]],
        y: list[int],
        feature_names: list[str] | None = None,
    ) -> None:
        """Train the classifier."""
        if isinstance(X, list):
            # Convert list-of-dicts to array
            if feature_names is None:
                feature_names = sorted(X[0].keys())
            self._feature_names = feature_names
            X_arr = np.array(
                [[d.get(f, 0.0) for f in feature_names] for d in X],
                dtype=float,
            )
        else:
            X_arr = np.asarray(X, dtype=float)
            self._feature_names = feature_names or [
                f"f{i}" for i in range(X_arr.shape[1])
            ]

        X_arr = np.nan_to_num(X_arr, nan=0.0)
        logger.info(
            "Layer 0 training on %d samples, %d features",
            X_arr.shape[0],
            X_arr.shape[1],
        )
        self._model.fit(X_arr, y)
        self._is_fitted = True

    # ── Prediction ──────────────────────────────────────────────────────

    def predict_one(
        self, features: dict[str, float]
    ) -> tuple[Optional[int], float]:
        """Predict a single entity.

        Returns
        -------
        (verdict, confidence)
            verdict is 0 or 1 when confidence is outside thresholds,
            or None when the model is uncertain (escalate).
        """
        if not self._is_fitted:
            return None, 0.5  # escalate everything if untrained

        x = np.array(
            [[features.get(f, 0.0) for f in self._feature_names]],
            dtype=float,
        )
        x = np.nan_to_num(x, nan=0.0)
        proba = self._model.predict_proba(x)[0]  # [p_class0, p_class1]
        conf_positive = float(proba[1]) if len(proba) > 1 else float(proba[0])

        if conf_positive >= self._upper:
            return 1, conf_positive
        if conf_positive <= self._lower:
            return 0, 1.0 - conf_positive
        return None, conf_positive  # uncertain → escalate

    def predict_batch(
        self, feature_dicts: list[dict[str, float]]
    ) -> list[tuple[Optional[int], float]]:
        """Predict a batch of entities."""
        return [self.predict_one(fd) for fd in feature_dicts]

    def to_verdict(
        self,
        entity_id: str,
        features: dict[str, float],
    ) -> tuple[AgentVerdict | None, float]:
        """Return (AgentVerdict | None, complexity_score).

        If confident, returns (verdict, 0.0).
        If uncertain (escalate), returns (None, complexity_score).
        """
        verdict, conf = self.predict_one(features)
        complexity = self.get_complexity(features)
        if verdict is not None:
            return AgentVerdict(
                agent_name="L0_DeterministicRouter",
                prediction=verdict,
                confidence=conf,
                reasoning=(
                    f"ML model confidence {conf:.3f} "
                    f"{'above' if verdict == 1 else 'below'} thresholds "
                    f"[{self._lower}, {self._upper}]"
                ),
            ), 0.0
        return None, complexity

    def get_complexity(self, features: dict[str, float]) -> float:
        """Return a complexity score in [0.0, 1.0].

        1 - 2 * |proba - 0.5| maps probabilities to complexity:
          proba=0.5  → complexity=1.0  (maximally uncertain)
          proba=0.0  → complexity=0.0  (fully certain class 0)
          proba=1.0  → complexity=0.0  (fully certain class 1)

        If the model is not fitted, returns 1.0 (maximum complexity).
        """
        if not self._is_fitted:
            return 1.0
        _, conf_positive = self.predict_one(features)
        return 1.0 - 2.0 * abs(conf_positive - 0.5)

    # ── Private ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_model():
        if _HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        )
