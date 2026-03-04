"""
Metrics and reporting utilities.
Computes Confusion Matrix, Precision, Recall, F1,
Value Recovery, and the Reply Mirror composite score.
"""

from __future__ import annotations

import logging

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from settings import get_settings

logger = logging.getLogger(__name__)
console = Console()


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute classification metrics.

    Returns
    -------
    dict with keys: tp, fp, fn, tn, precision, recall, f1
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    prec = precision_score(y_true, y_pred, zero_division=0.0)
    rec = recall_score(y_true, y_pred, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def compute_value_recovery(
    y_true: list[int],
    y_pred: list[int],
    fp_cost: float | None = None,
    fn_cost: float | None = None,
) -> float:
    """Simulate Value Recovery.

    VR = 1 - (total_loss / max_possible_loss)
    where total_loss = FP * fp_cost + FN * fn_cost
    and   max_possible_loss = n_positive * fn_cost  (all positives missed)
    """
    cfg = get_settings()
    fp_cost = fp_cost or cfg.fp_cost
    fn_cost = fn_cost or cfg.fn_cost

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    total_loss = fp * fp_cost + fn * fn_cost
    n_positive = sum(1 for y in y_true if y == 1)
    max_loss = n_positive * fn_cost if n_positive > 0 else 1.0

    vr = 1.0 - (total_loss / max_loss) if max_loss > 0 else 1.0
    return max(0.0, vr)


def compute_reply_mirror_score(f1: float, vr: float) -> float:
    """Composite Reply Mirror score: (F1 + VR) / 2."""
    return (f1 + vr) / 2.0


def print_report(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute and pretty-print a full evaluation report.

    Returns the metrics dict for programmatic use.
    """
    m = compute_metrics(y_true, y_pred)
    vr = compute_value_recovery(y_true, y_pred)
    score = compute_reply_mirror_score(m["f1"], vr)
    m["value_recovery"] = vr
    m["mirror_score"] = score

    # ── Confusion Matrix ────────────────────────────────────────────
    cm_table = Table(title="Confusion Matrix", show_lines=True)
    cm_table.add_column("", style="bold")
    cm_table.add_column("Pred = 0", justify="center")
    cm_table.add_column("Pred = 1", justify="center")
    cm_table.add_row("True = 0", f"{m['tn']:.0f}", f"{m['fp']:.0f}")
    cm_table.add_row("True = 1", f"{m['fn']:.0f}", f"{m['tp']:.0f}")
    console.print(cm_table)

    # ── Metrics ─────────────────────────────────────────────────────
    metrics_table = Table(title="Evaluation Metrics", show_lines=True)
    metrics_table.add_column("Metric", style="bold cyan")
    metrics_table.add_column("Value", justify="right")
    for key in ("precision", "recall", "f1", "value_recovery", "mirror_score"):
        metrics_table.add_row(key.replace("_", " ").title(), f"{m[key]:.4f}")
    console.print(metrics_table)

    return m
