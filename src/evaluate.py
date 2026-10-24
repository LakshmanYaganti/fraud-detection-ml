"""Evaluation utilities for highly imbalanced fraud classification.

At a 0.17% fraud rate, accuracy is meaningless (predicting "legitimate" for
every transaction scores 99.83%). This module therefore centers on the
precision/recall trade-off: PR-AUC, and metrics at an explicitly chosen
decision threshold rather than the default 0.5.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Compute threshold-dependent and threshold-free metrics."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def find_best_f1_threshold(y_true, y_prob) -> tuple[float, float]:
    """Return the threshold (and its F1) maximizing F1 on the PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
    # precision_recall_curve returns one more precision/recall than thresholds.
    best_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def find_threshold_for_recall(y_true, y_prob, target_recall: float = 0.90) -> tuple[float, float]:
    """Return the highest-precision threshold reaching ``target_recall``.

    Also returns the precision achieved at that threshold. If the target
    recall is unreachable, returns the lowest threshold on the curve.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    candidates = [
        (float(thresholds[i]), float(precisions[i]))
        for i in range(len(thresholds))
        if recalls[i] >= target_recall
    ]
    if not candidates:
        return float(thresholds[0]), float(precisions[0])
    # Highest threshold that still meets the recall target => best precision.
    return max(candidates, key=lambda t: t[0])


def format_report(name: str, metrics: dict) -> str:
    cm = metrics["confusion_matrix"]
    return (
        f"{name}\n"
        f"  threshold : {metrics['threshold']}\n"
        f"  precision : {metrics['precision']:.4f}\n"
        f"  recall    : {metrics['recall']:.4f}\n"
        f"  F1        : {metrics['f1']:.4f}\n"
        f"  ROC-AUC   : {metrics['roc_auc']:.4f}\n"
        f"  PR-AUC    : {metrics['pr_auc']:.4f}\n"
        f"  confusion : TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}"
    )


if __name__ == "__main__":
    import json
    from pathlib import Path

    metrics_path = Path(__file__).resolve().parent.parent / "reports" / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"No metrics file at {metrics_path}. Run `python -m src.train` first.")
    saved = json.loads(metrics_path.read_text())
    for section in ("baseline_logreg", "xgboost"):
        print(format_report(section, saved[section]["test"]))
        print()
    tuned = saved["xgboost"].get("test_at_validation_threshold")
    if tuned:
        print(format_report("xgboost @ validation-chosen threshold", tuned))
