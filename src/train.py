"""Train the fraud-detection models and persist artifacts.

Methodology
-----------
* Stratified 70/15/15 train/validation/test split (fixed seed).
* A logistic-regression baseline (``class_weight="balanced"``) and an
  XGBoost classifier (``scale_pos_weight`` set to the class ratio) share the
  same preprocessing pipeline.
* The operating threshold is chosen on the **validation** split by
  maximizing F1, then frozen and only *reported* on the held-out test split.
* Artifacts: ``models/model.joblib`` (fitted pipeline + threshold, gitignored)
  and ``reports/metrics.json`` (committed, so the README numbers are
  auditable).

Usage: ``python -m src.train [--data PATH]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import load_data
from src.evaluate import compute_metrics, find_best_f1_threshold, format_report
from src.features import build_preprocessor, split_xy

RANDOM_STATE = 42
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "model.joblib"
METRICS_PATH = REPO_ROOT / "reports" / "metrics.json"


def build_models(y_train) -> dict[str, Pipeline]:
    """Build the baseline and XGBoost pipelines (unfitted)."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    return {
        "baseline_logreg": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        scale_pos_weight=scale_pos_weight,
                        tree_method="hist",
                        eval_metric="aucpr",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_and_save(data_path: str | Path | None = None) -> dict:
    """Train both models, tune the XGB threshold on validation, save artifacts."""
    df = load_data(data_path)
    X, y = split_xy(df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    models = build_models(y_train)
    results: dict = {
        "dataset": {
            "rows": int(len(df)),
            "fraud": int(y.sum()),
            "fraud_rate": round(float(y.mean()), 5),
            "split": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        }
    }

    chosen_threshold = 0.5
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        entry = {
            "validation": compute_metrics(y_val, val_prob, 0.5),
            "test": compute_metrics(y_test, test_prob, 0.5),
        }
        print(format_report(f"{name} (validation)", entry["validation"]))
        print(format_report(f"{name} (test)", entry["test"]))
        print()
        if name == "xgboost":
            chosen_threshold, best_val_f1 = find_best_f1_threshold(y_val, val_prob)
            entry["chosen_threshold"] = round(float(chosen_threshold), 4)
            entry["validation_best_f1"] = round(float(best_val_f1), 4)
            entry["test_at_validation_threshold"] = compute_metrics(
                y_test, test_prob, chosen_threshold
            )
            print(
                format_report(
                    "xgboost @ validation-chosen threshold (test)",
                    entry["test_at_validation_threshold"],
                )
            )
            print()
        results[name] = entry

    bundle = {
        "pipeline": models["xgboost"],
        "threshold": float(chosen_threshold),
        "feature_columns": list(X.columns),
        "model_name": "xgboost",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle -> {MODEL_PATH} (threshold={chosen_threshold:.4f})")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics -> {METRICS_PATH}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud-detection models.")
    parser.add_argument("--data", default=None, help="Path to creditcard.csv")
    args = parser.parse_args()
    train_and_save(args.data)
