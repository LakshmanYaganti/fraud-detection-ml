"""Tests for the fraud-detection pipeline: features, training, prediction, API.

Everything runs on a tiny synthetic dataset shaped like the real one
(same columns, similar imbalance) so the suite is fast and needs no
150 MB download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import EXPECTED_COLUMNS, validate_columns
from src.evaluate import compute_metrics, find_threshold_for_recall
from src.features import FEATURE_COLUMNS, build_preprocessor, split_xy
from src.predict import score_transaction


def make_synthetic_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Build a small dataframe with the real dataset's schema."""
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n,
        n_features=28,
        n_informative=10,
        n_redundant=5,
        weights=[0.98, 0.02],
        random_state=seed,
    )
    data = {f"V{i}": X[:, i - 1] for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172800, size=n)
    data["Amount"] = np.abs(rng.normal(80, 60, size=n)).round(2)
    data["Class"] = y
    return pd.DataFrame(data)[EXPECTED_COLUMNS]


def train_tiny_bundle(df: pd.DataFrame) -> dict:
    """Fit a small XGBoost pipeline on the synthetic frame."""
    X, y = split_xy(df)
    pipeline = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=20,
                    max_depth=3,
                    tree_method="hist",
                    eval_metric="aucpr",
                    random_state=0,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return {
        "pipeline": pipeline,
        "threshold": 0.5,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": "xgboost-test",
    }


@pytest.fixture(scope="module")
def tiny_bundle() -> dict:
    return train_tiny_bundle(make_synthetic_frame())


# --------------------------------------------------------------------------- #
# Data validation & features
# --------------------------------------------------------------------------- #


def test_validate_columns_accepts_valid_frame():
    df = make_synthetic_frame(n=20)
    assert validate_columns(df) is df


def test_validate_columns_rejects_missing_column():
    df = make_synthetic_frame(n=20).drop(columns=["V7"])
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_columns(df)


def test_preprocessor_output_shape():
    df = make_synthetic_frame(n=50)
    X, y = split_xy(df)
    transformed = build_preprocessor().fit_transform(X)
    assert transformed.shape == (50, len(FEATURE_COLUMNS))
    assert set(np.unique(y)) == {0, 1}


def test_preprocessor_preserves_feature_order():
    """Transformed columns must align with FEATURE_COLUMNS.

    The notebook labels transformed data with FEATURE_COLUMNS for its SHAP
    analysis; if the transformer reordered columns, those labels would lie.
    """
    df = make_synthetic_frame(n=30)
    X, _ = split_xy(df)
    names = build_preprocessor().fit(X).get_feature_names_out()
    suffixes = [n.split("__", 1)[-1] for n in names]
    assert suffixes == FEATURE_COLUMNS


def test_split_xy_rejects_missing_target():
    df = make_synthetic_frame(n=20).drop(columns=["Class"])
    with pytest.raises(ValueError, match="missing columns"):
        split_xy(df)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.2, 0.9, 0.99])
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_find_threshold_for_recall():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.4, 0.6, 0.9])
    threshold, precision = find_threshold_for_recall(y_true, y_prob, target_recall=1.0)
    assert threshold <= 0.6  # must be low enough to catch both frauds
    assert precision > 0


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


def test_score_transaction_shape(tiny_bundle):
    transaction = {c: 0.0 for c in FEATURE_COLUMNS}
    transaction["Amount"] = 120.50
    result = score_transaction(transaction, bundle=tiny_bundle)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert isinstance(result["is_fraud"], bool)
    assert result["threshold"] == 0.5
    assert result["model"] == "xgboost-test"


def test_score_transaction_missing_feature(tiny_bundle):
    transaction = {c: 0.0 for c in FEATURE_COLUMNS if c != "V3"}
    with pytest.raises(ValueError, match="missing features"):
        score_transaction(transaction, bundle=tiny_bundle)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(tiny_bundle, monkeypatch):
    from fastapi.testclient import TestClient

    import api.app as app_module

    monkeypatch.setattr(app_module, "get_bundle", lambda: tiny_bundle)
    monkeypatch.setattr(app_module, "_bundle", tiny_bundle)
    return TestClient(app_module.app)


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_score(client):
    payload = {c: 0.0 for c in FEATURE_COLUMNS}
    payload["Amount"] = 250.0
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "fraud_probability" in body and "is_fraud" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_api_score_missing_feature_is_422(client):
    payload = {c: 0.0 for c in FEATURE_COLUMNS if c != "Amount"}
    response = client.post("/score", json=payload)
    assert response.status_code == 422
