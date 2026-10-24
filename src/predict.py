"""Score a single transaction with the trained model bundle.

The bundle (``models/model.joblib``) is produced by ``python -m src.train``
and contains the fitted preprocessing + XGBoost pipeline, the decision
threshold chosen on validation data, and the expected feature order.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"


@lru_cache(maxsize=1)
def load_bundle(model_path: str | Path = MODEL_PATH) -> dict:
    """Load (and cache) the model bundle saved by ``src.train``."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def score_transaction(transaction: dict, bundle: dict | None = None) -> dict:
    """Score one transaction.

    ``transaction`` must provide every feature in ``FEATURE_COLUMNS``
    (``Time``, ``V1``-``V28``, ``Amount``). Returns the fraud probability,
    the binary decision at the bundle's threshold, and the threshold used.
    """
    if bundle is None:
        bundle = load_bundle()
    missing = [c for c in FEATURE_COLUMNS if c not in transaction]
    if missing:
        raise ValueError(f"Transaction is missing features: {missing}")
    row = pd.DataFrame(
        [{c: float(transaction[c]) for c in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS
    )
    probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
    threshold = float(bundle["threshold"])
    return {
        "fraud_probability": round(probability, 6),
        "is_fraud": bool(probability >= threshold),
        "threshold": threshold,
        "model": bundle.get("model_name", "xgboost"),
    }


if __name__ == "__main__":
    import json

    example = {c: 0.0 for c in FEATURE_COLUMNS}
    example["Amount"] = 149.62
    print(json.dumps(score_transaction(example), indent=2))
