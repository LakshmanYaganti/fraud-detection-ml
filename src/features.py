"""Feature definitions and preprocessing for the fraud dataset.

The V1-V28 columns are already PCA-transformed and roughly standardized by
the dataset creators, so they pass through untouched. ``Time`` (seconds since
the first transaction) and ``Amount`` are on raw, heavy-tailed scales, so
they are robust-scaled (median / IQR), which is resilient to the extreme
transaction amounts present in the data.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from src.data import TARGET_COLUMN

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
SCALED_FEATURES = ["Time", "Amount"]
FEATURE_COLUMNS = ["Time"] + PCA_FEATURES + ["Amount"]


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted preprocessing transformer for the raw features.

    The transformer blocks are ordered so the output columns line up exactly
    with ``FEATURE_COLUMNS`` (Time, V1-V28, Amount). Downstream code (e.g.
    the SHAP analysis in the notebook) relies on that alignment when it
    labels transformed data with ``FEATURE_COLUMNS``.
    """
    return ColumnTransformer(
        transformers=[
            ("scale_time", RobustScaler(), ["Time"]),
            ("pca", "passthrough", PCA_FEATURES),
            ("scale_amount", RobustScaler(), ["Amount"]),
        ],
        remainder="drop",
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a validated dataframe into feature matrix ``X`` and target ``y``."""
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].astype(int).copy()
