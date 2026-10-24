"""Load and validate the Kaggle Credit Card Fraud dataset.

The dataset (284,807 European card transactions over two days in September
2013, of which 492 / 0.173% are fraud) is NOT committed to this repository
because of its size and Kaggle licensing. ``load_data`` resolves it in this
order:

1. an explicit ``path`` argument,
2. the ``FRAUD_DATA_PATH`` environment variable,
3. ``data/creditcard.csv`` inside this repository,
4. otherwise it downloads the file from the TensorFlow mirror into
   ``data/creditcard.csv`` (see ``data/README.md`` for provenance).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd

MIRROR_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "creditcard.csv"

TARGET_COLUMN = "Class"
EXPECTED_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", TARGET_COLUMN]
)


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ``ValueError`` unless ``df`` has exactly the expected schema."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        raise ValueError(f"Dataset has unexpected columns: {extra}")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column 'Class' contains null values")
    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target column 'Class' must be binary (0/1)")
    return df


def download_data(dest: Path = DEFAULT_DATA_PATH) -> Path:
    """Download the dataset from the TensorFlow mirror to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {MIRROR_URL} -> {dest} (~150 MB)")
    urllib.request.urlretrieve(MIRROR_URL, dest)
    return dest


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the credit-card fraud dataset and validate its schema."""
    if path is None:
        env_path = os.environ.get("FRAUD_DATA_PATH")
        path = Path(env_path) if env_path else DEFAULT_DATA_PATH
    path = Path(path)
    if not path.exists():
        path = download_data(path)
    df = pd.read_csv(path)
    return validate_columns(df)


if __name__ == "__main__":
    data = load_data()
    n_fraud = int(data[TARGET_COLUMN].sum())
    print(f"Rows: {len(data):,}  Fraud: {n_fraud:,} ({n_fraud / len(data):.3%})")
    print(f"Columns: {list(data.columns)}")
