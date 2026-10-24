"""FastAPI scoring API for the fraud-detection model.

Endpoints
---------
GET  /health  -> service status and whether the model bundle is available
POST /score   -> fraud probability + decision for one transaction

The request body for /score is a JSON object with the transaction features
``Time``, ``V1``-``V28`` and ``Amount``. The response contains the fraud
probability and the binary decision at the threshold that was chosen on
validation data during training.

Model lifecycle: the bundle saved by ``python -m src.train``
(``models/model.joblib``) is loaded lazily on the first /score call. If it
does not exist yet, the API trains the model on the fly (the dataset loader
downloads the data if needed) so the service works from a fresh clone:

    uvicorn api.app:app --reload     # from the repository root
"""

from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException

from src.features import FEATURE_COLUMNS
from src.predict import MODEL_PATH, load_bundle, score_transaction

app = FastAPI(title="Fraud Detection Scoring API", version="1.0.0")

_bundle: dict | None = None


def get_bundle() -> dict:
    """Return the cached model bundle, training on the fly if necessary."""
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            from src.train import train_and_save

            train_and_save()
        _bundle = load_bundle()
    return _bundle


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_available": MODEL_PATH.exists() or _bundle is not None}


@app.post("/score")
def score(payload: dict = Body(...)) -> dict:
    missing = [c for c in FEATURE_COLUMNS if c not in payload]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"Missing transaction features: {missing}"
        )
    try:
        return score_transaction(payload, bundle=get_bundle())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
