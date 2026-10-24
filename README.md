# Fraud Detection ML — XGBoost on Highly Imbalanced Payment Data

**Catching the 492 fraudulent transactions hiding in 284,807 — where a model that says "legitimate" every time is already 99.83% accurate.**

An end-to-end, production-shaped fraud-detection project: exploratory analysis,
a logistic-regression baseline vs. an XGBoost classifier, precision/recall-driven
threshold tuning on validation data, SHAP explanations, a FastAPI scoring
service, and a full test suite. Built by a data scientist who works on payments
fraud and credit risk for a living.

---

## The problem

Card fraud is rare, expensive, and adversarial. In this dataset only
**0.173%** of transactions are fraudulent, which breaks the usual ML playbook:

- **Accuracy is meaningless.** Predicting "legitimate" for every transaction
  scores 99.83% accuracy and stops zero fraud.
- **ROC-AUC flatters.** With ~42,600 legitimate transactions per test split,
  hundreds of false alarms barely dent the false-positive rate — the baseline
  below posts a 0.968 ROC-AUC while drowning analysts in false positives.
- **What matters is the precision/recall trade-off** (summarized by PR-AUC)
  and the **decision threshold**: every flagged transaction costs analyst time;
  every missed fraud costs real money. The threshold is a business decision,
  so this project tunes it on validation data and freezes it before testing.

## Dataset

| | |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Worldline / MLG-ULB research data) |
| Contents | 284,807 European card transactions over two days in Sep 2013 |
| Fraud | 492 transactions (0.173%) |
| Features | `Time`, `Amount`, and `V1`–`V28` (PCA-anonymized) |
| Target | `Class` (1 = fraud) |

The ~150 MB CSV is **not committed** (size + Kaggle licensing). See
[`data/README.md`](data/README.md) for provenance and download steps — the
loader in `src/data.py` can fetch it automatically from a public mirror.

## Approach

1. **Preprocessing** (`src/features.py`) — `Time` and `Amount` are
   robust-scaled (median/IQR, resilient to amount outliers); the PCA features
   pass through already standardized.
2. **Split** — stratified 70/15/15 train/validation/test. Validation exists
   for one reason: choosing the threshold. Test is reported once, at the end.
3. **Baseline** — logistic regression with `class_weight="balanced"`.
4. **Main model** — XGBoost (300 trees, depth 6) with `scale_pos_weight`
   set to the train class ratio (~578:1) so the loss actually notices fraud.
5. **Threshold tuning** (`src/evaluate.py`) — pick the threshold maximizing
   F1 on validation; also computed: the threshold needed for ~90% recall.
6. **Serving** (`api/app.py`) — FastAPI scores a transaction and applies the
   frozen, validation-chosen threshold.

## Results

Held-out **test set** (42,722 transactions, 74 fraud). Full machine-readable
metrics, including validation numbers: [`reports/metrics.json`](reports/metrics.json).

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.50 | 0.067 | 0.878 | 0.124 | 0.968 | 0.792 |
| **XGBoost** | **0.50** | **0.896** | **0.811** | **0.851** | **0.969** | **0.840** |
| XGBoost @ validation-tuned threshold | 0.292 | 0.857 | 0.811 | 0.833 | 0.969 | 0.840 |

Confusion matrices on test:

| Model | TP | FP | FN | TN |
|---|---|---|---|---|
| Logistic Regression @ 0.50 | 65 | 907 | 9 | 41,741 |
| XGBoost @ 0.50 | 60 | 7 | 14 | 42,641 |
| XGBoost @ 0.292 (tuned) | 60 | 10 | 14 | 42,638 |

## Key findings

- **The baseline is a cautionary tale.** Logistic regression catches 88% of
  fraud — at 6.7% precision. That's **907 false alarms for 65 catches**
  (~14 false alarms per real fraud). No fraud-ops team can work that queue.
- **XGBoost changes the game on precision, not recall.** At the default
  threshold it catches 60 of 74 frauds with only **7 false alarms** —
  precision 0.896 vs. the baseline's 0.067, a ~13× improvement at
  comparable recall.
- **Threshold tuning is not free F1 — and this run proves it.** The
  validation-optimal threshold (0.292, validation F1 0.841) scored F1 0.833
  on test, slightly *below* the default 0.5 (0.851). With only ~74 frauds
  per split, the tuned operating point is itself a noisy estimate and
  doesn't always generalize. In production you would set the threshold from
  the business cost ratio (a missed fraud vs. an analyst review), re-estimate
  it on far more data, and monitor it — the notebook (§6) also computes the
  ~90%-recall operating point to make that trade-off explicit.
- **ROC-AUC hides all of this** — it calls the two models a tie
  (0.969 vs. 0.968) while their false-alarm counts differ by two orders of
  magnitude (7 vs. 907). PR-AUC (0.840 vs. 0.792) and the confusion matrix
  tell the real story.
- **SHAP** (notebook §7) shows a handful of PCA components — **V14, V4,
  V12, V10**, by mean |SHAP| — dominate the fraud score, which matches the
  strongest target correlations found in EDA.

## Project structure

```
fraud-detection-ml/
├── src/
│   ├── data.py        # dataset loading (auto-download) + schema validation
│   ├── features.py    # feature list + preprocessing (RobustScaler on Time/Amount)
│   ├── train.py       # train baseline + XGBoost, tune threshold, save artifacts
│   ├── evaluate.py    # PR-AUC / precision / recall / threshold analysis
│   └── predict.py     # score a single transaction from the saved bundle
├── api/
│   └── app.py         # FastAPI: GET /health, POST /score
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # executed EDA + modeling walkthrough
├── tests/
│   └── test_pipeline.py            # preprocessing, prediction, and API tests
├── data/README.md     # dataset provenance + download instructions
├── reports/metrics.json            # full metrics from the reference run
├── requirements.txt
└── LICENSE            # MIT
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Train (dataset downloads automatically on first run, ~150 MB)
python -m src.train

# 2. Score transactions via the API
uvicorn api.app:app --reload
curl http://localhost:8000/health
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" \
  -d '{"Time": 406, "V1": -2.31, "V2": 1.95, "...": "...", "Amount": 149.62}'
# -> {"fraud_probability": ..., "is_fraud": ..., "threshold": 0.2916, "model": "xgboost"}
# (If no trained bundle exists, the API trains one on the fly at first /score.)

# 3. Tests
pytest

# 4. Notebook
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

## Tech stack

Python · XGBoost · scikit-learn · pandas · SHAP · FastAPI · matplotlib/seaborn · pytest

## License

MIT — see [`LICENSE`](LICENSE). The dataset itself remains under Kaggle's /
the original providers' terms and is not redistributed here.

## Author

**Lakshman Sai Yaganti** — Data Scientist @ Square (payments fraud & credit risk)
[GitHub](https://github.com/LakshmanYaganti) · [LinkedIn](https://www.linkedin.com/in/y-lakshman/)
