# Machine Failure Early-Warning System

Predicts the risk that a machine is close to failure from its recent sensor readings, and serves the result through a REST API (FastAPI, Docker).

A score is a prioritization aid for maintenance. It is never proof that a machine is safe or unsafe.

## Overview

- **Data:** NASA C-MAPSS **FD001** (simulated turbofan engines, run to failure).
- **Task:** binary classification. Label = 1 if remaining useful life (RUL) <= 30 cycles, else 0.
- **Type:** snapshot model ("failure risk detection"), not a forecaster.
- **Model:** LightGBM, trained on 100 engines, with 48 features per reading.
- **Output:** risk score, a band (Low / Watch / Alarm) and the last 3 scores.

## Method

| Step | Decision |
|---|---|
| Label | RUL = max_cycle - cycle per engine; 1 if RUL <= 30. The 30 is an assumption; 20 and 50 were not tested. |
| Split | By **engine** (GroupShuffleSplit 80/20, seed 42), never by row. GroupKFold(5) for cross-validation. |
| Columns | 8 constant or near-constant columns dropped (`op3, s1, s5, s6, s10, s16, s18, s19`); 16 kept. |
| Features | 16 raw + 16 rolling means + 16 rolling slopes (10-cycle window, history only) = 48. |
| Imbalance | About 15% positives. Handled with class weights (`is_unbalance`). No resampling. |
| Metrics | Recall, precision, PR-AUC, confusion matrix. Accuracy is not used. |

The same `add_features` function (in `features.py`) runs in training and in the API, so features cannot drift between the two.

## Results

**Cross-validation** (GroupKFold, 5 folds, 100 engines, PR-AUC):

| Model | Mean | Std |
|---|---|---|
| Logistic Regression | 0.949 | 0.005 |
| Random Forest | 0.955 | 0.010 |
| **LightGBM (chosen)** | **0.961** | 0.009 |

LightGBM's edge is small (about 0.01). Its settings were tuned on engines that also sit inside the CV folds, so the number is slightly optimistic.

**Final test on NASA's official test set** (model trained on all 100 engines, run once):

| Level | Result |
|---|---|
| Last cycle of each engine, PR-AUC | 0.947 |
| Engine alarm rule (3 in a row >= 0.88) | caught 16 of 25 failing engines, 1 false alarm among 75 healthy engines |

All 9 missed engines had RUL between 18 and 29, right at the label boundary.

## Risk bands

| Band | Rule |
|---|---|
| **Alarm** | the last 3 scores are all >= 0.88 |
| **Watch** | the latest score is >= 0.5 |
| **Low** | the latest score is < 0.5 |

The Alarm rule (0.88, 3 in a row) was chosen in cross-validation before the final test. The Watch level (0.5) was chosen **after** seeing the final test, so treat it as a post-hoc choice.

Scores are uncalibrated risk scores, not probabilities. They are over-confident between 0.1 and 0.9.

## API

`POST /predict`: send the most recent readings of **one engine**.

- Needs `cycle` plus the 16 kept columns: `op1, op2, s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`.
- Send **12 readings** for a full history window (10 for the rolling window, plus 2 so each of the last 3 scored rows has a full window). Fewer still works, and the response includes a `warning`.
- Missing or null values return `422`. At most 500 readings per request.

Example response:

```json
{
  "score": 0.999,
  "band": "Alarm",
  "recent_probs": [0.998, 0.998, 0.999],
  "warning": null,
  "note": "Uncalibrated risk score, not a probability. Never proof a machine is safe or unsafe."
}
```

Interactive docs are at `http://127.0.0.1:8000/docs` once the server is running.

## Run with Docker

```
docker build -t machine-risk-api .
docker run -d --name risk-api --restart unless-stopped -p 8000:8000 machine-risk-api
```

Later: `docker start risk-api` and `docker stop risk-api`. After changing `api.py`, `features.py`, `meta.json`, the model or `requirements.txt`, remove the container and rebuild.

## Run locally

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
uvicorn api:app --reload
```

## Reproduce from scratch

The raw data is not in this repo. Download C-MAPSS from the NASA Prognostics Data Repository and place `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` in `data/raw/`. Then:

```
python prepare_data.py     # features, labels, engine split, meta.json
python train_final.py      # trains on all 100 engines, saves models/lgbm_final.joblib
python check_api.py        # compares API output with the model directly (container must be running)
```

Experiment scripts: `baseline_logreg.py`, `rf_model.py`, `lgbm_model.py`, `cv_compare.py`, `threshold_cost.py`, `engine_alarm.py`, `calibration_curves.py`, `final_check.py`.

## Project structure

```
api.py                 FastAPI app (/predict)
features.py            add_features(): rolling mean and slope, shared by training and API
prepare_data.py        raw data -> labeled, split, featured CSVs + meta.json
train_final.py         final model on all 100 engines
check_api.py           API vs direct model check
final_check.py         one-time NASA test evaluation (does not save a model)
cv_compare.py          GroupKFold comparison of the 3 models
engine_alarm.py        engine-level alarm study (k, threshold, gap, lead)
threshold_cost.py      row-level cost threshold sweep (superseded by the engine rule)
calibration_curves.py  calibration and learning curve
models/                lgbm_final.joblib
data/processed/        meta.json (CSVs are regenerated, not committed)
Dockerfile, requirements.txt, .dockerignore
```

## Limitations

- FD001 is simulated, clean and easy. Real machines will be noisier.
- The RUL <= 30 label threshold is an assumption and was not tested at 20 or 50.
- The 10x miss-to-false-alarm cost ratio, k = 3, gap = 10 and minimum lead = 5 are guesses and were never validated.
- Cross-validation was more lenient than the final test (0 misses in CV vs 9 of 25 on NASA's test). The NASA test has only 25 positives, so its percentages are shaky.
- Watch >= 0.5 was set after the final test. It was never checked in cross-validation.
- Model settings were tuned on engines that sit inside the CV folds, so the numbers are slightly optimistic.
- Rows within one engine are highly correlated. The data holds about 100 independent engines, not 20,000 rows.
- Differences between models (0.963 / 0.967 / 0.970 on the 20 validation engines) are within noise.
- Scores are uncalibrated and over-confident between 0.1 and 0.9.
- A short history (fewer than 12 readings) makes the score less reliable.

## Tech

Python 3.14, pandas, NumPy, scikit-learn, LightGBM, FastAPI, Uvicorn, Docker. Exact versions are pinned in `requirements.txt`.
