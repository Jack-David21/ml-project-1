## Run locally

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
uvicorn scripts.serving.api:app --reload
```

## Reproduce from scratch

The raw data is not in this repo. Download C-MAPSS from the NASA Prognostics Data Repository and place `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` in `data/raw/`. Run each command from the project root:

```
python -m scripts.data.prepare_data        # features, labels, engine split, meta.json
python -m scripts.models.train_final       # trains on all 100 engines, saves models/lgbm_final.joblib
python -m scripts.evaluation.check_api     # compares API output with the model (container must be running)
```

## Project structure

```
scripts/
  features.py                add_features(): rolling mean and slope, shared by training and API
  data/prepare_data.py       raw data -> labeled, split, featured CSVs + meta.json
  models/
    train_final.py           final model on all 100 engines
    baseline_logreg.py, rf_model.py, lgbm_model.py   single-split model experiments
  evaluation/
    check_api.py             API vs direct model check
    final_check.py           one-time NASA test evaluation (does not save a model)
    cv_compare.py            GroupKFold comparison of the 3 models
    engine_alarm.py          engine-level alarm study (k, threshold, gap, lead)
    threshold_cost.py        row-level cost sweep (superseded by the engine rule)
    calibration_curves.py    calibration and learning curve
  serving/api.py             FastAPI app (/predict)
models/                      lgbm_final.joblib
data/processed/              meta.json (CSVs are regenerated, not committed)
Dockerfile, requirements.txt, .dockerignore
```