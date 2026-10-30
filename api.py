import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import add_features

meta = json.load(open("data/processed/meta.json"))
KEEP, WINDOW = meta["keep"], meta["window"]
FEATURES = KEEP + [c + "_mean" for c in KEEP] + [c + "_slope" for c in KEEP]
model = joblib.load("models/lgbm_final.joblib")

ALARM_T, WATCH_T, K = 0.88, 0.5, 3
MIN_ROWS = WINDOW + K - 1  # 12 readings give all 3 scored rows a full window

app = FastAPI(title="Machine Failure Risk")


class Request(BaseModel):
    readings: list[dict[str, float]] = Field(min_length=1, max_length=500)


@app.post("/predict")
def predict(req: Request):
    df = pd.DataFrame(req.readings)
    cols = ["cycle"] + KEEP

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(422, f"missing columns: {missing}")
    if df[cols].isna().any().any():
        raise HTTPException(422, "null or missing values in readings")

    n_sent = len(df)
    df = df.sort_values("cycle").tail(MIN_ROWS).copy()
    df["unit"] = 1
    feats = add_features(df, KEEP, win=WINDOW)
    probs = model.predict_proba(feats[FEATURES])[:, 1][-K:]

    if len(probs) == K and (probs >= ALARM_T).all():
        band = "Alarm"
    elif probs[-1] >= WATCH_T:
        band = "Watch"
    else:
        band = "Low"

    warning = None
    if n_sent < MIN_ROWS:
        warning = (
            f"Only {n_sent} readings sent; {MIN_ROWS} are needed for a full "
            "history window. Score is less reliable."
        )

    return {
        "score": round(float(probs[-1]), 3),
        "band": band,
        "recent_probs": [round(float(p), 3) for p in probs],
        "warning": warning,
        "note": "Uncalibrated risk score, not a probability. Never proof a machine is safe or unsafe.",
    }