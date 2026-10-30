import json
import urllib.request

import joblib
import pandas as pd

meta = json.load(open("data/processed/meta.json"))
keep = meta["keep"]
features = keep + [c + "_mean" for c in keep] + [c + "_slope" for c in keep]
cols = ["cycle"] + keep

model = joblib.load("models/lgbm_final.joblib")
val = pd.read_csv("data/processed/val.csv")

worst = 0.0
for u in val["unit"].unique()[:3]:
    eng = val[val["unit"] == u].sort_values("cycle")
    for cut in (0, 25, 60):  # cut rows off the end: end of life, near label boundary, healthy
        sub = eng.iloc[: len(eng) - cut]
        direct = model.predict_proba(sub[features])[:, 1][-3:]

        body = json.dumps({"readings": sub.tail(12)[cols].to_dict(orient="records")}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8000/predict", data=body,
            headers={"Content-Type": "application/json"},
        )
        api = json.load(urllib.request.urlopen(req))["recent_probs"]

        diff = max(abs(round(float(d), 3) - a) for d, a in zip(direct, api))
        worst = max(worst, diff)
        print(f"unit {u} cut {cut}: direct {[round(float(d), 3) for d in direct]} api {api}")

print("OK" if worst <= 0.001 else f"MISMATCH, worst diff {worst}")