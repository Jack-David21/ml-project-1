import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

meta = json.load(open("data/processed/meta.json"))
keep = meta["keep"]
features = keep + [c + "_mean" for c in keep] + [c + "_slope" for c in keep]

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
full = pd.concat([train, val], ignore_index=True)

X = full[features]
y = full["label"]
print("X shape:", X.shape, "| positive rate:", round(y.mean(), 3))

model = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=7,
    min_child_samples=200,
    is_unbalance=True,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(X, y)

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/lgbm_final.joblib")
print("saved: models/lgbm_final.joblib")