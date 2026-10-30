import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (recall_score, precision_score,
                             average_precision_score, confusion_matrix)
from features import add_features

THRESH_ALARM, K = 0.88, 3

with open("data/processed/meta.json") as f:
    meta = json.load(f)
keep, win, rul_thr = meta["keep"], meta["window"], meta["threshold"]
features = keep + [c + "_mean" for c in keep] + [c + "_slope" for c in keep]

# 1) Train final model on all 100 engines (settings fixed, no tuning)
data = pd.concat([pd.read_csv("data/processed/train.csv"),
                  pd.read_csv("data/processed/val.csv")], ignore_index=True)
model = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=7,
                       min_child_samples=200, is_unbalance=True,
                       random_state=42, n_jobs=-1, verbose=-1)
model.fit(data[features], data["label"])
print("Engines used for training:", data["unit"].nunique())

# 2) Rebuild test features from full history, then keep last K rows per engine
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
test = pd.read_csv("data/raw/test_FD001.txt", sep=r"\s+", header=None, names=COLS)
rul_end = pd.read_csv("data/raw/RUL_FD001.txt", header=None)[0].to_numpy()
test = add_features(test, keep, win)

last = test.groupby("unit").tail(K).copy()
last["rul"] = last["unit"].map(lambda u: rul_end[u - 1]) \
              + last.groupby("unit")["cycle"].transform("max") - last["cycle"]
last["label"] = (last["rul"] <= rul_thr).astype(int)
last["prob"] = model.predict_proba(last[features])[:, 1]

# 3) Row level: final cycle of each engine (100 rows)
final = last.groupby("unit").tail(1)
print("\nROW LEVEL (final cycle, 100 rows)")
print("PR-AUC:", round(average_precision_score(final["label"], final["prob"]), 3))
for t in [0.5, THRESH_ALARM]:
    pred = (final["prob"] >= t).astype(int)
    print(f"\nThreshold {t}")
    print("Recall:   ", round(recall_score(final["label"], pred), 3))
    print("Precision:", round(precision_score(final["label"], pred, zero_division=0), 3))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(final["label"], pred))

# 4) Engine level: flagged if the last K probabilities are all >= threshold
eng = last.groupby("unit").agg(min_prob=("prob", "min"),
                               label=("label", "last"),
                               rul=("rul", "min"))
eng["flagged"] = (eng["min_prob"] >= THRESH_ALARM).astype(int)
tn, fp, fn, tp = confusion_matrix(eng["label"], eng["flagged"]).ravel()
print(f"\nENGINE LEVEL (rule: last {K} probs >= {THRESH_ALARM})")
print(f"Caught {tp} of {tp+fn} at-risk engines | Missed {fn}")
print(f"False alarms {fp} of {fp+tn} healthy-looking engines")
print("\nMissed engines (rul at last cycle, min_prob):")
print(eng[(eng.label == 1) & (eng.flagged == 0)][["rul", "min_prob"]])