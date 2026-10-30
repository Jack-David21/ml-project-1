import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier

data = pd.concat([
    pd.read_csv("data/processed/train.csv"),
    pd.read_csv("data/processed/val.csv"),
], ignore_index=True)

with open("data/processed/meta.json") as f:
    keep = json.load(f)["keep"]
features = keep + [c + "_mean" for c in keep] + [c + "_slope" for c in keep]
X, y, groups = data[features], data["label"], data["unit"]

def make_model():
    return LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=7,
                          min_child_samples=200, is_unbalance=True,
                          random_state=42, n_jobs=-1, verbose=-1)

# 1) Calibration: out-of-fold probabilities vs actual failure rate
oof = np.zeros(len(data))
for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
    m = make_model()
    m.fit(X.iloc[tr], y.iloc[tr])
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]

frac_pos, mean_pred = calibration_curve(y, oof, n_bins=10, strategy="quantile")
print("Predicted prob -> actual failure rate")
for p, f in zip(mean_pred, frac_pos):
    print(f"{p:.3f} -> {f:.3f}")

# 2) Learning curve: train on n engines, score on a fixed held-out set of 20 engines
engines = np.array(sorted(groups.unique()))
rng = np.random.RandomState(42)
rng.shuffle(engines)
hold, pool = engines[:20], engines[20:]
hold_mask = groups.isin(hold)
sizes = [10, 20, 40, 60, 80]
scores = []
print("\nTrain engines -> held-out PR-AUC")
for n in sizes:
    tr_mask = groups.isin(pool[:n])
    m = make_model()
    m.fit(X[tr_mask], y[tr_mask])
    s = average_precision_score(y[hold_mask], m.predict_proba(X[hold_mask])[:, 1])
    scores.append(s)
    print(f"{n} -> {s:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot([0, 1], [0, 1], "--", label="perfect")
ax[0].plot(mean_pred, frac_pos, "o-", label="LightGBM")
ax[0].set_xlabel("Predicted probability")
ax[0].set_ylabel("Actual failure rate")
ax[0].set_title("Calibration")
ax[0].legend()
ax[1].plot(sizes, scores, "o-")
ax[1].set_xlabel("Training engines")
ax[1].set_ylabel("Held-out PR-AUC")
ax[1].set_title("Learning curve")
plt.tight_layout()
plt.savefig("calibration_learning.png")
print("\nSaved calibration_learning.png")