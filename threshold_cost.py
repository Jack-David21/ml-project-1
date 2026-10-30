import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

data = pd.concat([
    pd.read_csv("data/processed/train.csv"),
    pd.read_csv("data/processed/val.csv"),
], ignore_index=True)

with open("data/processed/meta.json") as f:
    keep = json.load(f)["keep"]
features = keep + [c + "_mean" for c in keep] + [c + "_slope" for c in keep]

X, y, groups = data[features], data["label"], data["unit"]

# out-of-fold probabilities: every row scored by a model that never saw its engine
oof = np.zeros(len(data))
gkf = GroupKFold(n_splits=5)
for tr_idx, va_idx in gkf.split(X, y, groups):
    model = LGBMClassifier(
        n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=200,
        is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]

thresholds = np.arange(0.05, 0.96, 0.05)
y_arr = y.to_numpy()

def counts(t):
    pred = oof >= t
    misses = int(((~pred) & (y_arr == 1)).sum())
    false_alarms = int((pred & (y_arr == 0)).sum())
    return misses, false_alarms

for cost_miss in [5, 10, 50]:
    costs = []
    for t in thresholds:
        m, fa = counts(t)
        costs.append(cost_miss * m + fa)
    best = int(np.argmin(costs))
    m0, fa0 = counts(0.5)
    cost_at_05 = cost_miss * m0 + fa0
    m1, fa1 = counts(thresholds[best])
    print(f"\nMiss costs {cost_miss}x a false alarm")
    print(f"Best threshold: {thresholds[best]:.2f} | misses {m1}, false alarms {fa1}, cost {costs[best]}")
    print(f"At 0.50:        misses {m0}, false alarms {fa0}, cost {cost_at_05}")
    print(f"Cost saved vs 0.50: {round(100 * (cost_at_05 - costs[best]) / cost_at_05, 1)}%")