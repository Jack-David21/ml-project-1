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

oof = np.zeros(len(data))
for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
    m = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=7,
                       min_child_samples=200, is_unbalance=True,
                       random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[tr], y.iloc[tr])
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
data["prob"] = oof
data = data.sort_values(["unit", "cycle"])

GAP, MIN_LEAD, MAX_MISSED = 10, 5, 2
KS = [1, 3, 5]
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)

engines = []
for _, eng in data.groupby("unit"):
    p = pd.Series(eng["prob"].to_numpy())
    rolled = {k: p.rolling(k).min().to_numpy() for k in KS}
    engines.append((rolled, eng["rul"].to_numpy()))

rows = []
for k in KS:
    for t in THRESHOLDS:
        missed, false_eps, leads = 0, 0, []
        for rolled, rul in engines:
            idx = np.where(rolled[k] >= t)[0]
            if len(idx) == 0:
                missed += 1
                continue
            episodes = np.split(idx, np.where(np.diff(idx) > GAP)[0] + 1)
            false_eps += sum(rul[ep].min() > 30 for ep in episodes)
            caught = idx[(rul[idx] >= MIN_LEAD) & (rul[idx] <= 30)]
            if len(caught) == 0:
                missed += 1
            else:
                leads.append(rul[caught[0]])
        med = float(np.median(leads)) if leads else float("nan")
        rows.append((k, t, missed, false_eps, med))

res = pd.DataFrame(rows, columns=["k", "threshold", "missed", "false_eps", "median_lead"])
res["ok"] = res["missed"] <= MAX_MISSED
print(res.to_string(index=False))

print("\nBest feasible (missed <= 2, fewest false episodes) per k:")
ok = res[res["ok"]]
print(ok.loc[ok.groupby("k")["false_eps"].idxmin()].to_string(index=False))