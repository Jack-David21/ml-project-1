import json
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, recall_score

data = pd.concat([
    pd.read_csv("data/processed/train.csv"),
    pd.read_csv("data/processed/val.csv"),
], ignore_index=True)

with open("data/processed/meta.json") as f:
    keep = json.load(f)["keep"]
features = keep + [c + "_mean" for c in keep] + [c + "_slope" for c in keep]

X, y, groups = data[features], data["label"], data["unit"]
print("Features:", len(features), "| Engines:", groups.nunique())

models = {
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]),
    "RF": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=20, class_weight="balanced",
        random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(
        n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=200,
        is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
}

gkf = GroupKFold(n_splits=5)
for name, model in models.items():
    pr, rec = [], []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        prob = model.predict_proba(X.iloc[va_idx])[:, 1]
        pr.append(average_precision_score(y.iloc[va_idx], prob))
        rec.append(recall_score(y.iloc[va_idx], prob >= 0.5))
    print(f"\n{name}")
    print("PR-AUC per fold:", [round(p, 3) for p in pr])
    print("PR-AUC mean/std:", round(sum(pr)/5, 3), "/", round(pd.Series(pr).std(), 3))
    print("Recall@0.5 mean:", round(sum(rec)/5, 3))