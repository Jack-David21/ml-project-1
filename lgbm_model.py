import json
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score, precision_score, average_precision_score, confusion_matrix

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

with open("data/processed/meta.json") as f:
    keep = json.load(f)["keep"]
features = keep + [c + "_mean" for c in keep] + [c + "_slope" for c in keep]
print("Number of features:", len(features))

X_train, y_train = train[features], train["label"]
X_val, y_val = val[features], val["label"]

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
model.fit(X_train, y_train)

for name, X, y in [("TRAIN", X_train, y_train), ("VAL", X_val, y_val)]:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    print(f"\n{name}")
    print("Recall:   ", round(recall_score(y, pred), 3))
    print("Precision:", round(precision_score(y, pred), 3))
    print("PR-AUC:   ", round(average_precision_score(y, prob), 3))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y, pred))