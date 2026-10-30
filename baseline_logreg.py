import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, average_precision_score, confusion_matrix
import json

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

with open("data/processed/meta.json") as f:
    keep = json.load(f)["keep"]
features = keep + [c + "_mean" for c in keep] + [c + "_slope" for c in keep]
print("Number of features:", len(features))

X_train, y_train = train[features], train["label"]
X_val, y_val = val[features], val["label"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
])
model.fit(X_train, y_train)

for name, X, y in [("TRAIN", X_train, y_train), ("VAL", X_val, y_val)]:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    print(f"\n{name}")
    print("Recall:   ", round(recall_score(y, pred), 3))
    print("Precision:", round(precision_score(y, pred), 3))
    print("PR-AUC:   ", round(average_precision_score(y, prob), 3))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y, pred))