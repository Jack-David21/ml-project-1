import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from features import add_features

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed")
THRESH, WIN, SEED = 30, 10, 42

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load(name):
    return pd.read_csv(SRC / name, sep=r"\s+", header=None, names=COLS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = load("train_FD001.txt"), load("test_FD001.txt")
    rul_end = pd.read_csv(SRC / "RUL_FD001.txt", header=None)[0].to_numpy()

    # Clean: drop columns that are constant or have only 2 values (train only)
    base = [c for c in COLS if c not in ("unit", "cycle")]
    keep = [c for c in base if train[c].std() >= 1e-6 and train[c].nunique() > 2]
    print("dropped:", [c for c in base if c not in keep])

    # Label
    train["rul"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    train["label"] = (train["rul"] <= THRESH).astype(int)

    # Features
    train, test = add_features(train, keep, WIN), add_features(test, keep, WIN)

    # Split by engine
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_i, va_i = next(gss.split(train, groups=train["unit"]))
    tr, va = train.iloc[tr_i], train.iloc[va_i]

    # Final check set: last cycle of each NASA test engine
    last = test.groupby("unit").tail(1).reset_index(drop=True)
    last["rul"] = rul_end
    last["label"] = (last["rul"] <= THRESH).astype(int)

    tr.to_csv(OUT / "train.csv", index=False)
    va.to_csv(OUT / "val.csv", index=False)
    last.to_csv(OUT / "final_test.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps({"keep": keep, "window": WIN, "threshold": THRESH}))

    # Verify
    assert not set(tr.unit) & set(va.unit), "engine leakage"
    assert tr.label.nunique() == va.label.nunique() == 2
    print(f"train {tr.unit.nunique()} engines / {len(tr)} rows, pos {tr.label.mean():.3f}")
    print(f"val   {va.unit.nunique()} engines / {len(va)} rows, pos {va.label.mean():.3f}")
    print(f"final_test {len(last)} rows, pos {int(last.label.sum())}")


if __name__ == "__main__":
    main()