#!/usr/bin/env python3
"""
Model-quality gate for CI.

Trains a quick pipeline and FAILS (exit 1) if ROC-AUC or recall fall below the
committed floors in metrics_baseline.json. This turns CI into a *model* quality
gate, not just a build/test check: a change that quietly degrades the model fails
the build.

Downloads the dataset first if it is not already present (scripts/fetch_data.py).
"""

import json
import os
import sys

import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.fetch_data import DEST as DATA
from scripts.fetch_data import main as fetch_dataset
from src.features.pipeline import RAW_FEATURES, TARGET, build_pipeline, clean_data

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    with open(os.path.join(ROOT, "metrics_baseline.json")) as f:
        baseline = json.load(f)

    fetch_dataset()  # ensure the dataset is present (download if missing)
    df = clean_data(pd.read_csv(DATA))
    X, y = df[RAW_FEATURES], df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    pipe = build_pipeline(
        XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42,
            n_jobs=-1, scale_pos_weight=spw, eval_metric="logloss",
        )
    ).fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    rec = recall_score(y_te, (proba >= 0.30).astype(int))

    print(f"Data        : {os.path.relpath(DATA, ROOT)}")
    print(f"ROC-AUC     : {auc:.4f}  (floor {baseline['roc_auc_min']})")
    print(f"recall@0.30 : {rec:.4f}  (floor {baseline['recall_min']})")

    failures = []
    if auc < baseline["roc_auc_min"]:
        failures.append(f"ROC-AUC {auc:.4f} < {baseline['roc_auc_min']}")
    if rec < baseline["recall_min"]:
        failures.append(f"recall {rec:.4f} < {baseline['recall_min']}")
    if failures:
        print("GATE FAILED: " + "; ".join(failures))
        sys.exit(1)
    print("GATE PASSED.")


if __name__ == "__main__":
    main()
