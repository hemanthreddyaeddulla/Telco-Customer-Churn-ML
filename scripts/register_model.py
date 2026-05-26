#!/usr/bin/env python3
"""
Train a pipeline and register it to the MLflow Model Registry as the 'challenger'.

    python scripts/register_model.py

Logs the fitted pipeline + holdout metrics, registers a new version of
'telco-churn-classifier', and points the 'challenger' alias at it. Promotion to
'champion' is gated separately by scripts/promote_model.py.
"""

import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.pipeline import RAW_FEATURES, TARGET, build_pipeline, clean_data
from src.models.registry import DATA, MODEL_NAME, use_registry


def main():
    use_registry()
    mlflow.set_experiment("telco-churn-registry")

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

    with mlflow.start_run():
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("recall", rec)
        mlflow.sklearn.log_model(pipe, "model", registered_model_name=MODEL_NAME)

    client = MlflowClient()
    version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    print(f"Registered {MODEL_NAME} v{version} as 'challenger' | roc_auc={auc:.4f} recall={rec:.4f}")


if __name__ == "__main__":
    main()
