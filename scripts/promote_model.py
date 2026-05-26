#!/usr/bin/env python3
"""
Champion/challenger promotion gate.

    python scripts/promote_model.py

Scores the 'challenger' and the current 'champion' on the same holdout and promotes
the challenger to 'champion' ONLY if it does not regress (roc_auc >= champion - tol).
If there is no champion yet, the challenger is promoted. This is the model-registry
equivalent of the CI metric gate: no silent quality regressions reach production.
"""

import argparse
import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.pipeline import RAW_FEATURES, TARGET, clean_data
from src.models.registry import DATA, MODEL_NAME, use_registry


def _holdout():
    df = clean_data(pd.read_csv(DATA))
    X, y = df[RAW_FEATURES], df[TARGET].astype(int)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_te, y_te


def _auc_for_alias(client, alias, X_te, y_te):
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    except Exception:
        return None, None
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
    return float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])), mv.version


def main(args):
    use_registry()
    client = MlflowClient()
    X_te, y_te = _holdout()

    chal_auc, chal_v = _auc_for_alias(client, "challenger", X_te, y_te)
    if chal_auc is None:
        print("No 'challenger' registered. Run scripts/register_model.py first.")
        sys.exit(1)
    print(f"challenger v{chal_v}: roc_auc={chal_auc:.4f}")

    champ_auc, champ_v = _auc_for_alias(client, "champion", X_te, y_te)
    if champ_auc is None:
        client.set_registered_model_alias(MODEL_NAME, "champion", chal_v)
        print(f"No champion yet -> PROMOTED challenger v{chal_v} to 'champion'.")
        return
    print(f"champion   v{champ_v}: roc_auc={champ_auc:.4f}")

    if chal_auc >= champ_auc - args.tol:
        client.set_registered_model_alias(MODEL_NAME, "champion", chal_v)
        print(f"PROMOTED challenger v{chal_v} to 'champion' (>= champion - {args.tol}).")
    else:
        print(
            f"REJECTED: challenger {chal_auc:.4f} < champion {champ_auc:.4f} - {args.tol}. "
            "Champion unchanged."
        )
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Champion/challenger promotion gate")
    p.add_argument("--tol", type=float, default=0.0, help="allowed AUC regression before rejecting")
    main(p.parse_args())
