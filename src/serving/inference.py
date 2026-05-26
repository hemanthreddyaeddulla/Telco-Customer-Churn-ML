"""
Inference - load the trained Pipeline and predict.

The model artifact is a complete scikit-learn Pipeline (preprocessing + XGBoost)
logged by the training script. Serving just feeds it the *raw* customer columns;
the fitted ``OneHotEncoder`` inside the pipeline reproduces the exact training
encoding for a single row. There is deliberately no feature-engineering code in
this module - that was the source of the old train/serve skew.

Model resolution order:
1. ``$MODEL_DIR`` (set in the Docker image, default ``/app/model``)
2. newest ``model`` dir under ``./mlruns`` (local development after training)
3. newest ``model`` dir under ``src/serving/model`` (checked-in dev copies)

Loading is lazy and cached so importing this module (e.g. in tests or the app)
does not require a model to be present until the first prediction.
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

import mlflow.sklearn
import numpy as np
import pandas as pd

from src.features.pipeline import CATEGORICAL_FEATURES, align_raw_features
from src.serving.monitoring import log_prediction

DEFAULT_THRESHOLD = 0.35

_SEARCH_GLOBS = [
    "./mlruns/*/*/artifacts/model",
    "./src/serving/model/*/artifacts/model",
]


def _resolve_model_dir() -> str:
    """Find a usable model directory, preferring the container path."""
    env_dir = os.environ.get("MODEL_DIR", "/app/model")
    if os.path.exists(os.path.join(env_dir, "MLmodel")):
        return env_dir

    candidates = [p for pattern in _SEARCH_GLOBS for p in glob.glob(pattern)]
    candidates = [p for p in candidates if os.path.exists(os.path.join(p, "MLmodel"))]
    if not candidates:
        raise FileNotFoundError(
            f"No MLflow model found at MODEL_DIR={env_dir!r} or under {_SEARCH_GLOBS}. "
            "Train one with scripts/run_pipeline.py first."
        )
    return max(candidates, key=os.path.getmtime)


@lru_cache(maxsize=1)
def get_model():
    """Load and cache the serving Pipeline."""
    model_dir = _resolve_model_dir()
    model = mlflow.sklearn.load_model(model_dir)
    print(f"[inference] model loaded from {model_dir}")
    return model


@lru_cache(maxsize=1)
def get_threshold() -> float:
    """Decision threshold persisted alongside the model at training time.

    The training run writes feature_contract.json (with the cost-optimised
    threshold) next to the model artifact. Falls back to DEFAULT_THRESHOLD if the
    contract is absent (e.g. an older container image).
    """
    model_dir = _resolve_model_dir()
    contract_path = os.path.join(os.path.dirname(model_dir), "feature_contract.json")
    try:
        with open(contract_path) as f:
            return float(json.load(f).get("threshold", DEFAULT_THRESHOLD))
    except (OSError, ValueError, TypeError):
        return DEFAULT_THRESHOLD


def _to_frame(input_dict: dict) -> pd.DataFrame:
    """Raw payload -> single-row frame with exactly the expected raw columns."""
    return align_raw_features(pd.DataFrame([input_dict]))


def predict_proba(input_dict: dict) -> float:
    """Return the churn probability (class 1) for one customer."""
    model = get_model()
    proba = model.predict_proba(_to_frame(input_dict))[:, 1]
    return float(proba[0])


def predict(input_dict: dict, threshold: float | None = None) -> str:
    """Return a business-friendly churn decision for one customer.

    Args:
        input_dict: raw customer attributes (CustomerData schema).
        threshold: decision threshold on churn probability; when None, uses the
            cost-optimised threshold persisted with the model (see get_threshold).
    """
    thr = get_threshold() if threshold is None else threshold
    return "Likely to churn" if predict_proba(input_dict) >= thr else "Not likely to churn"


# === Explainability (SHAP) =================================================
# A churn score is only actionable if you know *why*. SHAP attributes the model
# output for one customer to individual features, so the API can return the top
# churn drivers and a recommended retention action - not just a label.

# Maps a raw feature to a concrete retention play when it pushes a customer
# toward churn. Keyed by the raw column name.
_RETENTION_ACTIONS = {
    "Contract": "Offer a discounted 1- or 2-year contract to lock in the customer.",
    "tenure": "Early-tenure customer - enroll in onboarding + a loyalty perk.",
    "MonthlyCharges": "Review pricing; offer a loyalty discount or right-sized plan.",
    "TotalCharges": "Review pricing; offer a loyalty discount or right-sized plan.",
    "InternetService": "Fiber customers churn more - bundle value-adds or price protection.",
    "TechSupport": "Offer complimentary or trial premium tech support.",
    "OnlineSecurity": "Offer a security add-on bundle.",
    "OnlineBackup": "Offer a backup/security add-on bundle.",
    "PaymentMethod": "Incentivise a switch to automatic payment.",
    "PaperlessBilling": "Confirm billing preferences; resolve any billing friction.",
}


def _base_column(transformed_name: str) -> str:
    """Map a transformed feature name back to its raw column.

    e.g. 'num__tenure' -> 'tenure', 'cat__InternetService_Fiber optic' ->
    'InternetService'.
    """
    name = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    for col in CATEGORICAL_FEATURES:
        if name.startswith(col + "_"):
            return col
    return name


@lru_cache(maxsize=1)
def _get_explainer():
    import shap  # imported lazily; heavy dependency only needed for explanations

    # Last pipeline step is the estimator (robust to the step's name).
    final = get_model()[-1]
    # If it's a calibrated wrapper, SHAP needs the underlying tree model.
    if hasattr(final, "calibrated_classifiers_"):
        final = final.calibrated_classifiers_[0].estimator
    return shap.TreeExplainer(final)


def explain(input_dict: dict, top_k: int = 5) -> list[dict]:
    """Return the top-k SHAP drivers for one customer's churn score.

    Each driver: {feature, impact, direction}. Positive impact pushes toward
    churn; negative pulls away from it.
    """
    model = get_model()
    # All steps except the final estimator = the preprocessing (robust to step names).
    preprocessor = model[:-1]
    transformed = preprocessor.transform(_to_frame(input_dict))
    names = preprocessor.get_feature_names_out()

    contributions = np.asarray(_get_explainer().shap_values(transformed))[0]

    # SHAP scores every one-hot column (including the inactive ones), which reads
    # confusingly. Sum the columns of each original feature and label it with the
    # customer's actual value (so OnlineSecurity=Yes shows "OnlineSecurity = Yes").
    totals: dict[str, float] = {}
    for name, value in zip(names, contributions, strict=False):
        base = _base_column(name)
        totals[base] = totals.get(base, 0.0) + float(value)

    ranked = sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return [
        {
            "feature": (
                f"{base} = {input_dict.get(base, '?')}"
                if base in CATEGORICAL_FEATURES
                else base
            ),
            "impact": round(total, 4),
            "direction": "increases churn" if total > 0 else "reduces churn",
        }
        for base, total in ranked
    ]


def _recommend_action(drivers: list[dict], names_transformed: list[str]) -> str:
    """Pick a retention play from the strongest churn-increasing driver."""
    for driver in drivers:
        if driver["impact"] > 0:
            # Recover the raw column from the pretty label.
            col = driver["feature"].split(" = ")[0]
            if col in _RETENTION_ACTIONS:
                return _RETENTION_ACTIONS[col]
    return "Reach out with a personalised retention offer."


def predict_full(
    input_dict: dict, threshold: float | None = None, top_k: int = 5, log: bool = True
) -> dict:
    """Full decision payload: label, probability, threshold, drivers, action.

    Each call is appended to the prediction log (best-effort) for monitoring; pass
    ``log=False`` to skip.
    """
    proba = predict_proba(input_dict)
    thr = get_threshold() if threshold is None else threshold
    likely = proba >= thr
    drivers = explain(input_dict, top_k=top_k)
    action = _recommend_action(drivers, []) if likely else "No action needed - low churn risk."
    payload = {
        "prediction": "Likely to churn" if likely else "Not likely to churn",
        "churn_probability": round(proba, 4),
        "threshold": round(thr, 4),
        "top_drivers": drivers,
        "recommended_action": action,
    }
    if log:
        log_prediction(input_dict, payload)
    return payload
