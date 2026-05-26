#!/usr/bin/env python3
"""
Train the Telco churn model end-to-end and log the full pipeline to MLflow.

This is the productionised version of notebooks/telco_churn_end_to_end.ipynb and
reproduces it step for step:

    load -> clean -> validate (pandera) -> split -> tune (Optuna) ->
    fit Pipeline(preprocess + XGBoost) -> cost-based threshold -> evaluate -> log.

The logged artifact is a single scikit-learn Pipeline that ingests *raw* customer
columns, so training and serving share one feature path (no train/serve skew).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Keep emoji/log output from crashing on a non-UTF-8 console (Windows cp1252, CI).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from src.data.load_data import load_data
from src.features.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_FEATURES,
    TARGET,
    build_pipeline,
    clean_data,
)
from src.models.threshold import ChurnEconomics, do_nothing_value, optimize_threshold
from src.models.tune import tune_xgb
from src.utils.validate_data import validate_telco_data


def main(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_dir = os.path.join(project_root, "mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri(args.mlflow_uri or Path(mlruns_dir).as_uri())
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_param("model", "xgboost_pipeline")
        mlflow.log_param("test_size", args.test_size)

        # === STAGE 1: Load + clean ===
        print("Loading data...")
        df = load_data(args.input)
        df = clean_data(df)
        print(f"Loaded & cleaned: {df.shape}")

        # === STAGE 2: Validate against the pandera contract (raw frame) ===
        is_valid, failed = validate_telco_data(load_data(args.input))
        mlflow.log_metric("data_quality_pass", int(is_valid))
        if not is_valid:
            mlflow.log_text(json.dumps(failed, indent=2), "failed_expectations.json")
            raise ValueError(f"Data quality check failed: {failed}")

        # === STAGE 3: Features + target ===
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' not found")
        X = df[RAW_FEATURES].copy()
        y = df[TARGET].astype(int)

        # Persist the cleaned modelling frame (matches the notebook artifact).
        processed_path = os.path.join(
            project_root, "data", "processed", "telco_churn_processed.csv"
        )
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)

        # === STAGE 4: Split (test = untouched holdout) ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
        )
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        mlflow.log_param("scale_pos_weight", round(float(scale_pos_weight), 4))

        # === STAGE 5: Hyperparameter tuning (Optuna) ===
        if args.tune:
            print(f"Tuning XGBoost with Optuna ({args.n_trials} trials)...")
            best_params = tune_xgb(
                X_train, y_train, n_trials=args.n_trials, random_state=args.random_state
            )
        else:
            best_params = {
                "n_estimators": 301,
                "learning_rate": 0.034,
                "max_depth": 7,
                "subsample": 0.95,
                "colsample_bytree": 0.98,
            }
        mlflow.log_params({f"xgb_{k}": v for k, v in best_params.items()})

        # === STAGE 6: Fit the final pipeline on the full training set ===
        xgb = XGBClassifier(
            **best_params,
            random_state=args.random_state,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
        # scale_pos_weight gives the tree good ranking on imbalanced data but inflates its
        # probabilities; isotonic calibration maps them back to honest probabilities.
        estimator = (
            CalibratedClassifierCV(xgb, method="isotonic", cv=5) if args.calibrate else xgb
        )
        mlflow.log_param("calibrated", args.calibrate)
        pipe = build_pipeline(estimator)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        mlflow.log_metric("train_time", time.time() - t0)

        # === STAGE 7: Cost-based threshold on the same inner-validation split ===
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=args.random_state
        )
        proba_val = pipe.predict_proba(X_val)[:, 1]
        econ = ChurnEconomics(
            clv=args.clv, retention_cost=args.retention_cost, save_rate=args.save_rate
        )
        mlflow.log_params(
            {"clv": econ.clv, "retention_cost": econ.retention_cost, "save_rate": econ.save_rate}
        )
        if args.threshold is not None:
            chosen_threshold, _, curve = (args.threshold, None, [])
            best_t, best_v, curve = optimize_threshold(y_val, proba_val, econ)
            mlflow.log_param("threshold_mode", "fixed")
        else:
            best_t, best_v, curve = optimize_threshold(y_val, proba_val, econ)
            chosen_threshold = best_t
            mlflow.log_param("threshold_mode", "cost_optimized")
        uplift = best_v - do_nothing_value(y_val, econ)
        mlflow.log_param("threshold", round(float(chosen_threshold), 3))
        mlflow.log_metric("val_uplift_vs_do_nothing", uplift)
        print(f"Cost-optimal threshold = {best_t:.3f} | uplift vs do-nothing = ${uplift:,.0f}")

        # === STAGE 8: Evaluate on the untouched holdout ===
        proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (proba >= chosen_threshold).astype(int)
        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        print(
            f"Holdout @ {chosen_threshold:.2f}: "
            + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        print(classification_report(y_test, y_pred, digits=3))

        # === STAGE 9: Cost curve artifact ===
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        if curve:
            ts = [t for t, _ in curve]
            vs = [v for _, v in curve]
            plt.figure(figsize=(7, 4))
            plt.plot(ts, vs)
            plt.axvline(
                chosen_threshold, color="red", ls="--", label=f"chosen={chosen_threshold:.2f}"
            )
            plt.xlabel("decision threshold")
            plt.ylabel("expected net value ($)")
            plt.title("Retention value vs threshold")
            plt.legend()
            plt.tight_layout()
            curve_path = os.path.join(artifacts_dir, "threshold_cost_curve.png")
            plt.savefig(curve_path, dpi=120)
            plt.close()
            mlflow.log_artifact(curve_path)

        # === STAGE 10: Log the full pipeline + feature contract ===
        signature = infer_signature(X_test, pipe.predict(X_test))
        mlflow.sklearn.log_model(
            pipe, artifact_path="model", signature=signature, input_example=X_test.head(3)
        )
        contract = {
            "raw_features": RAW_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "target": TARGET,
            "threshold": round(float(chosen_threshold), 4),
            "economics": {
                "clv": econ.clv,
                "retention_cost": econ.retention_cost,
                "save_rate": econ.save_rate,
            },
        }
        with open(os.path.join(artifacts_dir, "feature_contract.json"), "w") as f:
            json.dump(contract, f, indent=2)
        mlflow.log_dict(contract, "feature_contract.json")
        print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the churn pipeline (XGBoost + MLflow)")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to CSV (e.g., data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)",
    )
    p.add_argument(
        "--target",
        type=str,
        default="Churn",
        help="target column (fixed to Churn for this dataset)",
    )
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--tune", action="store_true", default=True, help="run Optuna tuning")
    p.add_argument("--no_tune", dest="tune", action="store_false", help="skip tuning, use defaults")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--calibrate", action="store_true", default=True,
                   help="isotonic-calibrate the final model's probabilities (default on)")
    p.add_argument("--no_calibrate", dest="calibrate", action="store_false")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="fixed decision threshold; default = cost-optimized",
    )
    p.add_argument("--clv", type=float, default=1000.0)
    p.add_argument("--retention_cost", type=float, default=200.0)
    p.add_argument("--save_rate", type=float, default=0.30)
    p.add_argument("--mlflow_uri", type=str, default=None)
    main(p.parse_args())
