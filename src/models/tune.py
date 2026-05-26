"""
Optuna hyperparameter tuning for the churn XGBoost model.

Extracted from notebooks/telco_churn_end_to_end.ipynb (section 17). The objective
is evaluated on a validation split carved from the *training* data - never the
test set - so the holdout stays untouched for final reporting. The seed makes the
search reproducible.
"""

from __future__ import annotations

import optuna
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.pipeline import build_pipeline


def tune_xgb(
    X_train,
    y_train,
    n_trials: int = 30,
    random_state: int = 42,
    tuning_threshold: float = 0.30,
) -> dict:
    """Return the best XGBoost hyper-parameters found by Optuna.

    Mirrors the notebook: recall objective at ``tuning_threshold`` on an inner
    validation split, TPE sampler seeded for reproducibility.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )
    spw_fit = (y_fit == 0).sum() / (y_fit == 1).sum()

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            random_state=random_state,
            n_jobs=-1,
            scale_pos_weight=spw_fit,
            eval_metric="logloss",
        )
        pipe = build_pipeline(XGBClassifier(**params)).fit(X_fit, y_fit)
        proba = pipe.predict_proba(X_val)[:, 1]
        return recall_score(y_val, (proba >= tuning_threshold).astype(int))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
