"""Tests for prediction logging and PSI drift detection."""
import json

import numpy as np
import pandas as pd

from src.features.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.serving.monitoring import (
    _psi_categorical,
    _psi_numeric,
    feature_drift,
    log_prediction,
)


def test_psi_zero_for_identical_distribution():
    x = np.random.RandomState(0).normal(size=2000)
    assert _psi_numeric(x, x) < 1e-6


def test_psi_numeric_detects_shift():
    rs = np.random.RandomState(0)
    assert _psi_numeric(rs.normal(0, 1, 2000), rs.normal(2.5, 1, 2000)) > 0.25


def test_psi_categorical_detects_shift():
    ref = pd.Series(["A"] * 900 + ["B"] * 100)
    cur = pd.Series(["A"] * 100 + ["B"] * 900)
    assert _psi_categorical(ref, cur) > 0.25


def test_feature_drift_flags_shifted_column(clean_df):
    current = clean_df.copy()
    current["MonthlyCharges"] = current["MonthlyCharges"] * 1.3
    report = feature_drift(clean_df, current, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    assert "MonthlyCharges" in report["feature"].values
    assert set(report["status"]) <= {"stable", "moderate", "drift"}


def test_log_prediction_writes_record(tmp_path):
    path = tmp_path / "preds.jsonl"
    log_prediction({"gender": "Male"}, {"prediction": "Likely to churn"}, path=str(path))
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["features"]["gender"] == "Male"
    assert record["result"]["prediction"] == "Likely to churn"
    assert "ts" in record
