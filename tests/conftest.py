"""Shared pytest fixtures."""

import os
import sys

import pandas as pd
import pytest
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.fetch_data import DEST as DATA
from scripts.fetch_data import main as fetch_dataset
from src.features.pipeline import RAW_FEATURES, TARGET, build_pipeline, clean_data


@pytest.fixture(scope="session")
def raw_df():
    fetch_dataset()  # download the dataset into data/raw if it isn't there yet
    return pd.read_csv(DATA)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture(scope="session")
def trained_pipeline(clean_df):
    """A small, fast pipeline trained on the real data (for behavioural tests)."""
    X = clean_df[RAW_FEATURES]
    y = clean_df[TARGET].astype(int)
    spw = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw,
        eval_metric="logloss",
    )
    return build_pipeline(model).fit(X, y)


@pytest.fixture
def sample_customer():
    return dict(
        gender="Female",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        tenure=2,
        MonthlyCharges=90.0,
        TotalCharges=180.0,
    )
