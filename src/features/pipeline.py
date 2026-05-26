"""
Single source of truth for cleaning + feature transformations.

Extracted from notebooks/telco_churn_end_to_end.ipynb. The SAME fitted
scikit-learn Pipeline is used at training and serving time, which eliminates
training/serving skew: the ``OneHotEncoder`` learns the category set when fit on
the training data and applies that identical encoding to a single serving row.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Churn"

# Raw input feature contract (matches the serving API schema, CustomerData).
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Defaults that make serving robust to a partial payload.
_CATEGORICAL_DEFAULT = "__missing__"
_NUMERIC_DEFAULT = 0


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco frame (mirrors the notebook's cleaning step).

    - strip header whitespace
    - drop the customerID identifier
    - coerce TotalCharges to numeric; the 11 blanks are tenure-0 (brand-new)
      customers with nothing billed yet, so they are filled with 0
    - normalise SeniorCitizen to int
    - map the target to 0/1 when present (training time)
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in ["customerID", "CustomerID"] if c in df.columns])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    if TARGET in df.columns and df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].str.strip().map({"No": 0, "Yes": 1}).astype(int)
    return df


def align_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict ``df`` to exactly ``RAW_FEATURES``, filling any missing columns.

    The ColumnTransformer selects columns by name and would raise if one is
    absent; filling missing categoricals with a sentinel (encoded as all-zeros by
    ``handle_unknown="ignore"``) and missing numerics with 0 keeps serving robust
    to partial payloads.
    """
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = _CATEGORICAL_DEFAULT
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = _NUMERIC_DEFAULT
    return df[RAW_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Impute numerics (median safety net) and one-hot encode categoricals.

    ``handle_unknown="ignore"`` makes an unseen category encode to all-zeros
    instead of erroring. No ``drop_first``: tree models need no reference category,
    and dropping one is what created the single-row serving bug.
    """
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_pipeline(model) -> Pipeline:
    """Wrap preprocessing + an estimator as one fit/serialise/serve unit."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])
