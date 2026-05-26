"""Unit tests for the preprocessing pipeline and serving alignment."""

import pandas as pd

from src.features.pipeline import (
    RAW_FEATURES,
    align_raw_features,
    build_preprocessor,
)


def test_align_fills_missing_columns():
    df = pd.DataFrame([{"gender": "Male", "tenure": 5}])
    out = align_raw_features(df)
    assert list(out.columns) == RAW_FEATURES
    assert out.shape == (1, len(RAW_FEATURES))


def test_preprocessor_ignores_unknown_category(clean_df):
    pre = build_preprocessor().fit(clean_df[RAW_FEATURES])
    row = clean_df[RAW_FEATURES].iloc[[0]].copy()
    row["InternetService"] = "Satellite"  # category never seen at fit time
    out = pre.transform(row)  # must not raise
    assert out.shape[0] == 1


def test_single_row_encoding_equals_batch(trained_pipeline, clean_df):
    # The core skew guarantee: a single row encodes identically to a batch.
    X = clean_df[RAW_FEATURES]
    p_single = trained_pipeline.predict_proba(X.iloc[[7]])[0, 1]
    p_in_batch = trained_pipeline.predict_proba(X.iloc[:10])[7, 1]
    assert abs(p_single - p_in_batch) < 1e-9
