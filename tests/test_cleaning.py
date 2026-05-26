"""Unit tests for src.features.pipeline.clean_data."""

from src.features.pipeline import TARGET, clean_data


def test_drops_customer_id(raw_df):
    assert "customerID" not in clean_data(raw_df).columns


def test_totalcharges_numeric_no_nan(raw_df):
    out = clean_data(raw_df)
    assert out["TotalCharges"].dtype.kind == "f"
    assert out["TotalCharges"].isna().sum() == 0


def test_tenure_zero_totalcharges_is_zero(raw_df):
    # The 11 originally-blank TotalCharges rows are all tenure-0 customers,
    # so cleaning should set them to 0 (not impute a large value).
    out = clean_data(raw_df)
    assert (out.loc[out["tenure"] == 0, "TotalCharges"] == 0).all()


def test_target_mapped_to_binary(raw_df):
    out = clean_data(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}


def test_clean_is_idempotent(raw_df):
    once = clean_data(raw_df)
    twice = clean_data(once)
    assert set(twice[TARGET].unique()) <= {0, 1}
    assert "customerID" not in twice.columns
