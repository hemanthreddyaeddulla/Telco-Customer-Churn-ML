"""Data-contract tests for the pandera schema."""

from src.utils.validate_data import validate_telco_data


def test_valid_data_passes(raw_df):
    ok, failed = validate_telco_data(raw_df)
    assert ok
    assert failed == []


def test_bad_gender_fails(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "gender"] = "Unknown"
    ok, failed = validate_telco_data(bad)
    assert not ok
    assert any("gender" in f for f in failed)


def test_negative_tenure_fails(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "tenure"] = -5
    ok, failed = validate_telco_data(bad)
    assert not ok
    assert any("tenure" in f for f in failed)


def test_invalid_contract_fails(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Contract"] = "Lifetime"
    ok, failed = validate_telco_data(bad)
    assert not ok
    assert any("Contract" in f for f in failed)
