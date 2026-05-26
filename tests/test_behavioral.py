"""Model behavioural tests - directional expectations & the skew guarantee."""

import pandas as pd

from src.features.pipeline import align_raw_features


def _proba(pipe, customer):
    return pipe.predict_proba(align_raw_features(pd.DataFrame([customer])))[0, 1]


def test_high_risk_scores_higher_than_low_risk(trained_pipeline):
    high = dict(
        gender="Female",
        SeniorCitizen=1,
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
        tenure=1,
        MonthlyCharges=95.0,
        TotalCharges=95.0,
    )
    low = dict(
        gender="Male",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="Yes",
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="Yes",
        DeviceProtection="Yes",
        TechSupport="Yes",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Two year",
        PaperlessBilling="No",
        PaymentMethod="Credit card (automatic)",
        tenure=70,
        MonthlyCharges=45.0,
        TotalCharges=3150.0,
    )
    assert _proba(trained_pipeline, high) > _proba(trained_pipeline, low)


def test_internet_service_changes_score(trained_pipeline, sample_customer):
    # The old single-row serving collapsed all internet values to the DSL
    # reference, making these identical. They must now differ.
    p_dsl = _proba(trained_pipeline, {**sample_customer, "InternetService": "DSL"})
    p_fiber = _proba(trained_pipeline, {**sample_customer, "InternetService": "Fiber optic"})
    assert abs(p_dsl - p_fiber) > 1e-6


def test_longer_tenure_not_more_risky(trained_pipeline, sample_customer):
    short = {**sample_customer, "tenure": 1}
    long = {**sample_customer, "tenure": 72, "TotalCharges": 72 * sample_customer["MonthlyCharges"]}
    # tenure is strongly protective; longer tenure must not raise churn risk.
    assert _proba(trained_pipeline, long) <= _proba(trained_pipeline, short) + 0.05
