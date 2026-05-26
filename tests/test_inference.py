"""Tests for the serving layer (model + threshold are monkeypatched)."""

import src.serving.inference as inf


def test_predict_full_payload(monkeypatch, trained_pipeline, sample_customer):
    # Point the serving layer at the in-memory test pipeline (no MLflow needed).
    inf.get_model.cache_clear()
    inf._get_explainer.cache_clear()
    monkeypatch.setattr(inf, "get_model", lambda: trained_pipeline)
    monkeypatch.setattr(inf, "get_threshold", lambda: 0.35)

    out = inf.predict_full(sample_customer)

    assert set(out) >= {
        "prediction",
        "churn_probability",
        "threshold",
        "top_drivers",
        "recommended_action",
    }
    assert out["prediction"] in ("Likely to churn", "Not likely to churn")
    assert 0.0 <= out["churn_probability"] <= 1.0
    assert isinstance(out["top_drivers"], list) and len(out["top_drivers"]) >= 1
    assert {"feature", "impact", "direction"} <= set(out["top_drivers"][0])


def test_predict_threshold_decision(monkeypatch, trained_pipeline, sample_customer):
    inf.get_model.cache_clear()
    monkeypatch.setattr(inf, "get_model", lambda: trained_pipeline)
    # A near-zero threshold should flag almost anyone as churn.
    assert inf.predict(sample_customer, threshold=0.001) == "Likely to churn"
    # A near-one threshold should flag almost no one.
    assert inf.predict(sample_customer, threshold=0.999) == "Not likely to churn"
