"""Unit tests for the cost-based threshold (expected-value framework)."""

import numpy as np

from src.models.threshold import (
    ChurnEconomics,
    do_nothing_value,
    expected_value,
    optimize_threshold,
)


def test_expected_value_matches_confusion_components():
    econ = ChurnEconomics(clv=1000, retention_cost=200, save_rate=0.3)
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.9, 0.8, 0.1, 0.2])
    # threshold 0.5 -> pred [1, 1, 0, 0]: TP=1, FP=1, FN=1, TN=1
    expected = 1 * (0.3 * 1000 - 200) + 1 * (-200) + 1 * (-1000)
    assert expected_value(y, proba, 0.5, econ) == expected


def test_optimize_returns_curve_maximum():
    econ = ChurnEconomics()
    y = np.array([1, 1, 0, 0, 1, 0])
    proba = np.array([0.6, 0.55, 0.4, 0.3, 0.7, 0.2])
    best_t, best_v, curve = optimize_threshold(y, proba, econ)
    assert best_v == max(v for _, v in curve)
    assert 0.05 <= best_t <= 0.95


def test_do_nothing_value_is_lost_clv():
    econ = ChurnEconomics(clv=1000)
    y = np.array([1, 1, 0])
    assert do_nothing_value(y, econ) == -2000
