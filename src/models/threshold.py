"""
Cost-based decision threshold via the expected-value framework.

A churn classifier outputs a probability; turning that into an *action* (contact
the customer with a retention offer, or not) requires a threshold. The default
0.35 in the original repo was arbitrary. Here the threshold is chosen to maximise
expected business value given the economics of retention, following the
expected-value framework (Provost & Fawcett, *Data Science for Business*).

Per-customer outcomes at a given threshold:
  * TP (flag a true churner):   save_rate * CLV - retention_cost
  * FP (flag a non-churner):    - retention_cost          (offer wasted)
  * FN (miss a churner):        - CLV                     (customer lost)
  * TN (correctly ignore):      0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChurnEconomics:
    """Economic assumptions that drive the optimal threshold."""

    clv: float = 1000.0  # margin retained when a would-be churner is saved
    retention_cost: float = 200.0  # cost to contact + offer, per flagged customer
    save_rate: float = 0.30  # P(retention succeeds | true churner contacted)


def expected_value(
    y_true: Sequence[int], proba: Sequence[float], threshold: float, econ: ChurnEconomics
) -> float:
    """Total expected net value ($) of acting at ``threshold`` on this set."""
    yt = np.asarray(y_true)
    pr = np.asarray(proba)
    pred = pr >= threshold

    tp = int(np.sum(pred & (yt == 1)))
    fp = int(np.sum(pred & (yt == 0)))
    fn = int(np.sum(~pred & (yt == 1)))

    return float(
        tp * (econ.save_rate * econ.clv - econ.retention_cost)
        + fp * (-econ.retention_cost)
        + fn * (-econ.clv)
    )


def optimize_threshold(
    y_true: Sequence[int],
    proba: Sequence[float],
    econ: ChurnEconomics | None = None,
    grid: Sequence[float] | None = None,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Pick the threshold maximising expected value.

    Returns ``(best_threshold, best_value, curve)`` where ``curve`` is the list of
    ``(threshold, expected_value)`` points (handy for logging/plotting).
    """
    econ = econ or ChurnEconomics()
    if grid is None:
        grid = np.round(np.linspace(0.05, 0.95, 91), 3)

    curve = [(float(t), expected_value(y_true, proba, t, econ)) for t in grid]
    best_threshold, best_value = max(curve, key=lambda kv: kv[1])
    return best_threshold, best_value, curve


def do_nothing_value(y_true: Sequence[int], econ: ChurnEconomics) -> float:
    """Baseline value when no one is contacted: every churner is lost."""
    return float(-int(np.sum(np.asarray(y_true) == 1)) * econ.clv)
