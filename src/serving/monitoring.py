"""
Production monitoring: prediction logging + data-drift detection (PSI).

Two pieces a deployed model needs but the original project lacked:

1. ``log_prediction`` - append every scored request (inputs + outputs) to a JSONL
   file, so predictions can be audited and replayed, and so "current" traffic can
   be compared against the training distribution.
2. ``feature_drift`` - Population Stability Index (PSI) per feature between a
   reference (training) frame and a current frame. PSI is the standard,
   dependency-free drift metric (no heavy library needed):
     PSI < 0.10  -> stable;  0.10-0.25 -> moderate shift;  > 0.25 -> significant drift.

(Evidently would give richer HTML reports but pulls a large dependency tree that
previously destabilised this project's pinned stack; PSI captures the same signal.)
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd

PREDICTION_LOG = os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl")
_EPS = 1e-6


def log_prediction(features: dict, result: dict, path: str | None = None) -> None:
    """Append one prediction record to the JSONL log. Never raises (best-effort)."""
    path = path or PREDICTION_LOG
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "features": features,
        "result": result,
    }
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass  # logging must never break serving


def _psi_numeric(reference, current, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    ref_pct = np.histogram(ref, edges)[0] / max(len(ref), 1)
    cur_pct = np.histogram(cur, edges)[0] / max(len(cur), 1)
    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref_freq = reference.value_counts(normalize=True)
    cur_freq = current.value_counts(normalize=True)
    psi = 0.0
    for cat in set(ref_freq.index) | set(cur_freq.index):
        r = max(float(ref_freq.get(cat, 0.0)), _EPS)
        c = max(float(cur_freq.get(cat, 0.0)), _EPS)
        psi += (c - r) * np.log(c / r)
    return float(psi)


def _label(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "drift"


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    """Per-feature PSI between reference and current frames, sorted worst-first."""
    rows = []
    for col in numeric:
        if col in reference and col in current:
            psi = _psi_numeric(reference[col], current[col])
            rows.append({"feature": col, "type": "numeric", "psi": round(psi, 4), "status": _label(psi)})
    for col in categorical:
        if col in reference and col in current:
            psi = _psi_categorical(reference[col].astype(str), current[col].astype(str))
            rows.append({"feature": col, "type": "categorical", "psi": round(psi, 4), "status": _label(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
