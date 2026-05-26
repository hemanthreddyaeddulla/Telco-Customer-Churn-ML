#!/usr/bin/env python3
"""
Generate a PSI feature-drift report between a reference and a current dataset.

    python scripts/drift_report.py --current path/to/new_batch.csv
    python scripts/drift_report.py            # demo: reference vs a synthetically-shifted copy

Writes artifacts/drift_report.md and prints the per-feature PSI table.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, clean_data
from src.serving.monitoring import feature_drift

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROCESSED = os.path.join(ROOT, "data", "processed", "telco_churn_processed.csv")
_RAW = os.path.join(ROOT, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


def main(args):
    reference = args.reference or (_PROCESSED if os.path.exists(_PROCESSED) else _RAW)
    ref = clean_data(pd.read_csv(reference))

    if args.current:
        cur = clean_data(pd.read_csv(args.current))
        current_desc = args.current
    else:
        # Demo: synthesise a drifted batch so the report shows detection working.
        cur = ref.copy()
        cur["MonthlyCharges"] = cur["MonthlyCharges"] * 1.20
        cur["Contract"] = "Month-to-month"
        current_desc = "SIMULATED drift (MonthlyCharges +20%, all Month-to-month)"

    report = feature_drift(ref, cur, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    n_drift = int((report["status"] == "drift").sum())
    print(report.to_string(index=False))
    print(f"\nFeatures with significant drift (PSI > 0.25): {n_drift}")

    out = os.path.join(ROOT, "artifacts", "drift_report.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Feature drift report (PSI)\n\n")
        f.write(f"- reference: `{os.path.relpath(reference, ROOT)}`\n")
        f.write(f"- current: {current_desc}\n")
        f.write(f"- significant-drift features (PSI > 0.25): **{n_drift}**\n\n")
        f.write("PSI: <0.10 stable | 0.10-0.25 moderate | >0.25 drift\n\n")
        f.write("```\n" + report.to_string(index=False) + "\n```\n")
    print(f"Wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PSI drift report")
    p.add_argument("--reference", default=None, help="reference CSV (default: training data)")
    p.add_argument("--current", default=None, help="current CSV (default: simulated drift)")
    main(p.parse_args())
