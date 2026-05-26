#!/usr/bin/env python3
"""
Download the public IBM Telco Customer Churn dataset into data/raw/.

The dataset is intentionally NOT committed to the repository. This script fetches it on
demand (locally, in CI, or at Docker build time) so the project stays reproducible
without storing data in git. Idempotent: skips the download if the file already exists.
"""

import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEST = os.path.join(ROOT, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


def main() -> None:
    if os.path.exists(DEST):
        print(f"Dataset already present: {os.path.relpath(DEST, ROOT)}")
        return
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"Downloading dataset -> {os.path.relpath(DEST, ROOT)}")
    urllib.request.urlretrieve(URL, DEST)  # noqa: S310 (trusted public dataset URL)
    with open(DEST, encoding="utf-8") as f:
        n_rows = sum(1 for _ in f) - 1
    print(f"Done: {n_rows} rows.")
    if n_rows < 7000:
        print(f"ERROR: unexpected row count ({n_rows}); download may be corrupt.")
        sys.exit(1)


if __name__ == "__main__":
    main()
