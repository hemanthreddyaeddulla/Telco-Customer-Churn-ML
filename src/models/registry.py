"""
Shared config + helpers for the MLflow Model Registry workflow.

The Model Registry requires a database-backed tracking store (file:// stores do
not support it), so we use a local SQLite database. In production this would point
at a managed MLflow tracking server.
"""

from __future__ import annotations

import os

import mlflow

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGISTRY_URI = os.environ.get(
    "MLFLOW_REGISTRY_URI", "sqlite:///" + os.path.join(ROOT, "mlflow_registry.db")
)
MODEL_NAME = "telco-churn-classifier"

_RAW = os.path.join(ROOT, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
DATA = os.environ.get("CHURN_DATA") or _RAW


def use_registry() -> None:
    """Point MLflow at the registry-capable (SQLite) tracking store."""
    mlflow.set_tracking_uri(REGISTRY_URI)
