"""
Data validation as a declarative contract using pandera.

This replaces the original Great Expectations implementation, which imported
``great_expectations.dataset.PandasDataset`` - an API removed in GX 1.x. The repo
pins ``great_expectations==1.5.8``, so the original validation crashed on import
of that symbol before a single check could run.

pandera expresses the same business rules as a single declarative schema that
doubles as a data-contract test fixture (see tests/). ``validate_telco_data``
keeps the original ``(is_valid, failed)`` return contract so the training
pipeline did not need to change how it calls validation.
"""

from __future__ import annotations

import pandas as pd

try:  # pandera >= 0.20 exposes the pandas API under pandera.pandas
    import pandera.pandas as pa
    from pandera.pandas import Check, Column, DataFrameSchema
except ImportError:  # pragma: no cover - older pandera fallback
    import pandera as pa  # type: ignore[no-redef]
    from pandera import Check, Column, DataFrameSchema  # type: ignore[no-redef]

# Yes/No binary service columns share the same allowed value set.
_YES_NO = Check.isin(["Yes", "No"])

TELCO_SCHEMA = DataFrameSchema(
    {
        # Identity - must be present and unique for business operations.
        "customerID": Column(str, nullable=False, unique=True),
        # Demographics
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1]), coerce=True),
        "Partner": Column(str, _YES_NO),
        "Dependents": Column(str, _YES_NO),
        # Services
        "PhoneService": Column(str, _YES_NO),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(str, _YES_NO),
        # Numeric / financial - ranges encode business constraints.
        "tenure": Column(int, Check.in_range(0, 120), coerce=True),
        "MonthlyCharges": Column(float, Check.in_range(0, 200), coerce=True),
        # TotalCharges is blank for brand-new customers -> nullable after coercion.
        "TotalCharges": Column(float, Check.ge(0), nullable=True, coerce=True),
        # Target is present at training time, absent at serving time.
        "Churn": Column(str, Check.isin(["Yes", "No"]), required=False),
    },
    strict=False,  # allow extra columns we don't constrain
    coerce=False,
)


def validate_telco_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate the raw Telco frame against the contract.

    Args:
        df: raw (or minimally cleaned) Telco dataframe. ``TotalCharges`` may still
            be an object column; pandera coerces a working copy for the check.

    Returns:
        (is_valid, failed): ``failed`` lists ``"<column>: <check>"`` strings for
        every violated expectation (collected lazily so all failures surface at
        once, not just the first).
    """
    print("Validating data against the pandera contract...")
    # Coerce TotalCharges on a copy first: it ships as text with blank cells for
    # tenure-0 customers, which a plain float cast cannot handle (they become NaN,
    # which the schema allows as nullable).
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    try:
        TELCO_SCHEMA.validate(df, lazy=True)
        n_checks = sum(len(col.checks) + 1 for col in TELCO_SCHEMA.columns.values())
        print(f"Data validation passed ({n_checks} column expectations).")
        return True, []
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        failed = sorted(
            {
                f"{row['column']}: {row['check']}"
                for _, row in cases.iterrows()
                if pd.notna(row.get("column"))
            }
        )
        # Schema-level failures (e.g. missing column) have no 'column' value.
        for _, row in cases.iterrows():
            if pd.isna(row.get("column")):
                failed.append(f"schema: {row['check']} ({row['failure_case']})")
        failed = sorted(set(failed))
        print(f"Data validation FAILED: {len(failed)} expectation(s) violated.")
        for item in failed:
            print(f"   - {item}")
        return False, failed
