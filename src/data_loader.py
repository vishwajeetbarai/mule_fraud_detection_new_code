"""Load and validate the real NeurIPS Bank Account Fraud benchmark (BaseDataset.csv)."""
from pathlib import Path

import numpy as np
import pandas as pd

import config

# Compact dtypes: the file is ~480 MB as default pandas dtypes, ~100 MB with these.
_FLOAT_COLS = [
    "income", "name_email_similarity", "days_since_request", "intended_balcon_amount",
    "velocity_6h", "velocity_24h", "velocity_4w", "session_length_in_minutes",
]
_INT_COLS = [
    "prev_address_months_count", "current_address_months_count", "customer_age",
    "zip_count_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
    "credit_risk_score", "bank_months_count", "proposed_credit_limit",
    "device_distinct_emails_8w", "device_fraud_count",
]
_BIN_COLS = [
    config.TARGET, "email_is_free", "phone_home_valid", "phone_mobile_valid",
    "has_other_cards", "foreign_request", "keep_alive_session", config.TIME_COL,
]

REQUIRED_COLUMNS = (
    _FLOAT_COLS + _INT_COLS + _BIN_COLS + config.CATEGORICAL_COLS
)


def _dtype_map():
    d = {c: "float32" for c in _FLOAT_COLS}
    d.update({c: "int16" for c in _INT_COLS})
    d.update({c: "int8" for c in _BIN_COLS})
    return d


def load_base_dataset(path=None, sample_frac=None, random_state=config.RANDOM_STATE):
    """Read BaseDataset.csv and validate its schema.

    Parameters
    ----------
    path : str | Path | None
        CSV location (defaults to config.DATA_PATH).
    sample_frac : float | None
        Optional random fraction of the REAL rows (for quick smoke tests only).
    """
    path = Path(path) if path else config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Place the NeurIPS 2022 Bank Account Fraud 'BaseDataset.csv' at "
            "data/BaseDataset.csv, pass --data /path/to/BaseDataset.csv, or set the "
            "BASE_DATASET_PATH environment variable.\n"
            "Source: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022"
        )

    df = pd.read_csv(path, dtype=_dtype_map())

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")
    if df.isna().any().any():
        raise ValueError("Unexpected NaN values in raw dataset.")
    if not set(df[config.TARGET].unique()) <= {0, 1}:
        raise ValueError(f"Target '{config.TARGET}' must be binary (0/1).")

    if sample_frac is not None and sample_frac < 1.0:
        df = (
            df.groupby(config.TIME_COL, group_keys=False)
            .sample(frac=sample_frac, random_state=random_state)
            .sort_index()
            .reset_index(drop=True)
        )

    return df


def temporal_split(df):
    """Split by `month` into train / validation / test (no shuffling, no overlap)."""
    parts = {}
    for name, months in (("train", config.TRAIN_MONTHS),
                         ("valid", config.VALID_MONTHS),
                         ("test", config.TEST_MONTHS)):
        parts[name] = df[df[config.TIME_COL].isin(months)].reset_index(drop=True)
        if parts[name].empty:
            raise ValueError(f"Empty {name} split for months {months}.")
    return parts["train"], parts["valid"], parts["test"]


def describe(df, name="data"):
    n, f = len(df), int(df[config.TARGET].sum())
    return f"{name:6s}: {n:>9,} rows | {f:>7,} fraud ({f / n:.3%})"
