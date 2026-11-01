import numpy as np
import pandas as pd
from typing import Dict, List, Any

def simulate_delayed_feedback(
    df: pd.DataFrame,
    time_col: str = 'month',
    target_col: str = 'fraud_bool',
    delay_months: int = 1
) -> pd.DataFrame:
    """Simulate realistic verification lag in mule account labels."""
    out = df.copy()
    out['verified_month'] = out[time_col] + delay_months
    return out

def get_training_slice_under_delay(
    df: pd.DataFrame,
    current_month: int,
    time_col: str = 'month',
    target_col: str = 'fraud_bool',
    delay_months: int = 1
) -> pd.DataFrame:
    """Return only applications whose labels are verified by current_month."""
    mask = (df[time_col] + delay_months) <= current_month
    return df[mask].copy()
