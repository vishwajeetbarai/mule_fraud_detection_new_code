import numpy as np
import pandas as pd
from typing import Dict, List, Any

def compute_fairness_cost_pareto(
    y_true: np.ndarray,
    proba: np.ndarray,
    sensitive_slice: pd.Series,
    cost_fn: float = 1500.0,
    cost_fp: float = 30.0,
    n_thresholds: int = 100
) -> pd.DataFrame:
    """Compute Pareto frontier between operational cost and disparate impact."""
    y = np.asarray(y_true)
    p = np.asarray(proba)
    slice_vals = sensitive_slice.astype(str).values
    groups = sorted(np.unique(slice_vals))
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    records = []

    for thr in thresholds:
        flagged = (p >= thr)
        approved = ~flagged
        fn = int(np.sum((y == 1) & approved))
        fp = int(np.sum((y == 0) & flagged))
        total_cost = fn * cost_fn + fp * cost_fp
        cost_per_app = total_cost / len(y)

        approval_rates = []
        for g in groups:
            mask = (slice_vals == g)
            if np.sum(mask) > 0:
                approval_rates.append(float(np.mean(approved[mask])))

        if approval_rates and max(approval_rates) > 0:
            min_di = min(approval_rates) / max(approval_rates)
        else:
            min_di = 1.0

        records.append({
            'threshold': float(thr),
            'cost_per_app': float(cost_per_app),
            'min_disparate_impact': float(min_di),
            'four_fifths_compliant': bool(min_di >= 0.80),
        })

    return pd.DataFrame(records)
