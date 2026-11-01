import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
import numpy as np
import pandas as pd
from review3_enhancements import (
    evaluate_multi_scenario_cost,
    evaluate_capacity_constrained,
    evaluate_enhanced_fairness
)

def test_scenario_cost():
    y_val = np.array([0] * 90 + [1] * 10)
    p_val = np.concatenate([np.random.uniform(0.01, 0.2, 90), np.random.uniform(0.7, 0.99, 10)])
    y_test = np.array([0] * 180 + [1] * 20)
    p_test = np.concatenate([np.random.uniform(0.01, 0.2, 180), np.random.uniform(0.7, 0.99, 20)])
    df_cost = evaluate_multi_scenario_cost(y_val, p_val, y_test, p_test)
    assert len(df_cost) == 3
    assert 'scenario' in df_cost.columns
    assert 'val_selected_threshold' in df_cost.columns
    assert 'realized_test_cost_per_app' in df_cost.columns

def test_capacity_constrained():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.random.uniform(0.01, 0.2, 90), np.random.uniform(0.7, 0.99, 10)])
    df_cap = evaluate_capacity_constrained(y, p)
    assert len(df_cap) == 3
    assert 'investigation_capacity' in df_cap.columns

def test_enhanced_fairness():
    y = np.array([0] * 80 + [1] * 20)
    p = np.concatenate([np.random.uniform(0.01, 0.2, 80), np.random.uniform(0.6, 0.99, 20)])
    slices = pd.Series(['GroupA'] * 50 + ['GroupB'] * 50)
    df_fair = evaluate_enhanced_fairness(y, p, threshold=0.3, slice_series=slices)
    assert len(df_fair) == 2
    assert 'group_fpr' in df_fair.columns
    assert 'group_fnr' in df_fair.columns
    assert 'approval_ci_95' in df_fair.columns
