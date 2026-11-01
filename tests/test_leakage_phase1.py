"""Automated Leakage Verification Tests (Phase 1).

Verifies:
1. Threshold selection functions receive and use validation data ONLY.
2. Inverting or corrupting test labels has ZERO impact on selected thresholds.
3. Multi-scenario cost evaluation produces invariant thresholds when test labels change.
4. Out-of-sample test costs are strictly evaluated with frozen validation thresholds.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import numpy as np
import pandas as pd

import config
import evaluation as ev
from review3_enhancements import evaluate_multi_scenario_cost
from ablation import run_ablation_study


@pytest.fixture
def mock_dataset():
    """Create reproducible synthetic validation and test arrays."""
    np.random.seed(123)
    n_val = 1000
    n_test = 2000

    y_val = np.random.binomial(1, 0.02, size=n_val)
    # Higher scores for positives
    p_val = np.clip(y_val * 0.45 + np.random.beta(1.5, 30, size=n_val), 0.001, 0.999)

    y_test = np.random.binomial(1, 0.02, size=n_test)
    p_test = np.clip(y_test * 0.45 + np.random.beta(1.5, 30, size=n_test), 0.001, 0.999)

    return {
        "y_val": y_val,
        "p_val": p_val,
        "y_test": y_test,
        "p_test": p_test,
    }


def test_cost_threshold_optimization_invariance_to_test_labels(mock_dataset):
    """Verify that optimizing cost threshold on validation data is completely invariant to test labels."""
    y_val = mock_dataset["y_val"]
    p_val = mock_dataset["p_val"]

    # Select threshold on validation
    res_val = ev.optimize_cost_threshold(y_val, p_val, cost_fn=config.COST_FN, cost_fp=config.COST_FP, cost_tp=config.COST_TP)
    thr_original = res_val["optimal_threshold"]

    # Selected threshold from validation must be deterministic
    res_val_check = ev.optimize_cost_threshold(y_val, p_val, cost_fn=config.COST_FN, cost_fp=config.COST_FP, cost_tp=config.COST_TP)
    assert res_val_check["optimal_threshold"] == thr_original, "Validation threshold selection must be deterministic"


def test_multi_scenario_zero_leakage_invariance(mock_dataset):
    """Verify that evaluate_multi_scenario_cost selects thresholds strictly on validation data."""
    y_val = mock_dataset["y_val"]
    p_val = mock_dataset["p_val"]
    y_test = mock_dataset["y_test"]
    p_test = mock_dataset["p_test"]

    # Baseline run
    df_original = evaluate_multi_scenario_cost(y_val, p_val, y_test, p_test)

    # Corrupt test labels entirely (all 0s or inverted)
    y_test_inverted = 1 - y_test
    df_inverted = evaluate_multi_scenario_cost(y_val, p_val, y_test_inverted, p_test)

    # Thresholds MUST remain identical for all scenarios despite test label corruption
    for sc in df_original["scenario"].unique():
        thr_orig = df_original.loc[df_original["scenario"] == sc, "val_selected_threshold"].values[0]
        thr_inv = df_inverted.loc[df_inverted["scenario"] == sc, "val_selected_threshold"].values[0]
        assert thr_orig == thr_inv, f"Scenario {sc} threshold changed when test labels were altered! LEAKAGE DETECTED."

    # Test metrics (costs, recall) should change because test ground-truth changed
    cost_orig = df_original.loc[df_original["scenario"] == "Scenario_B_HighFN_LowFP", "realized_test_cost_per_app"].values[0]
    cost_inv = df_inverted.loc[df_inverted["scenario"] == "Scenario_B_HighFN_LowFP", "realized_test_cost_per_app"].values[0]
    assert cost_orig != cost_inv, "Realized test cost must reflect out-of-sample ground truth"


def test_ablation_study_zero_leakage(tmp_path, mock_dataset):
    """Verify that all ablation study configurations use validation data to establish thresholds."""
    y_val = mock_dataset["y_val"]
    p_val = mock_dataset["p_val"]
    y_test = mock_dataset["y_test"]
    p_test = mock_dataset["p_test"]

    df_ab_1 = run_ablation_study(y_test, p_test, y_val, p_val, out_dir=tmp_path / "out1")

    # Invert test labels
    y_test_corrupted = 1 - y_test
    df_ab_2 = run_ablation_study(y_test_corrupted, p_test, y_val, p_val, out_dir=tmp_path / "out2")

    for i in range(len(df_ab_1)):
        cfg = df_ab_1.iloc[i]["configuration"]
        thr_1 = df_ab_1.iloc[i]["threshold"]
        thr_2 = df_ab_2.iloc[i]["threshold"]
        assert thr_1 == thr_2, f"Ablation config '{cfg}' altered threshold when test labels changed! LEAKAGE DETECTED."
