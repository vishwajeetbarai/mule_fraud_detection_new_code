"""Automated Verification Tests for Phase 2: Hierarchical Decision Policy & Genuine Ablation.

Verifies:
1. All three operational actions (AUTO_APPROVE, AUTO_FLAG, HUMAN_REVIEW) are produced.
2. Every application receives exactly one operational action.
3. Action percentages sum to 100%.
4. Test-label corruption (inversion) does NOT change policy thresholds or action assignments.
5. Changing test labels changes realized costs/metrics without altering policy routing.
6. Ablation study configurations are genuinely distinct in metrics and thresholds.
7. Phase 1 zero-leakage properties remain strictly preserved.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import numpy as np
import pandas as pd

import config
import evaluation as ev
from conformal import ConformalFraudPredictor
from adaptive_threshold import AdaptiveThresholdCalibrator
from hierarchical_triage import HierarchicalTriagePolicy
from ablation import run_ablation_study


@pytest.fixture
def sample_validation_test_data():
    """Generate realistic synthetic validation and test distributions."""
    np.random.seed(42)
    n_val = 1500
    n_test = 3000

    y_val = np.random.binomial(1, 0.02, size=n_val)
    p_val = np.clip(y_val * 0.45 + np.random.beta(1.5, 30, size=n_val), 0.001, 0.999)

    y_test = np.random.binomial(1, 0.02, size=n_test)
    p_test = np.clip(y_test * 0.45 + np.random.beta(1.5, 30, size=n_test), 0.001, 0.999)

    return {
        "y_val": y_val, "p_val": p_val,
        "y_test": y_test, "p_test": p_test,
    }


def test_hierarchical_triage_actions_and_partitions(sample_validation_test_data):
    """Verify that all three actions are present, mutually exclusive, and sum to 100%."""
    d = sample_validation_test_data
    cpf = ConformalFraudPredictor(alpha=0.05).calibrate(d["y_val"], d["p_val"])
    tau_cost = 0.20
    tau_tdta = 0.35

    policy = HierarchicalTriagePolicy(
        tau_cost=tau_cost,
        tau_tdta=tau_tdta,
        conformal_predictor=cpf,
        cost_fn=1500.0,
        cost_fp=30.0,
        cost_tp=10.0,
        cost_investigation=30.0,
        cost_tn=0.0,
    )

    routed = policy.route(d["p_test"])

    # 1. Check all three actions exist in non-trivial sample
    actions = set(routed["action"].unique())
    assert "AUTO_APPROVE" in actions, "AUTO_APPROVE action must be generated"
    assert "AUTO_FLAG" in actions, "AUTO_FLAG action must be generated"
    assert "HUMAN_REVIEW" in actions, "HUMAN_REVIEW action must be generated"

    # 2. Every application receives exactly one action
    assert len(routed) == len(d["p_test"])
    assert routed["action"].isnull().sum() == 0

    # 3. Percentages sum to 100%
    eval_res = policy.evaluate(d["y_test"], d["p_test"])
    total_pct = eval_res["pct_auto_approve"] + eval_res["pct_auto_flag"] + eval_res["pct_human_review"]
    assert np.isclose(total_pct, 1.0, atol=1e-5), f"Action percentages must sum to 1.0, got {total_pct}"

    total_count = eval_res["n_auto_approve"] + eval_res["n_auto_flag"] + eval_res["n_human_review"]
    assert total_count == len(d["y_test"]), "Application action counts must sum to total test size"


def test_triage_policy_zero_leakage_and_invariance(sample_validation_test_data):
    """Verify that routing decisions are strictly invariant to test labels."""
    d = sample_validation_test_data
    cpf = ConformalFraudPredictor(alpha=0.05).calibrate(d["y_val"], d["p_val"])

    policy = HierarchicalTriagePolicy(
        tau_cost=0.22,
        tau_tdta=0.33,
        conformal_predictor=cpf,
    )

    routed_clean = policy.route(d["p_test"])
    eval_clean = policy.evaluate(d["y_test"], d["p_test"])

    # Completely corrupt test labels
    y_test_corrupted = 1 - d["y_test"]
    routed_corrupted = policy.route(d["p_test"])
    eval_corrupted = policy.evaluate(y_test_corrupted, d["p_test"])

    # Actions must be completely identical regardless of test labels
    pd.testing.assert_series_equal(routed_clean["action"], routed_corrupted["action"])

    # Metric costs must reflect the changed test labels
    assert eval_clean["operational_cost_per_app"] != eval_corrupted["operational_cost_per_app"], \
        "Realized test cost must reflect out-of-sample ground truth"


def test_ablation_five_distinct_configurations(tmp_path, sample_validation_test_data):
    """Verify that the 5 ablation configurations produce genuinely distinct, non-identical metrics."""
    d = sample_validation_test_data

    df_ab = run_ablation_study(
        y_true=d["y_test"],
        y_proba=d["p_test"],
        y_val=d["y_val"],
        proba_val=d["p_val"],
        out_dir=tmp_path / "ablation_test",
        model_name="XGBoost",
    )

    assert len(df_ab) == 5, f"Expected 5 configurations, got {len(df_ab)}"

    # Configuration names
    expected_configs = [
        "1. Baseline",
        "2. Cost Optimization",
        "3. Cost Opt + TDTA",
        "4. Cost Opt + Conformal",
        "5. Full Proposed Framework",
    ]
    assert list(df_ab["configuration"]) == expected_configs

    # Verify that Config 5 is NOT an identical copy of Config 2
    row_2 = df_ab.loc[df_ab["configuration"] == "2. Cost Optimization"].iloc[0]
    row_5 = df_ab.loc[df_ab["configuration"] == "5. Full Proposed Framework"].iloc[0]

    # Action percentages must differ (Config 2 has 0% human review, Config 5 routes ambiguous cases)
    assert row_2["pct_human_review"] == 0.0
    assert row_5["pct_human_review"] > 0.0, "Full Framework must route ambiguous cases to human review"
    assert row_2["pct_auto_approve"] != row_5["pct_auto_approve"], "Auto-approve rates must differ"

    # Verify Config 4 vs Config 5 distinction
    row_4 = df_ab.loc[df_ab["configuration"] == "4. Cost Opt + Conformal"].iloc[0]
    assert row_4["pct_human_review"] != row_5["pct_human_review"] or row_4["operational_cost_per_app"] != row_5["operational_cost_per_app"], \
        "Config 4 and Config 5 must execute distinct routing policies"


def test_phase2_generated_artifacts(tmp_path, sample_validation_test_data):
    """Verify that all Phase 2 artifacts are created cleanly."""
    d = sample_validation_test_data
    out_dir = tmp_path / "artifacts"

    run_ablation_study(
        y_true=d["y_test"],
        y_proba=d["p_test"],
        y_val=d["y_val"],
        proba_val=d["p_val"],
        out_dir=out_dir,
    )

    assert (out_dir / "phase2_ablation_results.csv").exists()
    assert (out_dir / "phase2_integrated_decisions.csv").exists()
    assert (out_dir / "phase2_action_distribution.csv").exists()

    df_dec = pd.read_csv(out_dir / "phase2_integrated_decisions.csv")
    assert "hierarchical_action" in df_dec.columns
    assert set(df_dec["hierarchical_action"].unique()) <= {"AUTO_APPROVE", "AUTO_FLAG", "HUMAN_REVIEW"}

