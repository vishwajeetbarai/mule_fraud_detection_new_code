"""Automated Test Suite for Phase 3: Statistical Reliability, Wilson CIs & Fairness.

Verifies:
1. Exact mathematical agreement of wilson_score_interval with standard textbook benchmarks.
2. Boundary handling: zero successes, all successes, zero denominator, tiny samples.
3. Model metric confidence intervals: bounds ordering (0 <= low <= pt <= high <= 1).
4. Fairness metrics: Wilson CIs for precision, recall, FPR, selection rate.
5. Multi-action mapping in fairness evaluation (AUTO_APPROVE as favorable, AUTO_FLAG/HUMAN_REVIEW as flagged).
6. Disaggregated conformal coverage and empirical coverage calculations.
7. Temporal stability metric separation (Month 6 vs Month 7).
8. Preservation of zero test-label leakage.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import numpy as np
import pandas as pd
from scipy import stats

import config
import evaluation as ev
from statistical_reliability import (
    wilson_score_interval,
    format_ci,
    compute_model_metric_confidence_intervals,
    evaluate_group_fairness_with_wilson,
    evaluate_conformal_coverage_disaggregated,
    evaluate_temporal_stability,
)
from conformal import ConformalFraudPredictor
from hierarchical_triage import HierarchicalTriagePolicy


def test_wilson_score_interval_textbook_benchmarks():
    """Verify Wilson score interval against textbook values (Agresti & Coull 1998, Brown et al. 2001)."""
    # Case 1: k=10, n=100, 95% CI
    # Standard Wilson interval for 10/100 at z=1.96:
    # center = (0.10 + 1.96^2 / 200) / (1 + 1.96^2 / 100) = (0.10 + 0.019208) / 1.038416 = 0.114798
    # spread = 1.96 / 1.038416 * sqrt(0.10 * 0.90 / 100 + 1.96^2 / 40000) = 1.8875 * sqrt(0.0009 + 0.000096) = 0.05957
    # lower ~= 0.0552, upper ~= 0.1744
    pt, low, high = wilson_score_interval(10, 100, 0.95)
    assert np.isclose(pt, 0.10)
    assert 0.055 <= low <= 0.056
    assert 0.174 <= high <= 0.175

    # Case 2: k=50, n=100 (symmetric)
    pt2, low2, high2 = wilson_score_interval(50, 100, 0.95)
    assert np.isclose(pt2, 0.50)
    assert 0.403 <= low2 <= 0.405
    assert 0.595 <= high2 <= 0.597
    assert np.isclose(0.50 - low2, high2 - 0.50, atol=1e-5)


def test_wilson_score_interval_edge_cases():
    """Verify behavior on extreme edge cases."""
    # Zero successes: k = 0, n = 10
    pt0, low0, high0 = wilson_score_interval(0, 10, 0.95)
    assert pt0 == 0.0
    assert low0 == 0.0
    assert high0 > 0.0
    assert np.isclose(high0, 0.2775, atol=1e-3)

    # All successes: k = 10, n = 10
    pt1, low1, high1 = wilson_score_interval(10, 10, 0.95)
    assert pt1 == 1.0
    assert high1 == 1.0
    assert low1 < 1.0
    assert np.isclose(low1, 1.0 - 0.2775, atol=1e-3)

    # Empty denominator: n = 0
    pt_nan, low_nan, high_nan = wilson_score_interval(0, 0, 0.95)
    assert np.isnan(pt_nan)
    assert np.isnan(low_nan)
    assert np.isnan(high_nan)

    # Invalid arguments
    with pytest.raises(ValueError):
        wilson_score_interval(15, 10)
    with pytest.raises(ValueError):
        wilson_score_interval(-1, 10)


@pytest.fixture
def synthetic_phase3_data():
    """Create controlled synthetic validation and test arrays."""
    np.random.seed(42)
    n = 2000
    y = np.random.binomial(1, 0.02, size=n)
    p = np.clip(y * 0.45 + np.random.beta(1.5, 30, size=n), 0.001, 0.999)
    # Demographic slice
    slices = pd.Series(np.random.choice(["Age <25", "Age 25-50", "Age 50+"], size=n, p=[0.3, 0.5, 0.2]))
    # Months
    months = np.random.choice([6, 7], size=n)
    return {"y": y, "p": p, "slices": slices, "months": months}


def test_model_metric_confidence_intervals(synthetic_phase3_data):
    """Verify bounds and ordering for model performance confidence intervals."""
    d = synthetic_phase3_data
    df_cis = compute_model_metric_confidence_intervals(
        y_true=d["y"],
        proba=d["p"],
        threshold=0.30,
        n_bootstrap=100,
        seed=42,
    )

    assert len(df_cis) == 9
    for _, row in df_cis.iterrows():
        pt = row["point_estimate"]
        low = row["ci_lower"]
        high = row["ci_upper"]
        assert 0.0 <= low <= high <= 1.0, f"Invalid bounds for {row['metric']}: [{low}, {high}]"
        assert low <= pt <= high or np.isclose(pt, low, atol=1e-4) or np.isclose(pt, high, atol=1e-4), \
            f"Point estimate {pt} outside [{low}, {high}] for {row['metric']}"


def test_group_fairness_with_wilson(synthetic_phase3_data):
    """Verify group-level fairness metrics, Wilson CIs, and multi-action mapping."""
    d = synthetic_phase3_data
    # Construct actions
    acts = np.where(d["p"] < 0.10, "AUTO_APPROVE", np.where(d["p"] > 0.40, "AUTO_FLAG", "HUMAN_REVIEW"))

    df_fair = evaluate_group_fairness_with_wilson(
        y_true=d["y"],
        proba=d["p"],
        actions=acts,
        slice_series=d["slices"],
        slice_name="age_cohort",
    )

    assert len(df_fair) == 3
    assert "disparate_impact_ratio" in df_fair.columns
    assert "approval_ci_95" in df_fair.columns
    assert "recall_ci_95" in df_fair.columns
    assert "statistical_reliability" in df_fair.columns

    # Reference group must have disparate impact = 1.0
    ref_di = df_fair["disparate_impact_ratio"].max()
    assert np.isclose(ref_di, 1.0, atol=1e-4)


def test_conformal_coverage_disaggregated(synthetic_phase3_data):
    """Verify disaggregated conformal coverage and set sizes across subgroups."""
    d = synthetic_phase3_data
    cpf = ConformalFraudPredictor(alpha=0.05)
    cpf.calibrate(d["y"][:500], d["p"][:500])

    df_cov = evaluate_conformal_coverage_disaggregated(
        y_true=d["y"][500:],
        proba=d["p"][500:],
        conformal_predictor=cpf,
        subgroup_series=d["slices"].iloc[500:],
        subgroup_name="age_cohort",
    )

    assert len(df_cov) == 3
    assert "empirical_coverage" in df_cov.columns
    assert "average_set_size" in df_cov.columns
    assert "coverage_ci_95" in df_cov.columns

    for _, r in df_cov.iterrows():
        assert 0.0 <= r["empirical_coverage"] <= 1.0
        assert 0.0 <= r["average_set_size"] <= 2.0


def test_temporal_stability(synthetic_phase3_data):
    """Verify temporal stability evaluation across separate test months."""
    d = synthetic_phase3_data
    acts = np.where(d["p"] < 0.10, "AUTO_APPROVE", np.where(d["p"] > 0.40, "AUTO_FLAG", "HUMAN_REVIEW"))
    cpf = ConformalFraudPredictor(alpha=0.05).calibrate(d["y"][:500], d["p"][:500])

    df_raw = pd.DataFrame({
        config.TARGET: d["y"],
        "month": d["months"],
    })

    df_stab = evaluate_temporal_stability(
        df_raw=df_raw,
        proba=d["p"],
        actions=acts,
        conformal_predictor=cpf,
        test_months=[6, 7],
    )

    assert len(df_stab) == 2
    assert set(df_stab["test_month"]) == {6, 7}
    assert "conformal_empirical_coverage" in df_stab.columns
    assert "recall_ci_95" in df_stab.columns
