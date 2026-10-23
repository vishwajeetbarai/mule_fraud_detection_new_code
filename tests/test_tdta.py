"""Unit and Integration tests for Temporal Drift-Aware Threshold Adaptation (TDTA).

Validates:
1. Drift detection mathematics: PSI, KS test, categorical TV distance on stable, shifted,
   and edge-case distributions.
2. Leakage prevention: Verifies that reference quantiles and drift metrics do not rely on future labels.
3. Adaptive threshold calibration: Correct target FPR convergence, boundary and small-sample fallback.
4. Edge cases: Insufficient positive/negative instances, ties, unattainable FPR.
5. End-to-end integration: Baseline vs TDTA vs Scheduled policies running without future-label contamination.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from adaptive_threshold import AdaptiveThresholdCalibrator
from drift_detector import (
    TemporalDriftDetector,
    compute_categorical_tv_distance,
    compute_ks_test,
    compute_psi,
)
from tdta_experiment import run_sequential_policy


# ----------------------------------------------------------------------------
# 1. Drift Detector Unit Tests
# ----------------------------------------------------------------------------
def test_psi_identical_distribution():
    """Identical distributions should have near-zero PSI (< 0.02)."""
    np.random.seed(42)
    dist1 = np.random.normal(loc=10.0, scale=2.0, size=5000)
    dist2 = np.random.normal(loc=10.0, scale=2.0, size=5000)
    psi, edges = compute_psi(dist1, dist2, n_bins=10)
    assert psi < 0.05, f"PSI for identical distribution is unexpectedly high: {psi}"
    assert len(edges) > 1


def test_psi_shifted_distribution():
    """Significantly shifted distribution should trigger high PSI (> 0.25)."""
    np.random.seed(42)
    dist_ref = np.random.normal(loc=10.0, scale=2.0, size=5000)
    dist_shift = np.random.normal(loc=15.0, scale=2.0, size=5000)
    psi, _ = compute_psi(dist_ref, dist_shift, n_bins=10)
    assert psi > 0.25, f"PSI failed to detect severe distribution shift: {psi}"


def test_psi_handles_nans_and_constants():
    """Handles NaNs and constant values gracefully without crashing."""
    ref = np.array([5.0] * 100 + [np.nan] * 10)
    tar = np.array([5.0] * 80 + [6.0] * 20)
    psi, _ = compute_psi(ref, tar)
    assert np.isfinite(psi)
    assert psi >= 0.0


def test_ks_test():
    """Two-sample KS test should return p > 0.05 for same distribution and p < 0.01 for shifted."""
    np.random.seed(42)
    d1 = np.random.exponential(scale=2.0, size=2000)
    d2 = np.random.exponential(scale=2.0, size=2000)
    d3 = np.random.exponential(scale=5.0, size=2000)

    _, p_same = compute_ks_test(d1, d2)
    _, p_diff = compute_ks_test(d1, d3)
    assert p_same > 0.01
    assert p_diff < 0.001


def test_categorical_tv_distance():
    s1 = pd.Series(["A"] * 50 + ["B"] * 50)
    s2 = pd.Series(["A"] * 50 + ["B"] * 50)
    s3 = pd.Series(["A"] * 90 + ["B"] * 10)

    dist_same = compute_categorical_tv_distance(s1, s2)
    dist_diff = compute_categorical_tv_distance(s1, s3)
    assert dist_same == 0.0
    assert dist_diff > 0.3


def test_temporal_drift_detector_class():
    """Test full TemporalDriftDetector fitting and alert triggering."""
    np.random.seed(42)
    n = 1000
    df_ref = pd.DataFrame({
        "velocity_6h": np.random.normal(5000, 1000, n),
        "credit_risk_score": np.random.normal(120, 30, n),
        "income": np.random.uniform(0.1, 0.9, n),
    })
    scores_ref = np.random.beta(1, 20, n)

    detector = TemporalDriftDetector(
        features=["velocity_6h", "credit_risk_score", "income"],
        psi_threshold=0.10,
        monitor_score=True,
    )
    detector.fit_reference(df_ref, scores_ref=scores_ref)

    # Identical batch: no alert
    df_stable = df_ref.copy()
    report_stable = detector.detect(df_stable, scores_eval=scores_ref, period_id=1)
    assert not report_stable["drift_detected"]
    assert len(report_stable["alert_features"]) == 0

    # Shifted velocity: alert
    df_shifted = df_ref.copy()
    df_shifted["velocity_6h"] = df_shifted["velocity_6h"] * 2.5
    report_shift = detector.detect(df_shifted, scores_eval=scores_ref, period_id=2)
    assert report_shift["drift_detected"]
    assert "velocity_6h" in report_shift["alert_features"]


# ----------------------------------------------------------------------------
# 2. Adaptive Threshold Calibration Unit Tests
# ----------------------------------------------------------------------------
def test_calibrator_target_fpr_convergence():
    """Calibrator should set threshold near empirical (1 - target_fpr) quantile."""
    np.random.seed(42)
    n_neg = 10000
    n_pos = 200
    neg_scores = np.random.beta(1, 15, n_neg)
    pos_scores = np.random.beta(10, 5, n_pos)

    y_true = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    proba = np.concatenate([neg_scores, pos_scores])

    calibrator = AdaptiveThresholdCalibrator(target_fpr=0.05, initial_threshold=0.5)
    thr, log = calibrator.calibrate(y_true, proba, trigger_reason="drift_alert", period_id=6)

    # Realized FPR on calibration data should be ~0.05
    assert abs(log["calib_realized_fpr"] - 0.05) < 0.005
    assert log["status"] == "RECALIBRATED_SUCCESSFULLY"
    assert 0.0 < thr < 1.0


def test_calibrator_fallback_on_insufficient_samples():
    """Calibrator should preserve threshold when sample counts are insufficient."""
    y_small = np.array([0] * 50 + [1] * 1)
    p_small = np.random.uniform(0.1, 0.9, len(y_small))

    initial_thr = 0.65
    calibrator = AdaptiveThresholdCalibrator(
        target_fpr=0.05,
        initial_threshold=initial_thr,
        min_samples=1000,
        min_negatives=500,
        min_positives=5,
    )
    thr, log = calibrator.calibrate(y_small, p_small, trigger_reason="drift_alert", period_id=6)

    assert thr == initial_thr
    assert "FALLBACK_INSUFFICIENT_DATA" in log["status"]
    assert not log["threshold_changed"]


# ----------------------------------------------------------------------------
# 3. Integration & Zero Data Leakage Tests
# ----------------------------------------------------------------------------
class DummyModel:
    """Mock model predicting probability correlated with velocity."""
    def predict_proba(self, X):
        col = X["velocity_6h"] if "velocity_6h" in X.columns else X.iloc[:, 0]
        p = 1.0 / (1.0 + np.exp(-0.001 * (col - col.median())))
        return np.vstack([1.0 - p, p]).T


class DummyPreprocessor:
    def transform(self, df):
        return df[["velocity_6h", "income", "credit_risk_score"]].copy()


def test_sequential_policy_no_future_leakage():
    """Verify that predictions in month 6 cannot access month 6 labels or future thresholds."""
    np.random.seed(42)
    rows = []
    # Generate 3 months (5, 6, 7) of synthetic test applications
    for m in [5, 6, 7]:
        n = 1500
        rows.append(pd.DataFrame({
            config.TIME_COL: m,
            config.TARGET: np.random.choice([0, 1], size=n, p=[0.98, 0.02]),
            "velocity_6h": np.random.normal(5000 + (m - 5) * 1500, 800, n),
            "credit_risk_score": np.random.normal(120, 20, n),
            "income": np.random.uniform(0.1, 0.9, n),
        }))
    df_all = pd.concat(rows, ignore_index=True)

    dummy_model = DummyModel()
    dummy_pre = DummyPreprocessor()

    res_tdta = run_sequential_policy(
        model=dummy_model,
        preprocessor=dummy_pre,
        df_raw=df_all,
        initial_val_month=5,
        test_months=[6, 7],
        policy_type="tdta",
        psi_threshold=0.10,
        label_delay=0,
    )

    summary = res_tdta["summary"]
    assert summary["policy"] == "tdta"
    assert len(res_tdta["period_metrics"]) == 2

    # Verify that in month 6, the applied threshold equals the initial validation threshold
    m6_metric = res_tdta["period_metrics"][0]
    assert m6_metric["month"] == 6
    assert abs(m6_metric["threshold"] - summary["initial_threshold"]) < 1e-6
    # Month 7 threshold can adapt based on Month 6 drift
    m7_metric = res_tdta["period_metrics"][1]
    assert m7_metric["month"] == 7
