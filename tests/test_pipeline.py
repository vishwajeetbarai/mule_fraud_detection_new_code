"""Sanity tests on a small slice of the REAL BaseDataset.csv (no synthetic data)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402
from data_loader import load_base_dataset, temporal_split  # noqa: E402
from features import FeaturePreprocessor  # noqa: E402

pytestmark = pytest.mark.skipif(not config.DATA_PATH.exists(),
                                reason="BaseDataset.csv not found")


@pytest.fixture(scope="module")
def df():
    return load_base_dataset(sample_frac=0.02)


def test_schema_and_target(df):
    assert config.TARGET in df.columns
    assert set(df[config.TARGET].unique()) <= {0, 1}


def test_temporal_split_has_no_overlap(df):
    tr, va, te = temporal_split(df)
    assert tr[config.TIME_COL].max() < va[config.TIME_COL].min()
    assert va[config.TIME_COL].max() < te[config.TIME_COL].min()


def test_no_target_or_time_leakage_in_features(df):
    tr, va, _ = temporal_split(df)
    pre = FeaturePreprocessor().fit(tr)
    X = pre.transform(va)
    assert config.TARGET not in X.columns and config.TIME_COL not in X.columns
    assert list(X.columns) == pre.feature_names_
    assert not X.isin([float("inf"), float("-inf")]).any().any()


def test_unseen_category_is_all_zero(df):
    tr, va, _ = temporal_split(df)
    pre = FeaturePreprocessor().fit(tr)
    va = va.copy()
    va["device_os"] = "brand_new_os"
    X = pre.transform(va)
    assert X[[c for c in X.columns if c.startswith("device_os_")]].sum().sum() == 0


def test_behavioral_graph_and_drift_features(df):
    tr, va, _ = temporal_split(df)
    pre = FeaturePreprocessor().fit(tr)
    X = pre.transform(va)
    # Check that behavioral graph and drift features exist
    expected_feats = [
        "device_sharing_density", "device_recycling_speed", "graph_hub_degree_proxy",
        "velocity_surge_intensity", "proposed_credit_to_balcon", "velocity_6h_drift_zscore",
        "velocity_spike_index", "velocity_momentum_spike", "fraud_ring_network_density",
        "ring_clique_density_proxy", "device_to_address_collision_ratio"
    ]
    for ef in expected_feats:
        assert ef in X.columns, f"Missing engineered feature: {ef}"
        assert not X[ef].isna().any(), f"NaN values found in {ef}"


def test_concept_drift_tracker(df):
    from drift import ConceptDriftTracker, compute_psi, compute_ks_test
    import numpy as np

    tr, va, _ = temporal_split(df)
    tracker = ConceptDriftTracker(features_to_monitor=["velocity_6h", "credit_risk_score"])
    tracker.fit(tr)
    rep = tracker.evaluate_batch(va, batch_name="val_test")
    assert not rep.empty
    assert "psi" in rep.columns
    assert "ks_statistic" in rep.columns
    assert "drift_status" in rep.columns
    assert (rep["psi"] >= 0).all()

    # Synthetic sanity test on PSI
    ref = np.random.normal(0, 1, 1000)
    shifted = np.random.normal(2, 1, 1000)
    psi_drift, _ = compute_psi(ref, shifted)
    psi_stable, _ = compute_psi(ref, ref)
    assert psi_drift > 0.25
    assert psi_stable < 0.05


def test_fairness_bias_audit(df):
    import evaluation as ev
    import numpy as np
    import pandas as pd

    y_true = np.array([0]*80 + [1]*20)
    proba = np.concatenate([np.random.uniform(0.01, 0.2, 80), np.random.uniform(0.4, 0.9, 20)])
    slices = pd.Series(["GroupA"]*50 + ["GroupB"]*50)

    report = ev.evaluate_fairness_bias_audit(y_true, proba, threshold=0.3, slice_series=slices, slice_name="test_group")
    assert not report.empty
    assert "disparate_impact_ratio" in report.columns
    assert "four_fifths_compliant" in report.columns
    assert "statistical_parity_diff" in report.columns
    assert set(report["group"].unique()) == {"GroupA", "GroupB"}


def test_delong_test_synthetic():
    import evaluation as ev
    import numpy as np
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Model A is perfect, Model B is random
    p_a = np.array([0.1, 0.2, 0.1, 0.3, 0.8, 0.9, 0.7, 0.85])
    p_b = np.array([0.5, 0.4, 0.6, 0.3, 0.5, 0.4, 0.6, 0.7])
    res = ev.delong_roc_test(y_true, p_a, p_b)
    assert res["auc_model_a"] > res["auc_model_b"]
    assert "p_value" in res
    assert 0.0 <= res["p_value"] <= 1.0


def test_cost_sensitive_threshold_optimization():
    import evaluation as ev
    import numpy as np
    y_true = np.array([0]*90 + [1]*10)
    # High fraud probabilities for true positives
    proba = np.concatenate([np.random.uniform(0.01, 0.1, 90), np.random.uniform(0.6, 0.95, 10)])
    res = ev.optimize_cost_threshold(y_true, proba, cost_fn=1500, cost_fp=30)
    assert 0.0 < res["optimal_threshold"] < 1.0
    assert res["min_cost_per_app"] <= res["default_05_cost_per_app"]

