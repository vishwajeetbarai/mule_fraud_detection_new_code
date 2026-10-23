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
        "velocity_surge_intensity", "proposed_credit_to_balcon", "velocity_6h_drift_zscore"
    ]
    for ef in expected_feats:
        assert ef in X.columns, f"Missing engineered feature: {ef}"
        assert not X[ef].isna().any(), f"NaN values found in {ef}"


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

