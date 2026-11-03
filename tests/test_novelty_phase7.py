"""Unit tests for Phase 7 mathematical novelty modules:
1. Dynamic Bipartite Entity Graph & Spectral Dirichlet Energy.
2. Multi-Manifold Sliced Wasserstein Distance Drift Detector.
3. Mondrian (Group-Conditional) Conformal Predictor.
"""
import numpy as np
import pandas as pd
import pytest

from src.graph_topology import DynamicBipartiteEntityGraph
from src.wasserstein_drift import SlicedWassersteinDriftDetector
from src.conformal import ConformalFraudPredictor, MondrianConformalPredictor


def test_dynamic_bipartite_graph_temporal_invariants():
    """Verify that graph invariants compute correctly without NaN or negative values."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "month": np.random.choice([0, 1, 2], size=n),
        "device_os": np.random.choice(["windows", "macintosh", "linux", "other"], size=n),
        "device_distinct_emails_8w": np.random.randint(0, 5, size=n),
        "zip_count_4w": np.random.randint(100, 2000, size=n),
        "bank_branch_count_8w": np.random.randint(0, 20, size=n),
        "date_of_birth_distinct_emails_4w": np.random.randint(0, 3, size=n),
        "customer_age": np.random.randint(18, 80, size=n),
        "email_is_free": np.random.choice([0, 1], size=n),
        "phone_home_valid": np.random.choice([0, 1], size=n),
    })
    labels = (np.random.rand(n) < 0.1).astype(int)

    graph = DynamicBipartiteEntityGraph(decay_half_life_days=14.0)
    top_df = graph.compute_topological_invariants(df, fraud_labels=labels, update_graph=True)

    assert len(top_df) == n
    assert "spectral_dirichlet_energy" in top_df.columns
    assert "bipartite_clustering_coefficient" in top_df.columns
    assert "entity_collision_burstiness" in top_df.columns
    assert "harmonic_mule_density" in top_df.columns

    # Check validity of numeric ranges
    assert not top_df.isna().any().any()
    assert (top_df["spectral_dirichlet_energy"] >= 0.0).all()
    assert (top_df["harmonic_mule_density"] >= 0.0).all()
    assert (top_df["harmonic_mule_density"] <= 1.0).all()


def test_sliced_wasserstein_drift_detector():
    """Verify that SWD detects multivariate distribution shift reliably."""
    np.random.seed(42)
    n = 500
    d = 4
    # Reference distribution ~ N(0, I)
    X_ref = np.random.randn(n, d).astype(np.float32)
    
    # In-distribution sample ~ N(0, I)
    X_in = np.random.randn(200, d).astype(np.float32)
    
    # Shifted distribution ~ N(1.5, 2.0*I)
    X_shifted = (np.random.randn(200, d) * 2.0 + 1.5).astype(np.float32)

    detector = SlicedWassersteinDriftDetector(n_projections=32, threshold_factor=2.0, random_state=42)
    detector.fit(X_ref)

    res_in = detector.compute_drift(X_in)
    res_shifted = detector.compute_drift(X_shifted)

    assert "swd_distance" in res_in
    assert res_shifted["swd_distance"] > res_in["swd_distance"]
    assert res_shifted["drift_detected"] is True


def test_mondrian_conformal_group_coverage():
    """Verify that Mondrian conformal prediction guarantees group-conditional coverage."""
    np.random.seed(42)
    n = 1000
    groups = np.random.choice(["Young", "Middle", "Senior"], size=n)
    
    # Generate simulated true labels and probabilities
    y_true = (np.random.rand(n) < 0.1).astype(int)
    # Calibrated probabilities with mild noise
    proba = np.where(y_true == 1, np.random.beta(5, 2, size=n), np.random.beta(1, 9, size=n))

    # Split into cal (500) and test (500)
    y_cal, y_te = y_true[:500], y_true[500:]
    p_cal, p_te = proba[:500], proba[500:]
    g_cal, g_te = groups[:500], groups[500:]

    mondrian = MondrianConformalPredictor(alpha=0.05)
    mondrian.calibrate(y_cal, p_cal, g_cal)

    cov_df = mondrian.evaluate_group_coverage(y_te, p_te, g_te)
    
    assert len(cov_df) == 3
    for _, row in cov_df.iterrows():
        # Empirical coverage should be close to 95% (+- tolerance for n=500 finite sample)
        assert row["empirical_coverage"] >= 0.88, f"Coverage too low for group {row['group']}"

