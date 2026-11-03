"""Phase 7 High-Novelty Benchmark Execution & Empirical Verification.

Executes the elevated mathematical components on the benchmark dataset:
1. Dynamic Bipartite Entity Graph with Spectral Harmonic Dirichlet Energy.
2. Multi-Manifold Sliced Wasserstein Distance Drift Detection across temporal horizons.
3. Mondrian (Group-Conditional) Conformal Prediction across protected age cohorts.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

import config
from data_loader import load_base_dataset, temporal_split
from features import FeaturePreprocessor
from graph_topology import DynamicBipartiteEntityGraph
from wasserstein_drift import SlicedWassersteinDriftDetector
from conformal import ConformalFraudPredictor, MondrianConformalPredictor


def run_phase7_benchmark():
    print("=" * 80)
    print("PHASE 7 BENCHMARK: ELEVATED GRAPH, OPTIMAL TRANSPORT & MONDRIAN CONFORMAL")
    print("=" * 80)

    # 1. Load Data
    t0 = time.time()
    # Sample fraction 0.1 for rapid high-novelty validation benchmark
    df = load_base_dataset(sample_frac=0.1, random_state=42)
    print(f"Loaded benchmark slice: {len(df):,} records in {time.time() - t0:.1f}s")

    # 2. Extract Dynamic Bipartite Graph Topological Invariants
    print("\n--- [Step 1] Constructing Dynamic Bipartite Entity Graph & Dirichlet Invariants ---")
    t_graph = time.time()
    graph = DynamicBipartiteEntityGraph(decay_half_life_days=14.0)
    
    # Process graph along chronological order
    df = df.sort_values(by=["month"]).reset_index(drop=True)
    graph_feats = graph.compute_topological_invariants(
        df,
        fraud_labels=df["fraud_bool"].values,
        update_graph=True
    )
    print(f"Computed spectral graph invariants in {time.time() - t_graph:.1f}s")
    for col in graph_feats.columns:
        df[col] = graph_feats[col].values

    # 3. Temporal Split
    df_train, df_val, df_test = temporal_split(df)
    print(f"Splits: Train (M0-4)={len(df_train):,}, Val (M5)={len(df_val):,}, Test (M6-7)={len(df_test):,}")

    # 4. Feature Preprocessing
    pre = FeaturePreprocessor()
    X_tr = pre.fit_transform(df_train)
    y_tr = df_train["fraud_bool"].values.astype(int)

    X_va = pre.transform(df_val)
    y_va = df_val["fraud_bool"].values.astype(int)

    X_te = pre.transform(df_test)
    y_te = df_test["fraud_bool"].values.astype(int)

    # 5. Train LightGBM Model
    print("\n--- [Step 2] Training Classifier with Spectral Invariants ---")
    spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    clf = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_tr, y_tr)
    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]

    # 6. Sliced Wasserstein Distance Drift Engine
    print("\n--- [Step 3] Multi-Manifold Optimal Transport (SWD) Drift Detection ---")
    drift_detector = SlicedWassersteinDriftDetector(n_projections=64, threshold_factor=2.0, random_state=42)
    # Fit reference manifold on historical training features
    drift_detector.fit(X_tr)

    drift_val = drift_detector.compute_drift(X_va)
    drift_test = drift_detector.compute_drift(X_te)
    print(f"SWD Validation Distance : {drift_val['swd_distance']:.4f} (Threshold: {drift_val['threshold']:.4f}, Drift: {drift_val['drift_detected']})")
    print(f"SWD Test Horizon Distance: {drift_test['swd_distance']:.4f} (Threshold: {drift_test['threshold']:.4f}, Drift: {drift_test['drift_detected']})")

    # 7. Mondrian (Group-Conditional) Conformal Prediction
    print("\n--- [Step 4] Mondrian Group-Conditional Conformal Fairness ---")
    # Define age cohorts
    def get_cohort(age):
        if age < 25: return "<25"
        elif age <= 35: return "25-35"
        elif age <= 50: return "35-50"
        elif age <= 65: return "50-65"
        else: return "65+"

    g_va = df_val["customer_age"].apply(get_cohort).values
    g_te = df_test["customer_age"].apply(get_cohort).values

    # Marginal baseline conformal
    marginal_conf = ConformalFraudPredictor(alpha=0.05)
    marginal_conf.calibrate(y_va, p_va)
    marginal_metrics = marginal_conf.evaluate_coverage(y_te, p_te)

    # Mondrian group-conditional conformal
    mondrian_conf = MondrianConformalPredictor(alpha=0.05)
    mondrian_conf.calibrate(y_va, p_va, g_va)
    mondrian_eval = mondrian_conf.evaluate_group_coverage(y_te, p_te, g_te)

    print(f"Marginal Overall Coverage: {marginal_metrics['empirical_coverage']:.4f} (Avg Set Size: {marginal_metrics['avg_set_size']:.3f})")
    print("\nMondrian Disaggregated Group Coverage:")
    print(mondrian_eval.to_string(index=False))

    # 8. Save Phase 7 Results
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    mondrian_eval.to_csv(out_dir / "phase7_mondrian_fairness_coverage.csv", index=False)

    swd_summary = pd.DataFrame([
        {"horizon": "Validation (M5)", **drift_val},
        {"horizon": "Test Out-of-Sample (M6-7)", **drift_test},
    ])
    swd_summary.to_csv(out_dir / "phase7_wasserstein_drift_summary.csv", index=False)

    print("\nPhase 7 outputs successfully exported to outputs/phase7_*")
    return mondrian_eval, swd_summary


if __name__ == "__main__":
    run_phase7_benchmark()
