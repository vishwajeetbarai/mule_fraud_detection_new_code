"""End-to-end research benchmark training and evaluation for Mule Account Fraud Detection.

Elevates the baseline pipeline with:
1. Advanced behavioral graph metrics & temporal drift handling (via features.py).
2. Focal Loss & cost-sensitive threshold tuning (minimizing dollar loss).
3. Leakage-safe in-fold resampling (SMOTE / hybrid) inside temporal splits.
4. Rigorous academic evaluation (DeLong's test, paired CV t-tests/Wilcoxon).
5. Adversarial robustness stress-testing under evasion noise.
6. Regulatory XAI compliance audits (RBI / Basel III alignment).

Usage:
    python src/train.py
    python src/train.py --loss focal --resample smote
    python src/train.py --sample-frac 0.1 --skip-cv --skip-shap  # rapid smoke test
"""
import argparse
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import config
import evaluation as ev
from data_loader import describe, load_base_dataset, temporal_split
from features import FeaturePreprocessor
from drift import ConceptDriftTracker
from conformal import ConformalFraudPredictor
from pareto_fairness import compute_fairness_cost_pareto
from review3_enhancements import (
    evaluate_multi_scenario_cost,
    evaluate_capacity_constrained,
    evaluate_enhanced_fairness,
)
from delayed_feedback import simulate_delayed_feedback, get_training_slice_under_delay
from ablation import run_ablation_study


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
warnings.filterwarnings("ignore", message=".*NumPy global RNG.*")


# ----------------------------------------------------------------------------
# 1. Focal Loss & Objectives
# ----------------------------------------------------------------------------
def _spw(y):
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def focal_loss_objective_lgb(preds, train_data, gamma=config.FOCAL_GAMMA, alpha=config.FOCAL_ALPHA):
    """Numerically stable binary focal loss gradient and hessian for LightGBM."""
    labels = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    p = np.clip(p, 1e-15, 1.0 - 1e-15)

    grad = np.where(
        labels == 1,
        alpha * (1.0 - p)**gamma * (gamma * p * np.log(p) + p - 1.0),
        (1.0 - alpha) * p**gamma * (p - gamma * (1.0 - p) * np.log(1.0 - p))
    )
    hess = np.where(
        labels == 1,
        alpha * (1.0 - p)**gamma * p * (1.0 - p) * (1.0 + gamma),
        (1.0 - alpha) * p**gamma * p * (1.0 - p) * (1.0 + gamma)
    )
    hess = np.clip(hess, 1e-6, None)
    return grad, hess


# ----------------------------------------------------------------------------
# 2. In-Fold Resampling (Zero Leakage)
# ----------------------------------------------------------------------------
def apply_resampling(X, y, strategy="none", target_ratio=config.RESAMPLING_RATIO, seed=42):
    """Resample minority fraud class strictly inside inner training folds."""
    if strategy in ("none", None):
        return X, y

    # Try imbalanced-learn if installed
    try:
        if strategy == "smote":
            from imblearn.over_sampling import SMOTE
            return SMOTE(sampling_strategy=target_ratio, random_state=seed).fit_resample(X, y)
        elif strategy == "random_under":
            from imblearn.under_sampling import RandomUnderSampler
            return RandomUnderSampler(sampling_strategy=target_ratio, random_state=seed).fit_resample(X, y)
    except ImportError:
        pass

    # Pure NumPy/scikit-learn SMOTE fallback (no imblearn required)
    if strategy in ("smote", "hybrid"):
        pos_mask = (y == 1)
        X_pos = X[pos_mask]
        n_pos = len(X_pos)
        n_neg = int(np.sum(y == 0))
        target_n_pos = int(n_neg * target_ratio)
        n_synth = max(0, target_n_pos - n_pos)

        if n_synth > 0 and n_pos >= 5:
            from sklearn.neighbors import NearestNeighbors
            knn = NearestNeighbors(n_neighbors=min(6, n_pos)).fit(X_pos)
            _, indices = knn.kneighbors(X_pos)
            rng = np.random.RandomState(seed)
            synth_list = []
            for _ in range(n_synth):
                idx = rng.randint(0, n_pos)
                nn_idx = rng.choice(indices[idx, 1:])
                diff = X_pos.iloc[nn_idx].values - X_pos.iloc[idx].values
                gap = rng.uniform(0, 1)
                synth_list.append(X_pos.iloc[idx].values + gap * diff)

            synth_df = pd.DataFrame(synth_list, columns=X.columns)
            X_res = pd.concat([X, synth_df], ignore_index=True)
            y_res = np.concatenate([y, np.ones(n_synth, dtype=y.dtype)])
            return X_res, y_res

    return X, y


# ----------------------------------------------------------------------------
# 3. Model Fitting
# ----------------------------------------------------------------------------
def fit_xgboost(X_tr, y_tr, X_va, y_va, seed, loss_type="weighted"):
    params = dict(config.XGB_PARAMS)
    if loss_type == "weighted":
        params["scale_pos_weight"] = _spw(y_tr)

    model = XGBClassifier(**params, random_state=seed, n_jobs=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def fit_lightgbm(X_tr, y_tr, X_va, y_va, seed, loss_type="weighted"):
    params = dict(config.LGB_PARAMS)
    if loss_type == "weighted":
        params["scale_pos_weight"] = _spw(y_tr)
        fobj = None
    elif loss_type == "focal":
        fobj = focal_loss_objective_lgb

    model = lgb.LGBMClassifier(**params, random_state=seed, n_jobs=-1)
    model.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(config.LGB_EARLY_STOPPING_ROUNDS,
                                      first_metric_only=True, verbose=False)],
    )
    return model


def best_iterations(name, model):
    if name == "XGBoost":
        return int(model.best_iteration) + 1
    return int(model.best_iteration_)


# ----------------------------------------------------------------------------
# 4. Temporal Cross-Validation with Academic Significance Testing
# ----------------------------------------------------------------------------
def temporal_cv(df, seed, loss_type="weighted", resample_strategy="none"):
    fitters = {
        "XGBoost": lambda X_t, y_t, X_v, y_v: fit_xgboost(X_t, y_t, X_v, y_v, seed, loss_type),
        "LightGBM": lambda X_t, y_t, X_v, y_v: fit_lightgbm(X_t, y_t, X_v, y_v, seed, loss_type),
    }

    rows = []
    for test_m in config.CV_TEST_MONTHS:
        val_m = test_m - 1
        tr = df[df[config.TIME_COL] < val_m]
        va = df[df[config.TIME_COL] == val_m]
        te = df[df[config.TIME_COL] == test_m]
        if tr.empty or va.empty or te.empty:
            continue

        pre = FeaturePreprocessor().fit(tr)
        X_tr, X_va, X_te = (pre.transform(d) for d in (tr, va, te))
        y_tr, y_va, y_te = (d[config.TARGET].values for d in (tr, va, te))

        # Apply in-fold resampling strictly on training data
        if resample_strategy != "none":
            X_tr, y_tr = apply_resampling(X_tr, y_tr, strategy=resample_strategy, seed=seed)

        for name, fitter in fitters.items():
            m = fitter(X_tr, y_tr, X_va, y_va)
            thr = ev.threshold_for_fpr(y_va, m.predict_proba(X_va)[:, 1], config.TARGET_FPR)
            met = ev.compute_metrics(y_te, m.predict_proba(X_te)[:, 1], thr)
            rows.append({
                "fold_test_month": test_m, "train_months": f"0-{val_m - 1}",
                "model": name, "n_train": len(X_tr), "n_test": len(te), **met
            })
            print(f"  [CV] test month {test_m} | {name:8s} | "
                  f"AUC={met['ROC_AUC']:.4f} PR-AUC={met['PR_AUC']:.4f} "
                  f"TPR@5%FPR={met['TPR_at_5pct_FPR']:.4f}", flush=True)

    cv_df = pd.DataFrame(rows)
    return cv_df


# ----------------------------------------------------------------------------
# 5. Explainability & Regulatory Compliance
# ----------------------------------------------------------------------------
def gain_importance(model, feature_names):
    if isinstance(model, XGBClassifier):
        score = model.get_booster().get_score(importance_type="gain")
        vals = [score.get(f, 0.0) for f in feature_names]
    else:
        vals = model.booster_.feature_importance(importance_type="gain")
    out = pd.DataFrame({"feature": feature_names, "gain": np.asarray(vals, dtype=float)})
    out["gain_share"] = out["gain"] / max(out["gain"].sum(), 1e-12)
    return out.sort_values("gain", ascending=False).reset_index(drop=True)


def run_shap_and_regulatory_audit(model, X_test, y_test, out_dir, seed, threshold):
    import matplotlib.pyplot as plt
    import shap

    n = min(config.SHAP_SAMPLE_SIZE, len(X_test))
    sample = X_test.sample(n=n, random_state=seed)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]

    # Global SHAP importance
    shap_df = pd.DataFrame({
        "feature": sample.columns,
        "mean_abs_shap": np.abs(sv).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(out_dir / "shap_feature_importance_lgbm.csv", index=False)

    # 1. Global Bar & Beeswarm Plots
    shap.summary_plot(sv, sample, plot_type="bar", max_display=20, show=False)
    plt.tight_layout(); plt.savefig(out_dir / "shap_bar_lgbm.png", dpi=150); plt.close()
    shap.summary_plot(sv, sample, max_display=20, show=False)
    plt.tight_layout(); plt.savefig(out_dir / "shap_beeswarm_lgbm.png", dpi=150); plt.close()

    # 2. Global Feature Interaction Plot (top 2 interacting features)
    top_feature_1 = shap_df.iloc[0]["feature"]
    top_feature_2 = shap_df.iloc[1]["feature"]
    try:
        plt.figure(figsize=(7.5, 5))
        shap.dependence_plot(top_feature_1, sv, sample, interaction_index=top_feature_2, show=False)
        plt.title(f"SHAP Feature Interaction: {top_feature_1} vs {top_feature_2}")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_interaction_top_pair.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"SHAP interaction plot skipped: {e}")

    # 3. Regulatory Compliance Individual Audit Notice (RBI / Basel III alignment)
    # Find high-confidence positive mule fraud cases
    probas = model.predict_proba(sample)[:, 1]
    fraud_indices = np.where((probas >= threshold) & (y_test.iloc[sample.index] == 1))[0]
    if len(fraud_indices) > 0:
        target_idx = int(fraud_indices[0])
        ev.generate_regulatory_audit_card(
            shap_values=sv[target_idx],
            base_value=float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value),
            feature_names=sample.columns,
            instance_series=sample.iloc[target_idx],
            y_true_val=1,
            predicted_proba=float(probas[target_idx]),
            threshold=threshold,
            output_path=out_dir / "regulatory_compliance_audit_case.png",
            case_id=int(sample.index[target_idx]),
        )
        print("  Generated Regulatory Adverse Action Audit Notice -> regulatory_compliance_audit_case.png")


# ----------------------------------------------------------------------------
# 6. Main Orchestrator
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=None, help="Path to BaseDataset.csv")
    ap.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    ap.add_argument("--sample-frac", type=float, default=None,
                    help="Use a random fraction of the REAL rows (smoke tests only)")
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    ap.add_argument("--loss", choices=["weighted", "focal"], default="weighted",
                    help="Loss function: weighted cross-entropy or focal loss")
    ap.add_argument("--resample", choices=["none", "smote", "random_under", "hybrid"],
                    default=config.RESAMPLING_STRATEGY, help="In-fold resampling strategy")
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--skip-stress", action="store_true", help="Skip adversarial stress testing")
    ap.add_argument("--run-tdta", action="store_true", help="Run Temporal Drift-Aware Threshold Adaptation (TDTA) experiments")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    t0 = time.time()

    print("=" * 70, "\nRESEARCH-BENCHMARK: MULE ACCOUNT FRAUD DETECTION (NeurIPS BAF)\n" + "=" * 70)
    print(f"Loss Function: {args.loss} | Resampling Strategy: {args.resample}")
    df = load_base_dataset(args.data, args.sample_frac, args.seed)
    train, valid, test = temporal_split(df)
    print(describe(df, "all"), "\n" + describe(train, "train"),
          f"(months {config.TRAIN_MONTHS})\n" + describe(valid, "valid"),
          f"(months {config.VALID_MONTHS})\n" + describe(test, "test"),
          f"(months {config.TEST_MONTHS})")

    # Feature engineering with behavioral graph metrics & rolling drift baselines
    pre = FeaturePreprocessor().fit(train)
    X_tr, X_va, X_te = (pre.transform(d) for d in (train, valid, test))
    y_tr, y_va, y_te = (d[config.TARGET].values for d in (train, valid, test))

    print(f"Engineered Features: {X_tr.shape[1]} (including behavioral graph & drift baselines)")

    # In-fold resampling on training data only
    if args.resample != "none":
        print(f"Applying {args.resample} resampling to training split only (zero leakage)...")
        X_tr, y_tr = apply_resampling(X_tr, y_tr, strategy=args.resample, seed=args.seed)
        print(f"Resampled training rows: {len(X_tr):,} | fraud: {int(np.sum(y_tr)):,} ({np.mean(y_tr):.2%})")

    # Fit models
    fitters = {
        "XGBoost": lambda X_t, y_t, X_v, y_v: fit_xgboost(X_t, y_t, X_v, y_v, args.seed, args.loss),
        "LightGBM": lambda X_t, y_t, X_v, y_v: fit_lightgbm(X_t, y_t, X_v, y_v, args.seed, args.loss),
    }

    models, probas, rows = {}, {}, []
    val_fpr_thresholds = {}

    for name, fitter in fitters.items():
        t = time.time()
        m = fitter(X_tr, y_tr, X_va, y_va)
        p_va, p_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
        thr = ev.threshold_for_fpr(y_va, p_va, config.TARGET_FPR)
        val_fpr_thresholds[name] = thr
        met = ev.compute_metrics(y_te, p_te, thr)
        rows.append({"model": name, "best_iteration": best_iterations(name, m), **met})
        models[name], probas[name] = m, p_te
        print(f"[{name}] best_iter={rows[-1]['best_iteration']} | "
              f"AUC={met['ROC_AUC']:.4f} PR-AUC={met['PR_AUC']:.4f} "
              f"TPR@1%FPR={met['TPR_at_1pct_FPR']:.4f} TPR@5%FPR={met['TPR_at_5pct_FPR']:.4f} "
              f"| @valid-thr({thr:.4f}): P={met['Precision']:.3f} R={met['Recall']:.3f} "
              f"FPR={met['realized_FPR']:.3f} ({time.time() - t:.0f}s)", flush=True)
        joblib.dump(m, out_dir / "models" / f"{name.lower()}.joblib")
    joblib.dump(pre, out_dir / "models" / "preprocessor.joblib")

    pd.DataFrame(rows).to_csv(out_dir / "model_metrics_table.csv", index=False)

    # ------------------------------------------------------------------------
    # Statistical Significance Testing: DeLong's Test (XGBoost vs LightGBM)
    # ------------------------------------------------------------------------
    print("\n[Statistical Significance Testing]")
    delong_res = ev.delong_roc_test(y_te, probas["LightGBM"], probas["XGBoost"])
    delong_df = pd.DataFrame([{
        "comparison": "LightGBM vs XGBoost",
        "LightGBM_AUC": delong_res["auc_model_a"],
        "XGBoost_AUC": delong_res["auc_model_b"],
        "AUC_Difference": delong_res["auc_difference"],
        "Standard_Error": delong_res["se_difference"],
        "Z_Statistic": delong_res["z_statistic"],
        "P_Value": delong_res["p_value"],
        "Significant_at_p05": delong_res["statistically_significant_05"],
        "CI_95_Lower": delong_res["ci_95"][0],
        "CI_95_Upper": delong_res["ci_95"][1],
    }])
    delong_df.to_csv(out_dir / "delong_roc_test_results.csv", index=False)
    print(f"  DeLong's Test: Delta_AUC={delong_res['auc_difference']:+.4f} | "
          f"z={delong_res['z_statistic']:.3f} | p={delong_res['p_value']:.4e} "
          f"(Significant: {delong_res['statistically_significant_05']})")

    # ------------------------------------------------------------------------
    # Cost-Sensitive Threshold Optimization (Zero Test-Label Leakage)
    # Thresholds are selected on Validation Month 5 and frozen out-of-sample
    # ------------------------------------------------------------------------
    print("\n[Cost-Sensitive Optimization & Operational Loss Matrix]")
    cost_rows = []
    frozen_cost_thresholds = {}
    for name in models:
        p_v = models[name].predict_proba(X_va)[:, 1]
        p_t = probas[name]
        c_val = ev.optimize_cost_threshold(y_va, p_v, cost_fn=config.COST_FN, cost_fp=config.COST_FP, cost_tp=config.COST_TP)
        thr_frozen = float(c_val["optimal_threshold"])
        frozen_cost_thresholds[name] = thr_frozen

        # Realized performance on out-of-sample test set (Months 6-7)
        pred_t = (p_t >= thr_frozen).astype(int)
        c_test = ev.compute_cost_matrix(y_te, pred_t, cost_fn=config.COST_FN, cost_fp=config.COST_FP, cost_tp=config.COST_TP)
        def_test = ev.compute_cost_matrix(y_te, (p_t >= 0.5).astype(int), cost_fn=config.COST_FN, cost_fp=config.COST_FP, cost_tp=config.COST_TP)
        fpr_test = ev.compute_cost_matrix(y_te, (p_t >= val_fpr_thresholds[name]).astype(int), cost_fn=config.COST_FN, cost_fp=config.COST_FP, cost_tp=config.COST_TP)

        realized_cost = c_test["cost_per_application"]
        default_cost = def_test["cost_per_application"]
        savings_vs_default = default_cost - realized_cost
        savings_pct = (savings_vs_default / default_cost) * 100.0 if default_cost > 0 else 0.0

        tn_t, fp_t, fn_t, tp_t = c_test["TN"], c_test["FP"], c_test["FN"], c_test["TP"]
        prec_t = tp_t / max(tp_t + fp_t, 1)
        rec_t = tp_t / max(tp_t + fn_t, 1)

        cost_rows.append({
            "model": name,
            "val_selected_threshold": thr_frozen,
            "val_expected_cost_per_app": c_val["min_cost_per_app"],
            "realized_test_cost_per_app": realized_cost,
            "default_05_cost_per_app": default_cost,
            "savings_vs_default_per_app": savings_vs_default,
            "savings_pct_vs_default": savings_pct,
            "fpr_5pct_cost_per_app": fpr_test["cost_per_application"],
            "precision_at_optimal": prec_t,
            "recall_at_optimal": rec_t,
            "test_tp": tp_t,
            "test_fp": fp_t,
            "test_fn": fn_t,
            "test_tn": tn_t,
        })
        print(f"  [{name}] Val-Selected Thr={thr_frozen:.3f} | Realized Test Cost=${realized_cost:.2f}/app "
              f"(Saves {savings_pct:.1f}% vs default 0.5 threshold)")

    pd.DataFrame(cost_rows).to_csv(out_dir / "cost_optimization_results.csv", index=False)
    ev.plot_cost_curve(y_te, probas, config.COST_FN, config.COST_FP, out_dir / "cost_optimization_curve.png",
                       frozen_thresholds=frozen_cost_thresholds)
    best = max(rows, key=lambda r: r["PR_AUC"])["model"]
    # Multi-Scenario Cost Matrix Analysis & Investigation Capacity (Zero Leakage)
    proba_val_best = models[best].predict_proba(X_va)[:, 1]
    df_scenarios = evaluate_multi_scenario_cost(
        y_val=y_va, proba_val=proba_val_best, y_test=y_te, proba_test=probas[best]
    )
    df_scenarios.to_csv(out_dir / "cost_scenarios_multi.csv", index=False)
    df_capacity = evaluate_capacity_constrained(y_te, probas[best])
    df_capacity.to_csv(out_dir / "investigation_capacity_table.csv", index=False)
    print("  Fixed Capacity Triage:")
    for _, c_row in df_capacity.iterrows():
        print(f"    {c_row['investigation_capacity']}: Precision={c_row['precision']:.1%}, Recall={c_row['recall']:.1%}")

    # Standard plots
    ev.plot_roc(y_te, probas, out_dir / "roc_curve.png")
    ev.plot_pr(y_te, probas, out_dir / "precision_recall_curve.png")
    ev.plot_score_distribution(y_te, probas[best], best, out_dir / "fraud_probability_distribution.png")

    for name in models:
        imp = gain_importance(models[name], list(X_tr.columns))
        imp.to_csv(out_dir / f"feature_importance_{name.lower()}.csv", index=False)
    ev.plot_importance(pd.read_csv(out_dir / "feature_importance_lightgbm.csv"), "gain",
                       "LightGBM Feature Importance (Gain)",
                       out_dir / "lightgbm_feature_importance.png")

    # ------------------------------------------------------------------------
    # Concept Drift Detection Audit (Training vs Validation / Test Batches)
    # ------------------------------------------------------------------------
    print("\n[Concept Drift Detection: Population Stability Index & Feature Shift]")
    drift_tracker = ConceptDriftTracker(
        features_to_monitor=config.DRIFT_MONITORING_FEATURES,
        psi_threshold=config.DRIFT_PSI_THRESHOLD,
        alpha_pvalue=config.DRIFT_MAX_PVALUE_ALPHA,
    )
    # Fit reference on training split with model probability baseline
    p_tr_best = models[best].predict_proba(X_tr)[:, 1]
    drift_tracker.fit(reference_df=train, reference_scores=p_tr_best)

    # Evaluate validation month drift
    drift_val_rep = drift_tracker.evaluate_batch(
        target_df=valid,
        target_scores=models[best].predict_proba(X_va)[:, 1],
        batch_name=f"Validation_Month_{config.VALID_MONTHS[0]}",
    )
    # Evaluate temporal test months drift
    drift_test_rep = drift_tracker.evaluate_batch(
        target_df=test,
        target_scores=probas[best],
        batch_name=f"Test_Months_{config.TEST_MONTHS[0]}_{config.TEST_MONTHS[1]}",
    )
    drift_summary = pd.concat([drift_val_rep, drift_test_rep], ignore_index=True)
    drift_summary.to_csv(out_dir / "concept_drift_audit_summary.csv", index=False)
    print(f"  Concept Drift Audit saved -> concept_drift_audit_summary.csv")
    for _, d_row in drift_test_rep.iterrows():
        print(f"    Feature: {d_row['feature']:26s} | PSI: {d_row['psi']:.4f} "
              f"| KS Stat: {d_row['ks_statistic']:.4f} (p={d_row['ks_pvalue']:.3e}) "
              f"| Status: {d_row['drift_status']}")

    # ------------------------------------------------------------------------
    # Model Fairness & Demographic Bias Audit (Disparate Impact & Parity)
    # ------------------------------------------------------------------------
    print("\n[Model Fairness & Demographic Bias Audit]")
    best_thr = val_fpr_thresholds[best]
    # 1. Customer Age Bins
    if "customer_age" in test.columns:
        age_bins = pd.cut(
            test["customer_age"],
            bins=[0, 25, 35, 50, 65, 120],
            labels=["<25", "25-35", "35-50", "50-65", "65+"]
        )
        fairness_age = ev.evaluate_fairness_bias_audit(
            y_true=y_te,
            proba=probas[best],
            threshold=best_thr,
            slice_series=age_bins,
            slice_name="customer_age_group",
        )
        fairness_age.to_csv(out_dir / "fairness_audit_customer_age.csv", index=False)
        ev.plot_fairness_audit(fairness_age, "Customer Age Groups", out_dir / "fairness_audit_age.png")
        print("  Customer Age Group Audit:")
        for _, f_row in fairness_age.iterrows():
            print(f"    Age {f_row['group']:6s} | Size: {f_row['sample_size']:6d} ({f_row['sample_pct']:4.1f}%) "
                  f"| Approval Rate: {f_row['approval_rate']:.3f} | Disparate Impact: {f_row['disparate_impact_ratio']:.3f} "
                  f"| 80%-Compliant: {f_row['four_fifths_compliant']}")

    # 2. Device Operating System Slices
    if "device_os" in test.columns:
        fairness_os = ev.evaluate_fairness_bias_audit(
            y_true=y_te,
            proba=probas[best],
            threshold=best_thr,
            slice_series=test["device_os"],
            slice_name="device_os",
        )
        fairness_os.to_csv(out_dir / "fairness_audit_device_os.csv", index=False)
        ev.plot_fairness_audit(fairness_os, "Device Operating System", out_dir / "fairness_audit_device_os.png")
    # Enhanced Fairness Audit with Confidence Intervals & Error Rates
    if "customer_age" in test.columns:
        df_enhanced_fair = evaluate_enhanced_fairness(y_te, probas[best], best_thr, age_bins, "customer_age")
        df_enhanced_fair.to_csv(out_dir / "enhanced_fairness_audit.csv", index=False)

        print("  Device OS Audit:")
        for _, f_row in fairness_os.iterrows():
            print(f"    OS {f_row['group']:8s} | Size: {f_row['sample_size']:6d} ({f_row['sample_pct']:4.1f}%) "
                  f"| Approval Rate: {f_row['approval_rate']:.3f} | Disparate Impact: {f_row['disparate_impact_ratio']:.3f} "
                  f"| 80%-Compliant: {f_row['four_fifths_compliant']}")

    # ------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Conformal Prediction & Uncertainty Quantification
    # ----------------------------------------------------------------------
    print("\n[Conformal Prediction: Finite-Sample Coverage Guarantees]")
    cp = ConformalFraudPredictor(alpha=0.05)
    cp.calibrate(y_va, models[best].predict_proba(X_va)[:, 1])
    cp_res = cp.evaluate_coverage(y_te, probas[best])
    pd.DataFrame([cp_res]).to_csv(out_dir / "conformal_prediction_summary.csv", index=False)
    print(f"  Empirical Coverage: {cp_res['empirical_coverage']:.2%} | Auto-Approve: {cp_res['auto_approve_rate']:.1%} | Human-Review: {cp_res['human_review_rate']:.1%}")

    # ----------------------------------------------------------------------
    # Fairness vs Financial Cost Pareto Frontier
    # ----------------------------------------------------------------------
    if "customer_age" in test.columns:
        age_slice = pd.cut(test["customer_age"], bins=[0, 25, 35, 50, 65, 120], labels=["<25", "25-35", "35-50", "50-65", "65+"])
        pareto_df = compute_fairness_cost_pareto(y_te, probas[best], age_slice)
        pareto_df.to_csv(out_dir / "fairness_cost_pareto_frontier.csv", index=False)
        print("  Pareto Frontier Analysis saved -> fairness_cost_pareto_frontier.csv")

    # ----------------------------------------------------------------------
    # Systematic Ablation Study (Review 3 Research Validation)
    # ----------------------------------------------------------------------
    print("\n[Systematic Ablation Study]")
    df_ablation = run_ablation_study(
        y_true=y_te,
        y_proba=probas[best],
        y_val=y_va,
        proba_val=models[best].predict_proba(X_va)[:, 1],
        out_dir=out_dir,
        model_name=best,
    )
    print("  Ablation Study Summary (Phase 2 Hierarchical Triage):")
    for _, ab_row in df_ablation.iterrows():
        print(f"    {ab_row['configuration']}: Cost=${ab_row['operational_cost_per_app']:.2f}/app | Prec={ab_row['precision']:.3f} | Rec={ab_row['recall']:.3f} | AutoApprove={ab_row['pct_auto_approve']:.1f}% | Review={ab_row['pct_human_review']:.1f}%")

    # ----------------------------------------------------------------------
    # Phase 3: Statistical Reliability, Model CIs & Fairness Audit
    # ----------------------------------------------------------------------
    from statistical_reliability import (
        compute_model_metric_confidence_intervals,
        evaluate_group_fairness_with_wilson,
        evaluate_conformal_coverage_disaggregated,
        evaluate_temporal_stability,
    )
    from hierarchical_triage import HierarchicalTriagePolicy
    from adaptive_threshold import AdaptiveThresholdCalibrator

    print("\n[Phase 3: Statistical Reliability & Confidence Intervals]")
    # 1. Model Performance CIs (Bootstrap for AUC/F1, Exact Wilson for rates)
    df_model_cis = compute_model_metric_confidence_intervals(
        y_true=y_te,
        proba=probas[best],
        threshold=best_thr,
        n_bootstrap=500,
        seed=args.seed,
    )
    df_model_cis.to_csv(out_dir / "phase3_model_confidence_intervals.csv", index=False)
    print("  Model Performance 95% Confidence Intervals:")
    for _, r_ci in df_model_cis.iterrows():
        print(f"    {r_ci['metric']:24s}: {r_ci['ci_95']} ({r_ci['method']})")

    # 2. Hierarchical Action Extraction for Fairness Audit
    p_va_best = models[best].predict_proba(X_va)[:, 1]
    c_val_opt = ev.optimize_cost_threshold(y_va, p_va_best, config.COST_FN, config.COST_FP, config.COST_TP)
    tau_c = float(c_val_opt["optimal_threshold"])
    cal_tdta = AdaptiveThresholdCalibrator(target_fpr=config.TARGET_FPR)
    tau_t, _ = cal_tdta.calibrate(y_va, p_va_best, trigger_reason="phase3_triage")
    triage_pol = HierarchicalTriagePolicy(tau_cost=tau_c, tau_tdta=tau_t, conformal_predictor=cp)
    routed_test = triage_pol.route(probas[best])
    actions_test = routed_test["action"].values

    # 3. Enhanced Fairness Evaluation with Exact Wilson CIs
    if "customer_age" in test.columns:
        df_fair_age = evaluate_group_fairness_with_wilson(
            y_true=y_te,
            proba=probas[best],
            actions=actions_test,
            slice_series=age_bins,
            slice_name="customer_age_group",
        )
        df_fair_age.to_csv(out_dir / "phase3_fairness_metrics.csv", index=False)
        print("  Customer Age Group Fairness (Wilson CIs):")
        for _, f_row in df_fair_age.iterrows():
            print(f"    Age {f_row['group']:6s} | N={f_row['sample_size']:5d} (Frauds={f_row['fraud_count']:2d}) "
                  f"| Approval: {f_row['approval_ci_95']} | Recall: {f_row['recall_ci_95']} "
                  f"| DI: {f_row['disparate_impact_ratio']:.3f} ({f_row['statistical_reliability']})")

    # 4. Conformal Coverage Disaggregated across Slices
    if "customer_age" in test.columns:
        df_conf_cov = evaluate_conformal_coverage_disaggregated(
            y_true=y_te,
            proba=probas[best],
            conformal_predictor=cp,
            subgroup_series=age_bins,
            subgroup_name="customer_age_group",
        )
        df_conf_cov.to_csv(out_dir / "phase3_conformal_coverage.csv", index=False)

    # 5. Temporal Stability Tracking (Month 6 vs Month 7)
    df_temp_stab = evaluate_temporal_stability(
        df_raw=test,
        proba=probas[best],
        actions=actions_test,
        conformal_predictor=cp,
        time_col=config.TIME_COL,
        test_months=config.TEST_MONTHS,
    )
    df_temp_stab.to_csv(out_dir / "phase3_temporal_stability.csv", index=False)
    print("  Temporal Stability Across Test Months:")
    for _, t_row in df_temp_stab.iterrows():
        print(f"    Month {t_row['test_month']} | N={t_row['sample_size']:4d} (Frauds={t_row['fraud_count']:2d}) "
              f"| ROC-AUC: {t_row['ROC_AUC']:.4f} | Recall: {t_row['recall_ci_95']} "
              f"| Coverage: {t_row['conformal_empirical_coverage']}")

    # ------------------------------------------------------------------------
    # Phase 4: Realistic Adversarial Robustness Validation (Raw-Input Perturbations)
    # ------------------------------------------------------------------------
    if not args.skip_stress:
        print("\n[Phase 4: Realistic Adversarial Robustness Validation (Raw-Input Perturbations)]")
        from realistic_adversarial import (
            AdversarialPerturbationGenerator,
            evaluate_adversarial_threat_scenario,
        )
        adv_gen = AdversarialPerturbationGenerator(seed=args.seed)
        
        # 5 Realistic Threat Scenarios + Clean Baseline
        scenarios = {
            "Clean Baseline": test,
            "S1: Application Camouflage": adv_gen.perturb_application_camouflage(test, strength=0.2),
            "S2: Velocity Throttling": adv_gen.perturb_velocity_throttling(test, throttle_ratio=0.25),
            "S3: Device Cloaking": adv_gen.perturb_device_cloaking(test),
            "S4: Missingness Exploitation": adv_gen.perturb_missingness_exploitation(test),
            "S5: Multi-Vector Syndicate": adv_gen.perturb_multi_vector_syndicate(test),
        }
        
        phase4_results = []
        action_dist_rows = []
        clean_auc, clean_prauc, clean_rec, clean_fpr, clean_cost = None, None, None, None, None

        for sc_name, df_sc in scenarios.items():
            sc_res = evaluate_adversarial_threat_scenario(
                scenario_name=sc_name,
                df_perturbed_raw=df_sc,
                df_original_raw=test,
                preprocessor=pre,
                model=models[best],
                triage_policy=triage_pol,
                frozen_valid_threshold=best_thr,
            )
            
            if sc_name == "Clean Baseline":
                clean_auc = sc_res["ROC_AUC"]
                clean_prauc = sc_res["PR_AUC"]
                clean_rec = sc_res["Recall"]
                clean_fpr = sc_res["FPR"]
                clean_cost = sc_res["avg_cost_per_applicant"]
            
            sc_res["delta_ROC_AUC"] = sc_res["ROC_AUC"] - clean_auc
            sc_res["delta_PR_AUC"] = sc_res["PR_AUC"] - clean_prauc
            sc_res["delta_Recall"] = sc_res["Recall"] - clean_rec
            sc_res["delta_FPR"] = sc_res["FPR"] - clean_fpr
            sc_res["delta_Cost"] = sc_res["avg_cost_per_applicant"] - clean_cost
            
            phase4_results.append(sc_res)
            action_dist_rows.append({
                "scenario": sc_name,
                "pct_auto_approve": sc_res["pct_auto_approve"],
                "pct_human_review": sc_res["pct_human_review"],
                "pct_auto_flag": sc_res["pct_auto_flag"],
                "frauds_auto_flagged": sc_res["frauds_auto_flagged"],
                "frauds_referred_review": sc_res["frauds_referred_review"],
                "frauds_erroneously_approved": sc_res["frauds_erroneously_approved"],
                "triage_recall": sc_res["triage_recall"],
                "triage_recall_ci_95": sc_res["triage_recall_ci_95"],
            })
            
            print(f"  [{sc_name}] AUC={sc_res['ROC_AUC']:.4f} ({sc_res['delta_ROC_AUC']:+.4f}) | "
                  f"PR-AUC={sc_res['PR_AUC']:.4f} ({sc_res['delta_PR_AUC']:+.4f}) | "
                  f"Cost=${sc_res['avg_cost_per_applicant']:.2f} ({sc_res['delta_Cost']:+.2f}) | "
                  f"TriageRec={sc_res['triage_recall']:.3f} | "
                  f"AutoApprove={sc_res['pct_auto_approve']:.1f}% Review={sc_res['pct_human_review']:.1f}% Flag={sc_res['pct_auto_flag']:.1f}%")
                  
        df_p4_res = pd.DataFrame(phase4_results)
        df_p4_res.to_csv(out_dir / "phase4_adversarial_results.csv", index=False)
        df_act_dist = pd.DataFrame(action_dist_rows)
        df_act_dist.to_csv(out_dir / "phase4_action_distribution.csv", index=False)
        print(f"  Phase 4 adversarial evaluation completed across {len(scenarios)} realistic threat models.")

    # ------------------------------------------------------------------------
    # Phase 5: Delayed-Label Feedback & Temporal Prequential Validation
    # ------------------------------------------------------------------------
    print("\n[Phase 5: Delayed-Label Feedback & Temporal Prequential Validation]")
    from delayed_feedback_validation import run_delayed_feedback_evaluation

    # Define the 6 research delay configurations
    delay_configs = [
        {"name": "Scenario A: Static Validation Threshold", "policy": "Static", "delay": 0.0, "is_static": True},
        {"name": "Scenario B: TDTA (0-Day Delay)", "policy": "TDTA", "delay": 0.0, "is_static": False},
        {"name": "Scenario C: TDTA (7-Day Delay)", "policy": "TDTA", "delay": 7.0, "is_static": False},
        {"name": "Scenario D: TDTA (14-Day Delay)", "policy": "TDTA", "delay": 14.0, "is_static": False},
        {"name": "Scenario E: TDTA (30-Day Delay)", "policy": "TDTA", "delay": 30.0, "is_static": False},
        {"name": "Scenario F: TDTA (60-Day Delay)", "policy": "TDTA", "delay": 60.0, "is_static": False},
    ]

    p5_results = []
    p5_threshold_rows = []
    p5_availability_rows = []
    p5_action_dist_rows = []

    for dc in delay_configs:
        res_dc = run_delayed_feedback_evaluation(
            model=models[best],
            preprocessor=pre,
            df_raw=df,
            initial_val_month=5,
            test_months=[6, 7],
            delay_days=dc["delay"],
            policy_name=dc["policy"],
            is_static=dc["is_static"],
            conformal_predictor=cp,
        )

        s_dict = res_dc["summary"]
        s_dict["scenario"] = dc["name"]
        p5_results.append(s_dict)

        for th in res_dc["threshold_history"]:
            th["scenario"] = dc["name"]
            p5_threshold_rows.append(th)

        for al in res_dc["availability_logs"]:
            al["scenario"] = dc["name"]
            p5_availability_rows.append(al)

        p5_action_dist_rows.append({
            "scenario": dc["name"],
            "delay_days": dc["delay"],
            "pct_auto_approve": s_dict["pct_auto_approve"],
            "pct_human_review": s_dict["pct_human_review"],
            "pct_auto_flag": s_dict["pct_auto_flag"],
            "frauds_auto_flagged": s_dict["frauds_auto_flagged"],
            "frauds_referred_review": s_dict["frauds_referred_review"],
            "frauds_erroneously_approved": s_dict["frauds_erroneously_approved"],
            "direct_flag_recall": s_dict["direct_flag_recall"],
            "triage_recall": s_dict["triage_recall"],
            "triage_recall_ci_95": s_dict["triage_recall_ci_95"],
        })

        print(f"  [{dc['name']}] Thr={s_dict['final_threshold']:.4f} | "
              f"Cost=${s_dict['avg_cost_per_applicant']:.2f}/app | "
              f"DirectRec={s_dict['direct_flag_recall']:.3f} | "
              f"TriageRec={s_dict['triage_recall']:.3f} | "
              f"Approve={s_dict['pct_auto_approve']:.1f}% Review={s_dict['pct_human_review']:.1f}% Flag={s_dict['pct_auto_flag']:.1f}%")

    pd.DataFrame(p5_results).to_csv(out_dir / "phase5_delay_scenario_results.csv", index=False)
    pd.DataFrame(p5_threshold_rows).to_csv(out_dir / "phase5_threshold_evolution.csv", index=False)
    pd.DataFrame(p5_availability_rows).to_csv(out_dir / "phase5_label_availability_log.csv", index=False)
    pd.DataFrame(p5_action_dist_rows).to_csv(out_dir / "phase5_action_distribution.csv", index=False)
    print(f"  Phase 5 delayed-label validation completed across {len(delay_configs)} operational scenarios.")

    # ------------------------------------------------------------------------
    # Phase 6: Formal Deliverables Synthesis
    # ------------------------------------------------------------------------
    print("\n[Phase 6: Emitting Standardized Research Deliverables]")
    # 1. Model performance
    pd.DataFrame(rows).to_csv(out_dir / "phase6_model_performance.csv", index=False)
    # 2. Model confidence intervals
    df_model_cis.to_csv(out_dir / "phase6_model_confidence_intervals.csv", index=False)
    # 3. Hierarchical triage results
    df_triage_res = pd.DataFrame([{
        "scenario": "Full Proposed Framework (Hierarchical Triage)",
        "model": best,
        "tau_cost": tau_c,
        "tau_tdta": tau_t,
        "total_test_records": len(y_te),
        "total_frauds": int(np.sum(y_te == 1)),
        "pct_auto_approve": float(np.mean(actions_test == "AUTO_APPROVE") * 100.0),
        "pct_human_review": float(np.mean(actions_test == "HUMAN_REVIEW") * 100.0),
        "pct_auto_flag": float(np.mean(actions_test == "AUTO_FLAG") * 100.0),
        "frauds_auto_flagged": int(np.sum((actions_test == "AUTO_FLAG") & (y_te == 1))),
        "frauds_referred_review": int(np.sum((actions_test == "HUMAN_REVIEW") & (y_te == 1))),
        "frauds_erroneously_approved": int(np.sum((actions_test == "AUTO_APPROVE") & (y_te == 1))),
        "legit_auto_flagged": int(np.sum((actions_test == "AUTO_FLAG") & (y_te == 0))),
        "legit_referred_review": int(np.sum((actions_test == "HUMAN_REVIEW") & (y_te == 0))),
        "legit_auto_approved": int(np.sum((actions_test == "AUTO_APPROVE") & (y_te == 0))),
        "direct_flag_recall": float(np.sum((actions_test == "AUTO_FLAG") & (y_te == 1)) / max(np.sum(y_te == 1), 1)),
        "triage_recall": float(np.sum((actions_test != "AUTO_APPROVE") & (y_te == 1)) / max(np.sum(y_te == 1), 1)),
        "fraud_escape_rate": float(np.sum((actions_test == "AUTO_APPROVE") & (y_te == 1)) / max(np.sum(y_te == 1), 1)),
        "avg_cost_per_applicant": float(c_test["cost_per_application"]),
    }])
    df_triage_res.to_csv(out_dir / "phase6_hierarchical_triage_results.csv", index=False)
    # 4. Fairness
    if "customer_age" in test.columns:
        df_fair_age.to_csv(out_dir / "phase6_fairness_results.csv", index=False)
    # 5. Conformal
    if "customer_age" in test.columns:
        df_conf_cov.to_csv(out_dir / "phase6_conformal_results.csv", index=False)
    # 6. Adversarial results
    df_p4_res.to_csv(out_dir / "phase6_adversarial_results.csv", index=False)
    # 7. Delayed feedback results
    pd.DataFrame(p5_results).to_csv(out_dir / "phase6_delayed_feedback_results.csv", index=False)
    # 8. Ablation results
    df_ablation.to_csv(out_dir / "phase6_ablation_results.csv", index=False)
    # 9. Cost reconstruction
    df_scenarios.to_csv(out_dir / "phase6_cost_reconstruction.csv", index=False)
    # 10. Feature importance
    if best in models and hasattr(models[best], "feature_importances_"):
        df_imp = pd.DataFrame({
            "feature": pre.feature_names_,
            "importance": models[best].feature_importances_
        }).sort_values("importance", ascending=False)
        df_imp.to_csv(out_dir / "phase6_feature_importance.csv", index=False)
    print("  Phase 6 standardized CSV deliverables successfully saved.")

    # ------------------------------------------------------------------------
    # Explainability & Regulatory Compliance Audit
    # ------------------------------------------------------------------------
    if not args.skip_shap:
        print("\n[Explainability (XAI) & Regulatory Compliance Audit]")
        run_shap_and_regulatory_audit(models["LightGBM"], X_te, pd.Series(y_te), out_dir, args.seed,
                                      val_fpr_thresholds["LightGBM"])

    # ------------------------------------------------------------------------
    # Expanding-Window Temporal Cross-Validation
    # ------------------------------------------------------------------------
    if not args.skip_cv:
        print("\n[Expanding-Window Temporal Cross-Validation]")
        cv = temporal_cv(df, args.seed, loss_type=args.loss, resample_strategy=args.resample)
        cv.to_csv(out_dir / "cv_results_5fold_temporal.csv", index=False)

        summ = cv.groupby("model")[["ROC_AUC", "PR_AUC", "TPR_at_1pct_FPR", "TPR_at_5pct_FPR",
                                    "Precision", "Recall", "F1"]].agg(["mean", "std"])
        summ.to_csv(out_dir / "cv_summary_5fold_temporal.csv")
        print(summ.round(4).to_string())

        # Cross-validation paired significance test
        paired_test = ev.paired_cv_significance(cv, metric="PR_AUC", model_a="LightGBM", model_b="XGBoost")
        pd.DataFrame([paired_test]).to_csv(out_dir / "cv_paired_significance_test.csv", index=False)
        print(f"\n  CV Paired t-test (PR-AUC): t={paired_test['paired_t_stat']:.3f}, p={paired_test['paired_t_pvalue']:.4f}, Cohen's d={paired_test['cohens_d']:.3f}")

    # ------------------------------------------------------------------------
    # Temporal Drift-Aware Threshold Adaptation (TDTA) Research Pipeline
    # ------------------------------------------------------------------------
    if args.run_tdta:
        print("\n" + "=" * 70)
        print("LAUNCHING TEMPORAL DRIFT-AWARE THRESHOLD ADAPTATION (TDTA) SUITE")
        print("=" * 70)
        from tdta_experiment import run_all_research_experiments
        run_all_research_experiments(
            data_path=args.data,
            output_dir=str(out_dir),
            sample_frac=args.sample_frac,
            seed=args.seed,
        )

    print(f"\nAll benchmark components completed in {(time.time() - t0) / 60:.1f} min. Outputs -> {out_dir}")


if __name__ == "__main__":
    main()

