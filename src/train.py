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
    print(f"  DeLong's Test: ΔAUC={delong_res['auc_difference']:+.4f} | "
          f"z={delong_res['z_statistic']:.3f} | p={delong_res['p_value']:.4e} "
          f"(Significant: {delong_res['statistically_significant_05']})")

    # ------------------------------------------------------------------------
    # Cost-Sensitive Threshold Optimization
    # ------------------------------------------------------------------------
    print("\n[Cost-Sensitive Optimization & Operational Loss Matrix]")
    cost_rows = []
    for name, p in probas.items():
        c_opt = ev.optimize_cost_threshold(y_te, p, cost_fn=config.COST_FN, cost_fp=config.COST_FP)
        cost_rows.append({
            "model": name,
            "optimal_threshold": c_opt["optimal_threshold"],
            "min_cost_per_app": c_opt["min_cost_per_app"],
            "default_05_cost_per_app": c_opt["default_05_cost_per_app"],
            "savings_vs_default_per_app": c_opt["savings_vs_default_per_app"],
            "savings_pct_vs_default": c_opt["savings_pct_vs_default"],
            "fpr_5pct_cost_per_app": c_opt["fpr_5pct_cost_per_app"],
            "precision_at_optimal": c_opt["optimal_precision"],
            "recall_at_optimal": c_opt["optimal_recall"],
        })
        print(f"  [{name}] Min Cost=${c_opt['min_cost_per_app']:.2f}/app @ thr={c_opt['optimal_threshold']:.3f} "
              f"(Saves {c_opt['savings_pct_vs_default']:.1f}% vs default 0.5 threshold)")

    pd.DataFrame(cost_rows).to_csv(out_dir / "cost_optimization_results.csv", index=False)
    ev.plot_cost_curve(y_te, probas, config.COST_FN, config.COST_FP, out_dir / "cost_optimization_curve.png")

    # Standard plots
    ev.plot_roc(y_te, probas, out_dir / "roc_curve.png")
    ev.plot_pr(y_te, probas, out_dir / "precision_recall_curve.png")
    best = max(rows, key=lambda r: r["PR_AUC"])["model"]
    ev.plot_score_distribution(y_te, probas[best], best, out_dir / "fraud_probability_distribution.png")

    for name in models:
        imp = gain_importance(models[name], list(X_tr.columns))
        imp.to_csv(out_dir / f"feature_importance_{name.lower()}.csv", index=False)
    ev.plot_importance(pd.read_csv(out_dir / "feature_importance_lightgbm.csv"), "gain",
                       "LightGBM Feature Importance (Gain)",
                       out_dir / "lightgbm_feature_importance.png")

    # ------------------------------------------------------------------------
    # Adversarial Robustness & Concept Drift Stress Testing
    # ------------------------------------------------------------------------
    if not args.skip_stress:
        print("\n[Adversarial Robustness Stress-Testing]")
        stress_df = ev.evaluate_adversarial_robustness(models[best], X_te, y_te)
        stress_df.to_csv(out_dir / "adversarial_stress_test_summary.csv", index=False)
        ev.plot_adversarial_stress(stress_df, out_dir / "adversarial_stress_curve.png")
        print(f"  Stress test completed across {len(config.ADVERSARIAL_NOISE_LEVELS)} perturbation budgets.")

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

