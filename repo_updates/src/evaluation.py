"""Metrics, statistical significance, cost-sensitive optimization, and XAI for fraud detection."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
)

import config


def tpr_at_fpr(y_true, proba, target_fpr):
    """Best achievable TPR with FPR <= target_fpr on this set's own ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(tpr[np.searchsorted(fpr, target_fpr, side="right") - 1])


def threshold_for_fpr(y_true, proba, target_fpr):
    """Score threshold that yields ~target_fpr on the given (validation) set."""
    neg = np.asarray(proba)[np.asarray(y_true) == 0]
    return float(np.quantile(neg, 1.0 - target_fpr))


def compute_metrics(y_true, proba, threshold):
    """Ranking metrics + operating-point metrics at a PRE-SELECTED threshold."""
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, proba),
        "PR_AUC": average_precision_score(y_true, proba),
        "TPR_at_1pct_FPR": tpr_at_fpr(y_true, proba, 0.01),
        "TPR_at_5pct_FPR": tpr_at_fpr(y_true, proba, 0.05),
        "threshold": threshold,
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "realized_FPR": fp / max(fp + tn, 1),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


# ----------------------------------------------------------------------------
# 1. Cost-Sensitive Optimization & Operational Loss Matrix
# ----------------------------------------------------------------------------
def compute_cost_matrix(y_true, pred, cost_fn=config.COST_FN, cost_fp=config.COST_FP,
                        cost_tp=config.COST_TP, cost_tn=config.COST_TN):
    """Calculate absolute and per-application operational cost."""
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp) + (tp * cost_tp) + (tn * cost_tn)
    return {
        "total_cost": float(total_cost),
        "cost_per_application": float(total_cost / len(y_true)),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def optimize_cost_threshold(y_true, proba, cost_fn=config.COST_FN, cost_fp=config.COST_FP,
                            cost_tp=config.COST_TP, cost_tn=config.COST_TN, n_grid=200):
    """Scan candidate score thresholds to locate the minimum operational financial loss point."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    thresholds = np.linspace(0.005, 0.995, n_grid)
    costs, precs, recs, f1s = [], [], [], []

    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        c = compute_cost_matrix(y_true, pred, cost_fn, cost_fp, cost_tp, cost_tn)
        costs.append(c["cost_per_application"])
        precs.append(precision_score(y_true, pred, zero_division=0))
        recs.append(recall_score(y_true, pred, zero_division=0))
        f1s.append(f1_score(y_true, pred, zero_division=0))

    best_idx = int(np.argmin(costs))
    optimal_thr = float(thresholds[best_idx])
    min_cost_per_app = float(costs[best_idx])

    # Benchmarks: default 0.5 threshold vs 5% FPR threshold
    default_cost = compute_cost_matrix(y_true, (proba >= 0.5).astype(int), cost_fn, cost_fp, cost_tp, cost_tn)
    thr_5fpr = threshold_for_fpr(y_true, proba, 0.05)
    fpr_cost = compute_cost_matrix(y_true, (proba >= thr_5fpr).astype(int), cost_fn, cost_fp, cost_tp, cost_tn)

    savings_vs_default = default_cost["cost_per_application"] - min_cost_per_app
    savings_pct_vs_default = (savings_vs_default / default_cost["cost_per_application"]) * 100.0 if default_cost["cost_per_application"] > 0 else 0.0

    return {
        "optimal_threshold": optimal_thr,
        "min_cost_per_app": min_cost_per_app,
        "optimal_precision": float(precs[best_idx]),
        "optimal_recall": float(recs[best_idx]),
        "optimal_f1": float(f1s[best_idx]),
        "default_05_cost_per_app": default_cost["cost_per_application"],
        "fpr_5pct_cost_per_app": fpr_cost["cost_per_application"],
        "savings_vs_default_per_app": savings_vs_default,
        "savings_pct_vs_default": savings_pct_vs_default,
        "threshold_grid": thresholds,
        "cost_curve": np.array(costs),
    }


def plot_cost_curve(y_true, probas, cost_fn, cost_fp, path):
    """Plot operational loss curves per application across decision thresholds."""
    plt.figure(figsize=(7, 5))
    for name, p in probas.items():
        res = optimize_cost_threshold(y_true, p, cost_fn=cost_fn, cost_fp=cost_fp)
        plt.plot(res["threshold_grid"], res["cost_curve"],
                 label=f"{name} (Min Cost=${res['min_cost_per_app']:.2f} @ thr={res['optimal_threshold']:.3f})")
        plt.scatter([res["optimal_threshold"]], [res["min_cost_per_app"]], s=60, zorder=5)

    plt.xlabel("Decision Threshold (Probability)")
    plt.ylabel(f"Operational Cost per Application ($)")
    plt.title(f"Cost-Sensitive Optimization Curve\n(FN Cost=${cost_fn:.0f}, FP Investigation=${cost_fp:.0f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 2. DeLong's Asymptotic Test for Correlated ROC-AUC
# ----------------------------------------------------------------------------
def _fast_delong_structural_components(predictions, true_labels):
    """Vectorized O(N log N) computation of DeLong's mid-rank structural components."""
    true_labels = np.asarray(true_labels, dtype=bool)
    m = int(np.sum(true_labels))
    n = int(len(true_labels) - m)
    if m == 0 or n == 0:
        raise ValueError("Both positive and negative cases required for DeLong's test.")

    k = predictions.shape[0]
    V10 = np.empty((k, m), dtype=float)
    V01 = np.empty((k, n), dtype=float)

    for i in range(k):
        preds = predictions[i]
        # Mid-ranks across all instances
        order = np.argsort(preds)
        ranks = np.empty(len(preds), dtype=float)
        ranks[order] = np.arange(1, len(preds) + 1)
        # Handle ties
        unique_preds, inv, counts = np.unique(preds, return_inverse=True, return_counts=True)
        if len(unique_preds) < len(preds):
            tie_ranks = np.bincount(inv, weights=ranks) / counts
            ranks = tie_ranks[inv]

        pos_preds = preds[true_labels]
        order_pos = np.argsort(pos_preds)
        ranks_pos = np.empty(m, dtype=float)
        ranks_pos[order_pos] = np.arange(1, m + 1)
        unique_pos, inv_pos, counts_pos = np.unique(pos_preds, return_inverse=True, return_counts=True)
        if len(unique_pos) < m:
            tie_pos = np.bincount(inv_pos, weights=ranks_pos) / counts_pos
            ranks_pos = tie_pos[inv_pos]

        neg_preds = preds[~true_labels]
        order_neg = np.argsort(neg_preds)
        ranks_neg = np.empty(n, dtype=float)
        ranks_neg[order_neg] = np.arange(1, n + 1)
        unique_neg, inv_neg, counts_neg = np.unique(neg_preds, return_inverse=True, return_counts=True)
        if len(unique_neg) < n:
            tie_neg = np.bincount(inv_neg, weights=ranks_neg) / counts_neg
            ranks_neg = tie_neg[inv_neg]

        V10[i] = (ranks[true_labels] - ranks_pos) / n
        V01[i] = 1.0 - (ranks[~true_labels] - ranks_neg) / m

    return V10, V01, m, n


def delong_roc_test(y_true, proba_a, proba_b):
    """DeLong's statistical significance test for correlated ROC-AUC on identical test set.

    Reference:
        DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
        Comparing the areas under two or more correlated receiver operating
        characteristic curves: a nonparametric approach. Biometrics, 837-845.
    """
    y_true = np.asarray(y_true)
    preds = np.vstack([np.asarray(proba_a), np.asarray(proba_b)])
    V10, V01, m, n = _fast_delong_structural_components(preds, y_true == 1)

    auc_a = float(np.mean(V10[0]))
    auc_b = float(np.mean(V10[1]))
    diff = auc_a - auc_b

    # Covariance estimation of pseudo-values
    S10 = np.cov(V10)
    S01 = np.cov(V01)
    S = (S10 / m) + (S01 / n)

    # Contrast vector L = [1, -1]
    var_diff = float(S[0, 0] + S[1, 1] - 2.0 * S[0, 1])
    var_diff = max(var_diff, 1e-12)
    se_diff = np.sqrt(var_diff)
    z_stat = diff / se_diff
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    ci_95 = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)

    return {
        "auc_model_a": auc_a,
        "auc_model_b": auc_b,
        "auc_difference": diff,
        "se_difference": se_diff,
        "z_statistic": float(z_stat),
        "p_value": float(p_val),
        "statistically_significant_05": bool(p_val < 0.05),
        "ci_95": ci_95,
    }


def paired_cv_significance(cv_df, metric="ROC_AUC", model_a="XGBoost", model_b="LightGBM"):
    """Paired Student's t-test and Wilcoxon signed-rank test across temporal CV folds."""
    sub_a = cv_df[cv_df["model"] == model_a].sort_values("fold_test_month")[metric].values
    sub_b = cv_df[cv_df["model"] == model_b].sort_values("fold_test_month")[metric].values
    if len(sub_a) != len(sub_b) or len(sub_a) < 2:
        return {"error": "Insufficient folds for paired testing"}

    diffs = sub_a - sub_b
    t_stat, t_pval = stats.ttest_rel(sub_a, sub_b)
    # Effect size (Cohen's d for paired samples)
    cohen_d = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-9))

    try:
        w_stat, w_pval = stats.wilcoxon(sub_a, sub_b)
    except Exception:
        w_stat, w_pval = np.nan, np.nan

    return {
        "metric": metric,
        "mean_model_a": float(np.mean(sub_a)),
        "mean_model_b": float(np.mean(sub_b)),
        "mean_diff": float(np.mean(diffs)),
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(t_pval),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "cohens_d": cohen_d,
    }


# ----------------------------------------------------------------------------
# 3. Adversarial Robustness & Concept Drift Stress-Testing
# ----------------------------------------------------------------------------
def evaluate_adversarial_robustness(model, X_te, y_te, noise_levels=None, target_cols=None):
    """Stress-test model resilience under adversarial feature evasion and sensor noise.

    Simulates fraudster evasion (e.g. concealing device footprint or throttle burst velocities).
    """
    if noise_levels is None:
        noise_levels = config.ADVERSARIAL_NOISE_LEVELS
    if target_cols is None:
        target_cols = [c for c in config.ADVERSARIAL_TARGET_FEATURES if c in X_te.columns]

    rows = []
    base_proba = model.predict_proba(X_te)[:, 1]
    base_auc = roc_auc_score(y_te, base_proba)
    base_prauc = average_precision_score(y_te, base_proba)
    base_tpr5 = tpr_at_fpr(y_te, base_proba, 0.05)

    for eps in noise_levels:
        if eps == 0.0:
            rows.append({
                "noise_level": 0.0,
                "ROC_AUC": base_auc,
                "PR_AUC": base_prauc,
                "TPR_at_5pct_FPR": base_tpr5,
                "delta_ROC_AUC": 0.0,
                "delta_PR_AUC": 0.0,
            })
            continue

        # Perturb target behavioral and device features with multiplicative evasion noise
        X_perturbed = X_te.copy()
        for col in target_cols:
            sigma = float(X_te[col].std()) if X_te[col].std() > 0 else 1.0
            noise = np.random.normal(0, eps * sigma, size=len(X_te))
            # Fraudsters dampen suspicion (attenuate velocity/distinct count)
            X_perturbed[col] = (X_perturbed[col] * (1.0 - eps) + noise).clip(lower=0)

        p_noise = model.predict_proba(X_perturbed)[:, 1]
        auc_n = roc_auc_score(y_te, p_noise)
        prauc_n = average_precision_score(y_te, p_noise)
        tpr5_n = tpr_at_fpr(y_te, p_noise, 0.05)

        rows.append({
            "noise_level": float(eps),
            "ROC_AUC": float(auc_n),
            "PR_AUC": float(prauc_n),
            "TPR_at_5pct_FPR": float(tpr5_n),
            "delta_ROC_AUC": float(auc_n - base_auc),
            "delta_PR_AUC": float(prauc_n - base_prauc),
        })

    return pd.DataFrame(rows)


def plot_adversarial_stress(stress_df, path):
    """Plot performance degradation curves under increasing adversarial noise budgets."""
    plt.figure(figsize=(7, 4.8))
    plt.plot(stress_df["noise_level"], stress_df["ROC_AUC"], marker="o", label="ROC-AUC")
    plt.plot(stress_df["noise_level"], stress_df["PR_AUC"], marker="s", label="PR-AUC (Average Precision)")
    plt.plot(stress_df["noise_level"], stress_df["TPR_at_5pct_FPR"], marker="^", label="TPR @ 5% FPR")
    plt.xlabel("Adversarial Evasion Perturbation Budget (ε)")
    plt.ylabel("Metric Score")
    plt.title("Model Robustness Under Adversarial Feature Evasion")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 4. Regulatory Explainability Audit Notice (RBI / Basel III Alignment)
# ----------------------------------------------------------------------------
def generate_regulatory_audit_card(shap_values, base_value, feature_names, instance_series,
                                   y_true_val, predicted_proba, threshold, output_path, case_id=0):
    """Produce a formal Regulatory Adverse Action & Explainability Audit Notice.

    Complies with:
    - RBI Regulatory Sandbox / Master Direction on Digital Lending (2022)
    - Basel Committee on Banking Supervision (BCBS) Model Risk Management / Fed SR 11-7
    """
    sv = np.asarray(shap_values)
    feat_contrib = pd.DataFrame({
        "feature": feature_names,
        "feature_value": instance_series.values,
        "shap_importance": sv,
    })

    # Sort into positive risk drivers (increasing fraud score) and mitigating drivers
    risk_drivers = feat_contrib[feat_contrib["shap_importance"] > 0].sort_values(
        "shap_importance", ascending=False).head(5)
    mitigating_factors = feat_contrib[feat_contrib["shap_importance"] < 0].sort_values(
        "shap_importance", ascending=True).head(5)

    decision = "DECLINE / AML INVESTIGATION" if predicted_proba >= threshold else "APPROVE"

    fig, ax = plt.subplots(figsize=(9, 6.5))
    top_feats = pd.concat([risk_drivers, mitigating_factors]).sort_values("shap_importance")
    colors = ["#2ca02c" if val < 0 else "#d62728" for val in top_feats["shap_importance"]]

    y_pos = np.arange(len(top_feats))
    ax.barh(y_pos, top_feats["shap_importance"], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    labels = [f"{row.feature} = {row.feature_value:.2f}" for _, row in top_feats.iterrows()]
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value (Contribution to Log-Odds of Fraud)")

    plt.title(
        f"Mule Account Regulatory Audit Notice (Case #{case_id})\n"
        f"Decision: {decision} | Fraud Probability: {predicted_proba:.3%} (Threshold: {threshold:.3%})\n"
        f"Basel III / RBI Digital Lending Compliance Audit Card",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# Standard Benchmark Plots
# ----------------------------------------------------------------------------
def plot_roc(y_true, probas, path):
    plt.figure(figsize=(6.5, 5.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.axvline(0.05, color="grey", ls=":", lw=0.8, label="5% FPR")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC — temporal test (months 6-7)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_pr(y_true, probas, path):
    plt.figure(figsize=(6.5, 5.5))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        plt.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_true, p):.3f})")
    plt.axhline(np.mean(y_true), color="k", ls="--", lw=0.8, label="Base rate")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall — temporal test"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_score_distribution(y_true, proba, name, path):
    y_true = np.asarray(y_true)
    plt.figure(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 51)
    plt.hist(proba[y_true == 0], bins=bins, alpha=.6, density=True, label="Legitimate")
    plt.hist(proba[y_true == 1], bins=bins, alpha=.6, density=True, label="Fraud")
    plt.yscale("log"); plt.xlabel(f"{name} fraud probability"); plt.ylabel("Density (log)")
    plt.title("Score distribution — temporal test"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_importance(imp_df, value_col, title, path, top=20):
    top_df = imp_df.sort_values(value_col, ascending=False).head(top).iloc[::-1]
    plt.figure(figsize=(8, 0.32 * len(top_df) + 1.5))
    plt.barh(top_df["feature"], top_df[value_col])
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

