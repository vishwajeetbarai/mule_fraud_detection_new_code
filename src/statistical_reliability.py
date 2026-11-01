"""Phase 3 Statistical Reliability & Fairness Analysis Module.

Implements:
1. Exact Wilson Score Confidence Intervals for binomial proportions with boundary edge-case handling.
2. Bootstrap & Analytical Confidence Intervals for ranking (ROC-AUC, PR-AUC) and threshold metrics.
3. Multi-Action Operational Fairness Auditing across protected slices (demographic parity, disparate impact, equal opportunity).
4. Conformal Coverage Breakdown (overall, temporal, demographic slices, risk deciles).
5. Temporal Stability Tracking across distinct evaluation months.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve
)

import config
import evaluation as ev
from conformal import ConformalFraudPredictor
from hierarchical_triage import HierarchicalTriagePolicy


# ----------------------------------------------------------------------------
# 1. Exact Wilson Score Confidence Interval
# ----------------------------------------------------------------------------
def wilson_score_interval(
    k: int,
    n: int,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Calculate exact Wilson Score interval for binomial proportions.

    Handles edge cases:
    - n == 0: returns (np.nan, np.nan, np.nan)
    - k == 0: lower bound is strictly 0.0, upper bound calculated via formula
    - k == n: upper bound is strictly 1.0, lower bound calculated via formula

    Formula:
    --------
    p_hat = k / n
    center = (p_hat + z^2 / (2n)) / (1 + z^2 / n)
    spread = (z / (1 + z^2 / n)) * sqrt(p_hat * (1 - p_hat) / n + z^2 / (4n^2))
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    """
    if n <= 0:
        return np.nan, np.nan, np.nan

    if k < 0 or k > n:
        raise ValueError(f"Successes k={k} must be in [0, n={n}]")

    p_hat = float(k) / float(n)
    alpha = 1.0 - confidence
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    n_f = float(n)

    denominator = 1.0 + z2 / n_f
    center = (p_hat + z2 / (2.0 * n_f)) / denominator
    term_inside = (p_hat * (1.0 - p_hat) / n_f) + (z2 / (4.0 * n_f * n_f))
    spread = (z / denominator) * np.sqrt(max(0.0, term_inside))

    ci_low = max(0.0, center - spread)
    ci_high = min(1.0, center + spread)

    if k == 0:
        ci_low = 0.0
    elif k == n:
        ci_high = 1.0

    return p_hat, float(ci_low), float(ci_high)


def format_ci(pt: float, low: float, high: float, decimals: int = 4) -> str:
    """Format confidence interval as human-readable string."""
    if np.isnan(pt):
        return "N/A"
    return f"{pt:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]"


# ----------------------------------------------------------------------------
# 2. Comprehensive Model Metric Confidence Intervals
# ----------------------------------------------------------------------------
def compute_model_metric_confidence_intervals(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute statistically rigorous confidence intervals for test set performance.

    - Threshold metrics (Precision, Recall, F1, FPR, FNR): Wilson score interval.
    - Ranking metrics (ROC-AUC, PR-AUC, TPR@1%FPR, TPR@5%FPR): Stratified bootstrap.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    n = len(y)
    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    pos = tp + fn
    neg = tn + fp
    flagged = tp + fp

    # 1. Exact Wilson Score Intervals for Binomial Proportions
    rec_pt, rec_low, rec_high = wilson_score_interval(tp, pos, confidence)
    prec_pt, prec_low, prec_high = wilson_score_interval(tp, flagged, confidence)
    fpr_pt, fpr_low, fpr_high = wilson_score_interval(fp, neg, confidence)
    fnr_pt, fnr_low, fnr_high = wilson_score_interval(fn, pos, confidence)

    # 2. Stratified Bootstrap for Continuous / Ranking Metrics
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    boot_roc = []
    boot_pr = []
    boot_f1 = []
    boot_tpr1 = []
    boot_tpr5 = []

    valid_replicates = 0
    for _ in range(n_bootstrap):
        # Stratified sampling
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        boot_idx = np.concatenate([sample_pos, sample_neg])

        y_b = y[boot_idx]
        p_b = p[boot_idx]
        pred_b = (p_b >= threshold).astype(int)

        # Check binary presence
        if len(np.unique(y_b)) < 2:
            continue

        valid_replicates += 1
        boot_roc.append(roc_auc_score(y_b, p_b))
        boot_pr.append(average_precision_score(y_b, p_b))
        boot_f1.append(f1_score(y_b, pred_b, zero_division=0))
        boot_tpr1.append(ev.tpr_at_fpr(y_b, p_b, 0.01))
        boot_tpr5.append(ev.tpr_at_fpr(y_b, p_b, 0.05))

    alpha_tail = (1.0 - confidence) / 2.0
    q_low = alpha_tail * 100.0
    q_high = (1.0 - alpha_tail) * 100.0

    roc_pt = float(roc_auc_score(y, p))
    pr_pt = float(average_precision_score(y, p))
    f1_pt = float(f1_score(y, pred, zero_division=0))
    tpr1_pt = float(ev.tpr_at_fpr(y, p, 0.01))
    tpr5_pt = float(ev.tpr_at_fpr(y, p, 0.05))

    def b_ci(arr, pt):
        if not arr:
            return pt, np.nan, np.nan
        return pt, float(np.percentile(arr, q_low)), float(np.percentile(arr, q_high))

    roc_val, roc_low, roc_high = b_ci(boot_roc, roc_pt)
    pr_val, pr_low, pr_high = b_ci(boot_pr, pr_pt)
    f1_val, f1_low, f1_high = b_ci(boot_f1, f1_pt)
    tpr1_val, tpr1_low, tpr1_high = b_ci(boot_tpr1, tpr1_pt)
    tpr5_val, tpr5_low, tpr5_high = b_ci(boot_tpr5, tpr5_pt)

    rows = [
        {"metric": "ROC_AUC", "point_estimate": roc_val, "ci_lower": roc_low, "ci_upper": roc_high, "method": f"Stratified Bootstrap ({valid_replicates} reps)", "type": "Ranking"},
        {"metric": "PR_AUC", "point_estimate": pr_val, "ci_lower": pr_low, "ci_upper": pr_high, "method": f"Stratified Bootstrap ({valid_replicates} reps)", "type": "Ranking"},
        {"metric": "TPR_at_1pct_FPR", "point_estimate": tpr1_val, "ci_lower": tpr1_low, "ci_upper": tpr1_high, "method": f"Stratified Bootstrap ({valid_replicates} reps)", "type": "Threshold-Constrained"},
        {"metric": "TPR_at_5pct_FPR", "point_estimate": tpr5_val, "ci_lower": tpr5_low, "ci_upper": tpr5_high, "method": f"Stratified Bootstrap ({valid_replicates} reps)", "type": "Threshold-Constrained"},
        {"metric": "Recall (TPR)", "point_estimate": rec_pt, "ci_lower": rec_low, "ci_upper": rec_high, "method": "Exact Wilson Score", "type": "Threshold-Dependent"},
        {"metric": "Precision", "point_estimate": prec_pt, "ci_lower": prec_low, "ci_upper": prec_high, "method": "Exact Wilson Score", "type": "Threshold-Dependent"},
        {"metric": "F1_Score", "point_estimate": f1_val, "ci_lower": f1_low, "ci_upper": f1_high, "method": f"Stratified Bootstrap ({valid_replicates} reps)", "type": "Threshold-Dependent"},
        {"metric": "False Positive Rate (FPR)", "point_estimate": fpr_pt, "ci_lower": fpr_low, "ci_upper": fpr_high, "method": "Exact Wilson Score", "type": "Threshold-Dependent"},
        {"metric": "False Negative Rate (FNR)", "point_estimate": fnr_pt, "ci_lower": fnr_low, "ci_upper": fnr_high, "method": "Exact Wilson Score", "type": "Threshold-Dependent"},
    ]
    df_metrics = pd.DataFrame(rows)
    df_metrics["ci_95"] = [format_ci(r["point_estimate"], r["ci_lower"], r["ci_upper"]) for _, r in df_metrics.iterrows()]
    return df_metrics


# ----------------------------------------------------------------------------
# 3. Enhanced Demographic & Operational Action Fairness Evaluation
# ----------------------------------------------------------------------------
def evaluate_group_fairness_with_wilson(
    y_true: np.ndarray,
    proba: np.ndarray,
    actions: np.ndarray,
    slice_series: pd.Series,
    slice_name: str = "customer_age_group",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Evaluate algorithmic fairness across demographic/device subgroups with exact Wilson CIs.

    Considers:
    - Binary Effective Action: AUTO_APPROVE (Favorable = 0) vs Flagged/Review (1).
    - Multi-Action Partition: Auto-Approve, Auto-Flag, Human-Review.
    - Error rates (FPR, FNR, Recall, Precision).
    - Parity metrics (Disparate Impact, Demographic Parity Diff, Equal Opportunity Diff).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    acts = np.asarray(actions, dtype=str)
    slice_str = slice_series.astype(str).values
    groups = sorted(np.unique(slice_str))

    rows = []
    for g in groups:
        m = (slice_str == g)
        n_g = int(np.sum(m))
        if n_g == 0:
            continue

        y_g = y[m]
        p_g = p[m]
        acts_g = acts[m]

        n_pos_g = int(np.sum(y_g == 1))
        n_neg_g = int(np.sum(y_g == 0))

        # Binary effective decisions:
        # Favorable outcome = AUTO_APPROVE
        # Flagged = AUTO_FLAG or HUMAN_REVIEW
        flagged_g = (acts_g != "AUTO_APPROVE").astype(int)
        approved_g = (acts_g == "AUTO_APPROVE").astype(int)

        tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_g, flagged_g, labels=[0, 1]).ravel()

        # Approval (Selection) Rate with Wilson CI
        k_app = int(np.sum(approved_g == 1))
        sel_rate, sel_low, sel_high = wilson_score_interval(k_app, n_g, confidence)

        # Recall (True Positive Rate) with Wilson CI
        rec_rate, rec_low, rec_high = wilson_score_interval(tp_g, n_pos_g, confidence)

        # False Positive Rate with Wilson CI
        fpr_rate, fpr_low, fpr_high = wilson_score_interval(fp_g, n_neg_g, confidence)

        # Precision with Wilson CI
        prec_rate, prec_low, prec_high = wilson_score_interval(tp_g, tp_g + fp_g, confidence)

        # Multi-Action counts
        n_auto_app = int(np.sum(acts_g == "AUTO_APPROVE"))
        n_auto_flg = int(np.sum(acts_g == "AUTO_FLAG"))
        n_human_rev = int(np.sum(acts_g == "HUMAN_REVIEW"))

        rows.append({
            "slice_name": slice_name,
            "group": g,
            "sample_size": n_g,
            "fraud_count": n_pos_g,
            "fraud_prevalence": round(float(n_pos_g / n_g), 4),
            "approval_rate": round(sel_rate, 4),
            "approval_ci_95": format_ci(sel_rate, sel_low, sel_high),
            "recall": round(rec_rate, 4),
            "recall_ci_95": format_ci(rec_rate, rec_low, rec_high),
            "fpr": round(fpr_rate, 4),
            "fpr_ci_95": format_ci(fpr_rate, fpr_low, fpr_high),
            "precision": round(prec_rate, 4),
            "precision_ci_95": format_ci(prec_rate, prec_low, prec_high),
            "pct_auto_approve": round(n_auto_app / n_g * 100.0, 2),
            "pct_auto_flag": round(n_auto_flg / n_g * 100.0, 2),
            "pct_human_review": round(n_human_rev / n_g * 100.0, 2),
            "statistical_reliability": "Sufficient" if n_g >= 100 and n_pos_g >= 5 else "Small Sample (High Uncertainty)",
        })

    df_rep = pd.DataFrame(rows)
    if df_rep.empty:
        return df_rep

    # Reference group is group with maximum approval rate
    ref_idx = df_rep["approval_rate"].idxmax()
    ref_rate = df_rep.loc[ref_idx, "approval_rate"]
    ref_rec = df_rep.loc[ref_idx, "recall"]
    ref_group = df_rep.loc[ref_idx, "group"]

    df_rep["reference_group"] = ref_group
    df_rep["disparate_impact_ratio"] = [
        round(r["approval_rate"] / max(ref_rate, 1e-6), 4) for _, r in df_rep.iterrows()
    ]
    df_rep["demographic_parity_diff"] = [
        round(r["approval_rate"] - ref_rate, 4) for _, r in df_rep.iterrows()
    ]
    df_rep["equal_opportunity_diff"] = [
        round(r["recall"] - ref_rec, 4) if not np.isnan(r["recall"]) and not np.isnan(ref_rec) else np.nan
        for _, r in df_rep.iterrows()
    ]
    df_rep["four_fifths_heuristic_compliant"] = df_rep["disparate_impact_ratio"] >= 0.80

    return df_rep


# ----------------------------------------------------------------------------
# 4. Conformal Prediction Coverage Disaggregation
# ----------------------------------------------------------------------------
def evaluate_conformal_coverage_disaggregated(
    y_true: np.ndarray,
    proba: np.ndarray,
    conformal_predictor: ConformalFraudPredictor,
    subgroup_series: pd.Series,
    subgroup_name: str = "slice",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Evaluate conformal empirical coverage and set sizes disaggregated across temporal/demographic slices."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    sub = subgroup_series.astype(str).values
    groups = sorted(np.unique(sub))

    conf_df = conformal_predictor.predict_sets(p)
    sets = conf_df["prediction_set"].values
    sizes = conf_df["set_size"].values

    # Determine empirical coverage per instance
    covered = np.zeros(len(y), dtype=bool)
    for i in range(len(y)):
        yi = y[i]
        si = sets[i]
        if yi == 0 and "Legitimate" in si:
            covered[i] = True
        elif yi == 1 and "Fraud" in si:
            covered[i] = True

    rows = []
    for g in groups:
        m = (sub == g)
        n_g = int(np.sum(m))
        if n_g == 0:
            continue

        cov_g = covered[m]
        k_cov = int(np.sum(cov_g))
        cov_pt, cov_low, cov_high = wilson_score_interval(k_cov, n_g, confidence)

        sets_g = sets[m]
        sizes_g = sizes[m]

        pct_singleton_0 = float(np.mean(sets_g == "{Legitimate}"))
        pct_singleton_1 = float(np.mean(sets_g == "{Fraud}"))
        pct_ambiguous = float(np.mean(sets_g == "{Legitimate, Fraud}"))
        avg_size = float(np.mean(sizes_g))

        rows.append({
            "subgroup_dimension": subgroup_name,
            "subgroup": g,
            "sample_size": n_g,
            "empirical_coverage": round(cov_pt, 4),
            "coverage_ci_95": format_ci(cov_pt, cov_low, cov_high),
            "target_coverage": 1.0 - conformal_predictor.alpha,
            "coverage_deficit": round(max(0.0, (1.0 - conformal_predictor.alpha) - cov_pt), 4),
            "average_set_size": round(avg_size, 4),
            "pct_singleton_legit": round(pct_singleton_0 * 100.0, 2),
            "pct_singleton_fraud": round(pct_singleton_1 * 100.0, 2),
            "pct_ambiguous_set": round(pct_ambiguous * 100.0, 2),
            "exchangeability_status": "Covariate Shift (Empirical Only)",
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. Temporal Stability Tracking
# ----------------------------------------------------------------------------
def evaluate_temporal_stability(
    df_raw: pd.DataFrame,
    proba: np.ndarray,
    actions: np.ndarray,
    conformal_predictor: ConformalFraudPredictor,
    time_col: str = "month",
    test_months: List[int] = [6, 7],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compare performance, calibration, and conformal coverage across distinct temporal test periods."""
    y_all = df_raw[config.TARGET].values
    m_all = df_raw[time_col].values

    rows = []
    for tm in test_months:
        idx = np.where(m_all == tm)[0]
        n_m = len(idx)
        if n_m == 0:
            continue

        y_m = y_all[idx]
        p_m = proba[idx]
        act_m = actions[idx]

        n_fraud_m = int(np.sum(y_m == 1))
        fraud_prev_m = float(n_fraud_m / n_m)

        # ROC & PR AUC
        roc_m = float(roc_auc_score(y_m, p_m)) if len(np.unique(y_m)) > 1 else np.nan
        pr_m = float(average_precision_score(y_m, p_m)) if len(np.unique(y_m)) > 1 else np.nan

        # Triage metrics (favorable = AUTO_APPROVE)
        flagged_m = (act_m != "AUTO_APPROVE").astype(int)
        tn, fp, fn, tp = confusion_matrix(y_m, flagged_m, labels=[0, 1]).ravel()

        rec_pt, rec_low, rec_high = wilson_score_interval(tp, tp + fn, confidence)
        fpr_pt, fpr_low, fpr_high = wilson_score_interval(fp, fp + tn, confidence)
        prec_pt, prec_low, prec_high = wilson_score_interval(tp, tp + fp, confidence)

        # Conformal coverage on month tm
        cov_m = conformal_predictor.evaluate_coverage(y_m, p_m)

        rows.append({
            "test_month": tm,
            "sample_size": n_m,
            "fraud_count": n_fraud_m,
            "fraud_prevalence": round(fraud_prev_m, 4),
            "ROC_AUC": round(roc_m, 4),
            "PR_AUC": round(pr_m, 4),
            "recall": round(rec_pt, 4),
            "recall_ci_95": format_ci(rec_pt, rec_low, rec_high),
            "fpr": round(fpr_pt, 4),
            "fpr_ci_95": format_ci(fpr_pt, fpr_low, fpr_high),
            "precision": round(prec_pt, 4),
            "precision_ci_95": format_ci(prec_pt, prec_low, prec_high),
            "pct_auto_approve": round(float(np.mean(act_m == "AUTO_APPROVE")) * 100.0, 2),
            "pct_auto_flag": round(float(np.mean(act_m == "AUTO_FLAG")) * 100.0, 2),
            "pct_human_review": round(float(np.mean(act_m == "HUMAN_REVIEW")) * 100.0, 2),
            "conformal_empirical_coverage": f"{cov_m['empirical_coverage']*100.0:.2f}%",
            "conformal_avg_set_size": round(cov_m["avg_set_size"], 4),
        })

    return pd.DataFrame(rows)

