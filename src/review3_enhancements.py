"""Review 3 Enhancements: Multi-Scenario Cost, Capacity Triage, and Fairness Audits.

Leakage-Safe Design:
- In evaluate_multi_scenario_cost():
  Thresholds are selected strictly on VALIDATION data (y_val, proba_val).
  Selected thresholds are then frozen and applied to TEST data (y_test, proba_test).
  Test labels are used solely to evaluate out-of-sample confusion matrix and realized cost.
- Cost formulation uses consistent financial parameters:
  Total cost = FN * cost_fn + FP * cost_fp + TP * cost_tp + TN * cost_tn.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import scipy.stats as stats
import config


def evaluate_multi_scenario_cost(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    y_test: np.ndarray,
    proba_test: np.ndarray,
    cost_tp: float = config.COST_TP,
    cost_tn: float = config.COST_TN,
) -> pd.DataFrame:
    """Evaluate multi-scenario cost without test-label leakage.

    Parameters
    ----------
    y_val : np.ndarray
        Ground-truth labels from validation month (Month 5).
    proba_val : np.ndarray
        Model predicted fraud probabilities for validation month.
    y_test : np.ndarray
        Ground-truth labels from test months (Months 6-7).
    proba_test : np.ndarray
        Model predicted fraud probabilities for test months.
    cost_tp : float
        Cost of true positive (automated step-up KYC verification friction).
    cost_tn : float
        Cost of seamless legitimate approval.

    Returns
    -------
    pd.DataFrame
        Table with validation-selected threshold, expected validation cost,
        and realized out-of-sample test cost/metrics.
    """
    scenarios = {
        'Scenario_A_LowFN_HighFP': {'cost_fn': 500.0, 'cost_fp': 100.0},
        'Scenario_B_HighFN_LowFP': {'cost_fn': 1500.0, 'cost_fp': 30.0},
        'Scenario_C_SevereFN_ModFP': {'cost_fn': 3000.0, 'cost_fp': 50.0},
    }

    y_v = np.asarray(y_val, dtype=int)
    p_v = np.asarray(proba_val, dtype=float)
    y_t = np.asarray(y_test, dtype=int)
    p_t = np.asarray(proba_test, dtype=float)

    thresholds = np.linspace(0.005, 0.995, 200)
    rows = []

    for name, params in scenarios.items():
        cfn = float(params['cost_fn'])
        cfp = float(params['cost_fp'])

        # 1. Select optimal threshold STRICTLY on Validation Data (Month 5)
        val_costs = []
        for t in thresholds:
            pred_v = (p_v >= t).astype(int)
            tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_v, pred_v, labels=[0, 1]).ravel()
            val_cost = (fn_v * cfn + fp_v * cfp + tp_v * cost_tp + tn_v * cost_tn) / len(y_v)
            val_costs.append(val_cost)

        best_i = int(np.argmin(val_costs))
        frozen_thr = float(thresholds[best_i])
        val_expected_cost = float(val_costs[best_i])

        # 2. Apply Frozen Threshold OUT-OF-SAMPLE to Test Data (Months 6-7)
        pred_test = (p_t >= frozen_thr).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_t, pred_test, labels=[0, 1]).ravel()
        realized_cost = (fn_t * cfn + fp_t * cfp + tp_t * cost_tp + tn_t * cost_tn) / len(y_t)
        prec_t = float(precision_score(y_t, pred_test, zero_division=0))
        rec_t = float(recall_score(y_t, pred_test, zero_division=0))
        fpr_t = float(fp_t / max(fp_t + tn_t, 1))

        # Default 0.5 threshold benchmark on test set
        pred_def = (p_t >= 0.5).astype(int)
        tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_t, pred_def, labels=[0, 1]).ravel()
        default_cost = (fn_d * cfn + fp_d * cfp + tp_d * cost_tp + tn_d * cost_tn) / len(y_t)
        savings_vs_default = default_cost - realized_cost
        savings_pct = (savings_vs_default / default_cost) * 100.0 if default_cost > 0 else 0.0

        rows.append({
            'scenario': name,
            'cost_fn': cfn,
            'cost_fp': cfp,
            'cost_tp': cost_tp,
            'val_selected_threshold': round(frozen_thr, 4),
            'val_expected_cost_per_app': round(val_expected_cost, 2),
            'realized_test_cost_per_app': round(realized_cost, 2),
            'default_05_test_cost_per_app': round(default_cost, 2),
            'savings_pct_vs_default': round(savings_pct, 1),
            'test_precision': round(prec_t, 4),
            'test_recall': round(rec_t, 4),
            'test_fpr': round(fpr_t, 4),
            'test_tp': int(tp_t),
            'test_fp': int(fp_t),
            'test_fn': int(fn_t),
            'test_tn': int(tn_t),
        })

    return pd.DataFrame(rows)


def evaluate_capacity_constrained(y_true, proba, capacities=[0.01, 0.02, 0.05]):
    """Evaluate detection metrics at fixed investigation capacities (ranking-based)."""
    y = np.asarray(y_true)
    p = np.asarray(proba)
    n = len(y)
    rows = []
    order = np.argsort(p)[::-1]
    for cap in capacities:
        k = max(1, int(n * cap))
        top_idx = order[:k]
        tp = int(np.sum(y[top_idx] == 1))
        total_pos = max(1, int(np.sum(y == 1)))
        rows.append({
            'investigation_capacity': f"Top {cap:.0%}",
            'n_investigated': k,
            'frauds_detected': tp,
            'precision': round(tp / k, 4),
            'recall': round(tp / total_pos, 4),
            'threshold_cutoff': round(float(p[top_idx[-1]]), 4),
        })
    return pd.DataFrame(rows)


from statistical_reliability import wilson_score_interval, format_ci

def evaluate_enhanced_fairness(y_true, proba, threshold, slice_series, slice_name='customer_age'):
    """Compute group-wise error rates and fairness compliance with exact Wilson Score CIs."""
    y = np.asarray(y_true)
    p = np.asarray(proba)
    pred = (p >= threshold).astype(int)
    slice_str = slice_series.astype(str).values
    groups = sorted(np.unique(slice_str))
    rows = []
    for g in groups:
        m = (slice_str == g)
        n_g = int(np.sum(m))
        if n_g == 0:
            continue
        y_g = y[m]
        pred_g = pred[m]
        tn, fp, fn, tp = confusion_matrix(y_g, pred_g, labels=[0, 1]).ravel()
        pos_g = tp + fn
        neg_g = fp + tn
        flagged_g = tp + fp
        
        # Exact Wilson Score CIs for binomial proportions
        k_app = int(np.sum(pred_g == 0))
        app_pt, app_low, app_high = wilson_score_interval(k_app, n_g, 0.95)
        rec_pt, rec_low, rec_high = wilson_score_interval(tp, pos_g, 0.95)
        fpr_pt, fpr_low, fpr_high = wilson_score_interval(fp, neg_g, 0.95)
        prec_pt, prec_low, prec_high = wilson_score_interval(tp, flagged_g, 0.95)
        fnr_pt, fnr_low, fnr_high = wilson_score_interval(fn, pos_g, 0.95)

        rows.append({
            'slice': slice_name,
            'group': g,
            'sample_size': n_g,
            'fraud_count': pos_g,
            'approval_rate': round(app_pt, 4),
            'approval_ci_95': format_ci(app_pt, app_low, app_high),
            'group_fpr': round(fpr_pt, 4),
            'fpr_ci_95': format_ci(fpr_pt, fpr_low, fpr_high),
            'group_fnr': round(fnr_pt, 4),
            'group_recall': round(rec_pt, 4),
            'recall_ci_95': format_ci(rec_pt, rec_low, rec_high),
            'group_precision': round(prec_pt, 4),
            'precision_ci_95': format_ci(prec_pt, prec_low, prec_high),
            'statistical_reliability': 'Sufficient' if n_g >= 100 and pos_g >= 5 else 'Small Sample (High Uncertainty)',
        })
    df_rep = pd.DataFrame(rows)
    ref_rate = df_rep['approval_rate'].max()
    df_rep['disparate_impact_ratio'] = round(df_rep['approval_rate'] / max(0.0001, ref_rate), 4)
    df_rep['four_fifths_compliant'] = df_rep['disparate_impact_ratio'] >= 0.80
    return df_rep
