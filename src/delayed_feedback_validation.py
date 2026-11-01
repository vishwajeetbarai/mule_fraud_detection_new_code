"""Phase 5 Delayed-Label Feedback & Temporal Prequential Validation.

Evaluates adaptive threshold calibration (TDTA) and the Hierarchical Triage Policy
under realistic verification latency (0, 7, 14, 30, and 60 days).

Zero-Leakage Guarantees:
- Decisions are made strictly at application time.
- Labels become accessible ONLY after t_verified = t_apply + delay_days.
- Threshold adaptations apply forward to subsequent decision periods, NEVER retroactively.
- Conformal prediction remains calibrated on validation data (Month 5).
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
import evaluation as ev
from adaptive_threshold import AdaptiveThresholdCalibrator
from drift_detector import TemporalDriftDetector
from features import FeaturePreprocessor
from hierarchical_triage import HierarchicalTriagePolicy
from statistical_reliability import wilson_score_interval


def build_application_timeline(
    df_test: pd.DataFrame,
    start_month: int = 6,
    days_per_month: float = 30.0,
) -> pd.DataFrame:
    """Assigns continuous application timestamps to chronological test applications."""
    df = df_test.copy().reset_index(drop=True)
    timestamps = np.zeros(len(df), dtype=float)

    for m in df[config.TIME_COL].unique():
        mask = (df[config.TIME_COL] == m)
        n_m = int(mask.sum())
        if n_m > 0:
            # Normalized arrival times strictly inside month: [m * 30 + step, (m+1) * 30]
            month_offset = float(m) * days_per_month
            step = days_per_month / max(n_m, 1)
            timestamps[mask] = month_offset + (np.arange(n_m) + 1.0) * (days_per_month / (n_m + 1.0))

    df["t_apply"] = timestamps
    return df


def run_delayed_feedback_evaluation(
    model: Any,
    preprocessor: FeaturePreprocessor,
    df_raw: pd.DataFrame,
    initial_val_month: int = 5,
    test_months: Optional[List[int]] = None,
    delay_days: float = 0.0,
    policy_name: str = "TDTA",
    is_static: bool = False,
    drift_method: str = "psi",
    psi_threshold: float = 0.10,
    target_fpr: float = config.TARGET_FPR,
    conformal_predictor: Optional[Any] = None,
    cost_fn: float = config.COST_FN,
    cost_fp: float = config.COST_FP,
    cost_tp: float = config.COST_TP,
) -> Dict[str, Any]:
    """Evaluates a threshold adaptation policy under specified feedback delay in days."""
    if test_months is None:
        test_months = list(config.TEST_MONTHS)

    # 1. Validation Month 5 Reference Setup
    df_val = df_raw[df_raw[config.TIME_COL] == initial_val_month].reset_index(drop=True)
    X_val = preprocessor.transform(df_val)
    y_val = df_val[config.TARGET].values
    val_proba = model.predict_proba(X_val)[:, 1]

    # Baseline threshold targeting 5% FPR on validation month
    val_target_thr = float(ev.threshold_for_fpr(y_val, val_proba, target_fpr))

    # Optimal cost threshold on validation data (zero leakage)
    opt_cost_res = ev.optimize_cost_threshold(y_val, val_proba, cost_fn, cost_fp, cost_tp)
    val_cost_thr = float(opt_cost_res["optimal_threshold"])

    # Setup calibrator
    calibrator = AdaptiveThresholdCalibrator(
        target_fpr=target_fpr,
        initial_threshold=val_target_thr,
        min_samples=config.CALIBRATION_MIN_SAMPLES,
        min_negatives=config.CALIBRATION_MIN_NEGATIVES,
        min_positives=config.CALIBRATION_MIN_POSITIVES,
    )

    # Setup drift detector on validation reference
    detector = TemporalDriftDetector(
        features=config.DRIFT_MONITORING_FEATURES,
        method=drift_method,
        psi_threshold=psi_threshold,
        monitor_score=config.DRIFT_SCORE_MONITORING,
        score_psi_threshold=config.DRIFT_SCORE_PSI_THRESHOLD,
    )
    detector.fit_reference(df_val, scores_ref=val_proba)

    # Build continuous timeline for test records
    df_test_raw = df_raw[df_raw[config.TIME_COL].isin(test_months)].copy().reset_index(drop=True)
    df_test_timeline = build_application_timeline(df_test_raw)

    # Precompute model predictions on test set (frozen model)
    X_test_all = preprocessor.transform(df_test_timeline)
    probas_test_all = model.predict_proba(X_test_all)[:, 1]
    df_test_timeline["proba"] = probas_test_all

    # Assign verification time: t_verified = t_apply + delay_days
    df_test_timeline["t_verified"] = df_test_timeline["t_apply"] + float(delay_days)

    # Historical label pool initialized with validation data (already verified by month 6)
    # Validation timestamps: Month 5 arrival (t_apply in [150, 180])
    historical_val_records = pd.DataFrame({
        "t_apply": np.linspace(150.0, 180.0, len(y_val)),
        "t_verified": np.linspace(150.0, 180.0, len(y_val)),
        "y_true": y_val,
        "proba": val_proba,
        "month": initial_val_month,
    })

    # Historical pool containing all verified records
    history_pool = [historical_val_records]

    active_threshold = val_target_thr
    threshold_history: List[Dict[str, Any]] = [{
        "scenario": f"{policy_name}_{int(delay_days)}d" if not is_static else "Static_Validation_Threshold",
        "month": initial_val_month,
        "threshold": active_threshold,
        "reason": "validation_initialization",
    }]

    availability_logs: List[Dict[str, Any]] = []
    month_evaluation_rows: List[Dict[str, Any]] = []
    month_triage_rows: List[Dict[str, Any]] = []

    # Sequential processing month-by-month
    for i, m in enumerate(test_months):
        m_mask = (df_test_timeline[config.TIME_COL] == m)
        df_m = df_test_timeline[m_mask].copy().reset_index(drop=True)
        if df_m.empty:
            continue

        start_of_month_t = float(m) * 30.0
        y_m = df_m[config.TARGET].values
        proba_m = df_m["proba"].values
        n_m = len(y_m)

        # -------------------------------------------------------------
        # STEP A: Make decisions using current active threshold
        # -------------------------------------------------------------
        current_thr = active_threshold
        preds_binary_m = (proba_m >= current_thr).astype(int)

        # Hierarchical triage routing at current active threshold
        triage_policy_m = HierarchicalTriagePolicy(
            tau_cost=val_cost_thr,
            tau_tdta=current_thr,
            conformal_predictor=conformal_predictor,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            cost_tp=cost_tp,
        )
        triage_res_m = triage_policy_m.route(proba_m)
        actions_m = triage_res_m["action"].values

        # -------------------------------------------------------------
        # STEP B: Measure drift using current unlabeled observations
        # -------------------------------------------------------------
        drift_report = detector.detect(df_m, scores_eval=proba_m, period_id=m)
        drift_detected = drift_report["drift_detected"]

        # -------------------------------------------------------------
        # STEP C: Record metrics for month m
        # -------------------------------------------------------------
        cm = ev.confusion_matrix(y_m, preds_binary_m)
        if isinstance(cm, dict):
            tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
        elif hasattr(cm, "ravel") and len(cm.ravel()) == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = int(np.sum((y_m == 0) & (preds_binary_m == 0)))
            fp = int(np.sum((y_m == 0) & (preds_binary_m == 1)))
            fn = int(np.sum((y_m == 1) & (preds_binary_m == 0)))
            tp = int(np.sum((y_m == 1) & (preds_binary_m == 1)))

        cost_val = ev.compute_cost_matrix(y_m, preds_binary_m, cost_fn, cost_fp, cost_tp)
        cost_m = float(cost_val["cost_per_application"])

        fraud_mask_m = (y_m == 1)
        n_frauds_m = int(np.sum(fraud_mask_m))

        frauds_flagged_m = int(np.sum((actions_m == "AUTO_FLAG") & fraud_mask_m))
        frauds_review_m = int(np.sum((actions_m == "HUMAN_REVIEW") & fraud_mask_m))
        frauds_approve_m = int(np.sum((actions_m == "AUTO_APPROVE") & fraud_mask_m))

        month_evaluation_rows.append({
            "month": m,
            "threshold": current_thr,
            "ROC_AUC": float(roc_auc_score(y_m, proba_m)),
            "PR_AUC": float(average_precision_score(y_m, proba_m)),
            "Precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "Recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "FPR": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "TPR_at_1pct_FPR": float(ev.tpr_at_fpr(y_m, proba_m, 0.01)),
            "TPR_at_5pct_FPR": float(ev.tpr_at_fpr(y_m, proba_m, 0.05)),
            "cost_per_applicant": cost_m,
            "pct_auto_approve": float(np.mean(actions_m == "AUTO_APPROVE") * 100.0),
            "pct_human_review": float(np.mean(actions_m == "HUMAN_REVIEW") * 100.0),
            "pct_auto_flag": float(np.mean(actions_m == "AUTO_FLAG") * 100.0),
            "frauds_auto_flagged": frauds_flagged_m,
            "frauds_referred_review": frauds_review_m,
            "frauds_erroneously_approved": frauds_approve_m,
            "direct_flag_recall": float(frauds_flagged_m / n_frauds_m) if n_frauds_m > 0 else 0.0,
            "triage_recall": float((frauds_flagged_m + frauds_review_m) / n_frauds_m) if n_frauds_m > 0 else 0.0,
            "drift_detected": drift_detected,
        })

        # Add month m's records to tracking pool
        history_pool.append(pd.DataFrame({
            "t_apply": df_m["t_apply"].values,
            "t_verified": df_m["t_verified"].values,
            "y_true": y_m,
            "proba": proba_m,
            "month": m,
        }))

        # -------------------------------------------------------------
        # STEP D: Threshold Adaptation for Next Period (m + 1)
        # -------------------------------------------------------------
        if (i + 1 < len(test_months)):
            next_month = test_months[i + 1]
            end_of_current_month_t = float(next_month) * 30.0

            # Gather ALL records verified prior to end_of_current_month_t
            full_pool_df = pd.concat(history_pool, ignore_index=True)
            eligible_verified_mask = (full_pool_df["t_verified"] <= end_of_current_month_t)
            eligible_df = full_pool_df[eligible_verified_mask]

            n_available_labels = len(eligible_df)
            n_verified_month_m = int(np.sum((eligible_df["month"] == m)))

            # Audit log entry for label availability
            availability_logs.append({
                "decision_point_t": end_of_current_month_t,
                "current_month": m,
                "next_month": next_month,
                "delay_days": delay_days,
                "policy": policy_name if not is_static else "Static",
                "total_available_labels": n_available_labels,
                "newly_verified_from_month_m": n_verified_month_m,
                "unverified_pending_from_month_m": n_m - n_verified_month_m,
                "active_threshold": current_thr,
                "drift_detected": drift_detected,
                "recalibration_triggered": bool(drift_detected and not is_static),
            })

            # Execute recalibration if policy allows and drift triggered
            if not is_static and drift_detected:
                calib_y = eligible_df["y_true"].values
                calib_p = eligible_df["proba"].values

                # Use recent calibration window (most recent 30-day verified slice)
                recent_calib_mask = (eligible_df["t_verified"] >= end_of_current_month_t - 30.0)
                if np.sum(recent_calib_mask) >= config.CALIBRATION_MIN_SAMPLES:
                    calib_y = eligible_df.loc[recent_calib_mask, "y_true"].values
                    calib_p = eligible_df.loc[recent_calib_mask, "proba"].values

                new_thr, cal_log = calibrator.calibrate(
                    calib_y_true=calib_y,
                    calib_proba=calib_p,
                    trigger_reason=f"drift_alert_month_{m}_delay_{delay_days}d",
                    period_id=next_month,
                    force_recalibrate=True,
                )
                active_threshold = new_thr
                threshold_history.append({
                    "scenario": f"{policy_name}_{int(delay_days)}d",
                    "month": next_month,
                    "threshold": active_threshold,
                    "reason": cal_log["status"],
                })
            else:
                threshold_history.append({
                    "scenario": f"{policy_name}_{int(delay_days)}d" if not is_static else "Static_Validation_Threshold",
                    "month": next_month,
                    "threshold": active_threshold,
                    "reason": "static_or_no_drift",
                })

    # Combine overall out-of-sample metrics across test months
    all_y_test = df_test_timeline[config.TARGET].values
    all_proba_test = df_test_timeline["proba"].values
    n_total_test = len(all_y_test)
    n_frauds_total = int(np.sum(all_y_test == 1))

    # Overall predictions using time-varying active thresholds
    all_preds_binary = np.zeros(n_total_test, dtype=int)
    all_actions = np.empty(n_total_test, dtype=object)

    for r in month_evaluation_rows:
        m = r["month"]
        thr_m = r["threshold"]
        m_idx = (df_test_timeline[config.TIME_COL] == m).values
        all_preds_binary[m_idx] = (all_proba_test[m_idx] >= thr_m).astype(int)

        triage_m = HierarchicalTriagePolicy(
            tau_cost=val_cost_thr,
            tau_tdta=thr_m,
            conformal_predictor=conformal_predictor,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            cost_tp=cost_tp,
        )
        routed_m = triage_m.route(all_proba_test[m_idx])
        all_actions[m_idx] = routed_m["action"].values

    # Aggregate metrics
    cm_all = ev.confusion_matrix(all_y_test, all_preds_binary)
    if isinstance(cm_all, dict):
        tp, fp, fn, tn = cm_all["TP"], cm_all["FP"], cm_all["FN"], cm_all["TN"]
    elif hasattr(cm_all, "ravel") and len(cm_all.ravel()) == 4:
        tn, fp, fn, tp = cm_all.ravel()
    else:
        tn = int(np.sum((all_y_test == 0) & (all_preds_binary == 0)))
        fp = int(np.sum((all_y_test == 0) & (all_preds_binary == 1)))
        fn = int(np.sum((all_y_test == 1) & (all_preds_binary == 0)))
        tp = int(np.sum((all_y_test == 1) & (all_preds_binary == 1)))

    overall_cost = float(ev.compute_cost_matrix(all_y_test, all_preds_binary, cost_fn, cost_fp, cost_tp)["cost_per_application"])

    fraud_mask_all = (all_y_test == 1)
    frauds_auto_flagged = int(np.sum((all_actions == "AUTO_FLAG") & fraud_mask_all))
    frauds_referred_review = int(np.sum((all_actions == "HUMAN_REVIEW") & fraud_mask_all))
    frauds_erroneously_approved = int(np.sum((all_actions == "AUTO_APPROVE") & fraud_mask_all))

    triage_recall = float((frauds_auto_flagged + frauds_referred_review) / n_frauds_total) if n_frauds_total > 0 else 0.0
    direct_flag_recall = float(frauds_auto_flagged / n_frauds_total) if n_frauds_total > 0 else 0.0

    _, rec_low, rec_high = wilson_score_interval(frauds_auto_flagged + frauds_referred_review, n_frauds_total)

    scenario_label = f"Scenario {policy_name} ({int(delay_days)}d lag)" if not is_static else "Scenario A: Static Threshold"

    summary_result = {
        "scenario": scenario_label,
        "policy": policy_name if not is_static else "Static",
        "delay_days": delay_days,
        "is_static": is_static,
        "test_sample_size": n_total_test,
        "total_frauds": n_frauds_total,
        "final_threshold": active_threshold,
        "ROC_AUC": float(roc_auc_score(all_y_test, all_proba_test)),
        "PR_AUC": float(average_precision_score(all_y_test, all_proba_test)),
        "Precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "Recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "FPR": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "TPR_at_1pct_FPR": float(ev.tpr_at_fpr(all_y_test, all_proba_test, 0.01)),
        "TPR_at_5pct_FPR": float(ev.tpr_at_fpr(all_y_test, all_proba_test, 0.05)),
        "avg_cost_per_applicant": overall_cost,
        "pct_auto_approve": float(np.mean(all_actions == "AUTO_APPROVE") * 100.0),
        "pct_human_review": float(np.mean(all_actions == "HUMAN_REVIEW") * 100.0),
        "pct_auto_flag": float(np.mean(all_actions == "AUTO_FLAG") * 100.0),
        "frauds_auto_flagged": frauds_auto_flagged,
        "frauds_referred_review": frauds_referred_review,
        "frauds_erroneously_approved": frauds_erroneously_approved,
        "direct_flag_recall": direct_flag_recall,
        "triage_recall": triage_recall,
        "triage_recall_ci_95": f"{triage_recall:.4f} [{rec_low:.4f}, {rec_high:.4f}]",
    }

    return {
        "summary": summary_result,
        "threshold_history": threshold_history,
        "availability_logs": availability_logs,
        "month_evaluations": month_evaluation_rows,
    }
