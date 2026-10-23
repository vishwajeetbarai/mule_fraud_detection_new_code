"""Temporal Drift-Aware Threshold Adaptation (TDTA) Sequential Policy & Experiment Runner.

Orchestrates sequential prequential evaluation over chronological periods (months):
1. Evaluates baseline fixed threshold policy.
2. Evaluates drift-monitoring only policy (logging drift without adapting threshold).
3. Evaluates TDTA policy (recalibrating threshold when drift detector flags alert).
4. Evaluates scheduled recalibration policy (recalibrating on every period regardless of drift).
5. Executes ablation & sensitivity sweeps across trigger thresholds, calibration windows,
   label delays, and models (XGBoost vs LightGBM).

Zero Data Leakage Protocol:
- Historical training: fitted on training months (0-4).
- Initial threshold: calibrated on validation month (5).
- Test periods: scored chronologically (month 6, then month 7).
- In month m:
  a. Applications are scored using features available at scoring time.
  b. Drift is measured comparing month m features/scores against reference (validation).
  c. If policy adapts for month m + 1, ONLY labels up to month m - label_delay are used.
  d. Threshold is applied forward to the subsequent test period, NEVER retroactively to
     the period that triggered it.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from data_loader import describe, load_base_dataset, temporal_split
from drift_detector import TemporalDriftDetector
from features import FeaturePreprocessor


def run_sequential_policy(
    model: Any,
    preprocessor: FeaturePreprocessor,
    df_raw: pd.DataFrame,
    initial_val_month: int = 5,
    test_months: Optional[List[int]] = None,
    policy_type: str = "tdta",  # "baseline", "drift_monitoring", "tdta", "scheduled"
    drift_method: str = "psi",
    psi_threshold: float = 0.10,
    target_fpr: float = config.TARGET_FPR,
    calibration_window: int = 1,
    label_delay: int = 0,
    model_name: str = "LightGBM",
) -> Dict[str, Any]:
    """Run sequential evaluation on chronological test periods under a specified threshold policy.
    
    Parameters
    ----------
    model : fitted classifier
    preprocessor : fitted FeaturePreprocessor
    df_raw : pd.DataFrame
        Complete raw dataset with month and fraud_bool.
    initial_val_month : int
        Month used to establish initial threshold (default 5).
    test_months : List[int]
        Chronological test months (default [6, 7]).
    policy_type : str
        'baseline': Fixed threshold from validation set.
        'drift_monitoring': Fixed threshold + drift logging.
        'tdta': Recalibrate only when drift triggered.
        'scheduled': Recalibrate every period unconditionally.
    """
    if test_months is None:
        test_months = list(config.TEST_MONTHS)

    # 1. Establish initial threshold on validation month
    df_val = df_raw[df_raw[config.TIME_COL] == initial_val_month].reset_index(drop=True)
    X_val = preprocessor.transform(df_val)
    y_val = df_val[config.TARGET].values
    val_proba = model.predict_proba(X_val)[:, 1]

    # Baseline threshold targeting 5% FPR on validation set
    initial_threshold = ev.threshold_for_fpr(y_val, val_proba, target_fpr)

    # Setup calibrator
    calibrator = AdaptiveThresholdCalibrator(
        target_fpr=target_fpr,
        initial_threshold=initial_threshold,
        min_samples=config.CALIBRATION_MIN_SAMPLES,
        min_negatives=config.CALIBRATION_MIN_NEGATIVES,
        min_positives=config.CALIBRATION_MIN_POSITIVES,
    )

    # Setup drift detector with reference = validation month features & scores
    detector = TemporalDriftDetector(
        features=config.DRIFT_MONITORING_FEATURES,
        method=drift_method,
        psi_threshold=psi_threshold,
        monitor_score=config.DRIFT_SCORE_MONITORING,
        score_psi_threshold=config.DRIFT_SCORE_PSI_THRESHOLD,
    )
    detector.fit_reference(df_val, scores_ref=val_proba)

    # Store cache of predictions and labels by month for delayed calibration
    month_cache = {
        initial_val_month: {
            "df": df_val,
            "y": y_val,
            "proba": val_proba,
        }
    }

    current_threshold = initial_threshold
    period_metrics = []
    drift_logs = []
    predictions_records = []

    # Chronological loop through test months
    for i, m in enumerate(test_months):
        df_m = df_raw[df_raw[config.TIME_COL] == m].reset_index(drop=True)
        if df_m.empty:
            continue

        X_m = preprocessor.transform(df_m)
        y_m = df_m[config.TARGET].values
        proba_m = model.predict_proba(X_m)[:, 1]

        # Record for future calibration pools
        month_cache[m] = {
            "df": df_m,
            "y": y_m,
            "proba": proba_m,
        }

        # Step A: Evaluate incoming batch m using CURRENT active threshold
        # (This ensures decisions on batch m are NEVER influenced by batch m labels or outcome drift)
        active_thr_for_m = current_threshold
        pred_m = (proba_m >= active_thr_for_m).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_m, pred_m, labels=[0, 1]).ravel()
        roc_auc = float(roc_auc_score(y_m, proba_m))
        pr_auc = float(average_precision_score(y_m, proba_m))
        prec = float(precision_score(y_m, pred_m, zero_division=0))
        rec = float(recall_score(y_m, pred_m, zero_division=0))
        f1 = float(f1_score(y_m, pred_m, zero_division=0))
        realized_fpr = float(fp / max(fp + tn, 1))

        # Step B: Compute drift on batch m using scoring features and model scores
        drift_report = detector.detect(df_m, scores_eval=proba_m, period_id=m)
        drift_detected = drift_report["drift_detected"]
        drift_logs.append({
            "policy": policy_type,
            "month": m,
            "model": model_name,
            "drift_detected": drift_detected,
            "alert_features": ", ".join(drift_report["alert_features"]),
            "trigger_reason": drift_report["trigger_reason"],
            "score_psi": drift_report["score_drift"]["score_psi"] if drift_report["score_drift"] else np.nan,
        })

        # Record period evaluation
        period_metrics.append({
            "policy": policy_type,
            "model": model_name,
            "month": m,
            "threshold": float(active_thr_for_m),
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Target_FPR": target_fpr,
            "Realized_FPR": realized_fpr,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "N_total": len(y_m),
            "N_fraud": int(np.sum(y_m)),
            "drift_detected": bool(drift_detected),
        })

        # Save sample predictions for record
        for idx in range(min(500, len(y_m))):
            predictions_records.append({
                "policy": policy_type,
                "model": model_name,
                "month": m,
                "idx": idx,
                "y_true": int(y_m[idx]),
                "proba": float(proba_m[idx]),
                "threshold": float(active_thr_for_m),
                "pred": int(pred_m[idx]),
            })

        # Step C: Threshold Adaptation Decision for SUBSEQUENT Month (m + 1)
        # Determine whether recalibration is triggered
        recalibrate_trigger = False
        recal_reason = "none"

        if policy_type == "baseline":
            recalibrate_trigger = False
        elif policy_type == "drift_monitoring":
            recalibrate_trigger = False
        elif policy_type == "tdta":
            if drift_detected:
                recalibrate_trigger = True
                recal_reason = f"drift_alert: {drift_report['trigger_reason']}"
        elif policy_type == "scheduled":
            recalibrate_trigger = True
            recal_reason = "scheduled_periodic"

        if recalibrate_trigger and (i + 1 < len(test_months)):
            # Eligible calibration pool: months strictly <= (m - label_delay)
            # and within the recent calibration_window
            max_eligible_month = m - label_delay
            eligible_months = [
                mon for mon in sorted(month_cache.keys())
                if mon <= max_eligible_month and mon > (max_eligible_month - calibration_window)
            ]

            if not eligible_months:
                # If delay prevents new labels, fall back to validation month
                eligible_months = [initial_val_month]

            calib_y_list = [month_cache[em]["y"] for em in eligible_months]
            calib_p_list = [month_cache[em]["proba"] for em in eligible_months]
            pool_y = np.concatenate(calib_y_list)
            pool_p = np.concatenate(calib_p_list)

            next_month = test_months[i + 1]
            updated_thr, log_entry = calibrator.calibrate(
                pool_y,
                pool_p,
                trigger_reason=recal_reason,
                period_id=next_month,
                force_recalibrate=True,
            )
            current_threshold = updated_thr

    # Aggregated metrics across all test months
    total_tp = sum(r["TP"] for r in period_metrics)
    total_fp = sum(r["FP"] for r in period_metrics)
    total_fn = sum(r["FN"] for r in period_metrics)
    total_tn = sum(r["TN"] for r in period_metrics)
    total_neg = total_fp + total_tn
    total_pos = total_tp + total_fn

    overall_prec = total_tp / max(total_tp + total_fp, 1)
    overall_rec = total_tp / max(total_pos, 1)
    overall_fpr = total_fp / max(total_neg, 1)
    overall_f1 = (2 * overall_prec * overall_rec) / max(overall_prec + overall_rec, 1e-12)

    all_y = np.concatenate([month_cache[m]["y"] for m in test_months])
    all_p = np.concatenate([month_cache[m]["proba"] for m in test_months])
    overall_roc_auc = float(roc_auc_score(all_y, all_p))
    overall_pr_auc = float(average_precision_score(all_y, all_p))

    summary = {
        "policy": policy_type,
        "model": model_name,
        "target_fpr": target_fpr,
        "drift_method": drift_method,
        "psi_threshold": psi_threshold,
        "calibration_window": calibration_window,
        "label_delay": label_delay,
        "overall_ROC_AUC": overall_roc_auc,
        "overall_PR_AUC": overall_pr_auc,
        "overall_Precision": overall_prec,
        "overall_Recall": overall_rec,
        "overall_F1": overall_f1,
        "overall_Realized_FPR": overall_fpr,
        "FPR_deviation_from_target": abs(overall_fpr - target_fpr),
        "total_TP": int(total_tp),
        "total_FP": int(total_fp),
        "total_FN": int(total_fn),
        "total_TN": int(total_tn),
        "total_drift_alerts": sum(1 for d in drift_logs if d["drift_detected"]),
        "total_threshold_updates": sum(
            1 for c in calibrator.calibration_history_ if c.get("threshold_changed", False)
        ),
        "initial_threshold": float(initial_threshold),
        "final_threshold": float(current_threshold),
    }

    return {
        "summary": summary,
        "period_metrics": period_metrics,
        "drift_logs": drift_logs,
        "audit_trail": calibrator.get_audit_trail(),
        "predictions_sample": pd.DataFrame(predictions_records),
    }


def run_all_research_experiments(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    sample_frac: Optional[float] = None,
    seed: int = config.RANDOM_STATE,
    models_to_test: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute complete suite of Experiments 1 to 5.
    
    Experiment 1: Baseline fixed validation threshold (XGBoost & LightGBM)
    Experiment 2: Drift monitoring only
    Experiment 3: TDTA (Temporal Drift-Aware Threshold Adaptation)
    Experiment 4: Scheduled recalibration without drift triggering
    Experiment 5: Ablation and sensitivity analysis (PSI thresholds, window sizes, label delays)
    """
    out_dir = Path(output_dir or config.OUTPUT_DIR)
    research_dir = out_dir / "research_tdta"
    research_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RESEARCH EXPERIMENT SUITE: TEMPORAL DRIFT-AWARE THRESHOLD ADAPTATION (TDTA)")
    print(f"Dataset: {data_path or config.DATA_PATH} | Sample Frac: {sample_frac} | Seed: {seed}")
    print("=" * 80)

    # 1. Load Data
    df = load_base_dataset(data_path, sample_frac, seed)
    train, valid, test = temporal_split(df)
    print(describe(df, "all"))
    print(describe(train, "train (months 0-4)"))
    print(describe(valid, "valid (month 5)"))
    print(describe(test, "test  (months 6-7)"))

    # 2. Fit Feature Preprocessor on train
    pre = FeaturePreprocessor().fit(train)
    X_tr = pre.transform(train)
    y_tr = train[config.TARGET].values
    X_va = pre.transform(valid)
    y_va = valid[config.TARGET].values

    # Check for existing models or fit
    if models_to_test is None:
        models_to_test = ["LightGBM", "XGBoost"]

    fitted_models = {}
    for name in models_to_test:
        model_path = out_dir / "models" / f"{name.lower()}.joblib"
        if model_path.exists() and sample_frac is None:
            print(f"Loading existing trained {name} from {model_path}...")
            fitted_models[name] = joblib.load(model_path)
        else:
            print(f"Fitting {name} on train split (months 0-4)...")
            from train import fit_lightgbm, fit_xgboost
            if name == "LightGBM":
                fitted_models[name] = fit_lightgbm(X_tr, y_tr, X_va, y_va, seed)
            elif name == "XGBoost":
                fitted_models[name] = fit_xgboost(X_tr, y_tr, X_va, y_va, seed)

    # Containers for results
    all_summaries = []
    all_period_metrics = []
    all_drift_logs = []
    all_calib_audits = []

    # ------------------------------------------------------------------------
    # Experiments 1 to 4 across models
    # ------------------------------------------------------------------------
    policies = ["baseline", "drift_monitoring", "tdta", "scheduled"]

    for model_name, model_obj in fitted_models.items():
        print(f"\n>>> Running Experiments 1-4 for {model_name}...")
        for pol in policies:
            res = run_sequential_policy(
                model=model_obj,
                preprocessor=pre,
                df_raw=df,
                initial_val_month=5,
                test_months=[6, 7],
                policy_type=pol,
                drift_method="psi",
                psi_threshold=0.10,
                target_fpr=config.TARGET_FPR,
                calibration_window=1,
                label_delay=0,
                model_name=model_name,
            )
            all_summaries.append(res["summary"])
            all_period_metrics.extend(res["period_metrics"])
            all_drift_logs.extend(res["drift_logs"])
            if not res["audit_trail"].empty:
                df_audit = res["audit_trail"].copy()
                df_audit["policy"] = pol
                df_audit["model"] = model_name
                all_calib_audits.append(df_audit)

            s = res["summary"]
            print(
                f"  [{model_name:8s} | {pol:16s}] Recall={s['overall_Recall']:.4f} "
                f"| Realized FPR={s['overall_Realized_FPR']:.4f} (Dev={s['FPR_deviation_from_target']:.4f}) "
                f"| F1={s['overall_F1']:.4f} | Alerts={s['total_drift_alerts']} | Updates={s['total_threshold_updates']}"
            )

    # ------------------------------------------------------------------------
    # Experiment 5: Ablation and Sensitivity Analysis
    # ------------------------------------------------------------------------
    print("\n>>> Running Experiment 5: Ablation and Sensitivity Analysis...")
    primary_model_name = "LightGBM" if "LightGBM" in fitted_models else models_to_test[0]
    primary_model = fitted_models[primary_model_name]

    # Vary PSI threshold
    psi_grid = [0.05, 0.10, 0.15, 0.25]
    for psi_th in psi_grid:
        res = run_sequential_policy(
            model=primary_model,
            preprocessor=pre,
            df_raw=df,
            policy_type="tdta",
            psi_threshold=psi_th,
            model_name=f"{primary_model_name}_psi_{psi_th}",
        )
        all_summaries.append(res["summary"])
        all_period_metrics.extend(res["period_metrics"])

    # Vary Calibration Window
    for cal_win in [1, 2]:
        res = run_sequential_policy(
            model=primary_model,
            preprocessor=pre,
            df_raw=df,
            policy_type="tdta",
            calibration_window=cal_win,
            model_name=f"{primary_model_name}_win_{cal_win}",
        )
        all_summaries.append(res["summary"])
        all_period_metrics.extend(res["period_metrics"])

    # Vary Target FPR
    for tgt_fpr in [0.01, 0.03, 0.05, 0.08]:
        res = run_sequential_policy(
            model=primary_model,
            preprocessor=pre,
            df_raw=df,
            policy_type="tdta",
            target_fpr=tgt_fpr,
            model_name=f"{primary_model_name}_target_fpr_{tgt_fpr}",
        )
        all_summaries.append(res["summary"])
        all_period_metrics.extend(res["period_metrics"])

    # Vary Label Delay (0 vs 1 month lag)
    for l_delay in [0, 1]:
        res = run_sequential_policy(
            model=primary_model,
            preprocessor=pre,
            df_raw=df,
            policy_type="tdta",
            label_delay=l_delay,
            model_name=f"{primary_model_name}_delay_{l_delay}m",
        )
        all_summaries.append(res["summary"])
        all_period_metrics.extend(res["period_metrics"])

    # ------------------------------------------------------------------------
    # Export Machine-Readable Datasets
    # ------------------------------------------------------------------------
    df_summaries = pd.DataFrame(all_summaries)
    df_periods = pd.DataFrame(all_period_metrics)
    df_drift = pd.DataFrame(all_drift_logs).drop_duplicates()
    df_audits = pd.concat(all_calib_audits, ignore_index=True) if all_calib_audits else pd.DataFrame()

    df_summaries.to_csv(research_dir / "experiment_summaries.csv", index=False)
    df_periods.to_csv(research_dir / "period_level_metrics.csv", index=False)
    df_drift.to_csv(research_dir / "drift_monitoring_logs.csv", index=False)
    df_audits.to_csv(research_dir / "calibration_audit_trail.csv", index=False)

    # Core policy comparison table
    core_comparison = df_summaries[
        df_summaries["model"].isin(["LightGBM", "XGBoost"])
        & (df_summaries["target_fpr"] == 0.05)
        & (df_summaries["label_delay"] == 0)
    ][[
        "model", "policy", "overall_Recall", "overall_Realized_FPR",
        "FPR_deviation_from_target", "overall_Precision", "overall_F1",
        "total_drift_alerts", "total_threshold_updates", "initial_threshold", "final_threshold"
    ]]
    core_comparison.to_csv(research_dir / "policy_comparison_table.csv", index=False)

    # Save JSON summary
    summary_json_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(data_path or config.DATA_PATH),
        "target_fpr": config.TARGET_FPR,
        "sample_frac": sample_frac,
        "comparison": core_comparison.to_dict(orient="records"),
    }
    with open(research_dir / "experiment_results_summary.json", "w") as f:
        json.dump(summary_json_payload, f, indent=2)

    # ------------------------------------------------------------------------
    # Generate Research Visualizations
    # ------------------------------------------------------------------------
    plot_research_figures(df_summaries, df_periods, research_dir)

    print(f"\nAll TDTA experiments and plots successfully saved to: {research_dir}")
    return {
        "summaries": df_summaries,
        "period_metrics": df_periods,
        "comparison_table": core_comparison,
        "research_dir": research_dir,
    }


def plot_research_figures(
    df_summaries: pd.DataFrame,
    df_periods: pd.DataFrame,
    output_dir: Path,
):
    """Generate publication-quality figures for the research report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    core = df_periods[
        df_periods["model"].isin(["LightGBM", "XGBoost"])
        & (df_periods["Target_FPR"] == 0.05)
    ]

    # 1. Temporal Performance Plot: Recall & Realized FPR across Months 6 and 7
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    policies = ["baseline", "drift_monitoring", "tdta", "scheduled"]
    markers = {"baseline": "o", "drift_monitoring": "s", "tdta": "^", "scheduled": "D"}
    colors = {"baseline": "#1f77b4", "drift_monitoring": "#7f7f7f", "tdta": "#2ca02c", "scheduled": "#d62728"}

    for pol in ["baseline", "tdta", "scheduled"]:
        sub = core[(core["model"] == "LightGBM") & (core["policy"] == pol)].sort_values("month")
        axes[0].plot(sub["month"], sub["Recall"], marker=markers[pol], color=colors[pol], lw=2, label=pol.upper())
        axes[1].plot(sub["month"], sub["Realized_FPR"], marker=markers[pol], color=colors[pol], lw=2, label=pol.upper())

    axes[0].set_title("LightGBM: Temporal Recall Across Test Months", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Chronological Test Month")
    axes[0].set_ylabel("Fraud Recall")
    axes[0].set_xticks([6, 7])
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("LightGBM: Realized False Positive Rate vs 5% Target", fontsize=11, fontweight="bold")
    axes[1].axhline(0.05, color="black", linestyle="--", lw=1.2, label="Target 5% FPR")
    axes[1].set_xlabel("Chronological Test Month")
    axes[1].set_ylabel("Realized FPR")
    axes[1].set_xticks([6, 7])
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "temporal_performance_comparison.png", dpi=160)
    plt.close()

    # 2. Threshold Evolution Plot across chronological periods
    plt.figure(figsize=(8, 4.8))
    for pol in ["baseline", "tdta", "scheduled"]:
        sub = core[(core["model"] == "LightGBM") & (core["policy"] == pol)].sort_values("month")
        plt.plot(sub["month"], sub["threshold"], marker=markers[pol], color=colors[pol], lw=2, label=pol.upper())
    plt.title("Decision Threshold Evolution Across Test Months (LightGBM)", fontsize=11, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Decision Score Threshold")
    plt.xticks([6, 7])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_evolution.png", dpi=160)
    plt.close()

    # 3. FPR Deviation & Stability Comparison (Bar Chart)
    comp_df = df_summaries[
        df_summaries["model"].isin(["LightGBM", "XGBoost"])
        & (df_summaries["target_fpr"] == 0.05)
        & (df_summaries["label_delay"] == 0)
    ]
    plt.figure(figsize=(9, 4.8))
    x = np.arange(len(comp_df))
    plt.bar(
        x, comp_df["FPR_deviation_from_target"] * 100,
        color=["#1f77b4", "#7f7f7f", "#2ca02c", "#d62728"] * (len(comp_df) // 4 + 1),
        alpha=0.85,
    )
    plt.xticks(
        x, [f"{r.model}\n{r.policy}" for _, r in comp_df.iterrows()],
        rotation=20, ha="right", fontsize=9,
    )
    plt.ylabel("Absolute FPR Deviation from 5% Target (pp)")
    plt.title("FPR Error Minimization: Proposed TDTA vs Baselines", fontsize=11, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fpr_deviation_comparison.png", dpi=160)
    plt.close()

    # 4. Sensitivity to Target FPR (Trade-off Curve)
    fpr_sweep = df_summaries[df_summaries["model"].str.startswith("LightGBM_target_fpr_")]
    if not fpr_sweep.empty:
        plt.figure(figsize=(7.5, 5))
        plt.plot(fpr_sweep["target_fpr"], fpr_sweep["overall_Recall"], marker="o", lw=2, color="#2ca02c", label="Recall")
        plt.plot(fpr_sweep["target_fpr"], fpr_sweep["overall_Realized_FPR"], marker="s", lw=2, color="#d62728", label="Realized FPR")
        plt.plot([0, 0.1], [0, 0.1], "k--", lw=1, label="Ideal Calibration Line")
        plt.xlabel("Configured Target FPR")
        plt.ylabel("Observed Metric")
        plt.title("TDTA Sensitivity across Target False Positive Rates", fontsize=11, fontweight="bold")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "sensitivity_target_fpr.png", dpi=160)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="TDTA Research Experiment Runner")
    parser.add_argument("--data", default=None, help="Path to BaseDataset.csv")
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR), help="Output directory")
    parser.add_argument("--sample-frac", type=float, default=None, help="Fraction of data for rapid smoke test")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args()

    run_all_research_experiments(
        data_path=args.data,
        output_dir=args.output_dir,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
