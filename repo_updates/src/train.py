"""End-to-end training / evaluation on the real BaseDataset.csv.

Usage (from the repository root):
    python src/train.py                         # full run
    python src/train.py --data /path/BaseDataset.csv
    python src/train.py --sample-frac 0.1 --skip-cv --skip-shap   # quick smoke test
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
# Model fitting (early stopping uses ONLY the validation months)
# ----------------------------------------------------------------------------
def _spw(y):
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def fit_xgboost(X_tr, y_tr, X_va, y_va, seed):
    model = XGBClassifier(
        **config.XGB_PARAMS, scale_pos_weight=_spw(y_tr),
        random_state=seed, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def fit_lightgbm(X_tr, y_tr, X_va, y_va, seed):
    model = lgb.LGBMClassifier(
        **config.LGB_PARAMS, scale_pos_weight=_spw(y_tr),
        random_state=seed, n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)],
        # first_metric_only: stop on average precision only (not on implicit logloss,
        # which degrades immediately when scale_pos_weight is large).
        callbacks=[lgb.early_stopping(config.LGB_EARLY_STOPPING_ROUNDS,
                                      first_metric_only=True, verbose=False)],
    )
    return model


def best_iterations(name, model):
    if name == "XGBoost":
        return int(model.best_iteration) + 1
    return int(model.best_iteration_)


FITTERS = {"XGBoost": fit_xgboost, "LightGBM": fit_lightgbm}


# ----------------------------------------------------------------------------
# Temporal cross-validation (expanding window, split on `month`)
# ----------------------------------------------------------------------------
def temporal_cv(df, seed):
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
        for name, fitter in FITTERS.items():
            m = fitter(X_tr, y_tr, X_va, y_va, seed)
            thr = ev.threshold_for_fpr(y_va, m.predict_proba(X_va)[:, 1], config.TARGET_FPR)
            met = ev.compute_metrics(y_te, m.predict_proba(X_te)[:, 1], thr)
            rows.append({"fold_test_month": test_m, "train_months": f"0-{val_m - 1}",
                         "model": name, "n_train": len(tr), "n_test": len(te), **met})
            print(f"  [CV] test month {test_m} | {name:8s} | "
                  f"AUC={met['ROC_AUC']:.4f} PR-AUC={met['PR_AUC']:.4f} "
                  f"TPR@5%FPR={met['TPR_at_5pct_FPR']:.4f}", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Explainability
# ----------------------------------------------------------------------------
def gain_importance(model, feature_names):
    if isinstance(model, XGBClassifier):
        score = model.get_booster().get_score(importance_type="gain")
        vals = [score.get(f, 0.0) for f in feature_names]  # booster keeps DataFrame column names
    else:
        vals = model.booster_.feature_importance(importance_type="gain")
    out = pd.DataFrame({"feature": feature_names, "gain": np.asarray(vals, dtype=float)})
    out["gain_share"] = out["gain"] / max(out["gain"].sum(), 1e-12)
    return out.sort_values("gain", ascending=False).reset_index(drop=True)


def run_shap(model, X_test, out_dir, seed):
    import matplotlib.pyplot as plt
    import shap

    n = min(config.SHAP_SAMPLE_SIZE, len(X_test))
    sample = X_test.sample(n=n, random_state=seed)
    sv = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(sv, list):  # older shap/lightgbm: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]

    shap_df = pd.DataFrame({"feature": sample.columns,
                            "mean_abs_shap": np.abs(sv).mean(axis=0)})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(out_dir / "shap_feature_importance_lgbm.csv", index=False)

    shap.summary_plot(sv, sample, plot_type="bar", max_display=20, show=False)
    plt.tight_layout(); plt.savefig(out_dir / "shap_bar_lgbm.png", dpi=150); plt.close()
    shap.summary_plot(sv, sample, max_display=20, show=False)
    plt.tight_layout(); plt.savefig(out_dir / "shap_beeswarm_lgbm.png", dpi=150); plt.close()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=None, help="Path to BaseDataset.csv")
    ap.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    ap.add_argument("--sample-frac", type=float, default=None,
                    help="Use a random fraction of the REAL rows (smoke tests only)")
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    t0 = time.time()

    print("=" * 70, "\nBANK ACCOUNT FRAUD DETECTION — BaseDataset.csv\n" + "=" * 70)
    df = load_base_dataset(args.data, args.sample_frac, args.seed)
    train, valid, test = temporal_split(df)
    print(describe(df, "all"), "\n" + describe(train, "train"),
          f"(months {config.TRAIN_MONTHS})\n" + describe(valid, "valid"),
          f"(months {config.VALID_MONTHS})\n" + describe(test, "test"),
          f"(months {config.TEST_MONTHS})")

    pre = FeaturePreprocessor().fit(train)
    X_tr, X_va, X_te = (pre.transform(d) for d in (train, valid, test))
    y_tr, y_va, y_te = (d[config.TARGET].values for d in (train, valid, test))
    print(f"Features: {X_tr.shape[1]} | scale_pos_weight={_spw(y_tr):.1f}")

    models, probas, rows = {}, {}, []
    for name, fitter in FITTERS.items():
        t = time.time()
        m = fitter(X_tr, y_tr, X_va, y_va, args.seed)
        p_va, p_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
        thr = ev.threshold_for_fpr(y_va, p_va, config.TARGET_FPR)  # chosen on VALID only
        met = ev.compute_metrics(y_te, p_te, thr)
        rows.append({"model": name, "best_iteration": best_iterations(name, m), **met})
        models[name], probas[name] = m, p_te
        print(f"[{name}] best_iter={rows[-1]['best_iteration']} | "
              f"AUC={met['ROC_AUC']:.4f} PR-AUC={met['PR_AUC']:.4f} "
              f"TPR@1%FPR={met['TPR_at_1pct_FPR']:.4f} TPR@5%FPR={met['TPR_at_5pct_FPR']:.4f} "
              f"| @valid-thr: P={met['Precision']:.3f} R={met['Recall']:.3f} "
              f"FPR={met['realized_FPR']:.3f} ({time.time() - t:.0f}s)", flush=True)
        joblib.dump(m, out_dir / "models" / f"{name.lower()}.joblib")
    joblib.dump(pre, out_dir / "models" / "preprocessor.joblib")

    pd.DataFrame(rows).to_csv(out_dir / "model_metrics_table.csv", index=False)
    pd.DataFrame({"row_in_test": np.arange(len(y_te)), config.TARGET: y_te,
                  "month": test[config.TIME_COL].values,
                  "xgboost_proba": probas["XGBoost"],
                  "lightgbm_proba": probas["LightGBM"]}
                 ).to_csv(out_dir / "test_predictions.csv", index=False)

    # Feature importance + plots
    for name in models:
        imp = gain_importance(models[name], list(X_tr.columns))
        imp.to_csv(out_dir / f"feature_importance_{name.lower()}.csv", index=False)
    ev.plot_importance(pd.read_csv(out_dir / "feature_importance_lightgbm.csv"), "gain",
                       "LightGBM feature importance (gain)",
                       out_dir / "lightgbm_feature_importance.png")
    ev.plot_roc(y_te, probas, out_dir / "roc_curve.png")
    ev.plot_pr(y_te, probas, out_dir / "precision_recall_curve.png")
    best = max(rows, key=lambda r: r["PR_AUC"])["model"]
    ev.plot_score_distribution(y_te, probas[best], best, out_dir / "fraud_probability_distribution.png")

    if not args.skip_shap:
        print("Computing SHAP (LightGBM)…", flush=True)
        run_shap(models["LightGBM"], X_te, out_dir, args.seed)

    if not args.skip_cv:
        print("Temporal cross-validation (expanding window)…", flush=True)
        cv = temporal_cv(df, args.seed)
        cv.to_csv(out_dir / "cv_results_5fold_temporal.csv", index=False)
        summ = cv.groupby("model")[["ROC_AUC", "PR_AUC", "TPR_at_1pct_FPR", "TPR_at_5pct_FPR",
                                    "Precision", "Recall", "F1"]].agg(["mean", "std"])
        summ.to_csv(out_dir / "cv_summary_5fold_temporal.csv")
        print(summ.round(4).to_string())

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
