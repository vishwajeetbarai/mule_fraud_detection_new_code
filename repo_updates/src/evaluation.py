"""Metrics and plots for highly imbalanced fraud detection."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
)


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
# Plots
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
