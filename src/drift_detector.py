"""Temporal Drift Detection Module for Bank Account Fraud Detection.

Implements rigorous, leak-free statistical drift detection between a reference distribution
(e.g., historical validation/training period) and subsequent chronological application batches.

Supported Metrics:
- Population Stability Index (PSI) with robust quantile binning and zero-frequency smoothing.
- Kolmogorov-Smirnov (KS) two-sample test with Bonferroni multiple testing correction.
- Categorical distribution distance (Jensen-Shannon divergence / Total Variation).
- Classifier model-score distribution drift.

Zero-Leakage Guarantees:
- Uses ONLY features available at application scoring time.
- Fraud target (fraud_bool) and future labels are STRICTLY EXCLUDED.
- Reference bins/quantiles are learned strictly on the reference period, frozen, and applied
  to the evaluation batch.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

import config


def compute_psi(
    reference: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
    bin_edges: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Calculate Population Stability Index (PSI) between reference and target distributions.
    
    Parameters
    ----------
    reference : np.ndarray
        Array of baseline/reference continuous values.
    target : np.ndarray
        Array of monitoring/target continuous values.
    n_bins : int
        Number of quantile-based bins to establish on reference distribution.
    eps : float
        Laplace / additive smoothing epsilon to avoid division by zero or log(0).
    bin_edges : np.ndarray, optional
        Pre-computed bin edges (if previously frozen on reference).
        
    Returns
    -------
    psi_value : float
        Summed PSI across all bins.
    bin_edges : np.ndarray
        Bin edges utilized.
    """
    ref = np.asarray(reference, dtype=float)
    tar = np.asarray(target, dtype=float)

    # Filter out NaNs/Infs
    ref = ref[np.isfinite(ref)]
    tar = tar[np.isfinite(tar)]

    if len(ref) < 2 or len(tar) < 2:
        return 0.0, np.array([])

    if bin_edges is None:
        # Quantile binning on reference
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(ref, quantiles)
        # Ensure distinct strictly monotonic edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            # Degenerate reference (e.g. all constant values)
            bin_edges = np.array([ref.min() - 1e-3, ref.max() + 1e-3])

    # Extend outermost bounds to prevent out-of-range clipping
    edges = bin_edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf

    # Compute empirical frequencies
    ref_counts, _ = np.histogram(ref, bins=edges)
    tar_counts, _ = np.histogram(tar, bins=edges)

    ref_pct = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
    tar_pct = (tar_counts + eps) / (len(tar) + eps * len(tar_counts))

    # PSI formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    # where Actual = target, Expected = reference
    psi_vec = (tar_pct - ref_pct) * np.log(tar_pct / ref_pct)
    psi_val = float(np.sum(psi_vec))
    return max(0.0, psi_val), bin_edges


def compute_ks_test(
    reference: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test for continuous distributions."""
    ref = np.asarray(reference, dtype=float)
    tar = np.asarray(target, dtype=float)
    ref = ref[np.isfinite(ref)]
    tar = tar[np.isfinite(tar)]

    if len(ref) < 2 or len(tar) < 2:
        return 0.0, 1.0

    stat, pval = stats.ks_2samp(ref, tar)
    return float(stat), float(pval)


def compute_categorical_tv_distance(
    reference: pd.Series,
    target: pd.Series,
) -> float:
    """Total Variation (TV) distance between two categorical distributions.
    
    TV = 0.5 * sum(|P(x) - Q(x)|) in [0, 1].
    """
    ref_counts = reference.value_counts(normalize=True)
    tar_counts = target.value_counts(normalize=True)

    all_cats = set(ref_counts.index) | set(tar_counts.index)
    ref_probs = np.array([ref_counts.get(c, 0.0) for c in all_cats])
    tar_probs = np.array([tar_counts.get(c, 0.0) for c in all_cats])

    tv_dist = 0.5 * np.sum(np.abs(tar_probs - ref_probs))
    return float(tv_dist)


class TemporalDriftDetector:
    """Configurable temporal drift detector comparing reference period to incoming batches.
    
    Parameters
    ----------
    features : List[str]
        List of continuous and categorical features to monitor.
    method : str
        Drift detection method: 'psi', 'ks', or 'combined'.
    psi_threshold : float
        PSI alert threshold (default 0.10).
    alpha : float
        Base significance level for hypothesis tests before multiple testing correction.
    min_alert_features : int
        Minimum number of feature alerts required to flag overall drift.
    monitor_score : bool
        Whether to also compute and alert on model probability score drift.
    score_psi_threshold : float
        PSI alert threshold for model score distribution.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        method: str = config.DRIFT_METHOD,
        psi_threshold: float = config.DRIFT_PSI_THRESHOLD,
        alpha: float = config.DRIFT_MAX_PVALUE_ALPHA,
        min_alert_features: int = config.DRIFT_MIN_ALERT_FEATURES,
        monitor_score: bool = config.DRIFT_SCORE_MONITORING,
        score_psi_threshold: float = config.DRIFT_SCORE_PSI_THRESHOLD,
    ):
        self.features = features or list(config.DRIFT_MONITORING_FEATURES)
        self.method = method
        self.psi_threshold = psi_threshold
        self.alpha = alpha
        self.min_alert_features = min_alert_features
        self.monitor_score = monitor_score
        self.score_psi_threshold = score_psi_threshold

        self.reference_data_: Optional[pd.DataFrame] = None
        self.reference_scores_: Optional[np.ndarray] = None
        self.frozen_bin_edges_: Dict[str, np.ndarray] = {}
        self.frozen_score_bins_: Optional[np.ndarray] = None

    def fit_reference(
        self,
        df_ref: pd.DataFrame,
        scores_ref: Optional[np.ndarray] = None,
    ) -> "TemporalDriftDetector":
        """Store and pre-bin the reference dataset.
        
        STRICT REQUIREMENT: df_ref must not include fraud labels or test data.
        """
        # Exclude target and temporal index explicitly
        clean_cols = [c for c in self.features if c in df_ref.columns and c not in (config.TARGET, config.TIME_COL)]
        self.features = clean_cols
        self.reference_data_ = df_ref[self.features].copy()

        # Learn and freeze quantile bins for continuous features
        self.frozen_bin_edges_ = {}
        for col in self.features:
            if pd.api.types.is_numeric_dtype(self.reference_data_[col]):
                vals = self.reference_data_[col].dropna().values
                _, edges = compute_psi(vals, vals, n_bins=10)
                self.frozen_bin_edges_[col] = edges

        # Pre-bin reference scores if provided
        if scores_ref is not None and self.monitor_score:
            self.reference_scores_ = np.asarray(scores_ref, dtype=float)
            _, s_edges = compute_psi(self.reference_scores_, self.reference_scores_, n_bins=10)
            self.frozen_score_bins_ = s_edges
        else:
            self.reference_scores_ = None
            self.frozen_score_bins_ = None

        return self

    def detect(
        self,
        df_eval: pd.DataFrame,
        scores_eval: Optional[np.ndarray] = None,
        period_id: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an incoming chronological batch against the reference.
        
        Returns
        -------
        report : Dict[str, Any]
            Detailed report including drift statistics per feature, score drift,
            alert count, and final boolean drift decision.
        """
        if self.reference_data_ is None:
            raise ValueError("Reference data not fitted. Call fit_reference first.")

        report = {
            "period_id": period_id,
            "n_reference": len(self.reference_data_),
            "n_eval": len(df_eval),
            "method": self.method,
            "psi_threshold": self.psi_threshold,
            "feature_reports": {},
            "alert_features": [],
            "score_drift": None,
            "drift_detected": False,
            "trigger_reason": "",
        }

        # Bonferroni corrected significance level across monitored features
        n_feats = max(1, len(self.features))
        bonferroni_alpha = self.alpha / n_feats

        for col in self.features:
            if col not in df_eval.columns:
                continue

            ref_series = self.reference_data_[col].dropna()
            eval_series = df_eval[col].dropna()

            if pd.api.types.is_numeric_dtype(ref_series):
                edges = self.frozen_bin_edges_.get(col, None)
                psi_val, _ = compute_psi(ref_series.values, eval_series.values, bin_edges=edges)
                ks_stat, ks_pval = compute_ks_test(ref_series.values, eval_series.values)

                feat_alert = False
                if self.method == "psi":
                    feat_alert = (psi_val >= self.psi_threshold)
                elif self.method == "ks":
                    feat_alert = (ks_pval < bonferroni_alpha)
                elif self.method == "combined":
                    feat_alert = (psi_val >= self.psi_threshold) or (ks_pval < bonferroni_alpha)

                report["feature_reports"][col] = {
                    "type": "numeric",
                    "psi": float(psi_val),
                    "ks_stat": float(ks_stat),
                    "ks_pvalue": float(ks_pval),
                    "alert": bool(feat_alert),
                }

                if feat_alert:
                    report["alert_features"].append(col)
            else:
                # Categorical TV distance
                tv_dist = compute_categorical_tv_distance(ref_series, eval_series)
                cat_alert = (tv_dist >= 0.10)
                report["feature_reports"][col] = {
                    "type": "categorical",
                    "tv_distance": float(tv_dist),
                    "alert": bool(cat_alert),
                }
                if cat_alert:
                    report["alert_features"].append(col)

        # Check model prediction score drift if provided
        score_alert = False
        if (
            self.monitor_score
            and self.reference_scores_ is not None
            and scores_eval is not None
        ):
            s_psi, _ = compute_psi(
                self.reference_scores_,
                np.asarray(scores_eval),
                bin_edges=self.frozen_score_bins_,
            )
            s_ks, s_pval = compute_ks_test(
                self.reference_scores_,
                np.asarray(scores_eval),
            )
            score_alert = (s_psi >= self.score_psi_threshold)
            report["score_drift"] = {
                "score_psi": float(s_psi),
                "score_ks_stat": float(s_ks),
                "score_ks_pvalue": float(s_pval),
                "score_psi_threshold": self.score_psi_threshold,
                "score_alert": bool(score_alert),
            }

        # Final trigger decision rule
        feat_drift_trigger = len(report["alert_features"]) >= self.min_alert_features
        drift_detected = feat_drift_trigger or score_alert

        reasons = []
        if feat_drift_trigger:
            reasons.append(f"Features drifted: {report['alert_features']}")
        if score_alert:
            reasons.append(f"Model score distribution drifted (PSI={report['score_drift']['score_psi']:.4f})")

        report["drift_detected"] = bool(drift_detected)
        report["trigger_reason"] = "; ".join(reasons) if drift_detected else "Distributions stable"

        return report
