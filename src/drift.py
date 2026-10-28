"""Concept Drift Detection Module for Bank Account Fraud Detection.

Implements lightweight Population Stability Index (PSI) and feature/score distribution
drift tracking comparing baseline reference distributions (e.g. training/validation months)
against incoming chronological test/validation months.

Research Novelty & Compliance:
- Standard Basel / credit-risk PSI binning (<0.10 stable, 0.10-0.25 moderate drift, >0.25 significant).
- Kolmogorov-Smirnov (KS) non-parametric 2-sample test for continuous distribution shift.
- Categorical distribution drift via Total Variation Distance (TVD) and Jensen-Shannon divergence.
- Leakage-free: frozen reference quantiles/frequencies learned strictly on reference periods.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

try:
    import config
except ImportError:
    from src import config


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
    n_bins : int, default=10
        Number of quantile-based bins to establish on the reference distribution.
    eps : float, default=1e-4
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
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([ref.min() - 1e-3, ref.max() + 1e-3])

    edges = bin_edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf

    # Empirical bin counts
    ref_counts, _ = np.histogram(ref, bins=edges)
    tar_counts, _ = np.histogram(tar, bins=edges)

    ref_pct = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
    tar_pct = (tar_counts + eps) / (len(tar) + eps * len(tar_counts))

    # PSI formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_vec = (tar_pct - ref_pct) * np.log(tar_pct / ref_pct)
    psi_val = float(np.sum(psi_vec))
    return max(0.0, psi_val), bin_edges


def compute_ks_test(
    reference: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test for continuous feature distribution shift.

    Returns
    -------
    stat : float
        KS test statistic (supremum distance between empirical CDFs).
    p_value : float
        Two-sided asymptotic p-value.
    """
    ref = np.asarray(reference, dtype=float)
    tar = np.asarray(target, dtype=float)
    ref = ref[np.isfinite(ref)]
    tar = tar[np.isfinite(tar)]
    if len(ref) < 2 or len(tar) < 2:
        return 0.0, 1.0
    res = stats.ks_2samp(ref, tar)
    return float(res.statistic), float(res.pvalue)


def compute_categorical_drift(
    reference: pd.Series,
    target: pd.Series,
    eps: float = 1e-5,
) -> Dict[str, float]:
    """Compute Total Variation Distance (TVD) and Jensen-Shannon distance for categorical slices."""
    all_cats = sorted(set(reference.dropna().unique()) | set(target.dropna().unique()))
    if not all_cats:
        return {"tvd": 0.0, "js_divergence": 0.0}

    ref_counts = reference.value_counts(normalize=True)
    tar_counts = target.value_counts(normalize=True)

    p = np.array([ref_counts.get(c, 0.0) for c in all_cats], dtype=float)
    q = np.array([tar_counts.get(c, 0.0) for c in all_cats], dtype=float)

    # Total Variation Distance: 0.5 * sum(|p_i - q_i|)
    tvd = 0.5 * np.sum(np.abs(p - q))

    # Jensen-Shannon Divergence
    p_smooth = (p + eps) / np.sum(p + eps)
    q_smooth = (q + eps) / np.sum(q + eps)
    m = 0.5 * (p_smooth + q_smooth)
    kl_pm = np.sum(p_smooth * np.log(p_smooth / m))
    kl_qm = np.sum(q_smooth * np.log(q_smooth / m))
    js_div = 0.5 * (kl_pm + kl_qm)

    return {
        "tvd": float(tvd),
        "js_divergence": float(js_div),
    }


class ConceptDriftTracker:
    """Production-grade feature and model-score concept drift monitor.

    Monitors incoming temporal application batches against frozen baseline reference statistics
    (e.g., training split or previous validation window).
    """

    def __init__(
        self,
        features_to_monitor: Optional[List[str]] = None,
        psi_threshold: float = 0.10,
        n_bins: int = 10,
        alpha_pvalue: float = 0.05,
    ):
        self.features_to_monitor = features_to_monitor or [
            "velocity_6h",
            "credit_risk_score",
            "proposed_credit_limit",
            "income",
            "name_email_similarity",
            "session_length_in_minutes",
        ]
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins
        self.alpha_pvalue = alpha_pvalue

        self.reference_bin_edges_: Dict[str, np.ndarray] = {}
        self.reference_stats_: Dict[str, Dict[str, float]] = {}
        self.reference_arrays_: Dict[str, np.ndarray] = {}
        self.is_fitted_ = False

    def fit(self, reference_df: pd.DataFrame, reference_scores: Optional[np.ndarray] = None) -> "ConceptDriftTracker":
        """Fit reference distribution baselines on training/reference window.

        Parameters
        ----------
        reference_df : pd.DataFrame
            Reference feature dataframe (e.g. train months 0-4).
        reference_scores : np.ndarray, optional
            Classifier predicted fraud probability scores on the reference dataset.
        """
        for feat in self.features_to_monitor:
            if feat in reference_df.columns:
                vals = reference_df[feat].dropna().values.astype(float)
                if len(vals) > 0:
                    _, edges = compute_psi(vals, vals, n_bins=self.n_bins)
                    self.reference_bin_edges_[feat] = edges
                    self.reference_stats_[feat] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "median": float(np.median(vals)),
                        "q25": float(np.percentile(vals, 25)),
                        "q75": float(np.percentile(vals, 75)),
                    }
                    self.reference_arrays_[feat] = vals

        if reference_scores is not None:
            scores = np.asarray(reference_scores, dtype=float)
            scores = scores[np.isfinite(scores)]
            if len(scores) > 0:
                _, edges = compute_psi(scores, scores, n_bins=self.n_bins)
                self.reference_bin_edges_["__model_score__"] = edges
                self.reference_stats_["__model_score__"] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "median": float(np.median(scores)),
                    "q25": float(np.percentile(scores, 25)),
                    "q75": float(np.percentile(scores, 75)),
                }
                self.reference_arrays_["__model_score__"] = scores

        self.is_fitted_ = True
        return self

    def evaluate_batch(
        self,
        target_df: pd.DataFrame,
        target_scores: Optional[np.ndarray] = None,
        batch_name: str = "target_batch",
    ) -> pd.DataFrame:
        """Evaluate drift metrics for a target incoming monthly or validation batch.

        Returns
        -------
        pd.DataFrame
            Summary table of PSI, KS statistic, KS p-value, drift status, and mean shifts.
        """
        if not self.is_fitted_:
            raise ValueError("ConceptDriftTracker must be fitted on reference data before evaluation.")

        records = []

        for feat in self.features_to_monitor:
            if feat in target_df.columns and feat in self.reference_bin_edges_:
                ref_vals = self.reference_arrays_[feat]
                tar_vals = target_df[feat].dropna().values.astype(float)
                edges = self.reference_bin_edges_[feat]

                psi, _ = compute_psi(ref_vals, tar_vals, bin_edges=edges)
                ks_stat, ks_p = compute_ks_test(ref_vals, tar_vals)

                tar_mean = float(np.mean(tar_vals)) if len(tar_vals) > 0 else 0.0
                ref_mean = self.reference_stats_[feat]["mean"]
                mean_delta = tar_mean - ref_mean

                # Drift categorization:
                # PSI < 0.10: Stable / No Drift
                # 0.10 <= PSI < 0.25: Moderate Drift
                # PSI >= 0.25: Significant Drift
                if psi >= 0.25:
                    status = "SIGNIFICANT_DRIFT"
                elif psi >= self.psi_threshold:
                    status = "MODERATE_DRIFT"
                else:
                    status = "STABLE"

                records.append({
                    "batch": batch_name,
                    "feature": feat,
                    "psi": round(psi, 4),
                    "ks_statistic": round(ks_stat, 4),
                    "ks_pvalue": ks_p,
                    "ks_significant": bool(ks_p < self.alpha_pvalue),
                    "ref_mean": round(ref_mean, 4),
                    "target_mean": round(tar_mean, 4),
                    "mean_shift": round(mean_delta, 4),
                    "drift_status": status,
                })

        if target_scores is not None and "__model_score__" in self.reference_bin_edges_:
            ref_scores = self.reference_arrays_["__model_score__"]
            tar_scores = np.asarray(target_scores, dtype=float)
            tar_scores = tar_scores[np.isfinite(tar_scores)]
            edges = self.reference_bin_edges_["__model_score__"]

            psi, _ = compute_psi(ref_scores, tar_scores, bin_edges=edges)
            ks_stat, ks_p = compute_ks_test(ref_scores, tar_scores)
            tar_mean = float(np.mean(tar_scores)) if len(tar_scores) > 0 else 0.0
            ref_mean = self.reference_stats_["__model_score__"]["mean"]

            status = "SIGNIFICANT_DRIFT" if psi >= 0.25 else ("MODERATE_DRIFT" if psi >= self.psi_threshold else "STABLE")

            records.append({
                "batch": batch_name,
                "feature": "__model_score__",
                "psi": round(psi, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": ks_p,
                "ks_significant": bool(ks_p < self.alpha_pvalue),
                "ref_mean": round(ref_mean, 4),
                "target_mean": round(tar_mean, 4),
                "mean_shift": round(tar_mean - ref_mean, 4),
                "drift_status": status,
            })

        return pd.DataFrame(records)

    def track_temporal_stream(
        self,
        df: pd.DataFrame,
        time_col: str = "month",
        model: Optional[Any] = None,
        feature_preprocessor: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Scan across multiple chronological periods (e.g. months 0 through 7) tracking PSI progression."""
        months = sorted(df[time_col].dropna().unique())
        all_results = []

        for m in months:
            m_df = df[df[time_col] == m]
            m_scores = None
            if model is not None and feature_preprocessor is not None:
                X_m = feature_preprocessor.transform(m_df)
                m_scores = model.predict_proba(X_m)[:, 1]

            m_report = self.evaluate_batch(
                target_df=m_df,
                target_scores=m_scores,
                batch_name=f"Month_{m}",
            )
            all_results.append(m_report)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()
