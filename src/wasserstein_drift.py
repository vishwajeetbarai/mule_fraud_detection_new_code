"""Multi-Manifold Optimal Transport & Sliced Wasserstein Distance Drift Detector.

Replaces heuristic 1D univariate Kolmogorov-Smirnov tests with true multi-dimensional
manifold distribution shift detection via the Sliced Wasserstein Distance (SWD).

Mathematical Formulation:
SWD(P, Q) = integral_{theta in S^{d-1}} W_1(theta_# P, theta_# Q) d_theta
approximated by Monte Carlo integration over L uniform random projections theta_l:
SWD_L(P, Q) = (1/L) sum_{l=1}^L W_1(X P theta_l, Y Q theta_l)
"""
import numpy as np
import pandas as pd


class SlicedWassersteinDriftDetector:
    """Detects multi-manifold concept drift using Sliced Wasserstein Distance (SWD)."""

    def __init__(self, n_projections=64, threshold_factor=2.0, random_state=42):
        self.n_projections = int(n_projections)
        self.threshold_factor = float(threshold_factor)
        self.random_state = int(random_state)
        self.rng = np.random.RandomState(self.random_state)
        
        self.feature_names = None
        self.reference_mean_ = None
        self.reference_std_ = None
        self.reference_projections_ = None
        self.baseline_swd_mean_ = 0.0
        self.baseline_swd_std_ = 1.0
        self.drift_threshold_ = 0.0
        self.projections_ = None

    def fit(self, X_reference, n_bootstrap_subsamples=20):
        """Fit reference manifold and calculate empirical baseline variation threshold.
        
        Parameters:
        -----------
        X_reference : pd.DataFrame or np.ndarray
            Historical reference data (e.g., training or validation partition).
        n_bootstrap_subsamples : int
            Number of bootstrap slices used to estimate natural baseline variation.
        """
        if isinstance(X_reference, pd.DataFrame):
            self.feature_names = list(X_reference.columns)
            X = X_reference.values.astype(np.float32)
        else:
            X = np.asarray(X_reference, dtype=np.float32)

        # Standardize features
        self.reference_mean_ = np.nanmean(X, axis=0)
        self.reference_std_ = np.nanstd(X, axis=0)
        self.reference_std_[self.reference_std_ < 1e-6] = 1.0
        
        X_norm = (np.nan_to_num(X, nan=0.0) - self.reference_mean_) / self.reference_std_
        n_samples, d = X_norm.shape

        # Generate L random projection vectors on the unit sphere S^{d-1}
        raw_projections = self.rng.randn(self.n_projections, d).astype(np.float32)
        norms = np.linalg.norm(raw_projections, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        self.projections_ = raw_projections / norms  # (L, d)

        # Project reference distribution: (N, L)
        self.reference_projections_ = np.dot(X_norm, self.projections_.T)

        # Estimate baseline variation by comparing bootstrap subsamples against reference
        baseline_distances = []
        sub_size = min(1000, n_samples // 2)
        if sub_size > 10:
            for _ in range(n_bootstrap_subsamples):
                idx = self.rng.choice(n_samples, size=sub_size, replace=False)
                sub_norm = X_norm[idx]
                sub_proj = np.dot(sub_norm, self.projections_.T)
                swd = self._compute_swd_from_projections(self.reference_projections_, sub_proj)
                baseline_distances.append(swd)

            self.baseline_swd_mean_ = float(np.mean(baseline_distances))
            self.baseline_swd_std_ = float(np.std(baseline_distances)) if np.std(baseline_distances) > 0 else 0.05
        else:
            self.baseline_swd_mean_ = 0.05
            self.baseline_swd_std_ = 0.02

        self.drift_threshold_ = self.baseline_swd_mean_ + self.threshold_factor * self.baseline_swd_std_
        return self

    @staticmethod
    def _compute_swd_from_projections(proj_ref, proj_curr):
        """Compute average 1D Wasserstein distance across all 1D sorted projections."""
        # proj_ref: (N_ref, L), proj_curr: (N_curr, L)
        L = proj_ref.shape[1]
        n_ref = proj_ref.shape[0]
        n_curr = proj_curr.shape[0]
        
        # Sort along each projection dimension
        sorted_ref = np.sort(proj_ref, axis=0)
        sorted_curr = np.sort(proj_curr, axis=0)
        
        # Interpolate to common quantile grid for exact L1 distance comparison
        grid = np.linspace(0.0, 1.0, 100)
        q_ref = np.quantile(sorted_ref, grid, axis=0)  # (100, L)
        q_curr = np.quantile(sorted_curr, grid, axis=0) # (100, L)
        
        # Mean absolute difference across grid and across all L projections
        l1_distances = np.mean(np.abs(q_ref - q_curr), axis=0) # (L,)
        return float(np.mean(l1_distances))

    def compute_drift(self, X_current):
        """Compute Sliced Wasserstein Distance of current evaluation window against reference manifold.
        
        Returns:
        --------
        dict:
            - swd_distance: float
            - drift_detected: bool
            - drift_ratio: float (distance / threshold)
            - baseline_mean: float
            - threshold: float
        """
        if isinstance(X_current, pd.DataFrame):
            X = X_current.values.astype(np.float32)
        else:
            X = np.asarray(X_current, dtype=np.float32)

        X_norm = (np.nan_to_num(X, nan=0.0) - self.reference_mean_) / self.reference_std_
        curr_proj = np.dot(X_norm, self.projections_.T)

        swd = self._compute_swd_from_projections(self.reference_projections_, curr_proj)
        drift_detected = bool(swd > self.drift_threshold_)
        ratio = float(swd / max(self.drift_threshold_, 1e-6))

        return {
            "swd_distance": float(swd),
            "drift_detected": drift_detected,
            "drift_ratio": ratio,
            "baseline_mean": float(self.baseline_swd_mean_),
            "threshold": float(self.drift_threshold_),
        }

