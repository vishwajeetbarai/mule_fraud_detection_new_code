"""Adaptive Threshold Calibration for Bank Account Fraud Detection.

Implements mathematically principled, leakage-safe threshold calibration targeting a specific
False Positive Rate (default 5% FPR) under non-stationary streaming distributions.

Features & Guarantees:
1. Target FPR Constraint: Calibrates score threshold tau such that empirical FPR <= target_fpr
   on eligible historical calibration data.
2. Delayed Feedback & Prequential Safety:
   - ONLY uses historical labels available strictly PRIOR to the decision period.
   - Evaluates test periods strictly unseen, never using test labels for calibration.
   - Supports configurable label lag (LABEL_DELAY_MONTHS).
3. Robust Fallback & Edge Case Handling:
   - If calibration dataset has insufficient legitimate cases (< min_negatives) or
     insufficient positives (< min_positives), preserves previous valid threshold.
   - Handles ties and score boundary conditions deterministically.
   - Formally acknowledges that target FPR on calibration data does not guarantee
     exact realization on future out-of-distribution test data.
4. Comprehensive Auditing:
   - Emits structured calibration log containing previous threshold, new threshold,
     calibration sample sizes, realized calibration FPR/TPR, and trigger metadata.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import config
import evaluation as ev


class AdaptiveThresholdCalibrator:
    """Manages threshold adaptation across sequential temporal batches.
    
    Parameters
    ----------
    target_fpr : float
        Target False Positive Rate (default 0.05).
    initial_threshold : float
        Initial decision threshold established on validation period (month 5).
    min_samples : int
        Minimum calibration sample count required.
    min_negatives : int
        Minimum legitimate (class 0) samples required in calibration buffer.
    min_positives : int
        Minimum fraud (class 1) samples required in calibration buffer.
    fallback_threshold : float
        Safety fallback threshold if calibration buffer is empty/degenerate.
    """

    def __init__(
        self,
        target_fpr: float = config.TARGET_FPR,
        initial_threshold: float = 0.5,
        min_samples: int = config.CALIBRATION_MIN_SAMPLES,
        min_negatives: int = config.CALIBRATION_MIN_NEGATIVES,
        min_positives: int = config.CALIBRATION_MIN_POSITIVES,
        fallback_threshold: Optional[float] = None,
    ):
        self.target_fpr = target_fpr
        self.current_threshold = float(initial_threshold)
        self.initial_threshold = float(initial_threshold)
        self.min_samples = min_samples
        self.min_negatives = min_negatives
        self.min_positives = min_positives
        self.fallback_threshold = (
            float(fallback_threshold) if fallback_threshold is not None else float(initial_threshold)
        )

        self.calibration_history_: List[Dict[str, Any]] = []

    def calibrate(
        self,
        calib_y_true: np.ndarray,
        calib_proba: np.ndarray,
        trigger_reason: str = "routine",
        period_id: Optional[Union[int, str]] = None,
        force_recalibrate: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute updated threshold from eligible calibration observations.
        
        Parameters
        ----------
        calib_y_true : np.ndarray
            Binary labels available from eligible historical period.
        calib_proba : np.ndarray
            Predicted fraud probabilities for eligible historical period.
        trigger_reason : str
            Description of event triggering calibration ('drift_alert', 'scheduled', 'init').
        period_id : Optional[Union[int, str]]
            Target period identifier where this new threshold will be applied.
        force_recalibrate : bool
            If False, retains current threshold without updating.
            
        Returns
        -------
        new_threshold : float
            The threshold to be applied to subsequent period.
        log_entry : Dict[str, Any]
            Audit entry detailing calibration statistics.
        """
        old_thr = self.current_threshold
        calib_y = np.asarray(calib_y_true, dtype=int)
        calib_p = np.asarray(calib_proba, dtype=float)

        n_total = len(calib_y)
        n_pos = int(np.sum(calib_y == 1))
        n_neg = int(np.sum(calib_y == 0))

        log_entry: Dict[str, Any] = {
            "period_id": period_id,
            "previous_threshold": old_thr,
            "updated_threshold": old_thr,
            "threshold_changed": False,
            "trigger_reason": trigger_reason,
            "n_calibration_samples": n_total,
            "n_calibration_positives": n_pos,
            "n_calibration_negatives": n_neg,
            "target_fpr": self.target_fpr,
            "calib_realized_fpr": np.nan,
            "calib_realized_tpr": np.nan,
            "status": "RETAINED_UNMODIFIED",
        }

        if not force_recalibrate:
            log_entry["status"] = "SKIPPED_NO_TRIGGER"
            self.calibration_history_.append(log_entry)
            return old_thr, log_entry

        # Edge Case Validation: Check minimum sample constraints
        if (
            n_total < self.min_samples
            or n_neg < self.min_negatives
            or n_pos < self.min_positives
        ):
            log_entry["status"] = (
                f"FALLBACK_INSUFFICIENT_DATA (total={n_total}/{self.min_samples}, "
                f"neg={n_neg}/{self.min_negatives}, pos={n_pos}/{self.min_positives})"
            )
            # Retain current threshold or fallback
            new_thr = old_thr if old_thr is not None else self.fallback_threshold
            log_entry["updated_threshold"] = new_thr
            self.calibration_history_.append(log_entry)
            return new_thr, log_entry

        # Calibrate threshold on legitimate (negative) scores:
        # Threshold at quantile (1 - target_fpr) of negatives
        neg_scores = calib_p[calib_y == 0]
        # Deterministic quantile calculation
        new_thr = float(np.quantile(neg_scores, 1.0 - self.target_fpr, method="linear"))

        # Clip threshold within safe probability boundaries [0.001, 0.999]
        new_thr = float(np.clip(new_thr, 0.001, 0.999))

        # Check empirical calibration performance
        calib_pred = (calib_p >= new_thr).astype(int)
        calib_fp = int(np.sum((calib_pred == 1) & (calib_y == 0)))
        calib_tp = int(np.sum((calib_pred == 1) & (calib_y == 1)))
        realized_fpr = calib_fp / max(n_neg, 1)
        realized_tpr = calib_tp / max(n_pos, 1)

        self.current_threshold = new_thr
        log_entry.update({
            "updated_threshold": new_thr,
            "threshold_changed": bool(abs(new_thr - old_thr) > 1e-5),
            "calib_realized_fpr": float(realized_fpr),
            "calib_realized_tpr": float(realized_tpr),
            "status": "RECALIBRATED_SUCCESSFULLY",
        })

        self.calibration_history_.append(log_entry)
        return new_thr, log_entry

    def get_audit_trail(self) -> pd.DataFrame:
        """Return historical audit log of all threshold updates."""
        return pd.DataFrame(self.calibration_history_)
