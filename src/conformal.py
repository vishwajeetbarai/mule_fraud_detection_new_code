"""Split and Mondrian (Group-Conditional) Conformal Prediction for Fraud Detection.

Provides finite-sample coverage guarantees P(Y in Gamma_alpha) >= 1 - alpha:
1. Standard Split Conformal Prediction (marginal coverage across entire population).
2. Mondrian Conformal Prediction (group-conditional coverage across demographic cohorts,
   e.g., customer age categories), eliminating coverage disparities under non-uniform shift.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


class ConformalFraudPredictor:
    """Standard Marginal Conformal Prediction for Fraud Detection."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.q_neg_ = None
        self.q_pos_ = None

    def calibrate(self, y_cal: np.ndarray, proba_cal: np.ndarray):
        y = np.asarray(y_cal)
        p = np.asarray(proba_cal)
        s_0 = p[y == 0]
        n_0 = len(s_0)
        q_level_0 = np.ceil((n_0 + 1) * (1.0 - self.alpha)) / n_0
        self.q_neg_ = float(np.quantile(s_0, min(1.0, q_level_0)))
        s_1 = 1.0 - p[y == 1]
        n_1 = len(s_1)
        if n_1 > 0:
            q_level_1 = np.ceil((n_1 + 1) * (1.0 - self.alpha)) / n_1
            self.q_pos_ = float(np.quantile(s_1, min(1.0, q_level_1)))
        else:
            self.q_pos_ = 0.95
        return self

    def predict_sets(self, proba: np.ndarray) -> pd.DataFrame:
        p = np.asarray(proba)
        include_0 = p <= self.q_neg_
        include_1 = (1.0 - p) <= self.q_pos_
        set_type = []
        decision = []
        for i0, i1 in zip(include_0, include_1):
            if i0 and not i1:
                set_type.append("{Legitimate}")
                decision.append("AUTO_APPROVE")
            elif i1 and not i0:
                set_type.append("{Fraud}")
                decision.append("HARD_DECLINE")
            elif i0 and i1:
                set_type.append("{Legitimate, Fraud}")
                decision.append("HUMAN_REVIEW")
            else:
                set_type.append("{Empty}")
                decision.append("HUMAN_REVIEW")
        return pd.DataFrame({
            "probability": p,
            "prediction_set": set_type,
            "triage_decision": decision,
            "set_size": include_0.astype(int) + include_1.astype(int),
        })

    def evaluate_coverage(self, y_true: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
        y = np.asarray(y_true)
        df_sets = self.predict_sets(proba)
        covered = np.zeros(len(y), dtype=bool)
        for i, (yi, set_str) in enumerate(zip(y, df_sets["prediction_set"])):
            if yi == 0 and "Legitimate" in set_str:
                covered[i] = True
            elif yi == 1 and "Fraud" in set_str:
                covered[i] = True

        return {
            "empirical_coverage": float(np.mean(covered)),
            "avg_set_size": float(df_sets["set_size"].mean()),
            "auto_approve_rate": float((df_sets["triage_decision"] == "AUTO_APPROVE").mean()),
            "human_review_rate": float((df_sets["triage_decision"] == "HUMAN_REVIEW").mean()),
            "hard_decline_rate": float((df_sets["triage_decision"] == "HARD_DECLINE").mean()),
        }


class MondrianConformalPredictor:
    """Mondrian (Group-Conditional) Conformal Predictor.
    
    Guarantees valid coverage P(Y in Gamma_alpha | G = g) >= 1 - alpha for each
    individual demographic group or category g independently.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        # Mapping: group_name -> {"q_neg": float, "q_pos": float}
        self.group_quantiles_ = {}
        # Global fallback if an unseen group appears
        self.global_fallback_ = ConformalFraudPredictor(alpha=self.alpha)

    def calibrate(self, y_cal: np.ndarray, proba_cal: np.ndarray, groups_cal: np.ndarray):
        """Calibrate group-specific non-conformity quantiles on validation data."""
        y = np.asarray(y_cal)
        p = np.asarray(proba_cal)
        groups = np.asarray(groups_cal)

        # Calibrate global fallback
        self.global_fallback_.calibrate(y, p)

        unique_groups = pd.Series(groups).dropna().unique()
        for g in unique_groups:
            mask = (groups == g)
            y_g = y[mask]
            p_g = p[mask]

            s_0 = p_g[y_g == 0]
            n_0 = len(s_0)
            if n_0 > 0:
                q_level_0 = np.ceil((n_0 + 1) * (1.0 - self.alpha)) / n_0
                q_neg = float(np.quantile(s_0, min(1.0, q_level_0)))
            else:
                q_neg = self.global_fallback_.q_neg_

            s_1 = 1.0 - p_g[y_g == 1]
            n_1 = len(s_1)
            if n_1 > 0:
                q_level_1 = np.ceil((n_1 + 1) * (1.0 - self.alpha)) / n_1
                q_pos = float(np.quantile(s_1, min(1.0, q_level_1)))
            else:
                q_pos = self.global_fallback_.q_pos_

            self.group_quantiles_[g] = {"q_neg": q_neg, "q_pos": q_pos}

        return self

    def predict_sets(self, proba: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
        """Predict group-conditional conformal prediction sets."""
        p = np.asarray(proba)
        g_arr = np.asarray(groups)
        n = len(p)

        include_0 = np.zeros(n, dtype=bool)
        include_1 = np.zeros(n, dtype=bool)

        for i in range(n):
            g = g_arr[i]
            q_dict = self.group_quantiles_.get(g, None)
            if q_dict is None:
                q_neg = self.global_fallback_.q_neg_
                q_pos = self.global_fallback_.q_pos_
            else:
                q_neg = q_dict["q_neg"]
                q_pos = q_dict["q_pos"]

            include_0[i] = (p[i] <= q_neg)
            include_1[i] = ((1.0 - p[i]) <= q_pos)

        set_type = []
        decision = []
        for i0, i1 in zip(include_0, include_1):
            if i0 and not i1:
                set_type.append("{Legitimate}")
                decision.append("AUTO_APPROVE")
            elif i1 and not i0:
                set_type.append("{Fraud}")
                decision.append("HARD_DECLINE")
            elif i0 and i1:
                set_type.append("{Legitimate, Fraud}")
                decision.append("HUMAN_REVIEW")
            else:
                set_type.append("{Empty}")
                decision.append("HUMAN_REVIEW")

        return pd.DataFrame({
            "probability": p,
            "group": g_arr,
            "prediction_set": set_type,
            "triage_decision": decision,
            "set_size": include_0.astype(int) + include_1.astype(int),
        })

    def evaluate_group_coverage(self, y_true: np.ndarray, proba: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
        """Evaluate empirical coverage and set size partitioned by each demographic group."""
        y = np.asarray(y_true)
        df_sets = self.predict_sets(proba, groups)
        df_sets["y_true"] = y

        df_sets["covered"] = False
        df_sets.loc[(df_sets["y_true"] == 0) & (df_sets["prediction_set"].str.contains("Legitimate")), "covered"] = True
        df_sets.loc[(df_sets["y_true"] == 1) & (df_sets["prediction_set"].str.contains("Fraud")), "covered"] = True

        records = []
        for g, sub in df_sets.groupby("group"):
            records.append({
                "group": g,
                "n_samples": len(sub),
                "fraud_count": int(sub["y_true"].sum()),
                "empirical_coverage": float(sub["covered"].mean()),
                "avg_set_size": float(sub["set_size"].mean()),
                "auto_approve_rate": float((sub["triage_decision"] == "AUTO_APPROVE").mean()),
                "human_review_rate": float((sub["triage_decision"] == "HUMAN_REVIEW").mean()),
                "hard_decline_rate": float((sub["triage_decision"] == "HARD_DECLINE").mean()),
            })

        return pd.DataFrame(records)
