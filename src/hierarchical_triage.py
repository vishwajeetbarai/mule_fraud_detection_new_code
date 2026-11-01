"""Hierarchical Operational Triage Policy for Mule Account Fraud Detection (Phase 2).

Integrates:
1. Asymmetric Financial Loss Optimization (Validation-selected cost threshold)
2. Temporal Drift-Aware Threshold Adaptation (TDTA target-FPR calibration)
3. Split Conformal Prediction Uncertainty Sets ({0}, {1}, {0, 1})
4. Three-Tier Operational Decision Routing:
   - AUTO_APPROVE: Frictionless onboarding ($C_TN = $0).
   - AUTO_FLAG: Flagged for high-priority investigation / automated step-up friction ($C_TP = $10, $C_FP = $30).
   - HUMAN_REVIEW: Routed to compliance investigation queue ($C_investigation = $30).

Zero Test-Label Leakage:
All thresholds, non-conformity quantiles, and policy cutoffs are tuned strictly on Validation data (Month 5).
Evaluated out-of-sample on Test data (Months 6-7).
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score

import config
import evaluation as ev
from conformal import ConformalFraudPredictor
from adaptive_threshold import AdaptiveThresholdCalibrator


class HierarchicalTriagePolicy:
    """Three-tier operational routing policy combining cost thresholds, TDTA, and conformal uncertainty.
    
    Mathematical Specification:
    ---------------------------
    Let:
      - p_i in [0, 1] be the predicted probability of fraud for application i.
      - tau_cost in [0, 1] be the cost-optimal decision threshold selected on validation data.
      - tau_tdta in [0, 1] be the drift-calibrated decision threshold (target FPR <= 0.05 on calibration).
      - Gamma_alpha(x_i) in {{0}, {1}, {0, 1}} be the split conformal prediction set at nominal coverage 1 - alpha.
    
    Decision Rules:
    --------------
    Action A: AUTO_APPROVE (Action = 0)
      Assigned if:
        Gamma_alpha(x_i) == {0}  AND  p_i < min(tau_cost, tau_tdta)
      Justification: Both the model's cost/drift threshold and its conformal uncertainty set agree
                     that the instance is provably low-risk. Approved frictionlessly.
                     
    Action B: AUTO_FLAG (Action = 1)
      Assigned if:
        p_i >= max(tau_cost, tau_tdta)  OR  Gamma_alpha(x_i) == {1}
      Justification: Model score exceeds both the cost-minimizing threshold and the drift safeguard,
                     or conformal calibration confirms unambiguous fraud. Flagged for fraud containment.
                     
    Action C: HUMAN_REVIEW (Action = 2)
      Assigned if:
        Neither Auto-Approve nor Auto-Flag conditions are met.
        Explicitly covers:
        - Ambiguous conformal prediction sets Gamma_alpha(x_i) == {0, 1}.
        - Conflict zone: tau_min <= p_i < tau_max (e.g. flagged by cost optimizer but guarded by TDTA drift threshold).
      Justification: High epistemic or aleatoric uncertainty requires manual investigator oversight.
    """

    def __init__(
        self,
        tau_cost: float,
        tau_tdta: float,
        conformal_predictor: ConformalFraudPredictor,
        cost_fn: float = config.COST_FN,
        cost_fp: float = config.COST_FP,
        cost_tp: float = config.COST_TP,
        cost_investigation: float = 30.0,
        cost_tn: float = config.COST_TN,
    ):
        self.tau_cost = float(tau_cost)
        self.tau_tdta = float(tau_tdta)
        self.cpf = conformal_predictor
        self.cost_fn = float(cost_fn)
        self.cost_fp = float(cost_fp)
        self.cost_tp = float(cost_tp)
        self.cost_investigation = float(cost_investigation)
        self.cost_tn = float(cost_tn)

    def route(self, proba: np.ndarray) -> pd.DataFrame:
        """Route applications into AUTO_APPROVE, AUTO_FLAG, or HUMAN_REVIEW."""
        p = np.asarray(proba, dtype=float)
        n = len(p)

        conf_df = self.cpf.predict_sets(p)
        set_type = conf_df["prediction_set"].values  # e.g., '{Legitimate}', '{Fraud}', '{Legitimate, Fraud}'

        actions = np.empty(n, dtype=object)
        action_codes = np.empty(n, dtype=int)  # 0: Approve, 1: Flag, 2: Review

        tau_min = min(self.tau_cost, self.tau_tdta)
        tau_max = max(self.tau_cost, self.tau_tdta)

        for i in range(n):
            pi = p[i]
            s = set_type[i]

            # Rule B: AUTO_FLAG (High-confidence fraud)
            if (pi >= tau_max) or (s == "{Fraud}"):
                actions[i] = "AUTO_FLAG"
                action_codes[i] = 1
            # Rule A: AUTO_APPROVE (High-confidence legitimate)
            elif (s == "{Legitimate}") and (pi < tau_min):
                actions[i] = "AUTO_APPROVE"
                action_codes[i] = 0
            # Rule C: HUMAN_REVIEW (Ambiguity, conflict, or margin cases)
            else:
                actions[i] = "HUMAN_REVIEW"
                action_codes[i] = 2

        out_df = pd.DataFrame({
            "probability": p,
            "prediction_set": set_type,
            "action": actions,
            "action_code": action_codes,
        })
        return out_df

    def evaluate(self, y_true: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
        """Evaluate operational metrics, costs, and triage distribution under ground-truth labels."""
        y = np.asarray(y_true, dtype=int)
        p = np.asarray(proba, dtype=float)
        n = len(y)

        routed = self.route(p)
        acts = routed["action"].values

        # Action proportions
        n_approve = int(np.sum(acts == "AUTO_APPROVE"))
        n_flag = int(np.sum(acts == "AUTO_FLAG"))
        n_review = int(np.sum(acts == "HUMAN_REVIEW"))

        pct_approve = n_approve / n
        pct_flag = n_flag / n
        pct_review = n_review / n

        # Binary effective decisions for bank security:
        # AUTO_APPROVE is non-flagged (0); AUTO_FLAG and HUMAN_REVIEW are flagged for manual/step-up triage (1)
        flagged_total = (acts != "AUTO_APPROVE").astype(int)
        tn, fp, fn, tp = confusion_matrix(y, flagged_total, labels=[0, 1]).ravel()

        prec = float(precision_score(y, flagged_total, zero_division=0))
        rec = float(recall_score(y, flagged_total, zero_division=0))
        f1 = float(f1_score(y, flagged_total, zero_division=0))
        fpr = float(fp / max(fp + tn, 1))
        fnr = float(fn / max(fn + tp, 1))

        # Action-level breakdown: Frauds and Legitimates per Action
        y_fraud = (y == 1)
        y_legit = (y == 0)

        fraud_approved = int(np.sum(y_fraud & (acts == "AUTO_APPROVE")))  # True False Negatives (escaped)
        fraud_flagged = int(np.sum(y_fraud & (acts == "AUTO_FLAG")))      # Direct fraud catch
        fraud_reviewed = int(np.sum(y_fraud & (acts == "HUMAN_REVIEW")))  # Caught in review queue

        legit_approved = int(np.sum(y_legit & (acts == "AUTO_APPROVE")))  # Frictionless approval
        legit_flagged = int(np.sum(y_legit & (acts == "AUTO_FLAG")))      # False alarm auto-flag
        legit_reviewed = int(np.sum(y_legit & (acts == "HUMAN_REVIEW")))  # Investigator burden

        # Operational Financial Cost Calculation:
        # - Escaped fraud (fraud_approved): Incurs full FN cost
        # - Flagged fraud (fraud_flagged): Step-up verification cost
        # - Reviewed fraud (fraud_reviewed): Investigation cost + step-up cost
        # - Approved legit (legit_approved): Zero friction
        # - Flagged legit (legit_flagged): False positive friction cost
        # - Reviewed legit (legit_reviewed): Investigation cost
        cost_escaped_fraud = fraud_approved * self.cost_fn
        cost_flagged_fraud = fraud_flagged * self.cost_tp
        cost_reviewed_fraud = fraud_reviewed * (self.cost_investigation + self.cost_tp)

        cost_approved_legit = legit_approved * self.cost_tn
        cost_flagged_legit = legit_flagged * self.cost_fp
        cost_reviewed_legit = legit_reviewed * self.cost_investigation

        total_cost = (
            cost_escaped_fraud
            + cost_flagged_fraud
            + cost_reviewed_fraud
            + cost_approved_legit
            + cost_flagged_legit
            + cost_reviewed_legit
        )
        cost_per_app = float(total_cost / n)

        # Conformal coverage
        cov_res = self.cpf.evaluate_coverage(y, p)

        return {
            "n_total": n,
            "n_fraud": int(np.sum(y)),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "n_auto_approve": n_approve,
            "n_auto_flag": n_flag,
            "n_human_review": n_review,
            "pct_auto_approve": pct_approve,
            "pct_auto_flag": pct_flag,
            "pct_human_review": pct_review,
            "fraud_approved_escaped": fraud_approved,
            "fraud_flagged": fraud_flagged,
            "fraud_reviewed": fraud_reviewed,
            "legit_approved": legit_approved,
            "legit_flagged": legit_flagged,
            "legit_reviewed": legit_reviewed,
            "conformal_empirical_coverage": cov_res["empirical_coverage"],
            "operational_cost_per_app": cost_per_app,
            "tau_cost": self.tau_cost,
            "tau_tdta": self.tau_tdta,
        }

