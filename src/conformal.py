import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class ConformalFraudPredictor:
    """Conformal Prediction for Fraud Detection."""
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
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
                set_type.append('{Legitimate}')
                decision.append('AUTO_APPROVE')
            elif i1 and not i0:
                set_type.append('{Fraud}')
                decision.append('HARD_DECLINE')
            elif i0 and i1:
                set_type.append('{Legitimate, Fraud}')
                decision.append('HUMAN_REVIEW')
            else:
                set_type.append('{Empty}')
                decision.append('HUMAN_REVIEW')
        return pd.DataFrame({
            'probability': p,
            'prediction_set': set_type,
            'triage_decision': decision,
            'set_size': include_0.astype(int) + include_1.astype(int),
        })

    def evaluate_coverage(self, y_true: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
        y = np.asarray(y_true)
        df_sets = self.predict_sets(proba)
        covered = np.zeros(len(y), dtype=bool)
        for i, (yi, set_str) in enumerate(zip(y, df_sets['prediction_set'])):
            if yi == 0 and 'Legitimate' in set_str:
                covered[i] = True
            elif yi == 1 and 'Fraud' in set_str:
                covered[i] = True

        return {
            'empirical_coverage': float(np.mean(covered)),
            'avg_set_size': float(df_sets['set_size'].mean()),
            'auto_approve_rate': float((df_sets['triage_decision'] == 'AUTO_APPROVE').mean()),
            'human_review_rate': float((df_sets['triage_decision'] == 'HUMAN_REVIEW').mean()),
            'hard_decline_rate': float((df_sets['triage_decision'] == 'HARD_DECLINE').mean()),
        }
