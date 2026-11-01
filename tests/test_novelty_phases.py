import numpy as np
import pandas as pd
from src.conformal import ConformalFraudPredictor
from src.pareto_fairness import compute_fairness_cost_pareto
from src.delayed_feedback import simulate_delayed_feedback, get_training_slice_under_delay

def test_conformal_predictor():
    y_cal = np.array([0] * 100 + [1] * 10)
    p_cal = np.concatenate([np.random.uniform(0.01, 0.2, 100), np.random.uniform(0.7, 0.99, 10)])
    cp = ConformalFraudPredictor(alpha=0.05)
    cp.calibrate(y_cal, p_cal)
    res = cp.predict_sets(np.array([0.05, 0.95, 0.45]))
    assert len(res) == 3
    assert 'triage_decision' in res.columns
    assert set(res['triage_decision']).issubset({'AUTO_APPROVE', 'HUMAN_REVIEW', 'HARD_DECLINE'})

def test_pareto_fairness():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.random.uniform(0.01, 0.2, 90), np.random.uniform(0.6, 0.99, 10)])
    slices = pd.Series(['Age1'] * 50 + ['Age2'] * 50)
    pareto_df = compute_fairness_cost_pareto(y, p, slices)
    assert not pareto_df.empty
    assert 'cost_per_app' in pareto_df.columns
    assert 'min_disparate_impact' in pareto_df.columns

def test_delayed_feedback():
    df = pd.DataFrame({'month': [0, 1, 2, 3], 'fraud_bool': [0, 1, 0, 1]})
    sim = simulate_delayed_feedback(df, delay_months=1)
    assert 'verified_month' in sim.columns
    train_at_2 = get_training_slice_under_delay(df, current_month=2, delay_months=1)
    assert len(train_at_2) == 2
