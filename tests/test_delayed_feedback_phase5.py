"""Automated Unit Tests for Phase 5: Delayed-Label Feedback & Temporal Prequential Validation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import config
from features import FeaturePreprocessor
from delayed_feedback_validation import (
    build_application_timeline,
    run_delayed_feedback_evaluation,
)
from conformal import ConformalFraudPredictor


@pytest.fixture
def mock_dataset():
    """Generates synthetic dataset spanning months 5, 6, and 7."""
    np.random.seed(42)
    rows = []
    # Month 5 (Validation reference): 300 rows
    # Month 6 (Test period 1): 300 rows
    # Month 7 (Test period 2): 300 rows
    for m in [5, 6, 7]:
        n = 300
        df_m = pd.DataFrame({
            "fraud_bool": np.random.choice([0, 1], size=n, p=[0.97, 0.03]),
            "income": np.random.uniform(0.1, 0.9, size=n),
            "name_email_similarity": np.random.uniform(0.1, 0.9, size=n),
            "prev_address_months_count": np.random.choice([-1, 10, 24], size=n),
            "current_address_months_count": np.random.choice([12, 36, 120], size=n),
            "customer_age": np.random.choice([20, 30, 40, 50], size=n),
            "days_since_request": np.random.uniform(0.01, 2.0, size=n),
            "intended_balcon_amount": np.random.uniform(-5.0, 50.0, size=n),
            "payment_type": np.random.choice(["AA", "AB", "AC"], size=n),
            "zip_count_4w": np.random.randint(500, 3000, size=n),
            "velocity_6h": np.random.uniform(1000.0, 10000.0, size=n),
            "velocity_24h": np.random.uniform(2000.0, 8000.0, size=n),
            "velocity_4w": np.random.uniform(3000.0, 7000.0, size=n),
            "bank_branch_count_8w": np.random.randint(10, 500, size=n),
            "date_of_birth_distinct_emails_4w": np.random.randint(1, 20, size=n),
            "employment_status": np.random.choice(["CA", "CB", "CC"], size=n),
            "credit_risk_score": np.random.randint(50, 250, size=n),
            "email_is_free": np.random.choice([0, 1], size=n),
            "housing_status": np.random.choice(["BA", "BB", "BC"], size=n),
            "phone_home_valid": np.random.choice([0, 1], size=n),
            "phone_mobile_valid": np.random.choice([0, 1], size=n),
            "bank_months_count": np.random.choice([-1, 5, 12], size=n),
            "has_other_cards": np.random.choice([0, 1], size=n),
            "proposed_credit_limit": np.random.uniform(200.0, 1500.0, size=n),
            "foreign_request": np.random.choice([0, 1], size=n),
            "source": np.random.choice(["INTERNET", "TELEAPP"], size=n),
            "session_length_in_minutes": np.random.uniform(1.0, 30.0, size=n),
            "device_os": np.random.choice(["windows", "linux", "macintosh"], size=n),
            "keep_alive_session": np.random.choice([0, 1], size=n),
            "device_distinct_emails_8w": np.random.choice([1, 2], size=n),
            "device_fraud_count": np.zeros(n),
            "month": np.full(n, m),
        })
        rows.append(df_m)
    return pd.concat(rows, ignore_index=True)


class MockModel:
    def predict_proba(self, X):
        n = len(X)
        scores = np.clip(X["credit_limit_to_income"].values / 5000.0, 0.01, 0.99)
        return np.column_stack([1.0 - scores, scores])


def test_timeline_ordering(mock_dataset):
    """Verify that build_application_timeline assigns non-decreasing chronological arrival times."""
    df_timeline = build_application_timeline(mock_dataset)
    assert "t_apply" in df_timeline.columns
    # Within each month, timestamps must be strictly monotonic
    for m in [5, 6, 7]:
        m_times = df_timeline[df_timeline["month"] == m]["t_apply"].values
        assert (np.diff(m_times) >= 0).all()
        # Must fall within [m * 30, (m+1) * 30]
        assert (m_times >= m * 30.0).all() and (m_times <= (m + 1) * 30.0).all()


def test_delay_inaccessibility_constraint(mock_dataset):
    """Verify that labels are strictly inaccessible before t_apply + delay_days."""
    pre = FeaturePreprocessor().fit(mock_dataset)
    model = MockModel()

    cp = ConformalFraudPredictor(alpha=0.05)
    cp.calibrate(np.array([0, 0, 0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9]))

    # Test 30-day delay
    res_30d = run_delayed_feedback_evaluation(
        model=model,
        preprocessor=pre,
        df_raw=mock_dataset,
        initial_val_month=5,
        test_months=[6, 7],
        delay_days=30.0,
        policy_name="TDTA",
        conformal_predictor=cp,
    )

    avail_logs = res_30d["availability_logs"]
    assert len(avail_logs) == 1
    log_entry = avail_logs[0]

    # At decision point t = 210 (end of month 6, start of month 7):
    # With a 30-day delay, applications that arrived throughout month 6 (t in [180, 210])
    # have verification times t_verified in [210, 240].
    # Therefore, at t = 210, none of Month 6's applications are verified!
    assert log_entry["newly_verified_from_month_m"] == 0
    assert log_entry["unverified_pending_from_month_m"] == 300


def test_zero_delay_allows_immediate_month_boundary_verification(mock_dataset):
    """Verify that with 0-day delay, all month m applications are verified at month m+1 start."""
    pre = FeaturePreprocessor().fit(mock_dataset)
    model = MockModel()

    cp = ConformalFraudPredictor(alpha=0.05)
    cp.calibrate(np.array([0, 0, 0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9]))

    res_0d = run_delayed_feedback_evaluation(
        model=model,
        preprocessor=pre,
        df_raw=mock_dataset,
        initial_val_month=5,
        test_months=[6, 7],
        delay_days=0.0,
        policy_name="TDTA",
        conformal_predictor=cp,
    )

    log_entry = res_0d["availability_logs"][0]
    # Under zero delay, all 300 applications from Month 6 are verified by start of Month 7
    assert log_entry["newly_verified_from_month_m"] == 300
    assert log_entry["unverified_pending_from_month_m"] == 0


def test_threshold_updates_affect_future_only(mock_dataset):
    """Verify that threshold updates in month 7 do not change decisions in month 6."""
    pre = FeaturePreprocessor().fit(mock_dataset)
    model = MockModel()

    cp = ConformalFraudPredictor(alpha=0.05)
    cp.calibrate(np.array([0, 0, 0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9]))

    res = run_delayed_feedback_evaluation(
        model=model,
        preprocessor=pre,
        df_raw=mock_dataset,
        initial_val_month=5,
        test_months=[6, 7],
        delay_days=0.0,
        policy_name="TDTA",
        conformal_predictor=cp,
    )

    m_evals = res["month_evaluations"]
    assert len(m_evals) == 2
    # Month 6 decision was evaluated at Month 5 threshold
    thr_m6 = m_evals[0]["threshold"]
    assert thr_m6 == res["threshold_history"][0]["threshold"]


def test_action_partitions_are_mutually_exclusive(mock_dataset):
    """Verify that AUTO_APPROVE + HUMAN_REVIEW + AUTO_FLAG sum to 100%."""
    pre = FeaturePreprocessor().fit(mock_dataset)
    model = MockModel()

    cp = ConformalFraudPredictor(alpha=0.05)
    cp.calibrate(np.array([0, 0, 0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9]))

    res = run_delayed_feedback_evaluation(
        model=model,
        preprocessor=pre,
        df_raw=mock_dataset,
        initial_val_month=5,
        test_months=[6, 7],
        delay_days=14.0,
        policy_name="TDTA",
        conformal_predictor=cp,
    )

    summ = res["summary"]
    total_pct = summ["pct_auto_approve"] + summ["pct_human_review"] + summ["pct_auto_flag"]
    assert np.isclose(total_pct, 100.0, atol=1e-3)
    total_frauds = summ["frauds_auto_flagged"] + summ["frauds_referred_review"] + summ["frauds_erroneously_approved"]
    assert total_frauds == summ["total_frauds"]

