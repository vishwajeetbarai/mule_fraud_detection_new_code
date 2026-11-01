"""Unit tests for Phase 4: Realistic Adversarial Robustness Validation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import config
from features import FeaturePreprocessor
from realistic_adversarial import (
    DomainValidator,
    AdversarialPerturbationGenerator,
    evaluate_adversarial_threat_scenario,
)
from hierarchical_triage import HierarchicalTriagePolicy


@pytest.fixture
def sample_raw_data():
    """Generates a small realistic raw batch representing BaseDataset."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "fraud_bool": np.random.choice([0, 1], size=n, p=[0.98, 0.02]),
        "income": np.random.uniform(0.1, 0.9, size=n),
        "name_email_similarity": np.random.uniform(0.1, 0.9, size=n),
        "prev_address_months_count": np.random.choice([-1, 10, 24, 48], size=n),
        "current_address_months_count": np.random.choice([12, 36, 120], size=n),
        "customer_age": np.random.choice([20, 30, 40, 50, 60], size=n),
        "days_since_request": np.random.uniform(0.01, 5.0, size=n),
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
        "bank_months_count": np.random.choice([-1, 5, 12, 24], size=n),
        "has_other_cards": np.random.choice([0, 1], size=n),
        "proposed_credit_limit": np.random.uniform(200.0, 1500.0, size=n),
        "foreign_request": np.random.choice([0, 1], size=n),
        "source": np.random.choice(["INTERNET", "TELEAPP"], size=n),
        "session_length_in_minutes": np.random.uniform(1.0, 30.0, size=n),
        "device_os": np.random.choice(["windows", "linux", "macintosh"], size=n),
        "keep_alive_session": np.random.choice([0, 1], size=n),
        "device_distinct_emails_8w": np.random.choice([1, 2], size=n),
        "device_fraud_count": np.zeros(n),
        "month": np.random.choice([6, 7], size=n),
    })
    return df


def test_domain_validator_detects_illegal_records(sample_raw_data):
    """Test that DomainValidator rejects out-of-bound or tampered records."""
    df_clean = sample_raw_data.copy()
    df_bad = df_clean.copy()

    # Tamper with immutable column
    df_bad.loc[0, "credit_risk_score"] = 999
    # Violate bounded probability
    df_bad.loc[1, "name_email_similarity"] = 1.5
    # Violate categorical vocabulary
    df_bad.loc[2, "device_os"] = "android_jailbreak"
    # Violate non-negativity
    df_bad.loc[3, "proposed_credit_limit"] = -100.0

    valid_df, n_rejected, reasons = DomainValidator.filter_valid_records(df_bad, df_clean)
    assert n_rejected >= 4
    assert len(valid_df) == len(df_clean) - n_rejected
    assert any("tampered immutable column" in r for r in reasons)
    assert any("out of [0.0, 1.0]" in r for r in reasons)
    assert any("invalid categorical" in r for r in reasons)
    assert any("negative proposed_credit_limit" in r for r in reasons)


def test_adversarial_generators_respect_constraints(sample_raw_data):
    """Test that all threat generators produce physically and domain-valid records."""
    gen = AdversarialPerturbationGenerator(seed=42)

    s1 = gen.perturb_application_camouflage(sample_raw_data)
    valid_s1, rej_s1, _ = DomainValidator.filter_valid_records(s1, sample_raw_data)
    assert rej_s1 == 0
    assert (s1["name_email_similarity"] >= 0.0).all() and (s1["name_email_similarity"] <= 1.0).all()
    assert (s1["email_is_free"] == 0).all()
    assert (s1["proposed_credit_limit"] >= 200.0).all()

    s2 = gen.perturb_velocity_throttling(sample_raw_data)
    valid_s2, rej_s2, _ = DomainValidator.filter_valid_records(s2, sample_raw_data)
    assert rej_s2 == 0
    assert (s2["velocity_6h"] >= 0.0).all()

    s3 = gen.perturb_device_cloaking(sample_raw_data)
    valid_s3, rej_s3, _ = DomainValidator.filter_valid_records(s3, sample_raw_data)
    assert rej_s3 == 0
    assert (s3["device_distinct_emails_8w"] <= 1).all()
    assert (s3["device_os"] == "windows").all()

    s4 = gen.perturb_missingness_exploitation(sample_raw_data)
    valid_s4, rej_s4, _ = DomainValidator.filter_valid_records(s4, sample_raw_data)
    assert rej_s4 == 0
    assert (s4["prev_address_months_count"] == -1).all()
    assert (s4["phone_home_valid"] == 0).all()

    s5 = gen.perturb_multi_vector_syndicate(sample_raw_data)
    valid_s5, rej_s5, _ = DomainValidator.filter_valid_records(s5, sample_raw_data)
    assert rej_s5 == 0


def test_derived_features_recalculated_on_perturbed_raw(sample_raw_data):
    """Ensure that modifying raw features properly propagates to derived engineered features."""
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(sample_raw_data)

    X_clean = preprocessor.transform(sample_raw_data)

    gen = AdversarialPerturbationGenerator(seed=42)
    s2 = gen.perturb_velocity_throttling(sample_raw_data, throttle_ratio=0.50)
    X_perturbed = preprocessor.transform(s2)

    # Derived velocity features must change
    assert not np.allclose(X_clean["velocity_surge_intensity"], X_perturbed["velocity_surge_intensity"])
    assert not np.allclose(X_clean["velocity_burst_acceleration"], X_perturbed["velocity_burst_acceleration"])
    assert not np.allclose(X_clean["velocity_spike_index"], X_perturbed["velocity_spike_index"])


def test_original_dataset_immutability(sample_raw_data):
    """Verify that generating perturbations does not mutate the source dataframe in-place."""
    original_copy = sample_raw_data.copy(deep=True)
    gen = AdversarialPerturbationGenerator(seed=42)

    _ = gen.perturb_multi_vector_syndicate(sample_raw_data)

    pd.testing.assert_frame_equal(sample_raw_data, original_copy)


def test_end_to_end_scenario_evaluation(sample_raw_data):
    """Test end-to-end evaluation with a mock predictor and triage policy."""
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(sample_raw_data)

    class DummyModel:
        def predict_proba(self, X):
            n = len(X)
            # deterministic dummy proba correlated with credit_limit_to_income
            scores = np.clip(X["credit_limit_to_income"].values / 5000.0, 0.01, 0.99)
            return np.column_stack([1 - scores, scores])

    model = DummyModel()
    from conformal import ConformalFraudPredictor

    cp = ConformalFraudPredictor(alpha=0.05)
    # calibrate dummy CP on validation-like scores
    dummy_val_p = np.linspace(0.01, 0.99, 100)
    dummy_val_y = (dummy_val_p > 0.5).astype(int)
    cp.calibrate(dummy_val_y, dummy_val_p)

    triage = HierarchicalTriagePolicy(tau_cost=0.30, tau_tdta=0.40, conformal_predictor=cp)

    gen = AdversarialPerturbationGenerator(seed=42)
    s1 = gen.perturb_application_camouflage(sample_raw_data)

    result = evaluate_adversarial_threat_scenario(
        scenario_name="S1_Camouflage",
        df_perturbed_raw=s1,
        df_original_raw=sample_raw_data,
        preprocessor=preprocessor,
        model=model,
        triage_policy=triage,
        frozen_valid_threshold=0.30,
    )

    assert "ROC_AUC" in result
    assert "triage_recall" in result
    assert "pct_auto_approve" in result
    assert result["valid_records"] == len(sample_raw_data)
    assert result["rejected_records"] == 0
    assert result["frauds_auto_flagged"] + result["frauds_referred_review"] + result["frauds_erroneously_approved"] == result["fraud_count"]
