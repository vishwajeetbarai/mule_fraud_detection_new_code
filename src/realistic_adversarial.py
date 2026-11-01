"""Phase 4 Realistic Adversarial Robustness Validation Suite.

Executes constrained, domain-valid perturbations on RAW input data prior to
feature engineering, guaranteeing that derived behavioral graph, velocity ratio,
and incoherence features are recomputed dynamically.

Evaluates clean vs perturbed performance against frozen models and validation-selected
thresholds under the Hierarchical Triage Policy (AUTO_APPROVE, HUMAN_REVIEW, AUTO_FLAG).
"""
import copy
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import config
import evaluation as ev
from statistical_reliability import wilson_score_interval


class DomainValidator:
    """Validates that perturbed raw records strictly respect physical, domain,
    and business logic boundaries of the BAF benchmark.
    """

    ALLOWED_CATEGORIES = {
        "payment_type": {"AA", "AB", "AC", "AD", "AE"},
        "employment_status": {"CA", "CB", "CC", "CD", "CE", "CF", "CG"},
        "housing_status": {"BA", "BB", "BC", "BD", "BE", "BF", "BG"},
        "source": {"INTERNET", "TELEAPP"},
        "device_os": {"linux", "macintosh", "other", "windows", "x11"},
    }

    IMMUTABLE_COLS = [
        "month",
        "fraud_bool",
        "credit_risk_score",
        "bank_months_count",
        "zip_count_4w",
        "bank_branch_count_8w",
        "date_of_birth_distinct_emails_4w",
        "customer_age",
        "days_since_request",
    ]

    @classmethod
    def validate_record(cls, raw_row: pd.Series, original_row: pd.Series = None) -> Tuple[bool, str]:
        """Validate a single raw record. Returns (is_valid, failure_reason)."""
        # 1. Non-negativity for physical counts/amounts (except sentinels)
        if raw_row["income"] < 0.0 or raw_row["income"] > 1.0:
            return False, "income out of [0.0, 1.0]"
        if raw_row["name_email_similarity"] < 0.0 or raw_row["name_email_similarity"] > 1.0:
            return False, "name_email_similarity out of [0.0, 1.0]"
        if raw_row["proposed_credit_limit"] < 0.0:
            return False, "negative proposed_credit_limit"
        if raw_row["velocity_6h"] < 0.0 or raw_row["velocity_24h"] < 0.0 or raw_row["velocity_4w"] < 0.0:
            return False, "negative velocity count"
        if raw_row["session_length_in_minutes"] < -1.0 or raw_row["session_length_in_minutes"] == 0:
            return False, "invalid session_length"

        # 2. Sentinels
        if raw_row["current_address_months_count"] < -1.0 or raw_row["prev_address_months_count"] < -1.0:
            return False, "address months < -1"
        if raw_row["device_distinct_emails_8w"] < -1.0:
            return False, "device_distinct_emails_8w < -1"

        # 3. Categorical vocabulary
        for cat_col, valid_set in cls.ALLOWED_CATEGORIES.items():
            if cat_col in raw_row and raw_row[cat_col] not in valid_set:
                return False, f"invalid categorical value {raw_row[cat_col]} for {cat_col}"

        # 4. Biological sanity: address tenure cannot exceed age in months
        if raw_row["current_address_months_count"] > 0:
            max_possible_months = (raw_row["customer_age"] + 10) * 12.0
            if raw_row["current_address_months_count"] > max_possible_months:
                return False, "address tenure exceeds applicant biological age"

        # 5. Immutability checks against original record
        if original_row is not None:
            for imm_col in cls.IMMUTABLE_COLS:
                if imm_col in raw_row and imm_col in original_row:
                    if raw_row[imm_col] != original_row[imm_col]:
                        return False, f"tampered immutable column {imm_col}"

        return True, "valid"

    @classmethod
    def filter_valid_records(cls, df_perturbed: pd.DataFrame, df_original: pd.DataFrame) -> Tuple[pd.DataFrame, int, List[str]]:
        """Filter out records failing domain constraints.
        
        Returns:
            (valid_df, n_rejected, rejection_reasons)
        """
        valid_indices = []
        reasons = []
        n_rejected = 0

        for idx in df_perturbed.index:
            p_row = df_perturbed.loc[idx]
            o_row = df_original.loc[idx]
            is_valid, reason = cls.validate_record(p_row, o_row)
            if is_valid:
                valid_indices.append(idx)
            else:
                n_rejected += 1
                reasons.append(reason)

        return df_perturbed.loc[valid_indices].copy(), n_rejected, reasons


class AdversarialPerturbationGenerator:
    """Generates realistic, domain-constrained perturbations on RAW tabular data."""

    def __init__(self, seed: int = config.RANDOM_STATE):
        self.rng = np.random.RandomState(seed)

    def perturb_application_camouflage(self, df_raw: pd.DataFrame, strength: float = 0.2) -> pd.DataFrame:
        """Scenario 1: Applicant form tuning (name-email similarity, free email, credit limit)."""
        df = df_raw.copy()
        # Camouflage email matching
        df["name_email_similarity"] = np.clip(df["name_email_similarity"] + strength, 0.0, 1.0)
        # Shift away from free disposable email
        df["email_is_free"] = 0
        # Reduce requested credit limit towards safer range (e.g. 500)
        df["proposed_credit_limit"] = np.maximum(200.0, df["proposed_credit_limit"] * (1.0 - strength))
        # Increase reported residential stability
        col_dtype = df["current_address_months_count"].dtype
        mask = df["current_address_months_count"] >= 0
        df["current_address_months_count"] = np.where(
            mask,
            df["current_address_months_count"] + 12,
            df["current_address_months_count"]
        ).astype(col_dtype)
        return df

    def perturb_velocity_throttling(self, df_raw: pd.DataFrame, throttle_ratio: float = 0.25) -> pd.DataFrame:
        """Scenario 2: Throttling burst rate submissions and increasing session dwell time."""
        df = df_raw.copy()
        # Dampen 6h short-term velocity burst
        df["velocity_6h"] = np.maximum(0.0, df["velocity_6h"] * (1.0 - throttle_ratio))
        # Increase dwell time on page
        mask_session = df["session_length_in_minutes"] > 0
        df.loc[mask_session, "session_length_in_minutes"] = df.loc[mask_session, "session_length_in_minutes"] + 3.0
        return df

    def perturb_device_cloaking(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Scenario 3: Device OS masquerade and profile isolation."""
        df = df_raw.copy()
        # Conceal device email collisions down to 1
        df["device_distinct_emails_8w"] = df["device_distinct_emails_8w"].apply(
            lambda x: 1 if x > 1 else x
        )
        # Masquerade OS as standard windows desktop
        df["device_os"] = "windows"
        df["keep_alive_session"] = 1
        return df

    def perturb_missingness_exploitation(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Scenario 4: Intentional non-disclosure of optional form fields."""
        df = df_raw.copy()
        df["prev_address_months_count"] = -1
        df["intended_balcon_amount"] = -1.0
        df["phone_home_valid"] = 0
        return df

    def perturb_multi_vector_syndicate(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Scenario 5: Sophisticated joint evasion (S1 + S2 + S3)."""
        df = self.perturb_application_camouflage(df_raw, strength=0.2)
        df = self.perturb_velocity_throttling(df, throttle_ratio=0.25)
        df = self.perturb_device_cloaking(df)
        return df


def evaluate_adversarial_threat_scenario(
    scenario_name: str,
    df_perturbed_raw: pd.DataFrame,
    df_original_raw: pd.DataFrame,
    preprocessor: Any,
    model: Any,
    triage_policy: Any,
    frozen_valid_threshold: float,
    cost_fn: float = config.COST_FN,
    cost_fp: float = config.COST_FP,
    cost_tp: float = config.COST_TP,
) -> Dict[str, Any]:
    """Evaluates an adversarial scenario end-to-end through preprocessor, model, and triage."""
    # 1. Domain Validation & Rejection
    valid_df, n_rejected, reasons = DomainValidator.filter_valid_records(df_perturbed_raw, df_original_raw)
    n_valid = len(valid_df)
    n_total = len(df_original_raw)

    if n_valid == 0:
        raise ValueError(f"All records rejected in scenario {scenario_name}")

    y_true = valid_df[config.TARGET].values

    # 2. Run feature engineering on perturbed raw data
    X_engineered = preprocessor.transform(valid_df)

    # 3. Model score prediction
    probas = model.predict_proba(X_engineered)[:, 1]

    # 4. Standard ranking and threshold metrics
    auc = float(roc_auc_score(y_true, probas))
    pr_auc = float(average_precision_score(y_true, probas))
    tpr1 = float(ev.tpr_at_fpr(y_true, probas, 0.01))
    tpr5 = float(ev.tpr_at_fpr(y_true, probas, 0.05))

    # Binary metrics at frozen validation threshold
    preds_binary = (probas >= frozen_valid_threshold).astype(int)
    cm = ev.confusion_matrix(y_true, preds_binary)
    if isinstance(cm, dict):
        tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    elif hasattr(cm, "ravel") and len(cm.ravel()) == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = int(np.sum((y_true == 0) & (preds_binary == 0)))
        fp = int(np.sum((y_true == 0) & (preds_binary == 1)))
        fn = int(np.sum((y_true == 1) & (preds_binary == 0)))
        tp = int(np.sum((y_true == 1) & (preds_binary == 1)))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # 5. Hierarchical Triage Decisions
    triage_results = triage_policy.route(probas)
    actions = triage_results["action"].values

    n_app = len(actions)
    auto_approve_cnt = int(np.sum(actions == "AUTO_APPROVE"))
    human_review_cnt = int(np.sum(actions == "HUMAN_REVIEW"))
    auto_flag_cnt = int(np.sum(actions == "AUTO_FLAG"))

    pct_auto_approve = (auto_approve_cnt / n_app) * 100.0
    pct_human_review = (human_review_cnt / n_app) * 100.0
    pct_auto_flag = (auto_flag_cnt / n_app) * 100.0

    # Triage fraud routing breakdown
    fraud_mask = (y_true == 1)
    n_frauds = int(np.sum(fraud_mask))

    frauds_auto_flagged = int(np.sum((actions == "AUTO_FLAG") & fraud_mask))
    frauds_in_review = int(np.sum((actions == "HUMAN_REVIEW") & fraud_mask))
    frauds_auto_approved = int(np.sum((actions == "AUTO_APPROVE") & fraud_mask))

    triage_recall = float((frauds_auto_flagged + frauds_in_review) / n_frauds) if n_frauds > 0 else 0.0
    direct_flag_recall = float(frauds_auto_flagged / n_frauds) if n_frauds > 0 else 0.0

    # Simulated operational cost calculation
    cost_val = ev.compute_cost_matrix(
        y_true=y_true,
        pred=preds_binary,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        cost_tp=cost_tp,
    )
    avg_cost = float(cost_val["cost_per_application"])

    # Wilson score interval on triage recall
    _, rec_lower, rec_upper = wilson_score_interval(frauds_auto_flagged + frauds_in_review, n_frauds)

    return {
        "scenario": scenario_name,
        "total_records": n_total,
        "valid_records": n_valid,
        "rejected_records": n_rejected,
        "rejection_rate_pct": (n_rejected / n_total) * 100.0,
        "fraud_count": n_frauds,
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "TPR_at_1pct_FPR": tpr1,
        "TPR_at_5pct_FPR": tpr5,
        "Precision": prec,
        "Recall": rec,
        "FPR": fpr,
        "avg_cost_per_applicant": avg_cost,
        "pct_auto_approve": pct_auto_approve,
        "pct_human_review": pct_human_review,
        "pct_auto_flag": pct_auto_flag,
        "direct_flag_recall": direct_flag_recall,
        "triage_recall": triage_recall,
        "triage_recall_ci_95": f"{triage_recall:.4f} [{rec_lower:.4f}, {rec_upper:.4f}]",
        "frauds_auto_flagged": frauds_auto_flagged,
        "frauds_referred_review": frauds_in_review,
        "frauds_erroneously_approved": frauds_auto_approved,
    }
