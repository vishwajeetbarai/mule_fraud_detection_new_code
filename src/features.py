"""Row-level preprocessing and feature engineering for BaseDataset.csv.

Everything here is computed per row (or from category levels learned on the TRAIN
split only), so no information flows from validation/test months into training.
"""
import numpy as np
import pandas as pd

import config


class FeaturePreprocessor:
    """Fit on the training split, then transform any split to a float32 matrix.

    Includes:
    - Advanced velocity ratios, surge factors, and burst acceleration.
    - Behavioral graph metrics (device-sharing density, hub centrality proxy, recycling speed).
    - Cross-channel consistency indicators and leverage strain.
    - Temporal drift handling with frozen historical training baseline normalization.
    """

    def __init__(self):
        self.categories_ = {}
        self.feature_names_ = None
        self.historical_stats_ = {}

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _clean_missing(df):
        out = df.copy()
        # Missing-value indicators first (the fact that a value is missing is itself
        # predictive, e.g. 71% of prev_address_months_count is missing).
        for c in config.SENTINEL_MISSING_COLS:
            out[f"{c}_missing"] = (out[c] == -1).astype("int8")
        for c in config.NEGATIVE_IS_MISSING_COLS:
            out[f"{c}_missing"] = (out[c] < 0).astype("int8")

        for c in config.SENTINEL_MISSING_COLS:
            out[c] = out[c].astype("float32").where(out[c] != -1, np.nan)
        for c in config.NEGATIVE_IS_MISSING_COLS:
            out[c] = out[c].astype("float32").where(out[c] >= 0, np.nan)
        return out

    def _engineer(self, df):
        eps = np.float32(1e-6)

        # 1. Advanced Velocity & Surge Acceleration
        df["velocity_ratio_6h_24h"] = df["velocity_6h"] / (df["velocity_24h"] + eps)
        df["velocity_ratio_24h_4w"] = df["velocity_24h"] / (df["velocity_4w"] + eps)
        df["velocity_ratio_6h_4w"] = df["velocity_6h"] / (df["velocity_4w"] + eps)
        # Normalized hourly velocity rate (intensity ratio: 6h hourly rate vs 24h hourly rate)
        df["velocity_surge_intensity"] = (df["velocity_6h"] / 6.0) / ((df["velocity_24h"] / 24.0) + eps)
        df["velocity_burst_acceleration"] = (df["velocity_6h"] / 6.0) - (df["velocity_24h"] / 24.0)
        # Daily rate vs 4-week background rate
        df["velocity_daily_vs_monthly"] = (df["velocity_24h"] / 1.0) / ((df["velocity_4w"] / 28.0) + eps)

        # 2. Behavioral Graph & Bipartite Entity Projections
        # Bipartite entity node degree product: device-email collisions x DOB-email collisions
        device_deg = df["device_distinct_emails_8w"].fillna(0).clip(lower=0)
        dob_deg = df["date_of_birth_distinct_emails_4w"].fillna(0).clip(lower=0)
        zip_deg = df["zip_count_4w"].fillna(0).clip(lower=0)
        branch_deg = df["bank_branch_count_8w"].fillna(0).clip(lower=0)

        df["device_sharing_density"] = device_deg * dob_deg
        # Device recycling speed (fast churn through accounts in short browser/session time)
        session_len = df["session_length_in_minutes"].fillna(df["session_length_in_minutes"].median() if not df.empty else 5.0).clip(lower=0.1)
        df["device_recycling_speed"] = device_deg / (session_len + eps)
        # Composite graph hub centrality proxy (log-degree accumulation across entity edges)
        df["graph_hub_degree_proxy"] = (
            np.log1p(device_deg)
            + np.log1p(dob_deg)
            + np.log1p(zip_deg)
            + np.log1p(branch_deg)
        )
        # Geographic vs institutional dispersion
        df["geo_branch_dispersion_ratio"] = zip_deg / (branch_deg + 1.0)

        # 3. Cross-Channel Consistency & Financial Exposure
        df["credit_limit_to_income"] = df["proposed_credit_limit"] / (df["income"] + eps)
        intended_bal = df["intended_balcon_amount"].fillna(0).clip(lower=0)
        df["proposed_credit_to_balcon"] = df["proposed_credit_limit"] / (intended_bal + 1.0)
        # Identity-channel incoherence: low name similarity combined with free disposable email
        df["name_email_free_mismatch"] = (1.0 - df["name_email_similarity"]) * df["email_is_free"]
        # Cumulative channel risk friction (lack of phone verification + foreign request)
        df["channel_risk_index"] = (
            (1.0 - df["phone_home_valid"])
            + (1.0 - df["phone_mobile_valid"])
            + df["foreign_request"].astype("float32")
        )
        df["income_to_credit_risk"] = df["income"] / (df["credit_risk_score"].clip(lower=1) + eps)

        # 4. Residential and Banking Tenure Ratios
        df["total_address_months"] = (
            df["current_address_months_count"].fillna(0)
            + df["prev_address_months_count"].fillna(0)
        )
        df["address_stability_ratio"] = df["current_address_months_count"].fillna(0) / (df["total_address_months"] + 1.0)
        df["phone_valid_count"] = df["phone_home_valid"] + df["phone_mobile_valid"]
        df["bank_tenure_to_age_ratio"] = df["bank_months_count"].fillna(0) / (df["customer_age"] * 12.0 + eps)

        # 5. Temporal Drift Adjustments (using baseline frozen during fit)
        if self.historical_stats_:
            for col in ["velocity_6h", "proposed_credit_limit", "credit_risk_score"]:
                mean = self.historical_stats_.get(f"{col}_mean", 0.0)
                std = self.historical_stats_.get(f"{col}_std", 1.0)
                df[f"{col}_drift_zscore"] = (df[col] - mean) / (std + eps)

        return df

    # -- public API --------------------------------------------------------
    def fit(self, df):
        # Learn categorical levels
        for c in config.CATEGORICAL_COLS:
            self.categories_[c] = sorted(df[c].dropna().unique().tolist())

        # Fit historical rolling statistics on TRAIN split to calibrate drift detection
        for col in ["velocity_6h", "proposed_credit_limit", "credit_risk_score"]:
            vals = df[col].dropna()
            self.historical_stats_[f"{col}_mean"] = float(vals.mean())
            self.historical_stats_[f"{col}_std"] = float(vals.std()) if vals.std() > 0 else 1.0

        self.feature_names_ = list(self._build(df).columns)
        return self

    def transform(self, df):
        X = self._build(df)
        return X.reindex(columns=self.feature_names_, fill_value=0).astype("float32")

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def _build(self, df):
        drop = set(config.DROP_COLS) | {config.TARGET, config.TIME_COL}
        X = self._clean_missing(df.drop(columns=[c for c in drop if c in df.columns]))
        X = self._engineer(X)

        # One-hot with levels frozen from the training split (unseen level -> all zeros).
        for c in config.CATEGORICAL_COLS:
            levels = self.categories_[c]
            # Map levels never seen in TRAIN to NaN explicitly
            col = X[c].where(X[c].isin(levels))
            cat = pd.Categorical(col, categories=levels)
            dummies = pd.get_dummies(cat, prefix=c, dtype="int8")
            dummies.index = X.index
            X = pd.concat([X.drop(columns=[c]), dummies], axis=1)

        X = X.drop(columns=[c for c in config.EXTRA_EXCLUDE_FEATURES if c in X.columns])
        return X

