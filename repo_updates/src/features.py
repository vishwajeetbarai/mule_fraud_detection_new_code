"""Row-level preprocessing and feature engineering for BaseDataset.csv.

Everything here is computed per row (or from category levels learned on the TRAIN
split only), so no information flows from validation/test months into training.
"""
import numpy as np
import pandas as pd

import config


class FeaturePreprocessor:
    """Fit on the training split, then transform any split to a float32 matrix."""

    def __init__(self):
        self.categories_ = {}
        self.feature_names_ = None

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

    @staticmethod
    def _engineer(df):
        eps = np.float32(1e-6)
        df["velocity_ratio_6h_24h"] = df["velocity_6h"] / (df["velocity_24h"] + eps)
        df["velocity_ratio_24h_4w"] = df["velocity_24h"] / (df["velocity_4w"] + eps)
        df["velocity_ratio_6h_4w"] = df["velocity_6h"] / (df["velocity_4w"] + eps)
        df["credit_limit_to_income"] = df["proposed_credit_limit"] / (df["income"] + eps)
        df["total_address_months"] = (
            df["current_address_months_count"].fillna(0)
            + df["prev_address_months_count"].fillna(0)
        )
        df["phone_valid_count"] = df["phone_home_valid"] + df["phone_mobile_valid"]
        return df

    # -- public API --------------------------------------------------------
    def fit(self, df):
        for c in config.CATEGORICAL_COLS:
            self.categories_[c] = sorted(df[c].dropna().unique().tolist())
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
            # Map levels never seen in TRAIN to NaN explicitly (pandas >= 3 deprecates
            # passing out-of-category values to pd.Categorical and pandas 4 will raise).
            col = X[c].where(X[c].isin(levels))
            cat = pd.Categorical(col, categories=levels)
            dummies = pd.get_dummies(cat, prefix=c, dtype="int8")
            dummies.index = X.index
            X = pd.concat([X.drop(columns=[c]), dummies], axis=1)

        X = X.drop(columns=[c for c in config.EXTRA_EXCLUDE_FEATURES if c in X.columns])
        return X
