"""Central configuration for the Bank Account Fraud (BaseDataset.csv) pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Dataset location. Override with the BASE_DATASET_PATH env var or --data CLI flag.
DATA_PATH = Path(os.environ.get("BASE_DATASET_PATH", ROOT / "data" / "BaseDataset.csv"))
OUTPUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# Schema of the NeurIPS 2022 Bank Account Fraud "Base" variant
# ----------------------------------------------------------------------------
TARGET = "fraud_bool"
TIME_COL = "month"  # 0..7, used ONLY for temporal splitting, never as a feature

CATEGORICAL_COLS = [
    "payment_type", "employment_status", "housing_status", "source", "device_os",
]

# Columns that use -1 as a "missing" sentinel (see dataset datasheet).
SENTINEL_MISSING_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
# Negative values mean "missing" here (range is roughly -15..113).
NEGATIVE_IS_MISSING_COLS = ["intended_balcon_amount"]

# Constant (always 0) in the Base variant -> no information.
DROP_COLS = ["device_fraud_count"]

# Optional: add e.g. "customer_age" to remove a sensitive attribute from the model.
EXTRA_EXCLUDE_FEATURES = []

# ----------------------------------------------------------------------------
# Temporal split (months). Mirrors the benchmark's train=0-5 / test=6-7 protocol,
# with month 5 carved out of training as the early-stopping / threshold-setting
# validation set so the test months are never touched during model selection.
# ----------------------------------------------------------------------------
TRAIN_MONTHS = [0, 1, 2, 3, 4]
VALID_MONTHS = [5]
TEST_MONTHS = [6, 7]

# Expanding-window temporal CV: test on each of these months, train on earlier ones.
CV_TEST_MONTHS = [3, 4, 5, 6, 7]

# Operating point used by the benchmark (paper reports TPR at 5% FPR).
TARGET_FPR = 0.05

# ----------------------------------------------------------------------------
# Model hyper-parameters
# ----------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=1000, max_depth=6, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
    objective="binary:logistic", eval_metric="aucpr",
    tree_method="hist", early_stopping_rounds=50,
)
LGB_PARAMS = dict(
    n_estimators=1000, num_leaves=31, learning_rate=0.05,
    subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
    min_child_samples=50, objective="binary", metric="average_precision", verbosity=-1,
)
LGB_EARLY_STOPPING_ROUNDS = 50
SHAP_SAMPLE_SIZE = 5000
