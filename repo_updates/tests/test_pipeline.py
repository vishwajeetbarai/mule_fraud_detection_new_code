"""Sanity tests on a small slice of the REAL BaseDataset.csv (no synthetic data)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402
from data_loader import load_base_dataset, temporal_split  # noqa: E402
from features import FeaturePreprocessor  # noqa: E402

pytestmark = pytest.mark.skipif(not config.DATA_PATH.exists(),
                                reason="BaseDataset.csv not found")


@pytest.fixture(scope="module")
def df():
    return load_base_dataset(sample_frac=0.02)


def test_schema_and_target(df):
    assert config.TARGET in df.columns
    assert set(df[config.TARGET].unique()) <= {0, 1}


def test_temporal_split_has_no_overlap(df):
    tr, va, te = temporal_split(df)
    assert tr[config.TIME_COL].max() < va[config.TIME_COL].min()
    assert va[config.TIME_COL].max() < te[config.TIME_COL].min()


def test_no_target_or_time_leakage_in_features(df):
    tr, va, _ = temporal_split(df)
    pre = FeaturePreprocessor().fit(tr)
    X = pre.transform(va)
    assert config.TARGET not in X.columns and config.TIME_COL not in X.columns
    assert list(X.columns) == pre.feature_names_
    assert not X.isin([float("inf"), float("-inf")]).any().any()


def test_unseen_category_is_all_zero(df):
    tr, va, _ = temporal_split(df)
    pre = FeaturePreprocessor().fit(tr)
    va = va.copy()
    va["device_os"] = "brand_new_os"
    X = pre.transform(va)
    assert X[[c for c in X.columns if c.startswith("device_os_")]].sum().sum() == 0
