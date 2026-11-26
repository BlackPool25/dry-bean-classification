"""Data acquisition and validation tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import CLASS_ORDER, FEATURES, TARGET
from src.data_loader import load_dataset, validate


@pytest.fixture(scope="module")
def frame():
    return load_dataset()


def test_shape_and_columns(frame):
    assert frame.shape == (13611, 17)
    assert list(frame.columns) == FEATURES + [TARGET]


def test_no_missing_values(frame):
    assert int(frame.isna().sum().sum()) == 0


def test_features_are_numeric_and_finite(frame):
    values = frame[FEATURES].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert (values > 0).all()


def test_target_labels(frame):
    assert set(frame[TARGET].unique()) == set(CLASS_ORDER)
    assert frame[TARGET].nunique() == 7


def test_class_counts_match_published_dataset(frame):
    counts = frame[TARGET].value_counts().to_dict()
    assert counts == {"DERMASON": 3546, "SIRA": 2636, "SEKER": 2027,
                      "HOROZ": 1928, "CALI": 1630, "BARBUNYA": 1322,
                      "BOMBAY": 522}


def test_validation_report_is_complete(frame):
    report = validate(frame)
    assert report["rows"] == 13611
    assert report["feature_count"] == 16
    assert report["duplicate_rows"] == 68
    assert report["all_features_numeric"] is True
    assert report["constant_columns"] == []
    assert len(report["feature_pairs_abs_corr_ge_0.90"]) > 0
