"""Split integrity and leakage-prevention tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import FEATURES, RANDOM_STATE, TEST_SIZE
from src.preprocessing import build_pipeline, split_data


@pytest.fixture(scope="module")
def split():
    from src.data_loader import load_dataset
    return split_data(load_dataset())


def test_split_sizes(split):
    total = len(split.y_train) + len(split.y_test)
    assert total == 13611
    assert abs(len(split.y_test) / total - TEST_SIZE) < 0.005


def test_split_is_stratified(split):
    train_share = split.y_train.value_counts(normalize=True)
    test_share = split.y_test.value_counts(normalize=True)
    assert (train_share - test_share).abs().max() < 0.005


def test_split_is_deterministic(split):
    from src.data_loader import load_dataset
    again = split_data(load_dataset())
    assert again.X_train.equals(split.X_train)
    assert again.y_test.equals(split.y_test)
    assert RANDOM_STATE == 42


def test_partitions_reconstruct_the_dataset(split):
    """The two partitions are a partition of the original rows.

    A plain row-overlap test would fail: the dataset contains 68 exact duplicate
    rows, and a duplicate can legitimately land on both sides of the split.
    Comparing the combined multiset against the original is the correct check.
    """
    from src.data_loader import load_dataset
    from src.config import FEATURES

    original = load_dataset()[FEATURES].sort_values(by=FEATURES).reset_index(drop=True)
    combined = (pd.concat([split.X_train, split.X_test])
                .sort_values(by=FEATURES).reset_index(drop=True))
    pd.testing.assert_frame_equal(combined, original)


def test_pipeline_scaler_is_fitted_on_training_data_only(split):
    pipe = build_pipeline(LogisticRegression(), scale=True)
    pipe.fit(split.X_train, split.y_train)
    scaler = pipe.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, split.X_train.mean().to_numpy(),
                               rtol=1e-10)


def test_tree_pipeline_has_no_scaler(split):
    pipe = build_pipeline(RandomForestClassifier(n_estimators=5), scale=False)
    assert list(pipe.named_steps) == ["model"]


def test_standardisation_is_scale_invariant(split):
    """Rescaling every input by a constant must not change the scaled values.

    This is the property that makes StandardScaler the right preprocessing step
    for the distance-based models, and it is checked on the scaler output because
    a kNN prediction comparison would be sensitive to ties between duplicate rows.
    """
    from sklearn.preprocessing import StandardScaler

    values = split.X_train.iloc[:2000]
    first = StandardScaler().fit_transform(values)
    second = StandardScaler().fit_transform(values * 100.0)
    np.testing.assert_allclose(first, second, atol=1e-6)
