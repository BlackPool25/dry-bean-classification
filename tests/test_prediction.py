"""Saved-model and prediction-interface tests."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import CLASS_ORDER, FEATURES, MODELS, RESULTS
from src.predict import load_model, predict_records

pytestmark = pytest.mark.skipif(
    not (MODELS / "best_model.joblib").exists(),
    reason="run `python -m src.run_pipeline` first",
)


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def sample_rows():
    return pd.read_csv(RESULTS.parent / "data" / "processed" / "test.csv").head(10)


def test_model_file_exists():
    assert (MODELS / "best_model.joblib").exists()
    assert (RESULTS / "experiment_metadata.json").exists()


def test_prediction_returns_valid_class(model, sample_rows):
    out = predict_records(sample_rows[FEATURES].to_dict(orient="records"), model)
    assert len(out) == 10
    assert all(r["predicted_variety"] in CLASS_ORDER for r in out)


def test_probabilities_are_a_distribution(model, sample_rows):
    out = predict_records(sample_rows[FEATURES].to_dict(orient="records"), model)
    for record in out:
        total = sum(t["probability"] for t in record["top3"])
        assert 0.0 < total <= 1.0 + 1e-6
        assert record["confidence"] == pytest.approx(
            record["top3"][0]["probability"], abs=1e-4)


def test_missing_feature_is_rejected(model):
    record = {f: 1.0 for f in FEATURES[:-1]}
    with pytest.raises(ValueError, match="missing feature"):
        predict_records([record], model)


def test_non_finite_values_are_rejected(model):
    record = {f: 1.0 for f in FEATURES}
    record["Area"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        predict_records([record], model)


def test_metadata_matches_model_comparison_table():
    meta = json.loads((RESULTS / "experiment_metadata.json").read_text())
    table = pd.read_csv(RESULTS / "model_comparison.csv")
    row = table.loc[table["model"] == meta["best_model"]].iloc[0]
    assert row["f1_macro"] == pytest.approx(meta["best_model_test"]["f1_macro"],
                                            abs=1e-6)
    assert row["f1_macro"] == table["f1_macro"].max()


def test_cli_round_trip(tmp_path):
    from src.predict import main

    frame = pd.DataFrame([{f: 1.0 for f in FEATURES}])
    frame["Area"] = 50000.0
    csv = tmp_path / "one.csv"
    frame.to_csv(csv, index=False)
    assert main(["--csv", str(csv)]) == 0
