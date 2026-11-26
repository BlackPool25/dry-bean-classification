"""Command-line prediction using the saved pipeline (scaler + final model)."""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURES, MODELS, RESULTS


def load_model(path=None):
    return joblib.load(path or (MODELS / "best_model.joblib"))


def probabilities(model, frame: pd.DataFrame) -> np.ndarray:
    """Class probabilities, falling back to a softmax of the decision scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame)
    scores = np.asarray(model.decision_function(frame), dtype=float)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_records(records: list[dict], model=None) -> list[dict]:
    """Classify one or more measurement dictionaries.

    Each record must supply all 16 morphological features in any order.
    """
    model = model or load_model()
    frame = pd.DataFrame(records)
    missing = [f for f in FEATURES if f not in frame.columns]
    if missing:
        raise ValueError(f"missing feature(s): {', '.join(missing)}")
    frame = frame[FEATURES].astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("all feature values must be finite numbers")

    proba = probabilities(model, frame)
    classes = list(model.classes_)
    out = []
    for i in range(len(frame)):
        ranked = np.argsort(proba[i])[::-1]
        out.append({
            "predicted_variety": classes[ranked[0]],
            "confidence": round(float(proba[i, ranked[0]]), 4),
            "top3": [{"variety": classes[j],
                      "probability": round(float(proba[i, j]), 4)}
                     for j in ranked[:3]],
        })
    return out


def main(argv=None) -> int:
    meta = json.loads((RESULTS / "experiment_metadata.json").read_text())
    parser = argparse.ArgumentParser(
        description="Classify dry bean varieties from morphological measurements "
                    f"(model: {meta['best_model_label']}).")
    parser.add_argument("--csv", help="CSV file with the 16 feature columns")
    parser.add_argument("--json", help="JSON list of measurement dictionaries")
    for feat in FEATURES:
        parser.add_argument(f"--{feat}", type=float)
    args = parser.parse_args(argv)

    if args.csv:
        records = pd.read_csv(args.csv)[FEATURES].to_dict(orient="records")
    elif args.json:
        records = json.loads(open(args.json, encoding="utf-8").read())
    else:
        values = {f: getattr(args, f) for f in FEATURES}
        absent = [f for f, v in values.items() if v is None]
        if absent:
            parser.error("provide --csv, --json, or all 16 feature values; "
                         f"missing: {', '.join(absent)}")
        records = [values]

    for i, result in enumerate(predict_records(records), start=1):
        top3 = ", ".join(f"{t['variety']} ({t['probability']:.3f})"
                         for t in result["top3"])
        print(f"bean {i}: {result['predicted_variety']} "
              f"[confidence {result['confidence']:.3f}] | top-3: {top3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
