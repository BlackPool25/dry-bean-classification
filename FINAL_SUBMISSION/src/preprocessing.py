"""Preprocessing pipelines and the stratified train/test split.

Every transformation that learns from data (scaling) lives inside a
scikit-learn Pipeline, so it is fitted on training folds only. This is the
mechanism that prevents leakage; nothing is scaled before the split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLASS_ORDER, DATA_PROCESSED, FEATURES, METRICS, RANDOM_STATE, TARGET, TEST_SIZE,
)


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def summary(self) -> dict:
        total = len(self.y_train) + len(self.y_test)
        return {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "stratified": True,
            "train_rows": int(len(self.y_train)),
            "test_rows": int(len(self.y_test)),
            "train_pct": round(100 * len(self.y_train) / total, 2),
            "test_pct": round(100 * len(self.y_test) / total, 2),
            "train_class_counts": {k: int(v) for k, v in
                                   self.y_train.value_counts().reindex(CLASS_ORDER).items()},
            "test_class_counts": {k: int(v) for k, v in
                                  self.y_test.value_counts().reindex(CLASS_ORDER).items()},
        }


def split_data(df: pd.DataFrame) -> Split:
    """Stratified 80/20 split; the test set is touched only in evaluate.py."""
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return Split(X_train, X_test, y_train, y_test)


def build_pipeline(estimator, scale: bool) -> Pipeline:
    """Wrap an estimator in a pipeline, optionally with standardisation.

    Distance- and margin-based models (Logistic Regression, KNN, SVM) need
    features on a common scale; tree ensembles split on thresholds and are
    invariant to monotone rescaling, so they use the raw values.
    """
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("model", estimator))
    return Pipeline(steps)


def persist_split(split: Split) -> None:
    """Write the split to disk and record its composition for the report."""
    train = split.X_train.copy()
    train[TARGET] = split.y_train
    test = split.X_test.copy()
    test[TARGET] = split.y_test
    train.to_csv(DATA_PROCESSED / "train.csv", index=False)
    test.to_csv(DATA_PROCESSED / "test.csv", index=False)
    (METRICS / "split_summary.json").write_text(
        json.dumps(split.summary(), indent=2), encoding="utf-8"
    )


def split_figure() -> None:
    """Bar chart of train/test class counts used as Figure 4.1 in the report."""
    import matplotlib.pyplot as plt
    import numpy as np
    from src.config import FIG_EDA, PALETTE, setup_plot_style

    setup_plot_style()
    train = pd.read_csv(DATA_PROCESSED / "train.csv")[TARGET].value_counts()
    test = pd.read_csv(DATA_PROCESSED / "test.csv")[TARGET].value_counts()
    train, test = train.reindex(CLASS_ORDER), test.reindex(CLASS_ORDER)

    x = np.arange(len(CLASS_ORDER))
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.bar(x - 0.2, train.values, 0.4, label="Training (80%)",
           color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, test.values, 0.4, label="Test (20%)",
           color="#D55E00", edgecolor="black", linewidth=0.5)
    for i, (tr, te) in enumerate(zip(train.values, test.values)):
        ax.text(i - 0.2, tr + 30, f"{tr}", ha="center", fontsize=7.6)
        ax.text(i + 0.2, te + 30, f"{te}", ha="center", fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("Number of beans")
    ax.set_title("Stratified 80/20 partition: class counts in both subsets")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_EDA / "fig_4_1_split.png")
    plt.close(fig)


def load_split() -> Split:
    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    return Split(train[FEATURES], test[FEATURES], train[TARGET], test[TARGET])
