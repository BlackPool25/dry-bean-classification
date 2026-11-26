"""Dataset acquisition and validation for the UCI Dry Bean dataset."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASS_ORDER, DATA_RAW, DATASET_ACCESSED, DATASET_FILE, DATASET_URL,
    FEATURES, METRICS, TARGET,
)


def download_dataset(force: bool = False) -> Path:
    """Fetch the official UCI archive and extract the Excel workbook.

    Returns the path of the extracted workbook. Skips the download when the
    file already exists so the pipeline stays offline-reproducible.
    """
    xlsx = DATA_RAW / DATASET_FILE
    if xlsx.exists() and not force:
        return xlsx

    import urllib.request

    with urllib.request.urlopen(DATASET_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        member = next(n for n in zf.namelist() if n.endswith(".xlsx"))
        xlsx.write_bytes(zf.read(member))
    (DATA_RAW / "dataset.zip.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="utf-8"
    )
    return xlsx


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the workbook as distributed (no cleaning applied)."""
    path = path or (DATA_RAW / DATASET_FILE)
    return pd.read_excel(path)


def validate(df: pd.DataFrame) -> dict:
    """Run the data-integrity checks reported in Chapter 4.

    Returns a JSON-serialisable dictionary; nothing is modified here so the
    report can state exactly what was found before any cleaning decision.
    """
    feature_frame = df[FEATURES]
    numeric_ok = all(np.issubdtype(t, np.number) for t in feature_frame.dtypes)

    iqr_outliers = {}
    for col in FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_outliers[col] = int(((df[col] < low) | (df[col] > high)).sum())

    corr = feature_frame.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    strong_pairs = (
        upper.stack().loc[lambda s: s.abs() >= 0.90]
        .sort_values(key=abs, ascending=False)
        .round(4)
        .to_dict()
    )

    report = {
        "source": DATASET_URL,
        "accessed": DATASET_ACCESSED,
        "file": DATASET_FILE,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_count": len(FEATURES),
        "all_features_numeric": bool(numeric_ok),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_column": {c: int(v) for c, v in df.isna().sum().items()},
        "non_finite_values": int(
            np.isinf(feature_frame.to_numpy(dtype=float)).sum()
            + np.isnan(feature_frame.to_numpy(dtype=float)).sum()
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_excluding_target": int(df.duplicated(subset=FEATURES).sum()),
        "class_counts": {k: int(v) for k, v in df[TARGET].value_counts().items()},
        "class_counts_ordered": {k: int((df[TARGET] == k).sum()) for k in CLASS_ORDER},
        "class_imbalance_ratio": round(
            float(df[TARGET].value_counts().max() / df[TARGET].value_counts().min()), 2
        ),
        "unexpected_labels": sorted(set(df[TARGET].unique()) - set(CLASS_ORDER)),
        "constant_columns": [c for c in FEATURES if df[c].nunique() <= 1],
        "zero_or_negative_values": {
            c: int((df[c] <= 0).sum()) for c in FEATURES if (df[c] <= 0).any()
        },
        "iqr_outlier_counts": iqr_outliers,
        "feature_pairs_abs_corr_ge_0.90": {f"{a} ~ {b}": float(v)
                                          for (a, b), v in strong_pairs.items()},
    }
    return report


def save_validation(report: dict, path: Path | None = None) -> Path:
    path = path or (METRICS / "data_validation.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def load_dataset() -> pd.DataFrame:
    """Download if needed, load, and sanity-check the frame."""
    download_dataset()
    df = load_raw()
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


if __name__ == "__main__":
    frame = load_dataset()
    summary = validate(frame)
    out = save_validation(summary)
    print(json.dumps({k: summary[k] for k in (
        "rows", "columns", "missing_values_total", "duplicate_rows",
        "class_counts_ordered", "class_imbalance_ratio")}, indent=2))
    print(f"validation written to {out}")
