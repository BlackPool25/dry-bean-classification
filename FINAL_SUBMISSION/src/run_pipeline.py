"""End-to-end pipeline: validate -> split -> EDA -> train -> evaluate -> explain.

Run from the project root:  python -m src.run_pipeline
"""
from __future__ import annotations

import json
import time

from src import eda, evaluate, explainability, train
from src.config import METRICS, RESULTS
from src.data_loader import load_dataset, save_validation, validate
from src.preprocessing import persist_split, split_data, split_figure


def main() -> None:
    t_start = time.perf_counter()

    print("== 1/6 data acquisition and validation ==")
    df = load_dataset()
    report = validate(df)
    save_validation(report)
    print(f"   {report['rows']} rows x {report['columns']} columns, "
          f"{report['missing_values_total']} missing values, "
          f"{report['duplicate_rows']} duplicate rows")

    print("== 2/6 stratified split ==")
    split = split_data(df)
    persist_split(split)
    split_figure()
    print(f"   train {len(split.y_train)} / test {len(split.y_test)}")

    print("== 3/6 exploratory data analysis ==")
    eda.run_all(df)

    print("== 4/6 model training and tuning ==")
    trained = train.run(split)

    print("== 5/6 held-out evaluation ==")
    evaluate.run(split)

    print("== 6/6 interpretation ==")
    explainability.run(split, trained["best"])

    meta = json.loads((RESULTS / "experiment_metadata.json").read_text())
    meta["pipeline_runtime_s"] = round(time.perf_counter() - t_start, 1)
    (RESULTS / "experiment_metadata.json").write_text(
        json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print(f"\nDone in {meta['pipeline_runtime_s']} s. "
          f"Best model: {meta['best_model_label']}")


if __name__ == "__main__":
    main()
