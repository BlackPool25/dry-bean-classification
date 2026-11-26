"""Model training, cross-validated hyperparameter search and model selection.

The test set is not referenced anywhere in this module. All tuning uses
stratified 5-fold cross-validation on the training partition only.
"""
from __future__ import annotations

import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.config import (
    CV_FOLDS, METRICS, MODELS, MODEL_LABELS, RANDOM_STATE, RESULTS, SCORING,
)
from src.preprocessing import Split, build_pipeline, split_data

SEARCH_VERBOSE = 0


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def model_zoo() -> dict[str, dict[str, Any]]:
    """Estimator, search space, scaling requirement and search strategy."""
    rs = RANDOM_STATE
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                solver="lbfgs", max_iter=3000, random_state=rs),
            "params": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "scale": True,
            "search": "grid",
        },
        "knn": {
            "estimator": KNeighborsClassifier(),
            "params": {
                "model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 41],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
            "scale": True,
            "search": "grid",
        },
        "svm_rbf": {
            # Platt scaling costs 7x per fit, so it is applied once to the
            # winning configuration in finalise_svm() instead of to all 80 fits.
            "estimator": SVC(kernel="rbf", cache_size=600, random_state=rs),
            "params": {
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.001, 0.01, 0.1],
            },
            "scale": True,
            "search": "grid",
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=300, random_state=rs, n_jobs=1),
            "params": {
                "model__max_depth": [None, 20, 30, 40],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                # 16 features: sqrt == log2 == 0.3 (all 4), so only two
                # genuinely different values exist.
                "model__max_features": ["sqrt", 0.5],
            },
            "scale": False,
            "search": "random",
            "n_iter": 25,
        },
        "extra_trees": {
            "estimator": ExtraTreesClassifier(
                n_estimators=300, random_state=rs, n_jobs=1),
            "params": {
                "model__max_depth": [None, 20, 30, 40],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                # 16 features: sqrt == log2 == 0.3 (all 4), so only two
                # genuinely different values exist.
                "model__max_features": ["sqrt", 0.5],
            },
            "scale": False,
            "search": "random",
            "n_iter": 25,
        },
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingClassifier(random_state=rs),
            "params": {
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__max_iter": [200, 400, 600],
                "model__max_leaf_nodes": [31, 63],
                "model__min_samples_leaf": [20, 50],
                "model__l2_regularization": [0.0, 1.0],
            },
            "scale": False,
            "search": "random",
            "n_iter": 20,
        },
    }


def _search(model_name: str, spec: dict, split: Split):
    pipe = build_pipeline(spec["estimator"], spec["scale"])
    kwargs = dict(
        estimator=pipe,
        scoring=SCORING,
        cv=cv_splitter(),
        n_jobs=-1,
        refit=True,
        verbose=SEARCH_VERBOSE,
        return_train_score=False,
    )
    if spec["search"] == "grid":
        return GridSearchCV(param_grid=spec["params"], **kwargs)
    return RandomizedSearchCV(
        param_distributions=spec["params"], n_iter=spec["n_iter"],
        random_state=RANDOM_STATE, **kwargs,
    )


def _cv_frame(search, model_name: str) -> pd.DataFrame:
    res = pd.DataFrame(search.cv_results_)
    param_cols = [c for c in res.columns if c.startswith("param_")]
    keep = param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]
    out = res[keep].copy()
    out.insert(0, "model", model_name)
    for c in param_cols:
        out[c] = out[c].astype(str)
    return out


def train_all(split: Split) -> dict:
    zoo = model_zoo()
    summary_rows, cv_frames, per_fold = [], [], {}

    for name, spec in zoo.items():
        print(f"[{name}] searching ...", flush=True)
        search = _search(name, spec, split)
        t0 = time.perf_counter()
        search.fit(split.X_train, split.y_train)
        elapsed = time.perf_counter() - t0

        cv_frames.append(_cv_frame(search, name))
        best_idx = search.best_index_
        fold_scores = [
            search.cv_results_[f"split{k}_test_score"][best_idx]
            for k in range(CV_FOLDS)
        ]
        per_fold[name] = fold_scores

        joblib.dump(search.best_estimator_, MODELS / f"{name}.joblib")
        summary_rows.append({
            "model": name,
            "label": MODEL_LABELS[name],
            "search": spec["search"],
            "n_candidates": len(search.cv_results_["params"]),
            "cv_macro_f1_mean": round(float(search.best_score_), 5),
            "cv_macro_f1_std": round(float(np.std(fold_scores, ddof=1)), 5),
            "cv_macro_f1_min": round(float(np.min(fold_scores)), 5),
            "cv_macro_f1_max": round(float(np.max(fold_scores)), 5),
            "search_time_s": round(elapsed, 2),
            "best_params": json.dumps(
                {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                 for k, v in search.best_params_.items()},
                default=str),
        })
        print(f"    best CV macro-F1 = {search.best_score_:.4f} "
              f"({elapsed:.1f}s)", flush=True)

    summary = pd.DataFrame(summary_rows).sort_values(
        "cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    summary.to_csv(METRICS / "cv_summary.csv", index=False)
    pd.concat(cv_frames, ignore_index=True).to_csv(RESULTS / "cv_results.csv", index=False)
    pd.DataFrame(per_fold).to_csv(METRICS / "cv_fold_scores.csv", index=False)
    return {"summary": summary, "per_fold": per_fold}


REDUNDANT_SIZE_CLUSTER = ["Perimeter", "MajorAxisLength", "MinorAxisLength",
                          "ConvexArea", "EquivDiameter"]


def feature_ablation(split: Split, summary: pd.DataFrame) -> pd.DataFrame:
    """Does removing the redundant size cluster improve generalisation?

    Random Forest is used as the reference learner with its tuned settings so
    the comparison isolates the effect of the feature set.
    """
    from src.config import FEATURES

    best_rf_params = json.loads(
        summary.loc[summary["model"] == "random_forest", "best_params"].iloc[0])
    best_rf_params = {f"model__{k}": v for k, v in best_rf_params.items()}
    keep = [f for f in FEATURES if f not in REDUNDANT_SIZE_CLUSTER]
    rows = []
    for label, cols in (("all_16_features", FEATURES),
                        ("11_features_reduced", keep)):
        pipe = build_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1,
                                   **{k.replace("model__", ""): v
                                      for k, v in best_rf_params.items()}),
            scale=False)
        scores = cross_val_score(pipe, split.X_train[cols], split.y_train,
                                 cv=cv_splitter(), scoring=SCORING, n_jobs=-1)
        rows.append({"feature_set": label, "n_features": len(cols),
                     "cv_macro_f1_mean": round(float(scores.mean()), 5),
                     "cv_macro_f1_std": round(float(scores.std(ddof=1)), 5)})
    table = pd.DataFrame(rows)
    table.to_csv(METRICS / "feature_ablation.csv", index=False)
    return table


def class_weight_experiment(split: Split, best_name: str) -> pd.DataFrame:
    """Compare the original class distribution with balanced class weights."""
    from src.config import MODEL_LABELS

    zoo = model_zoo()
    spec = zoo[best_name]
    est = spec["estimator"]
    if not hasattr(est, "class_weight"):
        return pd.DataFrame()

    rows = []
    for label, weight in (("none", None), ("balanced", "balanced")):
        extra = {"n_jobs": 1} if "n_jobs" in est.get_params() else {}
        clone = type(est)(**{**est.get_params(), **extra,
                             "class_weight": weight})
        pipe = build_pipeline(clone, spec["scale"])
        scores = cross_val_score(pipe, split.X_train, split.y_train,
                                 cv=cv_splitter(), scoring=SCORING, n_jobs=-1)
        rows.append({"model": MODEL_LABELS[best_name], "class_weight": label,
                     "cv_macro_f1_mean": round(float(scores.mean()), 5),
                     "cv_macro_f1_std": round(float(scores.std(ddof=1)), 5)})
    table = pd.DataFrame(rows)
    table.to_csv(METRICS / "class_weight_experiment.csv", index=False)
    return table


def finalise_svm(split: Split, summary: pd.DataFrame) -> float:
    """Refit the winning SVM once with Platt-scaled class probabilities.

    The search itself runs without probabilities because that costs 7x per fit;
    the report needs calibrated scores only for the final ROC curves.
    """
    params = json.loads(summary.loc[summary["model"] == "svm_rbf",
                                    "best_params"].iloc[0])
    pipe = build_pipeline(
        SVC(kernel="rbf", cache_size=600, probability=True,
            random_state=RANDOM_STATE, **params),
        scale=True,
    )
    t0 = time.perf_counter()
    pipe.fit(split.X_train, split.y_train)
    joblib.dump(pipe, MODELS / "svm_rbf.joblib")
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"    SVM refitted with Platt scaling in {elapsed}s")
    return elapsed


def select_best(summary: pd.DataFrame) -> str:
    """Highest mean CV macro-F1; ties broken by the more stable fold scores."""
    ordered = summary.sort_values(
        ["cv_macro_f1_mean", "cv_macro_f1_std"], ascending=[False, True])
    return str(ordered.iloc[0]["model"])


def run(split: Split | None = None) -> dict:
    if split is None:
        from src.data_loader import load_dataset
        split = split_data(load_dataset())

    out = train_all(split)
    summary = out["summary"]
    svm_refit_s = finalise_svm(split, summary)
    best = select_best(summary)

    ablation = feature_ablation(split, summary)
    weights = class_weight_experiment(split, best)

    joblib.dump(joblib.load(MODELS / f"{best}.joblib"), MODELS / "best_model.joblib")
    meta = {
        "selection_metric": SCORING,
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "best_model": best,
        "best_model_label": MODEL_LABELS[best],
        "cv_macro_f1_mean": float(summary.iloc[0]["cv_macro_f1_mean"]),
        "cv_macro_f1_std": float(summary.iloc[0]["cv_macro_f1_std"]),
        "best_params": json.loads(summary.iloc[0]["best_params"]),
        "ranking": summary[["model", "cv_macro_f1_mean", "cv_macro_f1_std"]]
        .to_dict(orient="records"),
        "feature_ablation": ablation.to_dict(orient="records"),
        "class_weight_experiment": weights.to_dict(orient="records") if len(weights) else [],
        "svm_probability_refit_s": svm_refit_s,
    }
    (RESULTS / "experiment_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSelected model: {MODEL_LABELS[best]} "
          f"(CV macro-F1 = {meta['cv_macro_f1_mean']:.4f})")
    return {"summary": summary, "best": best, "meta": meta}


if __name__ == "__main__":
    run()
