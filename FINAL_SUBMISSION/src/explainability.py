"""Model interpretation: impurity importance, permutation importance and SHAP.

Importance measures describe how the fitted model uses a feature. They do not
establish a biological or physical cause, and correlated inputs share credit
between them, which the report states explicitly.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.config import (
    CLASS_ORDER, FEATURES, FIG_EVAL, METRICS, MODELS, MODEL_LABELS, RESULTS,
    RANDOM_STATE, setup_plot_style,
)
from src.preprocessing import Split

setup_plot_style()


def tree_importance(split: Split) -> pd.DataFrame:
    """Impurity-based importance of the two randomised tree ensembles."""
    frames = []
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for ax, name in zip(axes, ["random_forest", "extra_trees"]):
        model = joblib.load(MODELS / f"{name}.joblib")
        est = model.named_steps["model"]
        imp = pd.Series(est.feature_importances_, index=FEATURES).sort_values()
        imp.to_csv(METRICS / f"feature_importance_{name}.csv")
        frames.append(imp.rename(MODEL_LABELS[name]))
        ax.barh(imp.index, imp.values, color="#009E73", height=0.72)
        ax.set_xlabel("Mean decrease in impurity")
        ax.set_title(MODEL_LABELS[name], fontsize=10)
        ax.tick_params(labelsize=7.6)
    fig.suptitle("Impurity-based feature importance of the tree ensembles",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_EVAL / "fig_6_11_tree_feature_importance.png")
    plt.close(fig)
    return pd.concat(frames, axis=1).round(5)


def permutation(split: Split, best: str, n_repeats: int = 8) -> pd.DataFrame:
    model = joblib.load(MODELS / f"{best}.joblib")
    result = permutation_importance(
        model, split.X_test, split.y_test, scoring="f1_macro",
        n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1,
    )
    table = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    table.round(5).to_csv(METRICS / "permutation_importance_best.csv", index=False)

    ordered = table.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.barh(ordered["feature"], ordered["importance_mean"],
            xerr=ordered["importance_std"], color="#0072B2", height=0.7,
            error_kw=dict(ecolor="black", lw=0.8, capsize=2))
    ax.set_xlabel("Drop in test macro F1 when the feature is permuted")
    ax.set_title(f"Permutation importance - {MODEL_LABELS[best]}")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_EVAL / "fig_6_10_permutation_importance.png")
    plt.close(fig)
    return table


def shap_analysis(split: Split, sample_size: int = 400) -> dict:
    """SHAP values for an Extra Trees model (exact TreeSHAP, cheap here)."""
    try:
        import shap
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"status": f"skipped: {exc}"}

    model = joblib.load(MODELS / "extra_trees.joblib")
    est = model.named_steps["model"]
    X_sample = split.X_test.sample(n=min(sample_size, len(split.X_test)),
                                   random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(est)
    values = explainer.shap_values(X_sample)

    # shap returns either a list per class or an (n, features, classes) array.
    if isinstance(values, list):
        per_class = values
    else:
        arr = np.asarray(values)
        per_class = [arr[:, :, k] for k in range(arr.shape[2])] \
            if arr.ndim == 3 else [arr]

    overall = np.mean([np.abs(v).mean(axis=0) for v in per_class], axis=0)
    table = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": overall}) \
        .sort_values("mean_abs_shap", ascending=False)
    table.round(5).to_csv(METRICS / "shap_importance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ordered = table.sort_values("mean_abs_shap")
    axes[0].barh(ordered["feature"], ordered["mean_abs_shap"],
                 color="#CC79A7", height=0.72)
    axes[0].set_xlabel("mean(|SHAP value|), averaged over the 7 classes")
    axes[0].set_title("Overall SHAP importance (Extra Trees)", fontsize=10)
    axes[0].tick_params(labelsize=7.6)

    idx = CLASS_ORDER.index("BOMBAY") if len(per_class) == len(CLASS_ORDER) else 0
    sv = per_class[idx]
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:8]
    axes[1].set_title("SHAP value distribution for BOMBAY (rare class)",
                      fontsize=10)
    for pos, j in enumerate(order[::-1]):
        vals = sv[:, j]
        feat = X_sample.iloc[:, j].to_numpy()
        norm = (feat - feat.min()) / (np.ptp(feat) + 1e-12)
        axes[1].scatter(vals, np.full_like(vals, pos) + np.random.uniform(
            -0.16, 0.16, size=len(vals)), c=norm, cmap="coolwarm", s=7,
            alpha=0.75, edgecolor="none")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_yticks(range(len(order)))
    axes[1].set_yticklabels([FEATURES[j] for j in order[::-1]], fontsize=7.6)
    axes[1].set_xlabel("SHAP value (impact on BOMBAY log-odds)")
    axes[1].set_ylim(-0.7, len(order) - 0.3)
    fig.tight_layout()
    fig.savefig(FIG_EVAL / "fig_6_12_shap_summary.png")
    plt.close(fig)
    return {"status": "ok", "top_features": table.head(5)["feature"].tolist()}


def run(split: Split, best: str) -> dict:
    tree = tree_importance(split)
    perm = permutation(split, best)
    shap_info = shap_analysis(split)
    meta_path = RESULTS / "experiment_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["permutation_importance_top5"] = perm.head(5)[
        ["feature", "importance_mean"]].round(5).to_dict(orient="records")
    meta["shap"] = shap_info
    meta_path.write_text(json.dumps(meta, indent=2, default=float),
                         encoding="utf-8")
    return {"tree": tree, "permutation": perm, "shap": shap_info}


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.preprocessing import split_data

    s = split_data(load_dataset())
    m = json.loads((RESULTS / "experiment_metadata.json").read_text())
    run(s, m["best_model"])
