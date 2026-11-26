"""Final evaluation on the held-out test set and all evaluation figures."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import (
    CLASS_ORDER, FEATURES, FIG_EVAL, FIG_MODELS, METRICS, MODELS, MODEL_LABELS,
    PALETTE, PREDICTIONS, RESULTS, TARGET, setup_plot_style,
)
from src.predict import probabilities
from src.preprocessing import Split

setup_plot_style()
MODEL_ORDER = ["logistic_regression", "knn", "svm_rbf", "random_forest",
               "extra_trees", "hist_gradient_boosting"]


def _scores(model, X):
    return probabilities(model, X)


def _time_predict(model, X, repeats: int = 3) -> float:
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        best = min(best, time.perf_counter() - t0)
    return 1000.0 * best  # milliseconds


def evaluate_models(split: Split) -> pd.DataFrame:
    rows, reports, preds = [], {}, {}
    y = np.asarray(split.y_test, dtype=str)
    for name in MODEL_ORDER:
        model = joblib.load(MODELS / f"{name}.joblib")
        t0 = time.perf_counter()
        y_pred = model.predict(split.X_test)
        fit_predict_s = time.perf_counter() - t0
        preds[name] = y_pred

        scores = _scores(model, split.X_test)
        try:
            auc = roc_auc_score(y, scores, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")

        rows.append({
            "model": name,
            "label": MODEL_LABELS[name],
            "accuracy": accuracy_score(y, y_pred),
            "precision_macro": precision_score(y, y_pred, average="macro",
                                               zero_division=0),
            "recall_macro": recall_score(y, y_pred, average="macro",
                                         zero_division=0),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, y_pred, average="weighted",
                                    zero_division=0),
            "roc_auc_ovr_macro": auc,
            "predict_time_ms": _time_predict(model, split.X_test),
        })
        reports[name] = classification_report(
            y, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)

    table = pd.DataFrame(rows)
    train_time = pd.read_csv(METRICS / "cv_summary.csv")[
        ["model", "search_time_s", "cv_macro_f1_mean"]]
    table = table.merge(train_time, on="model", how="left")
    for col in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                "f1_weighted", "roc_auc_ovr_macro", "predict_time_ms",
                "search_time_s", "cv_macro_f1_mean"]:
        table[col] = table[col].round(4)
    table = table.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    table.to_csv(RESULTS / "model_comparison.csv", index=False)

    (METRICS / "classification_reports.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8")
    for name, rep in reports.items():
        pd.DataFrame(rep).T.round(4).to_csv(
            METRICS / f"classification_report_{name}.csv")
    np.save(PREDICTIONS / "y_test.npy", np.asarray(y, dtype=str))
    for name, y_pred in preds.items():
        np.save(PREDICTIONS / f"y_pred_{name}.npy", np.asarray(y_pred, dtype=str))
    return table


def fig_model_comparison(table: pd.DataFrame) -> None:
    metrics = [("accuracy", "Accuracy"), ("f1_macro", "Macro F1"),
               ("f1_weighted", "Weighted F1")]
    labels = table["label"].tolist()
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    colours = ["#0072B2", "#D55E00", "#009E73"]
    for i, (col, name) in enumerate(metrics):
        bars = ax.bar(x + (i - 1) * width, table[col], width, label=name,
                      color=colours[i], edgecolor="black", linewidth=0.5)
        for bar, value in zip(bars, table[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=6.6,
                    rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    lo = max(0.0, float(table[metrics[0][0]].min()) - 0.06)
    ax.set_ylim(lo, 1.02)
    ax.set_ylabel("Score on the held-out test set")
    ax.set_title("Test-set performance of the six classifiers")
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_EVAL / "fig_6_1_model_comparison.png")
    plt.close(fig)


def fig_confusion_grid(split: Split) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.6))
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        y_pred = np.load(PREDICTIONS / f"y_pred_{name}.npy")
        cm = confusion_matrix(split.y_test, y_pred, labels=CLASS_ORDER)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax,
                    annot_kws={"size": 7.0}, linewidths=0.4, linecolor="white")
        ax.set_title(MODEL_LABELS[name], fontsize=9.5)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.tick_params(labelsize=6.6)
        ax.grid(False)
    fig.suptitle("Confusion matrices on the test set (counts)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_EVAL / "fig_6_2_confusion_matrices.png")
    plt.close(fig)


def fig_confusion_best(split: Split, best: str) -> None:
    y_pred = np.load(PREDICTIONS / f"y_pred_{best}.npy")
    cm = confusion_matrix(split.y_test, y_pred, labels=CLASS_ORDER)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    sns.heatmap(norm, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax,
                annot_kws={"size": 8}, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Row-normalised proportion"})
    ax.set_xlabel("Predicted variety")
    ax.set_ylabel("Actual variety")
    ax.set_title(f"Row-normalised confusion matrix - {MODEL_LABELS[best]}")
    ax.tick_params(labelsize=8)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_EVAL / "fig_6_3_confusion_matrix_best.png")
    plt.close(fig)


def fig_per_class(best: str) -> pd.DataFrame:
    reports = json.loads((METRICS / "classification_reports.json").read_text())
    rep = pd.DataFrame(reports[best]).T.loc[CLASS_ORDER]
    rep.index.name = "class"
    rep.round(4).to_csv(METRICS / "per_class_metrics_best.csv")

    x = np.arange(len(CLASS_ORDER))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for i, (col, label, colour) in enumerate([
            ("precision", "Precision", "#0072B2"),
            ("recall", "Recall", "#D55E00"),
            ("f1-score", "F1-score", "#009E73")]):
        bars = ax.bar(x + (i - 1) * width, rep[col], width, label=label,
                      color=colour, edgecolor="black", linewidth=0.4)
        for bar, value in zip(bars, rep[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005,
                    f"{value:.2f}", ha="center", va="bottom", fontsize=6.2,
                    rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, rotation=15)
    ax.set_ylim(max(0.0, rep[["precision", "recall", "f1-score"]].min().min() - 0.08), 1.03)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-class precision, recall and F1 - {MODEL_LABELS[best]}")
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_EVAL / "fig_6_4_per_class_metrics.png")
    plt.close(fig)
    return rep


def fig_roc(split: Split, best: str) -> float:
    model = joblib.load(MODELS / f"{best}.joblib")
    scores = _scores(model, split.X_test)
    # predict_proba columns follow model.classes_ (alphabetical); reorder them to
    # the display order so each column matches the binarised label column.
    fitted = list(model.classes_)
    scores = scores[:, [fitted.index(c) for c in CLASS_ORDER]]
    y_bin = label_binarize(split.y_test, classes=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    aucs = {}
    for i, cls in enumerate(CLASS_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        aucs[cls] = roc_auc_score(y_bin[:, i], scores[:, i])
        ax.plot(fpr, tpr, color=PALETTE[cls], lw=1.3,
                label=f"{cls} (AUC = {aucs[cls]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    macro = float(np.mean(list(aucs.values())))
    ax.set_title(f"One-vs-rest ROC curves - {MODEL_LABELS[best]} "
                 f"(macro AUC = {macro:.4f})")
    ax.legend(loc="lower right", fontsize=7.4)
    fig.tight_layout()
    fig.savefig(FIG_EVAL / "fig_6_5_roc_curves.png")
    plt.close(fig)
    pd.Series(aucs, name="roc_auc").round(4).to_csv(METRICS / "roc_auc_best.csv")
    return macro


def fig_cv_distribution() -> None:
    folds = pd.read_csv(METRICS / "cv_fold_scores.csv")
    order = [c for c in MODEL_ORDER if c in folds.columns]
    data = [folds[c].to_numpy() for c in order]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=1.2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.6))
    for patch, name in zip(bp["boxes"], order):
        patch.set_facecolor("#9ecae1")
        patch.set_edgecolor("black")
        patch.set_linewidth(0.7)
    ax.set_xticklabels([MODEL_LABELS[n] for n in order], rotation=18,
                       ha="right", fontsize=8.5)
    ax.set_ylabel("Fold macro F1-score")
    ax.set_title("Cross-validation stability of the tuned configurations "
                 f"({folds.shape[0]}-fold)")
    ax.set_ylim(min(min(d) for d in data) - 0.01, 1.005)
    fig.tight_layout()
    fig.savefig(FIG_EVAL / "fig_6_6_cv_fold_distribution.png")
    plt.close(fig)


def fig_svm_heatmap() -> None:
    cv = pd.read_csv(RESULTS / "cv_results.csv")
    svm = cv[cv["model"] == "svm_rbf"].copy()
    svm["C"] = svm["param_model__C"].astype(float)
    svm["gamma"] = svm["param_model__gamma"].astype(str)
    grid = svm.pivot_table(index="gamma", columns="C",
                           values="mean_test_score")
    order = ["scale", "0.001", "0.01", "0.1"]
    grid = grid.reindex([g for g in order if g in grid.index])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sns.heatmap(grid, annot=True, fmt=".4f", cmap="viridis", ax=ax,
                annot_kws={"size": 8}, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Mean CV macro F1"})
    ax.set_xlabel("Regularisation parameter C")
    ax.set_ylabel("Kernel width parameter gamma")
    ax.set_title("RBF-SVM grid search: CV macro F1 over the (C, gamma) grid")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_MODELS / "fig_5_2_svm_grid_search.png")
    plt.close(fig)


def fig_error_analysis(split: Split, best: str) -> pd.DataFrame:
    y_pred = np.load(PREDICTIONS / f"y_pred_{best}.npy")
    y = split.y_test
    cm = confusion_matrix(y, y_pred, labels=CLASS_ORDER)
    off = cm.copy()
    np.fill_diagonal(off, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5),
                             gridspec_kw={"width_ratios": [1.05, 1]})
    sns.heatmap(off, annot=True, fmt="d", cmap="Reds", cbar=False,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
                ax=axes[0], annot_kws={"size": 8}, linewidths=0.5,
                linecolor="white")
    axes[0].set_xlabel("Predicted variety")
    axes[0].set_ylabel("Actual variety")
    axes[0].set_title("Off-diagonal confusions (all 2,723 test beans)")
    axes[0].grid(False)

    err = 1 - np.diag(cm) / cm.sum(axis=1)
    bars = axes[1].bar(CLASS_ORDER, err, color=[PALETTE[c] for c in CLASS_ORDER],
                       edgecolor="black", linewidth=0.5)
    for bar, value in zip(bars, err):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.002,
                     f"{100 * value:.2f}%", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("Misclassification rate")
    axes[1].set_title("Per-class error rate")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_ylim(0, max(err.max() * 1.25, 0.02))
    fig.tight_layout()
    fig.savefig(FIG_EVAL / "fig_6_7_error_analysis.png")
    plt.close(fig)

    pairs = (pd.DataFrame(off, index=CLASS_ORDER, columns=CLASS_ORDER)
             .stack().sort_values(ascending=False))
    pairs = pairs[pairs > 0].rename("count").reset_index()
    pairs.columns = ["actual", "predicted", "count"]
    pairs.to_csv(METRICS / "confusion_pairs_best.csv", index=False)
    return pairs


def fig_misclassified_scatter(split: Split, best: str) -> None:
    y_pred = pd.Series(np.load(PREDICTIONS / f"y_pred_{best}.npy"),
                       index=split.X_test.index)
    wrong = y_pred != split.y_test
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    for ax, (fx, fy) in zip(axes, [("Area", "ShapeFactor1"),
                                   ("Compactness", "roundness")]):
        ax.scatter(split.X_test.loc[~wrong, fx], split.X_test.loc[~wrong, fy],
                   s=5, color="#B0B0B0", alpha=0.45, label="Correct",
                   edgecolor="none", rasterized=True)
        ax.scatter(split.X_test.loc[wrong, fx], split.X_test.loc[wrong, fy],
                   s=14, color="#D55E00", alpha=0.85,
                   label=f"Misclassified (n = {int(wrong.sum())})",
                   edgecolor="black", linewidth=0.25, rasterized=True)
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.legend(loc="best")
    axes[0].set_title("Misclassified beans in the size-shape plane")
    axes[1].set_title("Misclassified beans in the compactness-roundness plane")
    fig.suptitle(f"Where the {MODEL_LABELS[best]} errors sit in feature space",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_EVAL / "fig_6_8_misclassified_scatter.png")
    plt.close(fig)


def fig_cost_tradeoff(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for _, row in table.iterrows():
        ax.scatter(row["search_time_s"], row["f1_macro"], s=70,
                   color="#0072B2", edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(row["label"], (row["search_time_s"], row["f1_macro"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=7.6)
    ax.set_xscale("log")
    ax.set_xlabel("Hyperparameter search time (s, log scale)")
    ax.set_ylabel("Test macro F1")
    ax.set_title("Accuracy obtained per unit of tuning cost")
    fig.tight_layout()
    fig.savefig(FIG_EVAL / "fig_6_9_cost_tradeoff.png")
    plt.close(fig)


def sample_predictions(split: Split, best: str, per_class: int = 1) -> pd.DataFrame:
    """Representative test beans, at least one from every variety."""
    model = joblib.load(MODELS / f"{best}.joblib")
    proba = probabilities(model, split.X_test)
    classes = list(model.classes_)
    y_pred = pd.Series(np.load(PREDICTIONS / f"y_pred_{best}.npy"),
                       index=split.X_test.index)
    rows = []
    rng = np.random.default_rng(42)
    for cls in CLASS_ORDER:
        idx = split.y_test[split.y_test == cls].index
        pick = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for i in pick:
            p = proba[split.X_test.index.get_loc(i)]
            top = classes[int(np.argmax(p))]
            rows.append({
                **{f: round(float(split.X_test.loc[i, f]), 4) for f in FEATURES},
                "actual": split.y_test.loc[i],
                "predicted": y_pred.loc[i],
                "correct": bool(y_pred.loc[i] == split.y_test.loc[i]),
                "confidence": round(float(p.max()), 4),
                "runner_up": classes[int(np.argsort(p)[-2])],
                "runner_up_prob": round(float(np.sort(p)[-2]), 4),
                "top_class_from_proba": top,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(PREDICTIONS / "sample_predictions.csv", index=False)
    return frame


def run(split: Split) -> dict:
    meta = json.loads((RESULTS / "experiment_metadata.json").read_text())
    best = meta["best_model"]

    table = evaluate_models(split)
    fig_model_comparison(table)
    fig_confusion_grid(split)
    fig_confusion_best(split, best)
    per_class = fig_per_class(best)
    auc_macro = fig_roc(split, best)
    fig_cv_distribution()
    fig_svm_heatmap()
    pairs = fig_error_analysis(split, best)
    fig_misclassified_scatter(split, best)
    fig_cost_tradeoff(table)
    samples = sample_predictions(split, best)

    meta["test_metrics"] = table.to_dict(orient="records")
    meta["best_model_test"] = table.loc[table["model"] == best].iloc[0].to_dict()
    meta["roc_auc_ovr_macro_best"] = round(auc_macro, 4)
    meta["top_confusions"] = pairs.head(6).to_dict(orient="records")
    (RESULTS / "experiment_metadata.json").write_text(
        json.dumps(meta, indent=2, default=float), encoding="utf-8")

    print(table[["label", "accuracy", "f1_macro", "f1_weighted"]].to_string(index=False))
    return {"table": table, "per_class": per_class, "samples": samples}


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.preprocessing import split_data

    run(split_data(load_dataset()))
