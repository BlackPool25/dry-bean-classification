"""Generate every report table and number macro from the experiment outputs.

Both the LaTeX and the Word report are built from the artefacts written here, so
the two documents cannot disagree with each other or with results/.

Usage:  python scripts/build_report_assets.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MET = RES / "metrics"
FIG = ROOT / "figures"
REPORT = ROOT / "report"
TEX = REPORT / "latex"
TAB = TEX / "tables"
RFI = REPORT / "figures"
DOCX = REPORT / "docx"

for p in (TAB, RFI, DOCX):
    p.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA"]
PRETTY = {"SEKER": "Seker", "BARBUNYA": "Barbunya", "BOMBAY": "Bombay",
          "CALI": "Cali", "DERMASON": "Dermason", "HOROZ": "Horoz", "SIRA": "Sira"}

assets: dict = {"tables": {}, "figures": {}, "numbers": {}}


def read_csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(MET / name, **kw)


def tex_table(rows, header, align, caption, label, size="\\small",
              midrules=(), long=False):
    """Return a complete LaTeX table float."""
    lines = ["\\begin{table}[htbp]", "\\centering", size,
             f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    lines.append("\\begin{tabular}{" + align + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for i, row in enumerate(rows):
        if i in midrules:
            lines.append("\\midrule")
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    text = "\n".join(lines) + "\n"
    (TAB / f"{label.replace(':', '_')}.tex").write_text(text, encoding="utf-8")
    assets["tables"][label] = {
        "caption": caption, "header": list(header),
        "rows": [[str(c) for c in r] for r in rows],
    }
    return text


def fmt(value, digits=4):
    if isinstance(value, float) or isinstance(value, np.floating):
        return f"{value:.{digits}f}"
    return str(value)


def copy_figures():
    mapping = {
        "eda/fig_3_1_class_distribution.png": "fig_3_1_class_distribution.png",
        "eda/fig_3_2_missing_values.png": "fig_3_2_missing_values.png",
        "eda/fig_3_3_feature_histograms.png": "fig_3_3_feature_histograms.png",
        "eda/fig_3_4_boxplots_by_class.png": "fig_3_4_boxplots_by_class.png",
        "eda/fig_3_5_correlation_heatmap.png": "fig_3_5_correlation_heatmap.png",
        "eda/fig_3_6_pairplot.png": "fig_3_6_pairplot.png",
        "eda/fig_3_7_classwise_comparison.png": "fig_3_7_classwise_comparison.png",
        "eda/fig_3_8_scatter_relationships.png": "fig_3_8_scatter_relationships.png",
        "eda/fig_3_9_outlier_analysis.png": "fig_3_9_outlier_analysis.png",
        "eda/fig_3_10_discriminative_power.png": "fig_3_10_discriminative_power.png",
        "eda/fig_4_1_split.png": "fig_4_1_split.png",
        "models/fig_5_1_workflow.png": "fig_5_1_workflow.png",
        "models/fig_5_2_svm_grid_search.png": "fig_5_2_svm_grid_search.png",
        "models/fig_5_3_architecture.png": "fig_5_3_architecture.png",
        "evaluation/fig_6_1_model_comparison.png": "fig_6_1_model_comparison.png",
        "evaluation/fig_6_2_confusion_matrices.png": "fig_6_2_confusion_matrices.png",
        "evaluation/fig_6_3_confusion_matrix_best.png": "fig_6_3_confusion_matrix_best.png",
        "evaluation/fig_6_4_per_class_metrics.png": "fig_6_4_per_class_metrics.png",
        "evaluation/fig_6_5_roc_curves.png": "fig_6_5_roc_curves.png",
        "evaluation/fig_6_6_cv_fold_distribution.png": "fig_6_6_cv_fold_distribution.png",
        "evaluation/fig_6_7_error_analysis.png": "fig_6_7_error_analysis.png",
        "evaluation/fig_6_8_misclassified_scatter.png": "fig_6_8_misclassified_scatter.png",
        "evaluation/fig_6_9_cost_tradeoff.png": "fig_6_9_cost_tradeoff.png",
        "evaluation/fig_6_10_permutation_importance.png": "fig_6_10_permutation_importance.png",
        "evaluation/fig_6_11_tree_feature_importance.png": "fig_6_11_tree_feature_importance.png",
        "evaluation/fig_6_12_shap_summary.png": "fig_6_12_shap_summary.png",
    }
    copied = {}
    for src, dst in mapping.items():
        source = FIG / src
        if not source.exists():
            print(f"  missing figure: {src}")
            continue
        shutil.copy2(source, RFI / dst)
        copied[dst] = dst
    assets["figures"] = copied
    print(f"copied {len(copied)} figures -> report/figures/")


def dataset_tables():
    val = json.loads((MET / "data_validation.json").read_text())
    split = json.loads((MET / "split_summary.json").read_text())
    desc = read_csv("descriptive_statistics.csv", index_col=0)

    rows = [
        ["Dataset name", "Dry Bean Dataset (UCI ID 602)"],
        ["Provider", "UCI Machine Learning Repository"],
        ["Source URL", "\\url{https://archive.ics.uci.edu/dataset/602/dry+bean+dataset}"],
        ["DOI", "10.24432/C50S4B"],
        ["Date accessed", val["accessed"]],
        ["Licence", "CC BY 4.0"],
        ["File format", "Excel workbook (.xlsx), also ARFF and TXT"],
        ["Records", f"{val['rows']:,}"],
        ["Columns", f"{val['columns']} (16 features + 1 target)"],
        ["Target variable", "Class (7 nominal varieties)"],
        ["Task type", "Supervised multiclass classification"],
        ["Missing values", str(val["missing_values_total"])],
        ["Duplicate rows", str(val["duplicate_rows"])],
        ["Class imbalance ratio", f"{val['class_imbalance_ratio']}:1 (majority:minority)"],
    ]
    tex_table(rows, ["Property", "Details"], "p{0.30\\textwidth}p{0.60\\textwidth}",
              "Summary of the dry bean dataset as distributed by UCI",
              "tab:dataset_summary")

    counts = val["class_counts_ordered"]
    total = sum(counts.values())
    rows = [[i + 1, PRETTY[c], f"{counts[c]:,}", f"{100 * counts[c] / total:.2f}"]
            for i, c in enumerate(CLASS_ORDER)]
    rows.append(["", "Total", f"{total:,}", "100.00"])
    tex_table(rows, ["Sl. No.", "Variety", "Beans", "Share (\\%)"], "clrr",
              "Class distribution of the target variable",
              "tab:class_counts", midrules=(len(rows) - 1,))

    attrs = [
        ("Area", "Integer", "Pixels inside the bean boundary", "Input"),
        ("Perimeter", "Continuous", "Length of the bean boundary", "Input"),
        ("MajorAxisLength", "Continuous", "Longest line through the bean", "Input"),
        ("MinorAxisLength", "Continuous", "Shortest line through the bean", "Input"),
        ("AspectRation", "Continuous", "MajorAxisLength / MinorAxisLength", "Input"),
        ("Eccentricity", "Continuous", "Eccentricity of the equivalent ellipse", "Input"),
        ("ConvexArea", "Integer", "Pixels in the smallest enclosing convex polygon", "Input"),
        ("EquivDiameter", "Continuous", "Diameter of a circle of equal area", "Input"),
        ("Extent", "Continuous", "Bean area / bounding-box area", "Input"),
        ("Solidity", "Continuous", "Bean area / convex hull area", "Input"),
        ("roundness", "Continuous", "4 pi Area / Perimeter squared", "Input"),
        ("Compactness", "Continuous", "EquivDiameter / MajorAxisLength", "Input"),
        ("ShapeFactor1", "Continuous", "Area / MajorAxisLength cubed", "Input"),
        ("ShapeFactor2", "Continuous", "MinorAxisLength / MajorAxisLength", "Input"),
        ("ShapeFactor3", "Continuous", "Area / (Major x Minor axis length)", "Input"),
        ("ShapeFactor4", "Continuous", "Convex-hull based roundness ratio", "Input"),
        ("Class", "Nominal", "One of seven registered bean varieties", "Target"),
    ]
    rows = [[i + 1, a, b, c, d] for i, (a, b, c, d) in enumerate(attrs)]
    tex_table(rows, ["No.", "Attribute", "Type", "Description", "Role"],
              "clp{0.17\\textwidth}p{0.36\\textwidth}c",
              "Attributes of the dataset, in the order distributed by UCI",
              "tab:attributes", size="\\footnotesize")

    rows = []
    for f in desc.index:
        r = desc.loc[f]
        rows.append([f, f"{r['mean']:.4g}", f"{r['std']:.4g}",
                     f"{r['min']:.4g}", f"{r['50%']:.4g}", f"{r['max']:.4g}",
                     f"{r['skew']:.2f}"])
    tex_table(rows, ["Feature", "Mean", "Std. dev.", "Min", "Median", "Max", "Skew"],
              "lrrrrrr",
              "Descriptive statistics of the 16 morphological features (n = 13,611)",
              "tab:descriptive", size="\\footnotesize")

    rows = [[PRETTY[c],
             f"{val['class_counts_ordered'][c]:,}",
             f"{100 * val['class_counts_ordered'][c] / total:.1f}",
             f"{split['train_class_counts'][c]:,}",
             f"{split['test_class_counts'][c]:,}"] for c in CLASS_ORDER]
    tex_table(rows, ["Variety", "Full dataset", "Share (\\%)", "Training", "Test"],
              "lrrrr",
              "Stratified 80/20 partition: class counts are preserved in both subsets",
              "tab:split")
    assets["numbers"]["split"] = split
    return val, split


def correlation_table():
    corr = read_csv("feature_correlation.csv", index_col=0)
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    pairs = (upper.stack().sort_values(key=abs, ascending=False).head(10))
    rows = [[f"{a} -- {b}", f"{v:+.4f}"] for (a, b), v in pairs.items()]
    tex_table(rows, ["Feature pair", "Pearson r"], "lr",
              "The ten strongest feature correlations (all 21 pairs above 0.90 are "
              "listed in Appendix B)", "tab:corr_pairs")


def appendix_tables():
    folds = read_csv("cv_fold_scores.csv")
    labels = read_csv("cv_summary.csv").set_index("model")["label"]
    rows = []
    for col in folds.columns:
        vals = folds[col].to_numpy()
        rows.append([labels.get(col, col)] + [f"{v:.4f}" for v in vals] +
                    [f"{vals.mean():.4f}", f"{vals.std(ddof=1):.4f}"])
    tex_table(rows, ["Model", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5",
                     "Mean", "Std. dev."], "lrrrrrrr",
              "Macro F1 of the selected configuration in each cross-validation fold",
              "tab:cv_folds", size="\footnotesize")

    auc = read_csv("roc_auc_best.csv", index_col=0)
    rows = [[PRETTY[c], f"{auc.loc[c, 'roc_auc']:.4f}"] for c in CLASS_ORDER]
    tex_table(rows, ["Variety", "One-vs-rest ROC-AUC"], "lr",
              "Per-class ROC-AUC of the selected model on the test set",
              "tab:roc_class")

    corr = read_csv("feature_correlation.csv", index_col=0)
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    pairs = upper.stack().sort_values(key=abs, ascending=False)
    pairs = pairs[pairs.abs() >= 0.90]
    rows = [[i + 1, f"{a} -- {b}", f"{v:+.4f}"]
            for i, ((a, b), v) in enumerate(pairs.items())]
    tex_table(rows, ["No.", "Feature pair", "Pearson r"], "clr",
              f"All {len(rows)} feature pairs with $|r| \\ge 0.90$",
              "tab:corr_all", size="\footnotesize")


def model_tables():
    meta = json.loads((RES / "experiment_metadata.json").read_text())
    cv = read_csv("cv_summary.csv").sort_values("cv_macro_f1_mean", ascending=False)
    comp = read_csv("../results/model_comparison.csv") if False else pd.read_csv(
        RES / "model_comparison.csv")

    rows = []
    for _, r in cv.iterrows():
        rows.append([r["label"], r["search"], r["n_candidates"],
                     f"{r['cv_macro_f1_mean']:.4f}", f"{r['cv_macro_f1_std']:.4f}",
                     f"{r['cv_macro_f1_min']:.4f}", f"{r['cv_macro_f1_max']:.4f}",
                     f"{r['search_time_s']:.1f}"])
    tex_table(rows, ["Model", "Search", "Configs", "Mean", "Std. dev.", "Min", "Max",
                     "Time (s)"], "llcrrrrr",
              "Cross-validated macro F1 on the training partition (5 stratified folds). "
              "Sorted by mean macro F1", "tab:cv_results", size="\\footnotesize")

    best_params = {r["model"]: json.loads(r["best_params"]) for _, r in cv.iterrows()}
    rows = []
    for _, r in cv.iterrows():
        params = ", ".join(f"{k} = {v}" for k, v in best_params[r["model"]].items())
        params = params.replace("_", r"\_")
        rows.append([r["label"], params])
    tex_table(rows, ["Model", "Selected hyperparameters"],
              "p{0.26\\textwidth}p{0.64\\textwidth}",
              "Best configuration of every model, chosen by mean 5-fold CV macro F1",
              "tab:best_params", size="\\footnotesize")

    rows = []
    for _, r in comp.iterrows():
        rows.append([r["label"], f"{r['accuracy']:.4f}", f"{r['precision_macro']:.4f}",
                     f"{r['recall_macro']:.4f}", f"{r['f1_macro']:.4f}",
                     f"{r['f1_weighted']:.4f}", f"{r['roc_auc_ovr_macro']:.4f}"])
    tex_table(rows, ["Model", "Accuracy", "Macro prec.", "Macro rec.",
                     "Macro F1", "Weighted F1", "ROC-AUC"], "lrrrrrr",
              "Performance on the 2,723-bean held-out test set. Rows are ordered "
              "by test macro F1", "tab:model_comparison", size="\\scriptsize")

    best = meta["best_model"]
    y_true = np.load(RES / "predictions" / "y_test.npy")
    y_pred = np.load(RES / f"predictions/y_pred_{best}.npy")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    rows = []
    for i, c in enumerate(CLASS_ORDER):
        rows.append([PRETTY[c]] + [str(v) for v in cm[i]] + [str(cm[i].sum())])
    rows.append(["Total"] + [str(v) for v in cm.sum(axis=0)] + [str(cm.sum())])
    tex_table(rows, ["Actual \\textbackslash{} Predicted"] +
              [PRETTY[c] for c in CLASS_ORDER] + ["Recall base"], "l" + "r" * 8,
              f"Confusion matrix of the selected model ({meta['best_model_label']}) "
              "on the test set", "tab:confusion_best", size="\\scriptsize",
              midrules=(len(rows) - 1,))

    per_class = read_csv("per_class_metrics_best.csv", index_col=0)
    rows = []
    for c in CLASS_ORDER:
        r = per_class.loc[c]
        rows.append([PRETTY[c], f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                     f"{r['f1-score']:.4f}", f"{int(r['support'])}"])
    rows.append(["Macro average", f"{per_class.loc[CLASS_ORDER, 'precision'].mean():.4f}",
                 f"{per_class.loc[CLASS_ORDER, 'recall'].mean():.4f}",
                 f"{per_class.loc[CLASS_ORDER, 'f1-score'].mean():.4f}",
                 f"{int(per_class.loc[CLASS_ORDER, 'support'].sum())}"])
    tex_table(rows, ["Variety", "Precision", "Recall", "F1-score", "Support"],
              "lrrrr",
              f"Per-class performance of the selected model "
              f"({meta['best_model_label']})", "tab:per_class",
              midrules=(len(rows) - 1,))

    pairs = read_csv("confusion_pairs_best.csv")
    rows = [[PRETTY[r["actual"]], PRETTY[r["predicted"]], int(r["count"])]
            for _, r in pairs.head(6).iterrows()]
    tex_table(rows, ["Actual variety", "Predicted as", "Beans"],
              "lrr",
              "Most frequent confusions produced by the selected model",
              "tab:confusion_pairs")

    perm = read_csv("permutation_importance_best.csv")
    rows = [[i + 1, r["feature"], f"{r['importance_mean']:.4f}",
             f"{r['importance_std']:.4f}"] for i, (_, r) in enumerate(perm.iterrows())]
    tex_table(rows, ["Rank", "Feature", "Drop in macro F1", "Std. dev."], "clrr",
              "Permutation importance on the test set (8 repeats per feature)",
              "tab:permutation", size="\\footnotesize")

    abl = read_csv("feature_ablation.csv")
    rows = [[r["feature_set"].replace("_", " "), r["n_features"],
             f"{r['cv_macro_f1_mean']:.4f}", f"{r['cv_macro_f1_std']:.4f}"]
            for _, r in abl.iterrows()]
    tex_table(rows, ["Feature set", "Features", "CV macro F1", "Std. dev."], "lcrr",
              "Effect of dropping the five redundant size features (Random Forest, "
              "identical hyperparameters)", "tab:ablation")

    cw = read_csv("class_weight_experiment.csv")
    rows = [[r["model"], r["class_weight"], f"{r['cv_macro_f1_mean']:.4f}",
             f"{r['cv_macro_f1_std']:.4f}"] for _, r in cw.iterrows()]
    tex_table(rows, ["Model", "class\\_weight", "CV macro F1", "Std. dev."], "llrr",
              "Effect of balanced class weights on the selected model",
              "tab:class_weight")

    samples = read_csv("../results/predictions/sample_predictions.csv") if False else \
        pd.read_csv(RES / "predictions" / "sample_predictions.csv")
    keep = ["Area", "Perimeter", "Compactness", "roundness", "ShapeFactor1", "Extent"]
    def cell(feature, value):
        if feature in ("Area", "Perimeter"):
            return f"{value:,.0f}"
        return f"{value:.4f}"

    rows = []
    for i, (_, r) in enumerate(samples.iterrows(), start=1):
        rows.append([i] + [cell(f, r[f]) for f in keep] +
                    [PRETTY[r["actual"]], PRETTY[r["predicted"]],
                     "Yes" if r["correct"] else "No", f"{r['confidence']:.3f}"])
    tex_table(rows, ["\\#"] + keep + ["Actual", "Predicted", "Correct", "P(max)"],
              "clrrrrrrccc",
              "Representative test beans, one per variety, classified by the saved "
              "pipeline", "tab:sample_predictions", size="\\scriptsize")

    assets["numbers"].update({
        "best_model": meta["best_model_label"],
        "best_model_key": best,
        "cv_mean": f"{meta['cv_macro_f1_mean']:.4f}",
        "cv_std": f"{meta['cv_macro_f1_std']:.4f}",
        "test_accuracy": f"{meta['best_model_test']['accuracy']:.4f}",
        "test_f1": f"{meta['best_model_test']['f1_macro']:.4f}",
        "test_precision": f"{meta['best_model_test']['precision_macro']:.4f}",
        "test_recall": f"{meta['best_model_test']['recall_macro']:.4f}",
        "test_f1_weighted": f"{meta['best_model_test']['f1_weighted']:.4f}",
        "test_auc": f"{meta['roc_auc_ovr_macro_best']:.4f}",
        "predict_ms": f"{meta['best_model_test']['predict_time_ms']:.1f}",
        "search_s": f"{meta['best_model_test']['search_time_s']:.1f}",
        "runtime": f"{meta.get('pipeline_runtime_s', 0) / 60:.1f}",
        "n_errors": int((y_true != y_pred).sum()),
        "n_test": int(len(y_true)),
        "best_params": best_params[best],
        "second_model": comp.iloc[1]["label"],
        "second_f1": f"{comp.iloc[1]['f1_macro']:.4f}",
        "worst_model": comp.iloc[-1]["label"],
        "worst_f1": f"{comp.iloc[-1]['f1_macro']:.4f}",
        "shap_top": json.loads((MET / "shap_importance.csv").read_text()
                               if False else "{}") if False else None,
        "perm_top": perm.iloc[0]["feature"],
        "perm_top_value": f"{perm.iloc[0]['importance_mean']:.4f}",
        "ablation_16": f"{abl.iloc[0]['cv_macro_f1_mean']:.4f}",
        "ablation_11": f"{abl.iloc[1]['cv_macro_f1_mean']:.4f}",
        "cw_none": f"{cw.iloc[0]['cv_macro_f1_mean']:.4f}",
        "cw_balanced": f"{cw.iloc[1]['cv_macro_f1_mean']:.4f}",
        "top_pair": f"{PRETTY[pairs.iloc[0]['actual']]} -> "
                    f"{PRETTY[pairs.iloc[0]['predicted']]}",
        "top_pair_n": int(pairs.iloc[0]["count"]),
        "auc_by_class": read_csv("roc_auc_best.csv", index_col=0)["roc_auc"].round(4).to_dict(),
    })
    return meta, comp, cv


def numbers_tex():
    n = assets["numbers"]
    val = json.loads((MET / "data_validation.json").read_text())
    lines = ["% Auto-generated by scripts/build_report_assets.py - do not edit.",
             f"\\newcommand{{\\DatasetRows}}{{{val['rows']:,}}}",
             f"\\newcommand{{\\DatasetFeatures}}{{{val['feature_count']}}}",
             f"\\newcommand{{\\TrainRows}}{{{n['split']['train_rows']:,}}}",
             f"\\newcommand{{\\TestRows}}{{{n['split']['test_rows']:,}}}",
             f"\\newcommand{{\\TestPct}}{{{n['split']['test_pct']}}}",
             f"\\newcommand{{\\DuplicateRows}}{{{val['duplicate_rows']}}}",
             f"\\newcommand{{\\ImbalanceRatio}}{{{val['class_imbalance_ratio']}}}",
             f"\\newcommand{{\\BestModel}}{{{n['best_model']}}}",
             f"\\newcommand{{\\BestCV}}{{{n['cv_mean']}}}",
             f"\\newcommand{{\\BestCVStd}}{{{n['cv_std']}}}",
             f"\\newcommand{{\\BestAcc}}{{{n['test_accuracy']}}}",
             f"\\newcommand{{\\BestFOne}}{{{n['test_f1']}}}",
             f"\\newcommand{{\\BestPrec}}{{{n['test_precision']}}}",
             f"\\newcommand{{\\BestRec}}{{{n['test_recall']}}}",
             f"\\newcommand{{\\BestFWeighted}}{{{n['test_f1_weighted']}}}",
             f"\\newcommand{{\\BestAUC}}{{{n['test_auc']}}}",
             f"\\newcommand{{\\BestPredictMs}}{{{n['predict_ms']}}}",
             f"\\newcommand{{\\BestSearchS}}{{{n['search_s']}}}",
             f"\\newcommand{{\\PipelineMinutes}}{{{n['runtime']}}}",
             f"\\newcommand{{\\TestErrors}}{{{n['n_errors']}}}",
             f"\\newcommand{{\\TestErrorPct}}{{{100 * n['n_errors'] / n['n_test']:.2f}}}",
             f"\\newcommand{{\\SecondModel}}{{{n['second_model']}}}",
             f"\\newcommand{{\\SecondFOne}}{{{n['second_f1']}}}",
             f"\\newcommand{{\\WorstModel}}{{{n['worst_model']}}}",
             f"\\newcommand{{\\WorstFOne}}{{{n['worst_f1']}}}",
             f"\\newcommand{{\\TopPermFeature}}{{{n['perm_top']}}}",
             f"\\newcommand{{\\TopPermDrop}}{{{n['perm_top_value']}}}",
             f"\\newcommand{{\\AblationFull}}{{{n['ablation_16']}}}",
             f"\\newcommand{{\\AblationReduced}}{{{n['ablation_11']}}}",
             f"\\newcommand{{\\CWNone}}{{{n['cw_none']}}}",
             f"\\newcommand{{\\CWBalanced}}{{{n['cw_balanced']}}}",
             f"\\newcommand{{\\TopConfusion}}{{{n['top_pair']}}}",
             f"\\newcommand{{\\TopConfusionN}}{{{n['top_pair_n']}}}",
             ]
    (TEX / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    copy_figures()
    dataset_tables()
    correlation_table()
    appendix_tables()
    meta, comp, cv = model_tables()
    numbers_tex()
    (DOCX / "assets.json").write_text(
        json.dumps(assets, indent=2, default=str), encoding="utf-8")
    print(f"tables: {len(assets['tables'])}, numbers: {len(assets['numbers'])}")
    print("wrote report/latex/numbers.tex and report/docx/assets.json")


if __name__ == "__main__":
    main()
