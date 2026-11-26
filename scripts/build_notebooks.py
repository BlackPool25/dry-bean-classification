"""Build the four project notebooks from source so they stay in sync with src/.

Usage:  python scripts/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

BOOTSTRAP = '''\
import os, sys, pathlib
root = pathlib.Path.cwd()
while not (root / "src" / "config.py").exists():
    root = root.parent
sys.path.insert(0, str(root))
os.chdir(root)          # relative paths in the notebooks resolve from the root
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
pd.set_option("display.width", 160)
print("project root:", root)'''


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def nb(cells) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return notebook


def exploration() -> nbf.NotebookNode:
    return nb([
        md("""# 01 - Data Acquisition and Exploratory Analysis

**Dry Bean Variety Classification** - UCI dataset 602.

This notebook downloads (if needed) and validates the official workbook, then
reproduces the figures used in Chapter 3. All plotting code lives in
`src/eda.py`; the notebook calls it so the report and the notebook cannot drift
apart."""),
        code(BOOTSTRAP),
        md("## 1. Load and validate"),
        code('''from src.data_loader import load_dataset, validate
df = load_dataset()
print(df.shape)
df.head()'''),
        code('''report = validate(df)
{k: report[k] for k in ["rows", "columns", "feature_count", "missing_values_total",
                        "duplicate_rows", "class_imbalance_ratio"]}'''),
        code('''report["class_counts_ordered"]'''),
        code('''report["feature_pairs_abs_corr_ge_0.90"]'''),
        md("""68 rows repeat an earlier row on all 17 columns. Because every feature
is derived from integer pixel measurements, distinct beans can share identical
values; the duplicates are therefore retained and the decision is discussed in
Section 4.1."""),
        md("## 2. Descriptive statistics"),
        code('''from src.config import FEATURES
df[FEATURES].describe().T.round(3)'''),
        md("## 3. Class distribution"),
        code('''import matplotlib.pyplot as plt
from src.config import CLASS_ORDER, PALETTE
counts = df["Class"].value_counts().reindex(CLASS_ORDER)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index],
       edgecolor="black", linewidth=0.6)
for i, v in enumerate(counts.values):
    ax.text(i, v + 40, f"{v}\\n({100*v/counts.sum():.1f}%)", ha="center", fontsize=8)
ax.set_ylabel("Number of beans"); ax.set_xlabel("Variety")
ax.set_title("Class distribution"); ax.set_ylim(0, counts.max()*1.2)
plt.show()'''),
        md("## 4. Feature distributions"),
        code('''from src.eda import feature_histograms, boxplots_by_class
feature_histograms(df); boxplots_by_class(df)'''),
        code('''from IPython.display import Image, display
display(Image("figures/eda/fig_3_3_feature_histograms.png", width=980))'''),
        code('''display(Image("figures/eda/fig_3_4_boxplots_by_class.png", width=980))'''),
        md("""Area, perimeter, convex area and equivalent diameter are strongly
right-skewed with heavy tails, while `Solidity`, `Extent` and `ShapeFactor4` are
bounded and near-symmetric. BOMBAY sits far above the other six varieties on
every size feature, which suggests that a size axis alone will separate one
class almost perfectly while the remaining six need shape information."""),
        md("## 5. Correlation structure"),
        code('''from src.eda import correlation_heatmap
corr = correlation_heatmap(df)
display(Image("figures/eda/fig_3_5_correlation_heatmap.png", width=720))'''),
        code('''pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
             .stack().sort_values(key=abs, ascending=False).head(12).round(3))
pairs'''),
        md("""Twenty-one pairs exceed |r| = 0.90. `Area` and `ConvexArea` are almost
the same variable (r = 0.9999) because beans are nearly convex. This does not
justify deleting features: tree ensembles can exploit the small residual
difference, and Section 6.2 reports an ablation that tests the cost of dropping
the redundant size cluster."""),
        md("## 6. Class-wise separation"),
        code('''from src.eda import classwise_comparison, pairplot, scatter_relationships
classwise_comparison(df); scatter_relationships(df); pairplot(df)
display(Image("figures/eda/fig_3_7_classwise_comparison.png", width=1000))
display(Image("figures/eda/fig_3_8_scatter_relationships.png", width=1000))'''),
        code('''display(Image("figures/eda/fig_3_6_pairplot.png", width=760))'''),
        md("""The pair plot shows two clean clusters (BOMBAY and DERMASON) and one
dense, overlapping cloud containing SEKER, BARBUNYA, CALI, HOROZ and SIRA. The
expected error rate is therefore concentrated inside that cloud, which is what
the confusion matrices in Chapter 6 confirm."""),
        md("## 7. Outliers and discriminative power"),
        code('''from src.eda import outlier_analysis, discriminative_power
outlier_analysis(df)
anova = discriminative_power(df)
display(Image("figures/eda/fig_3_9_outlier_analysis.png", width=1000))
anova.head(10).round(2)'''),
        md("""Tukey's rule flags up to 843 observations per feature, but almost all of
them are legitimate large beans from BOMBAY rather than measurement errors. No
observation was removed. `ShapeFactor1` and `Compactness` carry the largest
between-class F-statistics after the size features."""),
        md("## 8. Observations carried into modelling"),
        md("""1. The classes are imbalanced by a factor of 6.8 (DERMASON 3,546 vs
BOMBAY 522), so macro-averaged F1 is the selection metric rather than accuracy.
2. Size features separate BOMBAY; shape features must separate the remaining six.
3. Feature magnitudes differ by four orders of magnitude, so distance- and
margin-based learners need standardisation.
4. Heavy-tailed features suggest that tree ensembles will be competitive
because they do not assume normality.
5. No missing values and no non-finite values, so imputation is unnecessary."""),
    ])


def preprocessing() -> nbf.NotebookNode:
    return nb([
        md("""# 02 - Preprocessing, Splitting and Leakage Control

The split is created once here and written to `data/processed/`. Every later
step reads the same files, so the test partition cannot change between
experiments."""),
        code(BOOTSTRAP),
        md("## 1. Stratified split"),
        code('''from src.data_loader import load_dataset
from src.preprocessing import split_data, persist_split
split = split_data(load_dataset())
persist_split(split)
split.summary()'''),
        md("""Stratification keeps the class proportions of the full dataset inside
both partitions; without it the smallest class (BOMBAY, 522 beans) could lose
several percent of its already small test representation."""),
        md("## 2. Why scaling must sit inside the pipeline"),
        code('''from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(split.X_train)          # fitted on TRAIN only
print("train mean Area :", round(scaler.mean_[0], 2))
print("test  mean Area :", round(split.X_test["Area"].mean(), 2))'''),
        code('''from src.config import FEATURES
raw = split.X_train[FEATURES].agg(["min", "max"]).T
raw["range"] = raw["max"] - raw["min"]
raw.round(3).sort_values("range", ascending=False).head(6)'''),
        md("""Area spans 20,420-254,616 pixels while `ShapeFactor1` stays below 0.011.
A distance-based model fitted on raw values would be driven almost entirely by
Area. Fitting the scaler on the whole dataset before splitting would leak test
statistics into training, so `src.preprocessing.build_pipeline` places
`StandardScaler` as the first pipeline step and only ever fits it on training
folds."""),
        md("## 3. Pipeline construction"),
        code('''from src.preprocessing import build_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
print(build_pipeline(LogisticRegression(), scale=True))
print(build_pipeline(RandomForestClassifier(n_estimators=10), scale=False))'''),
        md("""Tree ensembles receive the raw features: a split on `Area < 60,000` is
unchanged by a monotone rescaling, so standardisation would only add cost."""),
        md("## 4. Duplicate and outlier policy"),
        code('''import pandas as pd
train = pd.read_csv("data/processed/train.csv")
print("duplicate rows kept:", int(train.duplicated().sum()))
print(train.groupby("Class")["Area"].agg(["count", "min", "max"]).round(1))'''),
        md("""Duplicates were retained (pixel quantisation, not data entry errors) and
no outliers were removed, because the extreme values are genuine BOMBAY beans.
Both decisions are tested indirectly by the feature ablation in notebook 03."""),
    ])


def training() -> nbf.NotebookNode:
    return nb([
        md("""# 03 - Model Training and Hyperparameter Search

The full search (six models, 199 candidate configurations, 5-fold CV) is
executed by `python -m src.train` and takes several minutes. This notebook
demonstrates the identical code path on a reduced search so it runs quickly, then
reports the full saved results."""),
        code(BOOTSTRAP),
        md("## 1. A live, reduced search"),
        code('''from src.preprocessing import load_split
from src.preprocessing import build_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time
split = load_split()
pipe = build_pipeline(LogisticRegression(max_iter=3000, random_state=42), scale=True)
grid = GridSearchCV(pipe, {"model__C": [0.1, 1.0, 10.0]}, scoring="f1_macro",
                    cv=StratifiedKFold(2, shuffle=True, random_state=42), n_jobs=-1)
t0 = time.perf_counter(); grid.fit(split.X_train, split.y_train)
print(f"best C = {grid.best_params_['model__C']}, "
      f"CV macro F1 = {grid.best_score_:.4f} ({time.perf_counter()-t0:.1f}s)")'''),
        md("## 2. Full cross-validation results"),
        code('''cv = pd.read_csv("results/metrics/cv_summary.csv")
cv[["label", "search", "n_candidates", "cv_macro_f1_mean", "cv_macro_f1_std",
    "search_time_s"]].sort_values("cv_macro_f1_mean", ascending=False)'''),
        code('''import json
for _, row in cv.iterrows():
    print(f"{row['label']:<26} {row['best_params']}")'''),
        md("## 3. RBF-SVM (C, gamma) surface"),
        code('''from src.evaluate import fig_svm_heatmap
fig_svm_heatmap()
from IPython.display import Image, display
display(Image("figures/models/fig_5_2_svm_grid_search.png", width=560))'''),
        md("""The surface is flat between C = 1 and C = 100: once the margin penalty is
large enough, additional capacity buys nothing because the classes are already
almost separable in the induced feature space. Very small C (0.1) underfits and
large gamma (0.1) overfits, which is the expected bias-variance behaviour."""),
        md("## 4. Model selection"),
        code('''meta = json.loads(open("results/experiment_metadata.json").read())
print("selected:", meta["best_model_label"])
print("CV macro F1:", meta["cv_macro_f1_mean"], "+/-", meta["cv_macro_f1_std"])
print("best params:", meta["best_params"])'''),
        md("## 5. Feature ablation and class weighting"),
        code('''pd.read_csv("results/metrics/feature_ablation.csv")'''),
        code('''pd.read_csv("results/metrics/class_weight_experiment.csv")'''),
        md("""Removing the five redundant size features costs a small but measurable
amount of macro F1, and balancing the class weights does not improve macro F1 for
the selected model. The original class distribution was therefore kept."""),
    ])


def evaluation() -> nbf.NotebookNode:
    return nb([
        md("""# 04 - Held-out Evaluation and Demonstration

Everything below is computed from the saved test predictions; no model is
re-fitted here, so the numbers match the report exactly."""),
        code(BOOTSTRAP),
        md("## 1. Model comparison on the test set"),
        code('''table = pd.read_csv("results/model_comparison.csv")
table[["label", "accuracy", "precision_macro", "recall_macro", "f1_macro",
       "f1_weighted", "roc_auc_ovr_macro", "predict_time_ms"]]'''),
        code('''from IPython.display import Image, display
display(Image("figures/evaluation/fig_6_1_model_comparison.png", width=820))'''),
        md("## 2. Best model: confusion matrix and per-class scores"),
        code('''import json, numpy as np
meta = json.loads(open("results/experiment_metadata.json").read())
best = meta["best_model"]; print(best, "-", meta["best_model_label"])'''),
        code('''from src.preprocessing import load_split
from sklearn.metrics import classification_report
split = load_split()
y_pred = np.load(f"results/predictions/y_pred_{best}.npy")
print(classification_report(split.y_test, y_pred, digits=4))'''),
        code('''display(Image("figures/evaluation/fig_6_3_confusion_matrix_best.png", width=560))
display(Image("figures/evaluation/fig_6_2_confusion_matrices.png", width=1000))'''),
        md("""Nearly all remaining errors are DERMASON beans predicted as SIRA. The
two varieties share an elongated, low-compactness shape, so their feature ranges
overlap even though their mean areas differ."""),
        md("## 3. ROC curves and error analysis"),
        code('''display(Image("figures/evaluation/fig_6_5_roc_curves.png", width=520))
display(Image("figures/evaluation/fig_6_7_error_analysis.png", width=1000))'''),
        code('''pd.read_csv("results/metrics/confusion_pairs_best.csv").head(8)'''),
        md("## 4. Interpretation"),
        code('''display(Image("figures/evaluation/fig_6_10_permutation_importance.png", width=620))
display(Image("figures/evaluation/fig_6_11_tree_feature_importance.png", width=1000))'''),
        md("""`ShapeFactor1` and `Compactness` dominate every importance measure.
Because they are ratios of the size features, they summarise the elongation of a
bean in a scale-free way, which is exactly what separates the five overlapping
varieties."""),
        md("## 5. Prediction demonstration"),
        code('''from src.predict import predict_records, load_model
model = load_model()
samples = pd.read_csv("results/predictions/sample_predictions.csv")
samples[["actual", "predicted", "correct", "confidence", "runner_up"]]'''),
        code('''from src.config import FEATURES
record = samples.iloc[0]
features = {c: float(record[c]) for c in FEATURES}
for r in predict_records([features], model):
    print("prediction:", r["predicted_variety"], "confidence", r["confidence"])
    for t in r["top3"]:
        print(f"   {t['variety']:<10} {t['probability']:.4f}")'''),
        code('''!python -m src.predict --csv results/predictions/sample_predictions.csv'''),
        md("""The command-line interface accepts either a CSV of measurements or all
16 values as flags, and prints the predicted variety with its top-3
probabilities."""),
    ])


def main() -> None:
    NB_DIR.mkdir(exist_ok=True)
    for name, builder in (("01_data_exploration.ipynb", exploration),
                          ("02_preprocessing.ipynb", preprocessing),
                          ("03_model_training.ipynb", training),
                          ("04_evaluation.ipynb", evaluation)):
        nbf.write(builder(), NB_DIR / name)
        print("wrote", NB_DIR / name)


if __name__ == "__main__":
    main()
