"""Render the Chapter 5 workflow diagram as a publication-quality figure.

Mermaid is used for the README (GitHub renders it), but its automatic layout
produces a very narrow column for a twelve-step pipeline, so the report figure is
drawn explicitly to control the aspect ratio. The flow snakes through three
columns so that connectors never cross a box.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "models" / "fig_5_1_workflow.png"

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Liberation Serif", "DejaVu Serif"]})

COLUMNS = [
    dict(title="1. Data preparation", face="#EAF2FA", edge="#4C72B0", flow="down",
         boxes=[("UCI Dry Bean dataset\n13,611 beans x 17 columns", False),
                ("Validation: missing values,\nduplicates, ranges, labels", False),
                ("Stratified 80/20 split\nrandom_state = 42", False),
                ("Exploratory data analysis\ndistributions, correlation, outliers", False)]),
    dict(title="2. Model development", face="#F6F1E7", edge="#D55E00", flow="up",
         boxes=[("Leakage-safe pipelines\nscaler fitted on training folds", False),
                ("5-fold stratified CV\nscoring = macro F1", False),
                ("Grid search: LR, KNN, SVM\nRandom search: RF, ET, HistGB", False),
                ("Selection by mean\nCV macro F1", True)]),
    dict(title="3. Assessment and delivery", face="#EAF5F0", edge="#009E73",
         flow="down",
         boxes=[("Held-out test: accuracy,\nprecision, recall, F1, ROC-AUC", False),
                ("Error analysis and\ninterpretation (SHAP)", False),
                ("best_model.joblib\nprediction CLI, LaTeX + DOCX report", False)]),
]

W, H = 0.250, 0.135
XS = [0.135, 0.500, 0.865]
TOP, BOTTOM, GAP = 0.845, 0.205, 0.075


def draw_box(ax, x, y, text, face, edge, diamond=False):
    if diamond:
        pts = [(x - W / 2, y), (x, y + H / 2), (x + W / 2, y), (x, y - H / 2)]
        ax.add_patch(plt.Polygon(pts, closed=True, facecolor=face,
                                 edgecolor=edge, linewidth=1.2, zorder=2))
    else:
        ax.add_patch(FancyBboxPatch(
            (x - W / 2, y - H / 2), W, H,
            boxstyle="round,pad=0.004,rounding_size=0.014",
            facecolor=face, edgecolor=edge, linewidth=1.1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.6, zorder=4,
            linespacing=1.4)


def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.05, color="#333333",
                                 shrinkA=1, shrinkB=1, zorder=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.4, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pos = {}
    for ci, col in enumerate(COLUMNS):
        x = XS[ci]
        ax.text(x, 0.955, col["title"], ha="center", va="center", fontsize=10,
                fontweight="bold", color=col["edge"])
        for bi, (text, diamond) in enumerate(col["boxes"]):
            step = bi * (H + GAP)
            y = TOP - step if col["flow"] == "down" else BOTTOM + step
            draw_box(ax, x, y, text, col["face"], col["edge"], diamond)
            pos[(ci, bi)] = (x, y)

    for ci, col in enumerate(COLUMNS):
        n = len(col["boxes"])
        for bi in range(n - 1):
            x, y0 = pos[(ci, bi)]
            _, y1 = pos[(ci, bi + 1)]
            lo, hi = sorted((y0, y1))
            arrow(ax, (x, lo + H / 2 + 0.008), (x, hi - H / 2 - 0.008))

    for ci in (0, 1):
        x0, y0 = pos[(ci, len(COLUMNS[ci]["boxes"]) - 1)]
        x1, y1 = pos[(ci + 1, 0)]
        arrow(ax, (x0 + W / 2 + 0.008, y0), (x1 - W / 2 - 0.008, y1))

    ax.text(0.5, 0.075,
            "The test partition is opened only at the assessment stage; all tuning "
            "happens inside cross-validation on the training partition.",
            ha="center", va="center", fontsize=8.6, style="italic", color="#444444")

    fig.subplots_adjust(0, 0, 1, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
