"""Exploratory data analysis: descriptive statistics and report figures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config import (
    CLASS_ORDER, FEATURES, FIG_EDA, METRICS, PALETTE, TARGET, setup_plot_style,
)

setup_plot_style()

PAIR_FEATURES = ["Area", "Perimeter", "Compactness", "ShapeFactor1"]
BOX_FEATURES = ["Area", "Perimeter", "MajorAxisLength", "Eccentricity",
                "Compactness", "roundness", "ShapeFactor1", "Solidity"]
VIOLIN_FEATURES = ["Area", "Compactness", "roundness", "ShapeFactor1"]


def _save(fig, name: str) -> None:
    path = FIG_EDA / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(FIG_EDA.parents[1])}")


def _k_formatter(ax, axis: str = "x") -> None:
    """Shorten pixel-scale ticks (50,000 -> 50k) so labels never collide."""
    from matplotlib.ticker import FuncFormatter

    ticker = ax.xaxis if axis == "x" else ax.yaxis
    if max(abs(v) for v in ticker.get_ticklocs()) >= 10000:
        ticker.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))


def class_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[c] for c in counts.index],
                  edgecolor="black", linewidth=0.6)
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 45,
                f"{value}\n({100 * value / total:.1f}%)",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Number of beans")
    ax.set_xlabel("Bean variety")
    ax.set_title("Class distribution of the seven dry bean varieties")
    ax.set_ylim(0, counts.max() * 1.22)
    _save(fig, "fig_3_1_class_distribution.png")


def missing_values(df: pd.DataFrame) -> None:
    """Missing-value map. The dataset is complete, so the figure is a single
    flat line; it is included because the checklist requires the check."""
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    completeness = 100 * (1 - df[FEATURES].isna().mean())
    ax.barh(range(len(FEATURES)), completeness, color="#0072B2", height=0.75)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES, fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Non-missing values (%)")
    ax.set_title("Missing-value analysis: all 16 features are 100% complete")
    for i, value in enumerate(completeness):
        ax.text(value + 1.0, i, f"{value:.0f}%", va="center", fontsize=7.5)
    ax.invert_yaxis()
    _save(fig, "fig_3_2_missing_values.png")


def feature_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(11.5, 8.6))
    for ax, col in zip(axes.ravel(), FEATURES):
        ax.hist(df[col], bins=45, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.set_title(col, fontsize=9.5)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7.5)
        _k_formatter(ax)
        skew = stats.skew(df[col])
        ax.text(0.97, 0.93, f"skew={skew:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5))
    fig.suptitle("Distribution of the 16 morphological features (n = 13,611)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fig_3_3_feature_histograms.png")


def boxplots_by_class(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4))
    for ax, col in zip(axes.ravel(), BOX_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, order=CLASS_ORDER, ax=ax,
                    hue=TARGET, palette=PALETTE, legend=False,
                    linewidth=0.7, fliersize=1.4,
                    flierprops=dict(marker="o", alpha=0.35, markersize=2))
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=90, labelsize=7.5)
        _k_formatter(ax, "y")
    fig.suptitle("Feature spread by bean variety (boxes show the IQR, whiskers 1.5x IQR)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig_3_4_boxplots_by_class.png")


def correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, square=True, linewidths=0.4,
                annot_kws={"size": 6.6}, cbar_kws={"shrink": 0.75,
                                                   "label": "Pearson r"}, ax=ax)
    ax.set_title("Pearson correlation between morphological features")
    ax.tick_params(labelsize=7.6)
    _save(fig, "fig_3_5_correlation_heatmap.png")
    corr.to_csv(METRICS / "feature_correlation.csv")
    return corr


def pairplot(df: pd.DataFrame) -> None:
    sample = df.sample(n=min(4000, len(df)), random_state=42)
    grid = sns.pairplot(sample, vars=PAIR_FEATURES, hue=TARGET,
                        hue_order=CLASS_ORDER, palette=PALETTE,
                        diag_kind="kde", corner=False,
                        plot_kws={"s": 7, "alpha": 0.45, "edgecolor": "none",
                                  "rasterized": True},
                        diag_kws={"linewidth": 1.0, "fill": False})
    grid.figure.set_size_inches(8.6, 8.0)
    grid.figure.suptitle("Pairwise relationships of four informative features "
                         "(4,000-row sample)", y=1.005, fontsize=11.5,
                         fontweight="bold")
    grid._legend.set_title("Variety")
    grid.savefig(FIG_EDA / "fig_3_6_pairplot.png", dpi=300)
    plt.close(grid.figure)
    print("  saved figures/eda/fig_3_6_pairplot.png")


def classwise_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.9))
    for ax, col in zip(axes, VIOLIN_FEATURES):
        sns.violinplot(data=df, x=TARGET, y=col, order=CLASS_ORDER, ax=ax,
                       hue=TARGET, palette=PALETTE, legend=False,
                       inner="quartile", linewidth=0.6, cut=0,
                       density_norm="width")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=90, labelsize=7.2)
        _k_formatter(ax, "y")
    fig.suptitle("Class-wise density of four features that separate the varieties",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig_3_7_classwise_comparison.png")


def scatter_relationships(df: pd.DataFrame) -> None:
    sample = df.sample(n=min(5000, len(df)), random_state=7)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    panels = [("Area", "Perimeter", "Area (px) vs perimeter (px)"),
              ("Compactness", "roundness", "Compactness vs roundness")]
    for ax, (x, y, title) in zip(axes, panels):
        for cls in CLASS_ORDER:
            sub = sample[sample[TARGET] == cls]
            ax.scatter(sub[x], sub[y], s=7, alpha=0.5, color=PALETTE[cls],
                       label=cls, edgecolor="none", rasterized=True)
        ax.set_xlabel(f"{x}")
        ax.set_ylabel(f"{y}")
        ax.set_title(title, fontsize=10)
        _k_formatter(ax)
    axes[0].legend(title="Variety", markerscale=2.6, loc="upper left")
    fig.suptitle("Two feature pairs that shape the class boundaries",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig_3_8_scatter_relationships.png")


def outlier_analysis(df: pd.DataFrame) -> None:
    z = (df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std()
    counts = []
    for col in FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        counts.append(int(((df[col] < q1 - 1.5 * iqr) |
                           (df[col] > q3 + 1.5 * iqr)).sum()))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].boxplot([z[c].dropna() for c in FEATURES], vert=True,
                    tick_labels=FEATURES, showfliers=True,
                    flierprops=dict(marker=".", markersize=1.6, alpha=0.25,
                                    markeredgecolor="none"),
                    medianprops=dict(color="#D55E00", linewidth=1.0),
                    boxprops=dict(linewidth=0.6),
                    whiskerprops=dict(linewidth=0.6),
                    capprops=dict(linewidth=0.6))
    axes[0].axhline(3, color="#D55E00", ls="--", lw=0.9)
    axes[0].axhline(-3, color="#D55E00", ls="--", lw=0.9)
    axes[0].set_ylabel("Standardised value (z)")
    axes[0].set_title("Standardised feature spread; dashed lines at |z| = 3")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7.2)

    order = np.argsort(counts)
    axes[1].barh([FEATURES[i] for i in order], [counts[i] for i in order],
                 color="#0072B2", height=0.7)
    axes[1].set_xlabel("Observations outside 1.5 x IQR")
    axes[1].set_title("Tukey outlier counts per feature")
    axes[1].tick_params(labelsize=7.4)
    fig.tight_layout()
    _save(fig, "fig_3_9_outlier_analysis.png")


def discriminative_power(df: pd.DataFrame) -> pd.DataFrame:
    """One-way ANOVA F-statistic per feature (descriptive, not model output)."""
    rows = []
    for col in FEATURES:
        groups = [df.loc[df[TARGET] == c, col].to_numpy() for c in CLASS_ORDER]
        f, p = stats.f_oneway(*groups)
        rows.append({"feature": col, "F": f, "p_value": p})
    table = pd.DataFrame(rows).sort_values("F", ascending=False)
    table.to_csv(METRICS / "anova_feature_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.barh(table["feature"][::-1], table["F"][::-1], color="#009E73", height=0.72)
    ax.set_xscale("log")
    ax.set_xlabel("One-way ANOVA F-statistic (log scale)")
    ax.set_title("Between-class separation offered by each feature")
    ax.tick_params(labelsize=8)
    for i, value in enumerate(table["F"][::-1]):
        ax.text(value * 1.08, i, f"{value:,.0f}", va="center", fontsize=7.4)
    _save(fig, "fig_3_10_discriminative_power.png")
    return table


def descriptive_tables(df: pd.DataFrame) -> None:
    desc = df[FEATURES].describe().T
    desc["skew"] = df[FEATURES].skew()
    desc["kurtosis"] = df[FEATURES].kurtosis()
    desc.round(4).to_csv(METRICS / "descriptive_statistics.csv")

    class_means = df.groupby(TARGET)[FEATURES].mean().reindex(CLASS_ORDER)
    class_means.round(4).to_csv(METRICS / "class_feature_means.csv")


def run_all(df: pd.DataFrame) -> None:
    print("EDA figures -> figures/eda/")
    class_distribution(df)
    missing_values(df)
    feature_histograms(df)
    boxplots_by_class(df)
    correlation_heatmap(df)
    pairplot(df)
    classwise_comparison(df)
    scatter_relationships(df)
    outlier_analysis(df)
    discriminative_power(df)
    descriptive_tables(df)
    print("EDA tables -> results/metrics/")


if __name__ == "__main__":
    from src.data_loader import load_dataset

    run_all(load_dataset())
