"""Project-wide configuration: paths, seeds, feature names and plot styling."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
PREDICTIONS = RESULTS / "predictions"
FIG_EDA = ROOT / "figures" / "eda"
FIG_MODELS = ROOT / "figures" / "models"
FIG_EVAL = ROOT / "figures" / "evaluation"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS, METRICS, PREDICTIONS,
           FIG_EDA, FIG_MODELS, FIG_EVAL):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = "f1_macro"
DPI = 300

DATASET_FILE = "Dry_Bean_Dataset.xlsx"
DATASET_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
DATASET_ACCESSED = "2026-11-26"

# Column names are kept exactly as distributed by UCI, including the
# misspelling "AspectRation" in the original file.
FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]
TARGET = "Class"
CLASS_ORDER = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA"]

FEATURE_DESCRIPTIONS = {
    "Area": "Area of a bean zone in pixels; number of pixels inside the bean boundary.",
    "Perimeter": "Bean circumference, approximated by the length of its boundary.",
    "MajorAxisLength": "Length of the longest line that can be drawn through the bean.",
    "MinorAxisLength": "Length of the shortest line that can be drawn through the bean.",
    "AspectRation": "Ratio of MajorAxisLength to MinorAxisLength.",
    "Eccentricity": "Eccentricity of the ellipse that has the same moments as the bean.",
    "ConvexArea": "Number of pixels in the smallest convex polygon enclosing the bean.",
    "EquivDiameter": "Diameter of a circle with the same area as the bean.",
    "Extent": "Ratio of bean area to the area of its bounding box.",
    "Solidity": "Ratio of bean area to its convex hull area.",
    "roundness": "4*pi*Area / Perimeter^2; 1.0 for a perfect circle.",
    "Compactness": "EquivDiameter / MajorAxisLength.",
    "ShapeFactor1": "Ratio of Area to the cube of MajorAxisLength.",
    "ShapeFactor2": "Ratio of MinorAxisLength to MajorAxisLength.",
    "ShapeFactor3": "Ratio of Area to (MajorAxisLength * MinorAxisLength).",
    "ShapeFactor4": "Ratio of ConvexArea to Area-related product; roundness of the hull.",
}

# Colour-blind-safe qualitative palette (Okabe-Ito), one colour per variety.
PALETTE = {
    "SEKER": "#0072B2",
    "BARBUNYA": "#D55E00",
    "BOMBAY": "#009E73",
    "CALI": "#CC79A7",
    "DERMASON": "#E69F00",
    "HOROZ": "#56B4E9",
    "SIRA": "#666666",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "knn": "K-Nearest Neighbours",
    "svm_rbf": "RBF-SVM",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Hist Gradient Boosting",
}


def setup_plot_style():
    """Consistent, print-quality matplotlib styling for every figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.autolayout": False,
    })
