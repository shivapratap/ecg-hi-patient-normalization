#!/usr/bin/env python3
"""Publication-quality Figures 4-6 (+ supplementary) for the ECG-HI manuscript.

Target venue: IEEE Access (two-column, 181 mm / 7.16 in text width; single
column 88 mm / 3.46 in).  The script reads ONLY frozen result files, performs
no model refitting and no new inferential statistics, and writes every figure
as vector PDF/SVG plus a 600-dpi PNG.

Design conventions used here (matching how comparable ML-for-physiology papers
in IEEE Access / Physiol. Meas. / Comput. Biol. Med. present these results):

  Figure 4  Grouped bar charts for pooled metrics + violin/box/strip plots for
            the patient-level AUC distribution, with pooled AUC overlaid as a
            distinct marker so pooled and per-patient views are never confused.
  Figure 5  Horizontal grouped bar chart of feature-selection frequency across
            normalisation strategies (annotated heat map written separately as
            a supplementary alternative).
  Figure 6  Paired horizontal bar chart of per-patient AUC ranked by jackknife
            AUC, plus agreement and QC-burden scatter panels.

Legibility rules enforced throughout: no tick/label overlap (long category
names are wrapped or made horizontal), all legends are placed outside the data
area or in a guaranteed-empty corner, constrained_layout handles spacing, and
every numeric bar is labelled so values are readable in print.

Expected input files (default: --input-dir, else the script directory):
    loso_pooled_metrics.csv
    loso_patient_metrics.csv
    feature_selection_stability.csv
    patient_heterogeneity_master.csv
    final_feature_list.txt
    feature_stability_configuration_summary.csv   (optional, Jaccard panel)

Usage:
    python create_results_figures.py --input-dir data/results --output-dir figures
    python create_results_figures.py --font-family sans      # Arial-style
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Frozen analysis configuration
# -----------------------------------------------------------------------------
PRIMARY_MODE = "common_rows"
PRIMARY_CLASSIFIER = "extra_trees"
EXPECTED_FEATURE_COUNT = 34
EXPECTED_PATIENT_COUNT = 20
STABILITY_THRESHOLD = 0.75
BELOW_CHANCE_PATIENTS = (30851, 27245)

NORMALISATION_ORDER = [
    "none",
    "calibration_median",
    "calibration_median_iqr",
    "jackknife_median",
]

# Short labels for dense axes; wrapped labels for wide axes.
NORMALISATION_LABELS = {
    "none": "None",
    "calibration_median": "Calibration median",
    "calibration_median_iqr": "Calibration median\u2013IQR",
    "jackknife_median": "Jackknife median",
}
NORMALISATION_LABELS_WRAPPED = {
    "none": "None",
    "calibration_median": "Calibration\nmedian",
    "calibration_median_iqr": "Calibration\nmedian\u2013IQR",
    "jackknife_median": "Jackknife\nmedian",
}

CLASSIFIER_ORDER = ["logistic_regression", "extra_trees", "random_forest"]
CLASSIFIER_LABELS = {
    "logistic_regression": "Logistic\nregression",
    "extra_trees": "Extra Trees\n(primary)",
    "random_forest": "Random\nforest",
}

# Colour-blind-safe, print-safe palette (distinct in greyscale as well:
# light grey -> mid blue -> orange -> dark navy).
COLORS = {
    "none": "#B0B7BD",
    "calibration_median": "#5B9BD5",
    "calibration_median_iqr": "#E08214",
    "jackknife_median": "#1F4E79",
    "chance": "#4D4D4D",
    "connector": "#C4C4C4",
    "annotation": "#1A1A1A",
    "highlight": "#C0392B",
    "grid": "#CFCFCF",
}
# Hatches guarantee separability if the figure is printed in greyscale.
HATCHES = {
    "none": "",
    "calibration_median": "//",
    "calibration_median_iqr": "\\\\",
    "jackknife_median": "",
}

METRIC_ORDER = ["pooled_roc_auc", "pooled_average_precision", "balanced_accuracy"]
METRIC_LABELS = {
    "pooled_roc_auc": "ROC AUC",
    "pooled_average_precision": "Average\nprecision",
    "balanced_accuracy": "Balanced\naccuracy",
}

# IEEE Access column widths (inches).
SINGLE_COLUMN = 3.46
DOUBLE_COLUMN = 7.16

DEFAULT_FILES = {
    "pooled": "loso_pooled_metrics.csv",
    "patient": "loso_patient_metrics.csv",
    "stability": "feature_selection_stability.csv",
    "heterogeneity": "patient_heterogeneity_master.csv",
    "features": "final_feature_list.txt",
    "configuration": "feature_stability_configuration_summary.csv",
}

SERIF_STACK = [
    "Times New Roman",
    "Nimbus Roman",
    "Nimbus Roman No9 L",
    "Liberation Serif",
    "TeX Gyre Termes",
    "STIXGeneral",
    "DejaVu Serif",
]
SANS_STACK = [
    "Arial",
    "Helvetica",
    "Nimbus Sans",
    "Liberation Sans",
    "TeX Gyre Heros",
    "DejaVu Sans",
]


# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------
def configure_style(font_family: str = "serif") -> None:
    """Set a consistent, IEEE-compatible plotting theme.

    Font sizes are chosen so that text is >= 7 pt at final print size when the
    figure is placed at its native width (no downstream scaling required).
    """
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    stack = SERIF_STACK if font_family == "serif" else SANS_STACK
    generic = "serif" if font_family == "serif" else "sans-serif"
    plt.rcParams.update(
        {
            "font.family": generic,
            "font.serif": SERIF_STACK,
            "font.sans-serif": SANS_STACK,
            "mathtext.fontset": "stix" if font_family == "serif" else "dejavusans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.titlepad": 4.0,
            "axes.labelsize": 8.0,
            "axes.labelpad": 2.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "legend.title_fontsize": 7.5,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.1,
            "legend.borderpad": 0.35,
            "figure.titlesize": 9.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "grid.linewidth": 0.4,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.8,
            "lines.linewidth": 0.9,
            "patch.linewidth": 0.5,
            "hatch.linewidth": 0.4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,   # embed TrueType, editable text
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    plt.rcParams["font.serif"] = stack if font_family == "serif" else SERIF_STACK


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Write vector (PDF/SVG) and 600-dpi raster (PNG) versions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.svg")
    fig.savefig(output_dir / f"{stem}.png", dpi=600)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.085, y: float = 1.10) -> None:
    """IEEE-style lower-case panel tag placed outside the data area."""
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def style_axis(ax: plt.Axes, axis: str = "y") -> None:
    """Light reference grid behind the data, spines trimmed."""
    ax.set_axisbelow(True)
    ax.grid(axis=axis, linestyle="-", linewidth=0.4, color=COLORS["grid"], alpha=0.8)
    ax.grid(axis="x" if axis == "y" else "y", visible=False)
    sns.despine(ax=ax, top=True, right=True)


def label_bars(
    ax: plt.Axes,
    container,
    fmt: str = "{:.3f}",
    fontsize: float = 6.0,
    horizontal: bool = False,
    padding: float = 1.4,
) -> None:
    labels = [fmt.format(v) for v in (
        [p.get_width() for p in container] if horizontal else [p.get_height() for p in container]
    )]
    ax.bar_label(container, labels=labels, fontsize=fontsize, padding=padding, color="#1A1A1A")


# -----------------------------------------------------------------------------
# Input loading and validation
# -----------------------------------------------------------------------------
def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_inputs(input_dir: Path, filenames: dict[str, str]) -> dict[str, object]:
    paths = {key: input_dir / value for key, value in filenames.items()}
    required = [k for k in paths if k != "configuration"]
    missing = [str(paths[k]) for k in required if not paths[k].exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))

    pooled = pd.read_csv(paths["pooled"])
    patient = pd.read_csv(paths["patient"])
    stability = pd.read_csv(paths["stability"])
    heterogeneity = pd.read_csv(paths["heterogeneity"])
    features = [
        line.strip()
        for line in paths["features"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    configuration = (
        pd.read_csv(paths["configuration"]) if paths["configuration"].exists() else None
    )

    require_columns(
        pooled,
        {"evaluation_mode", "normalization", "classifier", "pooled_roc_auc",
         "pooled_average_precision", "balanced_accuracy"},
        "Pooled metrics",
    )
    require_columns(
        patient,
        {"outer_fold_patient_id", "evaluation_mode", "normalization", "classifier",
         "roc_auc"},
        "Patient metrics",
    )
    require_columns(
        stability,
        {"evaluation_mode", "normalization", "classifier", "feature_name",
         "final_selection_frequency"},
        "Feature stability",
    )
    require_columns(
        heterogeneity,
        {"patient_id", "jackknife_roc_auc", "calibration_iqr_roc_auc",
         "review_window_fraction", "later_continuous_block_used"},
        "Patient heterogeneity",
    )

    if len(features) != len(set(features)):
        raise ValueError("The frozen feature list contains duplicate entries")
    if len(features) != EXPECTED_FEATURE_COUNT:
        warnings.warn(
            f"Frozen feature list has {len(features)} entries; "
            f"{EXPECTED_FEATURE_COUNT} were expected.",
            stacklevel=2,
        )

    primary_pooled = pooled[
        (pooled["evaluation_mode"] == PRIMARY_MODE)
        & (pooled["classifier"] == PRIMARY_CLASSIFIER)
    ]
    if set(primary_pooled["normalization"]) != set(NORMALISATION_ORDER):
        raise ValueError("Primary pooled metrics do not contain all four normalisations")

    primary_patient = patient[
        (patient["evaluation_mode"] == PRIMARY_MODE)
        & (patient["classifier"] == PRIMARY_CLASSIFIER)
    ]
    counts = primary_patient.groupby("normalization")["outer_fold_patient_id"].nunique()
    if not (counts.reindex(NORMALISATION_ORDER) == EXPECTED_PATIENT_COUNT).all():
        raise ValueError(
            f"Each primary normalisation must contain {EXPECTED_PATIENT_COUNT} patients"
        )
    if len(heterogeneity) != EXPECTED_PATIENT_COUNT:
        raise ValueError(
            f"Patient heterogeneity table must contain {EXPECTED_PATIENT_COUNT} rows"
        )

    return {
        "pooled": pooled,
        "patient": patient,
        "stability": stability,
        "heterogeneity": heterogeneity,
        "features": features,
        "configuration": configuration,
    }


def primary_frames(pooled: pd.DataFrame, patient: pd.DataFrame):
    """Return (pooled primary rows, patient primary rows) in frozen order."""
    pooled_primary = pooled[
        (pooled["evaluation_mode"] == PRIMARY_MODE)
        & (pooled["classifier"] == PRIMARY_CLASSIFIER)
    ].copy()
    pooled_primary["normalization"] = pd.Categorical(
        pooled_primary["normalization"], categories=NORMALISATION_ORDER, ordered=True
    )
    pooled_primary = pooled_primary.sort_values("normalization")

    patient_primary = patient[
        (patient["evaluation_mode"] == PRIMARY_MODE)
        & (patient["classifier"] == PRIMARY_CLASSIFIER)
    ].copy()
    patient_primary["normalization"] = pd.Categorical(
        patient_primary["normalization"], categories=NORMALISATION_ORDER, ordered=True
    )
    return pooled_primary, patient_primary


# -----------------------------------------------------------------------------
# Figure 4: pooled and patient-level discrimination
# -----------------------------------------------------------------------------
def create_figure_4(pooled: pd.DataFrame, patient: pd.DataFrame, output_dir: Path) -> None:
    pooled_primary, patient_primary = primary_frames(pooled, patient)
    pooled_lookup = pooled_primary.set_index("normalization")["pooled_roc_auc"]

    fig = plt.figure(figsize=(DOUBLE_COLUMN, 6.9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], width_ratios=[1.28, 1.0])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    # ---- Panel (a): grouped bars, three pooled metrics x four normalisations --
    n_norm = len(NORMALISATION_ORDER)
    group_x = np.arange(len(METRIC_ORDER))
    bar_w = 0.20
    offsets = (np.arange(n_norm) - (n_norm - 1) / 2) * bar_w

    for j, norm in enumerate(NORMALISATION_ORDER):
        row = pooled_primary[pooled_primary["normalization"] == norm].iloc[0]
        values = [row[m] for m in METRIC_ORDER]
        bars = ax_a.bar(
            group_x + offsets[j],
            values,
            width=bar_w * 0.92,
            color=COLORS[norm],
            edgecolor="#2B2B2B",
            linewidth=0.5,
            hatch=HATCHES[norm],
            label=NORMALISATION_LABELS[norm],
            zorder=3,
        )
        label_bars(ax_a, bars, fmt="{:.3f}", fontsize=5.9, padding=1.6)

    chance_line = ax_a.axhline(
        0.5, color=COLORS["chance"], linestyle=(0, (4, 3)), linewidth=0.9, zorder=2,
        label="Chance (0.50)",
    )
    ax_a.set_xticks(group_x)
    ax_a.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
    ax_a.set_xlim(-0.55, len(METRIC_ORDER) - 0.45)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax_a.set_ylabel("Pooled out-of-fold performance")
    ax_a.set_title(
        "Pooled discrimination by patient-level normalisation "
        "(common rows, Extra Trees, 20 LOSO folds)"
    )
    style_axis(ax_a, axis="y")
    handles, labels = ax_a.get_legend_handles_labels()
    ax_a.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=5,
        frameon=False,
        title="Patient-level normalisation strategy",
    )
    add_panel_label(ax_a, "a", x=-0.055, y=1.14)

    # ---- Panel (b): patient-level AUC distributions (violin + box + points) ---
    dist = patient_primary.copy()
    dist["label"] = dist["normalization"].map(NORMALISATION_LABELS_WRAPPED)
    order = [NORMALISATION_LABELS_WRAPPED[n] for n in NORMALISATION_ORDER]
    palette = {NORMALISATION_LABELS_WRAPPED[n]: COLORS[n] for n in NORMALISATION_ORDER}

    sns.violinplot(
        data=dist, x="label", y="roc_auc", order=order, hue="label",
        hue_order=order, palette=palette, dodge=False, inner=None, cut=0,
        density_norm="width", linewidth=0.6, saturation=0.9, legend=False, ax=ax_b,
    )
    for collection in ax_b.collections:
        collection.set_alpha(0.45)
        collection.set_edgecolor("#3A3A3A")
    sns.boxplot(
        data=dist, x="label", y="roc_auc", order=order, width=0.20,
        showfliers=False, showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "#2B2B2B", "linewidth": 0.7,
                  "zorder": 4},
        whiskerprops={"linewidth": 0.7, "color": "#2B2B2B"},
        capprops={"linewidth": 0.7, "color": "#2B2B2B"},
        medianprops={"color": "#000000", "linewidth": 1.2},
        zorder=4, ax=ax_b,
    )
    sns.stripplot(
        data=dist, x="label", y="roc_auc", order=order, hue="label", hue_order=order,
        palette=palette, dodge=False, size=2.8, alpha=0.9, jitter=0.11,
        edgecolor="#2B2B2B", linewidth=0.3, legend=False, zorder=5, ax=ax_b,
    )
    pooled_values = [pooled_lookup[n] for n in NORMALISATION_ORDER]
    ax_b.scatter(
        np.arange(n_norm), pooled_values, marker="D", s=26, facecolor="white",
        edgecolor="#000000", linewidth=0.9, zorder=6,
    )
    ax_b.axhline(0.5, color=COLORS["chance"], linestyle=(0, (4, 3)), linewidth=0.8, zorder=2)
    ax_b.set_ylim(-0.03, 1.06)
    ax_b.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Held-out patient ROC AUC")
    ax_b.set_title("Patient-level AUC distribution vs. pooled AUC")
    style_axis(ax_b, axis="y")
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white",
                   markeredgecolor="black", markersize=4.2, label="Pooled ROC AUC"),
            Line2D([0], [0], color="black", linewidth=1.2, label="Median patient AUC"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#7F7F7F",
                   markeredgecolor="#2B2B2B", markersize=3.4,
                   label="Individual patient ($n=20$)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
        columnspacing=0.9,
    )
    add_panel_label(ax_b, "b", x=-0.14, y=1.12)

    # ---- Panel (c): classifier sensitivity of pooled ROC AUC ------------------
    classifier_frame = pooled[pooled["evaluation_mode"] == PRIMARY_MODE].copy()
    group_x = np.arange(len(CLASSIFIER_ORDER))
    offsets = (np.arange(n_norm) - (n_norm - 1) / 2) * bar_w
    for j, norm in enumerate(NORMALISATION_ORDER):
        values = [
            classifier_frame.loc[
                (classifier_frame["classifier"] == clf)
                & (classifier_frame["normalization"] == norm),
                "pooled_roc_auc",
            ].iloc[0]
            for clf in CLASSIFIER_ORDER
        ]
        bars = ax_c.bar(
            group_x + offsets[j], values, width=bar_w * 0.92, color=COLORS[norm],
            edgecolor="#2B2B2B", linewidth=0.5, hatch=HATCHES[norm], zorder=3,
        )
        label_bars(ax_c, bars, fmt="{:.2f}", fontsize=5.6, padding=1.3)

    ax_c.axhline(0.5, color=COLORS["chance"], linestyle=(0, (4, 3)), linewidth=0.8, zorder=2)
    ax_c.set_xticks(group_x)
    ax_c.set_xticklabels([CLASSIFIER_LABELS[c] for c in CLASSIFIER_ORDER])
    ax_c.set_xlim(-0.55, len(CLASSIFIER_ORDER) - 0.45)
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax_c.set_ylabel("Pooled ROC AUC")
    ax_c.set_title("Classifier sensitivity (secondary)")
    ax_c.text(
        0.015, 0.985, "Colours as in (a)", transform=ax_c.transAxes, fontsize=6.6,
        color="#444444", va="top", ha="left",
    )
    style_axis(ax_c, axis="y")
    add_panel_label(ax_c, "c", x=-0.16, y=1.12)

    save_figure(fig, output_dir, "Figure_4_primary_and_patient_discrimination")


# -----------------------------------------------------------------------------
# Figure 5: feature-selection stability
# -----------------------------------------------------------------------------
PRETTY_FEATURES = {
    "minimum": "Minimum",
    "maximum": "Maximum",
    "mean": "Mean",
    "median": "Median",
    "kurtosis": "Kurtosis",
    "skewness": "Skewness",
    "mean_absolute_value": "Mean absolute value",
    "root_mean_square": "Root mean square",
    "peak_to_peak": "Peak-to-peak",
    "waveform_length": "Waveform length",
    "zero_crossing_count": "Zero crossings",
    "slope_sign_change_count": "Slope sign changes",
    "approximate_entropy": "Approximate entropy",
    "permutation_entropy": "Permutation entropy",
    "fuzzy_entropy": "Fuzzy entropy",
    "distribution_entropy": "Distribution entropy",
    "svd_entropy": "SVD entropy",
    "lempel_ziv_complexity": "Lempel\u2013Ziv complexity",
    "hjorth_mobility": "Hjorth mobility",
    "hjorth_complexity": "Hjorth complexity",
    "katz_fractal_dimension": "Katz fractal dimension",
    "higuchi_fractal_dimension": "Higuchi fractal dimension",
    "detrended_fluctuation_analysis": "Detrended fluctuation analysis",
    "mean_frequency": "Mean frequency",
    "median_frequency": "Median frequency",
    "spectral_edge_frequency_95": "Spectral edge frequency (95%)",
    "spectral_entropy": "Spectral entropy",
    "AvgSampEn": "SampEn profile: mean",
    "MaxSampEn": "SampEn profile: maximum",
    "MedianSampEn": "SampEn profile: median",
    "StdSampEn": "SampEn profile: SD",
    "KurtosisSampEn": "SampEn profile: kurtosis",
    "SkewnessSampEn": "SampEn profile: skewness",
    "AUC_SampEn": "SampEn profile: AUC",
}


def prettify_feature_name(name: str) -> str:
    return PRETTY_FEATURES.get(name, name.replace("_", " ").capitalize())


def stability_matrix(stability: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    subset = stability[
        (stability["evaluation_mode"] == PRIMARY_MODE)
        & (stability["classifier"] == PRIMARY_CLASSIFIER)
        & stability["normalization"].isin(NORMALISATION_ORDER)
    ]
    pivot = subset.pivot(
        index="feature_name", columns="normalization", values="final_selection_frequency"
    ).reindex(index=features, columns=NORMALISATION_ORDER)
    if pivot.isna().any().any():
        raise ValueError("Feature-stability matrix contains missing configuration values")
    # Rank by mean selection frequency; frozen list order breaks ties.
    order = pivot.assign(_mean=pivot.mean(axis=1), _frozen=np.arange(len(pivot)))
    order = order.sort_values(["_mean", "_frozen"], ascending=[False, True])
    return pivot.loc[order.index]


def _feature_bar_panel(
    ax: plt.Axes,
    block: pd.DataFrame,
    labels: list[str],
    show_legend: bool = False,
) -> None:
    """Grouped horizontal bars of selection frequency for one block of features."""
    n_rows = len(block)
    n_norm = len(NORMALISATION_ORDER)
    y = np.arange(n_rows)[::-1]
    bar_h = 0.20
    offsets = ((np.arange(n_norm) - (n_norm - 1) / 2) * bar_h)[::-1]

    for i in range(n_rows):
        if i % 2 == 0:
            ax.axhspan(y[i] - 0.5, y[i] + 0.5, color="#F2F3F5", zorder=0)

    for j, norm in enumerate(NORMALISATION_ORDER):
        ax.barh(
            y + offsets[j],
            block[norm].to_numpy(),
            height=bar_h * 0.90,
            color=COLORS[norm],
            edgecolor="#2B2B2B",
            linewidth=0.35,
            label=NORMALISATION_LABELS[norm] if show_legend else None,
            zorder=3,
        )

    ax.axvline(
        STABILITY_THRESHOLD, color=COLORS["highlight"], linestyle=(0, (3.5, 2.5)),
        linewidth=0.9, zorder=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlim(0, 1.04)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_xlabel("Selection frequency (20 folds)")
    style_axis(ax, axis="x")
    ax.tick_params(axis="y", length=0)


def create_figure_5(
    stability: pd.DataFrame,
    features: list[str],
    output_dir: Path,
    configuration: pd.DataFrame | None = None,
) -> None:
    pivot = stability_matrix(stability, features)
    labels = [prettify_feature_name(name) for name in pivot.index]
    n_features = len(pivot)
    split = int(np.ceil(n_features / 2))

    has_jaccard = (
        configuration is not None
        and {"evaluation_mode", "classifier", "normalization",
             "mean_pairwise_jaccard"}.issubset(configuration.columns)
    )

    rows_height = 0.255 * split
    fig_height = rows_height + (2.55 if has_jaccard else 1.35)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, fig_height), constrained_layout=True)
    if has_jaccard:
        grid = fig.add_gridspec(2, 2, height_ratios=[rows_height, 1.15], hspace=0.06)
    else:
        grid = fig.add_gridspec(1, 2)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    # ---- Panels (a) and (b): ranked selection frequency, split into two columns
    _feature_bar_panel(ax_a, pivot.iloc[:split], labels[:split], show_legend=True)
    _feature_bar_panel(ax_b, pivot.iloc[split:], labels[split:])
    ax_a.set_title(f"Selection frequency, ranks 1\u2013{split}")
    ax_b.set_title(f"Selection frequency, ranks {split + 1}\u2013{n_features}")
    ax_a.set_ylabel("Frozen predictor (ranked by mean selection frequency)")
    add_panel_label(ax_a, "a", x=-0.66, y=1.05)
    add_panel_label(ax_b, "b", x=-0.50, y=1.05)

    handles, legend_labels = ax_a.get_legend_handles_labels()
    threshold_handle = Line2D(
        [0], [0], color=COLORS["highlight"], linestyle=(0, (3.5, 2.5)), linewidth=0.9,
        label="Stability threshold (0.75)",
    )
    fig.legend(
        handles + [threshold_handle],
        legend_labels + ["Stability threshold (0.75)"],
        loc="outside upper center", ncol=5, frameon=False,
        title="Patient-level normalisation strategy",
    )

    # ---- Panel (c): mean pairwise Jaccard similarity per strategy -------------
    if has_jaccard:
        ax_c = fig.add_subplot(grid[1, :])
        conf = configuration[
            (configuration["evaluation_mode"] == PRIMARY_MODE)
            & (configuration["classifier"] == PRIMARY_CLASSIFIER)
        ].set_index("normalization")
        order = [n for n in NORMALISATION_ORDER if n in conf.index]
        x = np.arange(len(order))
        values = conf.loc[order, "mean_pairwise_jaccard"].to_numpy()
        lows = (
            conf.loc[order, "minimum_pairwise_jaccard"].to_numpy()
            if "minimum_pairwise_jaccard" in conf.columns else None
        )
        highs = (
            conf.loc[order, "maximum_pairwise_jaccard"].to_numpy()
            if "maximum_pairwise_jaccard" in conf.columns else None
        )
        err = None
        if lows is not None and highs is not None:
            err = np.vstack([values - lows, highs - values])

        bars = ax_c.bar(
            x, values, width=0.46,
            color=[COLORS[n] for n in order], edgecolor="#2B2B2B", linewidth=0.5,
            yerr=err, capsize=2.0,
            error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#3A3A3A"},
            zorder=3,
        )
        label_bars(ax_c, bars, fmt="{:.3f}", fontsize=6.6, padding=2.0)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels([NORMALISATION_LABELS_WRAPPED[n] for n in order])
        ax_c.set_xlim(-0.6, len(order) - 0.4)
        ax_c.set_ylim(0, 1.12)
        ax_c.set_yticks(np.arange(0, 1.01, 0.25))
        ax_c.set_ylabel("Pairwise Jaccard\nsimilarity")
        ax_c.set_title(
            "Fold-to-fold agreement of retained feature sets "
            "(whiskers: minimum\u2013maximum over fold pairs)"
        )
        style_axis(ax_c, axis="y")
        add_panel_label(ax_c, "c", x=-0.075, y=1.16)

    save_figure(fig, output_dir, "Figure_5_feature_selection_stability")

    # ---- Supplementary alternative: annotated heat map ------------------------
    fig_h, ax_h = plt.subplots(
        figsize=(SINGLE_COLUMN + 1.35, 0.205 * n_features + 1.1), constrained_layout=True
    )
    heat = pivot.copy()
    heat.index = labels
    heat.columns = [NORMALISATION_LABELS_WRAPPED[n] for n in NORMALISATION_ORDER]
    sns.heatmap(
        heat, cmap=sns.light_palette(COLORS["jackknife_median"], as_cmap=True),
        vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"fontsize": 5.8},
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Final selection frequency", "shrink": 0.6, "pad": 0.02},
        ax=ax_h,
    )
    ax_h.set_xlabel("")
    ax_h.set_ylabel("Frozen predictor")
    ax_h.set_title("Selection frequency across 20 LOSO folds")
    ax_h.tick_params(axis="x", rotation=0, labelsize=6.6, length=0)
    ax_h.tick_params(axis="y", rotation=0, labelsize=6.6, length=0)
    ax_h.figure.axes[-1].yaxis.label.set_size(7.2)
    save_figure(fig_h, output_dir, "Figure_S1_feature_selection_stability_heatmap")


# -----------------------------------------------------------------------------
# Figure 6: patient-level heterogeneity
# -----------------------------------------------------------------------------
def create_figure_6(heterogeneity: pd.DataFrame, output_dir: Path) -> None:
    frame = heterogeneity.copy()
    frame["patient_id"] = frame["patient_id"].astype(int)
    frame["later_continuous_block_used"] = frame["later_continuous_block_used"].astype(bool)
    frame = frame.sort_values(
        ["jackknife_roc_auc", "patient_id"], ascending=[False, True]
    ).reset_index(drop=True)
    n = len(frame)

    fig = plt.figure(figsize=(DOUBLE_COLUMN, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.32, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    # ---- Panel (a): paired horizontal bars, ranked by jackknife AUC ----------
    y = np.arange(n)[::-1]
    bar_h = 0.38
    ax_a.barh(
        y + bar_h / 2, frame["jackknife_roc_auc"], height=bar_h * 0.95,
        color=COLORS["jackknife_median"], edgecolor="#2B2B2B", linewidth=0.4,
        label="Jackknife median (retrospective reference)", zorder=3,
    )
    ax_a.barh(
        y - bar_h / 2, frame["calibration_iqr_roc_auc"], height=bar_h * 0.95,
        color=COLORS["calibration_median_iqr"], edgecolor="#2B2B2B", linewidth=0.4,
        hatch=HATCHES["calibration_median_iqr"],
        label="Calibration median\u2013IQR (deployable)", zorder=3,
    )

    below = frame["jackknife_roc_auc"] < 0.5
    for idx in np.flatnonzero(below.to_numpy()):
        ax_a.axhspan(y[idx] - 0.5, y[idx] + 0.5, color="#FBE9E7", zorder=0)

    ax_a.axvline(0.5, color=COLORS["chance"], linestyle=(0, (4, 3)), linewidth=0.9, zorder=4)
    ax_a.text(
        0.5, -0.62, "chance", fontsize=6.5, color=COLORS["chance"],
        ha="center", va="bottom",
    )
    tick_labels = [
        f"{int(pid)}\u2020" if later else f"{int(pid)}"
        for pid, later in zip(frame["patient_id"], frame["later_continuous_block_used"])
    ]
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(tick_labels)
    for tick, is_below in zip(ax_a.get_yticklabels(), below):
        if is_below:
            tick.set_color(COLORS["highlight"])
            tick.set_fontweight("bold")
    ax_a.set_ylim(-0.7, n - 0.05)
    ax_a.set_xlim(0, 1.045)
    ax_a.set_xticks(np.arange(0, 1.01, 0.2))
    ax_a.set_xlabel("Held-out patient ROC AUC")
    ax_a.set_ylabel("Patient ID (ranked by jackknife AUC)")
    ax_a.set_title("Per-patient discrimination (common rows, Extra Trees)")
    style_axis(ax_a, axis="x")
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.045), ncol=1, frameon=False,
    )
    ax_a.text(
        0.0, -0.075,
        "\u2020 later continuous calibration block required; "
        "red rows: below-chance jackknife AUC.",
        transform=ax_a.transAxes, fontsize=6.3, color="#444444", ha="left", va="top",
    )
    add_panel_label(ax_a, "a", x=-0.155, y=1.10)

    # ---- Panel (b): agreement between the two strategies ---------------------
    qc = frame["review_window_fraction"].to_numpy()
    scatter = ax_b.scatter(
        frame["jackknife_roc_auc"], frame["calibration_iqr_roc_auc"],
        c=qc, cmap="cividis", vmin=0, vmax=1, s=26, edgecolor="#1A1A1A",
        linewidth=0.5, zorder=3,
    )
    ax_b.plot([0.2, 1.02], [0.2, 1.02], color="#8A8A8A", linestyle=(0, (4, 3)),
              linewidth=0.8, zorder=2)
    ax_b.axvline(0.5, color=COLORS["chance"], linestyle=(0, (1.5, 2)), linewidth=0.7, zorder=1)
    ax_b.axhline(0.5, color=COLORS["chance"], linestyle=(0, (1.5, 2)), linewidth=0.7, zorder=1)
    for pid in BELOW_CHANCE_PATIENTS:
        match = frame.index[frame["patient_id"] == pid]
        if len(match) == 1:
            row = frame.loc[match[0]]
            ax_b.annotate(
                str(pid),
                xy=(row["jackknife_roc_auc"], row["calibration_iqr_roc_auc"]),
                xytext=(9, -2), textcoords="offset points", fontsize=6.4,
                color=COLORS["highlight"], fontweight="bold",
            )
    rho = frame["jackknife_roc_auc"].corr(frame["calibration_iqr_roc_auc"], method="spearman")
    ax_b.text(
        0.03, 0.965, f"Spearman $\\rho$ = {rho:.3f}\n$n$ = {n} patients",
        transform=ax_b.transAxes, fontsize=6.8, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
              "edgecolor": "#C9C9C9", "linewidth": 0.5},
    )
    ax_b.set_xlim(0.2, 1.04)
    ax_b.set_ylim(0.2, 1.04)
    ax_b.set_xticks(np.arange(0.2, 1.01, 0.2))
    ax_b.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax_b.set_xlabel("Jackknife median AUC")
    ax_b.set_ylabel("Calibration median\u2013IQR AUC")
    ax_b.set_title("Agreement between strategies")
    style_axis(ax_b, axis="y")
    ax_b.grid(axis="x", visible=True, linestyle="-", linewidth=0.4,
              color=COLORS["grid"], alpha=0.8)
    cbar = fig.colorbar(scatter, ax=ax_b, pad=0.02, fraction=0.045)
    cbar.set_label("QC-review window fraction", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.4, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)
    add_panel_label(ax_b, "b", x=-0.20, y=1.14)

    # ---- Panel (c): AUC against QC-review burden ------------------------------
    ax_c.scatter(
        qc, frame["jackknife_roc_auc"], s=24, color=COLORS["jackknife_median"],
        edgecolor="white", linewidth=0.4, label="Jackknife median", zorder=3,
    )
    ax_c.scatter(
        qc, frame["calibration_iqr_roc_auc"], s=24, marker="^",
        color=COLORS["calibration_median_iqr"], edgecolor="white", linewidth=0.4,
        label="Calibration median\u2013IQR", zorder=3,
    )
    ax_c.axhline(0.5, color=COLORS["chance"], linestyle=(0, (4, 3)), linewidth=0.8, zorder=2)
    ax_c.set_xlim(-0.04, 1.04)
    ax_c.set_ylim(0.2, 1.34)
    ax_c.set_xticks(np.arange(0, 1.01, 0.25))
    ax_c.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax_c.set_xlabel("QC-review window fraction")
    ax_c.set_ylabel("Held-out patient ROC AUC")
    ax_c.set_title("Discrimination vs. signal-quality burden")
    style_axis(ax_c, axis="y")
    ax_c.legend(loc="upper left", bbox_to_anchor=(-0.012, 1.02), ncol=1, frameon=False)
    add_panel_label(ax_c, "c", x=-0.20, y=1.14)

    save_figure(fig, output_dir, "Figure_6_patient_heterogeneity")


# -----------------------------------------------------------------------------
# Draft captions (kept next to the figures for the manuscript session)
# -----------------------------------------------------------------------------
CAPTIONS = """\
Draft figure captions (verify every number against the frozen CSV files).

Fig. 4. Discrimination of pre-haemodynamic-instability versus haemodynamic-
instability windows under four patient-level normalisation strategies.
(a) Pooled out-of-fold ROC AUC, average precision and balanced accuracy for the
primary configuration (common-row evaluation, Extra Trees, 20 leave-one-subject-
out folds). (b) Distribution of per-patient ROC AUC (violin: kernel density;
box: median and interquartile range; points: individual held-out patients,
n = 20); white diamonds mark the corresponding pooled ROC AUC, which is lower
than the median patient AUC because prediction-score distributions remain
heterogeneous across patients. (c) Pooled ROC AUC for all three classifiers,
showing that the ordering of normalisation strategies is not classifier
specific. Dashed lines indicate chance (0.50). Jackknife normalisation is a
retrospective upper-bound reference and is not deployable.

Fig. 5. Reproducibility of feature selection across leave-one-subject-out
folds (common-row evaluation, Extra Trees). (a) Frequency with which each
frozen predictor survived the training-only selection pipeline in the 20 outer
folds, ranked by mean frequency across strategies; the dashed line marks the
0.75 (15/20 folds) descriptive stability threshold. (b) Mean pairwise Jaccard
similarity between the retained feature sets of different folds (whiskers:
minimum and maximum over all fold pairs). Stable features are reproducible
selections, not validated biomarkers.

Fig. 6. Patient-level heterogeneity in discrimination. (a) Per-patient ROC AUC
for the retrospective jackknife reference and the deployable calibration
median-IQR strategy, ranked by jackknife AUC; shaded rows and red labels mark
the two below-chance patients, and the dagger marks patients for whom a later
continuous calibration block was required. (b) Agreement between the two
strategies, coloured by QC-review window fraction; the dashed line is the line
of identity. (c) Per-patient ROC AUC against QC-review burden, showing that
signal-quality burden alone does not separate successful from unsuccessful
patients. Dashed lines indicate chance (0.50).

Fig. S1. Supplementary heat-map rendering of the selection frequencies shown in
Fig. 5(a).
"""


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, default=script_dir,
                        help="Directory containing the frozen result files.")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results_figures",
                        help="Directory for the PDF, SVG and PNG figures.")
    parser.add_argument("--font-family", choices=["serif", "sans"], default="serif",
                        help="Times-style serif (IEEE default) or Arial-style sans.")
    parser.add_argument("--pooled-file", default=DEFAULT_FILES["pooled"])
    parser.add_argument("--patient-file", default=DEFAULT_FILES["patient"])
    parser.add_argument("--stability-file", default=DEFAULT_FILES["stability"])
    parser.add_argument("--heterogeneity-file", default=DEFAULT_FILES["heterogeneity"])
    parser.add_argument("--feature-file", default=DEFAULT_FILES["features"])
    parser.add_argument("--configuration-file", default=DEFAULT_FILES["configuration"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    configure_style(args.font_family)

    filenames = {
        "pooled": args.pooled_file,
        "patient": args.patient_file,
        "stability": args.stability_file,
        "heterogeneity": args.heterogeneity_file,
        "features": args.feature_file,
        "configuration": args.configuration_file,
    }
    data = load_inputs(args.input_dir, filenames)

    create_figure_4(data["pooled"], data["patient"], args.output_dir)
    create_figure_5(data["stability"], data["features"], args.output_dir,
                    data["configuration"])
    create_figure_6(data["heterogeneity"], args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure_captions.txt").write_text(CAPTIONS, encoding="utf-8")

    print(f"Figures written to: {args.output_dir.resolve()}")
    for stem in [
        "Figure_4_primary_and_patient_discrimination",
        "Figure_5_feature_selection_stability",
        "Figure_6_patient_heterogeneity",
        "Figure_S1_feature_selection_stability_heatmap",
    ]:
        print(f"  {stem}.pdf / .svg / .png")
    print("  figure_captions.txt")


if __name__ == "__main__":
    main()
