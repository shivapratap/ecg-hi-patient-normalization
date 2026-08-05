#!/usr/bin/env python3
"""Create publication-ready Figures 4, 5, and 6 for the ECG-HI manuscript.

The script uses only frozen result files and performs no model refitting or new
statistical analysis. It writes each figure as vector PDF/SVG and 600-dpi PNG.

Expected input files (default: same directory as this script):
    loso_pooled_metrics(1).csv
    loso_patient_metrics(1).csv
    feature_selection_stability(1).csv
    patient_heterogeneity_master(1).csv
    final_feature_list(1).txt

Usage:
    python create_results_figures.py
    python create_results_figures.py --input-dir /path/to/results \
        --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns



# -----------------------------------------------------------------------------
# Frozen analysis configuration
# -----------------------------------------------------------------------------
PRIMARY_MODE = "common_rows"
PRIMARY_CLASSIFIER = "extra_trees"

NORMALISATION_ORDER = [
    "none",
    "calibration_median",
    "calibration_median_iqr",
    "jackknife_median",
]

NORMALISATION_LABELS = {
    "none": "None",
    "calibration_median": "Calibration\nmedian",
    "calibration_median_iqr": "Calibration\nmedian–IQR",
    "jackknife_median": "Jackknife\nmedian",
}

# Colour-blind-aware, restrained academic palette.
COLORS = {
    "none": "#7A7A7A",                    # neutral grey
    "calibration_median": "#6B8E9E",      # muted blue-grey
    "calibration_median_iqr": "#D97706",  # muted burnt orange
    "jackknife_median": "#1F4E79",        # dark navy
    "chance": "#4D4D4D",
    "connector": "#B8B8B8",
    "annotation": "#222222",
}

METRIC_LABELS = {
    "pooled_roc_auc": "ROC AUC",
    "pooled_average_precision": "Average precision",
    "balanced_accuracy": "Balanced accuracy",
}

/Users/shivangayathri/Work/HI Detection/data/results/

# Files are configurable but default to the uploaded frozen filenames.
DEFAULT_FILES = {
    "pooled": "loso_pooled_metrics.csv",
    "patient": "loso_patient_metrics.csv",
    "stability": "feature_selection_stability.csv",
    "heterogeneity": "patient_heterogeneity_master.csv",
    "features": "final_feature_list.txt",
}


# -----------------------------------------------------------------------------
# Styling and input validation
# -----------------------------------------------------------------------------
def configure_style() -> None:
    """Set a consistent IEEE-style plotting theme."""
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.titlesize": 10,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.35,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_inputs(input_dir: Path, filenames: dict[str, str]) -> dict[str, object]:
    paths = {key: input_dir / value for key, value in filenames.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
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

    require_columns(
        pooled,
        {
            "evaluation_mode",
            "normalization",
            "classifier",
            "pooled_roc_auc",
            "pooled_average_precision",
            "balanced_accuracy",
        },
        "Pooled metrics",
    )
    require_columns(
        patient,
        {
            "outer_fold_patient_id",
            "evaluation_mode",
            "normalization",
            "classifier",
            "roc_auc",
        },
        "Patient metrics",
    )
    require_columns(
        stability,
        {
            "evaluation_mode",
            "normalization",
            "classifier",
            "feature_name",
            "final_selection_frequency",
        },
        "Feature stability",
    )
    require_columns(
        heterogeneity,
        {
            "patient_id",
            "jackknife_roc_auc",
            "calibration_iqr_roc_auc",
            "review_window_fraction",
            "later_continuous_block_used",
        },
        "Patient heterogeneity",
    )

    if len(features) != 34 or len(set(features)) != 34:
        raise ValueError("The frozen feature list must contain exactly 34 unique features")

    primary_pooled = pooled[
        (pooled["evaluation_mode"] == PRIMARY_MODE)
        & (pooled["classifier"] == PRIMARY_CLASSIFIER)
    ]
    if set(primary_pooled["normalization"]) != set(NORMALISATION_ORDER):
        raise ValueError("Primary pooled metrics do not contain all four normalisations")

    primary_patient = patient[
        (patient["evaluation_mode"] == PRIMARY_MODE)
        & (patient["classifier"] == PRIMARY_CLASSIFIER)
        & patient["normalization"].isin(
            ["jackknife_median", "calibration_median_iqr"]
        )
    ]
    counts = primary_patient.groupby("normalization")["outer_fold_patient_id"].nunique()
    if not (counts == 20).all():
        raise ValueError("Each primary patient-level strategy must contain 20 patients")

    return {
        "pooled": pooled,
        "patient": patient,
        "stability": stability,
        "heterogeneity": heterogeneity,
        "features": features,
    }


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.svg")
    fig.savefig(output_dir / f"{stem}.png", dpi=600)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


# -----------------------------------------------------------------------------
# Figure 4: pooled and patient-level discrimination
# -----------------------------------------------------------------------------
def create_figure_4(
    pooled: pd.DataFrame,
    patient: pd.DataFrame,
    output_dir: Path,
) -> None:
    primary_pooled = pooled[
        (pooled["evaluation_mode"] == PRIMARY_MODE)
        & (pooled["classifier"] == PRIMARY_CLASSIFIER)
    ].copy()
    primary_pooled["normalization"] = pd.Categorical(
        primary_pooled["normalization"],
        categories=NORMALISATION_ORDER,
        ordered=True,
    )
    primary_pooled = primary_pooled.sort_values("normalization")

    metric_long = primary_pooled.melt(
        id_vars=["normalization"],
        value_vars=list(METRIC_LABELS),
        var_name="metric",
        value_name="value",
    )
    metric_long["metric_label"] = metric_long["metric"].map(METRIC_LABELS)
    metric_long["normalisation_label"] = metric_long["normalization"].map(
        NORMALISATION_LABELS
    )

    primary_patient = patient[
        (patient["evaluation_mode"] == PRIMARY_MODE)
        & (patient["classifier"] == PRIMARY_CLASSIFIER)
        & patient["normalization"].isin(
            ["jackknife_median", "calibration_median_iqr"]
        )
    ].copy()
    patient_wide = primary_patient.pivot(
        index="outer_fold_patient_id",
        columns="normalization",
        values="roc_auc",
    ).reset_index()
    patient_wide = patient_wide.sort_values(
        ["jackknife_median", "outer_fold_patient_id"], ascending=[False, True]
    )

    pooled_lookup = primary_pooled.set_index("normalization")["pooled_roc_auc"]

    fig = plt.figure(figsize=(7.25, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    # Panel A: grouped point plot for three pooled metrics.
    x_positions = np.arange(len(NORMALISATION_ORDER))
    offsets = {
        "ROC AUC": -0.18,
        "Average precision": 0.00,
        "Balanced accuracy": 0.18,
    }
    markers = {
        "ROC AUC": "o",
        "Average precision": "s",
        "Balanced accuracy": "D",
    }
    for metric_label in METRIC_LABELS.values():
        subset = metric_long[metric_long["metric_label"] == metric_label].copy()
        subset = subset.set_index("normalization").reindex(NORMALISATION_ORDER).reset_index()
        for i, row in subset.iterrows():
            norm = str(row["normalization"])
            ax_a.scatter(
                x_positions[i] + offsets[metric_label],
                row["value"],
                s=34,
                marker=markers[metric_label],
                color=COLORS[norm],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
    ax_a.axhline(0.5, color=COLORS["chance"], linestyle="--", linewidth=0.8)
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels([NORMALISATION_LABELS[n] for n in NORMALISATION_ORDER])
    ax_a.set_ylim(0.45, 0.88)
    ax_a.set_ylabel("Pooled performance")
    ax_a.set_title("Primary common-row Extra Trees performance")
    ax_a.grid(axis="x", visible=False)
    metric_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[label],
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=6,
            label=label,
        )
        for label in METRIC_LABELS.values()
    ]
    ax_a.legend(handles=metric_handles, frameon=False, ncol=3, loc="lower right")
    add_panel_label(ax_a, "A")

    # Panel B: paired patient AUC comparison.
    y = np.arange(len(patient_wide))
    for idx, row in patient_wide.reset_index(drop=True).iterrows():
        ax_b.plot(
            [row["jackknife_median"], row["calibration_median_iqr"]],
            [idx, idx],
            color=COLORS["connector"],
            linewidth=0.8,
            zorder=1,
        )
    ax_b.scatter(
        patient_wide["jackknife_median"],
        y,
        s=22,
        color=COLORS["jackknife_median"],
        edgecolor="white",
        linewidth=0.45,
        label="Jackknife median",
        zorder=3,
    )
    ax_b.scatter(
        patient_wide["calibration_median_iqr"],
        y,
        s=22,
        color=COLORS["calibration_median_iqr"],
        edgecolor="white",
        linewidth=0.45,
        label="Calibration median–IQR",
        zorder=3,
    )
    ax_b.axvline(0.5, color=COLORS["chance"], linestyle="--", linewidth=0.8)
    ax_b.set_xlim(0.25, 1.02)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(patient_wide["outer_fold_patient_id"].astype(str))
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Patient ROC AUC")
    ax_b.set_ylabel("Held-out patient")
    ax_b.set_title("Paired patient-level discrimination")
    ax_b.legend(frameon=False, loc="lower right")
    add_panel_label(ax_b, "B")

    # Panel C: distributions with pooled AUC markers.
    distribution = primary_patient.copy()
    distribution["Strategy"] = distribution["normalization"].map(
        {
            "jackknife_median": "Jackknife median",
            "calibration_median_iqr": "Calibration median–IQR",
        }
    )
    strategy_order = ["Jackknife median", "Calibration median–IQR"]
    strategy_palette = {
        "Jackknife median": COLORS["jackknife_median"],
        "Calibration median–IQR": COLORS["calibration_median_iqr"],
    }
    sns.violinplot(
        data=distribution,
        x="Strategy",
        y="roc_auc",
        order=strategy_order,
        palette=strategy_palette,
        inner=None,
        cut=0,
        linewidth=0.8,
        saturation=0.85,
        ax=ax_c,
        legend=False,
    )
    sns.boxplot(
        data=distribution,
        x="Strategy",
        y="roc_auc",
        order=strategy_order,
        width=0.24,
        showfliers=False,
        boxprops={"facecolor": "white", "zorder": 3},
        whiskerprops={"linewidth": 0.8},
        medianprops={"color": "#111111", "linewidth": 1.1},
        ax=ax_c,
    )
    sns.stripplot(
        data=distribution,
        x="Strategy",
        y="roc_auc",
        order=strategy_order,
        palette=strategy_palette,
        size=3.2,
        alpha=0.75,
        jitter=0.12,
        edgecolor="white",
        linewidth=0.35,
        ax=ax_c,
        legend=False,
    )
    pooled_points = [
        pooled_lookup["jackknife_median"],
        pooled_lookup["calibration_median_iqr"],
    ]
    ax_c.scatter(
        [0, 1],
        pooled_points,
        marker="*",
        s=95,
        color=[COLORS["jackknife_median"], COLORS["calibration_median_iqr"]],
        edgecolor="black",
        linewidth=0.55,
        zorder=6,
        label="Pooled ROC AUC",
    )
    ax_c.axhline(0.5, color=COLORS["chance"], linestyle="--", linewidth=0.8)
    ax_c.set_ylim(0.25, 1.03)
    ax_c.set_xlabel("")
    ax_c.set_ylabel("ROC AUC")
    ax_c.set_title("Patient-level distributions and pooled AUC")
    ax_c.tick_params(axis="x", rotation=12)
    ax_c.legend(frameon=False, loc="lower left")
    add_panel_label(ax_c, "C")

    save_figure(fig, output_dir, "Figure_4_primary_and_patient_discrimination")


# -----------------------------------------------------------------------------
# Figure 5: feature-selection frequency heatmap
# -----------------------------------------------------------------------------
def prettify_feature_name(name: str) -> str:
    replacements = {
        "mean_absolute_value": "Mean absolute value",
        "slope_sign_change_count": "Slope sign changes",
        "approximate_entropy": "Approximate entropy",
        "permutation_entropy": "Permutation entropy",
        "fuzzy_entropy": "Fuzzy entropy",
        "distribution_entropy": "Distribution entropy",
        "svd_entropy": "SVD entropy",
        "lempel_ziv_complexity": "Lempel–Ziv complexity",
        "hjorth_mobility": "Hjorth mobility",
        "hjorth_complexity": "Hjorth complexity",
        "katz_fractal_dimension": "Katz fractal dimension",
        "higuchi_fractal_dimension": "Higuchi fractal dimension",
        "detrended_fluctuation_analysis": "Detrended fluctuation analysis",
        "mean_frequency": "Mean frequency",
        "median_frequency": "Median frequency",
        "spectral_edge_frequency_95": "Spectral edge frequency 95%",
        "spectral_entropy": "Spectral entropy",
        "root_mean_square": "Root mean square",
        "peak_to_peak": "Peak-to-peak",
        "waveform_length": "Waveform length",
        "zero_crossing_count": "Zero crossings",
    }
    return replacements.get(name, name.replace("_", " ").replace("SampEn", " SampEn").strip().title())


def create_figure_5(
    stability: pd.DataFrame,
    features: list[str],
    output_dir: Path,
) -> None:
    subset = stability[
        (stability["evaluation_mode"] == PRIMARY_MODE)
        & (stability["classifier"] == PRIMARY_CLASSIFIER)
        & stability["normalization"].isin(NORMALISATION_ORDER)
    ].copy()

    pivot = subset.pivot(
        index="feature_name",
        columns="normalization",
        values="final_selection_frequency",
    ).reindex(index=features, columns=NORMALISATION_ORDER)

    if pivot.isna().any().any():
        raise ValueError("Feature-stability heatmap contains missing configuration values")

    # Rank features by average selection frequency, using frozen order as tie-break.
    order_table = pivot.copy()
    order_table["mean_frequency"] = order_table.mean(axis=1)
    order_table["frozen_order"] = np.arange(len(order_table))
    ordered_features = order_table.sort_values(
        ["mean_frequency", "frozen_order"], ascending=[False, True]
    ).index.tolist()
    pivot = pivot.loc[ordered_features]
    pivot.index = [prettify_feature_name(name) for name in pivot.index]
    pivot.columns = [
        "None",
        "Calibration median",
        "Calibration median–IQR",
        "Jackknife median",
    ]

    fig, ax = plt.subplots(figsize=(6.5, 9.2))
    cmap = sns.light_palette(COLORS["jackknife_median"], as_cmap=True)
    sns.heatmap(
        pivot,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 6.3},
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Final selection frequency", "shrink": 0.72},
        ax=ax,
    )
    ax.set_xlabel("Patient-level normalisation strategy")
    ax.set_ylabel("Frozen predictor")
    ax.set_title("Feature-selection stability across 20 LOSO folds")
    ax.tick_params(axis="x", rotation=24)
    ax.tick_params(axis="y", rotation=0)

    # Mark the 0.75 stability threshold in the colour bar text/caption context.
    ax.text(
        1.0,
        -0.075,
        "Values ≥0.75 indicate selection in at least 15 of 20 folds.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )

    save_figure(fig, output_dir, "Figure_5_feature_selection_stability")


# -----------------------------------------------------------------------------
# Figure 6: ranked patient heterogeneity
# -----------------------------------------------------------------------------
def create_figure_6(
    heterogeneity: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = heterogeneity.copy()
    frame["patient_id"] = frame["patient_id"].astype(int)
    frame = frame.sort_values(
        ["jackknife_roc_auc", "patient_id"], ascending=[False, True]
    ).reset_index(drop=True)
    y = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(6.9, 7.0))

    for idx, row in frame.iterrows():
        ax.plot(
            [row["jackknife_roc_auc"], row["calibration_iqr_roc_auc"]],
            [idx, idx],
            color=COLORS["connector"],
            linewidth=0.9,
            zorder=1,
        )

    ax.scatter(
        frame["jackknife_roc_auc"],
        y,
        s=28,
        color=COLORS["jackknife_median"],
        edgecolor="white",
        linewidth=0.5,
        label="Jackknife median",
        zorder=3,
    )
    ax.scatter(
        frame["calibration_iqr_roc_auc"],
        y,
        s=28,
        color=COLORS["calibration_median_iqr"],
        edgecolor="white",
        linewidth=0.5,
        label="Calibration median–IQR",
        zorder=3,
    )

    # Restrained annotation markers.
    high_qc = frame["review_window_fraction"] >= 0.50
    later_block = frame["later_continuous_block_used"].astype(bool)
    ax.scatter(
        np.full(high_qc.sum(), 1.012),
        y[high_qc.to_numpy()],
        marker="s",
        s=22,
        facecolor="none",
        edgecolor="#6F6F6F",
        linewidth=0.8,
        clip_on=False,
        label="High QC-review burden",
        zorder=4,
    )
    ax.scatter(
        np.full(later_block.sum(), 1.035),
        y[later_block.to_numpy()],
        marker="^",
        s=28,
        facecolor="#FFFFFF",
        edgecolor="#222222",
        linewidth=0.8,
        clip_on=False,
        label="Later calibration block",
        zorder=4,
    )

    # Label only the two below-chance difficult patients.
    for patient_id in [30851, 27245]:
        match = frame.index[frame["patient_id"] == patient_id]
        if len(match) == 1:
            idx = int(match[0])
            row = frame.loc[idx]
            anchor_x = min(row["jackknife_roc_auc"], row["calibration_iqr_roc_auc"])
            ax.annotate(
                str(patient_id),
                xy=(anchor_x, idx),
                xytext=(-34, 0),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=7.2,
                color=COLORS["annotation"],
                arrowprops={"arrowstyle": "-", "lw": 0.55, "color": "#777777"},
            )

    ax.axvline(0.5, color=COLORS["chance"], linestyle="--", linewidth=0.85)
    ax.text(
        0.505,
        -0.65,
        "Chance",
        fontsize=7,
        color=COLORS["chance"],
        va="bottom",
    )
    ax.set_xlim(0.24, 1.065)
    ax.set_ylim(len(frame) - 0.3, -0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["patient_id"].astype(str))
    ax.set_xlabel("Patient ROC AUC")
    ax.set_ylabel("Held-out patient (ranked by jackknife AUC)")
    ax.set_title("Patient-level performance heterogeneity")
    ax.grid(axis="y", visible=False)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        loc="lower right",
        borderaxespad=0.2,
    )

    ax.text(
        1.0,
        -0.075,
        "Clinical and QC markers are descriptive annotations only.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )

    save_figure(fig, output_dir, "Figure_6_patient_heterogeneity")


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir,
        help="Directory containing the frozen result files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results_figures",
        help="Directory in which PDF, SVG, and PNG figures will be written.",
    )
    parser.add_argument("--pooled-file", default=DEFAULT_FILES["pooled"])
    parser.add_argument("--patient-file", default=DEFAULT_FILES["patient"])
    parser.add_argument("--stability-file", default=DEFAULT_FILES["stability"])
    parser.add_argument("--heterogeneity-file", default=DEFAULT_FILES["heterogeneity"])
    parser.add_argument("--feature-file", default=DEFAULT_FILES["features"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    filenames = {
        "pooled": args.pooled_file,
        "patient": args.patient_file,
        "stability": args.stability_file,
        "heterogeneity": args.heterogeneity_file,
        "features": args.feature_file,
    }
    data = load_inputs(args.input_dir, filenames)

    create_figure_4(data["pooled"], data["patient"], args.output_dir)
    create_figure_5(data["stability"], data["features"], args.output_dir)
    create_figure_6(data["heterogeneity"], args.output_dir)

    print(f"Figures written to: {args.output_dir.resolve()}")
    for stem in [
        "Figure_4_primary_and_patient_discrimination",
        "Figure_5_feature_selection_stability",
        "Figure_6_patient_heterogeneity",
    ]:
        print(f"  {stem}.pdf / .svg / .png")


if __name__ == "__main__":
    main()
