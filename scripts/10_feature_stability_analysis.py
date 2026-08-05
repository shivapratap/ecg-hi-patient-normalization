"""Describe feature-selection stability across patient-level LOSO folds.

This is post hoc descriptive analysis only. It reads the existing LOSO audit
files, quantifies selection frequencies and fold-set similarity, and creates
publication-ready figures without refitting models or changing predictors.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"
FEATURE_LIST_PATH = PROJECT_ROOT / "data" / "modelling" / "final_feature_list.txt"

EXPECTED_FOLDS = 20
EXPECTED_MODEL_RUNS = 480
EXPECTED_FEATURE_COUNT = 34
STABILITY_THRESHOLDS = {"very_high": 0.90, "high": 0.75, "moderate": 0.50}
PRIMARY_EVALUATION_MODE = "common_rows"
PRIMARY_CONFIGURATIONS = (
    ("jackknife_median", "extra_trees"),
    ("calibration_median_iqr", "extra_trees"),
    ("calibration_median", "extra_trees"),
)
REMOVED_STAGES = {"variance_filter", "top_k", "correlation_pruning", "retained"}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the four LOSO audit inputs and the ordered frozen feature list."""
    selected = pd.read_csv(RESULTS_ROOT / "loso_selected_features.csv")
    pruning = pd.read_csv(RESULTS_ROOT / "loso_correlation_pruning.csv")
    fold_audit = pd.read_csv(RESULTS_ROOT / "loso_fold_audit.csv")
    run_summary = pd.read_csv(RESULTS_ROOT / "loso_run_summary.csv")
    features = [
        line.strip()
        for line in FEATURE_LIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_inputs(selected, pruning, fold_audit, run_summary, features)
    return selected, pruning, fold_audit, run_summary, features


def validate_inputs(
    selected: pd.DataFrame,
    pruning: pd.DataFrame,
    fold_audit: pd.DataFrame,
    run_summary: pd.DataFrame,
    features: list[str],
) -> None:
    """Validate run counts, feature coverage, and stage-status logic."""
    if int(run_summary.loc[0, "completed_model_runs"]) != EXPECTED_MODEL_RUNS:
        raise ValueError("LOSO run summary does not report 480 completed runs")
    if len(selected) != EXPECTED_MODEL_RUNS * EXPECTED_FEATURE_COUNT:
        raise ValueError("Unexpected selected-feature audit row count")
    if len(features) != EXPECTED_FEATURE_COUNT or len(set(features)) != len(features):
        raise ValueError("Frozen feature list must contain exactly 34 unique features")
    if set(selected["feature_name"]) != set(features):
        raise ValueError("Selected-feature audit does not match frozen feature set")

    key = ["outer_fold_patient_id", "evaluation_mode", "normalization", "classifier"]
    if selected.duplicated(key + ["feature_name"]).any():
        raise ValueError("Duplicate feature-audit keys found")
    run_counts = selected.groupby(key)["feature_name"].nunique()
    if not (run_counts == EXPECTED_FEATURE_COUNT).all():
        raise ValueError("Every model run must contain exactly 34 features")
    if selected[key].drop_duplicates().shape[0] != EXPECTED_MODEL_RUNS:
        raise ValueError("Unexpected number of model configurations")
    if selected["outer_fold_patient_id"].nunique() != EXPECTED_FOLDS:
        raise ValueError("Unexpected number of outer folds")
    if selected["evaluation_mode"].nunique() != 2:
        raise ValueError("Unexpected evaluation-mode count")
    if selected["normalization"].nunique() != 4:
        raise ValueError("Unexpected normalization count")
    if selected["classifier"].nunique() != 3:
        raise ValueError("Unexpected classifier count")

    required = {
        "outer_fold_patient_id", "evaluation_mode", "normalization", "classifier",
        "feature_name", "initial_frozen_order", "ranking_position",
        "model_input_position", "feature_importance", "survived_variance_filter",
        "survived_top_k", "survived_correlation_pruning", "removed_at_stage",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Selected-feature audit is missing columns: {missing}")
    if not set(selected["removed_at_stage"].dropna()).issubset(REMOVED_STAGES):
        raise ValueError("Unexpected removed_at_stage value")

    for _, row in selected.iterrows():
        variance = bool(row["survived_variance_filter"])
        top_k = bool(row["survived_top_k"])
        correlation = bool(row["survived_correlation_pruning"])
        stage = row["removed_at_stage"]
        valid = (
            (stage == "retained" and variance and top_k and correlation)
            or (stage == "correlation_pruning" and variance and top_k and not correlation)
            or (stage == "top_k" and variance and not top_k and not correlation)
            or (stage == "variance_filter" and not variance and not top_k and not correlation)
        )
        if not valid:
            raise ValueError(f"Inconsistent feature-stage status for {row['feature_name']}")

    if len(fold_audit) != EXPECTED_MODEL_RUNS:
        raise ValueError("Fold audit does not contain 480 rows")
    if not pruning.empty:
        pruning_key = [
            "outer_fold_patient_id", "evaluation_mode", "normalization", "classifier",
            "removed_feature", "retained_feature",
        ]
        if pruning.duplicated(pruning_key).any():
            raise ValueError("Duplicate correlation-pruning rows found")


def stability_category(frequency: float) -> str:
    """Classify a final-selection frequency using configured thresholds."""
    if frequency >= STABILITY_THRESHOLDS["very_high"]:
        return "very_high"
    if frequency >= STABILITY_THRESHOLDS["high"]:
        return "high"
    if frequency >= STABILITY_THRESHOLDS["moderate"]:
        return "moderate"
    return "low"


def calculate_feature_stability(
    selected: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Calculate per-feature selection frequencies and ranking summaries."""
    key = ["evaluation_mode", "normalization", "classifier"]
    rows = []
    for config, group in selected.groupby(key, sort=True):
        for feature in features:
            feature_rows = group[group["feature_name"] == feature]
            ranking = pd.to_numeric(feature_rows["ranking_position"], errors="coerce").dropna()
            model_position = pd.to_numeric(feature_rows["model_input_position"], errors="coerce").dropna()
            importance = pd.to_numeric(feature_rows["feature_importance"], errors="coerce").dropna()
            variance_count = int(feature_rows["survived_variance_filter"].sum())
            top_k_count = int(feature_rows["survived_top_k"].sum())
            final_count = int(feature_rows["survived_correlation_pruning"].sum())
            mean_importance = importance.mean() if not importance.empty else np.nan
            std_importance = importance.std(ddof=1) if len(importance) > 1 else np.nan
            importance_cv = (
                std_importance / abs(mean_importance)
                if pd.notna(std_importance) and pd.notna(mean_importance) and mean_importance != 0
                else np.nan
            )
            rows.append(
                {
                    "evaluation_mode": config[0],
                    "normalization": config[1],
                    "classifier": config[2],
                    "feature_name": feature,
                    "fold_count": len(feature_rows),
                    "variance_survival_count": variance_count,
                    "variance_survival_frequency": variance_count / EXPECTED_FOLDS,
                    "top_k_survival_count": top_k_count,
                    "top_k_survival_frequency": top_k_count / EXPECTED_FOLDS,
                    "final_selection_count": final_count,
                    "final_selection_frequency": final_count / EXPECTED_FOLDS,
                    "variance_removal_count": int((feature_rows["removed_at_stage"] == "variance_filter").sum()),
                    "top_k_removal_count": int((feature_rows["removed_at_stage"] == "top_k").sum()),
                    "correlation_pruning_removal_count": int((feature_rows["removed_at_stage"] == "correlation_pruning").sum()),
                    "median_ranking_position": ranking.median() if not ranking.empty else np.nan,
                    "mean_ranking_position": ranking.mean() if not ranking.empty else np.nan,
                    "ranking_position_standard_deviation": ranking.std(ddof=1) if len(ranking) > 1 else np.nan,
                    "median_model_input_position": model_position.median() if not model_position.empty else np.nan,
                    "mean_model_input_position": model_position.mean() if not model_position.empty else np.nan,
                    "mean_feature_importance": mean_importance,
                    "median_feature_importance": importance.median() if not importance.empty else np.nan,
                    "feature_importance_standard_deviation": std_importance,
                    "minimum_feature_importance": importance.min() if not importance.empty else np.nan,
                    "maximum_feature_importance": importance.max() if not importance.empty else np.nan,
                    "importance_cv": importance_cv,
                    "stability_category": stability_category(final_count / EXPECTED_FOLDS),
                }
            )
    return pd.DataFrame(rows)


def final_sets_by_fold(group: pd.DataFrame) -> dict[int, set[str]]:
    """Return final retained feature sets indexed by outer-fold patient."""
    sets = {
        fold: set(rows.loc[rows["survived_correlation_pruning"], "feature_name"])
        for fold, rows in group.groupby("outer_fold_patient_id")
    }
    if len(sets) != EXPECTED_FOLDS or any(not values for values in sets.values()):
        raise ValueError("Every configuration must have 20 non-empty final feature sets")
    return sets


def calculate_set_similarity(selected: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise Jaccard and overlap similarity for each configuration."""
    rows = []
    key = ["evaluation_mode", "normalization", "classifier"]
    for config, group in selected.groupby(key, sort=True):
        fold_sets = final_sets_by_fold(group)
        for fold_1, fold_2 in combinations(sorted(fold_sets), 2):
            first = fold_sets[fold_1]
            second = fold_sets[fold_2]
            intersection = len(first & second)
            union = len(first | second)
            rows.append(
                {
                    "evaluation_mode": config[0],
                    "normalization": config[1],
                    "classifier": config[2],
                    "fold_patient_1": fold_1,
                    "fold_patient_2": fold_2,
                    "selected_feature_count_1": len(first),
                    "selected_feature_count_2": len(second),
                    "intersection_size": intersection,
                    "union_size": union,
                    "jaccard_similarity": intersection / union,
                    "overlap_coefficient": intersection / min(len(first), len(second)),
                }
            )
    return pd.DataFrame(rows)


def configuration_summary(
    stability: pd.DataFrame, similarity: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Summarize final-set size and feature stability by model configuration."""
    rows = []
    key = ["evaluation_mode", "normalization", "classifier"]
    for config, group in stability.groupby(key, sort=True):
        pairs = similarity[
            (similarity["evaluation_mode"] == config[0])
            & (similarity["normalization"] == config[1])
            & (similarity["classifier"] == config[2])
        ]
        ranked = group.sort_values(
            ["final_selection_frequency", "median_ranking_position"],
            ascending=[False, True],
        )
        best_frequency = ranked.iloc[0]["final_selection_frequency"]
        tied = ranked[ranked["final_selection_frequency"] == best_frequency]
        best = tied.sort_values(
            "feature_name", key=lambda values: values.map({name: i for i, name in enumerate(features)})
        ).iloc[0]
        rows.append(
            {
                "evaluation_mode": config[0],
                "normalization": config[1],
                "classifier": config[2],
                "fold_count": int(group["fold_count"].iloc[0]),
                "starting_feature_count": EXPECTED_FEATURE_COUNT,
                "mean_final_feature_count": group["final_selection_count"].mean(),
                "standard_deviation_final_feature_count": group["final_selection_count"].std(ddof=1),
                "minimum_final_feature_count": group["final_selection_count"].min(),
                "maximum_final_feature_count": group["final_selection_count"].max(),
                "mean_pairwise_jaccard": pairs["jaccard_similarity"].mean(),
                "median_pairwise_jaccard": pairs["jaccard_similarity"].median(),
                "standard_deviation_pairwise_jaccard": pairs["jaccard_similarity"].std(ddof=1),
                "minimum_pairwise_jaccard": pairs["jaccard_similarity"].min(),
                "maximum_pairwise_jaccard": pairs["jaccard_similarity"].max(),
                "mean_overlap_coefficient": pairs["overlap_coefficient"].mean(),
                "very_high_stability_feature_count": int((group["stability_category"] == "very_high").sum()),
                "high_or_better_feature_count": int(group["final_selection_frequency"].ge(0.75).sum()),
                "moderate_or_better_feature_count": int(group["final_selection_frequency"].ge(0.50).sum()),
                "feature_never_selected_count": int((group["final_selection_count"] == 0).sum()),
                "most_stable_feature": best["feature_name"],
                "most_stable_feature_frequency": best_frequency,
            }
        )
    return pd.DataFrame(rows)


def primary_comparison(stability: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compare final-selection frequencies across the three primary configurations."""
    lookup = stability[
        (stability["evaluation_mode"] == PRIMARY_EVALUATION_MODE)
        & stability[["normalization", "classifier"]].apply(tuple, axis=1).isin(PRIMARY_CONFIGURATIONS)
    ]
    rows = []
    for feature in features:
        frequencies = {}
        for normalization, classifier in PRIMARY_CONFIGURATIONS:
            match = lookup[
                (lookup["feature_name"] == feature)
                & (lookup["normalization"] == normalization)
                & (lookup["classifier"] == classifier)
            ]
            frequencies[normalization] = float(match["final_selection_frequency"].iloc[0])
        rows.append(
            {
                "feature_name": feature,
                "frozen_feature_order": features.index(feature),
                "jackknife_final_selection_frequency": frequencies["jackknife_median"],
                "calibration_median_iqr_final_selection_frequency": frequencies["calibration_median_iqr"],
                "calibration_median_final_selection_frequency": frequencies["calibration_median"],
                "maximum_selection_frequency": max(frequencies.values()),
                "minimum_selection_frequency": min(frequencies.values()),
                "frequency_range": max(frequencies.values()) - min(frequencies.values()),
                "stable_across_all_three": all(value >= 0.75 for value in frequencies.values()),
                "jackknife_specific": frequencies["jackknife_median"] >= 0.75 and frequencies["calibration_median_iqr"] < 0.50 and frequencies["calibration_median"] < 0.50,
                "calibration_iqr_specific": frequencies["calibration_median_iqr"] >= 0.75 and frequencies["jackknife_median"] < 0.50 and frequencies["calibration_median"] < 0.50,
                "calibration_median_specific": frequencies["calibration_median"] >= 0.75 and frequencies["jackknife_median"] < 0.50 and frequencies["calibration_median_iqr"] < 0.50,
            }
        )
    return pd.DataFrame(rows)


def pruning_summaries(
    pruning: pd.DataFrame, selected: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize correlation-pruning pairs and removed-feature competitors."""
    pair_rows = []
    competitor_rows = []
    key = ["evaluation_mode", "normalization", "classifier"]
    for config, group in selected.groupby(key, sort=True):
        if not pruning.empty:
            subset = pruning[
                (pruning["evaluation_mode"] == config[0])
                & (pruning["normalization"] == config[1])
                & (pruning["classifier"] == config[2])
            ]
        else:
            subset = pruning
        for (removed, retained), pair in subset.groupby(["removed_feature", "retained_feature"]):
            values = pair["absolute_spearman_correlation"]
            pair_rows.append(
                {
                    "evaluation_mode": config[0],
                    "normalization": config[1],
                    "classifier": config[2],
                    "removed_feature": removed,
                    "retained_feature": retained,
                    "pruning_count": len(pair),
                    "pruning_frequency_out_of_20_folds": len(pair) / EXPECTED_FOLDS,
                    "mean_absolute_spearman_correlation": values.mean(),
                    "median_absolute_spearman_correlation": values.median(),
                    "minimum_absolute_spearman_correlation": values.min(),
                    "maximum_absolute_spearman_correlation": values.max(),
                }
            )
        for removed, removed_group in subset.groupby("removed_feature"):
            competitors = removed_group["retained_feature"].value_counts()
            most_common = competitors.index[0]
            total = int(competitors.sum())
            competitor_rows.append(
                {
                    "evaluation_mode": config[0],
                    "normalization": config[1],
                    "classifier": config[2],
                    "feature_name": removed,
                    "times_removed_by_correlation": total,
                    "most_common_retained_competitor": most_common,
                    "competitor_count": int(competitors.size),
                    "competitor_fraction_of_feature_pruning_events": int(competitors.iloc[0]) / total,
                }
            )
    pair_columns = [
        "evaluation_mode", "normalization", "classifier", "removed_feature",
        "retained_feature", "pruning_count", "pruning_frequency_out_of_20_folds",
        "mean_absolute_spearman_correlation", "median_absolute_spearman_correlation",
        "minimum_absolute_spearman_correlation", "maximum_absolute_spearman_correlation",
    ]
    competitor_columns = [
        "evaluation_mode", "normalization", "classifier", "feature_name",
        "times_removed_by_correlation", "most_common_retained_competitor",
        "competitor_count", "competitor_fraction_of_feature_pruning_events",
    ]
    return pd.DataFrame(pair_rows, columns=pair_columns), pd.DataFrame(competitor_rows, columns=competitor_columns)


def create_figures(
    stability: pd.DataFrame, similarity: pd.DataFrame, primary: pd.DataFrame, features: list[str]
) -> None:
    """Create the three requested matplotlib-only publication figures."""
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    heat = stability[
        (stability["evaluation_mode"] == "common_rows")
        & (stability["classifier"] == "extra_trees")
    ]
    pivot = heat.pivot(index="feature_name", columns="normalization", values="final_selection_frequency")
    pivot = pivot.reindex(columns=["none", "jackknife_median", "calibration_median_iqr", "calibration_median"])
    pivot["_order"] = pivot["jackknife_median"].fillna(0)
    pivot = pivot.sort_values(["_order", "calibration_median_iqr"], ascending=False).drop(columns="_order")
    figure, axis = plt.subplots(figsize=(8, 11))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), ["None", "Jackknife", "Calib. IQR", "Calib. median"], rotation=30, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            axis.text(column_index, row_index, f"{pivot.iloc[row_index, column_index]:.2f}", ha="center", va="center", color="white" if pivot.iloc[row_index, column_index] < 0.55 else "black", fontsize=7)
    axis.set_title("Final feature-selection frequency\nCommon rows, Extra Trees")
    figure.colorbar(image, ax=axis, label="Selection frequency")
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "feature_stability_heatmap_common_extra_trees.png", dpi=300)
    plt.close(figure)

    primary_plot = primary[
        (primary["jackknife_final_selection_frequency"] >= 0.50)
        | (primary["calibration_median_iqr_final_selection_frequency"] >= 0.50)
        | (primary["calibration_median_final_selection_frequency"] >= 0.50)
    ].copy()
    primary_plot = primary_plot.sort_values("frozen_feature_order", ascending=False)
    figure, axis = plt.subplots(figsize=(10, max(5, len(primary_plot) * 0.28)))
    y = np.arange(len(primary_plot))
    height = 0.24
    axis.barh(y - height, primary_plot["jackknife_final_selection_frequency"], height, label="Jackknife median")
    axis.barh(y, primary_plot["calibration_median_iqr_final_selection_frequency"], height, label="Calibration median IQR")
    axis.barh(y + height, primary_plot["calibration_median_final_selection_frequency"], height, label="Calibration median")
    axis.set_yticks(y, primary_plot["feature_name"])
    axis.set_xlim(0, 1)
    axis.axvline(0.75, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Final selection frequency")
    axis.set_title("Primary Extra Trees feature-selection stability")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "feature_selection_frequency_primary.png", dpi=300)
    plt.close(figure)

    common_similarity = similarity[similarity["evaluation_mode"] == "common_rows"].copy()
    common_similarity["configuration"] = common_similarity.apply(
        lambda row: f"{row['normalization']}\n{row['classifier']}", axis=1
    )
    configurations = sorted(common_similarity["configuration"].unique())
    values = [common_similarity.loc[common_similarity["configuration"] == config, "jaccard_similarity"] for config in configurations]
    figure, axis = plt.subplots(figsize=(13, 7))
    axis.boxplot(values, labels=configurations, showfliers=False)
    axis.set_ylabel("Pairwise Jaccard similarity")
    axis.set_xlabel("Configuration")
    axis.set_title("Final feature-set similarity across LOSO folds\nCommon rows")
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "feature_set_jaccard_distribution.png", dpi=300)
    plt.close(figure)


def main() -> None:
    """Run validation, stability summaries, figures, and output checks."""
    print("Loading and validating input files", flush=True)
    selected, pruning, fold_audit, run_summary, features = load_inputs()
    print("Calculating feature-level stability", flush=True)
    stability = calculate_feature_stability(selected, features)
    print("Calculating fold-set similarity", flush=True)
    similarity = calculate_set_similarity(selected)
    print("Summarizing correlation pruning", flush=True)
    configuration = configuration_summary(stability, similarity, features)
    primary = primary_comparison(stability, features)
    pruning_summary, competitor_summary = pruning_summaries(pruning, selected, features)
    print("Creating figures", flush=True)
    create_figures(stability, similarity, primary, features)

    # Stability is summarized across folds, so it has one row per
    # feature/configuration rather than one row per feature/fold.
    expected_stability_rows = 2 * 4 * 3 * EXPECTED_FEATURE_COUNT
    expected_similarity_rows = 2 * 4 * 3 * (EXPECTED_FOLDS * (EXPECTED_FOLDS - 1) // 2)
    if len(stability) != expected_stability_rows:
        raise ValueError("Unexpected feature-stability row count")
    if len(similarity) != expected_similarity_rows:
        raise ValueError("Unexpected feature-set similarity row count")
    if len(configuration) != 24 or len(primary) != EXPECTED_FEATURE_COUNT:
        raise ValueError("Unexpected summary row count")
    if not stability["final_selection_frequency"].between(0, 1).all():
        raise ValueError("Invalid selection frequency")
    if not similarity[["jaccard_similarity", "overlap_coefficient"]].apply(lambda column: column.between(0, 1).all()).all():
        raise ValueError("Invalid set-similarity value")
    if not np.isfinite(similarity[["jaccard_similarity", "overlap_coefficient"]].to_numpy()).all():
        raise ValueError("Non-finite set-similarity value")
    print("Validating outputs", flush=True)

    summary = pd.DataFrame([{
        "source_feature_audit_rows": len(selected),
        "expected_feature_audit_rows": EXPECTED_MODEL_RUNS * EXPECTED_FEATURE_COUNT,
        "source_model_runs": selected[["outer_fold_patient_id", "evaluation_mode", "normalization", "classifier"]].drop_duplicates().shape[0],
        "expected_model_runs": EXPECTED_MODEL_RUNS,
        "fold_count": EXPECTED_FOLDS,
        "frozen_feature_count": EXPECTED_FEATURE_COUNT,
        "feature_stability_rows": len(stability),
        "expected_feature_stability_rows": expected_stability_rows,
        "pairwise_similarity_rows": len(similarity),
        "expected_pairwise_similarity_rows": expected_similarity_rows,
        "configuration_summary_rows": len(configuration),
        "primary_comparison_rows": len(primary),
        "all_feature_audit_keys_unique": not selected.duplicated(["outer_fold_patient_id", "evaluation_mode", "normalization", "classifier", "feature_name"]).any(),
        "all_model_runs_have_34_features": bool((selected.groupby(["outer_fold_patient_id", "evaluation_mode", "normalization", "classifier"])["feature_name"].nunique() == EXPECTED_FEATURE_COUNT).all()),
        "all_stage_status_relationships_valid": True,
        "all_configurations_have_20_folds": bool((selected.groupby(["evaluation_mode", "normalization", "classifier"])["outer_fold_patient_id"].nunique() == EXPECTED_FOLDS).all()),
        "all_pairwise_comparison_counts_valid": bool((similarity.groupby(["evaluation_mode", "normalization", "classifier"]).size() == EXPECTED_FOLDS * (EXPECTED_FOLDS - 1) // 2).all()),
        "all_outputs_finite_where_expected": True,
    }])

    print("Writing result files", flush=True)
    stability.to_csv(RESULTS_ROOT / "feature_selection_stability.csv", index=False)
    similarity.to_csv(RESULTS_ROOT / "feature_set_similarity.csv", index=False)
    configuration.to_csv(RESULTS_ROOT / "feature_stability_configuration_summary.csv", index=False)
    primary.to_csv(RESULTS_ROOT / "primary_feature_stability_comparison.csv", index=False)
    pruning_summary.to_csv(RESULTS_ROOT / "correlation_pruning_stability.csv", index=False)
    competitor_summary.to_csv(RESULTS_ROOT / "feature_competitor_summary.csv", index=False)
    summary.to_csv(RESULTS_ROOT / "feature_stability_run_summary.csv", index=False)
    print("Feature stability analysis completed", flush=True)


if __name__ == "__main__":
    main()
