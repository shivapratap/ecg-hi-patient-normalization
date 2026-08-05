"""Explore patient-level heterogeneity in the ECG LOSO results.

This script is descriptive only. It does not fit models, alter model inputs,
remove patients, or make causal claims. It combines existing LOSO outputs,
feature values, calibration information, QC summaries, and optional clinical
metadata into reproducible patient-level tables and figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"
MODEL_ROOT = PROJECT_ROOT / "data" / "modelling"
NORMALIZATION_ROOT = PROJECT_ROOT / "data" / "normalization"
QUALITY_ROOT = PROJECT_ROOT / "data" / "quality"

PATIENT_COUNT = 20
PRIMARY_CONFIGS = {
    "jackknife": ("common_rows", "jackknife_median", "extra_trees"),
    "calibration_iqr": ("common_rows", "calibration_median_iqr", "extra_trees"),
}
STABILITY_THRESHOLD = 0.75
MIN_GROUP_SIZE = 3
LARGE_EFFECT_THRESHOLD = 0.474

COLORS = {
    "jackknife": "#1f4e79",
    "calibration_iqr": "#d97706",
    "prehi": "#4c78a8",
    "hi": "#e45756",
    "review": "#7f7f7f",
}

sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
plt.rcParams["font.family"] = "DejaVu Sans"


def read_required_inputs() -> dict[str, pd.DataFrame | list[str]]:
    """Read required inputs and locate optional clinical metadata."""
    paths = {
        "patient_metrics": RESULTS_ROOT / "loso_patient_metrics.csv",
        "predictions": RESULTS_ROOT / "loso_predictions.csv",
        "pooled_metrics": RESULTS_ROOT / "loso_pooled_metrics.csv",
        "stability": RESULTS_ROOT / "feature_selection_stability.csv",
        "modelling": MODEL_ROOT / "clean_modelling_table.csv",
        "calibration": NORMALIZATION_ROOT / "full_calibration_rows.csv",
        "qc": QUALITY_ROOT / "window_signal_quality.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    clinical_summary_candidates = [
        PROJECT_ROOT / "data" / "clinical" / "patient_clinical_summary.csv",
        MODEL_ROOT / "patient_clinical_summary.csv",
        PROJECT_ROOT / "patient_clinical_summary.csv",
    ]
    clinical_flags_candidates = [
        PROJECT_ROOT / "data" / "clinical" / "patient_clinical_flags.csv",
        MODEL_ROOT / "patient_clinical_flags.csv",
        PROJECT_ROOT / "patient_clinical_flags.csv",
    ]
    summary_path = next((path for path in clinical_summary_candidates if path.exists()), None)
    flags_path = next((path for path in clinical_flags_candidates if path.exists()), None)

    data: dict[str, pd.DataFrame | list[str]] = {
        name: pd.read_csv(path) for name, path in paths.items()
    }
    data["features"] = [
        line.strip()
        for line in (MODEL_ROOT / "final_feature_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    data["clinical_summary"] = pd.read_csv(summary_path) if summary_path else pd.DataFrame()
    data["clinical_flags"] = pd.read_csv(flags_path) if flags_path else pd.DataFrame()
    data["clinical_summary_path"] = summary_path
    data["clinical_flags_path"] = flags_path
    return data


def validate_inputs(data: dict[str, pd.DataFrame | list[str]]) -> None:
    """Validate patient coverage, prediction values, keys, and feature count."""
    metrics = data["patient_metrics"]
    predictions = data["predictions"]
    modelling = data["modelling"]
    features = data["features"]
    expected_patients = set(metrics["outer_fold_patient_id"].unique())
    if len(expected_patients) != PATIENT_COUNT:
        raise ValueError(f"Expected 20 LOSO patients, found {len(expected_patients)}")
    if set(predictions["patient_id"].unique()) != expected_patients:
        raise ValueError("Prediction patient IDs do not match LOSO metric patient IDs")
    if set(modelling["patient_id"].unique()) != expected_patients:
        raise ValueError("Modelling-table patient IDs do not match LOSO patient IDs")
    if len(features) != 34 or len(set(features)) != 34:
        raise ValueError("final_feature_list.txt must contain exactly 34 unique features")

    key = ["outer_fold_patient_id", "evaluation_mode", "normalization", "classifier"]
    if metrics.duplicated(key).any():
        raise ValueError("LOSO patient metrics contain duplicate patient/configuration rows")
    for name, config in PRIMARY_CONFIGS.items():
        subset = metrics[
            (metrics["evaluation_mode"] == config[0])
            & (metrics["normalization"] == config[1])
            & (metrics["classifier"] == config[2])
        ]
        if len(subset) != PATIENT_COUNT:
            raise ValueError(f"{name} configuration does not have one metric row per patient")
        held_out = predictions[
            (predictions["evaluation_mode"] == config[0])
            & (predictions["normalization"] == config[1])
            & (predictions["classifier"] == config[2])
        ]
        if not np.isfinite(held_out["predicted_probability"]).all():
            raise ValueError(f"{name} predictions contain non-finite probabilities")
        if not held_out["predicted_probability"].between(0, 1).all():
            raise ValueError(f"{name} predictions fall outside [0, 1]")
        if held_out.duplicated(["patient_id", "segment_id"]).any():
            raise ValueError(f"{name} predictions contain duplicate patient/segment IDs")
        class_counts = held_out.groupby("patient_id")["Class"].nunique()
        if len(class_counts) != PATIENT_COUNT or not (class_counts == 2).all():
            raise ValueError(f"{name} does not contain both classes for every patient")

    stability = data["stability"]
    stable = stability[
        (stability["evaluation_mode"] == "common_rows")
        & (stability["classifier"] == "extra_trees")
        & stability["normalization"].isin(["jackknife_median", "calibration_median_iqr"])
    ]
    if stable.duplicated(["normalization", "feature_name"]).any():
        raise ValueError("Feature-stability input contains duplicate configuration/feature rows")

    for name in ["clinical_summary", "clinical_flags"]:
        table = data[name]
        if not table.empty and "patient_id" not in table.columns:
            raise ValueError(f"{name} must contain a patient_id column")


def select_stable_features(stability: pd.DataFrame, features: list[str]) -> list[str]:
    """Select features stable in both primary configurations, capped at 12."""
    subset = stability[
        (stability["evaluation_mode"] == "common_rows")
        & (stability["classifier"] == "extra_trees")
        & stability["normalization"].isin(["jackknife_median", "calibration_median_iqr"])
    ]
    pivot = subset.pivot(index="feature_name", columns="normalization", values="final_selection_frequency")
    qualified = pivot[(pivot >= STABILITY_THRESHOLD).all(axis=1)]
    if len(qualified) < 6:
        qualified = pivot.assign(
            minimum_frequency=pivot.min(axis=1),
            frozen_order=pivot.index.map({feature: i for i, feature in enumerate(features)}),
        ).sort_values(["minimum_frequency", "frozen_order"], ascending=[False, True]).head(6)
        selected = qualified.index.tolist()
    else:
        selected = qualified.assign(
            minimum_frequency=qualified.min(axis=1),
            frozen_order=qualified.index.map({feature: i for i, feature in enumerate(features)}),
        ).sort_values(["minimum_frequency", "frozen_order"], ascending=[False, True]).head(12).index.tolist()
    selected.sort(key=lambda feature: features.index(feature))
    return selected


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    """Return Benjamini-Hochberg adjusted p-values, preserving missingness."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    if not valid.any():
        return result
    p = values[valid].astype(float).to_numpy()
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    output = np.empty(len(p))
    output[order] = adjusted
    result.loc[valid] = output
    return result


def cliffs_delta(class_1: np.ndarray, class_0: np.ndarray) -> float:
    """Calculate Cliff's delta with positive values indicating higher HI values."""
    comparisons = class_1[:, None] - class_0[None, :]
    return float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)


def pooled_iqr(values: pd.Series) -> float:
    """Calculate the global IQR used for robust within-patient scaling."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.quantile(0.75) - numeric.quantile(0.25))


def calculate_feature_separation(
    modelling: pd.DataFrame, selected_features: list[str]
) -> pd.DataFrame:
    """Calculate patient-level effect sizes for each selected feature."""
    rows = []
    global_iqrs = {feature: pooled_iqr(modelling[feature]) for feature in selected_features}
    for patient_id, patient in modelling.groupby("patient_id", sort=True):
        for feature in selected_features:
            class_0 = pd.to_numeric(patient.loc[patient["Class"] == 0, feature], errors="coerce").dropna().to_numpy()
            class_1 = pd.to_numeric(patient.loc[patient["Class"] == 1, feature], errors="coerce").dropna().to_numpy()
            n0, n1 = len(class_0), len(class_1)
            median_0 = np.median(class_0) if n0 else np.nan
            median_1 = np.median(class_1) if n1 else np.nan
            diff = median_1 - median_0 if n0 and n1 else np.nan
            pooled_sd = np.nan
            smd = np.nan
            if n0 > 1 and n1 > 1:
                pooled_sd = np.sqrt(((n0 - 1) * np.var(class_0, ddof=1) + (n1 - 1) * np.var(class_1, ddof=1)) / (n0 + n1 - 2))
                if pooled_sd > 0 and np.isfinite(pooled_sd):
                    d = (np.mean(class_1) - np.mean(class_0)) / pooled_sd
                    correction = 1 - 3 / (4 * (n0 + n1) - 9) if (4 * (n0 + n1) - 9) > 0 else 1
                    smd = d * correction
            delta = cliffs_delta(class_1, class_0) if n0 and n1 else np.nan
            p_value = stats.mannwhitneyu(class_1, class_0, alternative="two-sided").pvalue if n0 and n1 else np.nan
            robust = diff / global_iqrs[feature] if np.isfinite(global_iqrs[feature]) and global_iqrs[feature] > 0 and np.isfinite(diff) else np.nan
            rows.append({
                "patient_id": patient_id,
                "feature_name": feature,
                "class_0_count": n0,
                "class_1_count": n1,
                "class_0_median": median_0,
                "class_1_median": median_1,
                "median_difference": diff,
                "pooled_standard_deviation": pooled_sd,
                "standardized_mean_difference": smd,
                "robust_standardized_median_difference": robust,
                "cliffs_delta": delta,
                "absolute_cliffs_delta": abs(delta) if np.isfinite(delta) else np.nan,
                "mann_whitney_u_p_value": p_value,
                "raw_p_value": p_value,
                "direction_of_change": "higher_in_HI" if diff > 0 else "higher_in_PreHI" if diff < 0 else "no_change" if np.isfinite(diff) else "not_available",
            })
    output = pd.DataFrame(rows)
    output["exploratory_fdr_q_value"] = benjamini_hochberg(output["raw_p_value"])
    return output


def spearman_row(x: pd.Series, y: pd.Series) -> tuple[int, float, float]:
    """Calculate a complete-case Spearman correlation."""
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return len(pair), np.nan, np.nan
    result = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return len(pair), float(result.statistic), float(result.pvalue)


def calculate_feature_auc_associations(
    separation: pd.DataFrame, master: pd.DataFrame
) -> pd.DataFrame:
    """Associate feature effect sizes with patient AUCs using Spearman correlation."""
    rows = []
    for feature, group in separation.groupby("feature_name", sort=False):
        merged = group.merge(master[["patient_id", "jackknife_roc_auc", "calibration_iqr_roc_auc"]], on="patient_id")
        for outcome, effect in [
            ("jackknife_roc_auc", "absolute_cliffs_delta"),
            ("calibration_iqr_roc_auc", "absolute_cliffs_delta"),
            ("jackknife_roc_auc", "robust_absolute_median_difference"),
            ("calibration_iqr_roc_auc", "robust_absolute_median_difference"),
        ]:
            if effect == "robust_absolute_median_difference":
                merged = merged.copy()
                merged[effect] = merged["robust_standardized_median_difference"].abs()
            count, rho, p_value = spearman_row(merged[outcome], merged[effect])
            rows.append({"feature_name": feature, "outcome_name": outcome, "effect_measure": effect, "patient_count": count, "spearman_rho": rho, "raw_p_value": p_value})
    output = pd.DataFrame(rows)
    output["exploratory_fdr_q_value"] = output.groupby("effect_measure")["raw_p_value"].transform(benjamini_hochberg)
    return output


def overlap_measure(values_0: pd.Series, values_1: pd.Series) -> float:
    """Use a transparent fallback overlap measure for small window samples."""
    median_0 = values_0.median()
    return float((values_1 < median_0).mean()) if len(values_1) else np.nan


def calibration_summary(calibration: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected calibration blocks without inferring missing data."""
    selected = calibration[calibration["selected_for_calibration"].astype(bool)].copy()
    rows = []
    for patient_id, group in calibration.groupby("patient_id", sort=True):
        chosen = selected[selected["patient_id"] == patient_id]
        if chosen.empty:
            rows.append({"patient_id": patient_id, "selected_calibration_start_rank": np.nan, "calibration_candidate_rows_skipped": np.nan, "calibration_recording_id": np.nan, "calibration_start_timestamp": np.nan, "calibration_end_timestamp": np.nan, "later_continuous_block_used": np.nan})
            continue
        rows.append({
            "patient_id": patient_id,
            "selected_calibration_start_rank": chosen["selected_block_start_chronological_rank"].min(),
            "calibration_candidate_rows_skipped": chosen["candidate_rows_skipped_before_selected_block"].max(),
            "calibration_recording_id": chosen["recording_id"].iloc[0],
            "calibration_start_timestamp": chosen["start_timestamp"].min(),
            "calibration_end_timestamp": chosen["start_timestamp"].max(),
            "later_continuous_block_used": bool(chosen["candidate_rows_skipped_before_selected_block"].max() > 0),
        })
    return pd.DataFrame(rows)


def build_master_table(data: dict[str, pd.DataFrame | list[str]]) -> pd.DataFrame:
    """Build one descriptive row per patient from existing outputs."""
    metrics = data["patient_metrics"]
    predictions = data["predictions"]
    modelling = data["modelling"]
    qc = data["qc"]
    calibration = data["calibration"]
    patient_ids = sorted(metrics["outer_fold_patient_id"].unique())
    rows = []
    for patient_id in patient_ids:
        row = {"patient_id": patient_id}
        source = qc[qc["patient_id"] == patient_id]
        model_rows = modelling[modelling["patient_id"] == patient_id]
        row.update({
            "total_source_windows": len(source),
            "class_0_source_windows": int((model_rows["Class"] == 0).sum()),
            "class_1_source_windows": int((model_rows["Class"] == 1).sum()),
            "unusable_windows_previously_excluded": int((source["quality_status"] == "unusable").sum()),
            "usable_window_count": int((source["quality_status"] == "usable").sum()),
            "review_window_count": int((source["quality_status"] == "review").sum()),
        })
        row["review_window_fraction"] = row["review_window_count"] / row["total_source_windows"] if row["total_source_windows"] else np.nan
        for label, config in PRIMARY_CONFIGS.items():
            metric = metrics[(metrics["outer_fold_patient_id"] == patient_id) & (metrics["evaluation_mode"] == config[0]) & (metrics["normalization"] == config[1]) & (metrics["classifier"] == config[2])].iloc[0]
            prefix = "jackknife" if label == "jackknife" else "calibration_iqr"
            for field in ["roc_auc", "average_precision", "accuracy", "sensitivity", "specificity", "balanced_accuracy"]:
                row[f"{prefix}_{field}"] = metric[field]
            pred = predictions[(predictions["patient_id"] == patient_id) & (predictions["evaluation_mode"] == config[0]) & (predictions["normalization"] == config[1]) & (predictions["classifier"] == config[2])]
            class_0 = pred.loc[pred["Class"] == 0, "predicted_probability"]
            class_1 = pred.loc[pred["Class"] == 1, "predicted_probability"]
            row[f"{prefix}_class_0_probability_median"] = class_0.median()
            row[f"{prefix}_class_1_probability_median"] = class_1.median()
            row[f"{prefix}_class_0_probability_iqr"] = class_0.quantile(.75) - class_0.quantile(.25)
            row[f"{prefix}_class_1_probability_iqr"] = class_1.quantile(.75) - class_1.quantile(.25)
            row[f"{prefix}_probability_median_difference"] = class_1.median() - class_0.median()
            row[f"{prefix}_probability_overlap_measure"] = overlap_measure(class_0, class_1)
            row[f"{prefix}_evaluated_rows"] = len(pred)
            row[f"{prefix}_class_0_evaluated_rows"] = int((pred["Class"] == 0).sum())
            row[f"{prefix}_class_1_evaluated_rows"] = int((pred["Class"] == 1).sum())
        common_pred = predictions[(predictions["patient_id"] == patient_id) & (predictions["evaluation_mode"] == "common_rows") & (predictions["normalization"] == "jackknife_median") & (predictions["classifier"] == "extra_trees")]
        row["common_evaluation_windows"] = len(common_pred)
        row["common_class_0_windows"] = int((common_pred["Class"] == 0).sum())
        row["common_class_1_windows"] = int((common_pred["Class"] == 1).sum())
        row["auc_difference_jackknife_minus_calibration"] = row["jackknife_roc_auc"] - row["calibration_iqr_roc_auc"]
        row["average_precision_difference"] = row["jackknife_average_precision"] - row["calibration_iqr_average_precision"]
        row["balanced_accuracy_difference"] = row["jackknife_balanced_accuracy"] - row["calibration_iqr_balanced_accuracy"]
        row.update(calibration_summary(calibration[calibration["patient_id"] == patient_id]).iloc[0].to_dict())
        rows.append(row)
    master = pd.DataFrame(rows)
    master = add_clinical_columns(master, data["clinical_summary"], "clinical_")
    master = add_clinical_columns(master, data["clinical_flags"], "flag_")
    master["jackknife_auc_rank"] = master["jackknife_roc_auc"].rank(method="min", ascending=False).astype(int)
    master["jackknife_auc_quartile"] = pd.qcut(master["jackknife_roc_auc"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    master["descriptive_performance_group"] = pd.cut(master["jackknife_roc_auc"], [-np.inf, .65, .85, np.inf], labels=["low", "intermediate", "high"], right=False)
    return master


def add_clinical_columns(master: pd.DataFrame, clinical: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Merge interpretable clinical fields while preserving missingness."""
    if clinical.empty:
        return master
    clinical = clinical.drop_duplicates("patient_id").copy()
    rename = {}
    for column in clinical.columns:
        if column != "patient_id" and column in master.columns:
            rename[column] = prefix + column
    return master.merge(clinical.rename(columns=rename), on="patient_id", how="left")


def clinical_variable_tables(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate available clinical continuous and grouped associations."""
    excluded = {"patient_id", "jackknife_roc_auc", "calibration_iqr_roc_auc", "auc_difference_jackknife_minus_calibration"}
    clinical_columns = [column for column in master.columns if column.startswith(("clinical_", "flag_"))]
    availability = []
    continuous_rows = []
    grouped_rows = []
    for column in clinical_columns:
        values = master[column]
        numeric = pd.to_numeric(values, errors="coerce")
        numeric_count = numeric.notna().sum()
        unique = values.dropna().nunique()
        availability.append({"variable_name": column, "available": True, "nonmissing_count": int(values.notna().sum()), "missing_count": int(values.isna().sum()), "unique_value_count": int(unique), "analysis_type": "continuous_or_ordinal" if numeric_count >= MIN_GROUP_SIZE and unique > 1 else "categorical_or_insufficient"})
        if numeric_count >= MIN_GROUP_SIZE and unique > 1:
            for outcome in ["jackknife_roc_auc", "calibration_iqr_roc_auc", "auc_difference_jackknife_minus_calibration"]:
                count, rho, p_value = spearman_row(numeric, master[outcome])
                continuous_rows.append({"variable_name": column, "outcome_name": outcome, "patient_count": count, "spearman_rho": rho, "raw_p_value": p_value, "missing_count": int(values.isna().sum())})
        categories = values.dropna()
        counts = categories.value_counts()
        if len(counts) == 2:
            adequate = counts.min() >= MIN_GROUP_SIZE
            reason = "analysed" if adequate else f"fewer than {MIN_GROUP_SIZE} patients in one group"
            for outcome in ["jackknife_roc_auc", "calibration_iqr_roc_auc", "auc_difference_jackknife_minus_calibration"]:
                if adequate:
                    groups = [master.loc[master[column] == level, outcome].dropna() for level in counts.index]
                    test = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                    u = test.statistic
                    rank_biserial = 2 * u / (len(groups[0]) * len(groups[1])) - 1
                    grouped_rows.append({"variable_name": column, "outcome_name": outcome, "analysis_status": reason, "group_1": str(counts.index[0]), "group_2": str(counts.index[1]), "group_1_count": len(groups[0]), "group_2_count": len(groups[1]), "group_1_median": groups[0].median(), "group_2_median": groups[1].median(), "median_difference_group_1_minus_group_2": groups[0].median() - groups[1].median(), "group_1_iqr": groups[0].quantile(.75) - groups[0].quantile(.25), "group_2_iqr": groups[1].quantile(.75) - groups[1].quantile(.25), "mann_whitney_u_p_value": test.pvalue, "rank_biserial_effect_size": rank_biserial})
                else:
                    grouped_rows.append({"variable_name": column, "outcome_name": outcome, "analysis_status": reason})
        elif len(counts) > 2:
            adequate = counts.min() >= MIN_GROUP_SIZE
            grouped_rows.append({"variable_name": column, "outcome_name": "jackknife_roc_auc", "analysis_status": "analysed_kruskal_wallis" if adequate else f"fewer than {MIN_GROUP_SIZE} patients in one group", "group_count": len(counts), "kruskal_wallis_p_value": stats.kruskal(*[master.loc[master[column] == level, "jackknife_roc_auc"].dropna() for level in counts.index]).pvalue if adequate else np.nan})
    availability_columns = [
        "variable_name", "available", "nonmissing_count", "missing_count",
        "unique_value_count", "analysis_type",
    ]
    continuous_columns = [
        "variable_name", "outcome_name", "patient_count", "spearman_rho",
        "raw_p_value", "missing_count", "exploratory_fdr_q_value",
    ]
    grouped_columns = [
        "variable_name", "outcome_name", "analysis_status", "group_1",
        "group_2", "group_1_count", "group_2_count", "group_1_median",
        "group_2_median", "median_difference_group_1_minus_group_2",
        "group_1_iqr", "group_2_iqr", "mann_whitney_u_p_value",
        "rank_biserial_effect_size", "group_count", "kruskal_wallis_p_value",
    ]
    availability = pd.DataFrame(availability, columns=availability_columns)
    continuous = pd.DataFrame(continuous_rows, columns=continuous_columns)
    if not continuous.empty:
        continuous["exploratory_fdr_q_value"] = continuous.groupby("outcome_name")["raw_p_value"].transform(benjamini_hochberg)
    grouped = pd.DataFrame(grouped_rows, columns=grouped_columns)
    return pd.DataFrame(availability), continuous, grouped


def choose_representatives(master: pd.DataFrame) -> pd.DataFrame:
    """Choose high, median, and low patients using deterministic AUC rules."""
    ordered = master.sort_values(["jackknife_roc_auc", "patient_id"], ascending=[False, True]).reset_index(drop=True)
    cohort_median = master["jackknife_roc_auc"].median()
    median_row = master.assign(_distance=(master["jackknife_roc_auc"] - cohort_median).abs()).sort_values(["_distance", "patient_id"]).iloc[0]
    choices = [("high", ordered.iloc[0]), ("median", median_row), ("low", ordered.sort_values(["jackknife_roc_auc", "patient_id"]).iloc[0])]
    rows = []
    for category, row in choices:
        selected = {"patient_id", "jackknife_roc_auc", "calibration_iqr_roc_auc", "common_evaluation_windows", "review_window_fraction"}
        selected.update(column for column in master.columns if column.startswith(("clinical_", "flag_")))
        output = {"patient_id": row["patient_id"], "representative_category": category, **{column: row[column] for column in selected if column in row}}
        rows.append(output)
    return pd.DataFrame(rows)


def patient_summary(master: pd.DataFrame, clinical_availability: pd.DataFrame) -> pd.DataFrame:
    """Create a concise manuscript-oriented patient heterogeneity summary."""
    def median_iqr(series: pd.Series) -> tuple[float, float]:
        return float(series.median()), float(series.quantile(.75) - series.quantile(.25))
    jack_median, jack_iqr = median_iqr(master["jackknife_roc_auc"])
    cal_median, cal_iqr = median_iqr(master["calibration_iqr_roc_auc"])
    try:
        wilcoxon = stats.wilcoxon(master["jackknife_roc_auc"], master["calibration_iqr_roc_auc"]).pvalue
    except ValueError:
        wilcoxon = np.nan
    _, rho, rho_p = spearman_row(master["jackknife_roc_auc"], master["calibration_iqr_roc_auc"])
    above_qc = master["review_window_fraction"] > master["review_window_fraction"].median()
    counts = master["descriptive_performance_group"].value_counts()
    rows = [{
        "patient_count": len(master), "jackknife_auc_median": jack_median, "jackknife_auc_iqr": jack_iqr, "jackknife_auc_minimum": master["jackknife_roc_auc"].min(), "jackknife_auc_maximum": master["jackknife_roc_auc"].max(), "calibration_iqr_auc_median": cal_median, "calibration_iqr_auc_iqr": cal_iqr, "calibration_iqr_auc_minimum": master["calibration_iqr_roc_auc"].min(), "calibration_iqr_auc_maximum": master["calibration_iqr_roc_auc"].max(), "high_performer_count": counts.get("high", 0), "intermediate_performer_count": counts.get("intermediate", 0), "low_performer_count": counts.get("low", 0), "paired_auc_difference_median": master["auc_difference_jackknife_minus_calibration"].median(), "paired_auc_wilcoxon_p_value": wilcoxon, "jackknife_calibration_auc_spearman_rho": rho, "jackknife_calibration_auc_spearman_p_value": rho_p, "patients_calibration_auc_exceeds_jackknife": int((master["auc_difference_jackknife_minus_calibration"] < 0).sum()), "patients_review_fraction_above_cohort_median": int(above_qc.sum()), "clinical_variable_count": len(clinical_availability), "clinical_variables_with_nonmissing_values": int((clinical_availability["nonmissing_count"] > 0).sum()) if not clinical_availability.empty else 0,
    }]
    return pd.DataFrame(rows)


def save_figure(figure: plt.Figure, stem: str) -> None:
    """Save a figure as 300-dpi PNG and vector PDF, then close it."""
    figure.savefig(FIGURES_ROOT / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def create_figures(master: pd.DataFrame, separation: pd.DataFrame, reps: pd.DataFrame, predictions: pd.DataFrame, selected_features: list[str]) -> list[str]:
    """Create the requested publication figures and return their stems."""
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    created = []
    ordered = master.sort_values(["jackknife_roc_auc", "patient_id"], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(7, 7))
    for index, (_, row) in enumerate(ordered.iterrows()):
        ax.plot([row["jackknife_roc_auc"], row["calibration_iqr_roc_auc"]], [index, index], color="#bdbdbd", lw=1)
    ax.scatter(ordered["jackknife_roc_auc"], range(len(ordered)), color=COLORS["jackknife"], label="Jackknife median", zorder=3)
    ax.scatter(ordered["calibration_iqr_roc_auc"], range(len(ordered)), color=COLORS["calibration_iqr"], label="Calibration median-IQR", zorder=3)
    ax.set_yticks(range(len(ordered)), ordered["patient_id"].astype(str))
    ax.set_xlabel("Patient ROC AUC"); ax.set_ylabel("Patient ID"); ax.axvline(.5, color="black", ls="--", lw=.8); ax.legend(frameon=False)
    save_figure(fig, "patient_auc_comparison"); created.append("patient_auc_comparison")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(master["jackknife_roc_auc"], master["calibration_iqr_roc_auc"], color=COLORS["jackknife"], alpha=.85)
    limits = [0, 1]; ax.plot(limits, limits, color="#777777", ls="--", lw=.9); ax.axvline(.5, color="#aaaaaa", ls=":"); ax.axhline(.5, color="#aaaaaa", ls=":")
    _, rho, _ = spearman_row(master["jackknife_roc_auc"], master["calibration_iqr_roc_auc"])
    ax.text(.04, .95, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, va="top")
    notable = pd.concat([master.nlargest(1, "auc_difference_jackknife_minus_calibration"), master.nsmallest(1, "auc_difference_jackknife_minus_calibration"), master.nlargest(1, "jackknife_roc_auc"), master.nsmallest(1, "jackknife_roc_auc")]).drop_duplicates("patient_id")
    for _, row in notable.iterrows(): ax.annotate(str(row["patient_id"]), (row["jackknife_roc_auc"], row["calibration_iqr_roc_auc"]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlim=limits, ylim=limits, xlabel="Jackknife median ROC AUC", ylabel="Calibration median-IQR ROC AUC")
    save_figure(fig, "jackknife_vs_calibration_auc"); created.append("jackknife_vs_calibration_auc")

    effects = separation.pivot(index="patient_id", columns="feature_name", values="cliffs_delta").reindex(index=ordered.patient_id, columns=selected_features)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(selected_features) * .65 + 2), 7), gridspec_kw={"width_ratios": [len(selected_features), 1]})
    sns.heatmap(effects, ax=axes[0], cmap="vlag", vmin=-1, vmax=1, center=0, annot=effects.abs().ge(LARGE_EFFECT_THRESHOLD), fmt="", cbar_kws={"label": "Cliff's delta"})
    axes[0].set_xlabel("Selected stable feature"); axes[0].set_ylabel("Patient ID")
    sns.heatmap(ordered[["jackknife_roc_auc"]].set_index(ordered.patient_id), ax=axes[1], cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "Jackknife AUC"}, yticklabels=False)
    axes[1].set_xlabel("AUC"); axes[1].set_ylabel("")
    save_figure(fig, "patient_feature_effect_heatmap"); created.append("patient_feature_effect_heatmap")

    pred_rows = []
    for _, rep in reps.iterrows():
        for label, config in PRIMARY_CONFIGS.items():
            subset = predictions[(predictions.patient_id == rep.patient_id) & (predictions.evaluation_mode == config[0]) & (predictions.normalization == config[1]) & (predictions.classifier == config[2])]
            for _, row in subset.iterrows(): pred_rows.append({"patient_id": rep.patient_id, "representative_category": rep.representative_category, "configuration": label, "Class": row["Class"], "predicted_probability": row["predicted_probability"]})
    probability_data = pd.DataFrame(pred_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, category in zip(axes, ["high", "median", "low"]):
        subset = probability_data[probability_data.representative_category == category]
        sns.violinplot(data=subset, x="Class", y="predicted_probability", hue="configuration", split=False, inner="box", palette={"jackknife": COLORS["jackknife"], "calibration_iqr": COLORS["calibration_iqr"]}, ax=ax, cut=0)
        patient = reps[reps.representative_category == category].iloc[0]
        ax.set_title(f"{category}: {patient.patient_id}\nJ={patient.jackknife_roc_auc:.2f}, C={patient.calibration_iqr_roc_auc:.2f}"); ax.set_xlabel("Class"); ax.set_ylim(0, 1)
        if ax is not axes[0]: ax.set_ylabel("")
    save_figure(fig, "representative_patient_probabilities"); created.append("representative_patient_probabilities")

    top_features = selected_features[:4]
    feature_long = data_for_feature_plot = None
    modelling = pd.read_csv(MODEL_ROOT / "clean_modelling_table.csv")
    plot_rows = modelling[modelling.patient_id.isin(reps.patient_id)][["patient_id", "Class"] + top_features].melt(id_vars=["patient_id", "Class"], var_name="feature_name", value_name="feature_value")
    fig, axes = plt.subplots(len(reps), len(top_features), figsize=(max(10, len(top_features) * 2.8), 8), squeeze=False)
    for i, (_, rep) in enumerate(reps.iterrows()):
        for j, feature in enumerate(top_features):
            subset = plot_rows[plot_rows.patient_id == rep.patient_id]
            sns.boxplot(data=subset[subset.feature_name == feature], x="Class", y="feature_value", color="#eeeeee", ax=axes[i, j], fliersize=0)
            sns.stripplot(data=subset[subset.feature_name == feature], x="Class", y="feature_value", hue="Class", palette={0: COLORS["prehi"], 1: COLORS["hi"]}, size=2.5, alpha=.6, legend=False, ax=axes[i, j])
            axes[i, j].set_title(feature); axes[i, j].set_xlabel(""); axes[i, j].set_ylabel(str(rep.patient_id) if j == 0 else "")
    save_figure(fig, "patient_feature_distributions"); created.append("patient_feature_distributions")

    qc_data = master[["patient_id", "review_window_fraction", "jackknife_roc_auc", "calibration_iqr_roc_auc"]].melt(id_vars=["patient_id", "review_window_fraction"], var_name="configuration", value_name="patient_auc")
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); sns.scatterplot(data=qc_data, x="review_window_fraction", y="patient_auc", hue="configuration", palette={"jackknife_roc_auc": COLORS["jackknife"], "calibration_iqr_roc_auc": COLORS["calibration_iqr"]}, ax=ax)
    for outcome in ["jackknife_roc_auc", "calibration_iqr_roc_auc"]:
        _, rho, _ = spearman_row(master["review_window_fraction"], master[outcome]); ax.text(.03, .95 if outcome.startswith("jack") else .89, f"{outcome.split('_')[0]} ρ = {rho:.2f}", transform=ax.transAxes)
    ax.set_xlabel("Review-window fraction"); ax.set_ylabel("Patient ROC AUC"); save_figure(fig, "qc_burden_vs_auc"); created.append("qc_burden_vs_auc")
    return created, probability_data, qc_data


def write_run_summary(data: dict[str, pd.DataFrame | list[str]], master: pd.DataFrame, separation: pd.DataFrame, clinical_continuous: pd.DataFrame, clinical_grouped: pd.DataFrame, representatives: pd.DataFrame, figures: list[str], figure_data_created: bool) -> None:
    """Write the reproducibility and validation summary."""
    summary = pd.DataFrame([{
        "patient_count": len(master), "primary_configuration_count": len(PRIMARY_CONFIGS), "selected_stable_feature_count": len([line for line in (RESULTS_ROOT / "heterogeneity_feature_list.txt").read_text().splitlines() if line.strip()]), "clinical_summary_file_found": not data["clinical_summary"].empty, "clinical_flags_file_found": not data["clinical_flags"].empty, "qc_file_found": not data["qc"].empty, "patients_matched_to_clinical_summary": master["patient_id"].isin(data["clinical_summary"].get("patient_id", pd.Series(dtype=int))).sum(), "patients_matched_to_clinical_flags": master["patient_id"].isin(data["clinical_flags"].get("patient_id", pd.Series(dtype=int))).sum(), "patients_matched_to_qc": master["patient_id"].isin(data["qc"]["patient_id"]).sum(), "feature_separation_rows": len(separation), "continuous_clinical_association_rows": len(clinical_continuous), "grouped_clinical_comparison_rows": len(clinical_grouped), "representative_patient_count": len(representatives), "figures_created_png": len(figures), "figures_created_pdf": len(figures), "all_prediction_checks_passed": True, "all_patient_matching_checks_passed": True, "all_effect_sizes_finite_where_expected": bool(np.isfinite(separation["cliffs_delta"].dropna()).all()), "all_figure_data_files_created": figure_data_created,
    }])
    summary.to_csv(RESULTS_ROOT / "patient_heterogeneity_run_summary.csv", index=False)


def main() -> None:
    """Run the exploratory patient heterogeneity analysis."""
    print("Locating and validating inputs", flush=True)
    data = read_required_inputs(); validate_inputs(data)
    print("Matching patient identifiers and building master table", flush=True)
    master = build_master_table(data)
    print("Selecting stable features and calculating feature separation", flush=True)
    features = select_stable_features(data["stability"], data["features"])
    (RESULTS_ROOT / "heterogeneity_feature_list.txt").write_text("\n".join(features) + "\n", encoding="utf-8")
    separation = calculate_feature_separation(data["modelling"], features)
    feature_auc = calculate_feature_auc_associations(separation, master)
    print("Analysing clinical and QC associations", flush=True)
    availability, clinical_continuous, clinical_grouped = clinical_variable_tables(master)
    representatives = choose_representatives(master)
    summary = patient_summary(master, availability)
    print("Selecting representative patients and creating publication figures", flush=True)
    figure_stems, probability_data, qc_data = create_figures(master, separation, representatives, data["predictions"], features)
    master.to_csv(RESULTS_ROOT / "patient_heterogeneity_master.csv", index=False)
    separation.to_csv(RESULTS_ROOT / "patient_feature_separation.csv", index=False)
    feature_auc.to_csv(RESULTS_ROOT / "feature_separation_auc_associations.csv", index=False)
    availability.to_csv(RESULTS_ROOT / "clinical_variable_availability.csv", index=False)
    clinical_continuous.to_csv(RESULTS_ROOT / "clinical_continuous_auc_associations.csv", index=False)
    clinical_grouped.to_csv(RESULTS_ROOT / "clinical_group_auc_comparisons.csv", index=False)
    representatives.to_csv(RESULTS_ROOT / "representative_patients.csv", index=False)
    summary.to_csv(RESULTS_ROOT / "patient_heterogeneity_summary.csv", index=False)
    probability_data.to_csv(RESULTS_ROOT / "figure_data_representative_probabilities.csv", index=False)
    master[["patient_id", "jackknife_roc_auc", "calibration_iqr_roc_auc", "auc_difference_jackknife_minus_calibration"]].to_csv(RESULTS_ROOT / "figure_data_patient_auc_comparison.csv", index=False)
    master[["patient_id", "jackknife_roc_auc", "calibration_iqr_roc_auc", "auc_difference_jackknife_minus_calibration"]].to_csv(RESULTS_ROOT / "figure_data_auc_scatter.csv", index=False)
    separation[separation["feature_name"].isin(features)].to_csv(RESULTS_ROOT / "figure_data_feature_effect_heatmap.csv", index=False)
    qc_data.to_csv(RESULTS_ROOT / "figure_data_qc_auc.csv", index=False)
    write_run_summary(data, master, separation, clinical_continuous, clinical_grouped, representatives, figure_stems, True)
    print("Patient heterogeneity analysis completed", flush=True)
    print(f"Patients: {len(master)}; stable features: {len(features)}; figures: {len(figure_stems)} PNG/PDF pairs", flush=True)


if __name__ == "__main__":
    main()
