"""Exploratory clinical sensitivity analysis for the ECG LOSO study.

The primary modelling analysis is not changed here. This script merges the
existing patient-level results with structured clinical metadata, describes
low- and perfect-AUC patients, and subsets existing out-of-fold predictions
for prespecified secondary sensitivity cohorts. All findings are exploratory.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"
METADATA_ROOT = PROJECT_ROOT / "metadata"
MODEL_ROOT = PROJECT_ROOT / "data" / "modelling"
RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 2000
PATIENT_COUNT = 20
PRIMARY_CONFIGS = {
    "jackknife": ("common_rows", "jackknife_median", "extra_trees"),
    "calibration_iqr": ("common_rows", "calibration_median_iqr", "extra_trees"),
}
LARGE_EFFECT = 0.474
COLORS = {
    "jackknife": "#1f4e79",
    "calibration_iqr": "#d97706",
    "prehi": "#4c78a8",
    "hi": "#e45756",
    "missing": "#d9d9d9",
}

sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
plt.rcParams["font.family"] = "DejaVu Sans"


def read_inputs() -> dict[str, pd.DataFrame | list[str]]:
    """Read required analysis and clinical input files."""
    paths = {
        "clinical_summary": METADATA_ROOT / "patient_clinical_summary.csv",
        "clinical_flags": METADATA_ROOT / "patient_clinical_flags.csv",
        "master": RESULTS_ROOT / "patient_heterogeneity_master.csv",
        "feature_separation": RESULTS_ROOT / "patient_feature_separation.csv",
        "metrics": RESULTS_ROOT / "loso_patient_metrics.csv",
        "predictions": RESULTS_ROOT / "loso_predictions.csv",
        "modelling": MODEL_ROOT / "clean_modelling_table.csv",
        "stability": RESULTS_ROOT / "feature_selection_stability.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    data: dict[str, pd.DataFrame | list[str]] = {name: pd.read_csv(path) for name, path in paths.items()}
    data["features"] = [
        line.strip()
        for line in (MODEL_ROOT / "final_feature_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return data


def patient_ids(table: pd.DataFrame, column: str) -> set[str]:
    """Return patient IDs in a stable string representation."""
    return set(table[column].dropna().astype(str))


def validate_inputs(data: dict[str, pd.DataFrame | list[str]]) -> None:
    """Validate patient coverage, clinical uniqueness, and prediction ranges."""
    master = data["master"]
    clinical_summary = data["clinical_summary"].rename(columns={"Patient_ID": "patient_id"})
    clinical_flags = data["clinical_flags"].rename(columns={"Patient_ID": "patient_id"})
    if len(master) != PATIENT_COUNT or master["patient_id"].nunique() != PATIENT_COUNT:
        raise ValueError("The patient heterogeneity master must contain exactly 20 unique patients")
    expected = patient_ids(master, "patient_id")
    for name, table in [("clinical summary", clinical_summary), ("clinical flags", clinical_flags)]:
        if table["patient_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate patient records")
        if patient_ids(table, "patient_id") != expected:
            raise ValueError(f"Patient matching is incomplete for {name}")
    if {"30851", "27245"} - expected:
        raise ValueError("Patients 30851 and 27245 are missing from the primary results")
    if len(data["features"]) != 34 or len(set(data["features"])) != 34:
        raise ValueError("final_feature_list.txt must contain exactly 34 unique features")
    predictions = data["predictions"]
    if not np.isfinite(predictions["predicted_probability"]).all() or not predictions["predicted_probability"].between(0, 1).all():
        raise ValueError("Prediction probabilities must be finite and within [0, 1]")
    if predictions.duplicated(["patient_id", "segment_id", "evaluation_mode", "normalization", "classifier"]).any():
        raise ValueError("Duplicate patient/configuration/segment prediction rows found")
    separation = data["feature_separation"]
    if not expected.issubset(patient_ids(separation, "patient_id")):
        raise ValueError("Feature-separation results do not cover all 20 patients")


def matching_audit(data: dict[str, pd.DataFrame | list[str]]) -> pd.DataFrame:
    """Create the requested patient matching audit."""
    master = data["master"]
    expected = sorted(patient_ids(master, "patient_id"))
    sources = {
        "clinical_summary": patient_ids(data["clinical_summary"].rename(columns={"Patient_ID": "patient_id"}), "patient_id"),
        "clinical_flags": patient_ids(data["clinical_flags"].rename(columns={"Patient_ID": "patient_id"}), "patient_id"),
        "feature_separation": patient_ids(data["feature_separation"], "patient_id"),
        "qc_analysis": patient_ids(data["master"], "patient_id"),
    }
    rows = []
    for patient_id in expected:
        row = {"patient_id": patient_id, **{f"present_in_{name}": patient_id in values for name, values in sources.items()}}
        row["present_in_modelling_results"] = patient_id in patient_ids(master, "patient_id")
        row["fully_matched"] = all(row[column] for column in row if column.startswith("present_in_"))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_ef(value: object) -> float:
    """Parse an EF percentage or range to a documented midpoint when possible."""
    if pd.isna(value):
        return np.nan
    matches = re.findall(r"\d+(?:\.\d+)?", str(value))
    if not matches:
        return np.nan
    numbers = [float(item) for item in matches[:2]]
    return float(np.mean(numbers))


def variable_type(series: pd.Series, name: str) -> tuple[str, bool, str]:
    """Classify a clinical field without treating missing values as negative."""
    if name in {"Patient_ID", "patient_id"}:
        return "identifier", False, "identifier"
    if name == "Age":
        return "continuous", True, "continuous patient variable"
    if pd.api.types.is_numeric_dtype(series) and set(series.dropna().unique()).issubset({0, 1}):
        return "binary", True, "structured binary flag"
    if name in {"Sex", "Outcome"}:
        return "categorical", True, "structured categorical variable"
    if name == "EF":
        return "continuous_text", True, "numeric midpoint parsed separately; original EF preserved"
    if series.dtype == object:
        return ("free_text", False, "narrative field; preserved for context, not subgroup testing")
    return "categorical", False, "not eligible for prespecified analysis"


def metadata_dictionary(data: dict[str, pd.DataFrame | list[str]]) -> pd.DataFrame:
    """Describe every original clinical field and its missingness."""
    rows = []
    for source_name, table in [("patient_clinical_summary.csv", data["clinical_summary"]), ("patient_clinical_flags.csv", data["clinical_flags"])]:
        for column in table.columns:
            kind, eligible, reason = variable_type(table[column], column)
            rows.append({
                "source_file": source_name,
                "original_variable_name": column,
                "cleaned_variable_name": f"summary_{column}" if source_name.startswith("patient_clinical_summary") and column != "Patient_ID" else f"flag_{column}" if column != "Patient_ID" else "patient_id",
                "variable_type": kind,
                "non_missing_count": int(table[column].notna().sum()),
                "missing_count": int(table[column].isna().sum()),
                "unique_value_count": int(table[column].nunique(dropna=True)),
                "analysis_eligible": eligible,
                "reason_if_not_eligible": "" if eligible else reason,
            })
    rows.append({"source_file": "patient_clinical_summary.csv", "original_variable_name": "EF_numeric_midpoint", "cleaned_variable_name": "summary_EF_numeric_midpoint", "variable_type": "continuous_derived", "non_missing_count": int(data["clinical_summary"]["EF"].map(parse_ef).notna().sum()), "missing_count": int(data["clinical_summary"]["EF"].map(parse_ef).isna().sum()), "unique_value_count": int(data["clinical_summary"]["EF"].map(parse_ef).nunique(dropna=True)), "analysis_eligible": True, "reason_if_not_eligible": "explicitly parsed midpoint; original EF remains unchanged"})
    return pd.DataFrame(rows)


def prepare_clinical_master(data: dict[str, pd.DataFrame | list[str]]) -> pd.DataFrame:
    """Merge structured clinical variables with the existing patient master."""
    master = data["master"].copy()
    summary = data["clinical_summary"].rename(columns={"Patient_ID": "patient_id"}).copy()
    flags = data["clinical_flags"].rename(columns={"Patient_ID": "patient_id"}).copy()
    summary = summary.rename(columns={column: f"summary_{column}" for column in summary.columns if column != "patient_id"})
    flags = flags.rename(columns={column: f"flag_{column}" for column in flags.columns if column != "patient_id"})
    summary["summary_EF_numeric_midpoint"] = summary["summary_EF"].map(parse_ef)
    return master.merge(summary, on="patient_id", how="left", validate="one_to_one").merge(flags, on="patient_id", how="left", validate="one_to_one")


def add_performance_flags(master: pd.DataFrame) -> pd.DataFrame:
    """Add exact and below-chance AUC indicators."""
    master = master.copy()
    master["perfect_jackknife_auc"] = master["jackknife_roc_auc"].sub(1).abs() <= 1e-12
    master["below_chance_jackknife_auc"] = master["jackknife_roc_auc"] < 0.5
    master["below_chance_calibration_auc"] = master["calibration_iqr_roc_auc"] < 0.5
    return master


def bh_adjust(values: pd.Series) -> pd.Series:
    """Apply Benjamini-Hochberg correction while preserving missingness."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    if not valid.any():
        return result
    p = values[valid].to_numpy(float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1].clip(0, 1)
    output = np.empty(len(p)); output[order] = adjusted
    result.loc[valid] = output
    return result


def bootstrap_median_difference(group_1: np.ndarray, group_0: np.ndarray, rng: np.random.Generator, iterations: int = BOOTSTRAP_ITERATIONS) -> tuple[float, float]:
    """Bootstrap the group-1 minus group-0 median difference."""
    if len(group_1) == 0 or len(group_0) == 0:
        return np.nan, np.nan
    values = np.empty(iterations)
    for index in range(iterations):
        values[index] = np.median(rng.choice(group_1, len(group_1), replace=True)) - np.median(rng.choice(group_0, len(group_0), replace=True))
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def rank_biserial(group_1: np.ndarray, group_0: np.ndarray) -> float:
    """Calculate rank-biserial effect size with group 1 minus group 0 direction."""
    u = stats.mannwhitneyu(group_1, group_0, alternative="two-sided").statistic
    return float(2 * u / (len(group_1) * len(group_0)) - 1)


def clinical_subgroup_tables(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate binary and multi-group descriptive AUC comparisons."""
    binary_rows, categorical_rows = [], []
    rng = np.random.default_rng(RANDOM_SEED)
    binary_columns = [column for column in master.columns if column.startswith("flag_") and column != "flag_notes"]
    binary_columns += ["summary_Sex"]
    for column in binary_columns:
        values = master[column]
        levels = list(values.dropna().unique())
        if len(levels) != 2:
            continue
        level_1, level_0 = sorted(levels, key=str)[-1], sorted(levels, key=str)[0]
        group_1_mask, group_0_mask = values == level_1, values == level_0
        for outcome, family in [("jackknife_roc_auc", "jackknife_binary"), ("calibration_iqr_roc_auc", "calibration_iqr_binary")]:
            group_1 = master.loc[group_1_mask, outcome].dropna().to_numpy(); group_0 = master.loc[group_0_mask, outcome].dropna().to_numpy()
            eligible = len(group_1) >= 3 and len(group_0) >= 3
            row = {"variable_name": column, "outcome_name": outcome, "positive_level": str(level_1), "negative_level": str(level_0), "positive_count": len(group_1), "negative_count": len(group_0), "analysis_eligible": eligible, "reason_if_not_eligible": "" if eligible else "requires at least 3 patients per group"}
            if eligible:
                ci_low, ci_high = bootstrap_median_difference(group_1, group_0, rng)
                test = stats.mannwhitneyu(group_1, group_0, alternative="two-sided")
                row.update({"positive_median_auc": np.median(group_1), "negative_median_auc": np.median(group_0), "positive_iqr_auc": np.quantile(group_1, .75) - np.quantile(group_1, .25), "negative_iqr_auc": np.quantile(group_0, .75) - np.quantile(group_0, .25), "positive_min_auc": group_1.min(), "negative_min_auc": group_0.min(), "positive_max_auc": group_1.max(), "negative_max_auc": group_0.max(), "positive_mean_auc": group_1.mean(), "negative_mean_auc": group_0.mean(), "median_difference_positive_minus_negative": np.median(group_1) - np.median(group_0), "hodges_lehmann_location_shift": np.median((group_1[:, None] - group_0[None, :]).ravel()), "mann_whitney_u_statistic": test.statistic, "raw_p_value": test.pvalue, "rank_biserial_effect_size": rank_biserial(group_1, group_0), "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high, "fdr_family": family})
            binary_rows.append(row)
    for column in [column for column in master.columns if column.startswith("flag_") and column != "flag_notes"]:
        levels = master[column].dropna().unique()
        if len(levels) <= 2:
            continue
        groups = [master.loc[master[column] == level, "jackknife_roc_auc"].dropna().to_numpy() for level in levels]
        adequate = all(len(group) >= 3 for group in groups)
        row = {"variable_name": column, "group_count": len(levels), "analysis_eligible": adequate, "reason_if_not_eligible": "" if adequate else "requires at least 3 patients per group", "raw_p_value": stats.kruskal(*groups).pvalue if adequate else np.nan, "fdr_family": "categorical"}
        for level, group in zip(levels, groups):
            row[f"group_{level}_count"] = len(group); row[f"group_{level}_median_auc"] = np.median(group) if len(group) else np.nan; row[f"group_{level}_iqr_auc"] = np.quantile(group, .75) - np.quantile(group, .25) if len(group) else np.nan
        categorical_rows.append(row)
    binary = pd.DataFrame(binary_rows)
    if not binary.empty:
        binary["exploratory_fdr_q_value"] = binary.groupby("fdr_family")["raw_p_value"].transform(bh_adjust)
    categorical_columns = [
        "variable_name", "group_count", "analysis_eligible",
        "reason_if_not_eligible", "raw_p_value", "fdr_family",
        "exploratory_fdr_q_value",
    ]
    categorical = pd.DataFrame(categorical_rows, columns=categorical_columns)
    if not categorical.empty:
        categorical["exploratory_fdr_q_value"] = categorical["raw_p_value"].transform(bh_adjust)
    return binary, categorical


def continuous_associations(master: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman associations for eligible continuous fields."""
    columns = ["summary_Age", "summary_EF_numeric_midpoint"]
    rows = []
    outcomes = ["jackknife_roc_auc", "calibration_iqr_roc_auc", "auc_difference_jackknife_minus_calibration", "review_window_fraction"]
    for column in columns:
        for outcome in outcomes:
            pair = master[[column, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
            count = len(pair)
            if count >= 10 and pair[column].nunique() > 1 and pair[outcome].nunique() > 1:
                result = stats.spearmanr(pair[column], pair[outcome]); rho, p_value = result.statistic, result.pvalue
            else:
                rho, p_value = np.nan, np.nan
            rows.append({"variable_name": column, "outcome_name": outcome, "patient_count": count, "spearman_rho": rho, "raw_p_value": p_value, "exploratory_fdr_q_value": np.nan, "missing_count": int(master[column].isna().sum()), "coverage_flag": "limited_coverage" if count < 15 else "adequate_coverage"})
    output = pd.DataFrame(rows)
    if not output.empty:
        output["exploratory_fdr_q_value"] = output.groupby("outcome_name")["raw_p_value"].transform(bh_adjust)
    return output


def auc_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate ROC AUC from labels and probabilities without sklearn."""
    labels = np.asarray(labels); probabilities = np.asarray(probabilities)
    positive = labels == 1; negative = labels == 0
    if positive.sum() == 0 or negative.sum() == 0:
        return np.nan
    ranks = stats.rankdata(probabilities)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def average_precision_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate average precision for binary labels."""
    order = np.argsort(-probabilities, kind="mergesort")
    ordered = labels[order]
    positives = ordered.sum()
    if positives == 0:
        return np.nan
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / positives)


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate pooled prediction metrics for a prediction subset."""
    labels = predictions["Class"].to_numpy(); probabilities = predictions["predicted_probability"].to_numpy(); predicted = (probabilities >= .5).astype(int)
    tp = np.sum((labels == 1) & (predicted == 1)); tn = np.sum((labels == 0) & (predicted == 0)); fp = np.sum((labels == 0) & (predicted == 1)); fn = np.sum((labels == 1) & (predicted == 0))
    sensitivity = tp / (tp + fn) if tp + fn else np.nan; specificity = tn / (tn + fp) if tn + fp else np.nan
    return {"pooled_roc_auc": auc_score(labels, probabilities), "pooled_average_precision": average_precision_score(labels, probabilities), "pooled_sensitivity": sensitivity, "pooled_specificity": specificity, "pooled_balanced_accuracy": np.nanmean([sensitivity, specificity])}


def bootstrap_prediction_metrics(predictions: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    """Bootstrap pooled AUC and average precision by resampling patients."""
    patients = sorted(predictions["patient_id"].unique())
    auc_values, ap_values = [], []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = rng.choice(patients, len(patients), replace=True)
        subset = pd.concat([predictions[predictions["patient_id"] == patient] for patient in sampled], ignore_index=True)
        values = prediction_metrics(subset)
        auc_values.append(values["pooled_roc_auc"]); ap_values.append(values["pooled_average_precision"])
    return {"pooled_roc_auc_bootstrap_ci_low": np.nanquantile(auc_values, .025), "pooled_roc_auc_bootstrap_ci_high": np.nanquantile(auc_values, .975), "pooled_average_precision_bootstrap_ci_low": np.nanquantile(ap_values, .025), "pooled_average_precision_bootstrap_ci_high": np.nanquantile(ap_values, .975)}


def subgroup_performance(master: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Subset existing predictions for prespecified clinical sensitivity cohorts."""
    subgroup_masks: dict[str, pd.Series] = {"full_cohort": pd.Series(True, index=master.index)}
    flag = lambda name: master.get(f"flag_{name}", pd.Series(np.nan, index=master.index))
    subgroup_masks.update({
        "hrv_valid": flag("exclude_from_HRV_valid_subgroup") == 0,
        "without_implanted_device_or_pacing": flag("paced_or_icd") == 0,
        "without_major_arrhythmia": (flag("any_AF") == 0) & (flag("BBB") == 0) & (flag("major_HRV_outlier") == 0),
        "without_high_qc_burden": master["review_window_fraction"] <= master["review_window_fraction"].median(),
        "clean_sinus_rhythm_candidate": flag("clean_sinus_candidate") == 1,
        "without_atypical_hi": flag("atypical_HI") == 0,
    })
    rows = []; rng = np.random.default_rng(RANDOM_SEED)
    for subgroup, mask in subgroup_masks.items():
        included = set(master.loc[mask, "patient_id"].astype(str))
        for config_name, config in PRIMARY_CONFIGS.items():
            subset = predictions[(predictions["evaluation_mode"] == config[0]) & (predictions["normalization"] == config[1]) & (predictions["classifier"] == config[2]) & predictions["patient_id"].astype(str).isin(included)]
            values = prediction_metrics(subset) if len(subset) else {key: np.nan for key in ["pooled_roc_auc", "pooled_average_precision", "pooled_sensitivity", "pooled_specificity", "pooled_balanced_accuracy"]}
            row = {"analysis_type": "primary" if subgroup == "full_cohort" else "secondary", "subgroup_name": subgroup, "configuration": config_name, "included_patient_count": len(included), "included_prediction_row_count": len(subset), **values}
            if len(included) >= 2:
                row.update(bootstrap_prediction_metrics(subset, rng))
            else:
                row.update({"pooled_roc_auc_bootstrap_ci_low": np.nan, "pooled_roc_auc_bootstrap_ci_high": np.nan, "pooled_average_precision_bootstrap_ci_low": np.nan, "pooled_average_precision_bootstrap_ci_high": np.nan})
            patient_auc = master[master.patient_id.astype(str).isin(included)]["jackknife_roc_auc" if config_name == "jackknife" else "calibration_iqr_roc_auc"]
            row.update({"median_patient_auc": patient_auc.median(), "patient_auc_iqr": patient_auc.quantile(.75) - patient_auc.quantile(.25), "minimum_patient_auc": patient_auc.min(), "maximum_patient_auc": patient_auc.max()})
            rows.append(row)
    return pd.DataFrame(rows)


def feature_reference_directions(separation: pd.DataFrame, master: pd.DataFrame) -> dict[str, int]:
    """Get cohort reference directions among patients with jackknife AUC >= .85."""
    high = set(master.loc[master["jackknife_roc_auc"] >= .85, "patient_id"])
    reference = {}
    for feature, group in separation[separation.patient_id.isin(high)].groupby("feature_name"):
        value = np.median(group["median_difference"].dropna())
        reference[feature] = int(np.sign(value))
    return reference


def case_summary(master: pd.DataFrame, separation: pd.DataFrame) -> pd.DataFrame:
    """Create structured rule-based summaries for patients below chance."""
    reference = feature_reference_directions(separation, master)
    rows = []
    for patient_id in ["30851", "27245"]:
        patient = master[master.patient_id.astype(str) == patient_id].iloc[0]
        group = separation[separation.patient_id.astype(str) == patient_id]
        row = {"patient_id": patient_id}
        for field in ["jackknife_roc_auc", "calibration_iqr_roc_auc", "jackknife_average_precision", "calibration_iqr_average_precision", "jackknife_sensitivity", "calibration_iqr_sensitivity", "jackknife_specificity", "calibration_iqr_specificity", "jackknife_balanced_accuracy", "calibration_iqr_balanced_accuracy", "jackknife_evaluated_rows", "jackknife_class_0_evaluated_rows", "jackknife_class_1_evaluated_rows", "review_window_fraction", "usable_window_count", "review_window_count", "selected_calibration_start_rank", "calibration_candidate_rows_skipped", "later_continuous_block_used"]:
            row[field] = patient.get(field, np.nan)
        for prefix in ["jackknife", "calibration_iqr"]:
            for field in ["class_0_probability_median", "class_1_probability_median", "probability_median_difference", "probability_overlap_measure"]:
                row[f"{prefix}_{field}"] = patient.get(f"{prefix}_{field}", np.nan)
            row[f"{prefix}_median_probability_ordering_reversed"] = patient.get(f"{prefix}_probability_median_difference", np.nan) < 0
        row["feature_direction_agreement_with_high_auc_reference"] = np.mean([np.sign(value) == reference.get(feature, 0) for feature, value in zip(group.feature_name, group.median_difference) if reference.get(feature, 0) != 0])
        for _, feature_row in group.iterrows():
            feature = feature_row.feature_name
            row[f"{feature}_class_0_median"] = feature_row.class_0_median; row[f"{feature}_class_1_median"] = feature_row.class_1_median; row[f"{feature}_cliffs_delta"] = feature_row.cliffs_delta; row[f"{feature}_absolute_cliffs_delta"] = feature_row.absolute_cliffs_delta; row[f"{feature}_direction_of_change"] = feature_row.direction_of_change; row[f"{feature}_direction_agrees_with_high_auc_reference"] = np.sign(feature_row.median_difference) == reference.get(feature, 0)
        statements = []
        if patient.review_window_fraction >= master.review_window_fraction.median(): statements.append("extreme QC burden" if patient.review_window_fraction > .75 else "high QC burden")
        if patient.jackknife_probability_median_difference < 0 or patient.calibration_iqr_probability_median_difference < 0: statements.append("reversed prediction ordering")
        if group["median_difference"].apply(np.sign).nunique() > 1: statements.append("mixed feature directions")
        if patient.get("flag_paced_or_icd", 0) == 1 or patient.get("flag_any_AF", 0) == 1: statements.append("implanted-device or rhythm context")
        if patient.get("flag_major_HRV_outlier", 0) == 1: statements.append("major HRV outlier flag")
        row["rule_based_interpretation"] = "; ".join(statements) if statements else "insufficient evidence for a single explanation"
        rows.append(row)
    return pd.DataFrame(rows)


def perfect_summary(master: pd.DataFrame, separation: pd.DataFrame) -> pd.DataFrame:
    """Summarize every patient with exact jackknife AUC of 1.00."""
    perfect = master[master.perfect_jackknife_auc].copy(); reference = feature_reference_directions(separation, master)
    rows = []
    for _, patient in perfect.iterrows():
        group = separation[separation.patient_id == patient.patient_id]
        row = {"patient_id": patient.patient_id, "jackknife_roc_auc": patient.jackknife_roc_auc, "calibration_iqr_roc_auc": patient.calibration_iqr_roc_auc, "total_source_windows": patient.total_source_windows, "common_evaluation_windows": patient.common_evaluation_windows, "review_window_fraction": patient.review_window_fraction, "jackknife_probability_median_difference": patient.jackknife_probability_median_difference, "jackknife_probability_overlap_measure": patient.jackknife_probability_overlap_measure, "median_absolute_cliffs_delta": group.absolute_cliffs_delta.median(), "minimum_absolute_cliffs_delta": group.absolute_cliffs_delta.min(), "large_effect_feature_count": int((group.absolute_cliffs_delta >= LARGE_EFFECT).sum()), "very_large_effect_feature_count": int((group.absolute_cliffs_delta >= .70).sum()), "feature_direction_consistency_score": np.mean([np.sign(value) == reference.get(feature, 0) for feature, value in zip(group.feature_name, group.median_difference) if reference.get(feature, 0) != 0])}
        for column in master.columns:
            if column.startswith("flag_") and column != "flag_notes": row[column] = patient[column]
        for _, feature_row in group.iterrows(): row[f"{feature_row.feature_name}_cliffs_delta"] = feature_row.cliffs_delta
        rows.append(row)
    return pd.DataFrame(rows)


def perfect_comparison(master: pd.DataFrame, separation: pd.DataFrame) -> pd.DataFrame:
    """Compare perfect and non-perfect patients on separation and prediction fields."""
    rows = []
    metrics = {
        "review_window_fraction": master["review_window_fraction"],
        "jackknife_probability_median_difference": master["jackknife_probability_median_difference"],
        "jackknife_probability_overlap_measure": master["jackknife_probability_overlap_measure"],
    }
    patient_effect = separation.groupby("patient_id")["absolute_cliffs_delta"].median()
    metrics["median_absolute_cliffs_delta"] = master.patient_id.map(patient_effect)
    metrics["large_effect_feature_count"] = master.patient_id.map(separation.groupby("patient_id")["absolute_cliffs_delta"].apply(lambda x: (x >= LARGE_EFFECT).sum()))
    for name, values in metrics.items():
        one = values[master.perfect_jackknife_auc].dropna().to_numpy(); zero = values[~master.perfect_jackknife_auc].dropna().to_numpy(); eligible = len(one) >= 3 and len(zero) >= 3
        row = {"variable_name": name, "perfect_count": len(one), "nonperfect_count": len(zero), "analysis_eligible": eligible}
        if eligible:
            test = stats.mannwhitneyu(one, zero, alternative="two-sided"); row.update({"perfect_median": np.median(one), "nonperfect_median": np.median(zero), "median_difference_perfect_minus_nonperfect": np.median(one) - np.median(zero), "mann_whitney_u_statistic": test.statistic, "raw_p_value": test.pvalue, "rank_biserial_effect_size": rank_biserial(one, zero)})
        rows.append(row)
    for column in [column for column in master.columns if column.startswith("flag_") and column != "flag_notes"]:
        one = master.loc[master.perfect_jackknife_auc, column].dropna().astype(float); zero = master.loc[~master.perfect_jackknife_auc, column].dropna().astype(float); rows.append({"variable_name": column, "perfect_count": len(one), "nonperfect_count": len(zero), "analysis_eligible": len(one) >= 3 and len(zero) >= 3, "perfect_positive_count": int(one.sum()), "nonperfect_positive_count": int(zero.sum())})
    output = pd.DataFrame(rows)
    if "raw_p_value" in output:
        output["exploratory_fdr_q_value"] = bh_adjust(output["raw_p_value"])
    return output


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save PNG and vector PDF at publication resolution and close the figure."""
    fig.savefig(FIGURES_ROOT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figures(master: pd.DataFrame, binary: pd.DataFrame, subgroups: pd.DataFrame, case: pd.DataFrame, perfect: pd.DataFrame, separation: pd.DataFrame, predictions: pd.DataFrame, features: list[str]) -> dict[str, pd.DataFrame]:
    """Create the six requested figure families and return their source data."""
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    figure_data = {}
    eligible = binary[binary.analysis_eligible].copy() if not binary.empty else binary
    fig, ax = plt.subplots(figsize=(8, max(4, len(eligible) * .35)))
    if not eligible.empty:
        eligible["label"] = eligible.variable_name + " (n=" + eligible.positive_count.astype(str) + "/" + eligible.negative_count.astype(str) + ")"
        for config, color, offset in [("jackknife", COLORS["jackknife"], .12), ("calibration_iqr", COLORS["calibration_iqr"], -.12)]:
            values = eligible[eligible.outcome_name == ("jackknife_roc_auc" if config == "jackknife" else "calibration_iqr_roc_auc")].set_index("variable_name")
            y = np.arange(len(eligible)) + offset
            mapped = eligible.variable_name.map(values["median_difference_positive_minus_negative"])
            low = eligible.variable_name.map(values["bootstrap_ci_low"]); high = eligible.variable_name.map(values["bootstrap_ci_high"])
            ax.errorbar(mapped, y, xerr=[mapped - low, high - mapped], fmt="o", color=color, label=config.replace("_", " ").title())
        ax.set_yticks(range(len(eligible)), eligible["label"])
    ax.axvline(0, color="black", ls="--", lw=.8); ax.set_xlabel("Median AUC difference: positive minus negative"); ax.set_ylabel("Clinical variable"); ax.legend(frameon=False)
    save_figure(fig, "clinical_subgroup_auc_sensitivity"); figure_data["clinical_subgroup_auc"] = eligible

    flag_columns = [column for column in master.columns if column.startswith("flag_") and column != "flag_notes"]
    extreme_ids = ["30851", "27245"] + sorted(perfect.patient_id.astype(str).tolist())
    profile = master[master.patient_id.astype(str).isin(extreme_ids)].copy().set_index(master[master.patient_id.astype(str).isin(extreme_ids)].patient_id.astype(str))
    heat = profile[flag_columns].apply(pd.to_numeric, errors="coerce") if flag_columns else pd.DataFrame(index=profile.index)
    fig, ax = plt.subplots(figsize=(max(6, len(flag_columns) * .55), max(3, len(profile) * .45)))
    if not heat.empty: sns.heatmap(heat, cmap=sns.color_palette([COLORS["missing"], "#ffffff", COLORS["hi"]], as_cmap=True), vmin=0, vmax=1, annot=True, fmt=".0f", mask=heat.isna(), ax=ax, cbar=False)
    ax.set_xlabel("Structured clinical flag"); ax.set_ylabel("Patient ID"); save_figure(fig, "extreme_patient_clinical_profile"); figure_data["extreme_patient_clinical_profile"] = profile.reset_index(drop=True)

    extreme_sep = separation[separation.patient_id.astype(str).isin(extreme_ids)].pivot(index="patient_id", columns="feature_name", values="cliffs_delta").reindex(columns=features)
    fig, ax = plt.subplots(figsize=(max(8, len(features) * .65), max(3, len(extreme_sep) * .45)))
    sns.heatmap(extreme_sep, cmap="vlag", vmin=-1, vmax=1, center=0, annot=extreme_sep.abs().ge(LARGE_EFFECT), fmt="", cbar_kws={"label": "Cliff's delta"}, ax=ax); ax.set_xlabel("Stable feature"); ax.set_ylabel("Patient ID"); save_figure(fig, "extreme_patient_feature_separation"); figure_data["extreme_patient_feature_separation"] = extreme_sep.reset_index()

    below = predictions[predictions.patient_id.astype(str).isin(["30851", "27245"]) & predictions.evaluation_mode.eq("common_rows") & predictions.classifier.eq("extra_trees") & predictions.normalization.isin(["jackknife_median", "calibration_median_iqr"])].copy(); below["configuration"] = below.normalization.map({"jackknife_median": "jackknife", "calibration_median_iqr": "calibration_iqr"}); figure_data["below_chance_predictions"] = below
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, patient_id in zip(axes, ["30851", "27245"]):
        subset = below[below.patient_id.astype(str) == patient_id]; sns.violinplot(data=subset, x="Class", y="predicted_probability", hue="configuration", inner="box", cut=0, palette={"jackknife": COLORS["jackknife"], "calibration_iqr": COLORS["calibration_iqr"]}, ax=ax); row = master[master.patient_id.astype(str) == patient_id].iloc[0]; ax.set_title(f"{patient_id}: J={row.jackknife_roc_auc:.2f}, C={row.calibration_iqr_roc_auc:.2f}"); ax.set_ylim(0, 1); ax.set_xlabel("True class")
    axes[0].set_ylabel("Predicted probability"); save_figure(fig, "below_chance_patient_predictions")

    characteristic = pd.DataFrame({"patient_id": master.patient_id, "perfect_group": np.where(master.perfect_jackknife_auc, "perfect", "non-perfect"), "median_absolute_cliffs_delta": master.patient_id.map(separation.groupby("patient_id").absolute_cliffs_delta.median()), "probability_median_difference": master.jackknife_probability_median_difference, "probability_overlap_measure": master.jackknife_probability_overlap_measure, "review_window_fraction": master.review_window_fraction}); figure_data["perfect_auc_characteristics"] = characteristic
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5)); plot_columns = ["median_absolute_cliffs_delta", "probability_median_difference", "probability_overlap_measure", "review_window_fraction"]
    for ax, column in zip(axes, plot_columns): sns.boxplot(data=characteristic, x="perfect_group", y=column, color="#eeeeee", ax=ax); sns.stripplot(data=characteristic, x="perfect_group", y=column, color=COLORS["jackknife"], ax=ax, size=4); ax.set_xlabel(""); ax.set_title(column.replace("_", " ").title())
    save_figure(fig, "perfect_auc_characteristics")

    restricted = subgroups.copy(); figure_data["clinical_subgroup_performance"] = restricted
    fig, ax = plt.subplots(figsize=(8, max(4, restricted.subgroup_name.nunique() * .45)))
    for config, color in [("jackknife", COLORS["jackknife"]), ("calibration_iqr", COLORS["calibration_iqr"])]:
        part = restricted[restricted.configuration == config].sort_values(["subgroup_name"]); y = np.arange(len(part)) + (.12 if config == "jackknife" else -.12); ax.errorbar(part.pooled_roc_auc, y, xerr=[part.pooled_roc_auc - part.pooled_roc_auc_bootstrap_ci_low, part.pooled_roc_auc_bootstrap_ci_high - part.pooled_roc_auc], fmt="o", color=color, label=config.replace("_", " ").title())
    names = restricted[restricted.configuration == "jackknife"].sort_values("subgroup_name")["subgroup_name"]; ax.set_yticks(range(len(names)), names); ax.set_xlabel("Pooled ROC AUC"); ax.set_ylabel("Cohort"); ax.legend(frameon=False); save_figure(fig, "clinical_subgroup_performance")
    return figure_data


def write_summary(master: pd.DataFrame, dictionary: pd.DataFrame, binary: pd.DataFrame, continuous: pd.DataFrame, subgroups: pd.DataFrame, perfect: pd.DataFrame, comparison: pd.DataFrame, separation: pd.DataFrame, matching: pd.DataFrame) -> pd.DataFrame:
    """Write the manuscript-oriented summary and return it."""
    eligible_binary = binary[binary.analysis_eligible] if not binary.empty else binary
    eligible_continuous = continuous[continuous.patient_count >= 10] if not continuous.empty else continuous
    perfect_group = master[master.perfect_jackknife_auc]; nonperfect = master[~master.perfect_jackknife_auc]
    patient_effect = master.patient_id.map(
        separation.groupby("patient_id")["absolute_cliffs_delta"].median()
    )
    strongest_binary = eligible_binary.assign(abs_effect=eligible_binary.median_difference_positive_minus_negative.abs()).sort_values("abs_effect").iloc[-1].variable_name if not eligible_binary.empty else "none"
    strongest_cont = eligible_continuous.assign(abs_rho=eligible_continuous.spearman_rho.abs()).sort_values("abs_rho").iloc[-1].variable_name if not eligible_continuous.empty else "none"
    restricted = subgroups[subgroups.subgroup_name != "full_cohort"]
    full_auc = subgroups[subgroups.subgroup_name == "full_cohort"].set_index("configuration")["pooled_roc_auc"]
    material = False
    for _, row in restricted.iterrows():
        if row.configuration in full_auc.index and abs(row.pooled_roc_auc - full_auc[row.configuration]) > .05: material = True
    perfect_effect = patient_effect[master.perfect_jackknife_auc].median()
    nonperfect_effect = patient_effect[~master.perfect_jackknife_auc].median()
    summary = pd.DataFrame([{"total_patient_count": len(master), "clinical_summary_match_count": int(matching.present_in_clinical_summary.sum()), "clinical_flags_match_count": int(matching.present_in_clinical_flags.sum()), "eligible_binary_clinical_variables": ";".join(sorted(eligible_binary.variable_name.unique())) if not eligible_binary.empty else "none", "eligible_continuous_clinical_variables": ";".join(sorted(eligible_continuous.variable_name.unique())) if not eligible_continuous.empty else "none", "perfect_auc_patient_count": len(perfect_group), "below_chance_patient_count": int(master.below_chance_jackknife_auc.sum()), "median_qc_burden_perfect": perfect_group.review_window_fraction.median(), "median_qc_burden_nonperfect": nonperfect.review_window_fraction.median(), "median_absolute_feature_separation_perfect": perfect_effect, "median_absolute_feature_separation_nonperfect": nonperfect_effect, "strongest_clinical_binary_auc_effect": strongest_binary, "strongest_continuous_auc_association": strongest_cont, "number_of_secondary_sensitivity_cohorts": subgroups.subgroup_name.nunique() - 1, "restricted_cohort_changes_pooled_auc_by_more_than_0.05": material, "all_validation_checks_passed": bool(matching.fully_matched.all())}])
    return summary


def main() -> None:
    """Run the complete exploratory clinical sensitivity analysis."""
    print("Locating clinical metadata", flush=True)
    data = read_inputs()
    print("Validating patient matching", flush=True)
    validate_inputs(data)
    matching = matching_audit(data)
    print("Preparing clinical variables", flush=True)
    dictionary = metadata_dictionary(data)
    master = add_performance_flags(prepare_clinical_master(data))
    print("Building clinical sensitivity master table", flush=True)
    separation = data["feature_separation"]
    features = [line.strip() for line in (RESULTS_ROOT / "heterogeneity_feature_list.txt").read_text().splitlines() if line.strip()]
    print("Analysing clinical subgroups and continuous associations", flush=True)
    binary, categorical = clinical_subgroup_tables(master)
    continuous = continuous_associations(master)
    print("Analysing patients 30851 and 27245", flush=True)
    case = case_summary(master, separation)
    print("Analysing perfect-AUC patients", flush=True)
    perfect = perfect_summary(master, separation); perfect_comparison_table = perfect_comparison(master, separation)
    print("Calculating restricted-cohort sensitivity", flush=True)
    subgroups = subgroup_performance(master, data["predictions"])
    print("Creating publication figures", flush=True)
    figure_data = create_figures(master, binary, subgroups, case, perfect, separation, data["predictions"], features)
    master.to_csv(RESULTS_ROOT / "clinical_sensitivity_master.csv", index=False)
    matching.to_csv(RESULTS_ROOT / "clinical_patient_matching_audit.csv", index=False)
    dictionary.to_csv(RESULTS_ROOT / "clinical_metadata_dictionary.csv", index=False)
    binary.to_csv(RESULTS_ROOT / "clinical_binary_auc_sensitivity.csv", index=False)
    categorical.to_csv(RESULTS_ROOT / "clinical_categorical_auc_sensitivity.csv", index=False)
    continuous.to_csv(RESULTS_ROOT / "clinical_continuous_auc_sensitivity.csv", index=False)
    subgroups.to_csv(RESULTS_ROOT / "clinical_subgroup_performance_sensitivity.csv", index=False)
    case.to_csv(RESULTS_ROOT / "below_chance_patient_case_summary.csv", index=False)
    perfect.to_csv(RESULTS_ROOT / "perfect_auc_patient_summary.csv", index=False)
    perfect_comparison_table.to_csv(RESULTS_ROOT / "perfect_vs_nonperfect_auc_comparison.csv", index=False)
    summary = write_summary(master, dictionary, binary, continuous, subgroups, perfect, perfect_comparison_table, separation, matching)
    summary.to_csv(RESULTS_ROOT / "clinical_sensitivity_summary.csv", index=False)
    figure_data["clinical_subgroup_auc"].to_csv(RESULTS_ROOT / "figure_data_clinical_subgroup_auc.csv", index=False)
    figure_data["extreme_patient_clinical_profile"].to_csv(RESULTS_ROOT / "figure_data_extreme_patient_clinical_profile.csv", index=False)
    figure_data["extreme_patient_feature_separation"].to_csv(RESULTS_ROOT / "figure_data_extreme_patient_feature_separation.csv", index=False)
    figure_data["below_chance_predictions"].to_csv(RESULTS_ROOT / "figure_data_below_chance_predictions.csv", index=False)
    figure_data["perfect_auc_characteristics"].to_csv(RESULTS_ROOT / "figure_data_perfect_auc_characteristics.csv", index=False)
    figure_data["clinical_subgroup_performance"].to_csv(RESULTS_ROOT / "figure_data_clinical_subgroup_performance.csv", index=False)
    if not matching.fully_matched.all():
        raise ValueError("Patient matching audit failed")
    print("Clinical sensitivity analysis completed", flush=True)
    print(f"Matched patients: {matching.fully_matched.sum()}; perfect AUC patients: {len(perfect)}; secondary cohorts: {subgroups.subgroup_name.nunique() - 1}", flush=True)


if __name__ == "__main__":
    main()
