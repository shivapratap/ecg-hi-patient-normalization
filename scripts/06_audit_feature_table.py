"""Audit and freeze predictors in the clean ECG modelling table.

This script documents the feature table before patient normalization and LOSO
model fitting. It reports data-quality and redundancy findings but does not
modify, normalize, prune, or model any feature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "modelling" / "clean_modelling_table.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "modelling"

EXPECTED_ROW_COUNT = 1_283
EXPECTED_PREDICTOR_COUNT = 48
NEAR_CONSTANT_FREQUENCY_THRESHOLD = 0.99
NEAR_CONSTANT_UNIQUE_COUNT_THRESHOLD = 3
SMALL_CLASS_COUNT_THRESHOLD = 20
HIGH_CORRELATION_THRESHOLD = 0.90
IQR_MULTIPLIER = 1.5

METADATA_COLUMNS = [
    "patient_id",
    "condition",
    "recording_id",
    "segment_id",
    "source_file",
    "window_index",
    "start_timestamp",
    "end_timestamp",
]
EXCLUDED_COLUMNS = {
    *METADATA_COLUMNS,
    "Class",
    "quality_status",
    "quality_reasons",
    "sampen_profile_status",
}

ABFE_FEATURE_NAMES = [
    "minimum",
    "maximum",
    "sum_value",
    "mean",
    "median",
    "standard_deviation",
    "variance",
    "kurtosis",
    "skewness",
    "mean_absolute_value",
    "root_mean_square",
    "peak_to_peak",
    "integrated_absolute_value",
    "waveform_length",
    "zero_crossing_count",
    "slope_sign_change_count",
    "approximate_entropy",
    "permutation_entropy",
    "fuzzy_entropy",
    "distribution_entropy",
    "svd_entropy",
    "lempel_ziv_complexity",
    "hjorth_mobility",
    "hjorth_complexity",
    "fisher_information",
    "petrosian_fractal_dimension",
    "katz_fractal_dimension",
    "higuchi_fractal_dimension",
    "detrended_fluctuation_analysis",
    "peak_frequency",
    "mean_frequency",
    "median_frequency",
    "spectral_edge_frequency_95",
    "spectral_entropy",
]
SAMPEN_FEATURE_NAMES = [
    "n_r_points",
    "TotalSampEn",
    "AvgSampEn",
    "MaxSampEn",
    "MinSampEn",
    "MedianSampEn",
    "StdSampEn",
    "VarSampEn",
    "KurtosisSampEn",
    "SkewnessSampEn",
    "r_at_MaxSampEn",
    "r_at_MinSampEn",
    "AUC_SampEn",
    "ProfileRange",
]


def load_table() -> pd.DataFrame:
    """Load the clean modelling table and validate its core structure."""
    table = pd.read_csv(INPUT_PATH)
    if len(table) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} rows, found {len(table)}"
        )
    required = [*METADATA_COLUMNS, "condition", "Class", "quality_status"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if table["segment_id"].isna().any() or table["segment_id"].duplicated().any():
        raise ValueError("segment_id must be present and unique")
    if table["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values")
    if not table["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1")
    if not table["quality_status"].isin(["usable", "review"]).all():
        raise ValueError("quality_status must contain only usable or review")

    expected_class = table["condition"].map({"PreHI": 0, "HI": 1})
    if expected_class.isna().any() or not expected_class.equals(table["Class"]):
        raise ValueError("condition does not agree with Class")
    return table


def identify_predictors(table: pd.DataFrame) -> list[str]:
    """Identify the ordered numeric predictors after excluding non-predictors."""
    predictors = []
    for column in table.columns:
        if column in EXCLUDED_COLUMNS or column.startswith("qc_"):
            continue
        predictors.append(column)

    if len(predictors) != EXPECTED_PREDICTOR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PREDICTOR_COUNT} predictors, found {len(predictors)}: "
            f"{predictors}"
        )
    nonnumeric = []
    for column in predictors:
        converted = pd.to_numeric(table[column], errors="coerce")
        if converted.isna().any() and table[column].notna().any():
            nonnumeric.append(column)
    if nonnumeric:
        raise ValueError(f"Predictors are not numerically convertible: {nonnumeric}")
    return predictors


def make_patient_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Count class and quality coverage for every patient."""
    rows = []
    for patient_id, group in table.groupby("patient_id", sort=True):
        prehi_count = int((group["Class"] == 0).sum())
        hi_count = int((group["Class"] == 1).sum())
        rows.append(
            {
                "patient_id": patient_id,
                "PreHI_Class0_count": prehi_count,
                "HI_Class1_count": hi_count,
                "total_count": len(group),
                "usable_count": int((group["quality_status"] == "usable").sum()),
                "review_count": int((group["quality_status"] == "review").sum()),
                "has_both_classes": bool(prehi_count and hi_count),
                "small_class_count": bool(
                    min(prehi_count, hi_count) < SMALL_CLASS_COUNT_THRESHOLD
                ),
            }
        )
    counts = pd.DataFrame(rows)
    if len(counts) != table["patient_id"].nunique():
        raise ValueError("Patient count table does not cover every patient")
    if int(counts["total_count"].sum()) != len(table):
        raise ValueError("Patient counts do not represent every clean-table row")
    return counts


def feature_family(feature_name: str) -> str:
    """Return a reliable family label for known ABFE and SampEn features."""
    if feature_name in ABFE_FEATURE_NAMES:
        return "ABFE"
    if feature_name in SAMPEN_FEATURE_NAMES:
        return "SampEn profile"
    return "Unknown"


def feature_statistics(table: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Calculate numeric integrity, distribution, and IQR-outlier statistics."""
    rows = []
    for feature_name in predictors:
        raw_values = table[feature_name]
        values = pd.to_numeric(raw_values, errors="coerce")
        finite_values = values[np.isfinite(values)]
        missing_count = int(values.isna().sum())
        infinite_count = int(np.isinf(values).sum())
        unique_count = int(finite_values.nunique())
        if finite_values.empty:
            q1 = median = q3 = minimum = maximum = mean = std = iqr = np.nan
            outlier_count = 0
            frequent_fraction = np.nan
        else:
            q1 = float(finite_values.quantile(0.25))
            median = float(finite_values.median())
            q3 = float(finite_values.quantile(0.75))
            minimum = float(finite_values.min())
            maximum = float(finite_values.max())
            mean = float(finite_values.mean())
            std = float(finite_values.std(ddof=1))
            iqr = q3 - q1
            lower = q1 - IQR_MULTIPLIER * iqr
            upper = q3 + IQR_MULTIPLIER * iqr
            outlier_count = int(((finite_values < lower) | (finite_values > upper)).sum())
            frequent_fraction = float(finite_values.value_counts(normalize=True).iloc[0])

        rows.append(
            {
                "feature_name": feature_name,
                "dtype": str(raw_values.dtype),
                "non_missing_count": int(values.notna().sum()),
                "missing_count": missing_count,
                "infinite_count": infinite_count,
                "unique_count": unique_count,
                "most_frequent_value_fraction": frequent_fraction,
                "minimum": minimum,
                "q1": q1,
                "median": median,
                "q3": q3,
                "maximum": maximum,
                "mean": mean,
                "standard_deviation": std,
                "iqr": iqr,
                "iqr_outlier_count": outlier_count,
                "is_constant": bool(unique_count <= 1),
                "is_near_constant": bool(
                    (pd.notna(frequent_fraction) and frequent_fraction >= NEAR_CONSTANT_FREQUENCY_THRESHOLD)
                    or unique_count <= NEAR_CONSTANT_UNIQUE_COUNT_THRESHOLD
                ),
                "feature_family": feature_family(feature_name),
            }
        )
    return pd.DataFrame(rows)


def duplicate_feature_pairs(
    table: pd.DataFrame, predictors: list[str]
) -> pd.DataFrame:
    """Find exact and sign-inverted duplicate feature columns."""
    rows = []
    arrays = {
        name: pd.to_numeric(table[name], errors="coerce").to_numpy(float)
        for name in predictors
    }
    for first_index, first_name in enumerate(predictors):
        for second_name in predictors[first_index + 1 :]:
            first = arrays[first_name]
            second = arrays[second_name]
            if np.array_equal(first, second):
                relationship = "exact_duplicate"
            elif np.array_equal(first, -second):
                relationship = "sign_inverted_duplicate"
            else:
                continue
            rows.append(
                {
                    "feature_1": first_name,
                    "feature_2": second_name,
                    "relationship": relationship,
                }
            )
    return pd.DataFrame(rows, columns=["feature_1", "feature_2", "relationship"])


def high_correlation_pairs(
    table: pd.DataFrame, predictors: list[str]
) -> pd.DataFrame:
    """Find high absolute Spearman correlations without pruning features."""
    correlation = table[predictors].corr(method="spearman")
    rows = []
    for first_index, first_name in enumerate(predictors):
        for second_name in predictors[first_index + 1 :]:
            value = correlation.loc[first_name, second_name]
            if pd.notna(value) and abs(value) >= HIGH_CORRELATION_THRESHOLD:
                rows.append(
                    {
                        "feature_1": first_name,
                        "feature_2": second_name,
                        "spearman_correlation": float(value),
                        "absolute_correlation": float(abs(value)),
                    }
                )
    result = pd.DataFrame(
        rows,
        columns=[
            "feature_1",
            "feature_2",
            "spearman_correlation",
            "absolute_correlation",
        ],
    )
    if not result.empty:
        result = result.sort_values(
            "absolute_correlation", ascending=False
        ).reset_index(drop=True)
    return result


def make_summary(
    table: pd.DataFrame,
    patient_counts: pd.DataFrame,
    statistics: pd.DataFrame,
    duplicate_pairs: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    predictors: list[str],
) -> pd.DataFrame:
    """Create the one-row feature-audit summary."""
    missing_count = int(statistics["missing_count"].sum())
    infinite_count = int(statistics["infinite_count"].sum())
    patients_missing_class = int((~patient_counts["has_both_classes"]).sum())
    sign_inverted_count = int(
        (duplicate_pairs["relationship"] == "sign_inverted_duplicate").sum()
    )
    return pd.DataFrame(
        [
            {
                "total_rows": len(table),
                "patient_count": table["patient_id"].nunique(),
                "Class_0_count": int((table["Class"] == 0).sum()),
                "Class_1_count": int((table["Class"] == 1).sum()),
                "usable_count": int((table["quality_status"] == "usable").sum()),
                "review_count": int((table["quality_status"] == "review").sum()),
                "predictor_count": len(predictors),
                "missing_feature_value_count": missing_count,
                "infinite_feature_value_count": infinite_count,
                "constant_feature_count": int(statistics["is_constant"].sum()),
                "near_constant_feature_count": int(statistics["is_near_constant"].sum()),
                "exact_duplicate_pair_count": int(
                    (duplicate_pairs["relationship"] == "exact_duplicate").sum()
                ),
                "sign_inverted_duplicate_pair_count": sign_inverted_count,
                "high_correlation_pair_count": len(correlation_pairs),
                "patients_missing_either_class": patients_missing_class,
            }
        ]
    )


def validate_frozen_list(
    table: pd.DataFrame, predictors: list[str], patient_counts: pd.DataFrame
) -> None:
    """Validate the frozen predictor list and coverage audit outputs."""
    if len(predictors) != EXPECTED_PREDICTOR_COUNT:
        raise ValueError("Frozen predictor list does not contain exactly 48 features")
    if len(set(predictors)) != len(predictors):
        raise ValueError("Frozen predictor list contains duplicates")
    forbidden = set(METADATA_COLUMNS) | {
        "condition",
        "Class",
        "quality_status",
        "quality_reasons",
        "sampen_profile_status",
    }
    forbidden.update(column for column in table.columns if column.startswith("qc_"))
    included_forbidden = sorted(forbidden.intersection(predictors))
    if included_forbidden:
        raise ValueError(f"Frozen list contains excluded columns: {included_forbidden}")
    missing = [feature for feature in predictors if feature not in table.columns]
    if missing:
        raise ValueError(f"Frozen list contains missing table columns: {missing}")
    if int(patient_counts["total_count"].sum()) != len(table):
        raise ValueError("Patient counts do not cover every clean-table row")


def print_summary(summary: pd.DataFrame, patient_counts: pd.DataFrame) -> None:
    """Print the requested concise audit results."""
    row = summary.iloc[0]
    small_patients = patient_counts.loc[
        patient_counts["small_class_count"], "patient_id"
    ].tolist()
    print(f"Rows: {int(row['total_rows'])}")
    print(f"Patients: {int(row['patient_count'])}")
    print(f"Class counts: 0={int(row['Class_0_count'])}, 1={int(row['Class_1_count'])}")
    print(f"Predictors: {int(row['predictor_count'])}")
    print(f"Constant features: {int(row['constant_feature_count'])}")
    print(f"Near-constant features: {int(row['near_constant_feature_count'])}")
    print(
        "Duplicate pairs: "
        f"{int(row['exact_duplicate_pair_count'])} exact, "
        f"{int(row['sign_inverted_duplicate_pair_count'])} sign-inverted"
    )
    print(f"High-correlation pairs: {int(row['high_correlation_pair_count'])}")
    print(f"Every patient has both classes: {int(row['patients_missing_either_class']) == 0}")
    print(f"Patients below class-count threshold: {small_patients}")


def main() -> None:
    """Run the complete feature-table audit and save all requested outputs."""
    # Validate structure and identify the exact ordered predictor list.
    table = load_table()
    predictors = identify_predictors(table)

    # Audit patient/class coverage before any feature-level calculations.
    patient_counts = make_patient_counts(table)

    # Calculate feature distributions, outlier counts, duplicates, and correlations.
    statistics = feature_statistics(table, predictors)
    duplicate_pairs = duplicate_feature_pairs(table, predictors)
    correlation_pairs = high_correlation_pairs(table, predictors)
    summary = make_summary(
        table,
        patient_counts,
        statistics,
        duplicate_pairs,
        correlation_pairs,
        predictors,
    )
    validate_frozen_list(table, predictors, patient_counts)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_ROOT / "feature_audit_summary.csv", index=False)
    patient_counts.to_csv(OUTPUT_ROOT / "patient_class_counts.csv", index=False)
    statistics.to_csv(OUTPUT_ROOT / "feature_statistics.csv", index=False)
    duplicate_pairs.to_csv(
        OUTPUT_ROOT / "exact_duplicate_feature_pairs.csv", index=False
    )
    correlation_pairs.to_csv(OUTPUT_ROOT / "high_correlation_pairs.csv", index=False)
    (OUTPUT_ROOT / "frozen_feature_list.txt").write_text(
        "\n".join(predictors) + "\n", encoding="utf-8"
    )
    print_summary(summary, patient_counts)


if __name__ == "__main__":
    main()
