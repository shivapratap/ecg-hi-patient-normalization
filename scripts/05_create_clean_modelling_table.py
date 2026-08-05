"""Create a QC-aware clean ECG modelling table.

This script joins the complete feature table with the signal-quality table
using ``segment_id`` and prepares the final table for patient normalisation and
model fitting.

Policy
------
- Keep windows labelled ``usable`` or ``review``.
- Exclude only windows labelled ``unusable``.
- Preserve quality status, reasons, and QC measurements for traceability.
- Keep extracted feature names unchanged.
- Prefix QC measurement columns with ``qc_`` so they cannot collide with
  extracted feature names such as ``peak_to_peak``.

This script does not perform imputation, patient normalisation, feature
selection, scaling, or model fitting. Those operations belong inside the
validation folds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "ecg_features.csv"
QUALITY_PATH = (
    PROJECT_ROOT / "data" / "quality" / "window_signal_quality.csv"
)

MODELLING_ROOT = PROJECT_ROOT / "data" / "modelling"
CLEAN_TABLE_PATH = MODELLING_ROOT / "clean_modelling_table.csv"
EXCLUDED_PATH = MODELLING_ROOT / "excluded_unusable_windows.csv"
SUMMARY_PATH = MODELLING_ROOT / "modelling_table_summary.csv"

JOIN_KEY = "segment_id"
EXCLUDED_QUALITY_STATUS = "unusable"

FEATURE_METADATA_COLUMNS = [
    "patient_id",
    "condition",
    "recording_id",
    "segment_id",
    "source_file",
    "window_index",
    "start_timestamp",
    "end_timestamp",
    "Class",
]

# This is a status/provenance field, not a numeric model feature.
FEATURE_STATUS_COLUMNS = [
    "sampen_profile_status",
]

QUALITY_METADATA_COLUMNS = [
    "patient_id",
    "condition",
    "recording_id",
    "segment_id",
]

QUALITY_MEASUREMENT_COLUMNS = [
    "signal_standard_deviation",
    "peak_to_peak",
    "max_absolute_first_difference",
    "longest_flatline_samples",
    "longest_flatline_seconds",
    "clipping_fraction",
    "baseline_power_fraction",
    "high_frequency_power_fraction",
]

QUALITY_STATUS_COLUMNS = [
    "quality_status",
    "quality_reasons",
]

QUALITY_REQUIRED_COLUMNS = (
    QUALITY_METADATA_COLUMNS
    + QUALITY_MEASUREMENT_COLUMNS
    + QUALITY_STATUS_COLUMNS
)

SHARED_METADATA_COLUMNS = [
    "patient_id",
    "condition",
    "recording_id",
]


def read_csv_checked(path: Path, table_name: str) -> pd.DataFrame:
    """Read a CSV file and fail clearly if it is missing or empty."""
    if not path.is_file():
        raise FileNotFoundError(f"{table_name} was not found: {path}")

    table = pd.read_csv(path)
    if table.empty:
        raise ValueError(f"{table_name} is empty: {path}")

    return table


def require_columns(
    table: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Verify that a table contains every required column."""
    missing_columns = [
        column for column in required_columns
        if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{table_name} is missing required columns: {missing_columns}"
        )


def verify_unique_segment_ids(
    table: pd.DataFrame,
    table_name: str,
) -> None:
    """Verify that every segment appears exactly once in a table."""
    if table[JOIN_KEY].isna().any():
        raise ValueError(f"{table_name} contains missing segment IDs")

    duplicate_mask = table[JOIN_KEY].duplicated(keep=False)
    if duplicate_mask.any():
        examples = sorted(
            table.loc[duplicate_mask, JOIN_KEY]
            .astype(str)
            .unique()
            .tolist()
        )[:10]
        raise ValueError(
            f"{table_name} contains duplicate segment IDs. "
            f"Examples: {examples}"
        )


def verify_class_labels(feature_table: pd.DataFrame) -> None:
    """Verify that Class is 0 for PreHI and 1 for HI."""
    expected_class = feature_table["condition"].map(
        {"PreHI": 0, "HI": 1}
    )
    if expected_class.isna().any():
        unexpected = sorted(
            feature_table.loc[
                expected_class.isna(),
                "condition",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"Unexpected condition labels: {unexpected}")

    actual_class = pd.to_numeric(
        feature_table["Class"],
        errors="coerce",
    )
    mismatch_mask = actual_class != expected_class

    if mismatch_mask.any():
        examples = feature_table.loc[
            mismatch_mask,
            ["segment_id", "condition", "Class"],
        ].head(10).to_dict("records")
        raise ValueError(
            "Class labels do not agree with condition labels. "
            f"Examples: {examples}"
        )


def get_extracted_feature_columns(
    feature_table: pd.DataFrame,
) -> list[str]:
    """Return only extracted numeric feature columns.

    Metadata and status/provenance columns are excluded explicitly. This avoids
    mistakenly converting text columns such as ``sampen_profile_status`` to
    NaN when checking feature completeness.
    """
    excluded_columns = set(
        FEATURE_METADATA_COLUMNS + FEATURE_STATUS_COLUMNS
    )
    feature_columns = [
        column for column in feature_table.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No extracted feature columns were found")

    # Feature values should be numeric or convertible to numeric.
    non_numeric_columns = []
    for column in feature_columns:
        converted = pd.to_numeric(
            feature_table[column],
            errors="coerce",
        )
        newly_missing = (
            converted.isna()
            & feature_table[column].notna()
        )
        if newly_missing.any():
            non_numeric_columns.append(column)

    if non_numeric_columns:
        raise ValueError(
            "These extracted feature columns contain nonnumeric values: "
            f"{non_numeric_columns}"
        )

    return feature_columns


def prepare_quality_table(
    quality_table: pd.DataFrame,
) -> pd.DataFrame:
    """Prefix QC measurements with qc_ before joining.

    Quality status and reasons keep their readable names. Metadata are renamed
    temporarily so that agreement with feature-table metadata can be checked.
    """
    rename_map = {
        column: f"{column}_quality"
        for column in SHARED_METADATA_COLUMNS
    }
    rename_map.update({
        column: f"qc_{column}"
        for column in QUALITY_MEASUREMENT_COLUMNS
    })
    return quality_table.rename(columns=rename_map)


def prepare_feature_table(
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Temporarily rename shared feature metadata for comparison."""
    return feature_table.rename(
        columns={
            column: f"{column}_feature"
            for column in SHARED_METADATA_COLUMNS
        }
    )


def normalise_patient_ids(values: pd.Series) -> pd.Series:
    """Represent patient IDs consistently when comparing metadata."""
    return values.astype("string").str.strip()


def verify_shared_metadata(joined: pd.DataFrame) -> None:
    """Verify that feature and QC metadata agree for every segment."""
    mismatches: dict[str, list[str]] = {}

    for column in SHARED_METADATA_COLUMNS:
        feature_column = f"{column}_feature"
        quality_column = f"{column}_quality"

        if column == "patient_id":
            feature_values = normalise_patient_ids(
                joined[feature_column]
            )
            quality_values = normalise_patient_ids(
                joined[quality_column]
            )
        else:
            feature_values = joined[feature_column].astype("string")
            quality_values = joined[quality_column].astype("string")

        mismatch_mask = (
            feature_values.fillna("<missing>")
            != quality_values.fillna("<missing>")
        )

        if mismatch_mask.any():
            mismatches[column] = (
                joined.loc[mismatch_mask, JOIN_KEY]
                .astype(str)
                .head(10)
                .tolist()
            )

    if mismatches:
        raise ValueError(
            "Feature and QC metadata disagree for some segments: "
            f"{mismatches}"
        )


def join_features_and_quality(
    feature_table: pd.DataFrame,
    quality_table: pd.DataFrame,
) -> pd.DataFrame:
    """Perform a strict one-to-one outer join using segment_id."""
    feature_for_join = prepare_feature_table(feature_table)
    quality_for_join = prepare_quality_table(quality_table)

    joined = feature_for_join.merge(
        quality_for_join,
        on=JOIN_KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    feature_without_qc = joined["_merge"] == "left_only"
    qc_without_feature = joined["_merge"] == "right_only"

    if feature_without_qc.any() or qc_without_feature.any():
        feature_only_examples = (
            joined.loc[feature_without_qc, JOIN_KEY]
            .astype(str)
            .head(10)
            .tolist()
        )
        qc_only_examples = (
            joined.loc[qc_without_feature, JOIN_KEY]
            .astype(str)
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Feature and QC tables do not contain identical segment sets. "
            f"Features without QC: {feature_only_examples}; "
            f"QC without features: {qc_only_examples}"
        )

    verify_shared_metadata(joined)

    # Restore one canonical copy of each shared metadata field.
    for column in SHARED_METADATA_COLUMNS:
        joined[column] = joined.pop(f"{column}_feature")
        joined.drop(columns=f"{column}_quality", inplace=True)

    joined.drop(columns="_merge", inplace=True)

    # Guard against ambiguous automatic merge suffixes.
    ambiguous_columns = [
        column for column in joined.columns
        if column.endswith("_x") or column.endswith("_y")
    ]
    if ambiguous_columns:
        raise RuntimeError(
            "Ambiguous merge-generated columns remain: "
            f"{ambiguous_columns}"
        )

    return joined


def arrange_columns(
    joined: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Arrange metadata, QC information, and extracted features clearly."""
    qc_measurement_columns = [
        f"qc_{column}"
        for column in QUALITY_MEASUREMENT_COLUMNS
    ]

    leading_columns = [
        "patient_id",
        "condition",
        "recording_id",
        "segment_id",
        "source_file",
        "window_index",
        "start_timestamp",
        "end_timestamp",
        "Class",
        *FEATURE_STATUS_COLUMNS,
        "quality_status",
        "quality_reasons",
        *qc_measurement_columns,
    ]

    ordered_columns = [
        column for column in leading_columns
        if column in joined.columns
    ]
    ordered_columns.extend([
        column for column in feature_columns
        if column in joined.columns
    ])

    unexpected_columns = [
        column for column in joined.columns
        if column not in ordered_columns
    ]
    if unexpected_columns:
        raise RuntimeError(
            "Unexpected columns remain after assembly: "
            f"{unexpected_columns}"
        )

    return joined[ordered_columns]


def create_summary(
    full_joined_table: pd.DataFrame,
    clean_table: pd.DataFrame,
    excluded_table: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Create a concise audit summary using only true feature columns."""
    status_counts = (
        full_joined_table["quality_status"]
        .value_counts(dropna=False)
        .to_dict()
    )
    class_counts_before = (
        full_joined_table["Class"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    class_counts_after = (
        clean_table["Class"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    numeric_features = clean_table[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    rows_with_any_feature_nan = int(
        numeric_features.isna().any(axis=1).sum()
    )
    total_feature_nan_count = int(
        numeric_features.isna().sum().sum()
    )
    infinite_feature_value_count = int(
        np.isinf(numeric_features.to_numpy(dtype=float)).sum()
    )

    return pd.DataFrame([{
        "feature_input_rows": len(full_joined_table),
        "qc_input_rows": len(full_joined_table),
        "joined_rows": len(full_joined_table),
        "usable_rows": int(status_counts.get("usable", 0)),
        "review_rows": int(status_counts.get("review", 0)),
        "unusable_rows": int(status_counts.get("unusable", 0)),
        "excluded_rows": len(excluded_table),
        "clean_modelling_rows": len(clean_table),
        "class_0_rows_before_qc_exclusion": int(
            class_counts_before.get(0, 0)
        ),
        "class_1_rows_before_qc_exclusion": int(
            class_counts_before.get(1, 0)
        ),
        "class_0_rows_after_qc_exclusion": int(
            class_counts_after.get(0, 0)
        ),
        "class_1_rows_after_qc_exclusion": int(
            class_counts_after.get(1, 0)
        ),
        "retained_review_rows": int(
            (clean_table["quality_status"] == "review").sum()
        ),
        "feature_column_count": len(feature_columns),
        "feature_columns": "|".join(feature_columns),
        "rows_with_any_feature_nan": rows_with_any_feature_nan,
        "total_feature_nan_count": total_feature_nan_count,
        "infinite_feature_value_count": infinite_feature_value_count,
        "exclusion_rule": (
            f"quality_status == '{EXCLUDED_QUALITY_STATUS}'"
        ),
    }])


def verify_final_outputs(
    full_joined_table: pd.DataFrame,
    clean_table: pd.DataFrame,
    excluded_table: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Verify row accounting, uniqueness, and feature-table integrity."""
    for table, table_name in (
        (full_joined_table, "joined table"),
        (clean_table, "clean modelling table"),
        (excluded_table, "excluded table"),
    ):
        verify_unique_segment_ids(table, table_name)

    if len(clean_table) + len(excluded_table) != len(
        full_joined_table
    ):
        raise RuntimeError(
            "Retained plus excluded rows do not equal joined rows"
        )

    if (
        clean_table["quality_status"]
        == EXCLUDED_QUALITY_STATUS
    ).any():
        raise RuntimeError(
            "The clean table still contains unusable windows"
        )

    if not excluded_table.empty and not (
        excluded_table["quality_status"]
        == EXCLUDED_QUALITY_STATUS
    ).all():
        raise RuntimeError(
            "The excluded table contains non-unusable windows"
        )

    if clean_table[JOIN_KEY].isin(
        excluded_table[JOIN_KEY]
    ).any():
        raise RuntimeError(
            "A segment appears in both retained and excluded outputs"
        )

    missing_feature_columns = [
        column for column in feature_columns
        if column not in clean_table.columns
    ]
    if missing_feature_columns:
        raise RuntimeError(
            "Extracted features are missing from the clean table: "
            f"{missing_feature_columns}"
        )

    ambiguous_columns = [
        column for column in clean_table.columns
        if column.endswith("_x") or column.endswith("_y")
    ]
    if ambiguous_columns:
        raise RuntimeError(
            f"Ambiguous columns remain: {ambiguous_columns}"
        )

    qc_columns = [
        f"qc_{column}"
        for column in QUALITY_MEASUREMENT_COLUMNS
    ]
    missing_qc_columns = [
        column for column in qc_columns
        if column not in clean_table.columns
    ]
    if missing_qc_columns:
        raise RuntimeError(
            f"Expected QC columns are missing: {missing_qc_columns}"
        )


def print_summary(
    clean_table: pd.DataFrame,
    excluded_table: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Print a concise terminal summary after successful assembly."""
    numeric_features = clean_table[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    print(f"Clean modelling rows: {len(clean_table)}")
    print(f"Excluded unusable rows: {len(excluded_table)}")
    print(
        "Retained review rows: "
        f"{int((clean_table['quality_status'] == 'review').sum())}"
    )
    print(
        "Class counts after exclusion:",
        clean_table["Class"].value_counts().sort_index().to_dict(),
    )
    print(f"Extracted feature columns: {len(feature_columns)}")
    print(
        "Rows with any missing extracted feature: "
        f"{int(numeric_features.isna().any(axis=1).sum())}"
    )
    print(
        "Total missing extracted feature values: "
        f"{int(numeric_features.isna().sum().sum())}"
    )
    print(f"Clean table: {CLEAN_TABLE_PATH}")
    print(f"Excluded rows: {EXCLUDED_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


def main() -> None:
    """Build and save the corrected QC-aware modelling table."""
    feature_table = read_csv_checked(
        FEATURES_PATH,
        "Feature table",
    )
    quality_table = read_csv_checked(
        QUALITY_PATH,
        "Signal-quality table",
    )

    require_columns(
        feature_table,
        FEATURE_METADATA_COLUMNS + FEATURE_STATUS_COLUMNS,
        "Feature table",
    )
    require_columns(
        quality_table,
        QUALITY_REQUIRED_COLUMNS,
        "Signal-quality table",
    )

    verify_unique_segment_ids(feature_table, "Feature table")
    verify_unique_segment_ids(quality_table, "Signal-quality table")
    verify_class_labels(feature_table)

    feature_columns = get_extracted_feature_columns(feature_table)

    joined_table = join_features_and_quality(
        feature_table,
        quality_table,
    )
    joined_table = arrange_columns(
        joined_table,
        feature_columns,
    )

    excluded_table = joined_table[
        joined_table["quality_status"]
        == EXCLUDED_QUALITY_STATUS
    ].copy()

    clean_table = joined_table[
        joined_table["quality_status"]
        != EXCLUDED_QUALITY_STATUS
    ].copy()

    clean_table.reset_index(drop=True, inplace=True)
    excluded_table.reset_index(drop=True, inplace=True)

    verify_final_outputs(
        joined_table,
        clean_table,
        excluded_table,
        feature_columns,
    )

    summary = create_summary(
        joined_table,
        clean_table,
        excluded_table,
        feature_columns,
    )

    MODELLING_ROOT.mkdir(parents=True, exist_ok=True)
    clean_table.to_csv(CLEAN_TABLE_PATH, index=False)
    excluded_table.to_csv(EXCLUDED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print_summary(
        clean_table,
        excluded_table,
        feature_columns,
    )


if __name__ == "__main__":
    main()