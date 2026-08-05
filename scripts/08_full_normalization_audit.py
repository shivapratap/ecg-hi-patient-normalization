"""Audit four patient-level normalization strategies across the full cohort.

This script validates normalization mathematics and selects the earliest valid
continuous Class 0 calibration block for every patient. It does not fit models, modify the source modelling table,
or write a permanent combined normalized feature table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "modelling" / "clean_modelling_table.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "data" / "modelling" / "final_feature_list.txt"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "normalization"

CALIBRATION_ROWS_PER_PATIENT = 10
IQR_TOLERANCE = 1e-12
EXPECTED_PATIENT_COUNT = 20
EXPECTED_FEATURE_COUNT = 34
WINDOW_SPACING_SECONDS = 120.0
TIMESTAMP_TOLERANCE_SECONDS = 1.0

NORMALIZATION_STRATEGIES = (
    "none",
    "jackknife_median",
    "calibration_median",
    "calibration_median_iqr",
)


def load_and_validate_inputs() -> tuple[pd.DataFrame, list[str]]:
    """Load the source table and validate the frozen feature contract."""
    table = pd.read_csv(INPUT_PATH)
    feature_names = [
        line.strip()
        for line in FEATURE_LIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected 35 frozen features, found {len(feature_names)}")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Frozen feature list contains duplicate names")

    required_columns = [
        "patient_id",
        "segment_id",
        "Class",
        "start_timestamp",
        "recording_id",
        "window_index",
        *feature_names,
    ]
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise ValueError(f"Source table is missing required columns: {missing}")
    if table["segment_id"].isna().any() or table["segment_id"].duplicated().any():
        raise ValueError("segment_id must be present and unique")
    if not table["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1")

    numeric_features = table[feature_names].apply(pd.to_numeric, errors="coerce")
    if numeric_features.isna().any().any():
        raise ValueError("Frozen feature values must be numeric and non-missing")
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise ValueError("Frozen feature values must all be finite")
    table[feature_names] = numeric_features
    return table, feature_names


def sort_class_zero_rows(table: pd.DataFrame, patient_id: int) -> pd.DataFrame:
    """Sort one patient's Class 0 rows by the required chronology keys."""
    rows = table[(table["patient_id"] == patient_id) & (table["Class"] == 0)].copy()
    if len(rows) < 2:
        raise ValueError(f"Patient {patient_id} has fewer than two Class 0 rows")
    rows["_parsed_start_timestamp"] = pd.to_datetime(
        rows["start_timestamp"], errors="coerce"
    )
    if rows["_parsed_start_timestamp"].isna().any():
        raise ValueError(f"Patient {patient_id} has invalid start_timestamp values")
    rows = rows.sort_values(
        ["_parsed_start_timestamp", "recording_id", "window_index"]
    ).reset_index(drop=True)
    rows["chronological_rank"] = np.arange(1, len(rows) + 1)
    return rows


def is_valid_continuous_block(block: pd.DataFrame) -> bool:
    """Return True when a candidate block is continuous within one recording."""
    if len(block) != CALIBRATION_ROWS_PER_PATIENT:
        return False
    if block["recording_id"].nunique() != 1:
        return False

    window_indices = block["window_index"].to_numpy()
    if not np.array_equal(
        np.diff(window_indices),
        np.ones(CALIBRATION_ROWS_PER_PATIENT - 1),
    ):
        return False

    timestamp_gaps = (
        block["_parsed_start_timestamp"].diff().dt.total_seconds().dropna().to_numpy()
    )
    if len(timestamp_gaps) != CALIBRATION_ROWS_PER_PATIENT - 1:
        return False

    return bool(
        np.all(
            np.abs(timestamp_gaps - WINDOW_SPACING_SECONDS)
            <= TIMESTAMP_TOLERANCE_SECONDS
        )
    )


def find_earliest_continuous_block(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame | None, int | None]:
    """Find the earliest valid continuous calibration block by sliding search."""
    maximum_start = len(candidates) - CALIBRATION_ROWS_PER_PATIENT
    for start_position in range(maximum_start + 1):
        block = candidates.iloc[
            start_position : start_position + CALIBRATION_ROWS_PER_PATIENT
        ].copy()
        if is_valid_continuous_block(block):
            return block, start_position
    return None, None


def select_calibration_rows(
    table: pd.DataFrame, patient_ids: list[int]
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Select the earliest continuous Class 0 calibration block per patient."""
    selected_by_patient: dict[int, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []
    patients_without_valid_block: list[int] = []

    for patient_id in patient_ids:
        candidates = sort_class_zero_rows(table, patient_id)
        if len(candidates) < CALIBRATION_ROWS_PER_PATIENT:
            patients_without_valid_block.append(patient_id)
            selected = None
            start_position = None
        else:
            selected, start_position = find_earliest_continuous_block(candidates)
            if selected is None:
                patients_without_valid_block.append(patient_id)

        if selected is None:
            selected_ids: set[str] = set()
            selected_order: dict[str, int] = {}
            block_recording_count = 0
            within_single_recording = False
            crosses_boundary = False
            consecutive_indices = False
            expected_spacing = False
            timestamp_discontinuity = True
            selected_block_start_rank = np.nan
            skipped_candidate_rows = np.nan
            calibration_block_available = False
        else:
            selected = selected.copy()
            selected["calibration_order"] = np.arange(1, len(selected) + 1)
            selected_by_patient[patient_id] = selected

            selected_ids = set(selected["segment_id"])
            if len(selected_ids) != CALIBRATION_ROWS_PER_PATIENT:
                raise ValueError(
                    f"Selected calibration IDs are not unique for patient {patient_id}"
                )
            if not (selected["Class"] == 0).all():
                raise ValueError(
                    f"Calibration block contains Class 1 for patient {patient_id}"
                )

            block_recording_count = int(selected["recording_id"].nunique())
            within_single_recording = block_recording_count == 1
            crosses_boundary = block_recording_count > 1
            consecutive_indices = bool(
                np.array_equal(
                    np.diff(selected["window_index"].to_numpy()),
                    np.ones(CALIBRATION_ROWS_PER_PATIENT - 1),
                )
            )
            timestamp_gaps = (
                selected["_parsed_start_timestamp"]
                .diff()
                .dt.total_seconds()
                .dropna()
                .to_numpy()
            )
            expected_spacing = bool(
                len(timestamp_gaps) == CALIBRATION_ROWS_PER_PATIENT - 1
                and np.all(
                    np.abs(timestamp_gaps - WINDOW_SPACING_SECONDS)
                    <= TIMESTAMP_TOLERANCE_SECONDS
                )
            )
            timestamp_discontinuity = not expected_spacing
            selected_order = (
                selected.set_index("segment_id")["calibration_order"].to_dict()
            )
            selected_block_start_rank = int(selected["chronological_rank"].iloc[0])
            skipped_candidate_rows = int(start_position)
            calibration_block_available = True

            if not is_valid_continuous_block(selected):
                raise ValueError(
                    f"Internal error: selected calibration block is not continuous "
                    f"for patient {patient_id}"
                )

        for _, row in candidates.iterrows():
            audit_rows.append(
                {
                    "patient_id": patient_id,
                    "segment_id": row["segment_id"],
                    "start_timestamp": row["start_timestamp"],
                    "recording_id": row["recording_id"],
                    "window_index": row["window_index"],
                    "chronological_rank": row["chronological_rank"],
                    "calibration_order": selected_order.get(row["segment_id"], np.nan),
                    "Class": row["Class"],
                    "selected_for_calibration": row["segment_id"] in selected_ids,
                    "calibration_block_available": calibration_block_available,
                    "selected_block_start_chronological_rank": selected_block_start_rank,
                    "candidate_rows_skipped_before_selected_block": skipped_candidate_rows,
                    "selected_block_recording_count": block_recording_count,
                    "selected_block_within_single_recording": within_single_recording,
                    "selected_block_consecutive_window_indices": consecutive_indices,
                    "selected_block_expected_120s_spacing": expected_spacing,
                    "selected_block_crosses_recording_boundary": crosses_boundary,
                    "selected_block_has_timestamp_discontinuity": timestamp_discontinuity,
                }
            )

    calibration_rows = pd.DataFrame(audit_rows)

    if patients_without_valid_block:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        calibration_rows.to_csv(
            OUTPUT_ROOT / "full_calibration_rows.csv", index=False
        )
        patient_list = ", ".join(str(value) for value in patients_without_valid_block)
        raise ValueError(
            "No valid continuous calibration block was found for patient(s): "
            f"{patient_list}. Diagnostic flags were written to "
            "data/normalization/full_calibration_rows.csv."
        )

    return selected_by_patient, calibration_rows


def calculate_calibration_statistics(
    selected_by_patient: dict[int, pd.DataFrame], feature_names: list[str]
) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """Calculate fixed calibration medians and IQRs for every patient."""
    medians = {
        patient_id: rows[feature_names].median()
        for patient_id, rows in selected_by_patient.items()
    }
    iqrs = {
        patient_id: rows[feature_names].quantile(0.75)
        - rows[feature_names].quantile(0.25)
        for patient_id, rows in selected_by_patient.items()
    }
    return medians, iqrs


def make_iqr_fallback_audit(
    patient_ids: list[int], feature_names: list[str], iqrs: dict[int, pd.Series]
) -> pd.DataFrame:
    """Record valid IQRs and temporary fallback scales for every patient-feature."""
    rows = []
    for patient_id in patient_ids:
        for feature_name in feature_names:
            calibration_iqr = float(iqrs[patient_id][feature_name])
            valid_iqr = bool(
                np.isfinite(calibration_iqr) and calibration_iqr > IQR_TOLERANCE
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "feature_name": feature_name,
                    "configured_calibration_rows": CALIBRATION_ROWS_PER_PATIENT,
                    "calibration_iqr": calibration_iqr,
                    "fallback_required": not valid_iqr,
                    "temporary_scale_used": calibration_iqr if valid_iqr else 1.0,
                    "valid_iqr": valid_iqr,
                }
            )
    return pd.DataFrame(rows)


def normalize_patient_rows(
    patient_rows: pd.DataFrame,
    feature_names: list[str],
    selected_calibration: pd.DataFrame,
    strategy: str,
    medians: pd.Series | None = None,
    iqrs: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply one normalization strategy to all rows for one patient."""
    transformed = patient_rows.copy()
    feature_values = patient_rows[feature_names].astype(float)
    selected_ids = set(selected_calibration["segment_id"])
    transformed["is_common_calibration_row"] = patient_rows["segment_id"].isin(selected_ids)
    order_map = selected_calibration.set_index("segment_id")["calibration_order"].to_dict()
    transformed["calibration_order"] = patient_rows["segment_id"].map(order_map)

    if strategy == "none":
        normalized = feature_values.copy()
    elif strategy == "jackknife_median":
        class_zero = patient_rows[patient_rows["Class"] == 0]
        if len(class_zero) < 2:
            raise ValueError("Jackknife requires at least two Class 0 rows")
        class_zero_values = class_zero[feature_names].astype(float)
        normalized = pd.DataFrame(index=patient_rows.index, columns=feature_names, dtype=float)
        for row_index in patient_rows.index:
            if patient_rows.at[row_index, "Class"] == 1:
                reference_values = class_zero_values
            else:
                reference_values = class_zero_values.drop(index=row_index)
            normalized.loc[row_index] = feature_values.loc[row_index] - reference_values.median()
    elif strategy == "calibration_median":
        if medians is None:
            raise ValueError("Calibration medians are required")
        normalized = feature_values - medians
    elif strategy == "calibration_median_iqr":
        if medians is None or iqrs is None:
            raise ValueError("Calibration medians and IQRs are required")
        safe_scales = iqrs.where(
            np.isfinite(iqrs) & (iqrs > IQR_TOLERANCE), 1.0
        )
        # The temporary scale is only for this audit. Final LOSO code must
        # derive a fallback scale from outer-training patients.
        normalized = (feature_values - medians) / safe_scales
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")

    transformed[feature_names] = normalized[feature_names]
    return transformed


def manually_check_jackknife(
    source: pd.DataFrame,
    output: pd.DataFrame,
    feature_names: list[str],
    patient_ids: list[int],
) -> bool:
    """Recompute one Class 0 and one Class 1 jackknife row per patient."""
    for patient_id in patient_ids:
        patient = source[source["patient_id"] == patient_id]
        class_zero = patient[patient["Class"] == 0]
        class_zero_row = class_zero.iloc[0]
        class_one_row = patient[patient["Class"] == 1].iloc[0]
        zero_reference = class_zero.drop(index=class_zero_row.name)[feature_names].median()
        one_reference = class_zero[feature_names].median()
        expected_zero = class_zero_row[feature_names].astype(float) - zero_reference
        expected_one = class_one_row[feature_names].astype(float) - one_reference
        actual_zero = output[output["segment_id"] == class_zero_row["segment_id"]][feature_names].iloc[0]
        actual_one = output[output["segment_id"] == class_one_row["segment_id"]][feature_names].iloc[0]
        if not np.allclose(expected_zero.to_numpy(), actual_zero.to_numpy()):
            return False
        if not np.allclose(expected_one.to_numpy(), actual_one.to_numpy()):
            return False
    return True


def check_calibration_selection(
    selected_by_patient: dict[int, pd.DataFrame], calibration_rows: pd.DataFrame
) -> bool:
    """Verify selected counts, orders, uniqueness, Class 0 membership, and shared rows."""
    for patient_id, selected in selected_by_patient.items():
        if len(selected) != CALIBRATION_ROWS_PER_PATIENT:
            return False
        if list(selected["calibration_order"]) != list(
            range(1, CALIBRATION_ROWS_PER_PATIENT + 1)
        ):
            return False
        if selected["segment_id"].duplicated().any() or not (selected["Class"] == 0).all():
            return False
        if not is_valid_continuous_block(selected):
            return False
        audit = calibration_rows[calibration_rows["patient_id"] == patient_id]
        selected_audit = set(audit.loc[audit["selected_for_calibration"], "segment_id"])
        if selected_audit != set(selected["segment_id"]):
            return False
    return True


def build_normalization_audit(
    source: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    patient_ids: list[int],
    feature_names: list[str],
    jackknife_check: bool,
    calibration_check: bool,
) -> pd.DataFrame:
    """Build one validation row per patient and strategy."""
    rows = []
    for patient_id in patient_ids:
        patient_source = source[source["patient_id"] == patient_id]
        for strategy in NORMALIZATION_STRATEGIES:
            patient_output = outputs[strategy][outputs[strategy]["patient_id"] == patient_id]
            common_calibration_count = int(patient_output["is_common_calibration_row"].sum())
            native_evaluation_count = (
                len(patient_output)
                if strategy in ("none", "jackknife_median")
                else len(patient_output) - common_calibration_count
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "normalization": strategy,
                    "total_rows": len(patient_source),
                    "class_0_rows": int((patient_source["Class"] == 0).sum()),
                    "class_1_rows": int((patient_source["Class"] == 1).sum()),
                    "configured_calibration_rows": CALIBRATION_ROWS_PER_PATIENT,
                    "common_calibration_row_count": common_calibration_count,
                    "common_evaluation_row_count": len(patient_output) - common_calibration_count,
                    "native_evaluation_row_count": native_evaluation_count,
                    "feature_count": len(feature_names),
                    "self_exclusion_check_passed": (
                        jackknife_check if strategy == "jackknife_median" else True
                    ),
                    "calibration_selection_check_passed": calibration_check,
                    "finite_output_check_passed": bool(
                        np.isfinite(patient_output[feature_names].to_numpy()).all()
                    ),
                    "source_row_count_preserved": len(patient_output) == len(patient_source),
                    "unique_segment_ids_preserved": (
                        not patient_output["segment_id"].duplicated().any()
                    ),
                }
            )
    return pd.DataFrame(rows)


def validate_outputs(
    source: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    feature_names: list[str],
    calibration_rows: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """Run final full-cohort integrity checks before writing outputs."""
    for strategy, output in outputs.items():
        if len(output) != len(source):
            raise ValueError(f"Row count changed for {strategy}")
        if output["segment_id"].duplicated().any():
            raise ValueError(f"Segment IDs are not unique for {strategy}")
        if list(output[feature_names].columns) != feature_names:
            raise ValueError(f"Feature order changed for {strategy}")
        if not np.isfinite(output[feature_names].to_numpy()).all():
            raise ValueError(f"Non-finite normalized output for {strategy}")
        if strategy == "none":
            source_values = source.set_index("segment_id")[feature_names].loc[
                output["segment_id"]
            ]
            if not np.array_equal(source_values.to_numpy(), output[feature_names].to_numpy()):
                raise ValueError("None normalization changed source feature values")

    selected_count = int(calibration_rows["selected_for_calibration"].sum())
    expected_selected_count = CALIBRATION_ROWS_PER_PATIENT * source["patient_id"].nunique()
    if selected_count != expected_selected_count:
        raise ValueError("Unexpected total calibration-row selection count")
    if not audit.drop(columns=["patient_id", "normalization"]).all().all():
        raise ValueError("At least one normalization audit validation failed")


def make_summary(
    source: pd.DataFrame,
    patient_ids: list[int],
    calibration_rows: pd.DataFrame,
    fallback_audit: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Create one-row full-cohort normalization summary."""
    selected = calibration_rows[calibration_rows["selected_for_calibration"]]
    patient_flags = calibration_rows.groupby("patient_id").agg(
        calibration_block_available=("calibration_block_available", "first"),
        selected_block_start_rank=("selected_block_start_chronological_rank", "first"),
        crosses_recordings=("selected_block_crosses_recording_boundary", "first"),
        nonconsecutive_indices=(
            "selected_block_consecutive_window_indices",
            lambda values: not bool(values.iloc[0]),
        ),
        timestamp_discontinuity=(
            "selected_block_has_timestamp_discontinuity",
            "first",
        ),
    )
    return pd.DataFrame(
        [
            {
                "total_source_rows": len(source),
                "total_patients": len(patient_ids),
                "class_0_rows": int((source["Class"] == 0).sum()),
                "class_1_rows": int((source["Class"] == 1).sum()),
                "feature_count": EXPECTED_FEATURE_COUNT,
                "calibration_rows_per_patient": CALIBRATION_ROWS_PER_PATIENT,
                "total_selected_calibration_rows": len(selected),
                "patients_with_insufficient_class_0_rows": 0,
                "patients_without_valid_continuous_calibration_block": int((~patient_flags["calibration_block_available"]).sum()),
                "patients_using_later_continuous_calibration_block": int((patient_flags["selected_block_start_rank"] > 1).sum()),
                "patients_with_calibration_block_crossing_recordings": int(patient_flags["crosses_recordings"].sum()),
                "patients_with_nonconsecutive_window_indices": int(patient_flags["nonconsecutive_indices"].sum()),
                "patients_with_timestamp_discontinuity": int(patient_flags["timestamp_discontinuity"].sum()),
                "total_iqr_fallback_events": int(fallback_audit["fallback_required"].sum()),
                "patients_with_any_iqr_fallback": int(fallback_audit.groupby("patient_id")["fallback_required"].any().sum()),
                "all_jackknife_checks_passed": bool(audit.loc[audit["normalization"] == "jackknife_median", "self_exclusion_check_passed"].all()),
                "all_calibration_checks_passed": bool(audit["calibration_selection_check_passed"].all()),
                "all_finite_output_checks_passed": bool(audit["finite_output_check_passed"].all()),
                "all_source_row_counts_preserved": bool(audit["source_row_count_preserved"].all()),
                "all_unique_segment_id_checks_passed": bool(audit["unique_segment_ids_preserved"].all()),
            }
        ]
    )


def main() -> None:
    """Run the full-cohort normalization audit and save four audit files."""
    # Load the immutable source table and frozen ordered predictor list.
    source, feature_names = load_and_validate_inputs()
    patient_ids = sorted(source["patient_id"].unique())
    if len(patient_ids) != EXPECTED_PATIENT_COUNT:
        raise ValueError(f"Expected 20 patients, found {len(patient_ids)}")

    # Select one shared chronological Class 0 calibration block per patient.
    selected_by_patient, calibration_rows = select_calibration_rows(source, patient_ids)
    medians, iqrs = calculate_calibration_statistics(selected_by_patient, feature_names)
    fallback_audit = make_iqr_fallback_audit(patient_ids, feature_names, iqrs)

    # Apply each method in memory; no permanent normalized feature table is written.
    outputs = {}
    for strategy in NORMALIZATION_STRATEGIES:
        patient_outputs = []
        for patient_id in patient_ids:
            patient_rows = source[source["patient_id"] == patient_id]
            if strategy == "calibration_median":
                transformed = normalize_patient_rows(
                    patient_rows,
                    feature_names,
                    selected_by_patient[patient_id],
                    strategy,
                    medians=medians[patient_id],
                )
            elif strategy == "calibration_median_iqr":
                transformed = normalize_patient_rows(
                    patient_rows,
                    feature_names,
                    selected_by_patient[patient_id],
                    strategy,
                    medians=medians[patient_id],
                    iqrs=iqrs[patient_id],
                )
            else:
                transformed = normalize_patient_rows(
                    patient_rows,
                    feature_names,
                    selected_by_patient[patient_id],
                    strategy,
                )
            patient_outputs.append(transformed)
        outputs[strategy] = pd.concat(patient_outputs, ignore_index=True)

    # Validate the mathematics and common calibration-row contract.
    jackknife_check = manually_check_jackknife(
        source, outputs["jackknife_median"], feature_names, patient_ids
    )
    calibration_check = check_calibration_selection(
        selected_by_patient, calibration_rows
    )
    if not jackknife_check:
        raise ValueError("Jackknife manual recomputation failed")
    if not calibration_check:
        raise ValueError("Calibration selection validation failed")
    audit = build_normalization_audit(
        source,
        outputs,
        patient_ids,
        feature_names,
        jackknife_check,
        calibration_check,
    )
    validate_outputs(source, outputs, feature_names, calibration_rows, audit)
    summary = make_summary(source, patient_ids, calibration_rows, fallback_audit, audit)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_ROOT / "full_normalization_audit.csv", index=False)
    calibration_rows.to_csv(OUTPUT_ROOT / "full_calibration_rows.csv", index=False)
    fallback_audit.to_csv(OUTPUT_ROOT / "full_iqr_fallback_audit.csv", index=False)
    summary.to_csv(OUTPUT_ROOT / "full_normalization_summary.csv", index=False)

    print(f"Total patients: {len(patient_ids)}")
    print(f"Total rows: {len(source)}")
    print(f"Class 0 rows: {int((source['Class'] == 0).sum())}")
    print(f"Class 1 rows: {int((source['Class'] == 1).sum())}")
    print(f"Calibration size: {CALIBRATION_ROWS_PER_PATIENT}")
    print("All patients had enough Class 0 rows: True")
    print(
        "Patients using a later continuous calibration block:",
        int(summary.loc[0, "patients_using_later_continuous_calibration_block"]),
    )
    print(
        "Patients with calibration blocks crossing recordings:",
        int(summary.loc[0, "patients_with_calibration_block_crossing_recordings"]),
    )
    print(
        "Patients with timestamp discontinuities:",
        int(summary.loc[0, "patients_with_timestamp_discontinuity"]),
    )
    print(f"Total IQR fallback events: {int(fallback_audit['fallback_required'].sum())}")
    print(
        "Patients with any fallback:",
        int(fallback_audit.groupby("patient_id")["fallback_required"].any().sum()),
    )
    print("All validation checks passed: True")


if __name__ == "__main__":
    main()