"""Run and validate patient-level ECG feature normalization smoke tests.

The script applies four normalization strategies to all rows for the first
three patients in numeric order. The same chronological Class 0 calibration
block is marked for every strategy to support common-row comparisons. Output
contains only traceability fields and the frozen feature set.
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
SMOKE_TEST_PATIENT_COUNT = 3

NORMALIZATION_STRATEGIES = (
    "none",
    "jackknife_median",
    "calibration_median",
    "calibration_median_iqr",
)


# Only these non-feature fields are written to the smoke-test output. This
# prevents legacy features excluded from final_feature_list.txt from leaking
# into later modelling stages.
TRACEABILITY_COLUMNS = (
    "patient_id",
    "condition",
    "recording_id",
    "segment_id",
    "source_file",
    "window_index",
    "start_timestamp",
    "end_timestamp",
    "Class",
    "sampen_profile_status",
    "quality_status",
    "quality_reasons",
)


def load_inputs() -> tuple[pd.DataFrame, list[str]]:
    """Load the modelling table and exact ordered final feature list."""
    table = pd.read_csv(INPUT_PATH)
    feature_names = [
        line.strip()
        for line in FEATURE_LIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not feature_names or len(feature_names) != 34:
        raise ValueError(f"Expected 34 final features, found {len(feature_names)}")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Final feature list contains duplicate names")
    missing = [name for name in feature_names if name not in table.columns]
    if missing:
        raise ValueError(f"Final features missing from modelling table: {missing}")
    if CALIBRATION_ROWS_PER_PATIENT <= 0:
        raise ValueError("CALIBRATION_ROWS_PER_PATIENT must be positive")
    return table, feature_names


def select_smoke_patients(table: pd.DataFrame) -> list[int]:
    """Select the first configured patient IDs in numeric order."""
    patient_ids = sorted(table["patient_id"].unique())
    if len(patient_ids) < SMOKE_TEST_PATIENT_COUNT:
        raise ValueError("Not enough patients for the requested smoke test")
    return patient_ids[:SMOKE_TEST_PATIENT_COUNT]


def sort_patient_class_zero_rows(table: pd.DataFrame, patient_id: int) -> pd.DataFrame:
    """Return one patient's Class 0 rows in the required chronological order."""
    rows = table[(table["patient_id"] == patient_id) & (table["Class"] == 0)].copy()
    if rows.empty:
        raise ValueError(f"Patient {patient_id} has no Class 0 calibration rows")
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


def select_calibration_rows(
    table: pd.DataFrame, patient_ids: list[int]
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Select one fixed chronological Class 0 calibration block per patient."""
    selected_by_patient = {}
    audit_rows = []
    for patient_id in patient_ids:
        class_zero_rows = sort_patient_class_zero_rows(table, patient_id)
        if len(class_zero_rows) < CALIBRATION_ROWS_PER_PATIENT:
            raise ValueError(
                f"Patient {patient_id} has {len(class_zero_rows)} Class 0 rows; "
                f"at least {CALIBRATION_ROWS_PER_PATIENT} are required"
            )

        selected = class_zero_rows.iloc[:CALIBRATION_ROWS_PER_PATIENT].copy()
        selected["calibration_order"] = np.arange(1, len(selected) + 1)
        selected_by_patient[patient_id] = selected
        selected_ids = set(selected["segment_id"])
        selected_order_map = selected.set_index("segment_id")[
            "calibration_order"
        ].to_dict()
        if len(selected_ids) != CALIBRATION_ROWS_PER_PATIENT:
            raise ValueError(f"Duplicate calibration segment IDs for patient {patient_id}")
        if not (selected["Class"] == 0).all():
            raise ValueError(f"Non-Class 0 row selected for patient {patient_id}")

        for _, row in class_zero_rows.iterrows():
            audit_rows.append(
                {
                    "patient_id": patient_id,
                    "segment_id": row["segment_id"],
                    "start_timestamp": row["start_timestamp"],
                    "recording_id": row["recording_id"],
                    "window_index": row["window_index"],
                    "chronological_rank": row["chronological_rank"],
                    "calibration_order": selected_order_map.get(
                        row["segment_id"], np.nan
                    ),
                    "Class": row["Class"],
                    "selected_for_calibration": row["segment_id"] in selected_ids,
                }
            )

    calibration_rows = pd.DataFrame(audit_rows)
    return selected_by_patient, calibration_rows


def calibration_medians(
    selected_by_patient: dict[int, pd.DataFrame], feature_names: list[str]
) -> dict[int, pd.Series]:
    """Calculate fixed feature-wise medians from selected calibration rows."""
    return {
        patient_id: rows[feature_names].median(axis=0)
        for patient_id, rows in selected_by_patient.items()
    }


def calibration_medians_iqrs(
    selected_by_patient: dict[int, pd.DataFrame], feature_names: list[str]
) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """Calculate fixed feature-wise medians and IQRs from calibration rows."""
    medians = calibration_medians(selected_by_patient, feature_names)
    iqrs = {
        patient_id: rows[feature_names].quantile(0.75)
        - rows[feature_names].quantile(0.25)
        for patient_id, rows in selected_by_patient.items()
    }
    return medians, iqrs


def make_iqr_fallback_audit(
    patient_ids: list[int],
    feature_names: list[str],
    iqrs: dict[int, pd.Series],
) -> pd.DataFrame:
    """Record every patient-feature calibration scale requiring temporary fallback."""
    rows = []
    for patient_id in patient_ids:
        for feature_name in feature_names:
            calibration_iqr = float(iqrs[patient_id][feature_name])
            fallback_required = (
                not np.isfinite(calibration_iqr) or calibration_iqr <= IQR_TOLERANCE
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "feature_name": feature_name,
                    "configured_calibration_rows": CALIBRATION_ROWS_PER_PATIENT,
                    "calibration_iqr": calibration_iqr,
                    "fallback_required": fallback_required,
                    "temporary_scale_used": 1.0 if fallback_required else calibration_iqr,
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
) -> tuple[pd.DataFrame, bool]:
    """Apply one normalization strategy to one patient's rows."""
    transformed = patient_rows.copy()
    feature_values = patient_rows[feature_names].astype(float)
    is_calibration = patient_rows["segment_id"].isin(
        set(selected_calibration["segment_id"])
    )
    calibration_order_map = selected_calibration.set_index("segment_id")[
        "calibration_order"
    ].to_dict()
    # Mark the same rows for all strategies so a later common-row analysis can
    # exclude an identical calibration block from every normalization method.
    transformed["is_common_calibration_row"] = is_calibration
    transformed["calibration_order"] = patient_rows["segment_id"].map(
        calibration_order_map
    )

    if strategy == "none":
        normalized = feature_values.copy()
    elif strategy == "jackknife_median":
        class_zero = patient_rows[patient_rows["Class"] == 0]
        if len(class_zero) < 2:
            raise ValueError("Jackknife normalization requires at least two Class 0 rows")
        normalized = pd.DataFrame(index=patient_rows.index, columns=feature_names, dtype=float)
        class_zero_values = class_zero[feature_names].astype(float)
        for row_index in patient_rows.index:
            if patient_rows.at[row_index, "Class"] == 1:
                reference = class_zero_values
            else:
                reference = class_zero_values.drop(index=row_index)
            normalized.loc[row_index] = feature_values.loc[row_index] - reference.median()
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
        normalized = (feature_values - medians) / safe_scales
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")

    transformed[feature_names] = normalized[feature_names]
    finite_output = bool(np.isfinite(transformed[feature_names].to_numpy()).all())
    if not finite_output:
        raise ValueError(
            f"Non-finite normalized output for patient {patient_rows['patient_id'].iloc[0]}"
        )
    return transformed, finite_output


def validate_jackknife_examples(
    table: pd.DataFrame,
    feature_names: list[str],
    patient_ids: list[int],
    jackknife_output: pd.DataFrame,
) -> bool:
    """Manually verify one Class 0 and one Class 1 jackknife calculation per patient."""
    for patient_id in patient_ids:
        patient_rows = table[table["patient_id"] == patient_id]
        class_zero = patient_rows[patient_rows["Class"] == 0]
        class_one = patient_rows[patient_rows["Class"] == 1]
        class_zero_example = class_zero.iloc[0]
        class_one_example = class_one.iloc[0]
        zero_reference = class_zero.drop(index=class_zero_example.name)[feature_names].median()
        one_reference = class_zero[feature_names].median()
        zero_expected = class_zero_example[feature_names].astype(float) - zero_reference
        one_expected = class_one_example[feature_names].astype(float) - one_reference
        zero_actual = jackknife_output[
            jackknife_output["segment_id"] == class_zero_example["segment_id"]
        ][feature_names].iloc[0]
        one_actual = jackknife_output[
            jackknife_output["segment_id"] == class_one_example["segment_id"]
        ][feature_names].iloc[0]
        if not np.allclose(
            zero_expected.to_numpy(),
            zero_actual.to_numpy(),
        ):
            return False
        if not np.allclose(
            one_expected.to_numpy(),
            one_actual.to_numpy(),
        ):
            return False
    return True


def validate_calibration_selection(
    selected_by_patient: dict[int, pd.DataFrame], calibration_rows: pd.DataFrame
) -> bool:
    """Verify chronological ordering, selected-row counts, and shared calibration rows."""
    for patient_id, selected in selected_by_patient.items():
        if len(selected) != CALIBRATION_ROWS_PER_PATIENT:
            return False
        if list(selected["calibration_order"]) != list(
            range(1, CALIBRATION_ROWS_PER_PATIENT + 1)
        ):
            return False
        if selected["segment_id"].duplicated().any() or not (selected["Class"] == 0).all():
            return False
        patient_audit = calibration_rows[calibration_rows["patient_id"] == patient_id]
        selected_audit_ids = set(
            patient_audit.loc[patient_audit["selected_for_calibration"], "segment_id"]
        )
        if selected_audit_ids != set(selected["segment_id"]):
            return False
    return True


def make_normalization_audit(
    smoke_table: pd.DataFrame,
    feature_names: list[str],
    selected_by_patient: dict[int, pd.DataFrame],
    outputs: dict[str, pd.DataFrame],
    jackknife_check: bool,
    calibration_check: bool,
) -> pd.DataFrame:
    """Create one validation row per patient and normalization strategy."""
    rows = []
    for patient_id in sorted(selected_by_patient):
        patient_source = smoke_table[smoke_table["patient_id"] == patient_id]
        calibration_count = CALIBRATION_ROWS_PER_PATIENT
        for strategy in NORMALIZATION_STRATEGIES:
            patient_output = outputs[strategy]
            patient_output = patient_output[patient_output["patient_id"] == patient_id]
            rows.append(
                {
                    "patient_id": patient_id,
                    "normalization": strategy,
                    "total_rows": len(patient_source),
                    "class_0_rows": int((patient_source["Class"] == 0).sum()),
                    "class_1_rows": int((patient_source["Class"] == 1).sum()),
                    "configured_calibration_rows": CALIBRATION_ROWS_PER_PATIENT,
                    "common_calibration_row_count": int(patient_output["is_common_calibration_row"].sum()),
                    "evaluation_row_count": int((~patient_output["is_common_calibration_row"]).sum()),
                    "feature_count": len(feature_names),
                    "self_exclusion_check_passed": (
                        jackknife_check if strategy == "jackknife_median" else True
                    ),
                    "calibration_selection_check_passed": calibration_check,
                    "finite_output_check_passed": bool(
                        np.isfinite(patient_output[feature_names].to_numpy()).all()
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
    fallback_audit: pd.DataFrame,
) -> None:
    """Run final smoke-test integrity checks before writing outputs."""
    for strategy, output in outputs.items():
        if output["segment_id"].duplicated().any():
            raise ValueError(f"Duplicate segment IDs in {strategy} output")
        if list(output[feature_names].columns) != feature_names:
            raise ValueError(f"Feature order changed in {strategy} output")
        if not np.isfinite(output[feature_names].to_numpy()).all():
            raise ValueError(f"Non-finite values in {strategy} output")
        if "is_common_calibration_row" not in output.columns:
            raise ValueError(
                f"Common calibration-row marker missing in {strategy} output"
            )
        if strategy == "none":
            source_values = source.set_index("segment_id")[feature_names].loc[
                output["segment_id"]
            ]
            if not np.array_equal(
                source_values.to_numpy(), output[feature_names].to_numpy()
            ):
                raise ValueError("None normalization changed feature values")

    # The calibration audit intentionally contains candidates once per patient,
    # not once per normalization strategy.
    expected_calibration_rows = (
        CALIBRATION_ROWS_PER_PATIENT * len(outputs["none"]["patient_id"].unique())
    )
    if len(calibration_rows[calibration_rows["selected_for_calibration"]]) != expected_calibration_rows:
        raise ValueError("Unexpected calibration-row selection count")
    if not audit["finite_output_check_passed"].all():
        raise ValueError("A normalization audit row failed finite-output validation")
    if not fallback_audit["temporary_scale_used"].apply(np.isfinite).all():
        raise ValueError("IQR fallback audit contains non-finite temporary scales")


def main() -> None:
    """Run the three-patient normalization smoke test and save audit files."""
    table, feature_names = load_inputs()
    patient_ids = select_smoke_patients(table)
    smoke_table = table[table["patient_id"].isin(patient_ids)].copy()
    selected_by_patient, calibration_rows = select_calibration_rows(table, patient_ids)
    medians = calibration_medians(selected_by_patient, feature_names)
    iqr_medians, iqrs = calibration_medians_iqrs(selected_by_patient, feature_names)
    fallback_audit = make_iqr_fallback_audit(patient_ids, feature_names, iqrs)

    outputs = {}
    for strategy in NORMALIZATION_STRATEGIES:
        transformed_patients = []
        for patient_id in patient_ids:
            patient_rows = smoke_table[smoke_table["patient_id"] == patient_id]
            if strategy == "calibration_median":
                transformed, _ = normalize_patient_rows(
                    patient_rows,
                    feature_names,
                    selected_by_patient[patient_id],
                    strategy,
                    medians=iqr_medians[patient_id],
                )
            elif strategy == "calibration_median_iqr":
                transformed, _ = normalize_patient_rows(
                    patient_rows,
                    feature_names,
                    selected_by_patient[patient_id],
                    strategy,
                    medians=iqr_medians[patient_id],
                    iqrs=iqrs[patient_id],
                )
            else:
                transformed, _ = normalize_patient_rows(
                    patient_rows,
                    feature_names,
                    selected_by_patient[patient_id],
                    strategy,
                )
            transformed["normalization"] = strategy
            transformed_patients.append(transformed)
        outputs[strategy] = pd.concat(transformed_patients, ignore_index=True)

    jackknife_check = validate_jackknife_examples(
        table, feature_names, patient_ids, outputs["jackknife_median"]
    )
    calibration_check = validate_calibration_selection(
        selected_by_patient, calibration_rows
    )
    if not jackknife_check:
        raise ValueError("Jackknife self-exclusion or HI-centering check failed")
    if not calibration_check:
        raise ValueError("Calibration ordering or selection check failed")

    audit = make_normalization_audit(
        smoke_table,
        feature_names,
        selected_by_patient,
        outputs,
        jackknife_check,
        calibration_check,
    )
    validate_outputs(
        smoke_table,
        outputs,
        feature_names,
        calibration_rows,
        audit,
        fallback_audit,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    missing_traceability = [
        column for column in TRACEABILITY_COLUMNS if column not in table.columns
    ]
    if missing_traceability:
        raise ValueError(
            f"Required traceability columns missing from modelling table: "
            f"{missing_traceability}"
        )

    smoke_columns = list(TRACEABILITY_COLUMNS) + [
        "normalization",
        "is_common_calibration_row",
        "calibration_order",
    ] + feature_names
    smoke_output = pd.concat(
        [outputs[strategy] for strategy in NORMALIZATION_STRATEGIES], ignore_index=True
    )
    smoke_output[smoke_columns].to_csv(
        OUTPUT_ROOT / "normalization_smoke_test.csv", index=False
    )
    audit.to_csv(OUTPUT_ROOT / "normalization_audit.csv", index=False)
    calibration_rows.to_csv(OUTPUT_ROOT / "calibration_rows.csv", index=False)
    fallback_audit.to_csv(OUTPUT_ROOT / "iqr_fallback_audit.csv", index=False)

    print(f"Selected patients: {patient_ids}")
    print(
        "Rows per patient:",
        smoke_table.groupby("patient_id").size().to_dict(),
    )
    print(
        "Class 0 counts:",
        smoke_table[smoke_table["Class"] == 0].groupby("patient_id").size().to_dict(),
    )
    print(
        "Class 1 counts:",
        smoke_table[smoke_table["Class"] == 1].groupby("patient_id").size().to_dict(),
    )
    print(f"Configured calibration size: {CALIBRATION_ROWS_PER_PATIENT}")
    print(
        "Selected calibration rows per patient:",
        calibration_rows[calibration_rows["selected_for_calibration"]]
        .groupby("patient_id")
        .size()
        .to_dict(),
    )
    print(f"IQR fallback events: {int(fallback_audit['fallback_required'].sum())}")
    print("All validation checks passed: True")


if __name__ == "__main__":
    main()