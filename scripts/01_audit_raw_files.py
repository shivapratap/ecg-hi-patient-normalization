"""Audit raw Lead II ECG CSV files without changing the source data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
HI_ROOT = RAW_ROOT / "20Patients_HI"
PREHI_ROOT = RAW_ROOT / "20Patients_PreHI"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

EXPECTED_FREQUENCY_HZ = 125.0
WINDOW_SECONDS = 120.0
FREQUENCY_TOLERANCE_HZ = 2.0


def find_csv_files() -> list[Path]:
    """Return all CSV files below the two expected condition folders."""
    files = []
    for folder in (HI_ROOT, PREHI_ROOT):
        files.extend(folder.rglob("*.csv"))
    return sorted(files)


def infer_condition(path: Path) -> str:
    """Infer the condition from the parent folder name."""
    if path.parent == HI_ROOT or HI_ROOT in path.parents:
        return "HI"
    if path.parent == PREHI_ROOT or PREHI_ROOT in path.parents:
        return "PreHI"
    return "Unknown"


def infer_patient_id(path: Path) -> str:
    """Extract the leading numeric patient ID from a filename."""
    match = re.match(r"^(\d+)_", path.name)
    return match.group(1) if match else ""


def identify_columns(columns: list[str]) -> tuple[str, str]:
    """Identify the timestamp and Lead II columns from the actual headers."""
    normalized = {column.strip().lower(): column for column in columns}

    timestamp_column = ""
    for candidate in ("time and date", "timestamp", "time", "date and time"):
        if candidate in normalized:
            timestamp_column = normalized[candidate]
            break
    if not timestamp_column:
        timestamp_column = next(
            (column for column in columns if "time" in column.lower()), ""
        )

    ecg_column = next(
        (column for column in columns if column.strip().upper() == "II"), ""
    )
    return timestamp_column, ecg_column


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse the observed two-digit- and four-digit-year timestamp formats."""
    cleaned = values.astype("string").str.strip().str.strip("[]")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    four_digit_year = cleaned.str[-4:].str.fullmatch(r"\d{4}").fillna(False)
    if four_digit_year.any():
        parsed.loc[four_digit_year] = pd.to_datetime(
            cleaned.loc[four_digit_year],
            format="%H:%M:%S.%f %d/%m/%Y",
            errors="coerce",
        )

    two_digit_year = ~four_digit_year
    if two_digit_year.any():
        parsed.loc[two_digit_year] = pd.to_datetime(
            cleaned.loc[two_digit_year],
            format="%H:%M:%S.%f %d/%m/%y",
            errors="coerce",
        )
    return parsed


def timestamp_metrics(timestamps: pd.Series) -> dict[str, object]:
    """Calculate duration, frequency, ordering, duplicate, and gap metrics."""
    valid = timestamps.dropna()
    if valid.empty:
        return {
            "recording_start": "",
            "recording_end": "",
            "duration_seconds": np.nan,
            "estimated_sampling_frequency_hz": np.nan,
            "duplicate_timestamps": np.nan,
            "non_monotonic_timestamps": np.nan,
            "timestamp_gaps": np.nan,
            "maximum_timestamp_step_seconds": np.nan,
            "timestamp_parse_failures": int(timestamps.isna().sum()),
        }

    steps = timestamps.diff().dt.total_seconds()
    positive_steps = steps[steps > 0]
    typical_step = positive_steps.median()
    estimated_frequency = 1.0 / typical_step if pd.notna(typical_step) else np.nan
    gap_count = (
        int((steps > typical_step * 1.5).sum())
        if pd.notna(typical_step)
        else np.nan
    )

    return {
        "recording_start": valid.min().isoformat(sep=" "),
        "recording_end": valid.max().isoformat(sep=" "),
        "duration_seconds": (valid.max() - valid.min()).total_seconds(),
        "estimated_sampling_frequency_hz": estimated_frequency,
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "non_monotonic_timestamps": int((steps < 0).sum()),
        "timestamp_gaps": gap_count,
        "maximum_timestamp_step_seconds": steps.max(),
        "timestamp_parse_failures": int(timestamps.isna().sum()),
    }


def ecg_metrics(frame: pd.DataFrame, ecg_column: str) -> dict[str, object]:
    """Calculate missing and nonnumeric values for the identified ECG column."""
    if not ecg_column:
        return {"missing_ecg_values": np.nan, "nonnumeric_ecg_values": np.nan}

    numeric_values = pd.to_numeric(frame[ecg_column], errors="coerce")
    nonnumeric = frame[ecg_column].notna() & numeric_values.isna()
    return {
        "missing_ecg_values": int(frame[ecg_column].isna().sum()),
        "nonnumeric_ecg_values": int(nonnumeric.sum()),
    }


def audit_file(path: Path) -> dict[str, object]:
    """Read and audit one CSV file, leaving the file untouched."""
    frame = pd.read_csv(path)
    timestamp_column, ecg_column = identify_columns(frame.columns.tolist())
    timestamps = (
        parse_timestamps(frame[timestamp_column])
        if timestamp_column
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    )
    metrics = timestamp_metrics(timestamps)
    metrics.update(ecg_metrics(frame, ecg_column))

    frequency = metrics["estimated_sampling_frequency_hz"]
    if pd.notna(frequency) and abs(frequency - EXPECTED_FREQUENCY_HZ) <= FREQUENCY_TOLERANCE_HZ:
        samples_per_window = round(frequency * WINDOW_SECONDS)
        complete_windows = len(frame) // samples_per_window
        frequency_check = "approximately_125_hz"
    else:
        complete_windows = np.nan
        frequency_check = "outside_expected_range"

    return {
        "file_name": path.name,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "patient_id": infer_patient_id(path),
        "condition": infer_condition(path),
        "column_names": "|".join(frame.columns),
        "column_count": len(frame.columns),
        "row_count": len(frame),
        "timestamp_column": timestamp_column,
        "lead_ii_column": ecg_column,
        **metrics,
        "sampling_frequency_check": frequency_check,
        "complete_120_second_windows": complete_windows,
    }


def add_structure_checks(audit: pd.DataFrame) -> pd.DataFrame:
    """Flag files whose column structure differs from their condition's mode."""
    audit["structure_differs_from_condition_mode"] = False
    for condition, indices in audit.groupby("condition").groups.items():
        mode = audit.loc[indices, "column_names"].mode()
        if not mode.empty:
            audit.loc[indices, "structure_differs_from_condition_mode"] = (
                audit.loc[indices, "column_names"] != mode.iloc[0]
            )
    return audit


def make_patient_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Create one row per patient summarizing condition coverage and recordings."""
    rows = []
    for patient_id, group in audit.groupby("patient_id", dropna=False):
        hi = group[group["condition"] == "HI"]
        prehi = group[group["condition"] == "PreHI"]
        rows.append(
            {
                "patient_id": patient_id,
                "hi_file_count": len(hi),
                "prehi_file_count": len(prehi),
                "has_hi": bool(len(hi)),
                "has_prehi": bool(len(prehi)),
                "has_both_conditions": bool(len(hi) and len(prehi)),
                "total_file_count": len(group),
                "total_row_count": int(group["row_count"].sum()),
                "minimum_duration_seconds": group["duration_seconds"].min(),
                "maximum_duration_seconds": group["duration_seconds"].max(),
                "total_complete_120_second_windows": group[
                    "complete_120_second_windows"
                ].sum(),
                "file_names": "|".join(group["file_name"]),
            }
        )
    return pd.DataFrame(rows).sort_values("patient_id")


def print_summary(audit: pd.DataFrame, patient_summary: pd.DataFrame) -> None:
    """Print a concise audit summary for the terminal."""
    print(f"Audited {len(audit)} CSV files for {len(patient_summary)} patients.")
    print(
        f"Conditions: {int((audit['condition'] == 'HI').sum())} HI files, "
        f"{int((audit['condition'] == 'PreHI').sum())} PreHI files."
    )
    print(
        "Patients with both conditions: "
        f"{int(patient_summary['has_both_conditions'].sum())}/{len(patient_summary)}."
    )
    print(
        "Estimated sampling frequency: "
        f"{audit['estimated_sampling_frequency_hz'].min():.3f}-"
        f"{audit['estimated_sampling_frequency_hz'].max():.3f} Hz."
    )
    print(
        "Lead II missing values: "
        f"{int(audit['missing_ecg_values'].fillna(0).sum())}; "
        "nonnumeric values: "
        f"{int(audit['nonnumeric_ecg_values'].fillna(0).sum())}."
    )
    print(
        "Timestamp issues: "
        f"{int(audit['duplicate_timestamps'].fillna(0).sum())} duplicates, "
        f"{int(audit['non_monotonic_timestamps'].fillna(0).sum())} non-monotonic rows, "
        f"{int(audit['timestamp_gaps'].fillna(0).sum())} gaps."
    )
    print(
        "Files with non-mode condition structure: "
        f"{int(audit['structure_differs_from_condition_mode'].sum())}."
    )
    print(f"Detailed audit: {OUTPUT_ROOT / 'raw_file_audit.csv'}")
    print(f"Patient summary: {OUTPUT_ROOT / 'patient_file_summary.csv'}")


def main() -> None:
    """Run the raw-file audit and save both requested reports."""
    files = find_csv_files()
    if not files:
        raise FileNotFoundError(f"No CSV files found below {RAW_ROOT}")

    audit = add_structure_checks(pd.DataFrame(audit_file(path) for path in files))
    patient_summary = make_patient_summary(audit)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_ROOT / "raw_file_audit.csv", index=False)
    patient_summary.to_csv(OUTPUT_ROOT / "patient_file_summary.csv", index=False)
    print_summary(audit, patient_summary)


if __name__ == "__main__":
    main()
