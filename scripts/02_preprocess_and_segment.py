"""Filter and segment independent raw Lead II ECG recordings.

Each source CSV is validated, filtered as one complete recording, and split
into non-overlapping 120-second windows. Raw files are never modified and
separate recordings are never concatenated.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
HI_ROOT = RAW_ROOT / "20Patients_HI"
PREHI_ROOT = RAW_ROOT / "20Patients_PreHI"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
WINDOW_ROOT = PROCESSED_ROOT / "windows"
MANIFEST_PATH = PROCESSED_ROOT / "window_manifest.csv"
RECORDING_SUMMARY_PATH = PROCESSED_ROOT / "recording_processing_summary.csv"

SAMPLING_FREQUENCY_HZ = 125.0
FREQUENCY_TOLERANCE_HZ = 2.0
WINDOW_SECONDS = 120
SAMPLES_PER_WINDOW = int(SAMPLING_FREQUENCY_HZ * WINDOW_SECONDS)
FILTER_ORDER = 4
LOW_CUTOFF_HZ = 0.5
HIGH_CUTOFF_HZ = 40.0


def find_source_files() -> list[Path]:
    """Return all raw CSV files from both condition folders."""
    files = []
    for folder in (HI_ROOT, PREHI_ROOT):
        files.extend(folder.rglob("*.csv"))
    return sorted(files)


def infer_metadata(path: Path) -> tuple[str, str, str]:
    """Extract patient ID, condition, and recording ID from a source path."""
    match = re.match(r"^(\d+)_", path.stem)
    if not match:
        raise ValueError(f"Could not infer patient ID from filename: {path.name}")

    if HI_ROOT in path.parents:
        condition = "HI"
    elif PREHI_ROOT in path.parents:
        condition = "PreHI"
    else:
        raise ValueError(f"Source file is outside an expected condition folder: {path}")

    patient_id = match.group(1)
    recording_id = path.stem
    return patient_id, condition, recording_id


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse the two-digit- and four-digit-year timestamp formats in the data."""
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


def validate_recording(
    path: Path, frame: pd.DataFrame, timestamps: pd.Series, ecg_values: pd.Series
) -> float:
    """Validate one recording and return its estimated sampling frequency."""
    if timestamps.empty or timestamps.isna().any():
        raise ValueError(f"Invalid or missing timestamps in {path}")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"Non-monotonic timestamps in {path}")
    if timestamps.duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path}")

    numeric_ecg = pd.to_numeric(ecg_values, errors="coerce")
    if numeric_ecg.isna().any() or not np.isfinite(numeric_ecg.to_numpy()).all():
        raise ValueError(f"Missing or non-finite Lead II values in {path}")

    steps = timestamps.diff().dt.total_seconds().dropna()
    if (steps <= 0).any():
        raise ValueError(f"Non-positive timestamp steps in {path}")
    typical_step = steps.median()
    sampling_frequency = 1.0 / typical_step
    if abs(sampling_frequency - SAMPLING_FREQUENCY_HZ) > FREQUENCY_TOLERANCE_HZ:
        raise ValueError(
            f"Unexpected sampling frequency in {path}: "
            f"{sampling_frequency:.6f} Hz"
        )
    if (steps > typical_step * 1.5).any():
        raise ValueError(f"Unexpected timestamp gap in {path}")

    if len(frame) != len(timestamps):
        raise ValueError(f"Timestamp and ECG lengths differ in {path}")
    return float(sampling_frequency)


def make_filter_coefficients() -> np.ndarray:
    """Create the requested fourth-order Butterworth band-pass filter."""
    return butter(
        FILTER_ORDER,
        [LOW_CUTOFF_HZ, HIGH_CUTOFF_HZ],
        btype="bandpass",
        fs=SAMPLING_FREQUENCY_HZ,
        output="sos",
    )


def filter_complete_recording(ecg_values: pd.Series, filter_coefficients: np.ndarray) -> np.ndarray:
    """Filter one complete recording before any window boundaries are applied."""
    # Filtering the complete recording avoids introducing a filter boundary at
    # every 120-second window. Separate source files remain independent.
    return sosfiltfilt(
        filter_coefficients,
        pd.to_numeric(ecg_values, errors="raise").to_numpy(dtype=float),
    )


def save_windows(
    filtered_signal: np.ndarray,
    timestamps: pd.Series,
    patient_id: str,
    condition: str,
    recording_id: str,
    source_path: Path,
    sampling_frequency: float,
) -> list[dict[str, object]]:
    """Save complete non-overlapping windows and return manifest rows."""
    complete_window_count = len(filtered_signal) // SAMPLES_PER_WINDOW
    patient_window_root = WINDOW_ROOT / patient_id
    patient_window_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for window_index in range(complete_window_count):
        start_sample = window_index * SAMPLES_PER_WINDOW
        end_sample = start_sample + SAMPLES_PER_WINDOW
        segment_id = f"{recording_id}_w{window_index:03d}"
        output_path = patient_window_root / f"{segment_id}.csv"
        window_values = filtered_signal[start_sample:end_sample]

        # Incomplete trailing samples are intentionally discarded: windows are
        # exact 120-second blocks, with no padding or overlap.
        pd.DataFrame({"ecg_lead_ii": window_values}).to_csv(
            output_path, index=False, float_format="%.10g"
        )
        manifest_rows.append(
            {
                "patient_id": patient_id,
                "condition": condition,
                "recording_id": recording_id,
                "segment_id": segment_id,
                "source_file": str(source_path.relative_to(PROJECT_ROOT)),
                "window_index": window_index,
                "start_sample": start_sample,
                "end_sample_exclusive": end_sample,
                "start_timestamp": timestamps.iloc[start_sample].isoformat(sep=" "),
                "end_timestamp": timestamps.iloc[end_sample - 1].isoformat(sep=" "),
                "sampling_frequency_hz": sampling_frequency,
                "duration_seconds": WINDOW_SECONDS,
                "sample_count": len(window_values),
                "output_file": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )
    return manifest_rows


def process_recording(path: Path, filter_coefficients: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate, filter, and segment one independent source recording."""
    patient_id, condition, recording_id = infer_metadata(path)
    frame = pd.read_csv(path, usecols=["Time and date", "II"])
    timestamps = parse_timestamps(frame["Time and date"])
    ecg_values = pd.to_numeric(frame["II"], errors="coerce")
    sampling_frequency = validate_recording(path, frame, timestamps, ecg_values)

    # The source file is an independent recording; never concatenate it with
    # another file or filter across a gap between separate recordings.
    filtered_signal = filter_complete_recording(ecg_values, filter_coefficients)
    manifest_rows = save_windows(
        filtered_signal,
        timestamps,
        patient_id,
        condition,
        recording_id,
        path,
        sampling_frequency,
    )

    complete_window_count = len(manifest_rows)
    saved_sample_count = complete_window_count * SAMPLES_PER_WINDOW
    discarded_samples = len(filtered_signal) - saved_sample_count
    return manifest_rows, {
        "patient_id": patient_id,
        "condition": condition,
        "recording_id": recording_id,
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "input_sample_count": len(filtered_signal),
        "complete_window_count": complete_window_count,
        "saved_sample_count": saved_sample_count,
        "discarded_trailing_sample_count": discarded_samples,
        "discarded_trailing_duration_seconds": discarded_samples / sampling_frequency,
    }


def verify_outputs(manifest: pd.DataFrame, recording_summary: pd.DataFrame) -> None:
    """Verify saved windows, unique IDs, and manifest/summary consistency."""
    if manifest["segment_id"].duplicated().any():
        raise RuntimeError("Segment IDs are not unique")
    if not manifest.empty and (manifest["sample_count"] != SAMPLES_PER_WINDOW).any():
        raise RuntimeError("Manifest contains a window with the wrong sample count")

    output_paths = [PROJECT_ROOT / path for path in manifest["output_file"]]
    if len(output_paths) != len(set(output_paths)):
        raise RuntimeError("Manifest contains duplicate output paths")
    if not all(path.is_file() for path in output_paths):
        raise RuntimeError("Manifest references a missing window CSV")

    for path in output_paths:
        window = pd.read_csv(path)
        if list(window.columns) != ["ecg_lead_ii"]:
            raise RuntimeError(f"Unexpected columns in saved window: {path}")
        values = window["ecg_lead_ii"].to_numpy(dtype=float)
        if len(values) != SAMPLES_PER_WINDOW or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid saved window values: {path}")

    saved_window_files = list(WINDOW_ROOT.glob("*/**/*.csv"))
    if len(manifest) != len(saved_window_files):
        raise RuntimeError(
            "Manifest row count does not equal saved window CSV count: "
            f"{len(manifest)} != {len(saved_window_files)}"
        )
    if len(manifest) != int(recording_summary["complete_window_count"].sum()):
        raise RuntimeError("Manifest count does not match recording summary")


def print_summary(recording_summary: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Print the requested concise processing summary."""
    hi_windows = int((manifest["condition"] == "HI").sum())
    prehi_windows = int((manifest["condition"] == "PreHI").sum())
    discarded = int(recording_summary["discarded_trailing_sample_count"].sum())
    print(f"Processed recordings: {len(recording_summary)}")
    print(f"HI windows: {hi_windows}")
    print(f"PreHI windows: {prehi_windows}")
    print(f"Total windows: {len(manifest)}")
    print(f"Discarded trailing samples: {discarded}")
    print(f"Matches audited expectation of 1,291 windows: {len(manifest) == 1291}")
    print(f"Window manifest: {MANIFEST_PATH}")
    print(f"Recording summary: {RECORDING_SUMMARY_PATH}")


def main() -> None:
    """Process all raw recordings and write windows plus summary files."""
    source_files = find_source_files()
    if not source_files:
        raise FileNotFoundError(f"No source CSV files found below {RAW_ROOT}")

    # Create output directories before processing the recordings.
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    WINDOW_ROOT.mkdir(parents=True, exist_ok=True)
    filter_coefficients = make_filter_coefficients()

    all_manifest_rows = []
    recording_rows = []
    for source_path in source_files:
        manifest_rows, recording_row = process_recording(source_path, filter_coefficients)
        all_manifest_rows.extend(manifest_rows)
        recording_rows.append(recording_row)

    manifest = pd.DataFrame(
        all_manifest_rows,
        columns=[
            "patient_id", "condition", "recording_id", "segment_id", "source_file",
            "window_index", "start_sample", "end_sample_exclusive", "start_timestamp",
            "end_timestamp", "sampling_frequency_hz", "duration_seconds", "sample_count",
            "output_file",
        ],
    )
    recording_summary = pd.DataFrame(
        recording_rows,
        columns=[
            "patient_id", "condition", "recording_id", "source_file", "input_sample_count",
            "complete_window_count", "saved_sample_count", "discarded_trailing_sample_count",
            "discarded_trailing_duration_seconds",
        ],
    )

    manifest.to_csv(MANIFEST_PATH, index=False)
    recording_summary.to_csv(RECORDING_SUMMARY_PATH, index=False)
    verify_outputs(manifest, recording_summary)
    print_summary(recording_summary, manifest)


if __name__ == "__main__":
    main()
