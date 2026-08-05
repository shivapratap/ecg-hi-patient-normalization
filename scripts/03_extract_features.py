"""Extract scalar ECG features from processed windows.

Feature values and metadata are kept in the main feature table, while package
diagnostics and failures are written to a separate table for inspection.
"""

from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path

import amrita_biosignal_feature_engine as abfe
import numpy as np
import pandas as pd
from amrita_biosignal_feature_engine import (
    ExtractorConfig,
    FeatureExtractor,
    WelchPSDConfig,
)
from sampen_profile import sample_entropy_profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "window_manifest.csv"
FEATURE_ROOT = PROJECT_ROOT / "data" / "features"
FEATURES_PATH = FEATURE_ROOT / "ecg_features.csv"
FAILURES_PATH = FEATURE_ROOT / "feature_extraction_failures.csv"
SUMMARY_PATH = FEATURE_ROOT / "feature_extraction_summary.csv"

SAMPLING_FREQUENCY_HZ = 125.0
EXPECTED_SAMPLE_COUNT = 15_000

# These values are expressed in seconds by ABFE's public Welch configuration.
WELCH_WINDOW_LENGTH_SECONDS = 2.0
WELCH_OVERLAP_SECONDS = 1.0
WELCH_DETREND = "constant"
WELCH_SCALING = "density"
WELCH_WINDOW = "hann"

SAMPEN_EMBEDDING_DIMENSION = 2
RUN_SAMPEN_PROFILE = True

# Process every window in the manifest. Set this to 2 for a small smoke test.
MAX_TEST_WINDOWS = None
PROGRESS_EVERY = 10

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

SAMPEN_SCALAR_NAMES = [
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


def package_version(distribution_name: str) -> str:
    """Return an installed distribution version for the summary table."""
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def load_manifest() -> pd.DataFrame:
    """Load and validate the processed-window manifest."""
    manifest = pd.read_csv(MANIFEST_PATH)
    required_columns = METADATA_COLUMNS + ["output_file", "sample_count"]
    missing_columns = [column for column in required_columns if column not in manifest]
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {missing_columns}")
    if manifest["segment_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate segment IDs")
    return manifest


def select_test_windows(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select all windows or one HI and one PreHI smoke-test window."""
    if MAX_TEST_WINDOWS is None:
        return manifest.copy()

    selected_rows = []
    for condition in ("HI", "PreHI"):
        condition_rows = manifest[manifest["condition"] == condition]
        if condition_rows.empty:
            raise ValueError(f"Manifest has no {condition} windows")
        selected_rows.append(condition_rows.iloc[0])

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    if MAX_TEST_WINDOWS is not None:
        selected = selected.iloc[:MAX_TEST_WINDOWS].copy()
    return selected


def load_window(manifest_row: pd.Series) -> np.ndarray:
    """Load one window and verify its exact finite numeric signal contract."""
    window_path = PROJECT_ROOT / str(manifest_row["output_file"])
    window = pd.read_csv(window_path)
    if list(window.columns) != ["ecg_lead_ii"]:
        raise ValueError(f"Unexpected window columns in {window_path}")

    values = pd.to_numeric(window["ecg_lead_ii"], errors="coerce")
    signal = values.to_numpy(dtype=float)
    if len(signal) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} samples in {window_path}, "
            f"found {len(signal)}"
        )
    if not np.isfinite(signal).all():
        raise ValueError(f"Window contains non-finite values: {window_path}")
    return signal


def make_feature_extractor() -> FeatureExtractor:
    """Create one reusable ABFE extractor with explicit Welch settings."""
    welch_config = WelchPSDConfig(
        window_length=WELCH_WINDOW_LENGTH_SECONDS,
        overlap=WELCH_OVERLAP_SECONDS,
        detrend=WELCH_DETREND,
        scaling=WELCH_SCALING,
        window=WELCH_WINDOW,
    )
    return FeatureExtractor(
        ExtractorConfig(
            sampling_frequency=SAMPLING_FREQUENCY_HZ,
            psd=welch_config,
        )
    )


def get_abfe_feature_names() -> list[str]:
    """Return the installed ABFE default scalar feature names."""
    return list(getattr(abfe.feature_registry, "DEFAULT_FEATURE_NAMES", ()))


def verify_feature_name_uniqueness(
    abfe_names: list[str], sampen_names: list[str]
) -> None:
    """Stop before table creation if ABFE and SampEn names overlap."""
    duplicate_names = sorted(set(abfe_names).intersection(sampen_names))
    if duplicate_names:
        raise ValueError(
            "Duplicate feature names found between ABFE and SampEn: "
            f"{duplicate_names}"
        )


def metadata_from_row(manifest_row: pd.Series) -> dict[str, object]:
    """Copy required manifest metadata and add the binary class label."""
    metadata = {column: manifest_row[column] for column in METADATA_COLUMNS}
    metadata["Class"] = 1 if manifest_row["condition"] == "HI" else 0
    return metadata


def extract_abfe_features(
    extractor: FeatureExtractor,
    signal: np.ndarray,
    segment_id: str,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Extract ABFE defaults and convert diagnostics to failure rows."""
    feature_values = {}
    failures = []
    try:
        result = extractor.extract(signal)
        feature_values = {
            name: value for name, value in result.values.items()
        }
        for diagnostic in result.diagnostics:
            failures.append(
                {
                    "segment_id": segment_id,
                    "package": "amrita_biosignal_feature_engine",
                    "feature": diagnostic.feature_name,
                    "error_or_diagnostic": (
                        f"{diagnostic.severity.value}: "
                        f"{diagnostic.code.value}: {diagnostic.message}"
                    ),
                }
            )
    except Exception as error:
        feature_values = {name: np.nan for name in get_abfe_feature_names()}
        failures.append(
            {
                "segment_id": segment_id,
                "package": "amrita_biosignal_feature_engine",
                "feature": "extract",
                "error_or_diagnostic": f"{type(error).__name__}: {error}",
            }
        )
    return feature_values, failures


def extract_sampen_features(
    signal: np.ndarray,
    segment_id: str,
) -> tuple[dict[str, float], list[dict[str, object]], float, str]:
    """Run SampEn profiling once and retain only documented scalar outputs."""
    start_time = time.perf_counter()
    try:
        result = sample_entropy_profile(signal, m=SAMPEN_EMBEDDING_DIMENSION)
        values = {
            name: result[name] for name in SAMPEN_SCALAR_NAMES
        }
        failures = []
        status = "completed"
    except Exception as error:
        values = {name: np.nan for name in SAMPEN_SCALAR_NAMES}
        failures = [
            {
                "segment_id": segment_id,
                "package": "sampen_profile",
                "feature": "sample_entropy_profile",
                "error_or_diagnostic": f"{type(error).__name__}: {error}",
            }
        ]
        status = "failed"
    elapsed_seconds = time.perf_counter() - start_time
    return values, failures, elapsed_seconds, status


def empty_sampen_features() -> dict[str, float]:
    """Return NaN SampEn columns when profiling is disabled."""
    return {name: np.nan for name in SAMPEN_SCALAR_NAMES}


def print_progress(current: int, total: int, segment_id: str) -> None:
    """Print periodic progress with completed and remaining window counts."""
    should_print = current == 1 or current == total or current % PROGRESS_EVERY == 0
    if should_print:
        percent = 100.0 * current / total
        remaining = total - current
        print(
            f"Progress: {current}/{total} windows ({percent:.1f}%) completed; "
            f"{remaining} remaining; current={segment_id}",
            flush=True,
        )


def process_windows(
    selected_manifest: pd.DataFrame,
    extractor: FeatureExtractor,
) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    """Extract features from selected windows and collect failures and timings."""
    feature_rows = []
    failure_rows = []
    sampen_times = []

    total_windows = len(selected_manifest)
    for current_window, (_, manifest_row) in enumerate(
        selected_manifest.iterrows(), start=1
    ):
        segment_id = str(manifest_row["segment_id"])
        try:
            signal = load_window(manifest_row)
        except Exception as error:
            feature_row = metadata_from_row(manifest_row)
            feature_row["sampen_profile_status"] = "not_run"
            feature_row.update(empty_sampen_features())
            feature_rows.append(feature_row)
            failure_rows.append(
                {
                    "segment_id": segment_id,
                    "package": "window_validation",
                    "feature": "ecg_lead_ii",
                    "error_or_diagnostic": f"{type(error).__name__}: {error}",
                }
            )
            print_progress(current_window, total_windows, segment_id)
            continue

        # ABFE default scalar extraction uses the one shared extractor.
        abfe_values, abfe_failures = extract_abfe_features(
            extractor, signal, segment_id
        )
        failure_rows.extend(abfe_failures)

        # SampEn is intentionally switchable because its profile is quadratic
        # in time and memory for a 15,000-sample signal. With full-dataset
        # selection enabled, it is attempted for every selected window.
        if RUN_SAMPEN_PROFILE:
            sampen_values, sampen_failures, elapsed, sampen_status = extract_sampen_features(
                signal, segment_id
            )
            sampen_times.append(elapsed)
            failure_rows.extend(sampen_failures)
        else:
            sampen_values = empty_sampen_features()
            sampen_status = "not_run"

        feature_row = metadata_from_row(manifest_row)
        feature_row["sampen_profile_status"] = sampen_status
        feature_row.update(abfe_values)
        feature_row.update(sampen_values)
        feature_rows.append(feature_row)
        print_progress(current_window, total_windows, segment_id)

    return (
        pd.DataFrame(feature_rows),
        pd.DataFrame(
            failure_rows,
            columns=["segment_id", "package", "feature", "error_or_diagnostic"],
        ),
        sampen_times,
    )


def make_summary(
    manifest: pd.DataFrame,
    selected_manifest: pd.DataFrame,
    feature_table: pd.DataFrame,
    failures: pd.DataFrame,
    sampen_times: list[float],
    total_seconds: float,
) -> pd.DataFrame:
    """Create one-row summary metadata for the extraction run."""
    abfe_names = get_abfe_feature_names()
    abfe_failure_segments = failures.loc[
        failures["package"] == "amrita_biosignal_feature_engine", "segment_id"
    ].nunique()
    sampen_failure_segments = failures.loc[
        failures["package"] == "sampen_profile", "segment_id"
    ].nunique()
    summary = {
        "abfe_package_version": package_version("amrita-biosignal-feature-engine"),
        "sampen_package_version": package_version("sampen-profile"),
        "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
        "manifest_window_count": len(manifest),
        "input_windows_selected": len(selected_manifest),
        "windows_successfully_processed": len(feature_table),
        "windows_with_abfe_diagnostics_or_failures": int(abfe_failure_segments),
        "windows_with_sampen_failures": int(sampen_failure_segments),
        "sampen_completed_windows": int(
            (feature_table["sampen_profile_status"] == "completed").sum()
        ),
        "sampen_not_run_windows": int(
            (feature_table["sampen_profile_status"] == "not_run").sum()
        ),
        "sampen_failed_windows": int(
            (feature_table["sampen_profile_status"] == "failed").sum()
        ),
        "abfe_feature_names": "|".join(abfe_names),
        "sampen_scalar_names": "|".join(SAMPEN_SCALAR_NAMES),
        "welch_window_length_seconds": WELCH_WINDOW_LENGTH_SECONDS,
        "welch_overlap_seconds": WELCH_OVERLAP_SECONDS,
        "welch_detrend": WELCH_DETREND,
        "welch_scaling": WELCH_SCALING,
        "welch_window": WELCH_WINDOW,
        "sampen_embedding_dimension": SAMPEN_EMBEDDING_DIMENSION,
        "sampen_profile_enabled": RUN_SAMPEN_PROFILE,
        "sampen_benchmark_seconds": sampen_times[0] if sampen_times else np.nan,
        "total_execution_seconds": total_seconds,
    }
    return pd.DataFrame([summary])


def print_summary(
    selected_manifest: pd.DataFrame,
    feature_table: pd.DataFrame,
    failures: pd.DataFrame,
    sampen_times: list[float],
) -> None:
    """Print a concise smoke-test summary without printing feature rows."""
    abfe_columns = [column for column in get_abfe_feature_names() if column in feature_table]
    abfe_failures = failures[
        failures["package"] == "amrita_biosignal_feature_engine"
    ]
    sampen_failures = failures[failures["package"] == "sampen_profile"]
    print("Selected test segments:", ", ".join(selected_manifest["segment_id"]))
    print(f"ABFE scalar features returned: {len(abfe_columns)}")
    print(f"ABFE diagnostics/failures: {len(abfe_failures)}")
    if sampen_times:
        print(f"SampEn benchmark seconds: {sampen_times[0]:.3f}")
    else:
        print("SampEn benchmark: not run")
    print(f"SampEn failures: {len(sampen_failures)}")
    for _, row in feature_table[["segment_id", "sampen_profile_status"]].iterrows():
        print(f"SampEn status {row['segment_id']}: {row['sampen_profile_status']}")
    print("Outputs:", FEATURES_PATH, FAILURES_PATH, SUMMARY_PATH)


def main() -> None:
    """Run the configured initial feature-extraction smoke test."""
    start_time = time.perf_counter()
    manifest = load_manifest()
    selected_manifest = select_test_windows(manifest)
    # Check unprefixed names before creating any output table so no feature is
    # silently overwritten if the two packages ever expose the same name.
    verify_feature_name_uniqueness(
        get_abfe_feature_names(), SAMPEN_SCALAR_NAMES
    )
    extractor = make_feature_extractor()
    feature_table, failures, sampen_times = process_windows(
        selected_manifest, extractor
    )

    # Metadata is retained separately from feature values in code so that
    # provenance remains visible and failed numeric features remain NaN rather
    # than being silently replaced with zero.
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(FEATURES_PATH, index=False)
    failures.to_csv(FAILURES_PATH, index=False)
    summary = make_summary(
        manifest,
        selected_manifest,
        feature_table,
        failures,
        sampen_times,
        time.perf_counter() - start_time,
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    print_summary(selected_manifest, feature_table, failures, sampen_times)


if __name__ == "__main__":
    main()
