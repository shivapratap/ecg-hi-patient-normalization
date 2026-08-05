"""Audit all saved ECG windows for obvious signal-quality problems.

The script reads every 120-second Lead II window referenced by the processed
window manifest, calculates simple signal-quality measures, derives
label-independent review thresholds from the full dataset, and assigns each
window one of three statuses: usable, review, or unusable.

It never modifies or deletes the ECG window files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "window_manifest.csv"
QUALITY_ROOT = PROJECT_ROOT / "data" / "quality"
QUALITY_PATH = QUALITY_ROOT / "window_signal_quality.csv"
SUMMARY_PATH = QUALITY_ROOT / "signal_quality_summary.csv"
PLOT_ROOT = QUALITY_ROOT / "review_plots"

SAMPLING_FREQUENCY_HZ = 125.0
EXPECTED_SAMPLE_COUNT = 15_000

# None means audit every window in the manifest.
MAX_TEST_WINDOWS = None

# Flatline detection is based on very small sample-to-sample changes.
# The tolerance is 0.1% of the window's peak-to-peak range, with a small
# absolute floor for numerical stability.
NEAR_FLAT_DIFFERENCE_FRACTION = 0.001
MINIMUM_NEAR_FLAT_TOLERANCE = 1e-6
FLATLINE_SECONDS = 2.0

# Clipping is assessed separately by counting samples close to the observed
# window minimum or maximum.
NEAR_EXTREME_SCALE_FRACTION = 0.001
CLIPPING_FRACTION_THRESHOLD = 0.01

# Welch settings are fixed so PSD-derived measures are reproducible.
PSD_NPERSEG = 4096
PSD_NOVERLAP = 2048
BASELINE_CUTOFF_HZ = 0.5
HIGH_FREQUENCY_CUTOFF_HZ = 30.0

# Dataset-derived review thresholds use median + multiplier × MAD.
ROBUST_THRESHOLD_MAD_MULTIPLIER = 5.0

QUALITY_COLUMNS = [
    "patient_id",
    "condition",
    "recording_id",
    "segment_id",
    "signal_standard_deviation",
    "peak_to_peak",
    "max_absolute_first_difference",
    "longest_flatline_samples",
    "longest_flatline_seconds",
    "clipping_fraction",
    "baseline_power_fraction",
    "high_frequency_power_fraction",
    "quality_status",
    "quality_reasons",
]


def load_manifest() -> pd.DataFrame:
    """Load the processed-window manifest and validate its required fields."""
    manifest = pd.read_csv(MANIFEST_PATH)
    required_columns = [
        "patient_id",
        "condition",
        "recording_id",
        "segment_id",
        "output_file",
    ]
    missing_columns = [
        column for column in required_columns if column not in manifest.columns
    ]
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {missing_columns}")
    if manifest["segment_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate segment IDs")
    return manifest


def select_windows(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return all windows, or a deterministic balanced subset for testing."""
    if MAX_TEST_WINDOWS is None:
        return manifest.copy().reset_index(drop=True)

    if MAX_TEST_WINDOWS < 2 or MAX_TEST_WINDOWS % 2:
        raise ValueError("MAX_TEST_WINDOWS must be even and at least 2")

    patient_ids = sorted(manifest["patient_id"].astype(str).unique(), key=int)
    patient_ids = patient_ids[: MAX_TEST_WINDOWS // 2]
    selected_rows = []

    for patient_id in patient_ids:
        for condition in ("HI", "PreHI"):
            matches = manifest[
                (manifest["patient_id"].astype(str) == patient_id)
                & (manifest["condition"] == condition)
            ]
            if not matches.empty:
                selected_rows.append(matches.iloc[0])

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    if len(selected) != MAX_TEST_WINDOWS:
        raise ValueError(
            f"Expected {MAX_TEST_WINDOWS} test windows, found {len(selected)}"
        )
    return selected


def load_signal(manifest_row: pd.Series) -> np.ndarray:
    """Load one window and verify its exact finite numeric signal contract."""
    window_path = PROJECT_ROOT / str(manifest_row["output_file"])
    frame = pd.read_csv(window_path)

    if list(frame.columns) != ["ecg_lead_ii"]:
        raise ValueError(f"Unexpected columns in {window_path}")

    values = pd.to_numeric(frame["ecg_lead_ii"], errors="coerce")
    signal = values.to_numpy(dtype=float)

    if len(signal) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} samples in {window_path}; "
            f"found {len(signal)}"
        )
    if not np.isfinite(signal).all():
        raise ValueError(f"Signal contains non-finite values: {window_path}")

    return signal


def longest_true_run(mask: np.ndarray) -> int:
    """Return the length of the longest consecutive True run."""
    longest_run = 0
    current_run = 0

    for value in mask:
        current_run = current_run + 1 if value else 0
        longest_run = max(longest_run, current_run)

    return longest_run


def calculate_power_fraction(
    frequencies: np.ndarray,
    power: np.ndarray,
    low_frequency: float,
    high_frequency: float | None,
) -> float:
    """Integrate PSD power over a band and divide by total PSD power."""
    total_power = np.trapezoid(power, frequencies)
    if total_power <= 0 or not np.isfinite(total_power):
        return np.nan

    band_mask = frequencies >= low_frequency
    if high_frequency is not None:
        band_mask &= frequencies < high_frequency

    if np.count_nonzero(band_mask) < 2:
        return np.nan

    band_power = np.trapezoid(power[band_mask], frequencies[band_mask])
    return float(band_power / total_power)


def calculate_signal_measures(signal: np.ndarray) -> dict[str, float]:
    """Calculate amplitude, flatline, clipping, and PSD quality measures."""
    signal_minimum = float(np.min(signal))
    signal_maximum = float(np.max(signal))
    peak_to_peak = signal_maximum - signal_minimum

    # A true flatline is a sustained sequence of almost unchanged consecutive
    # samples. It is therefore detected from first differences, not from
    # samples lying near the window minimum or maximum.
    flatline_tolerance = max(
        MINIMUM_NEAR_FLAT_TOLERANCE,
        NEAR_FLAT_DIFFERENCE_FRACTION * peak_to_peak,
    )
    absolute_differences = np.abs(np.diff(signal))
    near_flat_steps = absolute_differences <= flatline_tolerance
    longest_flatline_steps = longest_true_run(near_flat_steps)
    longest_flatline_samples = (
        longest_flatline_steps + 1 if longest_flatline_steps > 0 else 0
    )

    # Clipping is evaluated separately using repeated samples close to the
    # observed minimum or maximum of the window.
    near_extreme_tolerance = max(
        MINIMUM_NEAR_FLAT_TOLERANCE,
        NEAR_EXTREME_SCALE_FRACTION * peak_to_peak,
    )
    near_extreme = (
        np.abs(signal - signal_minimum) <= near_extreme_tolerance
    ) | (
        np.abs(signal - signal_maximum) <= near_extreme_tolerance
    )

    frequencies, power = welch(
        signal,
        fs=SAMPLING_FREQUENCY_HZ,
        nperseg=min(PSD_NPERSEG, len(signal)),
        noverlap=min(
            PSD_NOVERLAP,
            min(PSD_NPERSEG, len(signal)) - 1,
        ),
    )

    return {
        "signal_standard_deviation": float(np.std(signal)),
        "peak_to_peak": peak_to_peak,
        "max_absolute_first_difference": float(
            np.max(absolute_differences)
        ),
        "longest_flatline_samples": float(longest_flatline_samples),
        "longest_flatline_seconds": (
            longest_flatline_samples / SAMPLING_FREQUENCY_HZ
        ),
        "clipping_fraction": float(near_extreme.mean()),
        "baseline_power_fraction": calculate_power_fraction(
            frequencies,
            power,
            0.0,
            BASELINE_CUTOFF_HZ,
        ),
        "high_frequency_power_fraction": calculate_power_fraction(
            frequencies,
            power,
            HIGH_FREQUENCY_CUTOFF_HZ,
            None,
        ),
    }


def robust_upper_threshold(values: pd.Series) -> float:
    """Return a label-independent upper threshold using median plus MAD."""
    finite_values = values[np.isfinite(values)].astype(float)
    if finite_values.empty:
        return np.nan

    median_value = float(finite_values.median())
    mad_value = float(
        np.median(np.abs(finite_values.to_numpy() - median_value))
    )

    # If the MAD is zero, use the observed maximum so identical values do not
    # cause every tiny numerical deviation to be flagged.
    if mad_value == 0:
        return float(finite_values.max())

    return median_value + ROBUST_THRESHOLD_MAD_MULTIPLIER * mad_value


def derive_thresholds(measures: pd.DataFrame) -> dict[str, float]:
    """Derive quality thresholds from the complete observed dataset."""
    finite_standard_deviations = measures.loc[
        np.isfinite(measures["signal_standard_deviation"]),
        "signal_standard_deviation",
    ]
    if finite_standard_deviations.empty:
        raise ValueError("No valid windows are available for threshold derivation")

    median_standard_deviation = float(finite_standard_deviations.median())

    return {
        "near_zero_standard_deviation": max(
            1e-6,
            0.01 * median_standard_deviation,
        ),
        "flatline_samples": int(
            FLATLINE_SECONDS * SAMPLING_FREQUENCY_HZ
        ),
        "clipping_fraction": CLIPPING_FRACTION_THRESHOLD,
        "max_absolute_first_difference": robust_upper_threshold(
            measures["max_absolute_first_difference"]
        ),
        "baseline_power_fraction": robust_upper_threshold(
            measures["baseline_power_fraction"]
        ),
        "high_frequency_power_fraction": robust_upper_threshold(
            measures["high_frequency_power_fraction"]
        ),
    }


def classify_quality(
    measures: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[str, str]:
    """Assign a quality label and explicit reasons from calculated measures."""
    unusable_reasons: list[str] = []
    review_reasons: list[str] = []

    if measures.get("invalid_signal", False):
        unusable_reasons.append("invalid_or_nonfinite_signal")

    if (
        measures["signal_standard_deviation"]
        <= thresholds["near_zero_standard_deviation"]
    ):
        unusable_reasons.append("nearly_zero_variation")

    if (
        measures["longest_flatline_samples"]
        >= thresholds["flatline_samples"]
    ):
        unusable_reasons.append("sustained_flatline")

    if (
        measures["clipping_fraction"]
        >= thresholds["clipping_fraction"]
    ):
        unusable_reasons.append("repeated_clipping")

    if (
        measures["max_absolute_first_difference"]
        > thresholds["max_absolute_first_difference"]
    ):
        review_reasons.append("abrupt_jumps")

    if (
        measures["baseline_power_fraction"]
        > thresholds["baseline_power_fraction"]
    ):
        review_reasons.append("high_baseline_drift")

    if (
        measures["high_frequency_power_fraction"]
        > thresholds["high_frequency_power_fraction"]
    ):
        review_reasons.append("high_frequency_noise")

    all_reasons = unusable_reasons + review_reasons

    if unusable_reasons:
        status = "unusable"
    elif review_reasons:
        status = "review"
    else:
        status = "usable"

    reasons_text = "|".join(all_reasons) if all_reasons else "none"
    return status, reasons_text


def audit_window(
    manifest_row: pd.Series,
    thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, object], np.ndarray | None]:
    """Audit one window and return its result row plus signal for plotting."""
    metadata = {
        "patient_id": manifest_row["patient_id"],
        "condition": manifest_row["condition"],
        "recording_id": manifest_row["recording_id"],
        "segment_id": manifest_row["segment_id"],
    }

    try:
        signal = load_signal(manifest_row)
        measures = calculate_signal_measures(signal)

        if thresholds is None:
            return {**metadata, **measures}, signal

        status, reasons = classify_quality(measures, thresholds)
        return {
            **metadata,
            **measures,
            "quality_status": status,
            "quality_reasons": reasons,
        }, signal

    except Exception as error:
        if thresholds is None:
            return {
                **metadata,
                "signal_standard_deviation": np.nan,
                "peak_to_peak": np.nan,
                "max_absolute_first_difference": np.nan,
                "longest_flatline_samples": np.nan,
                "longest_flatline_seconds": np.nan,
                "clipping_fraction": np.nan,
                "baseline_power_fraction": np.nan,
                "high_frequency_power_fraction": np.nan,
                "invalid_signal": True,
                "validation_error": f"{type(error).__name__}: {error}",
            }, None

        return {
            **metadata,
            "signal_standard_deviation": np.nan,
            "peak_to_peak": np.nan,
            "max_absolute_first_difference": np.nan,
            "longest_flatline_samples": np.nan,
            "longest_flatline_seconds": np.nan,
            "clipping_fraction": np.nan,
            "baseline_power_fraction": np.nan,
            "high_frequency_power_fraction": np.nan,
            "quality_status": "unusable",
            "quality_reasons": "invalid_or_nonfinite_signal",
        }, None


def calculate_review_score(
    row: pd.Series,
    thresholds: dict[str, float],
) -> float:
    """Return a simple score used only to prioritise review plots."""
    ratios = []

    for measure_name in (
        "max_absolute_first_difference",
        "baseline_power_fraction",
        "high_frequency_power_fraction",
    ):
        threshold = thresholds[measure_name]
        value = row[measure_name]

        if np.isfinite(value) and np.isfinite(threshold) and threshold > 0:
            ratios.append(float(value / threshold))

    return max(ratios, default=0.0)


def select_plot_rows(
    quality: pd.DataFrame,
    status: str,
    limit: int,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Select the most informative rows for each plot category."""
    selected = quality[quality["quality_status"] == status].copy()
    if selected.empty:
        return selected

    if status == "review":
        selected["plot_priority"] = selected.apply(
            calculate_review_score,
            axis=1,
            thresholds=thresholds,
        )
        return selected.sort_values(
            "plot_priority",
            ascending=False,
        ).head(limit)

    if status == "unusable":
        return selected.sort_values(
            [
                "longest_flatline_seconds",
                "clipping_fraction",
                "signal_standard_deviation",
            ],
            ascending=[False, False, True],
        ).head(limit)

    return selected.head(limit)


def make_plots(
    quality: pd.DataFrame,
    signals: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> None:
    """Save plots for selected unusable, review, and usable windows."""
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    for status, limit in (
        ("unusable", 10),
        ("review", 10),
        ("usable", 3),
    ):
        selected_rows = select_plot_rows(
            quality,
            status,
            limit,
            thresholds,
        )

        for _, row in selected_rows.iterrows():
            segment_id = str(row["segment_id"])
            signal = signals.get(segment_id)
            if signal is None:
                continue

            time_axis = np.arange(len(signal)) / SAMPLING_FREQUENCY_HZ
            first_ten_seconds = int(10 * SAMPLING_FREQUENCY_HZ)

            figure, axes = plt.subplots(
                2,
                1,
                figsize=(12, 7),
                constrained_layout=True,
            )

            axes[0].plot(time_axis, signal, linewidth=0.5)
            axes[0].set_title(
                f"{segment_id} — full 120 seconds — {status}"
            )
            axes[0].set_xlabel("Time (seconds)")
            axes[0].set_ylabel("Lead II")

            axes[1].plot(
                time_axis[:first_ten_seconds],
                signal[:first_ten_seconds],
                linewidth=0.7,
            )
            axes[1].set_title(
                f"First 10 seconds — reasons: {row['quality_reasons']}"
            )
            axes[1].set_xlabel("Time (seconds)")
            axes[1].set_ylabel("Lead II")

            figure.savefig(
                PLOT_ROOT / f"{status}_{segment_id}.png",
                dpi=140,
            )
            plt.close(figure)


def make_summary(
    quality: pd.DataFrame,
    thresholds: dict[str, float],
    manifest_count: int,
) -> pd.DataFrame:
    """Create a one-row summary containing counts, reasons, and thresholds."""
    reason_names = (
        "invalid_or_nonfinite_signal",
        "nearly_zero_variation",
        "sustained_flatline",
        "repeated_clipping",
        "abrupt_jumps",
        "high_baseline_drift",
        "high_frequency_noise",
    )

    reason_counts = {
        reason: int(
            quality["quality_reasons"]
            .fillna("")
            .str.contains(reason, regex=False)
            .sum()
        )
        for reason in reason_names
    }

    summary = {
        "manifest_window_count": manifest_count,
        "audited_window_count": len(quality),
        "usable_windows": int(
            (quality["quality_status"] == "usable").sum()
        ),
        "review_windows": int(
            (quality["quality_status"] == "review").sum()
        ),
        "unusable_windows": int(
            (quality["quality_status"] == "unusable").sum()
        ),
        "quality_reason_counts": "|".join(
            f"{reason}:{count}"
            for reason, count in reason_counts.items()
        ),
        "thresholds_used": "|".join(
            f"{name}:{value}"
            for name, value in thresholds.items()
        ),
        **{
            f"reason_{reason}_count": count
            for reason, count in reason_counts.items()
        },
        **{
            f"threshold_{name}": value
            for name, value in thresholds.items()
        },
    }
    return pd.DataFrame([summary])


def verify_quality_output(
    manifest: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    """Verify one-to-one coverage between manifest windows and QC rows."""
    if quality["segment_id"].duplicated().any():
        raise RuntimeError("QC output contains duplicate segment IDs")

    manifest_ids = set(manifest["segment_id"].astype(str))
    quality_ids = set(quality["segment_id"].astype(str))

    missing_ids = sorted(manifest_ids - quality_ids)
    unexpected_ids = sorted(quality_ids - manifest_ids)

    if missing_ids:
        raise RuntimeError(
            f"QC output is missing {len(missing_ids)} manifest segments"
        )
    if unexpected_ids:
        raise RuntimeError(
            f"QC output contains {len(unexpected_ids)} unexpected segments"
        )

    if len(quality) != len(manifest):
        raise RuntimeError(
            "QC row count does not equal manifest row count: "
            f"{len(quality)} != {len(manifest)}"
        )

    valid_statuses = {"usable", "review", "unusable"}
    observed_statuses = set(quality["quality_status"].dropna().unique())
    if not observed_statuses.issubset(valid_statuses):
        raise RuntimeError(
            f"Unexpected quality statuses: {sorted(observed_statuses)}"
        )

    status_total = int(
        quality["quality_status"].isin(valid_statuses).sum()
    )
    if status_total != len(quality):
        raise RuntimeError(
            "Usable + review + unusable does not equal total QC rows"
        )


def print_summary(
    quality: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    """Print a concise full-dataset quality-audit summary."""
    print(f"Audited windows: {len(quality)}")
    print(
        f"Usable: "
        f"{int((quality['quality_status'] == 'usable').sum())}"
    )
    print(
        f"Review: "
        f"{int((quality['quality_status'] == 'review').sum())}"
    )
    print(
        f"Unusable: "
        f"{int((quality['quality_status'] == 'unusable').sum())}"
    )
    print("Thresholds:", thresholds)
    print(f"Quality table: {QUALITY_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Review plots: {PLOT_ROOT}")


def main() -> None:
    """Run the configured full-dataset signal-quality audit."""
    manifest = load_manifest()
    selected_manifest = select_windows(manifest)

    # First calculate measures without assigning quality labels. This ensures
    # that review thresholds are derived without using HI or PreHI labels.
    measure_rows: list[dict[str, object]] = []
    signals: dict[str, np.ndarray] = {}

    for _, manifest_row in selected_manifest.iterrows():
        measure_row, signal = audit_window(manifest_row)
        measure_rows.append(measure_row)

        if signal is not None:
            signals[str(manifest_row["segment_id"])] = signal

    measures = pd.DataFrame(measure_rows)
    thresholds = derive_thresholds(measures)

    # Classify every window using the thresholds derived from the complete
    # selected dataset.
    quality_rows: list[dict[str, object]] = []

    for _, manifest_row in selected_manifest.iterrows():
        quality_row, signal = audit_window(
            manifest_row,
            thresholds,
        )
        quality_rows.append(quality_row)

        if signal is not None:
            signals[str(manifest_row["segment_id"])] = signal

    quality = pd.DataFrame(
        quality_rows,
        columns=QUALITY_COLUMNS,
    )

    verify_quality_output(selected_manifest, quality)
    summary = make_summary(
        quality,
        thresholds,
        len(manifest),
    )

    QUALITY_ROOT.mkdir(parents=True, exist_ok=True)
    quality.to_csv(QUALITY_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    make_plots(
        quality,
        signals,
        thresholds,
    )
    print_summary(quality, thresholds)


if __name__ == "__main__":
    main()