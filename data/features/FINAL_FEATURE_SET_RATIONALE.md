# Final Frozen ECG Feature Set

## Purpose

This document freezes the predictor set to be used in the next stage of the ECG haemodynamic-instability project.

The source modelling table contains 48 extracted signal features from 1,283 retained 120-second Lead II ECG windows. The audit showed no missing or infinite feature values, but it identified one constant feature, one near-constant feature, one exact duplicate pair, and several mathematically redundant feature groups.

The aim of this reduction is to remove only features that are clearly constant, deterministic transformations, exact duplicates, or practically redundant because of the fixed 120-second window length and the signal-processing design.

No class labels, patient outcomes, model performance, or held-out-patient results were used to choose the retained features.

## Frozen predictor count

**34 features**

## Final ordered feature list

1. `minimum`
2. `maximum`
3. `mean`
4. `median`
5. `kurtosis`
6. `skewness`
7. `mean_absolute_value`
8. `root_mean_square`
9. `peak_to_peak`
10. `waveform_length`
11. `zero_crossing_count`
12. `slope_sign_change_count`
13. `approximate_entropy`
14. `permutation_entropy`
15. `fuzzy_entropy`
16. `distribution_entropy`
17. `svd_entropy`
18. `lempel_ziv_complexity`
19. `hjorth_mobility`
20. `hjorth_complexity`
21. `katz_fractal_dimension`
22. `higuchi_fractal_dimension`
23. `detrended_fluctuation_analysis`
24. `mean_frequency`
25. `median_frequency`
26. `spectral_edge_frequency_95`
27. `spectral_entropy`
28. `AvgSampEn`
20. `MaxSampEn`
30. `MedianSampEn`
31. `StdSampEn`
32. `KurtosisSampEn`
33. `SkewnessSampEn`
34. `AUC_SampEn`

## Features removed before modelling

| Removed feature | Retained alternative | Reason |
|---|---|---|
| `sum_value` | `mean` | Every window contains exactly 15,000 samples, so sum_value is a fixed multiple of mean. |
| `standard_deviation` | `root_mean_square` | After 0.5–40 Hz filtering, window means are close to zero and standard deviation is effectively redundant with RMS. |
| `variance` | `root_mean_square` | Variance is the square of standard deviation and adds no independent information. |
| `integrated_absolute_value` | `mean_absolute_value` | Every window contains exactly 15,000 samples, so integrated absolute value is a fixed multiple of mean absolute value. |
| `petrosian_fractal_dimension` | `slope_sign_change_count` | The audit showed perfect rank redundancy; the Petrosian calculation is derived largely from sign changes. |
| `fisher_information` | `hjorth_mobility` | The audit showed near-deterministic inverse redundancy in this implementation. |
| `VarSampEn` | `StdSampEn` | Variance is the square of standard deviation. |
| `ProfileRange` | `MaxSampEn` | ProfileRange was an exact duplicate of MaxSampEn in this dataset because MinSampEn was constant. |
| `MinSampEn` | `None` | The feature was constant across all retained windows. |
| `r_at_MaxSampEn` | `None` | The feature was near-constant and therefore unlikely to contribute stable discrimination. |
| `n_r_points` | `peak_to_peak` | The audit showed near-perfect dependence on signal amplitude/range. |
| `r_at_MinSampEn` | `peak_to_peak` | The audit showed near-perfect correlation with peak-to-peak amplitude. |
| `TotalSampEn` | `AUC_SampEn` | The two were almost perfectly correlated; AUC retains tolerance-axis spacing and is the more informative profile summary. |

peak_frequency was removed because its calibration IQR was zero in 19 of 20 patients when estimated from the continuous 10-window Class 0 calibration block. The feature represents a discrete maximum spectral bin and showed insufficient within-calibration variation for stable median–IQR normalization. Broader spectral descriptors, including mean frequency, median frequency, spectral edge frequency, and spectral entropy, were retained.

## Engineering rationale

### RMS, standard deviation, and variance

RMS is retained as the primary overall amplitude/energy descriptor.

Standard deviation and variance are removed because the ECG windows have already been band-pass filtered and therefore have means close to zero. Under this condition, RMS and standard deviation carry almost the same information, while variance is simply the squared standard deviation.

Retaining RMS gives a clinically and engineering-relevant measure of signal magnitude without allowing one amplitude concept to be represented three times.

### Fixed-length deterministic features

Every retained window contains exactly 15,000 samples. Therefore:

- `sum_value` is a fixed multiple of `mean`.
- `integrated_absolute_value` is a fixed multiple of `mean_absolute_value`.

Keeping both members of either pair would give duplicate representation to the same underlying quantity.

### Entropy-profile features

The SampEn-profile audit showed several structural redundancies:

- `MinSampEn` was constant.
- `r_at_MaxSampEn` was nearly constant.
- `ProfileRange` duplicated `MaxSampEn`.
- `VarSampEn` was redundant with `StdSampEn`.
- `TotalSampEn` and `AUC_SampEn` were almost perfectly correlated.
- `n_r_points` and `r_at_MinSampEn` were strongly tied to signal amplitude.

`AUC_SampEn` is retained instead of `TotalSampEn` because it incorporates the spacing of the tolerance axis and is therefore the more informative profile-level summary.

### Features intentionally retained despite strong correlation

Some remaining features are correlated but were kept because they represent different biomedical or signal-processing concepts:

- `mean_absolute_value` and `root_mean_square`
- `mean_frequency` and `median_frequency`
- `permutation_entropy` and `slope_sign_change_count`
- `svd_entropy` and `spectral_edge_frequency_95`
- `hjorth_mobility` and frequency-domain summaries

These are not exact deterministic duplicates. They may carry complementary information in some patients or folds.

Any additional correlation pruning must therefore be performed **inside each LOSO training fold**, using only the training patients. It must not be performed globally on the complete dataset.

## Predictor exclusions

The following columns must never be used as predictors:

- `patient_id`
- `condition`
- `recording_id`
- `segment_id`
- `source_file`
- `window_index`
- `start_timestamp`
- `end_timestamp`
- `Class`
- `sampen_profile_status`
- `quality_status`
- `quality_reasons`
- every column beginning with `qc_`

These fields are retained only for traceability, validation, QC sensitivity analysis, grouping, and outcome definition.

## Use in the next stage

The next-stage code should:

1. Read `data/modelling/clean_modelling_table.csv`.
2. Read the frozen feature list from `data/modelling/final_feature_list.txt`.
3. Verify that all 34 features exist exactly once.
4. Use only these 34 columns as candidate predictors.
5. Apply patient normalisation inside each LOSO fold.
6. Apply any further variance filtering, correlation pruning, feature ranking, scaling, and imputation using training data only.
7. Preserve the same ordered feature list for all experiments.

## Reproducibility decision

This 34-feature list is the manually reduced, scientifically justified starting set.

It should not be changed after model results are inspected unless the change is explicitly documented as a new analysis version.
