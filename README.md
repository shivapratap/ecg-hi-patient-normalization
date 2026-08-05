# Patient-Level ECG Normalisation for Haemodynamic Instability

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Manuscript](https://img.shields.io/badge/Manuscript-under%20submission-lightgrey.svg)](#citation)

This repository contains the analysis pipeline for a study of **patient-level ECG normalisation for leakage-safe discrimination of haemodynamic instability (HI)**. The study compares retrospective patient-specific centring with limited fixed calibration under leave-one-subject-out (LOSO) validation.

The repository is intended for researchers who want to understand and reproduce the study. The recommended reproducibility entry point is the set of **processed feature-extraction outputs**, rather than restricted raw clinical waveform data.

> **Research use only.** This repository is intended for retrospective research and methodological evaluation. It is not a medical device and must not be used for clinical decision-making.

## Study overview

The analysis used non-overlapping 120-second Lead II ECG windows sampled at 125 Hz. The central methodological question was whether patient-specific normalisation could reduce between-patient ECG baseline variation without introducing leakage into patient-held-out validation.

Four patient-level normalisation strategies were evaluated:

1. **No normalisation**
2. **Retrospective jackknife median centring**
3. **Fixed calibration median centring**
4. **Fixed calibration median–IQR normalisation**

The jackknife strategy uses retrospective PreHI information from the held-out patient and therefore serves as a favourable reference rather than a deployable method. The fixed calibration strategies use a continuous block of ten labelled PreHI windows per patient, corresponding to 20 minutes of ECG.

All imputation, variance filtering, feature ranking, correlation pruning, scaling, and model fitting are confined to the appropriate outer LOSO training folds.

## Cohort and analytical accounting

| Item | Value |
|---|---:|
| Patients | 20 |
| Candidate ECG windows | 1,291 |
| Unusable windows excluded | 8 |
| Retained 120-second windows | 1,283 |
| PreHI windows | 585 |
| HI windows | 698 |
| Initially extracted predictors | 48 |
| Frozen predictors used for modelling | 34 |
| Calibration windows per patient | 10 |
| Common evaluation windows | 1,083 |
| Outer LOSO folds | 20 |
| Normalisation strategies | 4 |
| Classifiers | 3 |
| Evaluation modes | 2 |
| Completed model runs | 480 |

Repeated ECG windows are nested within patients and should not be interpreted as independent clinical observations.

## Main findings

The primary analysis used common evaluation rows and Extra Trees. The retrospective jackknife strategy provided the highest pooled discrimination, while fixed calibration median–IQR retained a substantial part of that improvement using a restricted patient-specific reference block.

Key findings were:

- Patient-level normalisation improved discrimination relative to no normalisation.
- Retrospective jackknife median centring provided the strongest reference performance.
- Fixed calibration median–IQR was the strongest deployable calibration strategy.
- Feature selection was more reproducible under the normalised configurations.
- Patient-level performance remained heterogeneous.
- Seven patients achieved jackknife ROC AUC 1.00.
- Two patients showed below-chance discrimination.
- Signal-quality burden alone did not explain patient-level performance.

The exact metrics used in the manuscript are provided in:

```text
data/results/loso_pooled_metrics.csv
data/results/loso_patient_metrics.csv
data/results/feature_stability_configuration_summary.csv
data/results/patient_heterogeneity_summary.csv
data/results/clinical_sensitivity_summary.csv
```

## Repository structure

```text
.
├── data/
│   ├── features/          # Processed feature-extraction outputs
│   ├── modelling/         # Clean modelling table and frozen feature list
│   └── results/           # LOSO outputs, stability tables, figures and summaries
├── metadata/              # Structured patient-level clinical summaries and flags
├── outputs/               # General audit outputs from early pipeline stages
├── scripts/               # Numbered analysis scripts
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Raw ECG files, processed waveform windows, normalisation intermediates, and signal-quality review plots are excluded from version control where required by `.gitignore`.

## Reproducibility entry point

The recommended public workflow begins from the processed feature-extraction outputs.

The main downstream inputs are:

```text
data/features/ecg_features.csv
data/quality/window_signal_quality.csv
data/modelling/final_feature_list.txt
metadata/patient_clinical_summary.csv
metadata/patient_clinical_flags.csv
```

To rebuild the analysis from feature and QC outputs, begin with:

```bash
python scripts/05_create_clean_modelling_table.py
```

Researchers with access to processed 120-second ECG windows may also run `04_check_signal_quality.py`. Researchers with authorised access to the source waveform data may run the complete pipeline from script 01.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/shivapratap/ecg-hi-patient-normalization.git
cd ecg-hi-patient-normalization
```

### 2. Create an isolated Python environment

Python 3.10 or later is recommended.

Using `venv` on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Using Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade `pip` and install the core dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The downstream analysis uses:

- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- seaborn

### 3. Install the feature-extraction packages

Feature extraction in `scripts/03_extract_features.py` uses two project-specific packages.

#### `amrita-biosignal-feature-engine`

Repository: <https://github.com/shivapratap/amrita-biosignal-feature-engine>

Install directly from GitHub:

```bash
python -m pip install "git+https://github.com/shivapratap/amrita-biosignal-feature-engine.git"
```

This package supplies the general time-domain, frequency-domain, nonlinear, entropy, complexity, and fractal feature-extraction interface used in the ECG pipeline.

#### `sampen-profile`

Repository: <https://github.com/shivapratap/sampen-profile>

Install directly from GitHub:

```bash
python -m pip install "git+https://github.com/shivapratap/sampen-profile.git"
```

This package computes the sample-entropy profile and its scalar summaries, including `AvgSampEn`, `MaxSampEn`, `MedianSampEn`, `StdSampEn`, `KurtosisSampEn`, `SkewnessSampEn`, and `AUC_SampEn`.

Verify both installations:

```bash
python -c "import amrita_biosignal_feature_engine; import sampen_profile; print('Feature packages installed successfully')"
```

These two packages are required only when rerunning `03_extract_features.py`. They are not required when starting from the processed feature table.

## Input data organisation

### Full raw-data workflow

The raw waveform data are not distributed in this repository. For the complete pipeline, the scripts expect:

```text
data/
└── raw/
    ├── 20Patients_HI/
    └── 20Patients_PreHI/
```

Input files are expected to contain timestamps and Lead II ECG samples. The raw files are never modified by the scripts.

### Processed-window workflow

Processed waveform windows are expected under:

```text
data/processed/windows/
```

Each complete window contains 15,000 samples, corresponding to 120 seconds at 125 Hz.

### Feature-table workflow

The primary public starting point is:

```text
data/features/ecg_features.csv
```

This table contains extracted features and traceability fields for each ECG window. Signal-quality information is joined by `segment_id` in script 05.

## Script execution order

The numbered scripts encode the intended execution order.

| Script | Purpose |
|---|---|
| `01_audit_raw_files.py` | Audits raw Lead II ECG files without modifying them. |
| `02_preprocess_and_segment.py` | Band-pass filters recordings and creates non-overlapping 120-second windows. |
| `03_extract_features.py` | Extracts scalar ECG features and SampEn-profile summaries. |
| `04_check_signal_quality.py` | Computes signal-quality measures and assigns usable, review, or unusable status. |
| `05_create_clean_modelling_table.py` | Joins features and QC results and excludes only unusable windows. |
| `06_audit_feature_table.py` | Audits feature quality and redundancy before modelling. |
| `07_patient_normalization.py` | Runs patient-normalisation smoke tests. |
| `08_full_normalization_audit.py` | Audits all patient-level normalisation strategies across the cohort. |
| `09_loso_modelling.py` | Runs the leakage-safe LOSO modelling comparison. |
| `10_feature_stability_analysis.py` | Quantifies feature-selection stability across LOSO folds. |
| `11_patient_heterogeneity_analysis.py` | Characterises patient-level discrimination heterogeneity. |
| `12_clinical_sensitivity_analysis.py` | Performs exploratory clinical and signal-quality sensitivity analyses. |

To reproduce the downstream analysis from processed feature outputs:

```bash
python scripts/05_create_clean_modelling_table.py
python scripts/06_audit_feature_table.py
python scripts/07_patient_normalization.py
python scripts/08_full_normalization_audit.py
python scripts/09_loso_modelling.py
python scripts/10_feature_stability_analysis.py
python scripts/11_patient_heterogeneity_analysis.py
python scripts/12_clinical_sensitivity_analysis.py
```

To run the complete workflow from authorised waveform files:

```bash
python scripts/01_audit_raw_files.py
python scripts/02_preprocess_and_segment.py
python scripts/03_extract_features.py
python scripts/04_check_signal_quality.py
python scripts/05_create_clean_modelling_table.py
python scripts/06_audit_feature_table.py
python scripts/07_patient_normalization.py
python scripts/08_full_normalization_audit.py
python scripts/09_loso_modelling.py
python scripts/10_feature_stability_analysis.py
python scripts/11_patient_heterogeneity_analysis.py
python scripts/12_clinical_sensitivity_analysis.py
```

## Frozen feature set

The source modelling table contained 48 extracted signal features. A label-independent engineering audit removed constant, near-constant, exact-duplicate, deterministic, or practically redundant predictors.

The final modelling stage used the following 34 features:

```text
minimum
maximum
mean
median
kurtosis
skewness
mean_absolute_value
root_mean_square
peak_to_peak
waveform_length
zero_crossing_count
slope_sign_change_count
approximate_entropy
permutation_entropy
fuzzy_entropy
distribution_entropy
svd_entropy
lempel_ziv_complexity
hjorth_mobility
hjorth_complexity
katz_fractal_dimension
higuchi_fractal_dimension
detrended_fluctuation_analysis
mean_frequency
median_frequency
spectral_edge_frequency_95
spectral_entropy
AvgSampEn
MaxSampEn
MedianSampEn
StdSampEn
KurtosisSampEn
SkewnessSampEn
AUC_SampEn
```

The ordered list is stored in:

```text
data/modelling/final_feature_list.txt
```

Any further variance filtering, ranking, and correlation pruning are performed inside each outer LOSO training fold.

## LOSO modelling design

The modelling stage evaluates:

- 20 outer patient-held-out folds
- four normalisation strategies
- three classifiers: logistic regression, Extra Trees, and random forest
- two evaluation modes: common rows and native rows

The common-row analysis excludes the same ten calibration windows from every normalisation strategy, ensuring that strategies are compared on identical held-out observations.

Within each outer fold, the following operations are fitted using permitted training data only:

1. Patient-level normalisation statistics
2. Missing-value imputation
3. Variance filtering
4. Random-forest feature ranking
5. Top-k feature selection
6. Spearman correlation pruning
7. Standardisation for logistic regression
8. Model fitting

The held-out patient is excluded from all population-level fallback estimation and model-training operations.

## Principal outputs

Important result files include:

```text
data/results/loso_predictions.csv
data/results/loso_pooled_metrics.csv
data/results/loso_patient_metrics.csv
data/results/loso_fold_audit.csv
data/results/loso_selected_features.csv
data/results/feature_selection_stability.csv
data/results/feature_stability_configuration_summary.csv
data/results/patient_heterogeneity_master.csv
data/results/patient_heterogeneity_summary.csv
data/results/clinical_sensitivity_summary.csv
data/results/clinical_subgroup_performance_sensitivity.csv
```

Publication figures are stored in:

```text
data/results/figures/
```

Many figures are supplied in both PNG and PDF formats. Corresponding figure-data CSV files are retained to support checking and regeneration.

## Reproducibility notes

- The random seed used in the final modelling pipeline is `42`.
- The fixed calibration block contains ten consecutive PreHI windows per patient.
- Calibration windows are excluded from calibration-based evaluation sets.
- Two patients required a later valid continuous calibration block.
- Zero or near-zero calibration IQRs are resolved using fold-specific fallback scales derived only from outer-training patients.
- No global normalised feature table is created for model fitting.
- Difficult patients are retained rather than excluded post hoc.
- Clinical and QC analyses are exploratory and do not alter the primary model.

## Data availability

The source ECG data were obtained from a controlled-access clinical waveform resource and are not redistributed in this repository. Users seeking to reproduce the complete raw-data workflow must independently obtain the appropriate authorisation and comply with the source database's data-use requirements.

This repository provides code, processed feature-level materials where permitted, model outputs, audit tables, and figure-generation resources.

## Manuscript status

A manuscript describing this study is currently **under submission**. Results and filenames in this repository correspond to the frozen analysis used for manuscript preparation. Citation information will be updated after publication.

## Citation

Until the manuscript is published, please cite the repository as:

> Shivasankar, A., Gopakumar, S., and Udhayakumar, R. *Patient-Level ECG Normalisation for Leakage-Safe Discrimination of Haemodynamic Instability*. GitHub repository, 2026. <https://github.com/shivapratap/ecg-hi-patient-normalization>

## Authors

- **Asha Shivasankar**
- **Shivapratap Gopakumar**
- **Radhagayathri Udhayakumar**

**Affiliation:** Amrita Vishwa Vidyapeetham, Amritapuri, India

### Code enquiries

**Shivapratap Gopakumar**  
Email: [shivapg@am.amrita.edu](mailto:shivapg@am.amrita.edu)

Bug reports and reproducibility questions may also be submitted through GitHub Issues.

## Licence

The source code in this repository is released under the [MIT License](LICENSE).

The MIT licence applies to the repository's original source code. It does not override restrictions attached to source clinical data, third-party software, published articles, institutional materials, or externally licensed content.

## Disclaimer

This repository is provided for retrospective research and methodological evaluation. The software and results have not been validated as a medical device, diagnostic system, bedside alarm, or clinical decision-support tool. They must not be used to guide patient care.
