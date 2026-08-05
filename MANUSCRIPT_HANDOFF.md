# ECG Haemodynamic Instability Study: Frozen Manuscript Handoff

**Freeze date:** 2 August 2026  
**Target journal:** IEEE Access  
**Analysis status:** Primary modelling, feature-stability analysis, patient-level heterogeneity analysis, and exploratory clinical sensitivity analysis are complete and frozen.

> **Use rule:** Final CSV outputs are the source of truth for all numerical results. Final scripts are the source of truth for processing order and implementation details. This document is a compact manuscript handoff and should not override a final CSV or script.

## 1. Study purpose and manuscript positioning

This study evaluates whether patient-level ECG normalization improves discrimination between pre-haemodynamic-instability (PreHI) and haemodynamic-instability (HI) windows in a heterogeneous ICU cohort. The work should be positioned as a patient-adaptation and validation study rather than as a generic classifier benchmark.

The central scientific themes are:

1. strong inter-patient baseline heterogeneity in ECG-derived features;
2. leakage-safe patient-held-out evaluation;
3. comparison of a retrospective jackknife reference with deployable fixed calibration strategies;
4. reproducibility of feature selection across held-out patients;
5. substantial patient-level heterogeneity in discrimination; and
6. exploratory clinical interpretation of successful and unsuccessful patients.

## 2. Frozen cohort and ECG dataset

- Final modelling rows: **1,283**
- Patients: **20**
- PreHI / Class 0: **585 windows**
- HI / Class 1: **698 windows**
- Initially segmented windows: **1,291**
- Unusable windows excluded before modelling: **8**
- ECG lead: **Lead II**
- Sampling frequency: **125 Hz**
- Window duration: **120 s**
- Windowing: **non-overlapping; no padding**
- Filtering: **0.5–40 Hz, fourth-order Butterworth bandpass, SOS implementation, zero-phase filtering**

All 20 patients contributed both classes to the final modelling table.

## 3. Feature extraction and frozen feature set

- Initial extracted features: **48**
- Frozen modelling features: **34**
- Feature domains include amplitude/statistical, temporal morphology, spectral, entropy/complexity, fractal, and sample-entropy-profile descriptors.
- Features were frozen before final LOSO modelling.

Important feature-engineering decisions:

- `TotalSampEn` was removed because it was redundant with `AUC_SampEn`; `AUC_SampEn` was retained as the profile-area summary.
- `MedianSampEn` was retained and later showed highly stable selection.
- `peak_frequency` was removed because its calibration IQR was zero or unstable in 19 of 20 patients.
- Deterministic, constant, redundant, and calibration-unstable features were removed before modelling.

The heterogeneity analysis used the following stability-derived feature subset:

- `kurtosis`
- `skewness`
- `mean_absolute_value`
- `waveform_length`
- `slope_sign_change_count`
- `approximate_entropy`
- `fuzzy_entropy`
- `median_frequency`
- `spectral_entropy`
- `MedianSampEn`
- `AUC_SampEn`

## 4. Patient-level normalization strategies

Four normalization strategies were compared:

1. **None:** original feature values.
2. **Jackknife median:** HI rows were centered using the median of all Class 0 rows from the same patient; each Class 0 row was centered using the median of all other Class 0 rows from that patient. This is retrospective and should be interpreted as an upper-bound patient-reference strategy, not a deployable method.
3. **Calibration median:** subtract the median estimated from a fixed patient calibration block.
4. **Calibration median–IQR:** subtract the calibration median and divide by the calibration IQR.

Calibration policy:

- Calibration length: **10 Class 0 windows = 20 minutes**.
- Earliest valid continuous Class 0 block.
- Same recording.
- Consecutive window indices.
- Approximately 120-s timestamp spacing, with the implemented tolerance.
- Sliding search used when the earliest candidate block was invalid.
- All 20 patients had a valid block.
- Two patients required a later block.
- No IQR fallback events remained after the final feature freeze.

Evaluation modes:

- **Common rows (primary):** calibration rows excluded from every normalization strategy to ensure a fair comparison.
- **Native rows (secondary):** none and jackknife retained all eligible rows; calibration strategies excluded their calibration rows.

## 5. Leakage-safe LOSO modelling

- Outer folds: **20 patient-level LOSO folds**
- Evaluation modes: **2**
- Normalizations: **4**
- Classifiers: **3**
- Completed model runs: **480/480**
- Out-of-fold prediction rows across all runs: **27,192**
- Random seed: **42**

Within each outer training fold, processing followed this order:

1. patient normalization;
2. row exclusion according to evaluation mode;
3. conversion of infinite values to missing;
4. training-only median imputation;
5. training-only zero-variance filtering;
6. training-only Random Forest feature-importance ranking;
7. top-20 restriction;
8. training-only Spearman pruning at `|rho| >= 0.90`;
9. `StandardScaler` for Logistic Regression only;
10. classifier fitting on the 19 training patients;
11. prediction for the untouched held-out patient.

Classifiers: Logistic Regression, Extra Trees, and Random Forest.

All patient-overlap, training-only preprocessing, finite-matrix, and calibration-exclusion checks passed.

## 6. Primary performance results

The primary analysis is **common rows + Extra Trees**.

| Normalization | Pooled ROC AUC | Average precision | Accuracy | Sensitivity | Specificity | Balanced accuracy | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| None | 0.489 | 0.630 | 0.559 | 0.683 | 0.332 | 0.508 | 1,083 |
| Jackknife median | 0.823 | 0.830 | 0.780 | 0.794 | 0.756 | 0.775 | 1,083 |
| Calibration median | 0.679 | 0.722 | 0.653 | 0.655 | 0.649 | 0.652 | 1,083 |
| Calibration median–IQR | 0.745 | 0.824 | 0.669 | 0.668 | 0.673 | 0.670 | 1,083 |

Primary interpretation:

- Jackknife Extra Trees achieved the highest full-cohort pooled AUC: **0.823**.
- Calibration median–IQR Extra Trees achieved pooled AUC **0.745**.
- The pooled AUC difference was **0.078** in favour of jackknife.
- Jackknife should be described as a retrospective upper-bound reference.
- Calibration median–IQR should be described as the more deployable strategy.

Patient-level summary:

- Median jackknife patient AUC: **0.949**
- Jackknife patient AUC IQR: **0.238**
- Jackknife range: **0.298–1.000**
- Median calibration median–IQR patient AUC: **0.940**
- Calibration median–IQR patient AUC IQR: **0.292**
- Calibration median–IQR range: **0.443–1.000**
- Spearman correlation between patient AUCs for the two methods: **rho = 0.836**

The difference between pooled AUC and the high median patient AUC is a central result: many patients were internally well separated, but cross-patient prediction-score distributions remained heterogeneous.

## 7. Feature-selection stability

- Feature-audit rows: **16,320**
- Feature-stability rows: **816**
- Pairwise fold-set comparisons: **4,560**
- All audit keys, stage relationships, fold counts, and similarity checks passed.

Mean pairwise Jaccard similarity for common-row Extra Trees:

- Jackknife median: **0.841**
- Calibration median: **0.810**
- Calibration median–IQR: **0.768**
- None: **0.752**

Features retained in at least 75% of folds under all three primary Extra Trees normalization configurations:

- `MedianSampEn`
- `approximate_entropy`
- `fuzzy_entropy`
- `kurtosis`
- `mean_absolute_value`
- `median_frequency`
- `skewness`
- `slope_sign_change_count`
- `spectral_entropy`
- `waveform_length`

Sample-entropy profile findings:

- `MedianSampEn` was the most reproducible SampEn-profile descriptor.
- `AUC_SampEn` also contributed, especially under jackknife and calibration median–IQR.
- `TotalSampEn` was not modelled because it had already been removed as redundant.

> **Known reporting caution:** The originally generated `feature_stability_configuration_summary.csv` calculated the fields labelled mean/minimum/maximum final feature count from per-feature fold-selection counts rather than per-fold retained-set sizes. Jaccard similarities and feature-selection frequencies are valid, but those feature-count summary fields must not be quoted unless regenerated with the corrected calculation.

## 8. Patient-level heterogeneity

- High-performing patients by the predefined descriptive rule: **12**
- Intermediate: **4**
- Low: **4**
- Below-chance jackknife patients: **27245, 30851**

### Patient 30851

- Jackknife AUC: approximately **0.298**
- Calibration median–IQR AUC: approximately **0.443**
- QC-review fraction: approximately **96.2%**
- Prediction ordering was reversed for both primary methods.
- Structured clinical context included temporary pacing/device context, LBBB, and antiarrhythmic exposure.
- Stable-feature directions were mixed.

Interpretation: performance was consistent with extreme QC burden combined with atypical conduction/device-related morphology. This is an association, not a causal conclusion.

### Patient 27245

- Jackknife AUC: approximately **0.425**
- Calibration median–IQR AUC: approximately **0.467**
- QC-review fraction: approximately **6.7%**
- Structured clinical context included ICD/device and LBBB flags.
- A later calibration block was required.
- Feature shifts were large for some variables but inconsistent across domains.
- Calibration probability ordering was reversed.

Interpretation: this case is more consistent with a patient-specific multidomain feature-pattern mismatch than with QC burden alone.

## 9. Perfect-AUC patients

Seven patients had jackknife Extra Trees patient AUC = 1.00: **10013, 30071, 80436, 88503, 90396, 906, 96879**.

What they shared:

- Median absolute Cliff's delta: **1.00**, versus **0.58** among non-perfect patients.
- Typically 7–11 stable features had large within-patient effects.
- Predicted-probability distributions showed strong or complete class separation.
- Strong separation occurred across multiple feature domains.

What they did not share:

- No single diagnosis, rhythm state, device status, or other structured clinical factor was common to all perfect-AUC patients.
- Their clinical flags were heterogeneous.

The defensible conclusion is that perfect discrimination was associated primarily with strong multidomain ECG feature and prediction separation, not with one clinical phenotype.

## 10. Exploratory clinical sensitivity analysis

- Clinical-summary matches: **20/20**
- Clinical-flag matches: **20/20**
- Analysed continuous variables included age and parsed ejection-fraction midpoint.
- Analysed structured variables included sex, pacing/ICD, AF, BBB, antiarrhythmic exposure, mechanical support, atypical HI, HRV outlier status, clean sinus-rhythm candidacy, and HRV eligibility.
- All subgroup analyses are exploratory and hypothesis-generating.

Selected subgroup performance:

| Cohort | Patients | Jackknife AUC | Jackknife 95% CI | Calibration-IQR AUC | Calibration-IQR 95% CI |
|---|---:|---:|---:|---:|---:|
| Full Cohort | 20 | 0.823 | 0.715–0.917 | 0.745 | 0.629–0.868 |
| Hrv Valid | 12 | 0.892 | 0.808–0.961 | 0.847 | 0.753–0.952 |
| Without Implanted Device Or Pacing | 15 | 0.897 | 0.818–0.958 | 0.841 | 0.757–0.934 |
| Without Major Arrhythmia | 10 | 0.885 | 0.790–0.962 | 0.835 | 0.732–0.947 |
| Without High Qc Burden | 10 | 0.798 | 0.679–0.908 | 0.686 | 0.564–0.828 |

Clinical sensitivity interpretation:

- HRV-valid, device-free, and no-major-arrhythmia cohorts showed higher pooled AUC than the full cohort.
- Some restricted cohorts changed pooled AUC by more than 0.05.
- The qualitative ordering of jackknife above calibration median–IQR remained.
- Excluding patients with above-median QC burden did not improve performance, showing that QC-review fraction is not a sufficient exclusion criterion.
- No clinical subgroup result should be treated as confirmatory because the cohort contains only 20 patients.

## 11. Main scientific interpretation

> Patient-level ECG normalization materially improves PreHI-versus-HI discrimination by reducing between-patient baseline offsets. Retrospective jackknife normalization provides the strongest and most stable performance but represents an upper-bound reference. Fixed calibration is more deployable, although its pooled performance is lower. Patient-level success depends primarily on whether the PreHI-to-HI transition produces a coherent multidomain ECG feature shift for that individual; rhythm/device context, signal quality, and clinical complexity may disrupt this pattern.

## 12. Recommended research questions

1. Does patient-level normalization improve patient-held-out discrimination of PreHI and HI ECG windows compared with no normalization?
2. How does retrospective jackknife normalization compare with fixed calibration strategies representing more deployable patient adaptation?
3. Are the selected ECG features reproducible across LOSO folds and normalization strategies?
4. How much does discrimination vary between patients, and what signal, feature-separation, and clinical factors are associated with that variation?

## 13. Recommended manuscript contribution statement

This work provides a leakage-safe evaluation of patient-level ECG normalization for haemodynamic-instability discrimination, distinguishes a retrospective upper-bound strategy from deployable calibration approaches, quantifies feature-selection stability across held-out patients, and characterizes clinically relevant patient-level heterogeneity without excluding difficult cases.

## 14. Primary versus secondary analyses

### Primary

- Full 20-patient cohort
- Common-row evaluation
- Fixed 34-feature set
- Extra Trees primary model comparison
- Four normalization strategies
- Pooled out-of-fold metrics
- Patient-level LOSO metrics

### Secondary

- Native-row evaluation
- Logistic Regression and Random Forest comparisons
- Feature-selection stability
- Patient-level heterogeneity
- Case descriptions for 30851 and 27245
- Perfect-AUC patient comparison
- Clinical and QC subgroup sensitivity analyses

## 15. Manuscript safeguards

- Do not call jackknife normalization deployable.
- Do not call stable features validated biomarkers.
- Do not claim that clinical factors caused model success or failure.
- Do not remove difficult patients from the primary analysis.
- Do not perform post hoc feature re-selection using stability results.
- Do not perform hyperparameter optimization after inspecting outer-fold results.
- Do not present patient-level p-values or subgroup tests as confirmatory.
- Do not infer that missing clinical metadata means a condition was absent.
- Do not claim external generalizability.
- Always distinguish pooled AUC from median patient AUC.
- Preserve the full-cohort result as primary even when restricted cohorts perform better.

## 16. Source-of-truth hierarchy for manuscript writing

1. Final result CSV files for numerical values.
2. Final scripts for methods, ordering, and implementation.
3. Frozen feature list and feature-rationale document.
4. Structured clinical summary and clinical flags.
5. This handoff document.
6. Collaborator reports and earlier notebooks only for background; they must not override final outputs.

## 17. Essential files for the new manuscript session

### Methods and frozen design

- `data/modelling/clean_modelling_table.csv`
- `data/modelling/final_feature_list.txt`
- `data/modelling/FINAL_FEATURE_SET_RATIONALE.md`
- `scripts/07_patient_normalization.py`
- `scripts/08_full_normalization_audit.py`
- `scripts/09_loso_modelling.py`
- `scripts/10_feature_stability_analysis.py`
- `scripts/11_patient_heterogeneity_analysis.py`
- `scripts/12_clinical_sensitivity_analysis.py`

### Results

- `data/results/loso_pooled_metrics.csv`
- `data/results/loso_patient_metrics.csv`
- `data/results/loso_run_summary.csv`
- `data/results/feature_selection_stability.csv`
- `data/results/feature_set_similarity.csv`
- `data/results/patient_heterogeneity_master.csv`
- `data/results/patient_heterogeneity_summary.csv`
- `data/results/below_chance_patient_case_summary.csv`
- `data/results/perfect_auc_patient_summary.csv`
- `data/results/clinical_sensitivity_summary.csv`
- `data/results/clinical_subgroup_performance_sensitivity.csv`

### Clinical context

- `metadata/patient_clinical_summary.csv`
- `metadata/patient_clinical_flags.csv`
- `metadata/ECG_Clinical_Summary_Report.docx`

### Manuscript and references

- `access.tex`
- `main.bib`
- `Ref.bib`
- `PatientNormalizationStrategy.pdf`

## 18. Final freeze declaration

The analysis is complete. Manuscript writing should proceed without changing the modelling dataset, feature set, normalization definitions, calibration policy, LOSO structure, fixed model configurations, patient inclusion, or primary outcome definitions. Any future analysis must be versioned separately and explicitly labelled as post-freeze.
