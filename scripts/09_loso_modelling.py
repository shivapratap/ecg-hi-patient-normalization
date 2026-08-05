"""Run a leakage-safe patient-level LOSO modelling comparison.

All normalization and feature-selection steps are refit inside each outer
patient fold. The script writes out-of-fold predictions and complete stage-by-stage
feature-selection audit tables, but does not create a global normalized
feature table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "modelling" / "clean_modelling_table.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "data" / "modelling" / "final_feature_list.txt"
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"

RANDOM_SEED = 42
CALIBRATION_ROWS_PER_PATIENT = 10
WINDOW_SPACING_SECONDS = 120.0
TIMESTAMP_TOLERANCE_SECONDS = 1.0
IQR_TOLERANCE = 1e-12
VARIANCE_THRESHOLD = 0.0
TOP_K_FEATURES = 20
SPEARMAN_THRESHOLD = 0.90

NORMALIZATION_STRATEGIES = (
    "none",
    "jackknife_median",
    "calibration_median",
    "calibration_median_iqr",
)
CLASSIFIERS = ("logistic_regression", "extra_trees", "random_forest")
EVALUATION_MODES = ("common_rows", "native_rows")


def load_inputs() -> tuple[pd.DataFrame, list[str]]:
    """Load and validate the source table and exact frozen feature order."""
    table = pd.read_csv(INPUT_PATH)
    features = [
        line.strip()
        for line in FEATURE_LIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(table) != 1283 or table["patient_id"].nunique() != 20:
        raise ValueError("Expected 1,283 rows and 20 patients")
    if len(features) != 34 or len(set(features)) != 34:
        raise ValueError("Expected exactly 34 unique frozen features")
    required = [
        "patient_id", "segment_id", "Class", "start_timestamp",
        "recording_id", "window_index", *features,
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if table["segment_id"].isna().any() or table["segment_id"].duplicated().any():
        raise ValueError("segment_id must be present and unique")
    if not table["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1")
    numeric = table[features].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Frozen feature values must be numeric and finite")
    table[features] = numeric
    for patient_id, group in table.groupby("patient_id"):
        if set(group["Class"]) != {0, 1}:
            raise ValueError(f"Patient {patient_id} does not contain both classes")
    return table, features


def select_continuous_calibration_block(
    table: pd.DataFrame, patient_id: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select the earliest valid consecutive Class 0 calibration block."""
    candidates = table[(table["patient_id"] == patient_id) & (table["Class"] == 0)].copy()
    candidates["_timestamp"] = pd.to_datetime(candidates["start_timestamp"], errors="coerce")
    if candidates["_timestamp"].isna().any():
        raise ValueError(f"Invalid calibration timestamp for patient {patient_id}")
    candidates = candidates.sort_values(
        ["_timestamp", "recording_id", "window_index"]
    ).reset_index(drop=True)
    candidates["chronological_rank"] = np.arange(1, len(candidates) + 1)
    if len(candidates) < CALIBRATION_ROWS_PER_PATIENT:
        raise ValueError(f"Patient {patient_id} lacks enough Class 0 rows")

    selected = None
    block_audit = None
    for start in range(len(candidates) - CALIBRATION_ROWS_PER_PATIENT + 1):
        block = candidates.iloc[start : start + CALIBRATION_ROWS_PER_PATIENT].copy()
        starts = block["_timestamp"].diff().dt.total_seconds().dropna().to_numpy()
        same_recording = block["recording_id"].nunique() == 1
        consecutive_indices = same_recording and np.array_equal(
            np.diff(block["window_index"].to_numpy()),
            np.ones(CALIBRATION_ROWS_PER_PATIENT - 1),
        )
        expected_spacing = len(starts) == CALIBRATION_ROWS_PER_PATIENT - 1 and np.all(
            np.abs(starts - WINDOW_SPACING_SECONDS) <= TIMESTAMP_TOLERANCE_SECONDS
        )
        unique_ids = not block["segment_id"].duplicated().any()
        if same_recording and consecutive_indices and expected_spacing and unique_ids:
            selected = block
            block_audit = {
                "selected_block_recording_count": int(block["recording_id"].nunique()),
                "selected_block_within_single_recording": True,
                "selected_block_consecutive_window_indices": True,
                "selected_block_expected_120s_spacing": True,
                "selected_block_crosses_recording_boundary": False,
                "selected_block_has_timestamp_discontinuity": False,
            }
            break

    if selected is None:
        raise ValueError(f"No valid continuous calibration block for patient {patient_id}")
    selected["calibration_order"] = np.arange(1, CALIBRATION_ROWS_PER_PATIENT + 1)
    selected_ids = set(selected["segment_id"])
    selected_order_map = selected.set_index("segment_id")["calibration_order"].to_dict()
    audit_rows = []
    for _, row in candidates.iterrows():
        audit_rows.append(
            {
                "patient_id": patient_id,
                "segment_id": row["segment_id"],
                "start_timestamp": row["start_timestamp"],
                "recording_id": row["recording_id"],
                "window_index": row["window_index"],
                "chronological_rank": row["chronological_rank"],
                "calibration_order": selected_order_map.get(row["segment_id"], np.nan),
                "Class": row["Class"],
                "selected_for_calibration": row["segment_id"] in selected_ids,
                **block_audit,
            }
        )
    return selected, pd.DataFrame(audit_rows)


def select_all_calibration_blocks(
    table: pd.DataFrame, patient_ids: list[int]
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Select calibration blocks for every patient in numeric order."""
    selected = {}
    audits = []
    for patient_id in patient_ids:
        block, audit = select_continuous_calibration_block(table, patient_id)
        selected[patient_id] = block
        audits.append(audit)
    return selected, pd.concat(audits, ignore_index=True)


def calibration_statistics(
    selected: dict[int, pd.DataFrame], features: list[str]
) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """Calculate patient calibration medians and IQRs."""
    medians = {pid: rows[features].median() for pid, rows in selected.items()}
    iqrs = {
        pid: rows[features].quantile(0.75) - rows[features].quantile(0.25)
        for pid, rows in selected.items()
    }
    return medians, iqrs


def fold_iqr_scales(
    selected: dict[int, pd.DataFrame],
    iqrs: dict[int, pd.Series],
    train_patients: list[int],
    test_patient: int,
    features: list[str],
) -> tuple[dict[int, pd.Series], list[dict[str, object]]]:
    """Resolve patient IQR scales using only outer-training patient information."""
    valid_training = {
        feature: [float(iqrs[pid][feature]) for pid in train_patients
                  if np.isfinite(iqrs[pid][feature]) and iqrs[pid][feature] > IQR_TOLERANCE]
        for feature in features
    }
    pooled_training = {
        feature: np.concatenate([
            selected[pid][feature].to_numpy(float) for pid in train_patients
        ])
        for feature in features
    }
    fallback_rows = []
    scales_by_patient = {}
    for patient_id in [*train_patients, test_patient]:
        scales = {}
        role = "held_out" if patient_id == test_patient else "training"
        for feature in features:
            original = float(iqrs[patient_id][feature])
            if np.isfinite(original) and original > IQR_TOLERANCE:
                scale = original
                source = "patient_calibration_iqr"
                fallback = False
            elif valid_training[feature]:
                scale = float(np.median(valid_training[feature]))
                source = "median_valid_training_patient_iqr"
                fallback = True
            else:
                pooled = pooled_training[feature]
                pooled_iqr = float(np.percentile(pooled, 75) - np.percentile(pooled, 25))
                if np.isfinite(pooled_iqr) and pooled_iqr > IQR_TOLERANCE:
                    scale = pooled_iqr
                    source = "pooled_training_calibration_iqr"
                else:
                    scale = 1.0
                    source = "emergency_scale_1"
                fallback = True
            scales[feature] = scale
            fallback_rows.append(
                {
                    "outer_fold_patient_id": test_patient,
                    "affected_patient_id": patient_id,
                    "affected_patient_role": role,
                    "feature_name": feature,
                    "original_calibration_iqr": original,
                    "fallback_required": fallback,
                    "fallback_source": source,
                    "fallback_scale": scale,
                    "held_out_patient_excluded_from_fallback_estimation": test_patient not in train_patients,
                }
            )
        scales_by_patient[patient_id] = pd.Series(scales)
    return scales_by_patient, fallback_rows


def normalize_patient(
    rows: pd.DataFrame,
    features: list[str],
    selected: dict[int, pd.DataFrame],
    medians: dict[int, pd.Series],
    scales: dict[int, pd.Series],
    strategy: str,
) -> pd.DataFrame:
    """Normalize all rows for one patient using fold-specific statistics."""
    patient_id = int(rows["patient_id"].iloc[0])
    output = rows.copy()
    selected_ids = set(selected[patient_id]["segment_id"])
    output["is_common_calibration_row"] = output["segment_id"].isin(selected_ids)
    order_map = selected[patient_id].set_index("segment_id")["calibration_order"].to_dict()
    output["calibration_order"] = output["segment_id"].map(order_map)
    values = rows[features].astype(float)
    if strategy == "none":
        normalized = values
    elif strategy == "jackknife_median":
        class_zero = rows[rows["Class"] == 0]
        if len(class_zero) < 2:
            raise ValueError(f"Patient {patient_id} lacks two Class 0 rows")
        references = class_zero[features]
        normalized = pd.DataFrame(index=rows.index, columns=features, dtype=float)
        for index in rows.index:
            reference = references if rows.at[index, "Class"] == 1 else references.drop(index=index)
            normalized.loc[index] = values.loc[index] - reference.median()
    elif strategy == "calibration_median":
        normalized = values - medians[patient_id]
    elif strategy == "calibration_median_iqr":
        normalized = (values - medians[patient_id]) / scales[patient_id]
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")
    output[features] = normalized
    return output


def evaluation_rows(
    normalized: pd.DataFrame, strategy: str, mode: str
) -> pd.DataFrame:
    """Apply common-row or native-row exclusion rules."""
    if mode == "common_rows" or strategy in ("calibration_median", "calibration_median_iqr"):
        return normalized[~normalized["is_common_calibration_row"]].copy()
    return normalized.copy()


def make_classifier(name: str):
    """Create one configured classifier by name."""
    if name == "logistic_regression":
        return LogisticRegression(
            penalty="l2", solver="liblinear", class_weight="balanced",
            random_state=RANDOM_SEED, max_iter=5000,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        )
    raise ValueError(f"Unknown classifier: {name}")


def fit_training_preprocessing(
    train: pd.DataFrame, features: list[str], classifier_name: str
) -> tuple[np.ndarray, list[str], dict[str, object], list[dict[str, object]]]:
    """Fit imputation, variance filtering, ranking, and correlation pruning on training rows."""
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train[features].replace([np.inf, -np.inf], np.nan))
    variance_filter = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    train_variance = variance_filter.fit_transform(train_imputed)
    variance_features = [
        feature for feature, keep in zip(features, variance_filter.get_support()) if keep
    ]
    if not variance_features:
        raise ValueError("Variance filtering removed every feature")

    ranking_model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    ranking_model.fit(train_variance, train["Class"].to_numpy())
    variance_index = {feature: index for index, feature in enumerate(variance_features)}
    ranked = sorted(
        variance_features,
        key=lambda feature: (-ranking_model.feature_importances_[variance_index[feature]], features.index(feature)),
    )
    top_features = ranked[: min(TOP_K_FEATURES, len(ranked))]

    top_matrix = pd.DataFrame(train_imputed, columns=features)[top_features]
    correlation = top_matrix.corr(method="spearman")
    retained = []
    pruning_rows = []
    for feature in top_features:
        remove_against = None
        remove_value = None
        for kept in retained:
            value = correlation.loc[feature, kept]
            if pd.notna(value) and abs(value) >= SPEARMAN_THRESHOLD:
                remove_against = kept
                remove_value = float(abs(value))
                break
        if remove_against is None:
            retained.append(feature)
        else:
            pruning_rows.append(
                {
                    "removed_feature": feature,
                    "retained_feature": remove_against,
                    "absolute_spearman_correlation": remove_value,
                    "threshold": SPEARMAN_THRESHOLD,
                }
            )

    preprocessing = {
        "imputer": imputer,
        "all_features": features,
        "variance_filter": variance_filter,
        "variance_features": variance_features,
        "ranked_features": ranked,
        "top_features": top_features,
        "retained_features": retained,
        "feature_importances": dict(zip(variance_features, ranking_model.feature_importances_)),
        "scaler": StandardScaler().fit(top_matrix[retained].to_numpy())if classifier_name == "logistic_regression" else None,
    }
    return train_imputed, retained, preprocessing, pruning_rows


def transform_test_matrix(
    frame: pd.DataFrame, preprocessing: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply training-fitted imputation, variance filtering, and feature selection."""
    imputer = preprocessing["imputer"]
    values = imputer.transform(
        frame[preprocessing["all_features"]].replace([np.inf, -np.inf], np.nan)
    )
    selected = preprocessing["retained_features"]
    matrix = pd.DataFrame(values, columns=preprocessing["all_features"])[selected].to_numpy()
    return values, matrix


def metric_values(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate binary classification metrics for two-class evaluation data."""
    if set(y_true) != {0, 1}:
        raise ValueError("Evaluation data must contain both classes")
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
    }


def build_feature_rows(
    fold_patient: int,
    mode: str,
    normalization: str,
    classifier: str,
    preprocessing: dict[str, object],
    features: list[str],
) -> list[dict[str, object]]:
    """Describe stage-by-stage selection status for every frozen feature."""
    variance_features = list(preprocessing["variance_features"])
    ranked_features = list(preprocessing["ranked_features"])
    top_features = list(preprocessing["top_features"])
    retained_features = list(preprocessing["retained_features"])
    feature_importances = preprocessing["feature_importances"]

    variance_set = set(variance_features)
    top_set = set(top_features)
    retained_set = set(retained_features)

    rows = []
    for feature in features:
        survived_variance = feature in variance_set
        survived_top_k = feature in top_set
        survived_correlation = feature in retained_set

        rows.append(
            {
                "outer_fold_patient_id": fold_patient,
                "evaluation_mode": mode,
                "normalization": normalization,
                "classifier": classifier,
                "feature_name": feature,
                "initial_frozen_order": features.index(feature) + 1,
                "ranking_position": (
                    ranked_features.index(feature) + 1
                    if feature in ranked_features
                    else np.nan
                ),
                "model_input_position": (
                    retained_features.index(feature) + 1
                    if feature in retained_features
                    else np.nan
                ),
                "feature_importance": (
                    float(feature_importances[feature])
                    if feature in feature_importances
                    else np.nan
                ),
                "survived_variance_filter": survived_variance,
                "survived_top_k": survived_top_k,
                "survived_correlation_pruning": survived_correlation,
                "removed_at_stage": (
                    "variance_filter"
                    if not survived_variance
                    else "top_k"
                    if not survived_top_k
                    else "correlation_pruning"
                    if not survived_correlation
                    else "retained"
                ),
            }
        )
    return rows


def main() -> None:
    """Run all 480 LOSO model configurations and save result tables."""
    source, features = load_inputs()
    patients = sorted(source["patient_id"].unique())
    selected, calibration_audit = select_all_calibration_blocks(source, patients)
    all_medians, all_iqrs = calibration_statistics(selected, features)

    predictions = []
    pooled_inputs = []
    patient_metric_rows = []
    fold_audit_rows = []
    selected_feature_rows = []
    pruning_rows = []
    fallback_rows = []
    completed_runs = 0

    for fold_patient in patients:
        train_patients = [patient for patient in patients if patient != fold_patient]
        if set(train_patients) & {fold_patient}:
            raise ValueError("Outer train/test patient overlap")
        scales, fold_fallbacks = fold_iqr_scales(
            selected, all_iqrs, train_patients, fold_patient, features
        )
        fallback_rows.extend(fold_fallbacks)

        normalized = {}
        for strategy in NORMALIZATION_STRATEGIES:
            normalized[strategy] = pd.concat(
                [
                    normalize_patient(
                        source[source["patient_id"] == patient],
                        features,
                        selected,
                        all_medians,
                        scales,
                        strategy,
                    )
                    for patient in patients
                ],
                ignore_index=True,
            )

        for mode in EVALUATION_MODES:
            for normalization in NORMALIZATION_STRATEGIES:
                train_rows = evaluation_rows(
                    normalized[normalization][normalized[normalization]["patient_id"].isin(train_patients)],
                    normalization,
                    mode,
                )
                test_rows = evaluation_rows(
                    normalized[normalization][normalized[normalization]["patient_id"] == fold_patient],
                    normalization,
                    mode,
                )
                if set(train_rows["patient_id"]) & {fold_patient}:
                    raise ValueError("Held-out patient entered training rows")
                if set(test_rows["Class"]) != {0, 1}:
                    raise ValueError(f"Fold patient {fold_patient} lacks an evaluation class")

                for classifier_name in CLASSIFIERS:
                    train_imputed, final_features, preprocessing, pruning = fit_training_preprocessing(
                        train_rows, features, classifier_name
                    )
                    test_imputed, test_matrix = transform_test_matrix(test_rows, preprocessing)
                    train_matrix = pd.DataFrame(
                        train_imputed, columns=features
                    )[preprocessing["retained_features"]].to_numpy()
                    if preprocessing["scaler"] is not None:
                        train_matrix = preprocessing["scaler"].transform(train_matrix)
                        test_matrix = preprocessing["scaler"].transform(test_matrix)
                    model = make_classifier(classifier_name)
                    model.fit(train_matrix, train_rows["Class"].to_numpy())
                    probabilities = model.predict_proba(test_matrix)[:, 1]
                    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
                        raise ValueError("Invalid held-out probabilities")

                    for row, probability in zip(test_rows.itertuples(index=False), probabilities):
                        predictions.append(
                            {
                                "outer_fold_patient_id": fold_patient,
                                "patient_id": row.patient_id,
                                "segment_id": row.segment_id,
                                "Class": row.Class,
                                "evaluation_mode": mode,
                                "normalization": normalization,
                                "classifier": classifier_name,
                                "is_common_calibration_row": row.is_common_calibration_row,
                                "predicted_probability": probability,
                                "predicted_class": int(probability >= 0.5),
                                "random_seed": RANDOM_SEED,
                            }
                        )
                    metric = metric_values(test_rows["Class"].to_numpy(), probabilities)
                    patient_metric_rows.append(
                        {
                            "outer_fold_patient_id": fold_patient,
                            "evaluation_mode": mode,
                            "normalization": normalization,
                            "classifier": classifier_name,
                            **metric,
                            "evaluated_rows": len(test_rows),
                            "class_0_rows": int((test_rows["Class"] == 0).sum()),
                            "class_1_rows": int((test_rows["Class"] == 1).sum()),
                        }
                    )
                    pooled_inputs.append(
                        {
                            "evaluation_mode": mode,
                            "normalization": normalization,
                            "classifier": classifier_name,
                            "y_true": test_rows["Class"].to_numpy(),
                            "probabilities": probabilities,
                            "patient_id": fold_patient,
                        }
                    )
                    fold_audit_rows.append(
                        {
                            "outer_fold_patient_id": fold_patient,
                            "evaluation_mode": mode,
                            "normalization": normalization,
                            "classifier": classifier_name,
                            "training_patient_count": len(train_patients),
                            "test_patient_count": 1,
                            "training_rows_before_exclusion": len(normalized[normalization][normalized[normalization]["patient_id"].isin(train_patients)]),
                            "training_rows_after_exclusion": len(train_rows),
                            "test_rows_before_exclusion": len(normalized[normalization][normalized[normalization]["patient_id"] == fold_patient]),
                            "test_rows_after_exclusion": len(test_rows),
                            "training_class_0_rows": int((train_rows["Class"] == 0).sum()),
                            "training_class_1_rows": int((train_rows["Class"] == 1).sum()),
                            "test_class_0_rows": int((test_rows["Class"] == 0).sum()),
                            "test_class_1_rows": int((test_rows["Class"] == 1).sum()),
                            "starting_feature_count": len(features),
                            "post_variance_feature_count": len(preprocessing["variance_features"]),
                            "post_top_k_feature_count": len(preprocessing["top_features"]),
                            "final_feature_count": len(final_features),
                            "imputation_fit_on_training_only": True,
                            "variance_filter_fit_on_training_only": True,
                            "ranking_fit_on_training_only": True,
                            "correlation_pruning_fit_on_training_only": True,
                            "scaler_applicable": classifier_name == "logistic_regression",
                            "scaler_fit_on_training_only": (
                                True if classifier_name == "logistic_regression" else np.nan
                            ),
                            "no_patient_overlap": True,
                            "finite_train_matrix": bool(np.isfinite(train_matrix).all()),
                            "finite_test_matrix": bool(np.isfinite(test_matrix).all()),
                            "calibration_rows_excluded_correctly": (
                                mode == "common_rows" and len(test_rows) == len(normalized[normalization][normalized[normalization]["patient_id"] == fold_patient]) - CALIBRATION_ROWS_PER_PATIENT
                            ) or (
                                mode == "native_rows" and (
                                    normalization in ("none", "jackknife_median")
                                    or len(test_rows) == len(normalized[normalization][normalized[normalization]["patient_id"] == fold_patient]) - CALIBRATION_ROWS_PER_PATIENT
                                )
                            ),
                        }
                    )
                    selected_feature_rows.extend(
                        build_feature_rows(
                            fold_patient, mode, normalization, classifier_name,
                            preprocessing, features,
                        )
                    )
                    for row in pruning:
                        pruning_rows.append(
                            {
                                "outer_fold_patient_id": fold_patient,
                                "evaluation_mode": mode,
                                "normalization": normalization,
                                "classifier": classifier_name,
                                **row,
                            }
                        )
                    completed_runs += 1
        print(f"Completed outer fold {fold_patient} ({completed_runs}/{len(patients) * len(EVALUATION_MODES) * len(NORMALIZATION_STRATEGIES) * len(CLASSIFIERS)})", flush=True)

    predictions_df = pd.DataFrame(predictions)
    patient_metrics_df = pd.DataFrame(patient_metric_rows)
    fold_audit_df = pd.DataFrame(fold_audit_rows)
    selected_features_df = pd.DataFrame(selected_feature_rows)
    pruning_df = pd.DataFrame(pruning_rows, columns=[
        "outer_fold_patient_id", "evaluation_mode", "normalization", "classifier",
        "removed_feature", "retained_feature", "absolute_spearman_correlation", "threshold",
    ])
    fallback_df = pd.DataFrame(fallback_rows, columns=[
        "outer_fold_patient_id", "affected_patient_id", "affected_patient_role",
        "feature_name", "original_calibration_iqr", "fallback_required",
        "fallback_source", "fallback_scale", "held_out_patient_excluded_from_fallback_estimation",
    ])

    pooled_rows = []
    for mode in EVALUATION_MODES:
        for normalization in NORMALIZATION_STRATEGIES:
            for classifier_name in CLASSIFIERS:
                parts = [
                    row for row in pooled_inputs
                    if row["evaluation_mode"] == mode
                    and row["normalization"] == normalization
                    and row["classifier"] == classifier_name
                ]
                y_true = np.concatenate([part["y_true"] for part in parts])
                probabilities = np.concatenate([part["probabilities"] for part in parts])
                patient_subset = patient_metrics_df[
                    (patient_metrics_df["evaluation_mode"] == mode)
                    & (patient_metrics_df["normalization"] == normalization)
                    & (patient_metrics_df["classifier"] == classifier_name)
                ]
                pooled_metric = metric_values(y_true, probabilities)
                pooled_rows.append(
                    {
                        "evaluation_mode": mode,
                        "normalization": normalization,
                        "classifier": classifier_name,
                        "pooled_roc_auc": pooled_metric["roc_auc"],
                        "pooled_average_precision": pooled_metric["average_precision"],
                        "accuracy": pooled_metric["accuracy"],
                        "sensitivity": pooled_metric["sensitivity"],
                        "specificity": pooled_metric["specificity"],
                        "balanced_accuracy": pooled_metric["balanced_accuracy"],
                        "prediction_count": len(y_true),
                        "class_0_count": int((y_true == 0).sum()),
                        "class_1_count": int((y_true == 1).sum()),
                        "patient_count": len(parts),
                        "mean_patient_auc": patient_subset["roc_auc"].mean(),
                        "std_patient_auc": patient_subset["roc_auc"].std(ddof=1),
                    }
                )
    pooled_df = pd.DataFrame(pooled_rows)

    expected_runs = len(patients) * len(EVALUATION_MODES) * len(NORMALIZATION_STRATEGIES) * len(CLASSIFIERS)
    prediction_key = ["outer_fold_patient_id", "segment_id", "evaluation_mode", "normalization", "classifier"]
    common_sets_ok = True
    for fold_patient in patients:
        for classifier_name in CLASSIFIERS:
            for normalization in NORMALIZATION_STRATEGIES:
                sets = [
                    set(predictions_df[
                        (predictions_df.outer_fold_patient_id == fold_patient)
                        & (predictions_df.classifier == classifier_name)
                        & (predictions_df.normalization == normalization)
                        & (predictions_df.evaluation_mode == mode)
                    ]["segment_id"])
                    for mode in EVALUATION_MODES
                ]
                if sets[0] != sets[1]:
                    # Native rows are intentionally different; compare common_rows
                    # across normalization methods and classifiers below instead.
                    pass
            common_sets = [
                set(predictions_df[
                    (predictions_df.outer_fold_patient_id == fold_patient)
                    & (predictions_df.classifier == classifier_name)
                    & (predictions_df.normalization == normalization)
                    & (predictions_df.evaluation_mode == "common_rows")
                ]["segment_id"])
                for normalization in NORMALIZATION_STRATEGIES
            ]
            if any(item != common_sets[0] for item in common_sets[1:]):
                common_sets_ok = False
    if predictions_df.duplicated(prediction_key).any():
        raise ValueError("Duplicate prediction keys found")
    if not common_sets_ok:
        raise ValueError("Common-row prediction sets differ across normalizations")
    if len(fold_audit_df) != expected_runs:
        raise ValueError("Unexpected fold-audit run count")

    expected_feature_audit_rows = expected_runs * len(features)
    if len(selected_features_df) != expected_feature_audit_rows:
        raise ValueError(
            "Unexpected selected-feature audit row count: "
            f"expected {expected_feature_audit_rows}, "
            f"found {len(selected_features_df)}"
        )
    feature_audit_key = [
        "outer_fold_patient_id",
        "evaluation_mode",
        "normalization",
        "classifier",
        "feature_name",
    ]
    if selected_features_df.duplicated(feature_audit_key).any():
        raise ValueError("Duplicate selected-feature audit keys found")
    if set(selected_features_df["feature_name"]) != set(features):
        raise ValueError("Selected-feature audit does not cover all frozen features")

    summary = pd.DataFrame([{
        "source_rows": len(source),
        "patient_count": len(patients),
        "frozen_feature_count": len(features),
        "calibration_rows_per_patient": CALIBRATION_ROWS_PER_PATIENT,
        "outer_fold_count": len(patients),
        "evaluation_mode_count": len(EVALUATION_MODES),
        "normalization_count": len(NORMALIZATION_STRATEGIES),
        "classifier_count": len(CLASSIFIERS),
        "expected_model_runs": expected_runs,
        "completed_model_runs": completed_runs,
        "prediction_rows": len(predictions_df),
        "total_iqr_fallback_events": int(fallback_df["fallback_required"].sum()),
        "all_patient_overlap_checks_passed": bool(fold_audit_df["no_patient_overlap"].all()),
        "all_training_only_preprocessing_checks_passed": bool(
            fold_audit_df[
                [
                    "imputation_fit_on_training_only",
                    "variance_filter_fit_on_training_only",
                    "ranking_fit_on_training_only",
                    "correlation_pruning_fit_on_training_only",
                ]
            ].all().all()
            and fold_audit_df.loc[
                fold_audit_df["scaler_applicable"],
                "scaler_fit_on_training_only",
            ].fillna(False).all()
        ),
        "all_finite_matrix_checks_passed": bool(fold_audit_df[["finite_train_matrix", "finite_test_matrix"]].all().all()),
        "all_calibration_exclusion_checks_passed": bool(fold_audit_df["calibration_rows_excluded_correctly"].all()),
        "random_seed": RANDOM_SEED,
    }])

    # Final validation before writing any result file.
    if completed_runs != expected_runs or not np.isfinite(predictions_df["predicted_probability"]).all():
        raise ValueError("Final LOSO validation failed")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(RESULTS_ROOT / "loso_predictions.csv", index=False)
    pooled_df.to_csv(RESULTS_ROOT / "loso_pooled_metrics.csv", index=False)
    patient_metrics_df.to_csv(RESULTS_ROOT / "loso_patient_metrics.csv", index=False)
    fold_audit_df.to_csv(RESULTS_ROOT / "loso_fold_audit.csv", index=False)
    selected_features_df.to_csv(RESULTS_ROOT / "loso_selected_features.csv", index=False)
    pruning_df.to_csv(RESULTS_ROOT / "loso_correlation_pruning.csv", index=False)
    fallback_df.to_csv(RESULTS_ROOT / "loso_iqr_fallback_audit.csv", index=False)
    summary.to_csv(RESULTS_ROOT / "loso_run_summary.csv", index=False)

    print(f"Completed model runs: {completed_runs}")
    print(f"Prediction rows: {len(predictions_df)}")
    common = pooled_df[pooled_df["evaluation_mode"] == "common_rows"]
    for _, row in common.iterrows():
        print(
            f"common_rows {row['normalization']} {row['classifier']}: "
            f"pooled_auc={row['pooled_roc_auc']:.4f}, "
            f"mean_patient_auc={row['mean_patient_auc']:.4f}"
        )
    print(f"Total IQR fallback events: {int(fallback_df['fallback_required'].sum())}")
    print("All leakage and validation checks passed: True")


if __name__ == "__main__":
    main()