from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import aqa_score_metrics
from .util import sha256_file, write_json
from .v5_data import V5_RESULTS_ROOT
from .v6_modeling import V6Assets, load_v6_assets
from .v7_data import (
    V7_RESULTS_ROOT,
    V7_RUN_ROOT,
    load_risk_manifest_v7,
    load_v7_contract,
    require_v7_audit,
)


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _reference_map_for_pool(assets: V6Assets, pool: np.ndarray) -> dict[str, np.ndarray]:
    contract = load_v7_contract()
    frame = assets.frame
    actions = frame.action_type.astype(str).to_numpy()
    families = frame.event_family.astype(str).to_numpy()
    normalized = _normalized_rows(assets.global_latent)
    primary = int(contract["data"]["references"])
    slots = primary + int(contract["data"]["alternate_references"])
    references = np.full((len(frame), slots), -1, dtype=np.int32)
    distances = np.full((len(frame), slots), np.nan, dtype=np.float32)
    valid_count = np.zeros(len(frame), dtype=np.int16)
    for query in range(len(frame)):
        candidates = pool[
            (actions[pool] == actions[query])
            & (families[pool] != families[query])
            & (pool != query)
        ]
        if not len(candidates):
            continue
        candidate_distance = 1.0 - normalized[candidates] @ normalized[query]
        order = np.argsort(candidate_distance, kind="stable")[:slots]
        selected = candidates[order]
        references[query, : len(selected)] = selected
        distances[query, : len(selected)] = candidate_distance[order]
        valid_count[query] = min(len(selected), primary)
    open_set = valid_count < int(contract["data"]["minimum_references"])
    q_true = frame.execution_quality.to_numpy(dtype=float)
    primary_refs = references[:, :primary]
    primary_distance = distances[:, :primary]
    available = primary_distance[np.isfinite(primary_distance)]
    temperature = max(float(np.median(available)), 1e-3)
    weights = np.zeros_like(primary_distance, dtype=np.float32)
    for query in range(len(frame)):
        valid = primary_refs[query] >= 0
        if not np.any(valid):
            continue
        ref = primary_refs[query, valid]
        ref_error = np.abs(q_true[ref] - assets.teacher_quality[ref])
        ref_dispersion = float(np.std(q_true[ref], ddof=0))
        reliability = 1.0 / (1.0 + ref_error + ref_dispersion)
        raw = np.exp(-primary_distance[query, valid] / temperature) * reliability
        weights[query, valid] = raw / max(float(raw.sum()), 1e-8)
    return {
        "references": references,
        "distances": distances,
        "weights": weights,
        "valid_reference_count": valid_count,
        "open_set": open_set,
        "pool_indices": np.asarray(pool, dtype=np.int32),
        "temperature": np.asarray([temperature], dtype=np.float32),
    }


def build_reference_map_v7(final: bool) -> dict:
    require_v7_audit()
    assets = load_v6_assets()
    roles = load_v7_contract()["adapter"]["final_reference_roles" if final else "development_reference_roles"]
    pool = np.flatnonzero(assets.frame.analysis_role.isin(roles).to_numpy())
    mapping = _reference_map_for_pool(assets, pool)
    suffix = "final" if final else "development"
    path = V7_RESULTS_ROOT / "02_BASELINES" / f"reference_map_{suffix}_v7.npz"
    np.savez_compressed(path, **mapping)
    actions = assets.frame.action_type.astype(str).to_numpy()
    families = assets.frame.event_family.astype(str).to_numpy()
    legal = True
    refs = mapping["references"]
    for query in range(len(refs)):
        for ref in refs[query, : int(mapping["valid_reference_count"][query])]:
            legal &= bool(ref in pool and actions[ref] == actions[query] and families[ref] != families[query])
    result = {
        "status": "PASS" if legal else "FAIL",
        "stage": suffix,
        "pool_roles": list(roles),
        "pool_rows": int(len(pool)),
        "open_set_rows": int(mapping["open_set"].sum()),
        "all_references_legal": bool(legal),
        "reference_map_sha256": sha256_file(path),
    }
    write_json(V7_RESULTS_ROOT / "02_BASELINES" / f"reference_summary_{suffix}_v7.json", result)
    return result


def load_reference_map_v7(final: bool) -> dict[str, np.ndarray]:
    suffix = "final" if final else "development"
    path = V7_RESULTS_ROOT / "02_BASELINES" / f"reference_map_{suffix}_v7.npz"
    if not path.exists():
        build_reference_map_v7(final)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _weighted_reference_statistics(assets: V6Assets, mapping: dict[str, np.ndarray]) -> np.ndarray:
    primary = int(load_v7_contract()["data"]["references"])
    refs = mapping["references"][:, :primary].astype(int)
    safe = np.maximum(refs, 0)
    valid = refs >= 0
    weights = np.where(valid, mapping["weights"][:, :primary], 0.0).astype(float)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    q_true = assets.frame.execution_quality.to_numpy(dtype=float)[safe]
    q_teacher = assets.teacher_quality[safe]
    distance = np.nan_to_num(mapping["distances"][:, :primary], nan=2.0)
    weighted_true = np.sum(weights * q_true, axis=1)
    weighted_teacher = np.sum(weights * q_teacher, axis=1)
    weighted_residual = np.sum(weights * (q_true - q_teacher), axis=1)
    weighted_distance = np.sum(weights * distance, axis=1)
    mean_true = np.sum(weights * q_true, axis=1, keepdims=True)
    dispersion = np.sqrt(np.sum(weights * (q_true - mean_true) ** 2, axis=1))
    return np.column_stack((weighted_true, weighted_teacher, weighted_residual, weighted_distance, dispersion)).astype(np.float32)


def make_ridge_features_v7(
    assets: V6Assets,
    mapping: dict[str, np.ndarray],
    global_latent: np.ndarray | None = None,
    teacher_quality: np.ndarray | None = None,
    query_indices: np.ndarray | None = None,
) -> np.ndarray:
    global_values = assets.global_latent if global_latent is None else np.asarray(global_latent, dtype=np.float32)
    teacher_values = assets.teacher_quality if teacher_quality is None else np.asarray(teacher_quality, dtype=np.float32)
    statistics = _weighted_reference_statistics(assets, mapping)
    if query_indices is not None:
        statistics = statistics[np.asarray(query_indices, dtype=int)]
    return np.concatenate((global_values, teacher_values[:, None], statistics), axis=1).astype(np.float32)


def _sample_weights(manifest: pd.DataFrame, indices: np.ndarray, lambda_error: float, lambda_judge: float) -> np.ndarray:
    error = manifest.high_error_risk_proxy.to_numpy(dtype=bool)[indices]
    judge = manifest.high_judge_risk.to_numpy(dtype=bool)[indices]
    eligible = manifest.disagreement_primary_eligible.to_numpy(dtype=bool)[indices]
    return 1.0 + float(lambda_error) * error + float(lambda_judge) * (judge & eligible)


def _fit_ridge_artifact(
    features: np.ndarray,
    residual: np.ndarray,
    indices: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None,
    clip_source: np.ndarray | None = None,
) -> dict:
    contract = load_v7_contract()
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    model.fit(features[indices], residual[indices], ridge__sample_weight=sample_weight)
    source = residual[indices] if clip_source is None else np.asarray(clip_source, dtype=float)
    lower_q, upper_q = contract["adapter"]["residual_clip_quantiles"]
    lower, upper = np.quantile(source, [float(lower_q), float(upper_q)])
    return {
        "model": model,
        "alpha": float(alpha),
        "clip_lower": float(lower),
        "clip_upper": float(upper),
    }


def predict_ridge_artifact(artifact: dict, features: np.ndarray, teacher_quality: np.ndarray) -> np.ndarray:
    delta = artifact["model"].predict(features)
    delta = np.clip(delta, float(artifact["clip_lower"]), float(artifact["clip_upper"]))
    return np.asarray(teacher_quality, dtype=float) + delta


def _score_metrics(frame: pd.DataFrame, quality: np.ndarray, indices: np.ndarray) -> dict:
    score = 3.0 * frame.difficulty.to_numpy(dtype=float) * quality
    return aqa_score_metrics(frame.dive_score.to_numpy(dtype=float)[indices], score[indices])


def _high_risk_mae(frame: pd.DataFrame, score: np.ndarray, manifest: pd.DataFrame, indices: np.ndarray) -> tuple[float, int]:
    high = manifest.high_judge_risk.to_numpy(dtype=bool)[indices]
    eligible = manifest.disagreement_primary_eligible.to_numpy(dtype=bool)[indices]
    selected = indices[high & eligible]
    if not len(selected):
        return float("nan"), 0
    error = np.abs(score[selected] - frame.dive_score.to_numpy(dtype=float)[selected])
    return float(error.mean()), int(len(selected))


def train_baselines_v7() -> dict:
    require_v7_audit()
    manifest = load_risk_manifest_v7()
    assets = load_v6_assets()
    frame = assets.frame
    contract = load_v7_contract()
    build_reference_map_v7(final=False)
    mapping = load_reference_map_v7(final=False)
    features = make_ridge_features_v7(assets, mapping)
    residual = frame.execution_quality.to_numpy(dtype=float) - assets.teacher_quality
    fit = np.flatnonzero(frame.analysis_role.to_numpy() == "fit")
    validation = np.flatnonzero(frame.analysis_role.to_numpy() == "validation")
    teacher_metrics = _score_metrics(frame, assets.teacher_quality, validation)
    teacher_score = 3.0 * frame.difficulty.to_numpy(dtype=float) * assets.teacher_quality
    rows: list[dict] = []
    artifacts: dict[str, dict] = {}

    linear = LinearRegression().fit(assets.teacher_quality[fit, None], frame.execution_quality.to_numpy(dtype=float)[fit])
    linear_quality = linear.predict(assets.teacher_quality[:, None])
    linear_metrics = _score_metrics(frame, linear_quality, validation)
    linear_score = 3.0 * frame.difficulty.to_numpy(dtype=float) * linear_quality
    linear_high, high_n = _high_risk_mae(frame, linear_score, manifest, validation)
    rows.append({"name": "global_linear_calibration", "kind": "linear", **linear_metrics, "high_judge_mae": linear_high, "high_judge_n": high_n})

    ref_stats = _weighted_reference_statistics(assets, mapping)
    same_quality = assets.teacher_quality + ref_stats[:, 2]
    same_quality[mapping["open_set"].astype(bool)] = assets.teacher_quality[mapping["open_set"].astype(bool)]
    same_metrics = _score_metrics(frame, same_quality, validation)
    same_score = 3.0 * frame.difficulty.to_numpy(dtype=float) * same_quality
    same_high, high_n = _high_risk_mae(frame, same_score, manifest, validation)
    rows.append({"name": "same_action_reference_residual", "kind": "reference", **same_metrics, "high_judge_mae": same_high, "high_judge_n": high_n})

    for alpha in contract["adapter"]["ridge_alphas"]:
        artifact = _fit_ridge_artifact(features, residual, fit, float(alpha), None)
        quality = predict_ridge_artifact(artifact, features, assets.teacher_quality)
        quality[mapping["open_set"].astype(bool)] = assets.teacher_quality[mapping["open_set"].astype(bool)]
        metrics = _score_metrics(frame, quality, validation)
        score = 3.0 * frame.difficulty.to_numpy(dtype=float) * quality
        high_mae, high_n = _high_risk_mae(frame, score, manifest, validation)
        name = f"plain_ridge_alpha_{float(alpha):g}"
        rows.append({"name": name, "kind": "plain_ridge", "alpha": float(alpha), "lambda_error": 0.0, "lambda_judge": 0.0, **metrics, "high_judge_mae": high_mae, "high_judge_n": high_n})
        artifacts[name] = artifact
        for lambda_error in contract["adapter"]["lambda_error"]:
            for lambda_judge in contract["adapter"]["lambda_judge"]:
                weight = _sample_weights(manifest, fit, float(lambda_error), float(lambda_judge))
                risk_artifact = _fit_ridge_artifact(features, residual, fit, float(alpha), weight)
                risk_quality = predict_ridge_artifact(risk_artifact, features, assets.teacher_quality)
                risk_quality[mapping["open_set"].astype(bool)] = assets.teacher_quality[mapping["open_set"].astype(bool)]
                risk_metrics = _score_metrics(frame, risk_quality, validation)
                risk_score = 3.0 * frame.difficulty.to_numpy(dtype=float) * risk_quality
                risk_high, risk_n = _high_risk_mae(frame, risk_score, manifest, validation)
                risk_name = f"risk_ridge_a{float(alpha):g}_le{float(lambda_error):g}_lj{float(lambda_judge):g}"
                rows.append({"name": risk_name, "kind": "risk_ridge", "alpha": float(alpha), "lambda_error": float(lambda_error), "lambda_judge": float(lambda_judge), **risk_metrics, "high_judge_mae": risk_high, "high_judge_n": risk_n})
                artifacts[risk_name] = risk_artifact

    trials = pd.DataFrame(rows)
    trials["spearman_drop_from_teacher"] = float(teacher_metrics["spearman"]) - trials.spearman
    trials["eligible"] = trials.spearman_drop_from_teacher <= float(contract["adapter"]["maximum_validation_spearman_drop"])
    risk_eligible = trials[(trials.kind == "risk_ridge") & trials.eligible & trials.high_judge_mae.notna()].copy()
    if risk_eligible.empty:
        raise RuntimeError("No risk-balanced Ridge candidate passed the validation non-inferiority condition")
    selected_row = risk_eligible.sort_values(["high_judge_mae", "mae", "alpha", "lambda_error", "lambda_judge"], kind="stable").iloc[0]
    plain_row = trials[(trials.kind == "plain_ridge") & trials.eligible].sort_values(["mae", "alpha"], kind="stable").iloc[0]
    trials_path = V7_RESULTS_ROOT / "02_BASELINES" / "baseline_trials_v7.csv"
    trials.to_csv(trials_path, index=False)
    selected_artifact = artifacts[str(selected_row["name"])]
    selected_path = V7_RUN_ROOT / "checkpoints" / "selected_risk_ridge_v7.joblib"
    joblib.dump(selected_artifact, selected_path)
    selection = {
        "status": "PASS",
        "teacher_validation_metrics": teacher_metrics,
        "selected": selected_row.where(pd.notna(selected_row), None).to_dict(),
        "plain_ridge": plain_row.where(pd.notna(plain_row), None).to_dict(),
        "selection_rule": "Spearman drop <= 0.01, then minimum high-judge-risk MAE",
        "official_test_used_for_selection": False,
        "trials_sha256": sha256_file(trials_path),
        "artifact_sha256": sha256_file(selected_path),
    }
    write_json(V7_RESULTS_ROOT / "02_BASELINES" / "selected_model_v7.json", selection)
    return selection


def _fit_from_config(
    assets: V6Assets,
    mapping: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    train_indices: np.ndarray,
    config: dict,
) -> dict:
    features = make_ridge_features_v7(assets, mapping)
    residual = assets.frame.execution_quality.to_numpy(dtype=float) - assets.teacher_quality
    weight = _sample_weights(
        manifest, train_indices, float(config["lambda_error"]), float(config["lambda_judge"])
    )
    return _fit_ridge_artifact(features, residual, train_indices, float(config["alpha"]), weight)


def train_final_v7() -> dict:
    selection_path = V7_RESULTS_ROOT / "02_BASELINES" / "selected_model_v7.json"
    if not selection_path.exists():
        raise RuntimeError("Run train-baselines --protocol v7 first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = load_risk_manifest_v7()
    assets = load_v6_assets()
    frame = assets.frame
    contract = load_v7_contract()
    build_reference_map_v7(final=True)
    mapping = load_reference_map_v7(final=True)
    features = make_ridge_features_v7(assets, mapping)
    residual = frame.execution_quality.to_numpy(dtype=float) - assets.teacher_quality
    development = np.flatnonzero(frame.analysis_role.isin(("fit", "validation")).to_numpy())
    calibration = np.flatnonzero(frame.analysis_role.to_numpy() == "calibration")
    test = np.flatnonzero(frame.analysis_role.to_numpy() == "official_test")
    selected_config = selection["selected"]
    final_artifact = _fit_from_config(assets, mapping, manifest, development, selected_config)
    final_path = V7_RUN_ROOT / "checkpoints" / "final_risk_ridge_v7.joblib"
    joblib.dump(final_artifact, final_path)
    selected_quality = predict_ridge_artifact(final_artifact, features, assets.teacher_quality)
    open_set = mapping["open_set"].astype(bool)
    selected_quality[open_set] = assets.teacher_quality[open_set]

    plain_config = selection["plain_ridge"]
    plain_artifact = _fit_ridge_artifact(features, residual, development, float(plain_config["alpha"]), None)
    plain_quality = predict_ridge_artifact(plain_artifact, features, assets.teacher_quality)
    plain_quality[open_set] = assets.teacher_quality[open_set]
    plain_path = V7_RUN_ROOT / "checkpoints" / "final_plain_ridge_v7.joblib"
    joblib.dump(plain_artifact, plain_path)

    linear = LinearRegression().fit(assets.teacher_quality[development, None], frame.execution_quality.to_numpy(dtype=float)[development])
    linear_quality = linear.predict(assets.teacher_quality[:, None])
    ref_stats = _weighted_reference_statistics(assets, mapping)
    same_quality = assets.teacher_quality + ref_stats[:, 2]
    same_quality[open_set] = assets.teacher_quality[open_set]
    score_scale = 3.0 * frame.difficulty.to_numpy(dtype=float)
    output = frame[[
        "clip_uid", "official_split", "analysis_role", "source_role", "event_family",
        "action_type", "difficulty", "dive_score", "execution_quality", "judge_sample_sd",
        "disagreement_primary_eligible",
    ]].copy()
    output["teacher_predicted_score"] = score_scale * assets.teacher_quality
    output["global_linear_predicted_score"] = score_scale * linear_quality
    output["same_action_reference_predicted_score"] = score_scale * same_quality
    output["plain_ridge_predicted_score"] = score_scale * plain_quality
    output["trustdive_predicted_quality"] = selected_quality
    output["trustdive_predicted_score"] = score_scale * selected_quality
    output["teacher_adapter_gap"] = np.abs(output.trustdive_predicted_score - output.teacher_predicted_score)
    output["reference_distance"] = ref_stats[:, 3]
    output["reference_dispersion"] = ref_stats[:, 4]
    output["valid_reference_count"] = mapping["valid_reference_count"]
    output["open_set"] = open_set
    teacher_v5 = pd.read_parquet(V5_RESULTS_ROOT / "01_REFERENCES" / "teacher_predictions_v5.parquet").set_index("clip_uid")
    output["teacher_uncertainty"] = teacher_v5.loc[output.clip_uid, "teacher_uncertainty"].to_numpy(dtype=float)
    calibration_error = np.abs(output.loc[calibration, "trustdive_predicted_score"] - output.loc[calibration, "dive_score"])
    conformal = float(np.quantile(calibration_error, float(contract["risk_task"]["conformal_coverage"]), method="higher"))
    local_scale = 1.0 + output.reference_dispersion.to_numpy(dtype=float)
    output["prediction_interval_width"] = 2.0 * conformal * local_scale / max(float(np.median(local_scale[calibration])), 1e-8)
    prediction_path = V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet"
    output.to_parquet(prediction_path, index=False)

    # Family-disjoint downstream out-of-fold predictions for error-risk labels.
    groups = frame.event_family.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=int(contract["risk_task"]["crossfit_folds"]))
    oof = np.full(len(frame), np.nan, dtype=float)
    oof_distance = np.full(len(frame), np.nan, dtype=float)
    oof_dispersion = np.full(len(frame), np.nan, dtype=float)
    oof_open = np.zeros(len(frame), dtype=bool)
    for fold, (train_local, held_local) in enumerate(splitter.split(development, groups=groups[development])):
        train_index = development[train_local]
        held_index = development[held_local]
        fold_map = _reference_map_for_pool(assets, train_index)
        fold_features = make_ridge_features_v7(assets, fold_map)
        fold_artifact = _fit_from_config(assets, fold_map, manifest, train_index, selected_config)
        fold_quality = predict_ridge_artifact(fold_artifact, fold_features, assets.teacher_quality)
        fold_quality[fold_map["open_set"].astype(bool)] = assets.teacher_quality[fold_map["open_set"].astype(bool)]
        oof[held_index] = score_scale[held_index] * fold_quality[held_index]
        fold_stats = _weighted_reference_statistics(assets, fold_map)
        oof_distance[held_index] = fold_stats[held_index, 3]
        oof_dispersion[held_index] = fold_stats[held_index, 4]
        oof_open[held_index] = fold_map["open_set"][held_index]
        joblib.dump(fold_artifact, V7_RUN_ROOT / "checkpoints" / f"crossfit_risk_ridge_fold_{fold}_v7.joblib")
    crossfit = output[["clip_uid", "analysis_role", "event_family", "difficulty", "dive_score"]].copy()
    crossfit["oof_predicted_score"] = oof
    crossfit["oof_teacher_adapter_gap"] = np.abs(oof - output.teacher_predicted_score.to_numpy(dtype=float))
    crossfit["oof_reference_distance"] = oof_distance
    crossfit["oof_reference_dispersion"] = oof_dispersion
    crossfit["oof_open_set"] = oof_open
    crossfit_path = V7_RESULTS_ROOT / "03_SCORE" / "crossfit_predictions_v7.parquet"
    crossfit.to_parquet(crossfit_path, index=False)

    comparison_rows = []
    for name, column in (
        ("RICA2 deterministic", "teacher_predicted_score"),
        ("global linear calibration", "global_linear_predicted_score"),
        ("same-action reference residual", "same_action_reference_predicted_score"),
        ("plain latent reference Ridge", "plain_ridge_predicted_score"),
        ("TrustDive-Risk", "trustdive_predicted_score"),
    ):
        comparison_rows.append({"model": name, **aqa_score_metrics(output.loc[test, "dive_score"], output.loc[test, column])})
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = V7_RESULTS_ROOT / "03_SCORE" / "ablation_summary_v7.csv"
    comparison.to_csv(comparison_path, index=False)
    result = {
        "status": "PASS",
        "selected_config": selected_config,
        "test_rows": int(len(test)),
        "open_set_test_rows": int(open_set[test].sum()),
        "conformal_half_width": conformal,
        "prediction_sha256": sha256_file(prediction_path),
        "crossfit_sha256": sha256_file(crossfit_path),
        "ablation_sha256": sha256_file(comparison_path),
        "final_artifact_sha256": sha256_file(final_path),
        "official_test_used_for_model_selection": False,
    }
    write_json(V7_RESULTS_ROOT / "03_SCORE" / "score_summary_v7.json", result)
    return result


def load_final_artifact_v7() -> dict:
    path = V7_RUN_ROOT / "checkpoints" / "final_risk_ridge_v7.joblib"
    if not path.exists():
        raise RuntimeError("Run train --protocol v7 first")
    return joblib.load(path)

