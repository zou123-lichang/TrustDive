from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, f1_score

from .config import PROJECT_ROOT, RESULTS_ROOT, RUNS_ROOT
from .metrics import aqa_score_metrics
from .modeling import ordered_phase_targets
from .util import git_head, sha256_file, write_json
from .v4_counterfactual import PHASES, exact_three_phase_shapley, hybrid_sequence
from .v5_counterfactual import _phase_labels
from .v6_attribution import _load_sequences, _predict_teacher_latent
from .v6_modeling import load_v6_assets
from .v7_attribution import _coalitions, _cosine_rows
from .v7_data import load_v7_contract
from .v7_modeling import (
    _fit_ridge_artifact,
    _score_metrics,
    _weighted_reference_statistics,
    load_final_artifact_v7,
    load_reference_map_v7,
    make_ridge_features_v7,
    predict_ridge_artifact,
)


REVISION_ROOT = RESULTS_ROOT / "MANUSCRIPT_REVISION_2026_09_03"
REVISION_RUN_ROOT = RUNS_ROOT / "manuscript_revision_2026_09_03"
CONTRACT_PATH = PROJECT_ROOT / "01_PROTOCOL" / "manuscript_revision_contract_2026-09-03.yaml"
V7_PREDICTIONS = RESULTS_ROOT / "V7_RISK_TASK" / "03_SCORE" / "predictions_v7.parquet"
V7_EVIDENCE = RESULTS_ROOT / "V7_RISK_TASK" / "04_PHASE_EVIDENCE" / "phase_evidence_v7.parquet"
PHASE_CACHE = RESULTS_ROOT / "V2_DISAGREEMENT" / "01_FEATURES" / "phase_predictions_videomae_official_v2.npz"


def ensure_revision_dirs() -> None:
    for path in (
        REVISION_ROOT / "01_COMPONENT_ABLATION",
        REVISION_ROOT / "02_PHASE_PARSER",
        REVISION_ROOT / "03_SHAPLEY_AUDIT",
        REVISION_ROOT / "04_ORACLE_ATTRIBUTION",
        REVISION_ROOT / "05_EFFICIENCY",
        REVISION_RUN_ROOT / "checkpoints",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _metric_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {
        "spearman": float(spearmanr(target, prediction).statistic),
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }


def _cluster_bootstrap_deltas(
    frame: pd.DataFrame,
    candidate: str,
    comparator: str,
    iterations: int = 10_000,
    seed: int = 20260903,
) -> list[dict]:
    target = frame.dive_score.to_numpy(dtype=float)
    cand = frame[candidate].to_numpy(dtype=float)
    comp = frame[comparator].to_numpy(dtype=float)
    groups = frame.event_family.astype(str).to_numpy()
    unique = np.asarray(sorted(np.unique(groups)), dtype=object)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    draws = {metric: np.empty(iterations, dtype=float) for metric in ("spearman", "mae", "rmse")}
    for iteration in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        cand_metrics = _metric_values(target[indices], cand[indices])
        comp_metrics = _metric_values(target[indices], comp[indices])
        for metric in draws:
            draws[metric][iteration] = cand_metrics[metric] - comp_metrics[metric]
    point_cand = _metric_values(target, cand)
    point_comp = _metric_values(target, comp)
    return [
        {
            "candidate": candidate.removesuffix("_predicted_score"),
            "comparator": comparator.removesuffix("_predicted_score"),
            "metric": metric,
            "delta_candidate_minus_comparator": point_cand[metric] - point_comp[metric],
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "iterations": int(iterations),
            "cluster_count": int(len(unique)),
            "seed": int(seed),
        }
        for metric, values in draws.items()
    ]


def _select_ridge(
    development_features: np.ndarray,
    final_features: np.ndarray,
    assets,
    name: str,
) -> tuple[np.ndarray, dict]:
    frame = assets.frame
    residual = frame.execution_quality.to_numpy(dtype=float) - assets.teacher_quality
    fit = np.flatnonzero(frame.analysis_role.to_numpy() == "fit")
    validation = np.flatnonzero(frame.analysis_role.to_numpy() == "validation")
    development = np.flatnonzero(frame.analysis_role.isin(("fit", "validation")).to_numpy())
    teacher_metrics = _score_metrics(frame, assets.teacher_quality, validation)
    rows: list[dict] = []
    for alpha in (10.0, 100.0, 1000.0):
        artifact = _fit_ridge_artifact(development_features, residual, fit, alpha, None)
        quality = predict_ridge_artifact(artifact, development_features, assets.teacher_quality)
        metrics = _score_metrics(frame, quality, validation)
        rows.append(
            {
                "alpha": alpha,
                **metrics,
                "eligible": bool(teacher_metrics["spearman"] - metrics["spearman"] <= 0.01),
            }
        )
    trials = pd.DataFrame(rows)
    eligible = trials[trials.eligible]
    if eligible.empty:
        raise RuntimeError(f"No eligible {name} Ridge candidate")
    selected = eligible.sort_values(["mae", "alpha"], kind="stable").iloc[0]
    artifact = _fit_ridge_artifact(final_features, residual, development, float(selected.alpha), None)
    joblib.dump(artifact, REVISION_RUN_ROOT / "checkpoints" / f"{name}.joblib")
    quality = predict_ridge_artifact(artifact, final_features, assets.teacher_quality)
    return quality, {
        "selected_alpha": float(selected.alpha),
        "validation": selected.where(pd.notna(selected), None).to_dict(),
        "selection_used_official_test": False,
    }


def run_component_ablation() -> dict:
    ensure_revision_dirs()
    assets = load_v6_assets()
    frame = assets.frame.reset_index(drop=True)
    development_map = load_reference_map_v7(final=False)
    final_map = load_reference_map_v7(final=True)
    ref_dev = _weighted_reference_statistics(assets, development_map)
    ref_final = _weighted_reference_statistics(assets, final_map)
    latent = np.concatenate((assets.global_latent, assets.teacher_quality[:, None]), axis=1).astype(np.float32)
    reference_dev = np.concatenate((assets.teacher_quality[:, None], ref_dev), axis=1).astype(np.float32)
    reference_final = np.concatenate((assets.teacher_quality[:, None], ref_final), axis=1).astype(np.float32)

    latent_quality, latent_selection = _select_ridge(latent, latent, assets, "latent_only_ridge")
    reference_quality, reference_selection = _select_ridge(
        reference_dev, reference_final, assets, "reference_only_ridge"
    )
    reference_quality[final_map["open_set"].astype(bool)] = assets.teacher_quality[
        final_map["open_set"].astype(bool)
    ]

    development = np.flatnonzero(frame.analysis_role.isin(("fit", "validation")).to_numpy())
    linear = LinearRegression().fit(
        assets.teacher_quality[development, None], frame.execution_quality.to_numpy(dtype=float)[development]
    )
    linear_quality = linear.predict(assets.teacher_quality[:, None])
    score_scale = 3.0 * frame.difficulty.to_numpy(dtype=float)
    frozen = pd.read_parquet(V7_PREDICTIONS).set_index("clip_uid").loc[frame.clip_uid].reset_index()
    output = frame[["clip_uid", "analysis_role", "event_family", "dive_score", "difficulty"]].copy()
    output["frozen_teacher_predicted_score"] = score_scale * assets.teacher_quality
    output["score_only_linear_predicted_score"] = score_scale * linear_quality
    output["latent_only_ridge_predicted_score"] = score_scale * latent_quality
    output["reference_only_ridge_predicted_score"] = score_scale * reference_quality
    output["full_latent_reference_ridge_predicted_score"] = frozen.plain_ridge_predicted_score.to_numpy(dtype=float)
    output["prespecified_trustdive_predicted_score"] = frozen.trustdive_predicted_score.to_numpy(dtype=float)
    output["reference_sparse"] = final_map["open_set"].astype(bool)
    prediction_path = REVISION_ROOT / "01_COMPONENT_ABLATION" / "component_predictions.parquet"
    output.to_parquet(prediction_path, index=False)

    test = output.analysis_role == "official_test"
    test_frame = output.loc[test].reset_index(drop=True)
    model_columns = [column for column in output if column.endswith("_predicted_score")]
    metrics_rows = []
    for column in model_columns:
        metrics = aqa_score_metrics(test_frame.dive_score, test_frame[column])
        metrics_rows.append({"model": column.removesuffix("_predicted_score"), **metrics})
    metrics = pd.DataFrame(metrics_rows)
    metrics_path = REVISION_ROOT / "01_COMPONENT_ABLATION" / "component_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    comparisons = (
        ("full_latent_reference_ridge_predicted_score", "latent_only_ridge_predicted_score"),
        ("full_latent_reference_ridge_predicted_score", "reference_only_ridge_predicted_score"),
        ("full_latent_reference_ridge_predicted_score", "score_only_linear_predicted_score"),
        ("prespecified_trustdive_predicted_score", "frozen_teacher_predicted_score"),
        ("prespecified_trustdive_predicted_score", "full_latent_reference_ridge_predicted_score"),
    )
    intervals = []
    for offset, (candidate, comparator) in enumerate(comparisons):
        intervals.extend(
            _cluster_bootstrap_deltas(test_frame, candidate, comparator, seed=20260903 + offset)
        )
    interval_frame = pd.DataFrame(intervals)
    interval_path = REVISION_ROOT / "01_COMPONENT_ABLATION" / "paired_clustered_intervals.csv"
    interval_frame.to_csv(interval_path, index=False)
    result = {
        "status": "COMPLETE",
        "official_test_n": int(test.sum()),
        "event_family_clusters": int(test_frame.event_family.nunique()),
        "primary_endpoint": "MAE",
        "latent_only_selection": latent_selection,
        "reference_only_selection": reference_selection,
        "component_predictions_sha256": sha256_file(prediction_path),
        "component_metrics_sha256": sha256_file(metrics_path),
        "paired_intervals_sha256": sha256_file(interval_path),
    }
    write_json(REVISION_ROOT / "01_COMPONENT_ABLATION" / "component_ablation_summary.json", result)
    return result


def _transition_positions(labels: np.ndarray) -> tuple[int, int]:
    first = int(np.flatnonzero(labels == 1)[0])
    second = int(np.flatnonzero(labels == 2)[0])
    return first, second


def _annotation_boundaries(frame_count: int, transitions_json: str) -> tuple[int, int]:
    transitions = [int(value) for value in json.loads(transitions_json)]
    boundaries = sorted(set([0, *[max(0, min(frame_count, value)) for value in transitions], frame_count]))
    if len(boundaries) < 4:
        boundaries = [0, round(frame_count / 3), round(2 * frame_count / 3), frame_count]
    return int(boundaries[1]), int(boundaries[-2])


def _ground_truth_labels(frame: pd.DataFrame, length: int) -> np.ndarray:
    return np.stack(
        [
            ordered_phase_targets(
                int(row.frame_count), json.loads(row.transition_frames_json), int(length)
            )
            for row in frame.itertuples(index=False)
        ]
    ).astype(np.int8)


def run_phase_parser_audit() -> dict:
    ensure_revision_dirs()
    assets = load_v6_assets()
    frame = assets.frame.reset_index(drop=True)
    with np.load(PHASE_CACHE, allow_pickle=False) as payload:
        predicted = payload["predictions"].astype(np.int8)
    target = _ground_truth_labels(frame, predicted.shape[1])
    rows = []
    per_video = []
    for role in ("validation", "official_test"):
        mask = frame.analysis_role.to_numpy() == role
        truth_flat = target[mask].reshape(-1)
        pred_flat = predicted[mask].reshape(-1)
        boundary_error = {"takeoff_to_flight": [], "flight_to_entry": []}
        projected_frame_fraction = []
        normalized_token_error = []
        for index in np.flatnonzero(mask):
            p1, p2 = _transition_positions(predicted[index])
            sample_positions = np.linspace(0, max(int(frame.loc[index, "frame_count"]) - 1, 0), predicted.shape[1])
            predicted_frames = (float(sample_positions[p1]), float(sample_positions[p2]))
            true_frames = _annotation_boundaries(
                int(frame.loc[index, "frame_count"]), str(frame.loc[index, "transition_frames_json"])
            )
            errors = np.abs(np.asarray(predicted_frames) - np.asarray(true_frames, dtype=float))
            target_positions = _transition_positions(target[index])
            token_errors = np.abs(np.asarray((p1, p2), dtype=float) - np.asarray(target_positions, dtype=float))
            boundary_error["takeoff_to_flight"].append(float(errors[0]))
            boundary_error["flight_to_entry"].append(float(errors[1]))
            projected_frame_fraction.append(
                float(errors.mean() / max(int(frame.loc[index, "frame_count"]), 1))
            )
            normalized_token_error.append(float(token_errors.mean() / predicted.shape[1]))
            per_video.append(
                {
                    "clip_uid": frame.loc[index, "clip_uid"],
                    "analysis_role": role,
                    "event_family": frame.loc[index, "event_family"],
                    "token_accuracy": float(np.mean(predicted[index] == target[index])),
                    "takeoff_to_flight_error_frames": float(errors[0]),
                    "flight_to_entry_error_frames": float(errors[1]),
                    "mean_normalized_token_boundary_error": normalized_token_error[-1],
                    "mean_projected_frame_boundary_error_fraction": projected_frame_fraction[-1],
                }
            )
        matrix = confusion_matrix(truth_flat, pred_flat, labels=[0, 1, 2])
        rows.append(
            {
                "analysis_role": role,
                "videos": int(mask.sum()),
                "temporal_tokens_per_video": int(predicted.shape[1]),
                "token_accuracy": float(np.mean(pred_flat == truth_flat)),
                "macro_f1": float(f1_score(truth_flat, pred_flat, labels=[0, 1, 2], average="macro")),
                "takeoff_to_flight_boundary_mae_frames": float(np.mean(boundary_error["takeoff_to_flight"])),
                "flight_to_entry_boundary_mae_frames": float(np.mean(boundary_error["flight_to_entry"])),
                "mean_normalized_token_boundary_error": float(np.mean(normalized_token_error)),
                "mean_projected_frame_boundary_error_fraction": float(np.mean(projected_frame_fraction)),
                "confusion_matrix": json.dumps(matrix.tolist(), separators=(",", ":")),
            }
        )
    metrics = pd.DataFrame(rows)
    per_video_frame = pd.DataFrame(per_video)
    metrics_path = REVISION_ROOT / "02_PHASE_PARSER" / "phase_parser_metrics.csv"
    per_video_path = REVISION_ROOT / "02_PHASE_PARSER" / "phase_parser_per_video.parquet"
    metrics.to_csv(metrics_path, index=False)
    per_video_frame.to_parquet(per_video_path, index=False)
    result = {
        "status": "COMPLETE",
        "metric_unit": "eight sampled temporal tokens per video",
        "original_frame_boundary_errors_are_linear_projections": True,
        "metrics_sha256": sha256_file(metrics_path),
        "per_video_sha256": sha256_file(per_video_path),
    }
    write_json(REVISION_ROOT / "02_PHASE_PARSER" / "phase_parser_summary.json", result)
    return result


def _pairwise_interactions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty((len(values), 3), dtype=float)
    pairs = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
    for column, (left, right, other) in enumerate(pairs):
        base = values[:, (1 << left) | (1 << right)] - values[:, 1 << left] - values[:, 1 << right] + values[:, 0]
        conditional = (
            values[:, 7]
            - values[:, (1 << left) | (1 << other)]
            - values[:, (1 << right) | (1 << other)]
            + values[:, 1 << other]
        )
        output[:, column] = 0.5 * (base + conditional)
    return output


def _cluster_bootstrap_median(
    values: np.ndarray, groups: np.ndarray, iterations: int = 10_000, seed: int = 20260903
) -> tuple[float, float]:
    unique = np.asarray(sorted(np.unique(groups.astype(str))), dtype=object)
    members = {group: np.flatnonzero(groups.astype(str) == group) for group in unique}
    rng = np.random.default_rng(seed)
    output = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([members[group] for group in sampled])
        output[iteration] = float(np.median(values[index]))
    return float(np.quantile(output, 0.025)), float(np.quantile(output, 0.975))


def _cluster_bootstrap_spearman(
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    iterations: int = 10_000,
    seed: int = 20260903,
) -> tuple[float, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    group_labels = groups.astype(str)
    unique = np.asarray(sorted(np.unique(group_labels)), dtype=object)
    members = {group: np.flatnonzero(group_labels == group) for group in unique}
    rng = np.random.default_rng(seed)
    output = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([members[group] for group in sampled])
        output[iteration] = float(spearmanr(left[index], right[index]).statistic)
    finite = output[np.isfinite(output)]
    if len(finite) != iterations:
        raise RuntimeError("Clustered Spearman bootstrap produced non-finite draws")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def run_shapley_audit() -> dict:
    ensure_revision_dirs()
    evidence = pd.read_parquet(V7_EVIDENCE)
    test = evidence[(evidence.analysis_role == "official_test") & (~evidence.open_set.astype(bool))].reset_index(drop=True)
    coalition = test[[f"coalition_{mask}" for mask in range(8)]].to_numpy(dtype=float)
    phi = test[[f"phi_{phase}" for phase in PHASES]].to_numpy(dtype=float)
    occlusion = test[[f"occlusion_{phase}" for phase in PHASES]].to_numpy(dtype=float)
    top = np.argmax(np.abs(phi), axis=1)
    abs_occ = np.abs(occlusion)
    targeted = abs_occ[np.arange(len(test)), top]
    nonselected = np.asarray([[abs_occ[row, phase] for phase in range(3) if phase != top[row]] for row in range(len(test))])
    mean_nonselected = nonselected.mean(axis=1)
    strongest_nonselected = nonselected.max(axis=1)
    interactions = _pairwise_interactions(coalition)
    third_order = (
        coalition[:, 7] - coalition[:, 3] - coalition[:, 5] - coalition[:, 6]
        + coalition[:, 1] + coalition[:, 2] + coalition[:, 4] - coalition[:, 0]
    )
    loo_reconstruction_error = np.abs(occlusion.sum(axis=1) - (coalition[:, 7] - coalition[:, 0]))
    sign_match = np.sign(phi) == np.sign(occlusion)
    groups = test.event_family.astype(str).to_numpy()
    comparisons = {
        "targeted_minus_mean_nonselected": targeted - mean_nonselected,
        "targeted_minus_strongest_nonselected": targeted - strongest_nonselected,
    }
    summary_comparisons = {}
    for offset, (name, values) in enumerate(comparisons.items()):
        low, high = _cluster_bootstrap_median(values, groups, seed=20260903 + offset)
        summary_comparisons[name] = {
            "median": float(np.median(values)), "cluster_ci": [low, high]
        }
    output = test[["clip_uid", "event_family", "top_phase", "actual_max_intervention_phase"]].copy()
    output["targeted_effect"] = targeted
    output["mean_nonselected_effect"] = mean_nonselected
    output["strongest_nonselected_effect"] = strongest_nonselected
    output["loo_reconstruction_error"] = loo_reconstruction_error
    output["interaction_takeoff_flight"] = interactions[:, 0]
    output["interaction_takeoff_entry"] = interactions[:, 1]
    output["interaction_flight_entry"] = interactions[:, 2]
    output["third_order_interaction"] = third_order
    for index, phase in enumerate(PHASES):
        output[f"sign_match_{phase}"] = sign_match[:, index]
    output_path = REVISION_ROOT / "03_SHAPLEY_AUDIT" / "shapley_additional_metrics.parquet"
    output.to_parquet(output_path, index=False)
    result = {
        "status": "COMPLETE",
        "test_closed_set_n": int(len(test)),
        "top_phase_match": float(np.mean(test.top_phase == test.actual_max_intervention_phase)),
        "deterministic_intervention_comparisons": summary_comparisons,
        "sign_consistency_overall": float(sign_match.mean()),
        "sign_consistency_by_phase": {
            phase: float(sign_match[:, index].mean()) for index, phase in enumerate(PHASES)
        },
        "median_absolute_pairwise_interaction": {
            "takeoff_flight": float(np.median(np.abs(interactions[:, 0]))),
            "takeoff_entry": float(np.median(np.abs(interactions[:, 1]))),
            "flight_entry": float(np.median(np.abs(interactions[:, 2]))),
        },
        "median_absolute_third_order_interaction": float(np.median(np.abs(third_order))),
        "median_loo_reconstruction_error": float(np.median(loo_reconstruction_error)),
        "fraction_loo_error_above_0_05": float(np.mean(loo_reconstruction_error > 0.05)),
        "mismatch_n": int(np.sum(test.top_phase != test.actual_max_intervention_phase)),
        "output_sha256": sha256_file(output_path),
    }
    write_json(REVISION_ROOT / "03_SHAPLEY_AUDIT" / "shapley_audit_summary.json", result)
    return result


def run_oracle_attribution() -> dict:
    ensure_revision_dirs()
    started = time.perf_counter()
    assets = load_v6_assets()
    mapping = load_reference_map_v7(final=True)
    artifact = load_final_artifact_v7()
    sequences, actions = _load_sequences()
    oracle_labels = _ground_truth_labels(assets.frame, sequences.shape[1])
    predictions = pd.read_parquet(V7_PREDICTIONS).set_index("clip_uid").loc[assets.frame.clip_uid].reset_index()
    existing = pd.read_parquet(V7_EVIDENCE).set_index("clip_uid").loc[assets.frame.clip_uid].reset_index()
    test_mask = assets.frame.analysis_role.to_numpy() == "official_test"
    closed = ~mapping["open_set"].astype(bool)
    indices = np.flatnonzero(test_mask & closed)
    coalition = np.full((len(indices), 8), np.nan, dtype=np.float32)
    chunk_size = 100
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start : start + chunk_size]
        values = _coalitions(
            chunk, "original", assets, mapping, artifact, sequences, actions, oracle_labels
        )
        values[:, 7] = predictions.trustdive_predicted_quality.to_numpy(dtype=np.float32)[chunk]
        coalition[start : start + len(chunk)] = values
        print(f"Oracle attribution: {min(start + len(chunk), len(indices))}/{len(indices)}", flush=True)
    phi = exact_three_phase_shapley(coalition[:, None, :])[:, 0, :]
    occlusion = np.column_stack(
        [coalition[:, 7] - coalition[:, 7 ^ (1 << phase)] for phase in range(3)]
    )
    oracle_top = np.argmax(np.abs(phi), axis=1)
    oracle_intervention_top = np.argmax(np.abs(occlusion), axis=1)
    predicted_phi = existing.loc[indices, [f"phi_{phase}" for phase in PHASES]].to_numpy(dtype=float)
    predicted_top = np.argmax(np.abs(predicted_phi), axis=1)
    contribution_cosine = _cosine_rows(predicted_phi, phi)
    targeted = np.abs(occlusion[np.arange(len(output := assets.frame.loc[indices, ["clip_uid", "event_family", "action_type"]].reset_index(drop=True))), oracle_top])
    nonselected = np.asarray(
        [
            [abs(occlusion[row, phase]) for phase in range(3) if phase != oracle_top[row]]
            for row in range(len(oracle_top))
        ]
    )
    groups = output.event_family.astype(str).to_numpy()
    oracle_mean_delta = targeted - nonselected.mean(axis=1)
    oracle_strongest_delta = targeted - nonselected.max(axis=1)
    mean_ci = _cluster_bootstrap_median(oracle_mean_delta, groups, seed=20260913)
    strongest_ci = _cluster_bootstrap_median(oracle_strongest_delta, groups, seed=20260914)
    for phase_index, phase in enumerate(PHASES):
        output[f"oracle_phi_{phase}"] = phi[:, phase_index]
        output[f"oracle_occlusion_{phase}"] = occlusion[:, phase_index]
    output["predicted_oracle_contribution_cosine"] = contribution_cosine
    output["predicted_oracle_top_phase_match"] = predicted_top == oracle_top
    output["oracle_top_intervention_match"] = oracle_top == oracle_intervention_top
    output["oracle_reconstruction_error"] = np.abs(coalition[:, 0] + phi.sum(axis=1) - coalition[:, 7])
    for mask in range(8):
        output[f"oracle_coalition_{mask}"] = coalition[:, mask]
    parser_video = pd.read_parquet(
        REVISION_ROOT / "02_PHASE_PARSER" / "phase_parser_per_video.parquet"
    ).set_index("clip_uid")
    output["mean_normalized_token_boundary_error"] = parser_video.loc[
        output.clip_uid, "mean_normalized_token_boundary_error"
    ].to_numpy(dtype=float)
    output["attribution_oracle_disagreement"] = 1.0 - contribution_cosine
    correlation = spearmanr(
        output.mean_normalized_token_boundary_error, output.attribution_oracle_disagreement
    )
    correlation_ci = _cluster_bootstrap_spearman(
        output.mean_normalized_token_boundary_error.to_numpy(dtype=float),
        output.attribution_oracle_disagreement.to_numpy(dtype=float),
        groups,
        seed=20260915,
    )
    path = REVISION_ROOT / "04_ORACLE_ATTRIBUTION" / "oracle_phase_evidence.parquet"
    output.to_parquet(path, index=False)
    result = {
        "status": "COMPLETE",
        "test_closed_set_n": int(len(output)),
        "maximum_reconstruction_error": float(output.oracle_reconstruction_error.max()),
        "predicted_oracle_contribution_cosine_median": float(np.median(contribution_cosine)),
        "predicted_oracle_top_phase_match": float(output.predicted_oracle_top_phase_match.mean()),
        "oracle_top_intervention_match": float(output.oracle_top_intervention_match.mean()),
        "oracle_targeted_minus_mean_nonselected": {
            "median": float(np.median(oracle_mean_delta)), "cluster_ci": list(mean_ci)
        },
        "oracle_targeted_minus_strongest_nonselected": {
            "median": float(np.median(oracle_strongest_delta)), "cluster_ci": list(strongest_ci)
        },
        "oracle_top_phase_share": {
            phase: float(np.mean(oracle_top == index)) for index, phase in enumerate(PHASES)
        },
        "boundary_error_vs_attribution_disagreement_spearman": float(correlation.statistic),
        "boundary_error_vs_attribution_disagreement_cluster_ci": list(correlation_ci),
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_sha256": sha256_file(path),
    }
    write_json(REVISION_ROOT / "04_ORACLE_ATTRIBUTION" / "oracle_attribution_summary.json", result)
    return result


def _reference_map_k(assets, base: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    refs = base["references"][:, :k].astype(int)
    distances = base["distances"][:, :k].astype(float)
    valid = refs >= 0
    safe = np.maximum(refs, 0)
    available = distances[np.isfinite(distances)]
    temperature = max(float(np.median(available)), 1e-3)
    q_true = assets.frame.execution_quality.to_numpy(dtype=float)
    q_teacher = assets.teacher_quality
    weights = np.zeros_like(distances, dtype=np.float32)
    for query in range(len(refs)):
        keep = valid[query]
        if not np.any(keep):
            continue
        selected = safe[query, keep]
        dispersion = float(np.std(q_true[selected], ddof=0))
        reliability = 1.0 / (1.0 + np.abs(q_true[selected] - q_teacher[selected]) + dispersion)
        raw = np.exp(-distances[query, keep] / temperature) * reliability
        weights[query, keep] = raw / max(float(raw.sum()), 1e-8)
    return {
        "references": refs,
        "distances": distances,
        "weights": weights,
        "valid_reference_count": valid.sum(axis=1).astype(np.int16),
        "open_set": valid.sum(axis=1) < 3,
        "temperature": np.asarray([temperature], dtype=np.float32),
    }


def _reference_statistics_k(assets, mapping: dict[str, np.ndarray]) -> np.ndarray:
    refs = mapping["references"].astype(int)
    safe = np.maximum(refs, 0)
    valid = refs >= 0
    weights = np.where(valid, mapping["weights"], 0.0).astype(float)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    q_true = assets.frame.execution_quality.to_numpy(dtype=float)[safe]
    q_teacher = assets.teacher_quality[safe]
    distance = np.nan_to_num(mapping["distances"], nan=2.0)
    mean_true = np.sum(weights * q_true, axis=1, keepdims=True)
    return np.column_stack(
        (
            np.sum(weights * q_true, axis=1),
            np.sum(weights * q_teacher, axis=1),
            np.sum(weights * (q_true - q_teacher), axis=1),
            np.sum(weights * distance, axis=1),
            np.sqrt(np.sum(weights * (q_true - mean_true) ** 2, axis=1)),
        )
    ).astype(np.float32)


def _features_k(assets, mapping: dict[str, np.ndarray], latent=None, teacher=None, query=None) -> np.ndarray:
    global_values = assets.global_latent if latent is None else np.asarray(latent, dtype=np.float32)
    teacher_values = assets.teacher_quality if teacher is None else np.asarray(teacher, dtype=np.float32)
    stats = _reference_statistics_k(assets, mapping)
    if query is not None:
        stats = stats[np.asarray(query, dtype=int)]
    return np.concatenate((global_values, teacher_values[:, None], stats), axis=1).astype(np.float32)


def _coalitions_k(indices, assets, mapping, artifact, sequences, actions, labels) -> np.ndarray:
    refs = mapping["references"].astype(int)
    hybrids: list[np.ndarray] = []
    hybrid_actions: list[np.ndarray] = []
    metadata: list[tuple[int, int, int, int]] = []
    for local, query in enumerate(indices):
        for slot, ref in enumerate(refs[query]):
            if ref < 0:
                continue
            for mask in range(8):
                hybrids.append(
                    hybrid_sequence(
                        sequences[query], sequences[ref], labels[query], labels[ref], mask
                    )
                )
                hybrid_actions.append(actions[query])
                metadata.append((local, query, slot, mask))
    teacher, latent = _predict_teacher_latent(np.stack(hybrids), np.stack(hybrid_actions))
    query_vector = np.asarray([query for _, query, _, _ in metadata], dtype=int)
    features = _features_k(assets, mapping, latent, teacher, query_vector)
    components = predict_ridge_artifact(artifact, features, teacher)
    values = np.full((len(indices), refs.shape[1], 8), np.nan, dtype=np.float32)
    for value, (local, _query, slot, mask) in zip(components, metadata):
        values[local, slot, mask] = value
    weights = mapping["weights"][indices].astype(float)
    weights = np.where(np.isfinite(values[..., 0]), weights, 0.0)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    return np.nansum(values * weights[:, :, None], axis=1).astype(np.float32)


def run_efficiency_sensitivity() -> dict:
    ensure_revision_dirs()
    assets = load_v6_assets()
    frame = assets.frame.reset_index(drop=True)
    base_dev = load_reference_map_v7(final=False)
    base_final = load_reference_map_v7(final=True)
    residual = frame.execution_quality.to_numpy(dtype=float) - assets.teacher_quality
    test = np.flatnonzero(frame.analysis_role.to_numpy() == "official_test")
    score_scale = 3.0 * frame.difficulty.to_numpy(dtype=float)
    rows = []
    artifacts = {}
    maps = {}
    frozen_predictions = pd.read_parquet(V7_PREDICTIONS).set_index("clip_uid").loc[frame.clip_uid]
    for k in (3, 5, 10):
        dev_map = _reference_map_k(assets, base_dev, k)
        final_map = _reference_map_k(assets, base_final, k)
        dev_features = _features_k(assets, dev_map)
        final_features = _features_k(assets, final_map)
        if k == 5:
            selection = {"selected_alpha": 10.0}
            quality = (
                frozen_predictions.plain_ridge_predicted_score.to_numpy(dtype=float)
                / score_scale
            )
            artifact = joblib.load(RUNS_ROOT / "v7_risk_task" / "checkpoints" / "final_plain_ridge_v7.joblib")
        else:
            quality, selection = _select_ridge(dev_features, final_features, assets, f"ridge_k{k}")
            quality[final_map["open_set"]] = assets.teacher_quality[final_map["open_set"]]
            artifact = joblib.load(REVISION_RUN_ROOT / "checkpoints" / f"ridge_k{k}.joblib")
        metrics = aqa_score_metrics(frame.dive_score.to_numpy(dtype=float)[test], (score_scale * quality)[test])
        rows.append(
            {
                "reference_count": k,
                "selected_alpha": selection["selected_alpha"],
                "reference_sparse_test_n": int(final_map["open_set"][test].sum()),
                **metrics,
            }
        )
        maps[k] = final_map
        artifacts[k] = artifact
    metrics_frame = pd.DataFrame(rows)
    metrics_path = REVISION_ROOT / "05_EFFICIENCY" / "reference_count_sensitivity.csv"
    metrics_frame.to_csv(metrics_path, index=False)

    sequences, actions = _load_sequences()
    labels = _phase_labels(sequences.shape[1])
    eligible = test[np.sum(base_final["references"][:, :10] >= 0, axis=1)[test] >= 10]
    sample = eligible[np.linspace(0, len(eligible) - 1, min(100, len(eligible))).round().astype(int)]
    phi_by_k = {}
    timing_rows = []
    # Warm model and file-system caches before timed runs. Timing includes the
    # complete batched coalition call and is summarized by the median of three
    # repeats, so the first K is not penalized by one-time TorchScript loading.
    _coalitions_k(sample[:10], assets, maps[5], artifacts[5], sequences, actions, labels)
    for k in (3, 5, 10):
        elapsed_repeats = []
        coalition = None
        for _ in range(3):
            started = time.perf_counter()
            coalition = _coalitions_k(sample, assets, maps[k], artifacts[k], sequences, actions, labels)
            elapsed_repeats.append(time.perf_counter() - started)
        elapsed = float(np.median(elapsed_repeats))
        phi_by_k[k] = exact_three_phase_shapley(coalition[:, None, :])[:, 0, :]
        timing_rows.append(
            {
                "reference_count": k,
                "benchmark_videos": int(len(sample)),
                "hybrid_evaluations_per_video": int(8 * k),
                "elapsed_seconds": float(elapsed),
                "milliseconds_per_video_batched": float(1000.0 * elapsed / max(len(sample), 1)),
                "repeat_seconds": json.dumps(elapsed_repeats, separators=(",", ":")),
            }
        )
    timing = pd.DataFrame(timing_rows)
    timing_path = REVISION_ROOT / "05_EFFICIENCY" / "attribution_runtime.csv"
    timing.to_csv(timing_path, index=False)
    stability_rows = []
    base_phi = phi_by_k[5]
    base_top = np.argmax(np.abs(base_phi), axis=1)
    for k in (3, 10):
        stability_rows.append(
            {
                "comparison": f"K={k} vs K=5",
                "contribution_cosine_median": float(np.median(_cosine_rows(base_phi, phi_by_k[k]))),
                "top_phase_agreement": float(
                    np.mean(base_top == np.argmax(np.abs(phi_by_k[k]), axis=1))
                ),
                "n": int(len(sample)),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability_path = REVISION_ROOT / "05_EFFICIENCY" / "reference_count_attribution_stability.csv"
    stability.to_csv(stability_path, index=False)
    result = {
        "status": "COMPLETE",
        "benchmark_sample_n": int(len(sample)),
        "benchmark_selection": "deterministic evenly spaced sample among test videos with at least ten legal references",
        "metrics_sha256": sha256_file(metrics_path),
        "runtime_sha256": sha256_file(timing_path),
        "stability_sha256": sha256_file(stability_path),
    }
    write_json(REVISION_ROOT / "05_EFFICIENCY" / "efficiency_summary.json", result)
    return result


def build_manifest() -> dict:
    ensure_revision_dirs()
    manifest_path = REVISION_ROOT / "run_manifest.json"
    output_files = sorted(
        path for path in REVISION_ROOT.rglob("*")
        if path.is_file() and path != manifest_path
    )
    result = {
        "status": "COMPLETE",
        "git_head": git_head(PROJECT_ROOT),
        "contract": str(CONTRACT_PATH.relative_to(PROJECT_ROOT)),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "outputs": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in output_files
        },
        "historical_v7_outputs_overwritten": False,
    }
    write_json(manifest_path, result)
    return result


def run_all(include_oracle: bool = True) -> dict:
    result = {
        "component_ablation": run_component_ablation(),
        "phase_parser": run_phase_parser_audit(),
        "shapley_audit": run_shapley_audit(),
    }
    if include_oracle:
        result["oracle_attribution"] = run_oracle_attribution()
    result["efficiency_sensitivity"] = run_efficiency_sensitivity()
    result["manifest"] = build_manifest()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reviewer-requested TrustDive manuscript analyses")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("component-ablation")
    commands.add_parser("phase-parser")
    commands.add_parser("shapley-audit")
    commands.add_parser("oracle-attribution")
    commands.add_parser("efficiency-sensitivity")
    run = commands.add_parser("run-all")
    run.add_argument("--skip-oracle", action="store_true")
    commands.add_parser("manifest")
    args = parser.parse_args(argv)
    if args.command == "component-ablation":
        result = run_component_ablation()
    elif args.command == "phase-parser":
        result = run_phase_parser_audit()
    elif args.command == "shapley-audit":
        result = run_shapley_audit()
    elif args.command == "oracle-attribution":
        result = run_oracle_attribution()
    elif args.command == "efficiency-sensitivity":
        result = run_efficiency_sensitivity()
    elif args.command == "manifest":
        result = build_manifest()
    else:
        result = run_all(include_oracle=not args.skip_oracle)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
