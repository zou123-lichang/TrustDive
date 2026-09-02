from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .util import sha256_file, write_json
from .v4_counterfactual import (
    PHASES,
    exact_three_phase_shapley,
    hybrid_sequence,
    resample_phase_labels,
    shapley_efficiency_error,
)
from .v5_data import V5_RESULTS_ROOT, load_v5_contract, load_v5_frame, require_v5_audit


def _load_teacher_assets() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    folder = V5_RESULTS_ROOT / "01_REFERENCES"
    summary = folder / "teacher_export_v5.json"
    if not summary.exists() or json.loads(summary.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("The frozen RICA2 teacher must be exported before reference construction")
    frame = load_v5_frame()
    teacher = pd.read_parquet(folder / "teacher_predictions_v5.parquet").set_index("clip_uid").loc[
        frame.clip_uid
    ]
    with np.load(folder / "teacher_sequences_v5.npz", allow_pickle=False) as payload:
        clip_uid = payload["clip_uid"].astype(str)
        if not np.array_equal(clip_uid, frame.clip_uid.to_numpy(dtype=str)):
            raise AssertionError("Teacher sequence order does not match the v5 frame")
        sequences = payload["sequence"].astype(np.float32)
        actions = payload["action_presence"].astype(np.float32)
    return teacher.reset_index(drop=True), sequences, actions


def _phase_labels(length: int) -> np.ndarray:
    phase_cache = (
        V5_RESULTS_ROOT.parent
        / "V2_DISAGREEMENT"
        / "01_FEATURES"
        / "phase_predictions_videomae_official_v2.npz"
    )
    with np.load(phase_cache, allow_pickle=False) as payload:
        original = payload["predictions"].astype(np.int8)
    if len(original) != len(load_v5_frame()):
        raise AssertionError("Predicted phase cache does not contain all 3000 videos")
    return np.stack([resample_phase_labels(item, length) for item in original])


def _select_references(
    frame: pd.DataFrame, sequences: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fit = np.flatnonzero(frame.analysis_role.to_numpy() == "fit")
    actions = frame.action_type.astype(str).to_numpy()
    families = frame.event_family.astype(str).to_numpy()
    global_features = sequences.mean(axis=1)
    normalized = global_features / np.maximum(
        np.linalg.norm(global_features, axis=1, keepdims=True), 1e-8
    )
    refs = np.empty((len(frame), count), dtype=np.int32)
    valid_count = np.zeros(len(frame), dtype=np.int16)
    mean_distance = np.zeros(len(frame), dtype=np.float32)
    dispersion = np.zeros(len(frame), dtype=np.float32)
    qualities = frame.execution_quality.to_numpy(dtype=float)
    for index in range(len(frame)):
        candidates = fit[
            (actions[fit] == actions[index])
            & (families[fit] != families[index])
            & (fit != index)
        ]
        distances = (
            1.0 - normalized[candidates] @ normalized[index]
            if len(candidates)
            else np.asarray([], dtype=float)
        )
        order = (
            candidates[np.argsort(distances)[:count]]
            if len(candidates)
            else np.asarray([], dtype=int)
        )
        valid_count[index] = len(order)
        if not len(order):
            fallback = fit[(actions[fit] == actions[index]) & (fit != index)]
            if not len(fallback):
                fallback = fit[fit != index]
            fallback_distance = 1.0 - normalized[fallback] @ normalized[index]
            order = fallback[np.argsort(fallback_distance)[:1]]
        selected = order.tolist()
        while len(selected) < count:
            selected.append(selected[len(selected) % len(order)])
        selected = selected[:count]
        refs[index] = selected
        ref_distances = 1.0 - normalized[selected] @ normalized[index]
        mean_distance[index] = float(np.mean(ref_distances))
        dispersion[index] = float(np.std(qualities[selected], ddof=0))
    return refs, valid_count, mean_distance, dispersion


def build_references_v5() -> dict:
    require_v5_audit()
    contract = load_v5_contract()
    frame = load_v5_frame()
    teacher, sequences, _ = _load_teacher_assets()
    labels = _phase_labels(sequences.shape[1])
    refs, valid_count, distance, dispersion = _select_references(
        frame, sequences, int(contract["data"]["reference_count"])
    )
    qualities = frame.execution_quality.to_numpy(dtype=np.float32)
    reference_path = V5_RESULTS_ROOT / "01_REFERENCES" / "reference_map_v5.npz"
    np.savez_compressed(
        reference_path,
        references=refs,
        valid_reference_count=valid_count,
        phase_labels=labels,
    )
    output = frame[
        [
            "clip_uid",
            "feature_key",
            "official_split",
            "analysis_role",
            "event_family",
            "action_type",
            "difficulty",
            "dive_score",
            "execution_quality",
        ]
    ].copy()
    output["reference_indices_json"] = [json.dumps(item.tolist()) for item in refs]
    output["valid_reference_count"] = valid_count
    output["open_set"] = valid_count < int(contract["data"]["minimum_valid_references"])
    output["reference_distance"] = distance
    output["reference_dispersion"] = dispersion
    output["base_quality"] = np.median(qualities[refs], axis=1)
    output["teacher_predicted_quality"] = teacher.teacher_predicted_quality.to_numpy(dtype=float)
    output_path = V5_RESULTS_ROOT / "01_REFERENCES" / "reference_targets_v5.parquet"
    output.to_parquet(output_path, index=False)
    fit = frame.analysis_role.to_numpy() == "fit"
    forbidden = np.any(frame.analysis_role.to_numpy()[refs] != "fit")
    same_action_including_fallback = np.all(
        frame.action_type.astype(str).to_numpy()[refs]
        == frame.action_type.astype(str).to_numpy()[:, None]
    )
    same_action_valid = True
    different_family_valid = True
    actions = frame.action_type.astype(str).to_numpy()
    families = frame.event_family.astype(str).to_numpy()
    for index in range(len(frame)):
        for slot in range(int(valid_count[index])):
            if actions[refs[index, slot]] != actions[index]:
                same_action_valid = False
            if families[refs[index, slot]] == families[index]:
                different_family_valid = False
    result = {
        "status": "PASS"
        if not forbidden
        and same_action_valid
        and different_family_valid
        and int(fit.sum()) == 1559
        else "FAIL",
        "rows": len(output),
        "fit_rows": int(fit.sum()),
        "all_references_from_fit": not bool(forbidden),
        "all_valid_references_same_action": bool(same_action_valid),
        "all_references_same_action_including_open_set_fallback": bool(
            same_action_including_fallback
        ),
        "open_set_fallback_note": (
            "Rows with no eligible same-action different-family fit reference use a "
            "nearest fit fallback only to keep tensor shapes defined; they remain "
            "open_set and are excluded from same-action robustness claims."
        ),
        "valid_references_different_family": different_family_valid,
        "open_set_count": int(output.open_set.sum()),
        "reference_map_sha256": sha256_file(reference_path),
        "reference_targets_sha256": sha256_file(output_path),
    }
    write_json(V5_RESULTS_ROOT / "01_REFERENCES" / "reference_summary_v5.json", result)
    return result


def _predict_torchscript(sequences: np.ndarray, actions: np.ndarray, batch_size: int = 512) -> np.ndarray:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(
        str(V5_RESULTS_ROOT / "01_REFERENCES" / "teacher_counterfactual_v5.pt"),
        map_location=device,
    ).eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            x = torch.from_numpy(
                np.asarray(sequences[start : start + batch_size], dtype=np.float32)
            ).to(device)
            a = torch.from_numpy(
                np.asarray(actions[start : start + batch_size], dtype=np.float32)
            ).to(device)
            outputs.append(model(x, a).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _ecdf_scaled(values: np.ndarray, reference: np.ndarray, clip_quantile: float) -> np.ndarray:
    cap = max(float(np.quantile(reference, clip_quantile)), 1e-8)
    return np.clip(values / cap, 0.0, 1.0)


def build_counterfactual_targets_v5() -> dict:
    contract = load_v5_contract()
    reference_summary = V5_RESULTS_ROOT / "01_REFERENCES" / "reference_summary_v5.json"
    if not reference_summary.exists() or json.loads(reference_summary.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Run build-references --protocol v5 first")
    frame = load_v5_frame()
    teacher, sequences, actions = _load_teacher_assets()
    with np.load(V5_RESULTS_ROOT / "01_REFERENCES" / "reference_map_v5.npz", allow_pickle=False) as payload:
        refs = payload["references"].astype(np.int32)
        valid_count = payload["valid_reference_count"].astype(np.int16)
        phase_labels = payload["phase_labels"].astype(np.int8)

    fit = frame.analysis_role.to_numpy() == "fit"
    global_features = sequences.mean(axis=1)
    center = global_features[fit].mean(axis=0)
    scale = global_features[fit].std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    real_ood = np.sqrt(np.mean(((global_features[fit] - center) / scale) ** 2, axis=1))
    ood_threshold = float(
        np.quantile(real_ood, float(contract["counterfactual"]["real_distribution_quantile"]))
    )

    coalition_values = np.empty((len(frame), refs.shape[1], 8), dtype=np.float32)
    hybrid_ood = np.empty_like(coalition_values, dtype=np.float32)
    chunk = 48
    for start in range(0, len(frame), chunk):
        stop = min(start + chunk, len(frame))
        hybrids: list[np.ndarray] = []
        hybrid_actions: list[np.ndarray] = []
        metadata: list[tuple[int, int, int]] = []
        for query_index in range(start, stop):
            for ref_slot, ref_index in enumerate(refs[query_index]):
                for mask in range(8):
                    hybrid = hybrid_sequence(
                        sequences[query_index],
                        sequences[ref_index],
                        phase_labels[query_index],
                        phase_labels[ref_index],
                        mask,
                    )
                    hybrids.append(hybrid)
                    hybrid_actions.append(actions[query_index])
                    metadata.append((query_index, ref_slot, mask))
                    hybrid_ood[query_index, ref_slot, mask] = float(
                        np.sqrt(np.mean(((hybrid.mean(axis=0) - center) / scale) ** 2))
                    )
        predictions = _predict_torchscript(np.stack(hybrids), np.stack(hybrid_actions))
        for value, (query_index, ref_slot, mask) in zip(predictions, metadata):
            coalition_values[query_index, ref_slot, mask] = value
        print(f"V5 counterfactual scoring: {stop}/{len(frame)}", flush=True)

    contributions = exact_three_phase_shapley(coalition_values).astype(np.float32)
    efficiency = shapley_efficiency_error(coalition_values, contributions)
    valid_mask = np.arange(refs.shape[1])[None, :] < valid_count[:, None]
    aggregated_phi = np.empty((len(frame), 3), dtype=np.float32)
    phi_mad = np.empty_like(aggregated_phi)
    for index in range(len(frame)):
        selected = contributions[index, valid_mask[index]]
        if not len(selected):
            selected = contributions[index]
        aggregated_phi[index] = np.median(selected, axis=0)
        phi_mad[index] = np.median(
            np.abs(selected - np.median(selected, axis=0, keepdims=True)), axis=0
        )
    query_ood = np.max(hybrid_ood, axis=(1, 2))
    fit_mad = phi_mad[fit].reshape(-1)
    mad_scaled = _ecdf_scaled(
        phi_mad,
        fit_mad,
        float(contract["counterfactual"]["reliability_reference_mad_percentile_clip"]),
    )
    ood_scaled = _ecdf_scaled(
        query_ood,
        query_ood[fit],
        float(contract["counterfactual"]["reliability_ood_percentile_clip"]),
    )
    valid_ratio = np.minimum(valid_count / float(refs.shape[1]), 1.0)[:, None]
    raw_reliability = valid_ratio * (1.0 - mad_scaled) * (1.0 - ood_scaled[:, None])
    floor = float(contract["counterfactual"]["reliability_floor"])
    reliability = floor + (1.0 - floor) * np.clip(raw_reliability, 0.0, 1.0)

    pair_rows = []
    for query_index in range(len(frame)):
        for ref_slot, ref_index in enumerate(refs[query_index]):
            row = {
                "clip_uid": frame.iloc[query_index].clip_uid,
                "reference_clip_uid": frame.iloc[int(ref_index)].clip_uid,
                "query_index": query_index,
                "reference_index": int(ref_index),
                "reference_slot": ref_slot,
                "valid_reference": bool(valid_mask[query_index, ref_slot]),
                "efficiency_error": float(efficiency[query_index, ref_slot]),
                "hybrid_max_ood_score": float(hybrid_ood[query_index, ref_slot].max()),
            }
            for mask in range(8):
                row[f"coalition_{mask}"] = float(coalition_values[query_index, ref_slot, mask])
            for phase_index, phase in enumerate(PHASES):
                row[f"phi_{phase}"] = float(contributions[query_index, ref_slot, phase_index])
            pair_rows.append(row)
    pair_frame = pd.DataFrame(pair_rows)
    pair_path = V5_RESULTS_ROOT / "03_COUNTERFACTUAL" / "counterfactual_targets_v5.parquet"
    pair_frame.to_parquet(pair_path, index=False)

    references = pd.read_parquet(
        V5_RESULTS_ROOT / "01_REFERENCES" / "reference_targets_v5.parquet"
    )
    query = references.copy()
    for phase_index, phase in enumerate(PHASES):
        query[f"teacher_phi_{phase}"] = aggregated_phi[:, phase_index]
        query[f"teacher_phi_{phase}_mad"] = phi_mad[:, phase_index]
        query[f"phase_reliability_{phase}"] = reliability[:, phase_index]
    query["hybrid_ood_score"] = query_ood
    query_path = V5_RESULTS_ROOT / "03_COUNTERFACTUAL" / "counterfactual_query_targets_v5.parquet"
    query.to_parquet(query_path, index=False)
    arrays_path = V5_RESULTS_ROOT / "03_COUNTERFACTUAL" / "counterfactual_arrays_v5.npz"
    np.savez_compressed(
        arrays_path,
        coalition_values=coalition_values,
        teacher_phi=aggregated_phi,
        phase_reliability=reliability,
        hybrid_ood=hybrid_ood,
    )

    full_query = coalition_values[:, :, 7].mean(axis=1)
    trace_difference = float(
        np.max(
            np.abs(full_query - teacher.teacher_predicted_quality.to_numpy(dtype=float))
        )
    )
    ref_indices = refs.reshape(-1)
    reference_rho = float(
        spearmanr(
            teacher.teacher_predicted_quality.to_numpy(dtype=float)[ref_indices],
            frame.execution_quality.to_numpy(dtype=float)[ref_indices],
        ).statistic
    )
    checks = {
        "rows": len(pair_frame) == len(frame) * int(contract["data"]["reference_count"]),
        "finite": bool(
            np.isfinite(coalition_values).all()
            and np.isfinite(contributions).all()
            and np.isfinite(reliability).all()
        ),
        "efficiency": float(efficiency.max())
        <= float(contract["counterfactual"]["maximum_efficiency_error"]),
        "reference_ranking": reference_rho > 0.0,
        "torchscript_matches_teacher": trace_difference <= 1e-4,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_efficiency_error": float(efficiency.max()),
        "hybrid_ood_threshold": ood_threshold,
        "hybrid_ood_fraction": float(np.mean(hybrid_ood > ood_threshold)),
        "reliability_median": float(np.median(reliability)),
        "reliability_minimum": float(np.min(reliability)),
        "reference_teacher_true_spearman": reference_rho,
        "maximum_torchscript_teacher_difference": trace_difference,
        "pair_target_sha256": sha256_file(pair_path),
        "query_target_sha256": sha256_file(query_path),
        "array_sha256": sha256_file(arrays_path),
    }
    write_json(V5_RESULTS_ROOT / "03_COUNTERFACTUAL" / "counterfactual_summary_v5.json", result)
    return result
