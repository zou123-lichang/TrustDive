from __future__ import annotations

import json
import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .util import sha256_file, write_json
from .v4_data import V4_RESULTS_ROOT, load_v4_contract, load_v4_frame


PHASES = ("takeoff", "flight", "entry")


def exact_three_phase_shapley(coalition_values: np.ndarray) -> np.ndarray:
    values = np.asarray(coalition_values, dtype=float)
    if values.shape[-1] != 8:
        raise ValueError("Three-phase Shapley requires exactly eight coalition values")
    output = np.zeros(values.shape[:-1] + (3,), dtype=float)
    players = range(3)
    for player in players:
        others = [item for item in players if item != player]
        for size in range(3):
            for subset in combinations(others, size):
                mask = sum(1 << item for item in subset)
                with_player = mask | (1 << player)
                weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
                output[..., player] += weight * (values[..., with_player] - values[..., mask])
    return output


def shapley_efficiency_error(coalition_values: np.ndarray, contributions: np.ndarray) -> np.ndarray:
    values = np.asarray(coalition_values, dtype=float)
    phi = np.asarray(contributions, dtype=float)
    return np.abs(phi.sum(axis=-1) - (values[..., 7] - values[..., 0]))


def resample_phase_labels(labels: np.ndarray, target_length: int = 9) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or len(labels) < 3:
        raise ValueError("Phase labels must be a one-dimensional sequence")
    first = int(np.flatnonzero(labels == 1)[0]) if np.any(labels == 1) else max(1, len(labels) // 3)
    second = int(np.flatnonzero(labels == 2)[0]) if np.any(labels == 2) else max(first + 1, 2 * len(labels) // 3)
    first_target = int(np.clip(round(first / len(labels) * target_length), 1, target_length - 2))
    second_target = int(np.clip(round(second / len(labels) * target_length), first_target + 1, target_length - 1))
    output = np.zeros(target_length, dtype=np.int8)
    output[first_target:second_target] = 1
    output[second_target:] = 2
    return output


def _resample_tokens(tokens: np.ndarray, target_count: int) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.float32)
    if target_count <= 0:
        return np.empty((0, tokens.shape[-1]), dtype=np.float32)
    if len(tokens) == 0:
        return np.zeros((target_count, tokens.shape[-1]), dtype=np.float32)
    if len(tokens) == target_count:
        return tokens.copy()
    positions = np.linspace(0.0, len(tokens) - 1, target_count)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    weight = (positions - lower)[:, None].astype(np.float32)
    return ((1.0 - weight) * tokens[lower] + weight * tokens[upper]).astype(np.float32)


def hybrid_sequence(
    query: np.ndarray,
    reference: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    coalition_mask: int,
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    output = query.copy()
    for phase in range(3):
        positions = np.flatnonzero(query_labels == phase)
        if coalition_mask & (1 << phase):
            output[positions] = query[positions]
        else:
            donor = reference[reference_labels == phase]
            output[positions] = _resample_tokens(donor, len(positions))
    return output


def _load_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    prediction_path = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_predictions_v4.parquet"
    sequence_path = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_sequences_v4.npz"
    model_path = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_counterfactual_v4.pt"
    if not prediction_path.exists() or not sequence_path.exists() or not model_path.exists():
        raise RuntimeError("Run export-teacher --protocol v4 before counterfactual generation")
    frame = load_v4_frame()
    teacher = pd.read_parquet(prediction_path).set_index("clip_uid").loc[frame.clip_uid]
    with np.load(sequence_path, allow_pickle=False) as payload:
        clip_uid = payload["clip_uid"].astype(str)
        if not np.array_equal(clip_uid, frame.clip_uid.to_numpy(dtype=str)):
            raise AssertionError("Teacher sequence order does not match the v4 frame")
        sequences = payload["sequence"].astype(np.float32)
        actions = payload["action_presence"].astype(np.float32)
    phase_cache = V4_RESULTS_ROOT.parent / "V2_DISAGREEMENT" / "01_FEATURES" / "phase_predictions_videomae_official_v2.npz"
    with np.load(phase_cache, allow_pickle=False) as payload:
        original = payload["predictions"].astype(np.int8)
    labels = np.stack([resample_phase_labels(item, sequences.shape[1]) for item in original])
    return teacher.reset_index(drop=True), sequences, actions, labels


def _select_references(frame: pd.DataFrame, sequences: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fit = np.flatnonzero(frame.analysis_role.to_numpy() == "fit")
    actions = frame.action_type.astype(str).to_numpy()
    families = frame.event_family.astype(str).to_numpy()
    global_features = sequences.mean(axis=1)
    norms = np.linalg.norm(global_features, axis=1, keepdims=True)
    normalized = global_features / np.maximum(norms, 1e-8)
    refs = np.empty((len(frame), count), dtype=np.int32)
    valid_count = np.zeros(len(frame), dtype=np.int16)
    mean_distance = np.zeros(len(frame), dtype=np.float32)
    dispersion = np.zeros(len(frame), dtype=np.float32)
    qualities = frame.execution_quality.to_numpy(dtype=float)
    for index in range(len(frame)):
        candidates = fit[(actions[fit] == actions[index]) & (families[fit] != families[index]) & (fit != index)]
        distances = 1.0 - normalized[candidates] @ normalized[index] if len(candidates) else np.array([])
        order = candidates[np.argsort(distances)[:count]] if len(candidates) else np.array([], dtype=int)
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


def _predict_torchscript(sequences: np.ndarray, actions: np.ndarray, batch_size: int = 512) -> np.ndarray:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_counterfactual_v4.pt"), map_location=device).eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            x = torch.from_numpy(np.asarray(sequences[start : start + batch_size], dtype=np.float32)).to(device)
            a = torch.from_numpy(np.asarray(actions[start : start + batch_size], dtype=np.float32)).to(device)
            outputs.append(model(x, a).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def build_counterfactual_targets_v4() -> dict:
    gate = json.loads((V4_RESULTS_ROOT / "01_TEACHER" / "teacher_gate_v4.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("Teacher gate must pass before counterfactual generation")
    contract = load_v4_contract()
    frame = load_v4_frame()
    teacher, sequences, actions, phase_labels = _load_inputs()
    refs, valid_count, reference_distance, reference_dispersion = _select_references(
        frame, sequences, int(contract["data"]["reference_count"])
    )

    fit = frame.analysis_role.to_numpy() == "fit"
    global_features = sequences.mean(axis=1)
    center = global_features[fit].mean(axis=0)
    scale = global_features[fit].std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    real_ood = np.sqrt(np.mean(((global_features[fit] - center) / scale) ** 2, axis=1))
    ood_threshold = float(np.quantile(real_ood, float(contract["counterfactual"]["hybrid_ood_quantile"])))

    coalition_values = np.empty((len(frame), refs.shape[1], 8), dtype=np.float32)
    hybrid_ood = np.empty_like(coalition_values, dtype=np.float32)
    chunk = 48
    for start in range(0, len(frame), chunk):
        stop = min(start + chunk, len(frame))
        hybrids = []
        hybrid_actions = []
        metadata = []
        for query_index in range(start, stop):
            for ref_slot, ref_index in enumerate(refs[query_index]):
                for mask in range(8):
                    hybrid = hybrid_sequence(
                        sequences[query_index], sequences[ref_index], phase_labels[query_index], phase_labels[ref_index], mask
                    )
                    hybrids.append(hybrid)
                    hybrid_actions.append(actions[query_index])
                    metadata.append((query_index, ref_slot, mask))
                    score = np.sqrt(np.mean(((hybrid.mean(axis=0) - center) / scale) ** 2))
                    hybrid_ood[query_index, ref_slot, mask] = float(score)
        predictions = _predict_torchscript(np.stack(hybrids), np.stack(hybrid_actions))
        for value, (query_index, ref_slot, mask) in zip(predictions, metadata):
            coalition_values[query_index, ref_slot, mask] = value
        print(f"Counterfactual teacher scoring: {stop}/{len(frame)}", flush=True)

    contributions = exact_three_phase_shapley(coalition_values).astype(np.float32)
    efficiency = shapley_efficiency_error(coalition_values, contributions)
    rows = []
    for query_index in range(len(frame)):
        for ref_slot, ref_index in enumerate(refs[query_index]):
            row = {
                "clip_uid": frame.iloc[query_index].clip_uid,
                "reference_clip_uid": frame.iloc[int(ref_index)].clip_uid,
                "query_index": query_index,
                "reference_index": int(ref_index),
                "reference_slot": ref_slot,
                "valid_reference": bool(ref_slot < valid_count[query_index]),
                "efficiency_error": float(efficiency[query_index, ref_slot]),
                "hybrid_max_ood_score": float(hybrid_ood[query_index, ref_slot].max()),
            }
            for mask in range(8):
                row[f"coalition_{mask}"] = float(coalition_values[query_index, ref_slot, mask])
            for phase_index, phase in enumerate(PHASES):
                row[f"phi_{phase}"] = float(contributions[query_index, ref_slot, phase_index])
            rows.append(row)
    pair_frame = pd.DataFrame(rows)
    pair_frame.to_parquet(V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "counterfactual_targets_v4.parquet", index=False)

    aggregated = frame[["clip_uid", "feature_key", "official_split", "analysis_role", "source_role", "event_family", "action_type", "difficulty", "dive_score", "execution_quality"]].copy()
    aggregated["reference_indices_json"] = [json.dumps(item.tolist()) for item in refs]
    aggregated["valid_reference_count"] = valid_count
    aggregated["open_set"] = valid_count < int(contract["data"]["minimum_valid_references"])
    aggregated["reference_distance"] = reference_distance
    aggregated["reference_dispersion"] = reference_dispersion
    aggregated["base_quality"] = np.median(frame.execution_quality.to_numpy()[refs], axis=1)
    aggregated["teacher_predicted_quality"] = teacher.teacher_predicted_quality.to_numpy(dtype=float)
    for phase_index, phase in enumerate(PHASES):
        aggregated[f"teacher_phi_{phase}"] = np.median(contributions[:, :, phase_index], axis=1)
    aggregated.to_parquet(V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "counterfactual_query_targets_v4.parquet", index=False)
    np.savez_compressed(
        V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "reference_map_v4.npz",
        references=refs,
        valid_reference_count=valid_count,
        phase_labels=phase_labels,
    )

    full_query = coalition_values[:, :, 7].mean(axis=1)
    teacher_trace_difference = float(np.max(np.abs(full_query - teacher.teacher_predicted_quality.to_numpy(dtype=float))))
    ref_indices = refs.reshape(-1)
    reference_rho = float(
        spearmanr(
            teacher.teacher_predicted_quality.to_numpy(dtype=float)[ref_indices],
            frame.execution_quality.to_numpy(dtype=float)[ref_indices],
        ).statistic
    )
    checks = {
        "rows": len(pair_frame) == len(frame) * int(contract["data"]["reference_count"]),
        "finite": bool(np.isfinite(coalition_values).all() and np.isfinite(contributions).all()),
        "efficiency": float(efficiency.max()) <= float(contract["counterfactual"]["maximum_efficiency_error"]),
        "hybrid_ood": float(np.mean(hybrid_ood > ood_threshold)) <= float(contract["counterfactual"]["maximum_hybrid_ood_fraction"]),
        "reference_ranking": reference_rho > 0.0,
        "torchscript_matches_teacher": teacher_trace_difference <= 1e-4,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "STOP",
        "checks": checks,
        "maximum_efficiency_error": float(efficiency.max()),
        "hybrid_ood_threshold": ood_threshold,
        "hybrid_ood_fraction": float(np.mean(hybrid_ood > ood_threshold)),
        "reference_teacher_true_spearman": reference_rho,
        "maximum_torchscript_teacher_difference": teacher_trace_difference,
        "open_set_count": int((valid_count < int(contract["data"]["minimum_valid_references"])).sum()),
        "pair_target_sha256": sha256_file(V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "counterfactual_targets_v4.parquet"),
        "query_target_sha256": sha256_file(V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "counterfactual_query_targets_v4.parquet"),
    }
    write_json(V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "counterfactual_summary_v4.json", result)
    return result
