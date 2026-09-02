from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .util import sha256_file, write_json
from .v4_counterfactual import PHASES, exact_three_phase_shapley, hybrid_sequence
from .v5_counterfactual import _phase_labels
from .v6_attribution import _jitter_labels, _load_sequences, _predict_teacher_latent
from .v6_modeling import load_v6_assets
from .v7_data import V7_RESULTS_ROOT, load_v7_contract
from .v7_modeling import (
    load_final_artifact_v7,
    load_reference_map_v7,
    make_ridge_features_v7,
    predict_ridge_artifact,
)


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)
    return numerator / denominator


def _coalitions(
    indices: np.ndarray,
    variant: str,
    assets,
    mapping: dict[str, np.ndarray],
    artifact: dict,
    sequences: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    primary = int(load_v7_contract()["data"]["references"])
    refs = mapping["references"][:, :primary].astype(int)
    if variant == "reference_replace":
        alternate = mapping["references"][:, primary : 2 * primary].astype(int)
        refs = np.where(alternate >= 0, alternate, refs)
    hybrids: list[np.ndarray] = []
    hybrid_actions: list[np.ndarray] = []
    metadata: list[tuple[int, int, int, int]] = []
    for local, query in enumerate(indices):
        for slot, ref in enumerate(refs[query]):
            if ref < 0:
                continue
            q_labels, r_labels = labels[query], labels[ref]
            if variant == "boundary_left":
                q_labels, r_labels = _jitter_labels(q_labels, -1), _jitter_labels(r_labels, -1)
            elif variant == "boundary_right":
                q_labels, r_labels = _jitter_labels(q_labels, 1), _jitter_labels(r_labels, 1)
            for mask in range(8):
                hybrids.append(hybrid_sequence(sequences[query], sequences[ref], q_labels, r_labels, mask))
                hybrid_actions.append(actions[query])
                metadata.append((local, query, slot, mask))
    if not hybrids:
        return np.full((len(indices), 8), np.nan, dtype=np.float32)
    teacher, latent = _predict_teacher_latent(np.stack(hybrids), np.stack(hybrid_actions))
    query_vector = np.asarray([query for _, query, _, _ in metadata], dtype=int)
    features = make_ridge_features_v7(
        assets,
        mapping,
        global_latent=latent,
        teacher_quality=teacher,
        query_indices=query_vector,
    )
    component = predict_ridge_artifact(artifact, features, teacher)
    values = np.full((len(indices), primary, 8), np.nan, dtype=np.float32)
    for value, (local, _query, slot, mask) in zip(component, metadata):
        values[local, slot, mask] = value
    weights = mapping["weights"][indices, :primary].astype(float)
    valid = np.isfinite(values[..., 0])
    weights = np.where(valid, weights, 0.0)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    return np.nansum(values * weights[:, :, None], axis=1).astype(np.float32)


def build_phase_evidence_v7() -> dict:
    score_path = V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet"
    if not score_path.exists():
        raise RuntimeError("Run train --protocol v7 first")
    assets = load_v6_assets()
    contract = load_v7_contract()
    mapping = load_reference_map_v7(final=True)
    artifact = load_final_artifact_v7()
    sequences, actions = _load_sequences()
    labels = _phase_labels(sequences.shape[1])
    predictions = pd.read_parquet(score_path).set_index("clip_uid").loc[assets.frame.clip_uid].reset_index()
    all_indices = np.arange(len(assets.frame), dtype=int)
    closed_indices = all_indices[~mapping["open_set"].astype(bool)]
    test_mask = assets.frame.analysis_role.to_numpy() == "official_test"
    test_closed = all_indices[test_mask & (~mapping["open_set"].astype(bool))]

    coalition = np.full((len(assets.frame), 8), np.nan, dtype=np.float32)
    phi = np.full((len(assets.frame), 3), np.nan, dtype=np.float32)
    boundary_cosine = np.full(len(assets.frame), np.nan, dtype=np.float32)
    boundary_top_match = np.full(len(assets.frame), np.nan, dtype=np.float32)
    reference_cosine = np.full(len(assets.frame), np.nan, dtype=np.float32)
    chunk_size = 100
    for start in range(0, len(closed_indices), chunk_size):
        chunk = closed_indices[start : start + chunk_size]
        original = _coalitions(chunk, "original", assets, mapping, artifact, sequences, actions, labels)
        # Anchor the full-query endpoint to the exact persisted scorer output.
        # TorchScript counterfactual inference is float32 and can otherwise
        # differ from the cached latent path by about 1e-4 quality points.
        original[:, 7] = predictions.trustdive_predicted_quality.to_numpy(dtype=np.float32)[chunk]
        left = _coalitions(chunk, "boundary_left", assets, mapping, artifact, sequences, actions, labels)
        right = _coalitions(chunk, "boundary_right", assets, mapping, artifact, sequences, actions, labels)
        original_phi = exact_three_phase_shapley(original[:, None, :])[:, 0, :]
        left_phi = exact_three_phase_shapley(left[:, None, :])[:, 0, :]
        right_phi = exact_three_phase_shapley(right[:, None, :])[:, 0, :]
        coalition[chunk] = original
        phi[chunk] = original_phi
        cosine = np.column_stack((_cosine_rows(original_phi, left_phi), _cosine_rows(original_phi, right_phi)))
        boundary_cosine[chunk] = np.mean(cosine, axis=1)
        original_top = np.argmax(np.abs(original_phi), axis=1)
        matches = np.column_stack((
            original_top == np.argmax(np.abs(left_phi), axis=1),
            original_top == np.argmax(np.abs(right_phi), axis=1),
        ))
        boundary_top_match[chunk] = np.mean(matches, axis=1)
        print(f"V7 phase evidence: {min(start + len(chunk), len(closed_indices))}/{len(closed_indices)}", flush=True)
    for start in range(0, len(test_closed), chunk_size):
        chunk = test_closed[start : start + chunk_size]
        alternate = _coalitions(chunk, "reference_replace", assets, mapping, artifact, sequences, actions, labels)
        alternate_phi = exact_three_phase_shapley(alternate[:, None, :])[:, 0, :]
        reference_cosine[chunk] = _cosine_rows(phi[chunk], alternate_phi)

    closed = ~mapping["open_set"].astype(bool)
    closed_rows = np.flatnonzero(closed)
    reconstruction = np.abs(coalition[:, 0] + np.nansum(phi, axis=1) - coalition[:, 7])
    scorer_alignment = np.abs(coalition[:, 7] - predictions.trustdive_predicted_quality.to_numpy(dtype=float))
    top_phase_index = np.full(len(assets.frame), -1, dtype=int)
    top_phase_index[closed] = np.argmax(np.abs(phi[closed]), axis=1)
    occlusion = np.full((len(assets.frame), 3), np.nan, dtype=np.float32)
    for phase in range(3):
        occlusion[closed, phase] = coalition[closed, 7] - coalition[closed, 7 ^ (1 << phase)]
    actual_top = np.full(len(assets.frame), -1, dtype=int)
    actual_top[closed] = np.argmax(np.abs(occlusion[closed]), axis=1)
    top_match = np.zeros(len(assets.frame), dtype=bool)
    top_match[closed] = top_phase_index[closed] == actual_top[closed]
    rng = np.random.default_rng(int(contract["statistics"]["seed"]))
    random_phase = rng.integers(0, 3, size=len(assets.frame))
    targeted = np.full(len(assets.frame), np.nan, dtype=float)
    random_effect = np.full(len(assets.frame), np.nan, dtype=float)
    for index in closed_rows:
        targeted[index] = abs(occlusion[index, top_phase_index[index]])
        random_effect[index] = abs(occlusion[index, random_phase[index]])
    phase_lengths = np.column_stack([(labels == phase).sum(axis=1) for phase in range(3)]).astype(float)
    phase_lengths /= np.maximum(phase_lengths.sum(axis=1, keepdims=True), 1.0)
    duration_contribution = (coalition[:, 7] - coalition[:, 0])[:, None] * phase_lengths

    output = assets.frame[[
        "clip_uid", "official_split", "analysis_role", "event_family", "action_type", "difficulty", "dive_score"
    ]].copy()
    output["open_set"] = mapping["open_set"].astype(bool)
    output["reference_baseline"] = coalition[:, 0]
    output["predicted_quality"] = coalition[:, 7]
    for phase_index, phase_name in enumerate(PHASES):
        output[f"phi_{phase_name}"] = phi[:, phase_index]
        output[f"occlusion_{phase_name}"] = occlusion[:, phase_index]
        output[f"duration_contribution_{phase_name}"] = duration_contribution[:, phase_index]
    names = np.asarray(PHASES, dtype=object)
    output["top_phase"] = np.where(top_phase_index >= 0, names[np.maximum(top_phase_index, 0)], "open_set")
    output["actual_max_intervention_phase"] = np.where(actual_top >= 0, names[np.maximum(actual_top, 0)], "open_set")
    output["top_phase_intervention_match"] = top_match
    output["targeted_intervention_effect"] = targeted
    output["random_intervention_effect"] = random_effect
    output["reconstruction_error"] = reconstruction
    output["scorer_alignment_error"] = scorer_alignment
    output["phase_boundary_cosine"] = boundary_cosine
    output["phase_boundary_top_match"] = boundary_top_match
    output["reference_change_cosine"] = reference_cosine
    for mask in range(8):
        output[f"coalition_{mask}"] = coalition[:, mask]
    path = V7_RESULTS_ROOT / "04_PHASE_EVIDENCE" / "phase_evidence_v7.parquet"
    output.to_parquet(path, index=False)

    test_closed_mask = test_mask & closed
    top_share = np.bincount(top_phase_index[test_closed_mask], minlength=3) / max(int(test_closed_mask.sum()), 1)
    result = {
        "status": "PASS" if (
            float(np.nanmax(reconstruction[closed])) <= float(contract["attribution"]["maximum_reconstruction_error"])
            and float(np.nanmax(scorer_alignment[closed])) <= float(contract["attribution"]["maximum_reconstruction_error"])
        ) else "FAIL",
        "rows": int(len(output)),
        "closed_set_rows": int(closed.sum()),
        "test_closed_set_rows": int(test_closed_mask.sum()),
        "maximum_reconstruction_error": float(np.nanmax(reconstruction[closed])),
        "maximum_scorer_alignment_error": float(np.nanmax(scorer_alignment[closed])),
        "targeted_minus_random_median_test": float(np.nanmedian(targeted[test_closed_mask] - random_effect[test_closed_mask])),
        "top_phase_intervention_match_test": float(np.mean(top_match[test_closed_mask])),
        "chance_match": float(contract["attribution"]["chance_top_phase_match"]),
        "boundary_cosine_median_test": float(np.nanmedian(boundary_cosine[test_closed_mask])),
        "boundary_top_match_test": float(np.nanmean(boundary_top_match[test_closed_mask])),
        "reference_change_cosine_median_test": float(np.nanmedian(reference_cosine[test_closed_mask])),
        "top_phase_share_test": {phase: float(value) for phase, value in zip(PHASES, top_share)},
        "evidence_sha256": sha256_file(path),
        "perturbation_is_supplementary_not_a_stop_gate": True,
    }
    write_json(V7_RESULTS_ROOT / "04_PHASE_EVIDENCE" / "phase_evidence_summary_v7.json", result)
    return result
