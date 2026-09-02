from __future__ import annotations

import json

import joblib
import numpy as np

from .util import sha256_file, write_json
from .v4_counterfactual import exact_three_phase_shapley, hybrid_sequence
from .v5_counterfactual import _phase_labels
from .v6_attribution import _load_sequences, _predict_teacher_latent
from .v6_modeling import load_v6_assets
from .v7_data import V7_RESULTS_ROOT, V7_RUN_ROOT
from .v7_modeling import (
    _fit_ridge_artifact,
    load_reference_map_v7,
    make_ridge_features_v7,
    predict_ridge_artifact,
)
from .v8_data import V8_RESULTS_ROOT, load_v8_contract, require_v8_audit


TOKEN_NAMES = (
    "shapley",
    "occlusion",
    "phase_distance",
    "reference_true_quality",
    "reference_teacher_quality",
    "query_reference_teacher_gap",
    "reference_teacher_residual",
    "global_reference_distance",
    "reference_weight",
    "hybrid_ood_fraction",
)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    return float(1.0 - np.dot(a, b) / denominator)


def _artifact_for_stage(final: bool, mapping: dict[str, np.ndarray]):
    if final:
        return joblib.load(V7_RUN_ROOT / "checkpoints" / "final_plain_ridge_v7.joblib")
    assets = load_v6_assets()
    selection = json.loads(
        (V7_RESULTS_ROOT / "02_BASELINES" / "selected_model_v7.json").read_text(encoding="utf-8")
    )
    alpha = float(selection["plain_ridge"]["alpha"])
    features = make_ridge_features_v7(assets, mapping)
    residual = assets.frame.execution_quality.to_numpy(dtype=float) - assets.teacher_quality
    fit = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "fit")
    return _fit_ridge_artifact(features, residual, fit, alpha, None)


def _per_reference_coalitions(
    indices: np.ndarray,
    assets,
    mapping: dict[str, np.ndarray],
    artifact: dict,
    sequences: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    refs = mapping["references"][:, :5].astype(int)
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
    if not hybrids:
        shape = (len(indices), 5, 8)
        return np.full(shape, np.nan, np.float32), np.full(shape + (256,), np.nan, np.float32)
    teacher, latent = _predict_teacher_latent(np.stack(hybrids), np.stack(hybrid_actions))
    query_vector = np.asarray([query for _, query, _, _ in metadata], dtype=int)
    features = make_ridge_features_v7(
        assets,
        mapping,
        global_latent=latent,
        teacher_quality=teacher,
        query_indices=query_vector,
    )
    quality = predict_ridge_artifact(artifact, features, teacher)
    values = np.full((len(indices), 5, 8), np.nan, dtype=np.float32)
    latent_values = np.full((len(indices), 5, 8, 256), np.nan, dtype=np.float32)
    for value, vector, (local, _query, slot, mask) in zip(quality, latent, metadata):
        values[local, slot, mask] = value
        latent_values[local, slot, mask] = vector
    return values, latent_values


def build_phase_tokens_v8(final: bool) -> dict:
    require_v8_audit()
    assets = load_v6_assets()
    frame = assets.frame
    contract = load_v8_contract()
    mapping = load_reference_map_v7(final=final)
    artifact = _artifact_for_stage(final, mapping)
    sequences, actions = _load_sequences()
    labels = _phase_labels(sequences.shape[1])
    refs = mapping["references"][:, :5].astype(int)
    valid = refs >= 0
    closed = ~mapping["open_set"].astype(bool)
    indices = np.flatnonzero(closed)

    values = np.full((len(frame), 5, 8), np.nan, dtype=np.float32)
    hybrid_ood = np.full((len(frame), 5), np.nan, dtype=np.float32)
    fit = frame.analysis_role.to_numpy() == "fit"
    center = assets.global_latent[fit].mean(axis=0)
    scale = np.where(assets.global_latent[fit].std(axis=0) > 1e-6, assets.global_latent[fit].std(axis=0), 1.0)
    real_ood = np.sqrt(np.mean(((assets.global_latent[fit] - center) / scale) ** 2, axis=1))
    ood_threshold = float(np.quantile(real_ood, float(contract["phase_tokens"]["hybrid_ood_quantile"])))

    chunk_size = 100
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start : start + chunk_size]
        chunk_values, chunk_latent = _per_reference_coalitions(
            chunk, assets, mapping, artifact, sequences, actions, labels
        )
        values[chunk] = chunk_values
        finite = np.isfinite(chunk_latent).all(axis=-1)
        standardized = np.zeros_like(chunk_latent, dtype=np.float32)
        standardized[finite] = (chunk_latent[finite] - center) / scale
        distance = np.sqrt(np.mean(standardized**2, axis=-1))
        hybrid_ood[chunk] = np.mean(np.where(finite, distance > ood_threshold, np.nan), axis=2)
        print(
            f"V8 phase tokens ({'final' if final else 'development'}): "
            f"{min(start + len(chunk), len(indices))}/{len(indices)}",
            flush=True,
        )

    phi = exact_three_phase_shapley(values)
    occlusion = np.full((len(frame), 5, 3), np.nan, dtype=np.float32)
    for phase in range(3):
        occlusion[:, :, phase] = values[:, :, 7] - values[:, :, 7 ^ (1 << phase)]
    tokens = np.zeros((len(frame), 5, 3, len(TOKEN_NAMES)), dtype=np.float32)
    ref_true = frame.execution_quality.to_numpy(dtype=float)[np.maximum(refs, 0)]
    ref_teacher = assets.teacher_quality[np.maximum(refs, 0)]
    distance = np.nan_to_num(mapping["distances"][:, :5], nan=2.0)
    weights = np.where(valid, mapping["weights"][:, :5], 0.0).astype(np.float32)
    for query in range(len(frame)):
        for slot, ref in enumerate(refs[query]):
            if ref < 0:
                continue
            for phase in range(3):
                q_phase = sequences[query][labels[query] == phase].mean(axis=0)
                r_phase = sequences[ref][labels[ref] == phase].mean(axis=0)
                tokens[query, slot, phase] = np.asarray(
                    [
                        phi[query, slot, phase],
                        occlusion[query, slot, phase],
                        _cosine_distance(q_phase, r_phase),
                        ref_true[query, slot],
                        ref_teacher[query, slot],
                        assets.teacher_quality[query] - ref_teacher[query, slot],
                        ref_true[query, slot] - ref_teacher[query, slot],
                        distance[query, slot],
                        weights[query, slot],
                        hybrid_ood[query, slot],
                    ],
                    dtype=np.float32,
                )
    tokens[~np.isfinite(tokens)] = 0.0

    normalized_weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    base_coalition = np.nansum(values * normalized_weights[:, :, None], axis=1).astype(np.float32)
    full_features = make_ridge_features_v7(assets, mapping)
    full_quality = predict_ridge_artifact(artifact, full_features, assets.teacher_quality)
    full_quality[mapping["open_set"].astype(bool)] = assets.teacher_quality[mapping["open_set"].astype(bool)]
    base_coalition[closed, 7] = full_quality[closed].astype(np.float32)

    stage = "final" if final else "development"
    path = V8_RESULTS_ROOT / "02_PHASE_TOKENS" / f"reference_phase_tokens_{stage}_v8.npz"
    np.savez_compressed(
        path,
        clip_uid=frame.clip_uid.to_numpy(dtype=str),
        token=tokens.astype(np.float32),
        token_names=np.asarray(TOKEN_NAMES),
        reference_indices=refs.astype(np.int32),
        reference_weights=normalized_weights.astype(np.float32),
        valid_reference=valid,
        open_set=mapping["open_set"].astype(bool),
        base_coalition=base_coalition.astype(np.float32),
        per_reference_coalition=values.astype(np.float32),
        hybrid_ood_fraction=hybrid_ood.astype(np.float32),
    )
    finite_ood = hybrid_ood[np.isfinite(hybrid_ood)]
    summary = {
        "status": "PASS",
        "stage": stage,
        "rows": int(len(frame)),
        "closed_set_rows": int(closed.sum()),
        "open_set_rows": int((~closed).sum()),
        "token_shape": list(tokens.shape),
        "token_names": list(TOKEN_NAMES),
        "hybrid_ood_threshold": ood_threshold,
        "hybrid_ood_fraction": float(np.mean(finite_ood)) if len(finite_ood) else float("nan"),
        "output_sha256": sha256_file(path),
    }
    write_json(V8_RESULTS_ROOT / "02_PHASE_TOKENS" / f"phase_token_summary_{stage}_v8.json", summary)
    return summary


def load_phase_tokens_v8(final: bool) -> dict[str, np.ndarray]:
    stage = "final" if final else "development"
    path = V8_RESULTS_ROOT / "02_PHASE_TOKENS" / f"reference_phase_tokens_{stage}_v8.npz"
    if not path.exists():
        raise RuntimeError(f"Run build-phase-tokens --protocol v8 for {stage} assets")
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}
