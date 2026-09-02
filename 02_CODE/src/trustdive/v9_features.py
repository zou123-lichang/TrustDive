from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .util import sha256_file, write_json
from .v7_data import V7_RESULTS_ROOT, load_v7_frame
from .v8_data import V8_RESULTS_ROOT
from .v9_data import V9_RESULTS_ROOT, load_v9_contract, require_v9_audit


TYPE_NAMES = ("persistent_bias", "action_preference", "phase_bias", "episodic_lapse")
PHASE_NAMES = ("takeoff", "flight", "entry")
JUDGE_FEATURES = (
    "score", "leave_one_median", "raw_residual", "jep_standardized_residual",
    "panel_rank", "officially_trimmed", "difference_from_rica", "difference_from_v7",
)
GLOBAL_FEATURES = (
    "difficulty", "teacher_uncertainty", "reference_distance", "reference_dispersion",
    "open_set", "teacher_quality", "v7_quality",
)


def _load_token_lookup() -> tuple[dict[str, int], dict[str, np.ndarray]]:
    path = V8_RESULTS_ROOT / "02_PHASE_TOKENS" / "reference_phase_tokens_development_v8.npz"
    with np.load(path, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    return {str(uid): i for i, uid in enumerate(data["clip_uid"])}, data


def _pool_phase_tokens(data: dict[str, np.ndarray]) -> np.ndarray:
    token = data["token"].astype(np.float32)
    weight = data["reference_weights"].astype(np.float32)
    weight = weight / np.maximum(weight.sum(axis=1, keepdims=True), 1e-8)
    return np.sum(token * weight[:, :, None, None], axis=1).astype(np.float32)


def build_judge_phase_features_v9(stage: str) -> dict:
    require_v9_audit()
    if stage not in {"development", "final"}:
        raise ValueError(stage)
    pair_path = V9_RESULTS_ROOT / "02_CONFLICTS" / f"synthetic_judge_pairs_{stage}_v9.parquet"
    if not pair_path.exists():
        raise RuntimeError(f"Generate {stage} conflicts first")
    pairs = pd.read_parquet(pair_path)
    artifact = joblib.load(V9_RESULTS_ROOT / "01_SIMULATOR" / "judge_simulator_v9.joblib")
    predictions = pd.read_parquet(V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet").set_index("clip_uid")
    token_lookup, token_data = _load_token_lookup()
    pooled = _pool_phase_tokens(token_data)
    type_map = {name: i for i, name in enumerate(TYPE_NAMES)}
    frame = load_v7_frame().set_index("clip_uid")
    action_names = sorted(frame.action_type.astype(str).unique())
    action_map = {name: i + 1 for i, name in enumerate(action_names)}
    n = len(pairs)
    judge = np.zeros((n, 7, len(JUDGE_FEATURES)), np.float32)
    phase = np.zeros((n, 3, pooled.shape[-1]), np.float32)
    global_feature = np.zeros((n, len(GLOBAL_FEATURES)), np.float32)
    action_index = np.zeros(n, np.int64)
    panels = np.stack(pairs.panel_scores_json.map(json.loads).map(np.asarray)).astype(float)
    controls = np.zeros_like(panels)
    for slot in range(7):
        controls[:, slot] = np.median(np.delete(panels, slot, axis=1), axis=1)
    scale_frame = pd.DataFrame({
        "control": controls.reshape(-1),
        "action_type": np.repeat(pairs.action_type.astype(str).to_numpy(), 7),
        "difficulty": np.repeat(pairs.difficulty.to_numpy(dtype=float), 7),
    })
    scales = np.maximum(np.exp(artifact["model"].predict(scale_frame)), 0.20).reshape(n, 7)
    for i, row in enumerate(pairs.itertuples(index=False)):
        values = panels[i]
        pred = predictions.loc[row.clip_uid]
        rica_q = float(pred.teacher_predicted_score) / (3.0 * float(row.difficulty))
        v7_q = float(pred.trustdive_predicted_score) / (3.0 * float(row.difficulty))
        rank = pd.Series(values).rank(method="average").to_numpy(dtype=float) - 1.0
        order = np.argsort(values, kind="stable")
        trimmed = np.zeros(7, dtype=float); trimmed[order[:2]] = 1.0; trimmed[order[-2:]] = 1.0
        for slot in range(7):
            residual = values[slot] - controls[i, slot]
            judge[i, slot] = [values[slot], controls[i, slot], residual, residual / scales[i, slot],
                              rank[slot] / 6.0, trimmed[slot], values[slot] - rica_q, values[slot] - v7_q]
        idx = token_lookup[str(row.clip_uid)]
        phase[i] = pooled[idx]
        global_feature[i] = [float(row.difficulty), float(pred.teacher_uncertainty),
                             float(pred.reference_distance), float(pred.reference_dispersion),
                             float(bool(pred.open_set)), rica_q, v7_q]
        action_index[i] = action_map.get(str(row.action_type), 0)
    out = V9_RESULTS_ROOT / "03_FEATURES" / f"judge_phase_features_{stage}_v9.npz"
    np.savez_compressed(
        out, judge_token=judge, phase_token=phase, global_feature=global_feature,
        action_index=action_index, clip_uid=pairs.clip_uid.to_numpy(dtype=str),
        pair_id=pairs.pair_id.to_numpy(dtype=str), analysis_role=pairs.analysis_role.to_numpy(dtype=str),
        source_role=pairs.source_role.to_numpy(dtype=str), event_family=pairs.event_family.to_numpy(dtype=str),
        is_anomaly=pairs.is_anomaly.to_numpy(dtype=np.int8), target_slot=pairs.target_slot.to_numpy(dtype=np.int8),
        type_index=np.asarray([type_map[x] for x in pairs.scenario_type], dtype=np.int8),
        target_phase=pairs.target_phase.to_numpy(dtype=np.int8), severity_sigma=pairs.severity_sigma.to_numpy(dtype=np.float32),
        stealth_anomaly=pairs.stealth_anomaly.to_numpy(dtype=np.int8), variant=pairs.variant.to_numpy(dtype=str),
        scenario_type=pairs.scenario_type.to_numpy(dtype=str), panel_aggregate=pairs.panel_aggregate.to_numpy(dtype=np.float32),
        original_panel=np.stack(pairs.original_panel_json.map(json.loads).map(np.asarray)).astype(np.float32),
        simulated_panel=np.stack(pairs.panel_scores_json.map(json.loads).map(np.asarray)).astype(np.float32),
        judge_feature_names=np.asarray(JUDGE_FEATURES), global_feature_names=np.asarray(GLOBAL_FEATURES),
        phase_feature_names=token_data["token_names"], action_names=np.asarray(action_names),
    )
    result = {"status":"PASS", "stage":stage, "rows":int(n), "judge_shape":list(judge.shape),
              "phase_shape":list(phase.shape), "global_shape":list(global_feature.shape),
              "generator_source":"frozen_v2_videomae_phase_contributions",
              "detector_source":"v8_rica2_i3d_counterfactual_tokens",
              "sources_independent":True, "output_sha256":sha256_file(out)}
    write_json(V9_RESULTS_ROOT / "03_FEATURES" / f"feature_summary_{stage}_v9.json", result)
    return result


def load_features_v9(stage: str) -> dict[str, np.ndarray]:
    path = V9_RESULTS_ROOT / "03_FEATURES" / f"judge_phase_features_{stage}_v9.npz"
    if not path.exists():
        raise RuntimeError(f"Build v9 {stage} judge-phase features first")
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}
