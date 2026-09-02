from __future__ import annotations

import numpy as np
import pandas as pd

from .util import sha256_file, write_json
from .v4_counterfactual import PHASES, exact_three_phase_shapley
from .v8_data import V8_RESULTS_ROOT, load_v8_contract
from .v8_modeling import _arrays, load_final_models_v8


def _evaluate_coalitions(models, arrays, tokens, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models:
        model.to(device)
        model.module.eval()
    score_values = np.full((len(indices), 8), np.nan, dtype=np.float32)
    disagreement_values = np.full((len(indices), 8), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), 512):
            chunk = indices[start : start + 512]
            token = torch.from_numpy(arrays.token[chunk]).to(device)
            weight = torch.from_numpy(arrays.weights[chunk]).to(device)
            for mask in range(8):
                phase_mask = torch.as_tensor(
                    [[float(bool(mask & (1 << phase))) for phase in range(3)]] * len(chunk),
                    dtype=torch.float32,
                    device=device,
                )
                base = torch.from_numpy(tokens["base_coalition"][chunk, mask].astype(np.float32)).to(device)
                model_outputs = [model.module(token, weight, base, phase_mask) for model in models]
                score_values[start : start + len(chunk), mask] = torch.stack(
                    [item["quality"] for item in model_outputs]
                ).mean(dim=0).cpu().numpy()
                disagreement_values[start : start + len(chunk), mask] = torch.stack(
                    [item["log_excess_scale"] for item in model_outputs]
                ).mean(dim=0).cpu().numpy()
    return score_values, disagreement_values


def build_dual_evidence_v8() -> dict:
    prediction_path = V8_RESULTS_ROOT / "04_FINAL" / "predictions_v8.parquet"
    if not prediction_path.exists():
        raise RuntimeError("Run train --protocol v8 --stage final first")
    arrays, manifest, tokens = _arrays(final=True)
    models = load_final_models_v8()
    closed = ~arrays.open_set
    indices = np.flatnonzero(closed)
    score_coalition = np.full((len(manifest), 8), np.nan, dtype=np.float32)
    risk_coalition = np.full((len(manifest), 8), np.nan, dtype=np.float32)
    score_values, risk_values = _evaluate_coalitions(models, arrays, tokens, indices)
    score_coalition[indices] = score_values
    risk_coalition[indices] = risk_values
    score_phi = np.full((len(manifest), 3), np.nan, dtype=np.float32)
    risk_phi = np.full((len(manifest), 3), np.nan, dtype=np.float32)
    score_phi[indices] = exact_three_phase_shapley(score_values[:, None, :])[:, 0]
    risk_phi[indices] = exact_three_phase_shapley(risk_values[:, None, :])[:, 0]
    score_reconstruction = np.abs(score_coalition[:, 0] + np.nansum(score_phi, axis=1) - score_coalition[:, 7])
    risk_reconstruction = np.abs(risk_coalition[:, 0] + np.nansum(risk_phi, axis=1) - risk_coalition[:, 7])

    score_occlusion = np.full((len(manifest), 3), np.nan, dtype=np.float32)
    risk_occlusion = np.full((len(manifest), 3), np.nan, dtype=np.float32)
    for phase in range(3):
        score_occlusion[:, phase] = score_coalition[:, 7] - score_coalition[:, 7 ^ (1 << phase)]
        risk_occlusion[:, phase] = risk_coalition[:, 7] - risk_coalition[:, 7 ^ (1 << phase)]
    score_top = np.full(len(manifest), -1, dtype=int)
    risk_top = np.full(len(manifest), -1, dtype=int)
    score_actual = np.full(len(manifest), -1, dtype=int)
    risk_actual = np.full(len(manifest), -1, dtype=int)
    score_top[indices] = np.argmax(np.abs(score_phi[indices]), axis=1)
    risk_top[indices] = np.argmax(np.abs(risk_phi[indices]), axis=1)
    score_actual[indices] = np.argmax(np.abs(score_occlusion[indices]), axis=1)
    risk_actual[indices] = np.argmax(np.abs(risk_occlusion[indices]), axis=1)
    rng = np.random.default_rng(int(load_v8_contract()["statistics"]["seed"]))
    random_phase = rng.integers(0, 3, size=len(manifest))

    output = manifest[[
        "clip_uid", "official_split", "analysis_role", "event_family", "action_type",
        "difficulty", "dive_score", "expected_log_judge_sd",
    ]].copy()
    output["open_set"] = arrays.open_set
    output["score_reference_baseline"] = score_coalition[:, 0]
    output["score_predicted_quality"] = score_coalition[:, 7]
    output["disagreement_reference_baseline"] = risk_coalition[:, 0]
    output["disagreement_predicted_excess"] = risk_coalition[:, 7]
    for phase_index, phase in enumerate(PHASES):
        output[f"score_phi_{phase}"] = score_phi[:, phase_index]
        output[f"score_occlusion_{phase}"] = score_occlusion[:, phase_index]
        output[f"disagreement_phi_{phase}"] = risk_phi[:, phase_index]
        output[f"disagreement_occlusion_{phase}"] = risk_occlusion[:, phase_index]
    names = np.asarray(PHASES, dtype=object)
    output["score_top_phase"] = np.where(score_top >= 0, names[np.maximum(score_top, 0)], "open_set")
    output["score_actual_intervention_phase"] = np.where(score_actual >= 0, names[np.maximum(score_actual, 0)], "open_set")
    output["disagreement_top_phase"] = np.where(risk_top >= 0, names[np.maximum(risk_top, 0)], "open_set")
    output["disagreement_actual_intervention_phase"] = np.where(risk_actual >= 0, names[np.maximum(risk_actual, 0)], "open_set")
    output["score_top_match"] = (score_top == score_actual) & closed
    output["disagreement_top_match"] = (risk_top == risk_actual) & closed
    output["score_targeted_effect"] = np.nan
    output["score_random_effect"] = np.nan
    output["disagreement_targeted_effect"] = np.nan
    output["disagreement_random_effect"] = np.nan
    for index in indices:
        output.loc[index, "score_targeted_effect"] = abs(score_occlusion[index, score_top[index]])
        output.loc[index, "score_random_effect"] = abs(score_occlusion[index, random_phase[index]])
        output.loc[index, "disagreement_targeted_effect"] = abs(risk_occlusion[index, risk_top[index]])
        output.loc[index, "disagreement_random_effect"] = abs(risk_occlusion[index, random_phase[index]])
    output["score_reconstruction_error"] = score_reconstruction
    output["disagreement_reconstruction_error"] = risk_reconstruction
    for mask in range(8):
        output[f"score_coalition_{mask}"] = score_coalition[:, mask]
        output[f"disagreement_coalition_{mask}"] = risk_coalition[:, mask]
    path = V8_RESULTS_ROOT / "05_DUAL_EVIDENCE" / "dual_phase_evidence_v8.parquet"
    output.to_parquet(path, index=False)

    test_closed = (manifest.analysis_role.to_numpy() == "official_test") & closed
    contract = load_v8_contract()
    max_score_error = float(np.nanmax(score_reconstruction[closed]))
    max_risk_error = float(np.nanmax(risk_reconstruction[closed]))
    score_match = float(output.loc[test_closed, "score_top_match"].mean())
    risk_match = float(output.loc[test_closed, "disagreement_top_match"].mean())
    checks = {
        "score_exact_reconstruction": max_score_error <= float(contract["attribution"]["maximum_reconstruction_error"]),
        "disagreement_exact_reconstruction": max_risk_error <= float(contract["attribution"]["maximum_reconstruction_error"]),
        "score_top_match": score_match >= float(contract["attribution"]["minimum_score_top_match"]),
        "disagreement_top_match": risk_match >= float(contract["attribution"]["minimum_disagreement_top_match"]),
        "score_targeted_exceeds_random": float(np.nanmedian(
            output.loc[test_closed, "score_targeted_effect"] - output.loc[test_closed, "score_random_effect"]
        )) > 0,
        "disagreement_targeted_exceeds_random": float(np.nanmedian(
            output.loc[test_closed, "disagreement_targeted_effect"] - output.loc[test_closed, "disagreement_random_effect"]
        )) > 0,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_score_reconstruction_error": max_score_error,
        "maximum_disagreement_reconstruction_error": max_risk_error,
        "score_top_match_test": score_match,
        "disagreement_top_match_test": risk_match,
        "score_targeted_minus_random_median_test": float(np.nanmedian(
            output.loc[test_closed, "score_targeted_effect"] - output.loc[test_closed, "score_random_effect"]
        )),
        "disagreement_targeted_minus_random_median_test": float(np.nanmedian(
            output.loc[test_closed, "disagreement_targeted_effect"] - output.loc[test_closed, "disagreement_random_effect"]
        )),
        "output_sha256": sha256_file(path),
    }
    write_json(V8_RESULTS_ROOT / "05_DUAL_EVIDENCE" / "dual_phase_summary_v8.json", result)
    return result
