from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .modeling import phase_pool
from .util import set_seed, write_json
from .v2_modeling import load_rgb_arrays, prepare_inputs
from .v3_data import V3_RESULTS_ROOT, V3_RUN_ROOT, load_v3_contract, load_v3_frame, require_v3_frozen
from .v3_modeling import PHASES, _trace_model_class


def _load_models(phase_dim: int):
    import torch

    contract = load_v3_contract()
    models = []
    for seed in contract["random"]["model_seeds"]:
        path = V3_RUN_ROOT / "checkpoints" / f"final_trace_seed_{int(seed)}.pth"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        Model = _trace_model_class(
            phase_dim, int(contract["model"]["hidden_dimension"]), float(payload["residual_bound"])
        )
        model = Model()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append((int(seed), model))
    return models


def _infer(models, phase_values: np.ndarray, base: np.ndarray) -> dict:
    import torch

    phase_tensor = torch.from_numpy(np.asarray(phase_values, dtype=np.float32))
    base_tensor = torch.from_numpy(np.asarray(base, dtype=np.float32))
    predictions, contributions = [], []
    with torch.no_grad():
        for _, model in models:
            prediction, contribution, _, _ = model(phase_tensor, base_tensor)
            predictions.append(prediction.numpy())
            contributions.append(contribution.numpy())
    prediction = np.stack(predictions)
    contribution = np.stack(contributions)
    return {
        "prediction": prediction.mean(axis=0),
        "contributions": contribution.mean(axis=0),
        "seed_predictions": prediction,
        "seed_contributions": contribution,
    }


def _boundary_variant(delta: np.ndarray, boundary: int, direction: int) -> np.ndarray:
    """One-token boundary motion represented as a conservative adjacent-phase exchange."""
    changed = np.asarray(delta, dtype=np.float32).copy()
    fraction = 1.0 / 8.0
    left, right = boundary, boundary + 1
    if direction > 0:
        changed[:, left] = (1 - fraction) * changed[:, left] + fraction * changed[:, right]
    else:
        changed[:, right] = (1 - fraction) * changed[:, right] + fraction * changed[:, left]
    return changed


def _token_drop_variant(delta: np.ndarray, predicted_phases: np.ndarray, token: int) -> np.ndarray:
    changed = np.asarray(delta, dtype=np.float32).copy()
    for row in range(len(changed)):
        phase = int(predicted_phases[row, token])
        count = max(1, int(np.sum(predicted_phases[row] == phase)))
        changed[row, phase] *= max(0.0, (count - 1) / count)
    return changed


def _append_rows(records: list[dict], frame: pd.DataFrame, kind: str, name: str, result: dict) -> None:
    contribution = result["contributions"]
    top = np.argmax(np.abs(contribution), axis=1)
    for index, row in enumerate(frame.itertuples(index=False)):
        records.append(
            {
                "clip_uid": row.clip_uid,
                "event_family": row.event_family,
                "official_split": row.official_split,
                "analysis_role": row.analysis_role,
                "perturbation_kind": kind,
                "perturbation_name": name,
                "predicted_quality": float(result["prediction"][index]),
                "takeoff_contribution": float(contribution[index, 0]),
                "flight_contribution": float(contribution[index, 1]),
                "entry_contribution": float(contribution[index, 2]),
                "top_phase": PHASES[int(top[index])],
            }
        )


def _counterfactual_rows(frame: pd.DataFrame, inputs, models) -> list[dict]:
    reference_pool = np.flatnonzero(frame.analysis_role.isin(["fit", "validation"]).to_numpy())
    action = frame.action_type.to_numpy()
    family = frame.event_family.to_numpy()
    records: list[dict] = []
    for target_phase in range(3):
        modified = inputs.rgb_delta.copy()
        donors = np.full(len(frame), -1, dtype=int)
        other = [value for value in range(3) if value != target_phase]
        for index in range(len(frame)):
            candidates = reference_pool[
                (action[reference_pool] == action[index]) & (family[reference_pool] != family[index])
            ]
            candidates = candidates[candidates != index]
            if not len(candidates):
                continue
            distance = np.mean(
                (inputs.rgb_delta[candidates][:, other] - inputs.rgb_delta[index, other]) ** 2,
                axis=(1, 2),
            )
            donor = int(candidates[int(np.argmin(distance))])
            donors[index] = donor
            modified[index, target_phase] = inputs.rgb_delta[donor, target_phase]
        result = _infer(models, modified, inputs.base_quality)
        contribution = result["contributions"]
        for index, row in enumerate(frame.itertuples(index=False)):
            donor = donors[index]
            if donor < 0:
                continue
            records.append(
                {
                    "clip_uid": row.clip_uid,
                    "event_family": row.event_family,
                    "official_split": row.official_split,
                    "analysis_role": row.analysis_role,
                    "perturbation_kind": "counterfactual",
                    "perturbation_name": f"replace_{PHASES[target_phase]}",
                    "target_phase": PHASES[target_phase],
                    "donor_clip_uid": str(frame.iloc[donor].clip_uid),
                    "donor_event_family": str(frame.iloc[donor].event_family),
                    "donor_role": str(frame.iloc[donor].analysis_role),
                    "predicted_quality": float(result["prediction"][index]),
                    "takeoff_contribution": float(contribution[index, 0]),
                    "flight_contribution": float(contribution[index, 1]),
                    "entry_contribution": float(contribution[index, 2]),
                    "top_phase": PHASES[int(np.argmax(np.abs(contribution[index])))],
                }
            )
    return records


def _fit_review_risks(frame: pd.DataFrame, predictions: pd.DataFrame, stress: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    calibration = predictions.analysis_role == "calibration"
    pivot = stress.loc[stress.perturbation_kind.isin(["boundary", "token_drop", "noise"])].copy()
    baseline = predictions.set_index("clip_uid")
    pivot["absolute_change"] = np.abs(
        pivot.predicted_quality.to_numpy() - baseline.loc[pivot.clip_uid, "predicted_quality"].to_numpy()
    )
    instability = pivot.groupby("clip_uid").absolute_change.median().rename("phase_instability")
    predictions = predictions.merge(instability, on="clip_uid", how="left")
    predictions["phase_instability"] = predictions.phase_instability.fillna(0.0)
    error_features = predictions[["sigma_model", "reference_distance", "open_set", "phase_instability"]].astype(float)
    errors = np.abs(predictions.predicted_quality - predictions.execution_quality)
    error_target = errors[calibration] >= errors[calibration].quantile(0.75)
    error_model = LogisticRegression(max_iter=1000, random_state=20260817)
    error_model.fit(error_features.loc[calibration], error_target.astype(int))
    predictions["risk_error"] = error_model.predict_proba(error_features)[:, 1]

    disagreement_features = predictions[["sigma_judge", "phase_instability", "reference_distance"]].astype(float)
    disagreement_calibration = calibration & predictions.disagreement_primary_eligible
    disagreement_target = predictions.loc[disagreement_calibration, "judge_sample_sd"] >= predictions.loc[
        disagreement_calibration, "judge_sample_sd"
    ].quantile(0.75)
    disagreement_model = LogisticRegression(max_iter=1000, random_state=20260817)
    disagreement_model.fit(
        disagreement_features.loc[disagreement_calibration], disagreement_target.astype(int)
    )
    predictions["risk_disagreement"] = disagreement_model.predict_proba(disagreement_features)[:, 1]
    for name in ("risk_error", "risk_disagreement"):
        calibration_values = np.sort(predictions.loc[calibration, name].to_numpy())
        predictions[f"{name}_percentile"] = np.searchsorted(
            calibration_values, predictions[name].to_numpy(), side="right"
        ) / max(1, len(calibration_values))
    predictions["review_priority"] = predictions[
        ["risk_error_percentile", "risk_disagreement_percentile"]
    ].max(axis=1)
    threshold = float(predictions.loc[calibration, "review_priority"].quantile(0.80))
    predictions["review_recommended"] = predictions.review_priority >= threshold
    details = {
        "error_coefficients": error_model.coef_[0].tolist(),
        "disagreement_coefficients": disagreement_model.coef_[0].tolist(),
        "review_threshold": threshold,
        "calibration_review_fraction": float(predictions.loc[calibration, "review_recommended"].mean()),
    }
    return predictions, details


def stress_test_v3() -> dict:
    require_v3_frozen()
    frame = load_v3_frame()
    prediction_path = V3_RESULTS_ROOT / "03_FINAL" / "predictions_trace_v3.parquet"
    if not prediction_path.exists():
        raise RuntimeError("Run final v3 training before stress testing")
    predictions = pd.read_parquet(prediction_path)
    inputs, _ = prepare_inputs(
        frame, "none", allow_test_metrics=True, backbone="videomae", reference_roles=("fit", "validation")
    )
    models = _load_models(inputs.rgb_delta.shape[2])
    records: list[dict] = []
    baseline = _infer(models, inputs.rgb_delta, inputs.base_quality)
    _append_rows(records, frame, "baseline", "ensemble", baseline)
    for boundary in range(2):
        for direction in (-1, 1):
            changed = _boundary_variant(inputs.rgb_delta, boundary, direction)
            _append_rows(records, frame, "boundary", f"b{boundary}_{direction:+d}", _infer(models, changed, inputs.base_quality))
    for token in range(inputs.predicted_phases.shape[1]):
        changed = _token_drop_variant(inputs.rgb_delta, inputs.predicted_phases, token)
        _append_rows(records, frame, "token_drop", f"token_{token}", _infer(models, changed, inputs.base_quality))
    for noise_seed in range(10):
        rng = np.random.default_rng(20260817 + noise_seed)
        changed = inputs.rgb_delta + rng.normal(0.0, 0.05, inputs.rgb_delta.shape).astype(np.float32)
        _append_rows(records, frame, "noise", f"seed_{noise_seed}", _infer(models, changed, inputs.base_quality))
    for seed_index, (seed, _) in enumerate(models):
        result = {
            "prediction": baseline["seed_predictions"][seed_index],
            "contributions": baseline["seed_contributions"][seed_index],
        }
        _append_rows(records, frame, "model_seed", str(seed), result)
    records.extend(_counterfactual_rows(frame, inputs, models))
    stress = pd.DataFrame(records)
    destination = V3_RESULTS_ROOT / "04_STRESS" / "trace_stress_v3.parquet"
    stress.to_parquet(destination, index=False)
    predictions, risk_details = _fit_review_risks(frame, predictions, stress)
    predictions.to_parquet(prediction_path, index=False)
    result = {
        "status": "COMPLETE",
        "rows": int(len(stress)),
        "perturbations": stress.perturbation_kind.value_counts().to_dict(),
        "counterfactual_test_donor_leakage": int(
            ((stress.perturbation_kind == "counterfactual") & (stress.donor_role.fillna("") == "official_test")).sum()
        ),
        "risk_models": risk_details,
        "destination": str(destination),
    }
    write_json(V3_RESULTS_ROOT / "04_STRESS" / "stress_summary_v3.json", result)
    return result
