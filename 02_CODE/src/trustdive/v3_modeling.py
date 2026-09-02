from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import RUNS_ROOT
from .metrics import restore_total_score, score_metrics
from .statistics import split_conformal_radius
from .util import set_seed, sha256_file, write_json
from .v2_modeling import _judge_matrix, _model_classes, prepare_inputs, train_single_model
from .v3_data import (
    V3_RESULTS_ROOT,
    V3_RUN_ROOT,
    load_v3_contract,
    load_v3_frame,
    require_v3_audit,
    require_v3_frozen,
)


PHASES = ("takeoff", "flight", "entry")


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else 0.0


def attribution_coverage(contributions: np.ndarray, residual: np.ndarray) -> np.ndarray:
    numerator = np.abs(np.asarray(contributions, dtype=float)).sum(axis=1)
    denominator = numerator + np.abs(np.asarray(residual, dtype=float))
    return np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 1e-12)


def _trace_model_class(phase_dim: int, hidden: int, residual_bound: float):
    import torch
    import torch.nn as nn

    class TraceStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.phase_heads = nn.ModuleList(
                [nn.Sequential(nn.Linear(phase_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)) for _ in range(3)]
            )
            self.residual_head = (
                nn.Sequential(
                    nn.Linear(3 * phase_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
                )
                if residual_bound > 0
                else None
            )
            self.scale_head = nn.Sequential(
                nn.Linear(3 * phase_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
            )

        def forward(self, phase_delta, base):
            contributions = torch.stack(
                [head(phase_delta[:, index]).squeeze(-1) for index, head in enumerate(self.phase_heads)],
                dim=1,
            )
            if self.residual_head is None:
                residual = torch.zeros(len(phase_delta), device=phase_delta.device)
            else:
                residual = float(residual_bound) * torch.tanh(
                    self.residual_head(phase_delta.flatten(1)).squeeze(-1)
                )
            sigma = torch.nn.functional.softplus(
                self.scale_head(phase_delta.flatten(1)).squeeze(-1)
            ) + 0.05
            prediction = base + contributions.sum(dim=1) + residual
            return prediction, contributions, residual, sigma

    return TraceStudent


def _teacher_from_checkpoint(frame: pd.DataFrame) -> np.ndarray:
    """Fit-only v2 teacher. Its validation labels were never used to fit this checkpoint."""
    import torch

    inputs, _ = prepare_inputs(frame, "none", allow_test_metrics=False, backbone="i3d")
    _, RelativeRGB, _ = _model_classes(
        inputs.global_features.shape[1], inputs.rgb_delta.shape[2], 0.0
    )
    checkpoint = RUNS_ROOT / "v2_disagreement" / "checkpoints" / "baseline_relative_v2.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = RelativeRGB()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        prediction, _, _, _ = model(
            torch.from_numpy(inputs.global_features).float(),
            torch.from_numpy(inputs.rgb_delta).float(),
            torch.from_numpy(inputs.base_quality).float(),
        )
    return prediction.numpy().astype(np.float32)


def build_teacher_targets_v3() -> pd.DataFrame:
    require_v3_audit()
    frame = load_v3_frame()
    pilot = _teacher_from_checkpoint(frame)
    final_path = V3_RESULTS_ROOT.parent / "V2_DISAGREEMENT" / "03_FINAL" / "relative_predictions_v2.parquet"
    final = pd.read_parquet(final_path)[["clip_uid", "predicted_quality"]].rename(
        columns={"predicted_quality": "teacher_final_quality"}
    )
    output = frame[
        [
            "clip_uid", "feature_key", "official_split", "analysis_role", "source_role",
            "event_family", "action_type", "difficulty", "execution_quality",
        ]
    ].copy()
    output["teacher_pilot_quality"] = pilot
    output = output.merge(final, on="clip_uid", how="left", validate="one_to_one")
    if output.teacher_final_quality.isna().any():
        raise RuntimeError("Frozen v2 final teacher is incomplete")
    destination = V3_RESULTS_ROOT / "01_TEACHER" / "teacher_targets_v3.parquet"
    output.to_parquet(destination, index=False)
    summary = {
        "status": "PASS",
        "rows": int(len(output)),
        "teacher_pilot_fit_roles": ["fit"],
        "teacher_final_fit_roles": ["fit", "validation"],
        "official_test_labels_used_for_teacher_fit": False,
        "pilot_checkpoint_sha256": sha256_file(
            RUNS_ROOT / "v2_disagreement" / "checkpoints" / "baseline_relative_v2.pth"
        ),
        "final_predictions_sha256": sha256_file(final_path),
        "destination": str(destination),
    }
    write_json(V3_RESULTS_ROOT / "01_TEACHER" / "teacher_targets_summary_v3.json", summary)
    return output


def load_teacher_targets_v3() -> pd.DataFrame:
    path = V3_RESULTS_ROOT / "01_TEACHER" / "teacher_targets_v3.parquet"
    if not path.exists():
        raise RuntimeError("Run build-teacher-targets --protocol v3 first")
    return pd.read_parquet(path)


def _fit_trace_student(
    frame: pd.DataFrame,
    inputs,
    teacher: np.ndarray,
    alpha: float,
    residual_bound: float,
    seed: int,
    fit_roles: tuple[str, ...] = ("fit",),
    validation_role: str = "validation",
    role_column: str = "analysis_role",
    fixed_epochs: int | None = None,
) -> dict:
    import torch
    import torch.nn.functional as F

    contract = load_v3_contract()
    set_seed(seed)
    fit_indices = np.flatnonzero(frame[role_column].isin(fit_roles).to_numpy())
    validation_indices = np.flatnonzero(frame[role_column].to_numpy() == validation_role)
    if not len(fit_indices) or not len(validation_indices):
        raise ValueError("Trace student requires nonempty fit and validation partitions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase = torch.from_numpy(inputs.rgb_delta).float().to(device)
    base = torch.from_numpy(inputs.base_quality).float().to(device)
    target = torch.from_numpy(frame.execution_quality.to_numpy(dtype=np.float32)).to(device)
    teacher_tensor = torch.from_numpy(np.asarray(teacher, dtype=np.float32)).to(device)
    fit = torch.from_numpy(fit_indices).long().to(device)
    validation = torch.from_numpy(validation_indices).long().to(device)
    judge_matrix, judge_eligible, _ = _judge_matrix(frame)
    judges = torch.from_numpy(judge_matrix).float().to(device)
    eligible_fit_np = np.intersect1d(fit_indices, np.flatnonzero(judge_eligible))
    eligible_fit = torch.from_numpy(eligible_fit_np).long().to(device)
    Model = _trace_model_class(
        phase.shape[2], int(contract["model"]["hidden_dimension"]), residual_bound
    )
    model = Model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract["model"]["learning_rate"]),
        weight_decay=float(contract["model"]["weight_decay"]),
    )
    best_state = None
    best_loss = np.inf
    best_epoch = 0
    stale = 0
    maximum = fixed_epochs or int(contract["model"]["maximum_epochs"])
    for epoch in range(maximum):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, _, residual, sigma = model(phase, base)
        true_loss = F.huber_loss(prediction[fit], target[fit], delta=1.0)
        teacher_loss = F.huber_loss(prediction[fit], teacher_tensor[fit], delta=1.0)
        loss = (1.0 - alpha) * true_loss + alpha * teacher_loss
        if len(eligible_fit_np):
            distribution = torch.distributions.StudentT(
                df=float(contract["model"]["student_t_df"]),
                loc=prediction[eligible_fit, None],
                scale=sigma[eligible_fit, None],
            )
            loss = loss + float(contract["model"]["judge_nll_weight"]) * (
                -distribution.log_prob(judges[eligible_fit]).mean()
            )
        if residual_bound > 0:
            loss = loss + float(contract["model"]["residual_l1_weight"]) * residual[fit].abs().mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prediction, _, _, _ = model(phase, base)
            val_loss = float(F.huber_loss(val_prediction[validation], target[validation], delta=1.0).cpu())
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if fixed_epochs is not None:
            best_state, best_loss, best_epoch = state, val_loss, epoch + 1
            continue
        if val_loss < best_loss - 1e-7:
            best_state, best_loss, best_epoch, stale = state, val_loss, epoch + 1, 0
        else:
            stale += 1
            if stale >= int(contract["model"]["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Trace student produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction, contributions, residual, sigma = model(phase, base)
        ablated = []
        kept = []
        for phase_index in range(3):
            removed_input = phase.clone()
            removed_input[:, phase_index] = 0
            ablated.append(model(removed_input, base)[0])
            kept_input = torch.zeros_like(phase)
            kept_input[:, phase_index] = phase[:, phase_index]
            kept.append(model(kept_input, base)[0])
    prediction_np = prediction.cpu().numpy()
    contributions_np = contributions.cpu().numpy()
    residual_np = residual.cpu().numpy()
    validation_scores = score_metrics(
        frame.dive_score.to_numpy()[validation_indices],
        restore_total_score(prediction_np[validation_indices], frame.difficulty.to_numpy()[validation_indices]),
    )
    return {
        "model": model,
        "state_dict": best_state,
        "prediction": prediction_np,
        "contributions": contributions_np,
        "residual": residual_np,
        "sigma_judge": sigma.cpu().numpy(),
        "ablated_predictions": torch.stack(ablated, dim=1).cpu().numpy(),
        "kept_predictions": torch.stack(kept, dim=1).cpu().numpy(),
        "coverage": attribution_coverage(contributions_np, residual_np),
        "best_epoch": int(best_epoch),
        "validation_loss": float(best_loss),
        "validation_score": validation_scores,
    }


def pilot_trace_v3() -> dict:
    require_v3_audit()
    frame = load_v3_frame()
    teachers = load_teacher_targets_v3().set_index("clip_uid").loc[frame.clip_uid]
    teacher = teachers.teacher_pilot_quality.to_numpy(dtype=np.float32)
    inputs, preparation = prepare_inputs(frame, "none", allow_test_metrics=False, backbone="videomae")
    contract = load_v3_contract()
    validation = frame.analysis_role.to_numpy() == "validation"
    records = []
    checkpoints = []
    for alpha in contract["model"]["alpha_grid"]:
        for residual_bound in contract["model"]["residual_bounds"]:
            result = _fit_trace_student(
                frame, inputs, teacher, float(alpha), float(residual_bound),
                int(contract["random"]["model_seeds"][0]),
            )
            coverage = result["coverage"][validation]
            residual = np.abs(result["residual"][validation])
            gates = {
                "score_noninferior": result["validation_score"]["spearman"] >= float(contract["pilot_gates"]["minimum_validation_spearman"]),
                "median_coverage": float(np.median(coverage)) >= float(contract["pilot_gates"]["minimum_median_attribution_coverage"]),
                "coverage_fraction": float(np.mean(coverage >= 0.70)) >= float(contract["pilot_gates"]["minimum_fraction_coverage_at_0_70"]),
                "residual": float(np.median(residual)) <= float(contract["pilot_gates"]["maximum_median_absolute_residual"]),
            }
            record = {
                "alpha": float(alpha), "residual_bound": float(residual_bound),
                "validation_spearman": float(result["validation_score"]["spearman"]),
                "validation_relative_l2": float(result["validation_score"]["relative_l2"]),
                "validation_mae": float(result["validation_score"]["mae"]),
                "median_coverage": float(np.median(coverage)),
                "fraction_coverage_ge_0_70": float(np.mean(coverage >= 0.70)),
                "median_absolute_residual": float(np.median(residual)),
                "best_epoch": int(result["best_epoch"]),
                **{f"gate_{key}": bool(value) for key, value in gates.items()},
                "eligible": bool(all(gates.values())),
            }
            records.append(record)
            checkpoint = V3_RUN_ROOT / "checkpoints" / f"pilot_alpha_{alpha}_residual_{residual_bound}.pth"
            import torch
            torch.save({"state_dict": result["state_dict"], "record": record}, checkpoint)
            checkpoints.append(checkpoint)
    trials = pd.DataFrame(records)
    trials.to_csv(V3_RESULTS_ROOT / "02_PILOT" / "pilot_trials_v3.csv", index=False)
    eligible = trials.loc[trials.eligible].copy()
    if eligible.empty:
        selection = {
            "status": "STOP", "reason": "No candidate passed every prespecified pilot gate",
            "candidates": records, "official_test_metrics_revealed": False,
            "preparation": preparation,
        }
    else:
        maximum = float(eligible.validation_spearman.max())
        tied = eligible.loc[
            eligible.validation_spearman >= maximum - float(contract["pilot_gates"]["spearman_tie_tolerance"])
        ].sort_values(["residual_bound", "median_coverage"], ascending=[True, False])
        chosen = tied.iloc[0].to_dict()
        selection = {
            "status": "PASS", "selected": chosen,
            "selection_rule": "validation Spearman; within 0.005 prefer pure additive then higher coverage",
            "official_test_metrics_revealed": False,
            "preparation": preparation,
        }
    write_json(V3_RESULTS_ROOT / "02_PILOT" / "selected_config_v3.json", selection)
    return selection


def _ensemble_predictions(frame: pd.DataFrame, inputs, teacher: np.ndarray, selection: dict) -> tuple[pd.DataFrame, dict]:
    contract = load_v3_contract()
    chosen = selection["selected"]
    results = []
    for seed in contract["random"]["model_seeds"]:
        result = _fit_trace_student(
            frame, inputs, teacher, float(chosen["alpha"]), float(chosen["residual_bound"]), int(seed),
            fit_roles=("fit", "validation"), fixed_epochs=int(chosen["best_epoch"]),
        )
        results.append(result)
        import torch
        torch.save(
            {
                "state_dict": result["state_dict"], "seed": int(seed),
                "alpha": float(chosen["alpha"]), "residual_bound": float(chosen["residual_bound"]),
                "phase_dim": int(inputs.rgb_delta.shape[2]),
            },
            V3_RUN_ROOT / "checkpoints" / f"final_trace_seed_{int(seed)}.pth",
        )
    predictions = np.stack([item["prediction"] for item in results])
    contributions = np.stack([item["contributions"] for item in results])
    residuals = np.stack([item["residual"] for item in results])
    sigmas = np.stack([item["sigma_judge"] for item in results])
    ablations = np.stack([item["ablated_predictions"] for item in results])
    kept = np.stack([item["kept_predictions"] for item in results])
    mean_prediction = predictions.mean(axis=0)
    mean_contribution = contributions.mean(axis=0)
    mean_residual = residuals.mean(axis=0)
    output = frame.copy()
    output["base_quality"] = inputs.base_quality
    for index, phase in enumerate(PHASES):
        output[f"phase_{phase}_contribution"] = mean_contribution[:, index]
        output[f"ablate_{phase}_quality"] = ablations.mean(axis=0)[:, index]
        output[f"keep_{phase}_quality"] = kept.mean(axis=0)[:, index]
    output["residual"] = mean_residual
    output["attribution_coverage"] = attribution_coverage(mean_contribution, mean_residual)
    output["predicted_quality"] = mean_prediction
    output["predicted_score"] = restore_total_score(mean_prediction, frame.difficulty)
    output["sigma_model"] = predictions.std(axis=0, ddof=0)
    output["sigma_judge"] = sigmas.mean(axis=0)
    output["reference_distance"] = inputs.reference_distance
    output["open_set"] = inputs.open_set
    output["reference_indices_json"] = [json.dumps(value) for value in inputs.references]
    output["phase_boundary_error_normalized"] = np.mean(inputs.predicted_phases != inputs.phase_targets, axis=1)
    calibration = frame.analysis_role.to_numpy() == "calibration"
    radius = split_conformal_radius(
        frame.execution_quality.to_numpy()[calibration], mean_prediction[calibration],
        float(contract["model"]["conformal_coverage"]),
    )
    output["lower_quality"] = mean_prediction - radius
    output["upper_quality"] = mean_prediction + radius
    for seed_index, seed in enumerate(contract["random"]["model_seeds"]):
        output[f"seed_{seed}_predicted_quality"] = predictions[seed_index]
        for phase_index, phase in enumerate(PHASES):
            output[f"seed_{seed}_{phase}_contribution"] = contributions[seed_index, :, phase_index]
    test = frame.official_split.to_numpy() == "test"
    summary = {
        "status": "TRAINED",
        "selected": chosen,
        "seeds": [int(seed) for seed in contract["random"]["model_seeds"]],
        "official_test_n": int(test.sum()),
        "official_test_score": score_metrics(
            frame.dive_score.to_numpy()[test], output.predicted_score.to_numpy()[test]
        ),
        "seed_official_test_scores": [
            score_metrics(
                frame.dive_score.to_numpy()[test],
                restore_total_score(predictions[index, test], frame.difficulty.to_numpy()[test]),
            )
            for index in range(len(predictions))
        ],
        "conformal_radius_quality": float(radius),
    }
    return output, summary


def train_final_trace_v3() -> dict:
    require_v3_frozen()
    frame = load_v3_frame()
    targets = load_teacher_targets_v3().set_index("clip_uid").loc[frame.clip_uid]
    inputs, preparation = prepare_inputs(
        frame, "none", allow_test_metrics=True, backbone="videomae", reference_roles=("fit", "validation")
    )
    selection = json.loads(
        (V3_RESULTS_ROOT / "02_PILOT" / "selected_config_v3.json").read_text(encoding="utf-8")
    )
    output, summary = _ensemble_predictions(
        frame, inputs, targets.teacher_final_quality.to_numpy(dtype=np.float32), selection
    )
    destination = V3_RESULTS_ROOT / "03_FINAL" / "predictions_trace_v3.parquet"
    output.to_parquet(destination, index=False)
    # One-seed event-family-isolated internal robustness analysis. The source
    # teacher and student see source_fit/source_validation labels only.
    source_inputs, source_preparation = prepare_inputs(
        frame,
        "none",
        allow_test_metrics=True,
        split_column="source_role",
        backbone="videomae",
        reference_roles=("source_fit",),
    )
    source_teacher = train_single_model(
        frame,
        source_inputs,
        "relative",
        int(load_v3_contract()["random"]["model_seeds"][0]),
        "none",
        fit_role="source_fit",
        validation_role="source_validation",
        role_column="source_role",
    )
    source_student = _fit_trace_student(
        frame,
        source_inputs,
        source_teacher["prediction"],
        float(selection["selected"]["alpha"]),
        float(selection["selected"]["residual_bound"]),
        int(load_v3_contract()["random"]["model_seeds"][0]),
        fit_roles=("source_fit", "source_validation"),
        validation_role="source_test",
        role_column="source_role",
        fixed_epochs=int(selection["selected"]["best_epoch"]),
    )
    source_mask = frame.source_role == "source_test"
    source_output = frame.loc[
        source_mask,
        ["clip_uid", "event_family", "action_type", "difficulty", "dive_score", "execution_quality"],
    ].copy()
    source_indices = np.flatnonzero(source_mask.to_numpy())
    source_output["predicted_quality"] = source_student["prediction"][source_indices]
    source_output["predicted_score"] = restore_total_score(
        source_output.predicted_quality, source_output.difficulty
    )
    source_output.to_parquet(
        V3_RESULTS_ROOT / "03_FINAL" / "source_isolated_predictions_v3.parquet", index=False
    )
    summary["preparation"] = preparation
    summary["source_isolated_preparation"] = source_preparation
    summary["source_isolated_score"] = score_metrics(
        source_output.dive_score, source_output.predicted_score
    )
    summary["destination"] = str(destination)
    write_json(V3_RESULTS_ROOT / "03_FINAL" / "final_training_summary_v3.json", summary)
    return summary
