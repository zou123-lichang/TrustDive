from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metrics import aqa_score_metrics, restore_total_score
from .statistics import split_conformal_radius
from .util import set_seed, write_json
from .v4_counterfactual import PHASES
from .v4_data import (
    V4_RESULTS_ROOT,
    V4_RUN_ROOT,
    load_v4_contract,
    load_v4_frame,
    require_v4_audit,
    require_v4_frozen,
)


@dataclass
class StudentInputs:
    pair_features: np.ndarray
    base_quality: np.ndarray
    phase_targets: np.ndarray
    teacher_quality: np.ndarray
    references: np.ndarray
    valid_reference_count: np.ndarray
    phase_labels: np.ndarray


def safe_spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else 0.0


def _pool_phases(sequence: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pooled = []
    for phase in range(3):
        selected = sequence[labels == phase]
        pooled.append(selected.mean(axis=0) if len(selected) else np.zeros(sequence.shape[-1]))
    return np.stack(pooled).astype(np.float32)


def prepare_student_inputs_v4(
    phase_labels_override: np.ndarray | None = None,
    sequences_override: np.ndarray | None = None,
    references_override: np.ndarray | None = None,
) -> StudentInputs:
    frame = load_v4_frame()
    target_path = V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "counterfactual_query_targets_v4.parquet"
    map_path = V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "reference_map_v4.npz"
    sequence_path = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_sequences_v4.npz"
    if not target_path.exists() or not map_path.exists() or not sequence_path.exists():
        raise RuntimeError("Counterfactual targets are incomplete")
    targets = pd.read_parquet(target_path).set_index("clip_uid").loc[frame.clip_uid]
    with np.load(map_path, allow_pickle=False) as payload:
        references = payload["references"].astype(np.int64)
        valid_reference_count = payload["valid_reference_count"].astype(np.int64)
        phase_labels = payload["phase_labels"].astype(np.int8)
    if references_override is not None:
        override = np.asarray(references_override, dtype=np.int64)
        if override.shape != references.shape:
            raise ValueError("Reference override has the wrong shape")
        if np.any((override < 0) | (override >= len(frame))):
            raise ValueError("Reference override contains an invalid sample index")
        references = override
    if phase_labels_override is not None:
        override = np.asarray(phase_labels_override, dtype=np.int8)
        if override.shape != phase_labels.shape:
            raise ValueError("Phase-label override has the wrong shape")
        phase_labels = override
    with np.load(sequence_path, allow_pickle=False) as payload:
        sequences = payload["sequence"].astype(np.float32)
    if sequences_override is not None:
        override = np.asarray(sequences_override, dtype=np.float32)
        if override.shape != sequences.shape:
            raise ValueError("Sequence override has the wrong shape")
        sequences = override
    pooled = np.stack([_pool_phases(sequence, labels) for sequence, labels in zip(sequences, phase_labels)])
    fit = np.flatnonzero(frame.analysis_role.to_numpy() == "fit")
    fit_values = pooled[fit].reshape(-1, pooled.shape[-1])
    center = np.median(fit_values, axis=0)
    scale = 1.4826 * np.median(np.abs(fit_values - center), axis=0)
    fallback = fit_values.std(axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(fallback > 1e-6, fallback, 1.0))
    pooled = np.clip((pooled - center) / scale, -12.0, 12.0).astype(np.float32)
    query = np.repeat(pooled[:, None, :, :], references.shape[1], axis=1)
    reference = pooled[references]
    pair = np.concatenate((query, reference, query - reference, np.abs(query - reference)), axis=-1)
    phase_targets = targets[[f"teacher_phi_{phase}" for phase in PHASES]].to_numpy(dtype=np.float32)
    base_quality = targets.base_quality.to_numpy(dtype=np.float32)
    if references_override is not None:
        base_quality = np.median(frame.execution_quality.to_numpy(dtype=np.float32)[references], axis=1)
    return StudentInputs(
        pair_features=pair.astype(np.float16),
        base_quality=base_quality,
        phase_targets=phase_targets,
        teacher_quality=targets.teacher_predicted_quality.to_numpy(dtype=np.float32),
        references=references,
        valid_reference_count=valid_reference_count,
        phase_labels=phase_labels,
    )


def cfpd_model_class(input_dim: int, hidden: int, dropout: float):
    import torch
    import torch.nn as nn

    class CFPDStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.phase_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(3)])

        def forward(self, pair_features, base_quality):
            encoded = self.shared(pair_features)
            per_reference = torch.stack(
                [self.phase_heads[index](encoded[:, :, index]).squeeze(-1) for index in range(3)],
                dim=-1,
            )
            contributions = per_reference.median(dim=1).values
            prediction = base_quality + contributions.sum(dim=1)
            return prediction, contributions, per_reference

    return CFPDStudent


def _ranking_loss(prediction, target):
    import torch
    import torch.nn.functional as F

    if len(prediction) < 2:
        return prediction.new_tensor(0.0)
    order = torch.argsort(target)
    pred_sorted = prediction[order]
    target_sorted = target[order]
    sign = torch.sign(target_sorted[1:] - target_sorted[:-1])
    valid = sign != 0
    if not bool(valid.any()):
        return prediction.new_tensor(0.0)
    return F.softplus(-sign[valid] * (pred_sorted[1:][valid] - pred_sorted[:-1][valid])).mean()


def _jitter_phase_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    output = np.zeros_like(labels, dtype=np.int8)
    rng = np.random.default_rng(seed)
    length = labels.shape[1]
    for index, row in enumerate(labels):
        first = int(np.flatnonzero(row == 1)[0])
        second = int(np.flatnonzero(row == 2)[0])
        first = int(np.clip(first + rng.integers(-1, 2), 1, length - 2))
        second = int(np.clip(second + rng.integers(-1, 2), first + 1, length - 1))
        output[index, first:second] = 1
        output[index, second:] = 2
    return output


def _pilot_augmented_inputs(base: StudentInputs, mode: str, seed: int) -> StudentInputs:
    labels = _jitter_phase_labels(base.phase_labels, seed)
    if mode == "boundary":
        return prepare_student_inputs_v4(phase_labels_override=labels)
    if mode in {"boundary_token", "boundary_token_balanced"}:
        with np.load(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_sequences_v4.npz", allow_pickle=False) as payload:
            sequences = payload["sequence"].astype(np.float32)
        rng = np.random.default_rng(seed + 101)
        for index in range(len(sequences)):
            selected = rng.choice(sequences.shape[1], size=max(1, int(round(0.10 * sequences.shape[1]))), replace=False)
            sequences[index, selected] = 0.0
        return prepare_student_inputs_v4(phase_labels_override=labels, sequences_override=sequences)
    raise ValueError(f"Unknown v4 pilot augmentation: {mode}")


def _fit_student(
    frame: pd.DataFrame,
    inputs: StudentInputs,
    phase_weight: float,
    ranking_weight: float,
    seed: int,
    fit_roles: tuple[str, ...] = ("fit",),
    validation_role: str = "validation",
    role_column: str = "analysis_role",
    fixed_epochs: int | None = None,
    training_inputs: StudentInputs | None = None,
    balanced_sampling: bool = False,
) -> dict:
    import torch
    import torch.nn.functional as F

    contract = load_v4_contract()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_indices = np.flatnonzero(frame[role_column].isin(fit_roles).to_numpy())
    validation_indices = np.flatnonzero(frame[role_column].to_numpy() == validation_role)
    if not len(fit_indices):
        raise ValueError("CFPD requires a nonempty fit partition")
    Model = cfpd_model_class(
        inputs.pair_features.shape[-1], int(contract["model"]["hidden_dimension"]), float(contract["model"]["dropout"])
    )
    model = Model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract["model"]["learning_rate"]),
        weight_decay=float(contract["model"]["weight_decay"]),
    )
    true_quality = frame.execution_quality.to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    optimization_inputs = training_inputs or inputs
    sampling_probability = None
    if balanced_sampling:
        score_quartile = pd.qcut(
            frame.execution_quality.rank(method="first"), 4, labels=False, duplicates="drop"
        ).astype(str)
        strata = frame.action_type.astype(str) + "::" + score_quartile
        frequency = strata.iloc[fit_indices].value_counts()
        weight = strata.iloc[fit_indices].map(lambda value: 1.0 / frequency[value]).to_numpy(dtype=float)
        sampling_probability = weight / weight.sum()
    best_state = None
    best_epoch = 0
    best_spearman = -np.inf
    best_loss = np.inf
    stale = 0
    maximum = fixed_epochs or int(contract["model"]["maximum_epochs"])
    batch_size = 128
    for epoch in range(maximum):
        model.train()
        if sampling_probability is None:
            shuffled = rng.permutation(fit_indices)
        else:
            shuffled = rng.choice(
                fit_indices, size=len(fit_indices), replace=True, p=sampling_probability
            )
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            pair = torch.from_numpy(optimization_inputs.pair_features[indices].astype(np.float32)).to(device)
            base = torch.from_numpy(optimization_inputs.base_quality[indices]).to(device)
            true = torch.from_numpy(true_quality[indices]).to(device)
            teacher = torch.from_numpy(optimization_inputs.teacher_quality[indices]).to(device)
            phi = torch.from_numpy(optimization_inputs.phase_targets[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, contributions, _ = model(pair, base)
            loss = F.huber_loss(prediction, true, delta=1.0)
            loss = loss + float(contract["model"]["teacher_loss_weight"]) * F.huber_loss(prediction, teacher, delta=1.0)
            loss = loss + float(phase_weight) * F.huber_loss(contributions, phi, delta=0.5)
            if ranking_weight > 0:
                loss = loss + float(ranking_weight) * _ranking_loss(prediction, true)
            loss.backward()
            optimizer.step()

        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if fixed_epochs is not None:
            best_state = state
            best_epoch = epoch + 1
            continue
        model.eval()
        with torch.inference_mode():
            prediction, _, _ = _predict_indices(model, inputs, validation_indices, device)
        validation_true = true_quality[validation_indices]
        validation_loss = float(np.mean(np.abs(prediction - validation_true)))
        validation_spearman = safe_spearman(prediction, validation_true)
        if validation_spearman > best_spearman + 1e-6 or (
            abs(validation_spearman - best_spearman) <= 1e-6 and validation_loss < best_loss
        ):
            best_state = state
            best_epoch = epoch + 1
            best_spearman = validation_spearman
            best_loss = validation_loss
            stale = 0
        else:
            stale += 1
            if stale >= int(contract["model"]["patience"]):
                break
    if best_state is None:
        raise RuntimeError("CFPD training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    all_indices = np.arange(len(frame))
    with torch.inference_mode():
        prediction, contributions, per_reference = _predict_indices(model, inputs, all_indices, device)
    return {
        "state_dict": best_state,
        "prediction": prediction,
        "contributions": contributions,
        "per_reference": per_reference,
        "best_epoch": int(best_epoch),
        "validation_spearman": float(best_spearman) if fixed_epochs is None else None,
        "validation_mae": float(best_loss) if fixed_epochs is None else None,
    }


def _predict_indices(model, inputs: StudentInputs, indices: np.ndarray, device):
    import torch

    predictions = []
    contributions = []
    per_reference = []
    for start in range(0, len(indices), 256):
        selected = indices[start : start + 256]
        pair = torch.from_numpy(inputs.pair_features[selected].astype(np.float32)).to(device)
        base = torch.from_numpy(inputs.base_quality[selected]).to(device)
        prediction, contribution, per_ref = model(pair, base)
        predictions.append(prediction.detach().cpu().numpy())
        contributions.append(contribution.detach().cpu().numpy())
        per_reference.append(per_ref.detach().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(contributions), np.concatenate(per_reference)


def pilot_cfpd_v4() -> dict:
    require_v4_audit()
    summary_path = V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "counterfactual_summary_v4.json"
    if not summary_path.exists() or json.loads(summary_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Counterfactual gate must pass before the CFPD pilot")
    frame = load_v4_frame()
    inputs = prepare_student_inputs_v4()
    contract = load_v4_contract()
    validation = frame.analysis_role.to_numpy() == "validation"
    records = []
    for phase_weight in contract["model"]["phase_loss_weights"]:
        for ranking_weight in contract["model"]["ranking_loss_options"]:
            result = _fit_student(
                frame, inputs, float(phase_weight), float(ranking_weight), int(contract["model"]["pilot_seed"])
            )
            phase_rho = safe_spearman(
                result["contributions"][validation].reshape(-1), inputs.phase_targets[validation].reshape(-1)
            )
            record = {
                "phase_loss_weight": float(phase_weight),
                "ranking_loss_weight": float(ranking_weight),
                "validation_spearman": float(result["validation_spearman"]),
                "validation_mae_quality": float(result["validation_mae"]),
                "validation_phase_teacher_spearman": phase_rho,
                "best_epoch": int(result["best_epoch"]),
                "augmentation": "none",
                "eligible": bool(result["validation_spearman"] >= float(contract["model"]["minimum_validation_spearman"])),
            }
            records.append(record)
    trials = pd.DataFrame(records)
    eligible = trials.loc[trials.eligible].sort_values(
        ["validation_spearman", "validation_phase_teacher_spearman", "phase_loss_weight"],
        ascending=[False, False, False],
    )
    if eligible.empty:
        best_base = trials.sort_values(
            ["validation_spearman", "validation_phase_teacher_spearman"], ascending=False
        ).iloc[0]
        for mode in ("boundary", "boundary_token", "boundary_token_balanced"):
            augmented = _pilot_augmented_inputs(inputs, mode, int(contract["model"]["pilot_seed"]))
            result = _fit_student(
                frame,
                inputs,
                float(best_base.phase_loss_weight),
                float(best_base.ranking_loss_weight),
                int(contract["model"]["pilot_seed"]),
                training_inputs=augmented,
                balanced_sampling=mode == "boundary_token_balanced",
            )
            phase_rho = safe_spearman(
                result["contributions"][validation].reshape(-1), inputs.phase_targets[validation].reshape(-1)
            )
            records.append(
                {
                    "phase_loss_weight": float(best_base.phase_loss_weight),
                    "ranking_loss_weight": float(best_base.ranking_loss_weight),
                    "validation_spearman": float(result["validation_spearman"]),
                    "validation_mae_quality": float(result["validation_mae"]),
                    "validation_phase_teacher_spearman": phase_rho,
                    "best_epoch": int(result["best_epoch"]),
                    "augmentation": mode,
                    "eligible": bool(
                        result["validation_spearman"] >= float(contract["model"]["minimum_validation_spearman"])
                    ),
                }
            )
        trials = pd.DataFrame(records)
        eligible = trials.loc[trials.eligible].sort_values(
            ["validation_spearman", "validation_phase_teacher_spearman", "phase_loss_weight"],
            ascending=[False, False, False],
        )
    trials.to_csv(V4_RESULTS_ROOT / "03_PILOT" / "pilot_trials_v4.csv", index=False)
    if eligible.empty:
        selection = {
            "status": "STOP",
            "reason": "No CFPD candidate reached the locked validation Spearman gate",
            "official_student_test_metrics_revealed": False,
            "candidates": records,
        }
    else:
        selection = {
            "status": "PASS",
            "selected": eligible.iloc[0].to_dict(),
            "selection_rule": "validation Spearman, then teacher-phase fidelity, then stronger phase supervision",
            "official_student_test_metrics_revealed": False,
        }
    write_json(V4_RESULTS_ROOT / "03_PILOT" / "selected_config_v4.json", selection)
    return selection


def train_final_cfpd_v4() -> dict:
    require_v4_frozen()
    frame = load_v4_frame()
    inputs = prepare_student_inputs_v4()
    contract = load_v4_contract()
    selection = json.loads((V4_RESULTS_ROOT / "03_PILOT" / "selected_config_v4.json").read_text(encoding="utf-8"))
    chosen = selection["selected"]
    results = []
    for seed in contract["model"]["final_seeds"]:
        result = _fit_student(
            frame,
            inputs,
            float(chosen["phase_loss_weight"]),
            float(chosen["ranking_loss_weight"]),
            int(seed),
            fit_roles=("fit", "validation"),
            fixed_epochs=int(chosen["best_epoch"]),
            training_inputs=(
                _pilot_augmented_inputs(inputs, str(chosen.get("augmentation", "none")), int(seed))
                if str(chosen.get("augmentation", "none")) != "none"
                else None
            ),
            balanced_sampling=str(chosen.get("augmentation", "none")) == "boundary_token_balanced",
        )
        results.append(result)
        import torch

        torch.save(
            {
                "state_dict": result["state_dict"],
                "input_dim": int(inputs.pair_features.shape[-1]),
                "hidden": int(contract["model"]["hidden_dimension"]),
                "dropout": float(contract["model"]["dropout"]),
                "seed": int(seed),
                "selected": chosen,
            },
            V4_RUN_ROOT / "checkpoints" / f"cfpd_seed_{int(seed)}.pth",
        )
    predictions = np.stack([item["prediction"] for item in results])
    contributions = np.stack([item["contributions"] for item in results])
    per_reference = np.stack([item["per_reference"] for item in results])
    mean_prediction = predictions.mean(axis=0)
    mean_contribution = contributions.mean(axis=0)
    output = frame.copy()
    output["base_quality"] = inputs.base_quality
    output["teacher_predicted_quality"] = inputs.teacher_quality
    output["predicted_quality"] = mean_prediction
    output["predicted_score"] = restore_total_score(mean_prediction, frame.difficulty)
    output["sigma_model_internal"] = predictions.std(axis=0, ddof=0)
    output["teacher_student_abs_difference"] = np.abs(mean_prediction - inputs.teacher_quality)
    output["valid_reference_count"] = inputs.valid_reference_count
    output["open_set"] = inputs.valid_reference_count < int(contract["data"]["minimum_valid_references"])
    targets = pd.read_parquet(V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "counterfactual_query_targets_v4.parquet").set_index("clip_uid").loc[frame.clip_uid]
    output["reference_distance"] = targets.reference_distance.to_numpy()
    output["reference_dispersion"] = targets.reference_dispersion.to_numpy()
    for phase_index, phase in enumerate(PHASES):
        output[f"phase_{phase}_contribution"] = mean_contribution[:, phase_index]
        output[f"teacher_phi_{phase}"] = inputs.phase_targets[:, phase_index]
        output[f"reference_{phase}_dispersion"] = per_reference.mean(axis=0)[:, :, phase_index].std(axis=1)
    calibration = frame.analysis_role.to_numpy() == "calibration"
    radius = split_conformal_radius(
        frame.execution_quality.to_numpy()[calibration], mean_prediction[calibration], float(contract["model"]["conformal_coverage"])
    )
    output["lower_quality"] = mean_prediction - radius
    output["upper_quality"] = mean_prediction + radius
    for seed_index, seed in enumerate(contract["model"]["final_seeds"]):
        output[f"seed_{seed}_predicted_quality"] = predictions[seed_index]
    destination = V4_RESULTS_ROOT / "04_FINAL" / "predictions_v4.parquet"
    output.to_parquet(destination, index=False)
    test = frame.official_split.to_numpy() == "test"
    metrics = aqa_score_metrics(frame.dive_score.to_numpy()[test], output.predicted_score.to_numpy()[test])
    seed_metrics = [
        aqa_score_metrics(
            frame.dive_score.to_numpy()[test],
            restore_total_score(predictions[index, test], frame.difficulty.to_numpy()[test]),
        )
        for index in range(len(results))
    ]
    teacher_metrics = aqa_score_metrics(
        frame.dive_score.to_numpy()[test],
        restore_total_score(inputs.teacher_quality[test], frame.difficulty.to_numpy()[test]),
    )
    ablation_results = []
    for seed in contract["model"]["final_seeds"]:
        ablation_results.append(
            _fit_student(
                frame,
                inputs,
                0.0,
                float(chosen["ranking_loss_weight"]),
                int(seed),
                fit_roles=("fit", "validation"),
                fixed_epochs=int(chosen["best_epoch"]),
                training_inputs=(
                    _pilot_augmented_inputs(inputs, str(chosen.get("augmentation", "none")), int(seed))
                    if str(chosen.get("augmentation", "none")) != "none"
                    else None
                ),
                balanced_sampling=str(chosen.get("augmentation", "none")) == "boundary_token_balanced",
            )
        )
    ablation_predictions = np.stack([item["prediction"] for item in ablation_results])
    ablation_score = restore_total_score(ablation_predictions.mean(axis=0), frame.difficulty)
    ablation_metrics = aqa_score_metrics(frame.dive_score.to_numpy()[test], ablation_score[test])
    ablation_seed_metrics = [
        aqa_score_metrics(
            frame.dive_score.to_numpy()[test],
            restore_total_score(item[test], frame.difficulty.to_numpy()[test]),
        )
        for item in ablation_predictions
    ]
    result = {
        "status": "TRAINED",
        "official_test_metrics": metrics,
        "teacher_official_test_metrics": teacher_metrics,
        "spearman_delta_teacher": float(metrics["spearman"] - teacher_metrics["spearman"]),
        "seed_metrics": seed_metrics,
        "seed_spearman_sd": float(np.std([item["spearman"] for item in seed_metrics], ddof=0)),
        "no_shapley_ablation_metrics": ablation_metrics,
        "no_shapley_ablation_seed_metrics": ablation_seed_metrics,
        "conformal_radius_quality": float(radius),
        "selected": chosen,
        "destination": str(destination),
    }
    write_json(V4_RESULTS_ROOT / "04_FINAL" / "final_training_summary_v4.json", result)
    return result
