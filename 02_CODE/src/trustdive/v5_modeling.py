from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metrics import aqa_score_metrics
from .util import sha256_file, write_json
from .v4_counterfactual import PHASES
from .v5_data import (
    V5_RESULTS_ROOT,
    V5_RUN_ROOT,
    load_v5_contract,
    load_v5_frame,
    require_v5_frozen,
)


@dataclass
class V5Inputs:
    pair_features: np.ndarray
    base_quality: np.ndarray
    teacher_quality: np.ndarray
    phase_targets: np.ndarray
    phase_reliability: np.ndarray
    references: np.ndarray
    valid_reference_count: np.ndarray
    phase_labels: np.ndarray


def safe_spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _pool_phases(sequence: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pooled = np.empty((len(sequence), 3, sequence.shape[-1]), dtype=np.float32)
    for index in range(len(sequence)):
        for phase in range(3):
            selected = sequence[index, labels[index] == phase]
            pooled[index, phase] = selected.mean(axis=0) if len(selected) else 0.0
    return pooled


def _load_core_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(
        V5_RESULTS_ROOT / "01_REFERENCES" / "teacher_sequences_v5.npz", allow_pickle=False
    ) as payload:
        sequences = payload["sequence"].astype(np.float32)
    with np.load(
        V5_RESULTS_ROOT / "01_REFERENCES" / "reference_map_v5.npz", allow_pickle=False
    ) as payload:
        references = payload["references"].astype(np.int64)
        valid_count = payload["valid_reference_count"].astype(np.int64)
        labels = payload["phase_labels"].astype(np.int8)
    return sequences, references, valid_count, labels


def prepare_inputs_v5(
    sequences_override: np.ndarray | None = None,
    labels_override: np.ndarray | None = None,
    references_override: np.ndarray | None = None,
) -> V5Inputs:
    core_sequences, references, valid_count, core_labels = _load_core_arrays()
    sequences = core_sequences
    labels = core_labels
    if sequences_override is not None:
        sequences = np.asarray(sequences_override, dtype=np.float32)
    if labels_override is not None:
        labels = np.asarray(labels_override, dtype=np.int8)
    if references_override is not None:
        references = np.asarray(references_override, dtype=np.int64)
    frame = load_v5_frame()
    target_path = (
        V5_RESULTS_ROOT / "03_COUNTERFACTUAL" / "counterfactual_query_targets_v5.parquet"
    )
    if target_path.exists():
        targets = pd.read_parquet(target_path).set_index("clip_uid").loc[frame.clip_uid]
        phase_targets = targets[[f"teacher_phi_{phase}" for phase in PHASES]].to_numpy(
            dtype=np.float32
        )
        reliability = targets[[f"phase_reliability_{phase}" for phase in PHASES]].to_numpy(
            dtype=np.float32
        )
        base_quality = targets.base_quality.to_numpy(dtype=np.float32)
        teacher_quality = targets.teacher_predicted_quality.to_numpy(dtype=np.float32)
    else:
        references_frame = pd.read_parquet(
            V5_RESULTS_ROOT / "01_REFERENCES" / "reference_targets_v5.parquet"
        ).set_index("clip_uid").loc[frame.clip_uid]
        phase_targets = np.zeros((len(frame), 3), dtype=np.float32)
        reliability = np.ones_like(phase_targets)
        base_quality = references_frame.base_quality.to_numpy(dtype=np.float32)
        teacher_quality = references_frame.teacher_predicted_quality.to_numpy(dtype=np.float32)
    if references_override is not None:
        base_quality = np.median(
            frame.execution_quality.to_numpy(dtype=np.float32)[references], axis=1
        )
    # The frozen I3D channels have very different numerical scales.  Estimate
    # one phase-specific z-score transform from fit rows only, then reuse it
    # unchanged for validation, calibration, test, and all stress variants.
    fit = frame.analysis_role.to_numpy() == "fit"
    core_pooled = _pool_phases(core_sequences, core_labels)
    center = core_pooled[fit].mean(axis=0, keepdims=True)
    scale = core_pooled[fit].std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    pooled = (_pool_phases(sequences, labels) - center) / scale
    query = np.repeat(pooled[:, None, :, :], references.shape[1], axis=1)
    reference = pooled[references]
    pair = np.concatenate(
        (query, reference, query - reference, np.abs(query - reference)), axis=-1
    ).astype(np.float16)
    return V5Inputs(
        pair_features=pair,
        base_quality=base_quality,
        teacher_quality=teacher_quality,
        phase_targets=phase_targets,
        phase_reliability=reliability,
        references=references,
        valid_reference_count=valid_count,
        phase_labels=labels,
    )


def cfpd_plus_model_class(input_dim: int, hidden: int, dropout: float):
    import torch
    import torch.nn as nn

    class CFPDPlus(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.phase_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(3)])

        def forward(self, pair_features, base_quality):
            encoded = self.shared(pair_features)
            per_reference = torch.stack(
                [
                    self.phase_heads[index](encoded[:, :, index]).squeeze(-1)
                    for index in range(3)
                ],
                dim=-1,
            )
            # Five references are fixed by contract. Sorting and taking the
            # middle element is the exact median and avoids CUDA's warned
            # non-deterministic median-with-indices backward implementation.
            contributions = torch.sort(per_reference, dim=1).values[
                :, per_reference.shape[1] // 2, :
            ]
            prediction = base_quality + contributions.sum(dim=1)
            return prediction, contributions, per_reference

    return CFPDPlus


def _ranking_loss_by_action(prediction, target, actions: np.ndarray, indices: np.ndarray):
    import torch
    import torch.nn.functional as F

    losses = []
    batch_actions = actions[indices]
    for action in np.unique(batch_actions):
        local = np.flatnonzero(batch_actions == action)
        if len(local) < 2:
            continue
        local_tensor = torch.as_tensor(local, dtype=torch.long, device=prediction.device)
        local_target = target[local_tensor]
        order = torch.argsort(local_target)
        pred_sorted = prediction[local_tensor][order]
        target_sorted = local_target[order]
        differences = target_sorted[1:] - target_sorted[:-1]
        valid = differences != 0
        if bool(valid.any()):
            losses.append(F.softplus(-(pred_sorted[1:] - pred_sorted[:-1])[valid]).mean())
    return torch.stack(losses).mean() if losses else prediction.new_tensor(0.0)


def _jitter_labels(labels: np.ndarray, seed: int) -> np.ndarray:
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


def augmented_inputs_v5(mode: str, seed: int) -> V5Inputs | None:
    if mode in {"none", "balanced"}:
        return None
    sequences, _, _, labels = _load_core_arrays()
    rng = np.random.default_rng(seed)
    labels = _jitter_labels(labels, seed)
    if mode in {"boundary_token", "boundary_token_noise"}:
        count = max(1, int(round(sequences.shape[1] * 0.10)))
        for index in range(len(sequences)):
            selected = rng.choice(sequences.shape[1], size=count, replace=False)
            sequences[index, selected] = 0.0
    if mode == "boundary_token_noise":
        frame = load_v5_frame()
        fit = frame.analysis_role.to_numpy() == "fit"
        robust_scale = np.median(
            np.abs(sequences[fit] - np.median(sequences[fit], axis=(0, 1), keepdims=True)),
            axis=(0, 1),
        )
        robust_scale = np.where(robust_scale > 1e-6, 1.4826 * robust_scale, 1.0)
        noise = rng.normal(size=sequences.shape).astype(np.float32)
        sequences = sequences + 0.02 * noise * robust_scale[None, None, :]
    return prepare_inputs_v5(sequences_override=sequences, labels_override=labels)


def _predict(model, inputs: V5Inputs, indices: np.ndarray, device, batch_size: int):
    import torch

    predictions = []
    contributions = []
    per_reference = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            pair = torch.from_numpy(inputs.pair_features[selected].astype(np.float32)).to(device)
            base = torch.from_numpy(inputs.base_quality[selected]).to(device)
            prediction, contribution, per_ref = model(pair, base)
            predictions.append(prediction.detach().cpu().numpy())
            contributions.append(contribution.detach().cpu().numpy())
            per_reference.append(per_ref.detach().cpu().numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(contributions),
        np.concatenate(per_reference),
    )


def _balanced_probabilities(frame: pd.DataFrame, fit_indices: np.ndarray) -> np.ndarray:
    quartile = pd.qcut(
        frame.execution_quality.rank(method="first"), 4, labels=False, duplicates="drop"
    ).astype(str)
    strata = frame.action_type.astype(str) + "::" + quartile
    frequency = strata.iloc[fit_indices].value_counts()
    weights = strata.iloc[fit_indices].map(lambda value: 1.0 / frequency[value]).to_numpy(float)
    return weights / weights.sum()


def fit_v5_model(
    inputs: V5Inputs,
    hidden: int,
    learning_rate: float,
    ranking_weight: float,
    phase_weight: float,
    consistency_weight: float,
    reliability_weighted: bool,
    augmentation_mode: str,
    seed: int,
    fit_roles: tuple[str, ...] = ("fit",),
    validation_role: str = "validation",
    fixed_epochs: int | None = None,
) -> dict:
    import torch
    import torch.nn.functional as F

    contract = load_v5_contract()
    frame = load_v5_frame()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_indices = np.flatnonzero(frame.analysis_role.isin(fit_roles).to_numpy())
    validation_indices = np.flatnonzero(frame.analysis_role.to_numpy() == validation_role)
    Model = cfpd_plus_model_class(
        inputs.pair_features.shape[-1], hidden, float(contract["model"]["dropout"])
    )
    model = Model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(contract["model"]["weight_decay"]),
    )
    true_quality = frame.execution_quality.to_numpy(dtype=np.float32)
    actions = frame.action_type.astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    consistency_mode = (
        "boundary_token_noise"
        if consistency_weight > 0 and augmentation_mode in {"none", "balanced"}
        else augmentation_mode
    )
    augmented = augmented_inputs_v5(consistency_mode, seed)
    balanced = augmentation_mode != "none"
    sampling = _balanced_probabilities(frame, fit_indices) if balanced else None
    best_state = None
    best_epoch = 0
    best_spearman = -np.inf
    best_mae = np.inf
    stale = 0
    maximum = fixed_epochs or int(contract["model"]["maximum_epochs"])
    batch_size = int(contract["model"]["batch_size"])
    for epoch in range(maximum):
        model.train()
        if sampling is None:
            shuffled = rng.permutation(fit_indices)
        else:
            shuffled = rng.choice(fit_indices, size=len(fit_indices), replace=True, p=sampling)
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            pair = torch.from_numpy(inputs.pair_features[indices].astype(np.float32)).to(device)
            base = torch.from_numpy(inputs.base_quality[indices]).to(device)
            true = torch.from_numpy(true_quality[indices]).to(device)
            teacher = torch.from_numpy(inputs.teacher_quality[indices]).to(device)
            target_phi = torch.from_numpy(inputs.phase_targets[indices]).to(device)
            reliability = torch.from_numpy(inputs.phase_reliability[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, contribution, _ = model(pair, base)
            loss = F.huber_loss(prediction, true, delta=1.0)
            loss = loss + float(contract["model"]["teacher_loss_weight"]) * F.huber_loss(
                prediction, teacher, delta=1.0
            )
            loss = loss + float(ranking_weight) * _ranking_loss_by_action(
                prediction, true, actions, indices
            )
            if phase_weight > 0:
                phase_loss = F.huber_loss(
                    contribution, target_phi, delta=0.5, reduction="none"
                )
                if reliability_weighted:
                    phase_loss = phase_loss * reliability
                loss = loss + float(phase_weight) * phase_loss.mean()
            if consistency_weight > 0 and augmented is not None:
                aug_pair = torch.from_numpy(
                    augmented.pair_features[indices].astype(np.float32)
                ).to(device)
                aug_base = torch.from_numpy(augmented.base_quality[indices]).to(device)
                aug_prediction, aug_contribution, _ = model(aug_pair, aug_base)
                consistency = F.smooth_l1_loss(
                    aug_prediction, prediction.detach(), beta=0.25
                )
                consistency = consistency + F.smooth_l1_loss(
                    aug_contribution, contribution.detach(), beta=0.10
                )
                loss = loss + float(consistency_weight) * consistency
            loss.backward()
            optimizer.step()

        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if fixed_epochs is not None:
            best_state = state
            best_epoch = epoch + 1
            continue
        validation_prediction, validation_contribution, _ = _predict(
            model, inputs, validation_indices, device, batch_size
        )
        rho = safe_spearman(validation_prediction, true_quality[validation_indices])
        mae = float(np.mean(np.abs(validation_prediction - true_quality[validation_indices])))
        if rho > best_spearman + 1e-6 or (
            abs(rho - best_spearman) <= 1e-6 and mae < best_mae
        ):
            best_state = state
            best_epoch = epoch + 1
            best_spearman = rho
            best_mae = mae
            stale = 0
        else:
            stale += 1
            if stale >= int(contract["model"]["patience"]):
                break
    if best_state is None:
        raise RuntimeError("V5 training produced no checkpoint")
    model.load_state_dict(best_state)
    all_indices = np.arange(len(frame))
    prediction, contribution, per_reference = _predict(
        model, inputs, all_indices, device, batch_size
    )
    validation_trace = safe_spearman(
        contribution[validation_indices].reshape(-1),
        inputs.phase_targets[validation_indices].reshape(-1),
    )
    return {
        "state_dict": best_state,
        "prediction": prediction,
        "contribution": contribution,
        "per_reference": per_reference,
        "best_epoch": best_epoch,
        "validation_spearman": safe_spearman(
            prediction[validation_indices], true_quality[validation_indices]
        ),
        "validation_mae": float(
            np.mean(np.abs(prediction[validation_indices] - true_quality[validation_indices]))
        ),
        "validation_trace_spearman": validation_trace,
    }


def _save_checkpoint(result: dict, path: Path, metadata: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": result["state_dict"], "metadata": metadata}, path)


def optimize_baseline_v5() -> dict:
    reference_summary = V5_RESULTS_ROOT / "01_REFERENCES" / "reference_summary_v5.json"
    if not reference_summary.exists() or json.loads(reference_summary.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Run build-references --protocol v5 first")
    contract = load_v5_contract()
    history_root = V5_RESULTS_ROOT / "02_BASELINE" / "history"
    previous_selected = V5_RESULTS_ROOT / "02_BASELINE" / "selected_baseline_v5.json"
    previous_trials = V5_RESULTS_ROOT / "02_BASELINE" / "baseline_trials_v5.csv"
    if previous_selected.exists() or previous_trials.exists():
        attempt = 1
        while (history_root / f"attempt_{attempt:03d}").exists():
            attempt += 1
        archive = history_root / f"attempt_{attempt:03d}"
        archive.mkdir(parents=True, exist_ok=False)
        for source in (previous_selected, previous_trials):
            if source.exists():
                shutil.copy2(source, archive / source.name)
        write_json(
            archive / "archive_note.json",
            {
                "reason": "Superseded after fit-only phase z-score and shared LayerNorm preprocessing correction",
                "negative_result_preserved": True,
            },
        )
    inputs = prepare_inputs_v5()
    rows = []
    results = []
    seed = int(contract["model"]["pilot_seed"])
    for hidden in contract["model"]["hidden_dimensions"]:
        for ranking in contract["model"]["ranking_weights"]:
            for learning_rate in contract["model"]["learning_rates"]:
                result = fit_v5_model(
                    inputs,
                    int(hidden),
                    float(learning_rate),
                    float(ranking),
                    0.0,
                    0.0,
                    False,
                    "none",
                    seed,
                )
                row = {
                    "hidden": int(hidden),
                    "ranking_weight": float(ranking),
                    "learning_rate": float(learning_rate),
                    "augmentation_mode": "none",
                    "validation_spearman": result["validation_spearman"],
                    "validation_mae": result["validation_mae"],
                    "best_epoch": result["best_epoch"],
                }
                rows.append(row)
                results.append((row, result))
                print(f"V5 baseline grid: {row}", flush=True)
    best_row, best_result = max(
        results, key=lambda item: (item[0]["validation_spearman"], -item[0]["validation_mae"])
    )
    if best_row["validation_spearman"] < float(contract["model"]["rescue_validation_spearman"]):
        for mode in contract["model"]["augmentation_ladder"][1:]:
            result = fit_v5_model(
                inputs,
                int(best_row["hidden"]),
                float(best_row["learning_rate"]),
                float(best_row["ranking_weight"]),
                0.0,
                0.0,
                False,
                str(mode),
                seed,
            )
            row = {
                "hidden": int(best_row["hidden"]),
                "ranking_weight": float(best_row["ranking_weight"]),
                "learning_rate": float(best_row["learning_rate"]),
                "augmentation_mode": str(mode),
                "validation_spearman": result["validation_spearman"],
                "validation_mae": result["validation_mae"],
                "best_epoch": result["best_epoch"],
            }
            rows.append(row)
            results.append((row, result))
            print(f"V5 baseline rescue: {row}", flush=True)
    best_row, best_result = max(
        results, key=lambda item: (item[0]["validation_spearman"], -item[0]["validation_mae"])
    )
    trials = pd.DataFrame(rows)
    trials_path = V5_RESULTS_ROOT / "02_BASELINE" / "baseline_trials_v5.csv"
    trials.to_csv(trials_path, index=False)
    checkpoint_path = V5_RUN_ROOT / "checkpoints" / "selected_baseline_v5.pt"
    _save_checkpoint(best_result, checkpoint_path, best_row)
    selected = {
        "status": "PASS",
        **best_row,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trials_sha256": sha256_file(trials_path),
        "official_test_labels_used": False,
    }
    write_json(V5_RESULTS_ROOT / "02_BASELINE" / "selected_baseline_v5.json", selected)
    return selected


def pilot_cfpd_plus_v5() -> dict:
    counter = V5_RESULTS_ROOT / "03_COUNTERFACTUAL" / "counterfactual_summary_v5.json"
    if not counter.exists() or json.loads(counter.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Run build-counterfactuals --protocol v5 first")
    contract = load_v5_contract()
    baseline = json.loads(
        (V5_RESULTS_ROOT / "02_BASELINE" / "selected_baseline_v5.json").read_text(
            encoding="utf-8"
        )
    )
    inputs = prepare_inputs_v5()
    rows = []
    results = []
    for phase_weight in contract["model"]["phase_loss_weights"]:
        for consistency_weight in contract["model"]["consistency_weights"]:
            result = fit_v5_model(
                inputs,
                int(baseline["hidden"]),
                float(baseline["learning_rate"]),
                float(baseline["ranking_weight"]),
                float(phase_weight),
                float(consistency_weight),
                True,
                str(baseline["augmentation_mode"]),
                int(contract["model"]["pilot_seed"]),
            )
            row = {
                "phase_loss_weight": float(phase_weight),
                "consistency_weight": float(consistency_weight),
                "validation_spearman": result["validation_spearman"],
                "validation_mae": result["validation_mae"],
                "validation_trace_spearman": result["validation_trace_spearman"],
                "best_epoch": result["best_epoch"],
            }
            rows.append(row)
            results.append((row, result))
            print(f"V5 CFPD+ pilot: {row}", flush=True)
    trace_floor = float(contract["model"]["minimum_validation_trace_spearman"])
    eligible = [item for item in results if item[0]["validation_trace_spearman"] >= trace_floor]
    pool = eligible or results
    selected_row, selected_result = max(
        pool, key=lambda item: (item[0]["validation_spearman"], item[0]["validation_trace_spearman"])
    )
    trials_path = V5_RESULTS_ROOT / "04_PILOT" / "cfpd_trials_v5.csv"
    pd.DataFrame(rows).to_csv(trials_path, index=False)
    checkpoint_path = V5_RUN_ROOT / "checkpoints" / "selected_cfpd_v5.pt"
    _save_checkpoint(selected_result, checkpoint_path, selected_row)
    selected = {
        "status": "PASS" if eligible else "WARNING_TRACE_FLOOR_NOT_MET",
        **selected_row,
        "hidden": int(baseline["hidden"]),
        "learning_rate": float(baseline["learning_rate"]),
        "ranking_weight": float(baseline["ranking_weight"]),
        "augmentation_mode": str(baseline["augmentation_mode"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trials_sha256": sha256_file(trials_path),
        "official_test_labels_used": False,
    }
    write_json(V5_RESULTS_ROOT / "04_PILOT" / "selected_cfpd_v5.json", selected)
    if not eligible:
        stop = {
            "status": "STOPPED_AT_PILOT",
            "reason": "No predeclared CFPD+ candidate reached the validation trace Spearman floor",
            "required_validation_trace_spearman": trace_floor,
            "maximum_observed_validation_trace_spearman": float(
                max(row["validation_trace_spearman"] for row in rows)
            ),
            "best_observed_validation_score_spearman": float(
                max(row["validation_spearman"] for row in rows)
            ),
            "official_test_unlocked": False,
            "official_test_labels_used": False,
            "next_stage_authorized": False,
        }
        write_json(V5_RESULTS_ROOT / "04_PILOT" / "stop_decision_v5.json", stop)
        (V5_RESULTS_ROOT / "RESULTS_DECISION_V5.md").write_text(
            "# TrustDive-CFPD+ v5 decision\n\n"
            "**STOPPED_AT_PILOT.** No predeclared candidate reached the "
            f"validation trace Spearman floor of {trace_floor:.2f}; the maximum "
            f"observed value was {stop['maximum_observed_validation_trace_spearman']:.4f}. "
            "The official 749-video test set therefore remains locked.\n\n"
            "The deterministic RICA2 teacher and exact counterfactual Shapley "
            "targets passed their technical gates, but the current three-phase "
            "mean-pooled transparent student did not recover either strong scoring "
            "or sufficient phase fidelity. No final test, stress-test claim, review "
            "claim, or formal paper figure is authorized under this contract.\n",
            encoding="utf-8",
        )
    return selected


def _variant_specs(baseline: dict, selected: dict) -> list[dict]:
    return [
        {
            "name": "rica2_ar",
            "phase_weight": 0.0,
            "consistency_weight": 0.0,
            "reliability_weighted": False,
            "epochs": int(baseline["best_epoch"]),
        },
        {
            "name": "rica2_ar_shapley",
            "phase_weight": float(selected["phase_loss_weight"]),
            "consistency_weight": 0.0,
            "reliability_weighted": False,
            "epochs": int(selected["best_epoch"]),
        },
        {
            "name": "rica2_ar_reliable_shapley",
            "phase_weight": float(selected["phase_loss_weight"]),
            "consistency_weight": 0.0,
            "reliability_weighted": True,
            "epochs": int(selected["best_epoch"]),
        },
        {
            "name": "cfpd_plus",
            "phase_weight": float(selected["phase_loss_weight"]),
            "consistency_weight": float(selected["consistency_weight"]),
            "reliability_weighted": True,
            "epochs": int(selected["best_epoch"]),
        },
    ]


def train_final_v5() -> dict:
    require_v5_frozen()
    contract = load_v5_contract()
    baseline = json.loads(
        (V5_RESULTS_ROOT / "02_BASELINE" / "selected_baseline_v5.json").read_text(
            encoding="utf-8"
        )
    )
    selected = json.loads(
        (V5_RESULTS_ROOT / "04_PILOT" / "selected_cfpd_v5.json").read_text(
            encoding="utf-8"
        )
    )
    inputs = prepare_inputs_v5()
    frame = load_v5_frame()
    variants = _variant_specs(baseline, selected)
    all_results: dict[str, list[dict]] = {}
    summary_rows = []
    for spec in variants:
        runs = []
        for seed in contract["model"]["final_seeds"]:
            result = fit_v5_model(
                inputs,
                int(baseline["hidden"]),
                float(baseline["learning_rate"]),
                float(baseline["ranking_weight"]),
                float(spec["phase_weight"]),
                float(spec["consistency_weight"]),
                bool(spec["reliability_weighted"]),
                str(baseline["augmentation_mode"]),
                int(seed),
                fit_roles=("fit", "validation"),
                validation_role="calibration",
                fixed_epochs=int(spec["epochs"]),
            )
            runs.append(result)
            checkpoint = V5_RUN_ROOT / "checkpoints" / f"{spec['name']}_{seed}.pt"
            _save_checkpoint(result, checkpoint, {**spec, "seed": int(seed)})
        all_results[spec["name"]] = runs
        predictions = np.mean(np.stack([run["prediction"] for run in runs]), axis=0)
        test = frame.official_split.to_numpy() == "test"
        scores = 3.0 * frame.difficulty.to_numpy(dtype=float) * predictions
        metrics = aqa_score_metrics(frame.dive_score.to_numpy(dtype=float)[test], scores[test])
        seed_rho = []
        for run in runs:
            seed_scores = 3.0 * frame.difficulty.to_numpy(dtype=float) * run["prediction"]
            seed_rho.append(
                safe_spearman(seed_scores[test], frame.dive_score.to_numpy(dtype=float)[test])
            )
        summary_rows.append(
            {
                "method": spec["name"],
                **metrics,
                "seed_spearman_mean": float(np.mean(seed_rho)),
                "seed_spearman_sd": float(np.std(seed_rho, ddof=0)),
            }
        )
        print(f"V5 final {spec['name']}: {summary_rows[-1]}", flush=True)

    teacher = pd.read_parquet(
        V5_RESULTS_ROOT / "01_REFERENCES" / "teacher_predictions_v5.parquet"
    ).set_index("clip_uid").loc[frame.clip_uid]
    test = frame.official_split.to_numpy() == "test"
    teacher_metrics = aqa_score_metrics(
        frame.dive_score.to_numpy(dtype=float)[test],
        teacher.teacher_predicted_score.to_numpy(dtype=float)[test],
    )
    summary_rows.insert(
        0,
        {
            "method": "rica2_frozen",
            **teacher_metrics,
            "seed_spearman_mean": teacher_metrics["spearman"],
            "seed_spearman_sd": 0.0,
        },
    )
    ablation_path = V5_RESULTS_ROOT / "05_FINAL" / "ablation_summary_v5.csv"
    pd.DataFrame(summary_rows).to_csv(ablation_path, index=False)

    def assemble(name: str) -> pd.DataFrame:
        runs = all_results[name]
        prediction = np.mean(np.stack([run["prediction"] for run in runs]), axis=0)
        contribution = np.mean(np.stack([run["contribution"] for run in runs]), axis=0)
        per_reference = np.mean(np.stack([run["per_reference"] for run in runs]), axis=0)
        output = frame.copy()
        output["predicted_quality"] = prediction
        output["predicted_score"] = 3.0 * output.difficulty * output.predicted_quality
        output["teacher_predicted_quality"] = teacher.teacher_predicted_quality.to_numpy()
        output["teacher_predicted_score"] = teacher.teacher_predicted_score.to_numpy()
        output["base_quality"] = inputs.base_quality
        output["valid_reference_count"] = inputs.valid_reference_count
        output["open_set"] = inputs.valid_reference_count < int(
            contract["data"]["minimum_valid_references"]
        )
        targets = pd.read_parquet(
            V5_RESULTS_ROOT / "03_COUNTERFACTUAL" / "counterfactual_query_targets_v5.parquet"
        ).set_index("clip_uid").loc[frame.clip_uid]
        output["reference_distance"] = targets.reference_distance.to_numpy()
        output["reference_dispersion"] = targets.reference_dispersion.to_numpy()
        output["hybrid_ood_score"] = targets.hybrid_ood_score.to_numpy()
        for phase_index, phase in enumerate(PHASES):
            output[f"phase_{phase}_contribution"] = contribution[:, phase_index]
            output[f"teacher_phi_{phase}"] = inputs.phase_targets[:, phase_index]
            output[f"phase_reliability_{phase}"] = inputs.phase_reliability[:, phase_index]
            output[f"reference_{phase}_dispersion"] = per_reference[:, :, phase_index].std(axis=1)
        for seed_index, seed in enumerate(contract["model"]["final_seeds"]):
            output[f"seed_{seed}_predicted_quality"] = runs[seed_index]["prediction"]
        output["ensemble_quality_sd"] = np.std(
            np.stack([run["prediction"] for run in runs]), axis=0, ddof=0
        )
        calibration = output.analysis_role == "calibration"
        conformal_q = float(
            np.quantile(
                np.abs(
                    output.loc[calibration, "predicted_quality"]
                    - output.loc[calibration, "execution_quality"]
                ),
                float(contract["risk"]["conformal_coverage"]),
                method="higher",
            )
        )
        output["conformal_half_width_quality"] = conformal_q
        output["lower_quality"] = output.predicted_quality - conformal_q
        output["upper_quality"] = output.predicted_quality + conformal_q
        return output

    baseline_output = assemble("rica2_ar")
    baseline_path = V5_RESULTS_ROOT / "05_FINAL" / "baseline_predictions_v5.parquet"
    baseline_output.to_parquet(baseline_path, index=False)
    full_output = assemble("cfpd_plus")
    full_path = V5_RESULTS_ROOT / "05_FINAL" / "predictions_cfpd_plus_v5.parquet"
    full_output.to_parquet(full_path, index=False)
    ablation_frames = []
    for name in ("rica2_ar_shapley", "rica2_ar_reliable_shapley"):
        output = assemble(name)[
            [
                "clip_uid",
                "predicted_quality",
                "predicted_score",
                *[f"phase_{phase}_contribution" for phase in PHASES],
            ]
        ].copy()
        output["method"] = name
        ablation_frames.append(output)
    ablation_prediction_path = V5_RESULTS_ROOT / "05_FINAL" / "ablation_predictions_v5.parquet"
    pd.concat(ablation_frames, ignore_index=True).to_parquet(
        ablation_prediction_path, index=False
    )
    result = {
        "status": "PASS",
        "methods": summary_rows,
        "baseline_predictions_sha256": sha256_file(baseline_path),
        "full_predictions_sha256": sha256_file(full_path),
        "ablation_predictions_sha256": sha256_file(ablation_prediction_path),
        "ablation_summary_sha256": sha256_file(ablation_path),
    }
    write_json(V5_RESULTS_ROOT / "05_FINAL" / "final_training_v5.json", result)
    return result
