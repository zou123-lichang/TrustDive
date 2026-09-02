from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .util import write_json
from .v4_counterfactual import PHASES
from .v4_data import V4_RESULTS_ROOT, V4_RUN_ROOT, load_v4_contract, load_v4_frame, require_v4_frozen
from .v4_modeling import cfpd_model_class, prepare_student_inputs_v4


def _shift_boundaries(labels: np.ndarray, first_shift: int = 0, second_shift: int = 0) -> np.ndarray:
    output = np.zeros_like(labels, dtype=np.int8)
    length = labels.shape[1]
    for index, row in enumerate(labels):
        first = int(np.flatnonzero(row == 1)[0])
        second = int(np.flatnonzero(row == 2)[0])
        first = int(np.clip(first + first_shift, 1, length - 2))
        second = int(np.clip(second + second_shift, first + 1, length - 1))
        output[index, first:second] = 1
        output[index, second:] = 2
    return output


def _load_model(seed: int, input_dim: int):
    import torch

    contract = load_v4_contract()
    checkpoint = V4_RUN_ROOT / "checkpoints" / f"cfpd_seed_{seed}.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    Model = cfpd_model_class(input_dim, int(contract["model"]["hidden_dimension"]), float(contract["model"]["dropout"]))
    model = Model()
    model.load_state_dict(payload["state_dict"])
    return model.cuda().eval()


def _predict(model, inputs, noise_seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    import torch

    predictions = []
    contributions = []
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    with torch.inference_mode():
        for start in range(0, len(inputs.base_quality), 256):
            pair = inputs.pair_features[start : start + 256].astype(np.float32)
            if rng is not None:
                pair = pair + rng.normal(0.0, 0.05, size=pair.shape).astype(np.float32)
            pair_tensor = torch.from_numpy(pair).cuda()
            base = torch.from_numpy(inputs.base_quality[start : start + 256]).cuda()
            prediction, contribution, _ = model(pair_tensor, base)
            predictions.append(prediction.cpu().numpy())
            contributions.append(contribution.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(contributions)


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 1e-12)


def _token_dropout_sequences(sequences: np.ndarray, seed: int, fraction: float = 0.10) -> np.ndarray:
    output = np.asarray(sequences, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    count = max(1, int(round(output.shape[1] * fraction)))
    for index in range(len(output)):
        selected = rng.choice(output.shape[1], size=count, replace=False)
        output[index, selected] = 0.0
    return output


def _alternate_references(frame: pd.DataFrame, sequences: np.ndarray, current: np.ndarray, seed: int) -> np.ndarray:
    fit = np.flatnonzero(frame.analysis_role.to_numpy() == "fit")
    actions = frame.action_type.astype(str).to_numpy()
    families = frame.event_family.astype(str).to_numpy()
    global_features = sequences.mean(axis=1)
    normalized = global_features / np.maximum(np.linalg.norm(global_features, axis=1, keepdims=True), 1e-8)
    rng = np.random.default_rng(seed)
    output = np.empty_like(current)
    for index in range(len(frame)):
        excluded = set(int(value) for value in current[index])
        candidates = fit[
            (actions[fit] == actions[index])
            & (families[fit] != families[index])
            & np.asarray([int(value) not in excluded for value in fit])
        ]
        if len(candidates):
            distance = 1.0 - normalized[candidates] @ normalized[index]
            jitter = rng.uniform(0.0, 1e-7, size=len(candidates))
            selected = candidates[np.argsort(distance + jitter)[: current.shape[1]]].tolist()
        else:
            selected = current[index, ::-1].tolist()
        while len(selected) < current.shape[1]:
            selected.append(selected[len(selected) % max(len(selected), 1)])
        output[index] = selected[: current.shape[1]]
    return output


def fleiss_kappa(labels: np.ndarray, categories: int = 3) -> float:
    labels = np.asarray(labels, dtype=int)
    n_items, raters = labels.shape
    counts = np.zeros((n_items, categories), dtype=float)
    for category in range(categories):
        counts[:, category] = np.sum(labels == category, axis=1)
    agreement = (np.sum(counts**2, axis=1) - raters) / (raters * (raters - 1))
    proportions = counts.sum(axis=0) / (n_items * raters)
    expected = float(np.sum(proportions**2))
    return float((agreement.mean() - expected) / (1.0 - expected)) if expected < 1.0 else 1.0


def stress_test_v4() -> dict:
    require_v4_frozen()
    contract = load_v4_contract()
    frame = load_v4_frame()
    base_inputs = prepare_student_inputs_v4()
    seeds = [int(value) for value in contract["model"]["final_seeds"]]
    seed_predictions = []
    seed_contributions = []
    for seed in seeds:
        model = _load_model(seed, base_inputs.pair_features.shape[-1])
        prediction, contribution = _predict(model, base_inputs)
        seed_predictions.append(prediction)
        seed_contributions.append(contribution)
    baseline_prediction = np.mean(seed_predictions, axis=0)
    baseline_contribution = np.mean(seed_contributions, axis=0)
    baseline_top = np.argmax(np.abs(baseline_contribution), axis=1)

    variants = {
        "first_left": _shift_boundaries(base_inputs.phase_labels, -1, 0),
        "first_right": _shift_boundaries(base_inputs.phase_labels, 1, 0),
        "second_left": _shift_boundaries(base_inputs.phase_labels, 0, -1),
        "second_right": _shift_boundaries(base_inputs.phase_labels, 0, 1),
    }
    variant_predictions = []
    variant_contributions = []
    primary_model = _load_model(seeds[0], base_inputs.pair_features.shape[-1])
    for labels in variants.values():
        variant_inputs = prepare_student_inputs_v4(labels)
        prediction, contribution = _predict(primary_model, variant_inputs)
        variant_predictions.append(prediction)
        variant_contributions.append(contribution)
    with np.load(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_sequences_v4.npz", allow_pickle=False) as payload:
        teacher_sequences = payload["sequence"].astype(np.float32)
    token_inputs = prepare_student_inputs_v4(
        sequences_override=_token_dropout_sequences(
            teacher_sequences, int(contract["statistics"]["seed"]), 0.10
        )
    )
    token_prediction, token_contribution = _predict(primary_model, token_inputs)
    variant_predictions.append(token_prediction)
    variant_contributions.append(token_contribution)
    alternate = _alternate_references(
        frame,
        teacher_sequences,
        base_inputs.references,
        int(contract["statistics"]["seed"]) + 11,
    )
    reference_inputs = prepare_student_inputs_v4(references_override=alternate)
    reference_prediction, reference_contribution = _predict(primary_model, reference_inputs)
    variant_predictions.append(reference_prediction)
    variant_contributions.append(reference_contribution)
    noise_predictions = []
    noise_contributions = []
    for offset in range(3):
        prediction, contribution = _predict(primary_model, base_inputs, int(contract["statistics"]["seed"]) + offset)
        noise_predictions.append(prediction)
        noise_contributions.append(contribution)

    all_perturbed_predictions = np.vstack((*variant_predictions, *noise_predictions))
    all_perturbed_contributions = np.stack((*variant_contributions, *noise_contributions))
    top_agreement = np.mean(np.argmax(np.abs(all_perturbed_contributions), axis=2) == baseline_top[None, :], axis=0)
    cosine = np.median(
        np.stack([_row_cosine(item, baseline_contribution) for item in all_perturbed_contributions]), axis=0
    )
    prediction_change = np.median(np.abs(all_perturbed_predictions - baseline_prediction[None, :]), axis=0)
    seed_top = np.stack([np.argmax(np.abs(item), axis=1) for item in seed_contributions], axis=1)

    pair = pd.read_parquet(V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "counterfactual_targets_v4.parquet")
    teacher_effects = np.empty((len(frame), 3), dtype=float)
    teacher_keep = np.empty((len(frame), 3), dtype=float)
    for query_index, group in pair.groupby("query_index", sort=True):
        full = group.coalition_7.to_numpy(dtype=float)
        for phase in range(3):
            removed = group[f"coalition_{7 ^ (1 << phase)}"].to_numpy(dtype=float)
            kept = group[f"coalition_{1 << phase}"].to_numpy(dtype=float)
            teacher_effects[int(query_index), phase] = float(np.median(np.abs(full - removed)))
            teacher_keep[int(query_index), phase] = float(np.median(np.abs(full - kept)))
    teacher_top = np.argmax(teacher_effects, axis=1)
    rng = np.random.default_rng(int(contract["statistics"]["seed"]))
    random_phase = rng.integers(0, 3, size=len(frame))
    targeted_effect = teacher_effects[np.arange(len(frame)), baseline_top]
    random_effect = teacher_effects[np.arange(len(frame)), random_phase]
    sufficiency_gap = teacher_keep[np.arange(len(frame)), baseline_top]

    output = frame[["clip_uid", "official_split", "analysis_role", "source_role", "event_family"]].copy()
    output["baseline_prediction"] = baseline_prediction
    output["baseline_top_phase"] = baseline_top
    output["teacher_top_phase"] = teacher_top
    output["top_phase_matches_teacher"] = baseline_top == teacher_top
    output["perturbed_top_phase_agreement"] = top_agreement
    output["perturbed_contribution_cosine"] = cosine
    output["perturbed_prediction_change"] = prediction_change
    output["targeted_teacher_effect"] = targeted_effect
    output["random_teacher_effect"] = random_effect
    output["targeted_minus_random"] = targeted_effect - random_effect
    output["top_phase_sufficiency_gap"] = sufficiency_gap
    output["cross_seed_top_agreement"] = np.mean(seed_top == seed_top[:, [0]], axis=1)
    output.to_parquet(V4_RESULTS_ROOT / "05_STRESS" / "trace_stress_v4.parquet", index=False)

    test = frame.official_split.to_numpy() == "test"
    result = {
        "status": "COMPLETE",
        "official_test_n": int(test.sum()),
        "median_perturbed_top_phase_agreement": float(np.median(top_agreement[test])),
        "median_contribution_cosine": float(np.median(cosine[test])),
        "median_prediction_change": float(np.median(prediction_change[test])),
        "cross_seed_fleiss_kappa": fleiss_kappa(seed_top[test]),
        "top_phase_teacher_match": float(np.mean((baseline_top == teacher_top)[test])),
        "median_targeted_minus_random": float(np.median((targeted_effect - random_effect)[test])),
        "token_dropout_top_phase_agreement": float(
            np.mean((np.argmax(np.abs(token_contribution), axis=1) == baseline_top)[test])
        ),
        "reference_change_top_phase_agreement": float(
            np.mean((np.argmax(np.abs(reference_contribution), axis=1) == baseline_top)[test])
        ),
        "reference_change_median_prediction_change": float(
            np.median(np.abs(reference_prediction - baseline_prediction)[test])
        ),
    }
    write_json(V4_RESULTS_ROOT / "05_STRESS" / "stress_summary_v4.json", result)
    return result
