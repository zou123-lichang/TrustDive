from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .util import sha256_file, write_json
from .v4_counterfactual import PHASES
from .v4_stress import (
    _alternate_references,
    _row_cosine,
    _shift_boundaries,
    _token_dropout_sequences,
    fleiss_kappa,
)
from .v5_data import V5_RESULTS_ROOT, V5_RUN_ROOT, load_v5_contract, load_v5_frame, require_v5_frozen
from .v5_modeling import _load_core_arrays, _predict, cfpd_plus_model_class, prepare_inputs_v5


def _load_full_model(seed: int, input_dim: int):
    import torch

    baseline = json.loads(
        (V5_RESULTS_ROOT / "02_BASELINE" / "selected_baseline_v5.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = V5_RUN_ROOT / "checkpoints" / f"cfpd_plus_{seed}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    Model = cfpd_plus_model_class(input_dim, int(baseline["hidden"]), 0.10)
    model = Model()
    model.load_state_dict(payload["state_dict"])
    return model.cuda().eval() if torch.cuda.is_available() else model.eval()


def _predict_all(model, inputs):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _predict(model, inputs, np.arange(len(inputs.base_quality)), device, 256)[:2]


def stress_test_v5() -> dict:
    require_v5_frozen()
    contract = load_v5_contract()
    frame = load_v5_frame()
    base_inputs = prepare_inputs_v5()
    seeds = [int(value) for value in contract["model"]["final_seeds"]]
    seed_predictions = []
    seed_contributions = []
    for seed in seeds:
        model = _load_full_model(seed, base_inputs.pair_features.shape[-1])
        prediction, contribution = _predict_all(model, base_inputs)
        seed_predictions.append(prediction)
        seed_contributions.append(contribution)
    baseline_prediction = np.mean(seed_predictions, axis=0)
    baseline_contribution = np.mean(seed_contributions, axis=0)
    baseline_top = np.argmax(np.abs(baseline_contribution), axis=1)
    primary_model = _load_full_model(seeds[0], base_inputs.pair_features.shape[-1])

    variants = {
        "first_left": _shift_boundaries(base_inputs.phase_labels, -1, 0),
        "first_right": _shift_boundaries(base_inputs.phase_labels, 1, 0),
        "second_left": _shift_boundaries(base_inputs.phase_labels, 0, -1),
        "second_right": _shift_boundaries(base_inputs.phase_labels, 0, 1),
    }
    perturbed_predictions = []
    perturbed_contributions = []
    for labels in variants.values():
        prediction, contribution = _predict_all(
            primary_model, prepare_inputs_v5(labels_override=labels)
        )
        perturbed_predictions.append(prediction)
        perturbed_contributions.append(contribution)

    sequences, current_references, _, _ = _load_core_arrays()
    token_inputs = prepare_inputs_v5(
        sequences_override=_token_dropout_sequences(
            sequences,
            int(contract["statistics"]["seed"]),
            float(contract["model"]["token_dropout_fraction"]),
        )
    )
    token_prediction, token_contribution = _predict_all(primary_model, token_inputs)
    perturbed_predictions.append(token_prediction)
    perturbed_contributions.append(token_contribution)

    alternate = _alternate_references(
        frame,
        sequences,
        current_references,
        int(contract["statistics"]["seed"]) + 11,
    )
    reference_prediction, reference_contribution = _predict_all(
        primary_model, prepare_inputs_v5(references_override=alternate)
    )
    perturbed_predictions.append(reference_prediction)
    perturbed_contributions.append(reference_contribution)

    fit = frame.analysis_role.to_numpy() == "fit"
    robust_scale = np.median(
        np.abs(sequences[fit] - np.median(sequences[fit], axis=(0, 1), keepdims=True)),
        axis=(0, 1),
    )
    robust_scale = np.where(robust_scale > 1e-6, 1.4826 * robust_scale, 1.0)
    for offset in range(3):
        rng = np.random.default_rng(int(contract["statistics"]["seed"]) + 100 + offset)
        noisy = sequences + float(contract["model"]["feature_noise_fraction"]) * rng.normal(
            size=sequences.shape
        ).astype(np.float32) * robust_scale[None, None, :]
        prediction, contribution = _predict_all(
            primary_model, prepare_inputs_v5(sequences_override=noisy)
        )
        perturbed_predictions.append(prediction)
        perturbed_contributions.append(contribution)

    all_prediction = np.vstack(perturbed_predictions)
    all_contribution = np.stack(perturbed_contributions)
    top_agreement = np.mean(
        np.argmax(np.abs(all_contribution), axis=2) == baseline_top[None, :], axis=0
    )
    cosine = np.median(
        np.stack([_row_cosine(item, baseline_contribution) for item in all_contribution]),
        axis=0,
    )
    prediction_change = np.median(
        np.abs(all_prediction - baseline_prediction[None, :]), axis=0
    )
    seed_top = np.stack(
        [np.argmax(np.abs(item), axis=1) for item in seed_contributions], axis=1
    )

    with np.load(
        V5_RESULTS_ROOT / "03_COUNTERFACTUAL" / "counterfactual_arrays_v5.npz",
        allow_pickle=False,
    ) as payload:
        coalition = payload["coalition_values"].astype(np.float32)
    teacher_effects = np.empty((len(frame), 3), dtype=np.float32)
    teacher_keep_gap = np.empty_like(teacher_effects)
    for phase in range(3):
        teacher_effects[:, phase] = np.median(
            np.abs(coalition[:, :, 7] - coalition[:, :, 7 ^ (1 << phase)]), axis=1
        )
        teacher_keep_gap[:, phase] = np.median(
            np.abs(coalition[:, :, 7] - coalition[:, :, 1 << phase]), axis=1
        )
    teacher_top = np.argmax(teacher_effects, axis=1)
    rng = np.random.default_rng(int(contract["statistics"]["seed"]))
    random_phase = rng.integers(0, 3, size=len(frame))
    lowest_phase = np.argmin(np.abs(baseline_contribution), axis=1)
    targeted = teacher_effects[np.arange(len(frame)), baseline_top]
    random_effect = teacher_effects[np.arange(len(frame)), random_phase]
    lowest_effect = teacher_effects[np.arange(len(frame)), lowest_phase]

    output = frame[
        ["clip_uid", "official_split", "analysis_role", "event_family", "action_type"]
    ].copy()
    output["baseline_prediction"] = baseline_prediction
    output["baseline_top_phase"] = baseline_top
    output["teacher_top_phase"] = teacher_top
    output["top_phase_matches_teacher"] = baseline_top == teacher_top
    output["perturbed_top_phase_agreement"] = top_agreement
    output["perturbed_contribution_cosine"] = cosine
    output["perturbed_prediction_change"] = prediction_change
    output["targeted_teacher_effect"] = targeted
    output["random_teacher_effect"] = random_effect
    output["lowest_teacher_effect"] = lowest_effect
    output["targeted_minus_random"] = targeted - random_effect
    output["top_phase_sufficiency_gap"] = teacher_keep_gap[
        np.arange(len(frame)), baseline_top
    ]
    output["cross_seed_top_agreement"] = np.mean(
        seed_top == seed_top[:, [0]], axis=1
    )
    for phase_index, phase in enumerate(PHASES):
        output[f"phase_{phase}_contribution"] = baseline_contribution[:, phase_index]
        output[f"teacher_effect_{phase}"] = teacher_effects[:, phase_index]
    stress_path = V5_RESULTS_ROOT / "06_STRESS" / "trace_stress_v5.parquet"
    output.to_parquet(stress_path, index=False)
    evidence_path = V5_RESULTS_ROOT / "06_STRESS" / "phase_evidence_v5.parquet"
    output.to_parquet(evidence_path, index=False)

    test = frame.official_split.to_numpy() == "test"
    result = {
        "status": "COMPLETE",
        "official_test_n": int(test.sum()),
        "median_perturbed_top_phase_agreement": float(np.median(top_agreement[test])),
        "median_contribution_cosine": float(np.median(cosine[test])),
        "median_prediction_change": float(np.median(prediction_change[test])),
        "cross_seed_fleiss_kappa": fleiss_kappa(seed_top[test]),
        "top_phase_teacher_match": float(np.mean((baseline_top == teacher_top)[test])),
        "median_targeted_minus_random": float(np.median((targeted - random_effect)[test])),
        "token_dropout_top_phase_agreement": float(
            np.mean((np.argmax(np.abs(token_contribution), axis=1) == baseline_top)[test])
        ),
        "reference_change_top_phase_agreement": float(
            np.mean((np.argmax(np.abs(reference_contribution), axis=1) == baseline_top)[test])
        ),
        "reference_change_median_prediction_change": float(
            np.median(np.abs(reference_prediction - baseline_prediction)[test])
        ),
        "stress_sha256": sha256_file(stress_path),
        "phase_evidence_sha256": sha256_file(evidence_path),
    }
    write_json(V5_RESULTS_ROOT / "06_STRESS" / "stress_summary_v5.json", result)
    return result
