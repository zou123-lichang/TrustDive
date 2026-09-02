from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .util import sha256_file, write_json
from .v4_counterfactual import PHASES, exact_three_phase_shapley, hybrid_sequence
from .v5_counterfactual import _phase_labels
from .v5_data import V5_RESULTS_ROOT
from .v6_data import V6_RESULTS_ROOT, V6_RUN_ROOT, load_v6_contract, require_v6_frozen
from .v6_modeling import (
    ReferenceMLP,
    load_reference_map_v6,
    load_selected_adapter_v6,
    load_v6_assets,
    predict_selected_v6,
)


def _load_sequences() -> tuple[np.ndarray, np.ndarray]:
    with np.load(
        V5_RESULTS_ROOT / "01_REFERENCES" / "teacher_sequences_v5.npz",
        allow_pickle=False,
    ) as payload:
        return (
            payload["sequence"].astype(np.float32),
            payload["action_presence"].astype(np.float32),
        )


def _load_models(final: bool) -> tuple[dict, list[object]]:
    selected, development_model = load_selected_adapter_v6()
    if not final:
        return selected, [development_model]
    models: list[object] = []
    model_type = selected["selected"]["model_type"]
    for seed in load_v6_contract()["statistics"]["model_seeds"]:
        suffix = ".pt" if model_type == "mlp" else ".joblib"
        path = V6_RUN_ROOT / "checkpoints" / f"final_adapter_seed_{seed}_v6{suffix}"
        if model_type == "mlp":
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = ReferenceMLP(payload["hidden"], payload["residual_limit"])
            model.module.load_state_dict(payload["state_dict"])
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            models.append(model)
        elif model_type == "reference_residual":
            models.append({"type": "reference_residual"})
        else:
            models.append(joblib.load(path))
    return selected, models


def _predict_teacher_latent(sequences: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    path = V6_RESULTS_ROOT / "01_LATENTS" / "teacher_latent_counterfactual_v6.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(path), map_location=device).eval()
    quality: list[np.ndarray] = []
    latent: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), 1024):
            output = model(
                torch.from_numpy(sequences[start : start + 1024]).to(device),
                torch.from_numpy(actions[start : start + 1024]).to(device),
            )
            quality.append(output[0].detach().cpu().numpy().reshape(-1))
            latent.append(output[1].detach().cpu().numpy())
    return np.concatenate(quality).astype(np.float32), np.concatenate(latent).astype(np.float32)


def _component_predictions(
    selected: dict,
    models: list[object],
    teacher_hybrid: np.ndarray,
    global_hybrid: np.ndarray,
    ref_indices: np.ndarray,
    query_indices: np.ndarray,
    assets,
    reference_map: dict[str, np.ndarray],
) -> np.ndarray:
    model_type = selected["selected"]["model_type"]
    outputs: list[np.ndarray] = []
    for model in models:
        if model_type == "linear":
            prediction = model.predict(teacher_hybrid[:, None])
        elif model_type == "ridge":
            x = np.concatenate((global_hybrid, teacher_hybrid[:, None]), axis=1)
            prediction = teacher_hybrid + model.predict(x)
        elif model_type == "reference_residual":
            target = assets.frame.execution_quality.to_numpy(dtype=float)
            prediction = teacher_hybrid + target[ref_indices] - assets.teacher_quality[ref_indices]
        elif model_type == "mlp":
            import torch

            gr = assets.global_latent[ref_indices]
            qt_ref = assets.teacher_quality[ref_indices, None]
            q_ref = assets.frame.execution_quality.to_numpy(dtype=np.float32)[ref_indices, None]
            distance = 1.0 - np.sum(
                global_hybrid / np.maximum(np.linalg.norm(global_hybrid, axis=1, keepdims=True), 1e-8)
                * gr / np.maximum(np.linalg.norm(gr, axis=1, keepdims=True), 1e-8),
                axis=1,
                keepdims=True,
            )
            dispersion = np.repeat(
                np.nanstd(
                    assets.frame.execution_quality.to_numpy(dtype=float)[
                        np.maximum(reference_map["references"][query_indices, :5], 0)
                    ], axis=1,
                )[:, None], 1, axis=1,
            )
            feature = np.concatenate(
                (
                    global_hybrid, gr, global_hybrid - gr, np.abs(global_hybrid - gr),
                    teacher_hybrid[:, None], qt_ref, q_ref, distance, dispersion,
                ), axis=1,
            ).astype(np.float32)
            device = next(model.module.parameters()).device
            model.module.eval()
            with torch.inference_mode():
                delta = model.residual_limit * torch.tanh(
                    model.module(torch.from_numpy(feature).to(device)).squeeze(-1)
                )
            prediction = teacher_hybrid + delta.detach().cpu().numpy()
        else:
            raise ValueError(model_type)
        outputs.append(np.asarray(prediction, dtype=np.float32))
    return np.mean(outputs, axis=0)


def _jitter_labels(labels: np.ndarray, direction: int) -> np.ndarray:
    labels = labels.copy()
    first_flight = int(np.flatnonzero(labels == 1)[0]) if np.any(labels == 1) else len(labels) // 3
    first_entry = int(np.flatnonzero(labels == 2)[0]) if np.any(labels == 2) else 2 * len(labels) // 3
    first_flight = int(np.clip(first_flight + direction, 1, len(labels) - 2))
    first_entry = int(np.clip(first_entry + direction, first_flight + 1, len(labels) - 1))
    result = np.zeros_like(labels)
    result[first_flight:first_entry] = 1
    result[first_entry:] = 2
    return result


def _coalitions_for_indices(
    indices: np.ndarray,
    assets,
    reference_map: dict[str, np.ndarray],
    selected: dict,
    models: list[object],
    sequences: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    variant: str = "original",
) -> tuple[np.ndarray, np.ndarray]:
    primary_refs = reference_map["references"][:, :5].astype(int)
    if variant == "reference_replace":
        alternates = reference_map["references"][:, 5:10].astype(int)
        valid_alt = alternates >= 0
        primary_refs = np.where(valid_alt, alternates, primary_refs)
    hybrids: list[np.ndarray] = []
    hybrid_actions: list[np.ndarray] = []
    metadata: list[tuple[int, int, int, int]] = []
    for local, query in enumerate(indices):
        for slot, ref in enumerate(primary_refs[query]):
            if ref < 0:
                continue
            q_labels = labels[query]
            r_labels = labels[ref]
            if variant == "boundary_left":
                q_labels, r_labels = _jitter_labels(q_labels, -1), _jitter_labels(r_labels, -1)
            elif variant == "boundary_right":
                q_labels, r_labels = _jitter_labels(q_labels, 1), _jitter_labels(r_labels, 1)
            for mask in range(8):
                hybrid = hybrid_sequence(
                    sequences[query], sequences[ref], q_labels, r_labels, mask
                )
                if variant == "token_dropout":
                    token = int((query * 2654435761) % len(hybrid))
                    hybrid = hybrid.copy()
                    hybrid[token] = 0.0
                hybrids.append(hybrid)
                hybrid_actions.append(actions[query])
                metadata.append((local, query, slot, mask))
    teacher, global_latent = _predict_teacher_latent(
        np.stack(hybrids), np.stack(hybrid_actions)
    )
    ref_vector = np.asarray(
        [primary_refs[query, slot] for _, query, slot, _ in metadata], dtype=int
    )
    query_vector = np.asarray([query for _, query, _, _ in metadata], dtype=int)
    components = _component_predictions(
        selected, models, teacher, global_latent, ref_vector, query_vector, assets, reference_map
    )
    values = np.full((len(indices), 5, 8), np.nan, dtype=np.float32)
    latent_values = np.full((len(indices), 5, 8, 256), np.nan, dtype=np.float32)
    for component, latent, (local, _query, slot, mask) in zip(components, global_latent, metadata):
        values[local, slot, mask] = component
        latent_values[local, slot, mask] = latent
    weights = reference_map["weights"][indices].astype(np.float32)
    valid = np.isfinite(values[..., 0])
    weights = np.where(valid, weights, 0.0)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    coalition = np.nansum(values * weights[:, :, None], axis=1)
    return coalition, latent_values


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)
    return numerator / denominator


def build_attributions_v6(final: bool = False) -> dict:
    if final:
        require_v6_frozen()
        score_summary = V6_RESULTS_ROOT / "05_FINAL" / "final_score_summary_v6.json"
        if not score_summary.exists() or json.loads(score_summary.read_text(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError("Final scoring gate must pass before final attribution")
    contract = load_v6_contract()
    assets = load_v6_assets()
    reference_map = load_reference_map_v6(final=final)
    selected, models = _load_models(final)
    sequences, actions = _load_sequences()
    labels = _phase_labels(sequences.shape[1])
    if final:
        # Final phase evidence is label-free, so all rows are evaluated. This
        # supplies cross-fit/calibration risk features without ever using test
        # outcomes to train the adapter or choose a threshold.
        indices = np.arange(len(assets.frame), dtype=int)
    else:
        indices = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "validation")
    closed = ~reference_map["open_set"][indices].astype(bool)
    analyzed = indices[closed]
    coalition, latent_hybrids = _coalitions_for_indices(
        analyzed, assets, reference_map, selected, models, sequences, actions, labels
    )
    phi = exact_three_phase_shapley(coalition[:, None, :])[:, 0, :]
    reconstruction = np.abs(coalition[:, 0] + phi.sum(axis=1) - coalition[:, 7])

    fit = assets.frame.analysis_role.to_numpy() == "fit"
    center = assets.global_latent[fit].mean(axis=0)
    scale = np.where(assets.global_latent[fit].std(axis=0) > 1e-6, assets.global_latent[fit].std(axis=0), 1.0)
    real_ood = np.sqrt(np.mean(((assets.global_latent[fit] - center) / scale) ** 2, axis=1))
    threshold = float(np.quantile(real_ood, float(contract["attribution"]["real_distribution_quantile"])))
    hybrid_ood = np.full(latent_hybrids.shape[:3], np.nan, dtype=np.float32)
    finite_hybrid = np.isfinite(latent_hybrids).all(axis=3)
    standardized = (latent_hybrids[finite_hybrid] - center) / scale
    hybrid_ood[finite_hybrid] = np.sqrt(np.mean(standardized**2, axis=1))
    ood_fraction = float(np.nanmean(hybrid_ood > threshold))

    top = np.argmax(np.abs(phi), axis=1)
    rng = np.random.default_rng(int(contract["statistics"]["seed"]))
    random_stage = rng.integers(0, 3, size=len(analyzed))
    full = coalition[:, 7]
    top_effect = np.asarray([abs(full[i] - coalition[i, 7 ^ (1 << int(top[i]))]) for i in range(len(analyzed))])
    random_effect = np.asarray([abs(full[i] - coalition[i, 7 ^ (1 << int(random_stage[i]))]) for i in range(len(analyzed))])

    variants = {}
    for name in ("boundary_left", "boundary_right", "token_dropout", "reference_replace"):
        variant_coalition, _ = _coalitions_for_indices(
            analyzed, assets, reference_map, selected, models, sequences, actions, labels, variant=name
        )
        variant_phi = exact_three_phase_shapley(variant_coalition[:, None, :])[:, 0, :]
        variants[name] = {
            "phi": variant_phi,
            "cosine": _cosine_rows(phi, variant_phi),
            "top_match": np.argmax(np.abs(variant_phi), axis=1) == top,
        }
    all_cosine = np.concatenate([item["cosine"] for item in variants.values()])
    all_top = np.concatenate([item["top_match"] for item in variants.values()])
    top_share = np.bincount(top, minlength=3) / max(len(top), 1)

    output = assets.frame.loc[indices, [
        "clip_uid", "official_split", "analysis_role", "event_family", "action_type", "difficulty", "dive_score"
    ]].copy()
    output["open_set"] = reference_map["open_set"][indices].astype(bool)
    output["reference_baseline"] = np.nan
    output["predicted_quality"] = np.nan
    output["phi_takeoff"] = np.nan
    output["phi_flight"] = np.nan
    output["phi_entry"] = np.nan
    output["top_phase"] = "open_set"
    output["reconstruction_error"] = np.nan
    output["targeted_intervention_effect"] = np.nan
    output["random_intervention_effect"] = np.nan
    output["phase_stability_cosine"] = np.nan
    output["top_phase_stability"] = np.nan
    for mask in range(8):
        output[f"coalition_{mask}"] = np.nan
    locations = np.flatnonzero(closed)
    output.iloc[locations, output.columns.get_loc("reference_baseline")] = coalition[:, 0]
    output.iloc[locations, output.columns.get_loc("predicted_quality")] = coalition[:, 7]
    for phase_index, phase in enumerate(PHASES):
        output.iloc[locations, output.columns.get_loc(f"phi_{phase}")] = phi[:, phase_index]
    output.iloc[locations, output.columns.get_loc("top_phase")] = np.asarray(PHASES)[top]
    output.iloc[locations, output.columns.get_loc("reconstruction_error")] = reconstruction
    output.iloc[locations, output.columns.get_loc("targeted_intervention_effect")] = top_effect
    output.iloc[locations, output.columns.get_loc("random_intervention_effect")] = random_effect
    output.iloc[locations, output.columns.get_loc("phase_stability_cosine")] = np.mean(
        np.stack([item["cosine"] for item in variants.values()]), axis=0
    )
    output.iloc[locations, output.columns.get_loc("top_phase_stability")] = np.mean(
        np.stack([item["top_match"] for item in variants.values()]), axis=0
    )
    for mask in range(8):
        output.iloc[locations, output.columns.get_loc(f"coalition_{mask}")] = coalition[:, mask]
    prefix = "final" if final else "pilot"
    path = V6_RESULTS_ROOT / ("05_FINAL" if final else "03_ATTRIBUTION") / f"phase_evidence_{prefix}_v6.parquet"
    output.to_parquet(path, index=False)

    checks = {
        "closed_set_available": len(analyzed) > 0,
        "exact_reconstruction": float(reconstruction.max(initial=0.0)) <= float(contract["attribution"]["maximum_reconstruction_error"]),
        "hybrid_ood": ood_fraction <= float(contract["attribution"]["maximum_hybrid_ood_fraction"]),
        "targeted_intervention": float(np.median(top_effect - random_effect)) > float(contract["attribution"]["minimum_targeted_minus_random"]),
        "all_phases_represented": bool(np.all(top_share >= float(contract["attribution"]["minimum_top_phase_share"]))),
        "not_entry_collapsed": float(top_share[2]) <= float(contract["attribution"]["maximum_entry_top_share"]),
        "contribution_stability": float(np.median(all_cosine)) >= float(contract["attribution"]["minimum_contribution_cosine"]),
        "top_phase_stability": float(np.mean(all_top)) >= float(contract["attribution"]["minimum_top_phase_agreement"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": prefix,
        "checks": checks,
        "rows": int(len(output)),
        "closed_set_rows": int(len(analyzed)),
        "open_set_rows": int((~closed).sum()),
        "maximum_reconstruction_error": float(reconstruction.max(initial=0.0)),
        "hybrid_ood_fraction": ood_fraction,
        "hybrid_ood_threshold": threshold,
        "targeted_minus_random_median": float(np.median(top_effect - random_effect)),
        "top_phase_share": {phase: float(value) for phase, value in zip(PHASES, top_share)},
        "perturbation_cosine_median": float(np.median(all_cosine)),
        "perturbation_top_phase_agreement": float(np.mean(all_top)),
        "evidence_sha256": sha256_file(path),
    }
    summary_path = V6_RESULTS_ROOT / ("05_FINAL" if final else "03_ATTRIBUTION") / f"{prefix}_attribution_summary_v6.json"
    write_json(summary_path, result)
    return result


def pilot_gate_v6() -> dict:
    selected_path = V6_RESULTS_ROOT / "02_ADAPTER" / "selected_adapter_v6.json"
    attribution_path = V6_RESULTS_ROOT / "03_ATTRIBUTION" / "pilot_attribution_summary_v6.json"
    if not selected_path.exists() or not attribution_path.exists():
        raise RuntimeError("Run optimize-adapter and build-attributions before pilot")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    checks = {
        "adapter_validation": selected.get("status") == "PASS",
        "exact_attribution": attribution.get("status") == "PASS",
        "official_test_locked": not bool(load_v6_contract()["state"]["official_test_unlocked"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "STOP",
        "checks": checks,
        "selected_adapter": selected.get("selected"),
        "attribution": attribution,
        "official_test_labels_accessed": False,
    }
    write_json(V6_RESULTS_ROOT / "04_PILOT" / "pilot_gate_v6.json", result)
    if result["status"] != "PASS":
        write_json(V6_RESULTS_ROOT / "04_PILOT" / "stop_decision_v6.json", result)
        (V6_RESULTS_ROOT / "RESULTS_DECISION_V6.md").write_text(
            "# TrustDive-ECR v6 decision\n\n"
            "**STOPPED_AT_PILOT.** The validation adapter gate passed, but the "
            "predeclared top-phase stability gate failed. The official 749-video "
            "test set, review analysis, and formal paper figures remain locked.\n",
            encoding="utf-8",
        )
    return result
