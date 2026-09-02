from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .config import Paths, load_contract
from .metrics import restore_total_score, score_metrics
from .splits import add_analysis_roles
from .statistics import split_conformal_radius
from .util import set_seed, write_json


def ordered_phase_targets(frame_count: int, transitions: list[int], length: int) -> np.ndarray:
    boundaries = [0] + [int(x) for x in transitions] + [int(frame_count)]
    boundaries = sorted(set(max(0, min(frame_count, x)) for x in boundaries))
    if len(boundaries) < 4:
        boundaries = [0, round(frame_count / 3), round(2 * frame_count / 3), frame_count]
    sample_positions = np.linspace(0, max(frame_count - 1, 0), length)
    target = np.zeros(length, dtype=np.int64)
    target[sample_positions >= boundaries[1]] = 1
    target[sample_positions >= boundaries[-2]] = 2
    return target


def monotonic_decode(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 3 or logits.shape[0] < 3:
        raise ValueError("Expected [time, 3] logits with at least three time steps")
    best_score = -np.inf
    best = (1, logits.shape[0] - 1)
    for first in range(1, logits.shape[0] - 1):
        for second in range(first + 1, logits.shape[0]):
            score = logits[:first, 0].sum() + logits[first:second, 1].sum() + logits[second:, 2].sum()
            if score > best_score:
                best_score = score
                best = (first, second)
    labels = np.zeros(logits.shape[0], dtype=np.int64)
    labels[best[0] : best[1]] = 1
    labels[best[1] :] = 2
    return labels


def phase_pool(sequence: np.ndarray, labels: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    pooled = []
    for phase in range(3):
        selected = sequence[labels == phase]
        if not len(selected):
            pooled.append(np.zeros(sequence.shape[1], dtype=np.float32))
        elif np.issubdtype(selected.dtype, np.floating) and np.isnan(selected).any():
            with np.errstate(invalid="ignore"):
                pooled.append(np.nanmean(selected, axis=0))
        else:
            pooled.append(selected.mean(axis=0))
    return np.stack(pooled)


class FeatureRepository:
    def __init__(self, manifest: pd.DataFrame, paths: Paths | None = None):
        self.manifest = manifest.set_index("clip_uid", drop=False)
        self.paths = paths or Paths()
        self._rgb: dict[str, dict] = {}
        self._pose: dict[str, dict] = {}
        splash_path = self.paths.feature_store / "splash" / "splash_features.parquet"
        self.splash = pd.read_parquet(splash_path).set_index("clip_uid") if splash_path.exists() else None

    def rgb(self, clip_uid: str) -> dict:
        if clip_uid not in self._rgb:
            key = self.manifest.loc[clip_uid, "feature_key"]
            path = self.paths.feature_store / "rgb" / f"{key}.npz"
            if not path.exists():
                raise FileNotFoundError(f"RGB feature missing for {clip_uid}: {path}")
            with np.load(path) as payload:
                self._rgb[clip_uid] = {
                    "temporal": payload["temporal"].astype(np.float32),
                    "global_feature": payload["global_feature"].astype(np.float32),
                }
        return self._rgb[clip_uid]

    def pose(self, clip_uid: str) -> dict | None:
        if clip_uid not in self._pose:
            key = self.manifest.loc[clip_uid, "feature_key"]
            path = self.paths.feature_store / "pose" / f"{key}.npz"
            if not path.exists():
                return None
            with np.load(path) as payload:
                self._pose[clip_uid] = {"concepts": payload["concepts"].astype(np.float32)}
        return self._pose.get(clip_uid)


def build_feature_table(manifest: pd.DataFrame, paths: Paths | None = None) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    repository = FeatureRepository(manifest, paths)
    global_features = {}
    temporal_features = {}
    phase_targets = {}
    pose_features = {}
    for row in manifest.itertuples(index=False):
        rgb = repository.rgb(row.clip_uid)
        transitions = json.loads(row.transition_frames_json)
        targets = ordered_phase_targets(row.frame_count, transitions, len(rgb["temporal"]))
        global_features[row.clip_uid] = rgb["global_feature"]
        phase_targets[row.clip_uid] = targets
        temporal_features[row.clip_uid] = rgb["temporal"]
        pose = repository.pose(row.clip_uid)
        if pose is None:
            pose_features[row.clip_uid] = np.full((3, 6), np.nan, dtype=np.float32)
        else:
            pose_sequence = pose["concepts"]
            pose_labels = ordered_phase_targets(len(pose_sequence), [len(pose_sequence) // 3, 2 * len(pose_sequence) // 3], len(pose_sequence))
            pose_features[row.clip_uid] = phase_pool(pose_sequence, pose_labels)
    return manifest, {
        "global": np.stack([global_features[x] for x in manifest.clip_uid]),
        "temporal": np.stack([temporal_features[x] for x in manifest.clip_uid]),
        "pose": np.stack([pose_features[x] for x in manifest.clip_uid]),
        "phase_targets": phase_targets,
    }


def train_phase_parser(
    temporal: np.ndarray,
    target: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    import torch
    import torch.nn as nn

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class PhaseParser(nn.Module):
        def __init__(self, feature_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(128, 3, kernel_size=1),
            )

        def forward(self, x):
            return self.net(x.transpose(1, 2)).transpose(1, 2)

    x = torch.from_numpy(temporal.astype(np.float32)).to(device)
    y = torch.from_numpy(target.astype(np.int64)).to(device)
    fit = torch.from_numpy(fit_indices).long().to(device)
    validation = torch.from_numpy(validation_indices).long().to(device)
    model = PhaseParser(temporal.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    stale = 0
    for _ in range(150):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits[fit].reshape(-1, 3), y[fit].reshape(-1))
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x[validation])
            val_loss = float(
                nn.functional.cross_entropy(val_logits.reshape(-1, 3), y[validation].reshape(-1)).cpu()
            )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 20:
                break
    if best_state is None:
        raise RuntimeError("Phase parser failed to train")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x).cpu().numpy()
    predictions = np.stack([monotonic_decode(item) for item in logits])
    accuracy = float(np.mean(predictions == target))
    boundary_errors = []
    for predicted, truth in zip(predictions, target):
        pred_bounds = [int(np.argmax(predicted == phase)) for phase in (1, 2)]
        true_bounds = [int(np.argmax(truth == phase)) for phase in (1, 2)]
        boundary_errors.extend(abs(a - b) / len(truth) for a, b in zip(pred_bounds, true_bounds))
    return predictions, {
        "validation_loss": best_loss,
        "frame_accuracy": accuracy,
        "mean_normalized_boundary_error": float(np.mean(boundary_errors)),
    }


def _cosine_neighbors(
    frame: pd.DataFrame,
    features: np.ndarray,
    fit_indices: np.ndarray,
    query_index: int,
    count: int,
) -> list[int]:
    query = frame.iloc[query_index]
    candidates = [
        index
        for index in fit_indices
        if frame.iloc[index].action_type == query.action_type
        and frame.iloc[index].event_family != query.event_family
        and index != query_index
    ]
    if not candidates:
        return []
    distance = cdist(features[[query_index]], features[candidates], metric="cosine")[0]
    order = np.argsort(distance)
    return [candidates[int(i)] for i in order[:count]]


def build_reference_inputs(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    fit_indices: np.ndarray,
    count: int,
) -> dict[str, np.ndarray | list[list[int]]]:
    rgb_delta = []
    concept_delta = []
    base_quality = []
    reference_indices = []
    open_set = []
    splash = np.zeros((len(frame), 4), dtype=np.float32)
    splash_valid = np.zeros(len(frame), dtype=np.float32)
    repository = FeatureRepository(frame)
    if repository.splash is not None:
        for i, clip in enumerate(frame.clip_uid):
            if clip in repository.splash.index:
                row = repository.splash.loc[clip]
                if bool(row.splash_valid):
                    splash[i] = [row.splash_peak, row.splash_auc, row.splash_duration, row.splash_expansion]
                    splash_valid[i] = 1.0
    for index in range(len(frame)):
        refs = _cosine_neighbors(frame, arrays["global"], fit_indices, index, count)
        reference_indices.append(refs)
        open_set.append(len(refs) < 3)
        if refs:
            base_quality.append(float(np.median(frame.iloc[refs].execution_quality)))
            rgb_reference = np.median(arrays["phase"][refs], axis=0)
            pose_reference = np.nanmedian(arrays["pose"][refs], axis=0)
        else:
            base_quality.append(float(np.median(frame.iloc[fit_indices].execution_quality)))
            rgb_reference = np.median(arrays["phase"][fit_indices], axis=0)
            pose_reference = np.nanmedian(arrays["pose"][fit_indices], axis=0)
        rgb_delta.append(arrays["phase"][index] - rgb_reference)
        delta = arrays["pose"][index] - pose_reference
        valid = np.isfinite(delta).astype(np.float32)
        delta = np.nan_to_num(delta, nan=0.0)
        concept_delta.append(np.concatenate([delta, valid], axis=1))
    concept = np.stack(concept_delta)
    # Splash is an entry-phase concept and is explicitly accompanied by validity.
    concept = np.concatenate(
        [concept, np.zeros((len(frame), 3, 5), dtype=np.float32)], axis=2
    )
    concept[:, 2, -5:-1] = splash
    concept[:, 2, -1] = splash_valid
    return {
        "rgb_delta": np.stack(rgb_delta).astype(np.float32),
        "concept_delta": concept.astype(np.float32),
        "base_quality": np.asarray(base_quality, dtype=np.float32),
        "references": reference_indices,
        "open_set": np.asarray(open_set, dtype=bool),
    }


def _torch_models(rgb_dim: int, concept_dim: int):
    import torch
    import torch.nn as nn

    class GlobalMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(rgb_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

        def forward(self, rgb):
            return self.net(rgb).squeeze(-1)

    class RelativeRGB(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3 * rgb_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)
            )

        def forward(self, rgb_delta, base):
            return base + self.net(rgb_delta.flatten(1)).squeeze(-1)

    class TrustDive(nn.Module):
        def __init__(self):
            super().__init__()
            self.concept_head = nn.Sequential(nn.Linear(concept_dim, 32), nn.GELU(), nn.Linear(32, 1))
            self.residual = nn.Sequential(
                nn.Linear(3 * rgb_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)
            )

        def forward(self, rgb_delta, concepts, base):
            contributions = self.concept_head(concepts).squeeze(-1)
            residual = torch.tanh(self.residual(rgb_delta.flatten(1)).squeeze(-1))
            prediction = base + contributions.sum(dim=1) + residual
            return prediction, contributions, residual

    return GlobalMLP, RelativeRGB, TrustDive


def _train_model(
    model,
    inputs: dict[str, np.ndarray],
    target: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    mode: str,
    seed: int,
    max_epochs: int = 200,
):
    import torch

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tensors = {key: torch.from_numpy(value).float().to(device) for key, value in inputs.items()}
    y = torch.from_numpy(target.astype(np.float32)).to(device)
    fit = torch.from_numpy(fit_indices).long().to(device)
    validation = torch.from_numpy(validation_indices).long().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    patience = 25
    stale = 0
    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if mode == "global":
            prediction = model(tensors["global"])
            penalty = 0.0
        elif mode == "relative":
            prediction = model(tensors["rgb_delta"], tensors["base"])
            penalty = 0.0
        else:
            prediction, _, residual = model(tensors["rgb_delta"], tensors["concepts"], tensors["base"])
            penalty = 0.01 * residual[fit].abs().mean()
        loss = ((prediction[fit] - y[fit]) ** 2).mean() + penalty
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            if mode == "global":
                val_prediction = model(tensors["global"])
            elif mode == "relative":
                val_prediction = model(tensors["rgb_delta"], tensors["base"])
            else:
                val_prediction, _, _ = model(tensors["rgb_delta"], tensors["concepts"], tensors["base"])
            val_loss = float(((val_prediction[validation] - y[validation]) ** 2).mean().cpu())
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if mode == "global":
            prediction = model(tensors["global"])
            contributions = torch.zeros((len(target), 3), device=device)
            residual = torch.zeros(len(target), device=device)
        elif mode == "relative":
            prediction = model(tensors["rgb_delta"], tensors["base"])
            contributions = torch.zeros((len(target), 3), device=device)
            residual = prediction - tensors["base"]
        else:
            prediction, contributions, residual = model(tensors["rgb_delta"], tensors["concepts"], tensors["base"])
    return {
        "model": model,
        "prediction": prediction.cpu().numpy(),
        "contributions": contributions.cpu().numpy(),
        "residual": residual.cpu().numpy(),
        "validation_mse": best_loss,
    }


def _intervention_audit(model, input_arrays: dict[str, np.ndarray], test_indices: np.ndarray, seed: int) -> dict:
    import torch
    from scipy.stats import wilcoxon

    device = next(model.parameters()).device
    rgb = torch.from_numpy(input_arrays["rgb_delta"]).float().to(device)
    concepts = torch.from_numpy(input_arrays["concepts"]).float().to(device)
    base = torch.from_numpy(input_arrays["base"]).float().to(device)
    model.eval()
    with torch.no_grad():
        original, contributions, _ = model(rgb, concepts, base)
        phase_effects = []
        for phase in range(3):
            rgb_ablated = rgb.clone()
            concepts_ablated = concepts.clone()
            rgb_ablated[:, phase, :] = 0.0
            concepts_ablated[:, phase, :] = 0.0
            ablated, _, _ = model(rgb_ablated, concepts_ablated, base)
            phase_effects.append((original - ablated).abs())
        phase_effects = torch.stack(phase_effects, dim=1).cpu().numpy()
        contributions_np = contributions.cpu().numpy()

        splash_ablated = concepts.clone()
        splash_ablated[:, 2, -5:] = 0.0
        _, contributions_without_splash, _ = model(rgb, splash_ablated, base)
        nonentry_before = contributions[:, :2].cpu().numpy().reshape(-1)
        nonentry_after = contributions_without_splash[:, :2].cpu().numpy().reshape(-1)

    idx = np.asarray(test_indices, dtype=int)
    top_phase = np.argmax(np.abs(contributions_np[idx]), axis=1)
    rng = np.random.default_rng(seed)
    random_phase = np.asarray([rng.choice([p for p in range(3) if p != top]) for top in top_phase])
    targeted = phase_effects[idx, top_phase]
    random_effect = phase_effects[idx, random_phase]
    difference = targeted - random_effect
    try:
        p_value = float(wilcoxon(difference, alternative="greater").pvalue)
    except ValueError:
        p_value = 1.0
    rank_stability = float(spearmanr(nonentry_before, nonentry_after).statistic)
    return {
        "n": int(len(idx)),
        "targeted_median_change": float(np.median(targeted)),
        "random_median_change": float(np.median(random_effect)),
        "median_paired_difference": float(np.median(difference)),
        "one_sided_wilcoxon_p": p_value,
        "targeted_greater_than_random": bool(np.median(difference) > 0 and p_value < 0.05),
        "nonentry_rank_stability_without_splash": rank_stability,
        "intervention_level": "phase_feature_and_concept_deletion",
    }


def train_experiment(
    manifest: pd.DataFrame,
    paths: Paths | None = None,
    model_kind: str = "relative",
    seeds: list[int] | None = None,
    output_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    paths = paths or Paths()
    contract = load_contract(paths.contract)
    seeds = seeds or list(contract["random"]["model_seeds"])
    frame = add_analysis_roles(manifest, int(contract["random"]["master_seed"])).reset_index(drop=True)
    frame, arrays = build_feature_table(frame, paths)
    fit_indices = np.flatnonzero(frame.analysis_role.to_numpy() == "fit")
    validation_indices = np.flatnonzero(frame.analysis_role.to_numpy() == "validation")
    calibration_indices = np.flatnonzero(frame.analysis_role.to_numpy() == "calibration")
    test_indices = np.flatnonzero(frame.analysis_role.to_numpy() == "official_test")
    phase_target_array = np.stack([arrays["phase_targets"][clip] for clip in frame.clip_uid])
    predicted_phases, phase_metrics = train_phase_parser(
        arrays["temporal"],
        phase_target_array,
        fit_indices,
        validation_indices,
        int(contract["random"]["master_seed"]),
    )
    arrays["phase"] = np.stack(
        [phase_pool(sequence, labels) for sequence, labels in zip(arrays["temporal"], predicted_phases)]
    )
    reference = build_reference_inputs(
        frame, arrays, fit_indices, int(contract["score"]["reference_count"])
    )
    GlobalMLP, RelativeRGB, TrustDive = _torch_models(arrays["global"].shape[1], reference["concept_delta"].shape[2])
    input_arrays = {
        "global": arrays["global"].astype(np.float32),
        "rgb_delta": reference["rgb_delta"].astype(np.float32),
        "concepts": reference["concept_delta"].astype(np.float32),
        "base": reference["base_quality"].astype(np.float32),
    }
    target = frame.execution_quality.to_numpy(dtype=np.float32)
    run_label = output_name or model_kind
    seed_outputs = []
    seed_metrics = []
    for seed in seeds:
        if model_kind == "global":
            model = GlobalMLP()
        elif model_kind == "relative":
            model = RelativeRGB()
        elif model_kind == "trustdive":
            model = TrustDive()
        else:
            raise ValueError(f"Unknown model kind: {model_kind}")
        result = _train_model(
            model, input_arrays, target, fit_indices, validation_indices, model_kind, int(seed)
        )
        seed_outputs.append(result)
        import torch

        checkpoint_dir = paths.runs / run_label / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_kind": model_kind,
                "seed": int(seed),
                "state_dict": result["model"].state_dict(),
                "rgb_dim": int(arrays["global"].shape[1]),
                "concept_dim": int(reference["concept_delta"].shape[2]),
            },
            checkpoint_dir / f"seed_{int(seed)}.pth",
        )
        score_prediction = restore_total_score(result["prediction"][test_indices], frame.difficulty.iloc[test_indices])
        seed_metrics.append(score_metrics(frame.dive_score.iloc[test_indices], score_prediction))
    predictions = np.stack([x["prediction"] for x in seed_outputs])
    contributions = np.stack([x["contributions"] for x in seed_outputs])
    residuals = np.stack([x["residual"] for x in seed_outputs])
    mean_prediction = predictions.mean(axis=0)
    conformal = split_conformal_radius(
        target[calibration_indices], mean_prediction[calibration_indices], float(contract["score"]["interval_coverage"])
    )
    output = frame[["clip_uid", "feature_key", "official_split", "analysis_role", "difficulty", "dive_score"]].copy()
    output["prediction_role"] = output.analysis_role
    output["predicted_quality"] = mean_prediction
    output["predicted_score"] = restore_total_score(mean_prediction, frame.difficulty)
    output["lower_quality"] = mean_prediction - conformal
    output["upper_quality"] = mean_prediction + conformal
    output["lower_score"] = restore_total_score(output.lower_quality, frame.difficulty)
    output["upper_score"] = restore_total_score(output.upper_quality, frame.difficulty)
    output["phase_takeoff_contribution"] = contributions.mean(axis=0)[:, 0]
    output["phase_flight_contribution"] = contributions.mean(axis=0)[:, 1]
    output["phase_entry_contribution"] = contributions.mean(axis=0)[:, 2]
    output["residual"] = residuals.mean(axis=0)
    absolute_contribution = np.abs(contributions.mean(axis=0)).sum(axis=1)
    output["trace_coverage"] = absolute_contribution / (absolute_contribution + np.abs(output.residual) + 1e-12)
    output["open_set"] = reference["open_set"]
    output["reference_indices_json"] = [json.dumps([int(x) for x in refs]) for refs in reference["references"]]
    output["ensemble_sd"] = predictions.std(axis=0)
    output["review_recommended"] = (
        (output.upper_quality - output.lower_quality) > np.quantile(output.upper_quality - output.lower_quality, 0.8)
    ) | output.open_set
    destination = paths.results / "02_SCORE" / f"{run_label}_predictions.parquet"
    output.to_parquet(destination, index=False)
    summary = {
        "model_kind": model_kind,
        "seeds": [int(x) for x in seeds],
        "seed_metrics": seed_metrics,
        "spearman_mean": float(np.mean([x["spearman"] for x in seed_metrics])),
        "spearman_sd": float(np.std([x["spearman"] for x in seed_metrics], ddof=0)),
        "conformal_radius_quality": conformal,
        "official_test_n": int(len(test_indices)),
        "official_test_open_set": int(reference["open_set"][test_indices].sum()),
        "median_trace_coverage": float(np.median(output.trace_coverage.iloc[test_indices])),
        "phase_parser": phase_metrics,
        "formal_phase_source": "predicted_monotonic_labels",
    }
    if model_kind == "trustdive":
        summary["intervention_audit"] = _intervention_audit(
            seed_outputs[0]["model"], input_arrays, test_indices, int(seeds[0])
        )
    write_json(paths.results / "02_SCORE" / f"{run_label}_summary.json", summary)
    return output, summary
