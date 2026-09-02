from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from .config import RUNS_ROOT
from .metrics import restore_total_score, score_metrics
from .modeling import monotonic_decode, ordered_phase_targets, phase_pool
from .statistics import split_conformal_radius
from .util import set_seed, write_json
from .v2_data import V2_RESULTS_ROOT, load_v2_contract, v2_paths


@dataclass
class PreparedInputs:
    global_features: np.ndarray
    rgb_delta: np.ndarray
    base_quality: np.ndarray
    reference_distance: np.ndarray
    references: list[list[int]]
    open_set: np.ndarray
    predicted_phases: np.ndarray
    phase_targets: np.ndarray


def _feature_path(
    feature_key: str, augmented: bool = False, backbone: str = "i3d"
) -> Path:
    if backbone not in {"i3d", "videomae"}:
        raise ValueError(f"Unknown RGB backbone: {backbone}")
    folder = "videomae_v2" if backbone == "videomae" else "rgb_aug_v2" if augmented else "rgb"
    return v2_paths().feature_store / folder / f"{feature_key}.npz"


def feature_inventory(
    frame: pd.DataFrame, augmented_fit: bool = False, backbone: str = "i3d"
) -> dict:
    base = np.asarray(
        [_feature_path(key, backbone=backbone).exists() for key in frame.feature_key], dtype=bool
    )
    result = {"base_present": int(base.sum()), "base_missing": int((~base).sum())}
    if augmented_fit:
        fit = frame.analysis_role.to_numpy() == "fit"
        aug = np.asarray([_feature_path(key, True).exists() for key in frame.feature_key], dtype=bool)
        result.update(
            {
                "augmented_fit_present": int((aug & fit).sum()),
                "augmented_fit_missing": int((~aug & fit).sum()),
            }
        )
    return result


def load_rgb_arrays(
    frame: pd.DataFrame, use_augmented_fit: bool = False, backbone: str = "i3d"
) -> dict[str, np.ndarray]:
    temporal: list[np.ndarray] = []
    global_features: list[np.ndarray] = []
    baseline_temporal: list[np.ndarray] = []
    baseline_global: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        with np.load(_feature_path(row.feature_key, backbone=backbone)) as payload:
            base_t = payload["temporal"].astype(np.float32)
            base_g = payload["global_feature"].astype(np.float32)
        baseline_temporal.append(base_t)
        baseline_global.append(base_g)
        augmented_path = _feature_path(row.feature_key, True, backbone=backbone)
        if (
            backbone == "i3d"
            and use_augmented_fit
            and row.analysis_role == "fit"
            and augmented_path.exists()
        ):
            with np.load(augmented_path) as payload:
                temporal.append(payload["temporal"].astype(np.float32))
                global_features.append(payload["global_feature"].astype(np.float32))
        else:
            temporal.append(base_t)
            global_features.append(base_g)
    lengths = {len(x) for x in temporal}
    if len(lengths) != 1:
        raise AssertionError(f"RGB temporal features have inconsistent lengths: {sorted(lengths)}")
    return {
        "temporal": np.stack(temporal),
        "global": np.stack(global_features),
        "baseline_temporal": np.stack(baseline_temporal),
        "baseline_global": np.stack(baseline_global),
    }


def _phase_boundary_error(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    output = np.zeros(len(predicted), dtype=np.float32)
    for index, (pred, target) in enumerate(zip(predicted, truth)):
        errors = []
        for phase in (1, 2):
            p = int(np.flatnonzero(pred == phase)[0]) if np.any(pred == phase) else len(pred) - 1
            t = int(np.flatnonzero(target == phase)[0]) if np.any(target == phase) else len(target) - 1
            errors.append(abs(p - t) / max(len(target), 1))
        output[index] = float(np.mean(errors))
    return output


def train_or_load_phase_parser(
    frame: pd.DataFrame,
    temporal: np.ndarray,
    allow_test_metrics: bool,
    role_column: str = "analysis_role",
    fit_role: str = "fit",
    validation_role: str = "validation",
    evaluation_role: str = "official_test",
    backbone: str = "i3d",
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    import torch.nn as nn

    contract = load_v2_contract()
    split_suffix = "source" if role_column == "source_role" else "official"
    cache_suffix = f"{backbone}_{split_suffix}_v2"
    cache = V2_RESULTS_ROOT / "01_FEATURES" / f"phase_predictions_{cache_suffix}.npz"
    targets = np.stack(
        [
            ordered_phase_targets(
                int(row.frame_count), json.loads(row.transition_frames_json), temporal.shape[1]
            )
            for row in frame.itertuples(index=False)
        ]
    )
    if cache.exists():
        with np.load(cache) as payload:
            predictions = payload["predictions"].astype(np.int64)
        if predictions.shape != targets.shape:
            raise AssertionError("Cached v2 phase predictions do not match current features")
    else:
        seed = int(contract["random"]["master_seed"])
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class Parser(nn.Module):
            def __init__(self, feature_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(feature_dim, 128, 3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(128, 128, 3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(128, 3, 1),
                )

            def forward(self, x):
                return self.net(x.transpose(1, 2)).transpose(1, 2)

        x = torch.from_numpy(temporal.astype(np.float32)).to(device)
        y = torch.from_numpy(targets.astype(np.int64)).to(device)
        fit_idx = np.flatnonzero(frame[role_column].to_numpy() == fit_role)
        val_idx = np.flatnonzero(frame[role_column].to_numpy() == validation_role)
        fit = torch.from_numpy(fit_idx).long().to(device)
        val = torch.from_numpy(val_idx).long().to(device)
        model = Parser(temporal.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_state = None
        best_loss = float("inf")
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
                val_loss = float(
                    nn.functional.cross_entropy(
                        model(x[val]).reshape(-1, 3), y[val].reshape(-1)
                    ).cpu()
                )
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= 20:
                    break
        if best_state is None:
            raise RuntimeError("The v2 phase parser failed to produce a checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = model(x).cpu().numpy()
        predictions = np.stack([monotonic_decode(item) for item in logits])
        np.savez_compressed(cache, predictions=predictions.astype(np.int8))
        checkpoint = (
            RUNS_ROOT
            / "v2_disagreement"
            / "checkpoints"
            / f"phase_parser_{cache_suffix}.pth"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "feature_dim": temporal.shape[-1]}, checkpoint)
    validation = frame[role_column].to_numpy() == validation_role
    metrics = {
        "validation_frame_accuracy": float(np.mean(predictions[validation] == targets[validation])),
        "validation_mean_normalized_boundary_error": float(
            _phase_boundary_error(predictions[validation], targets[validation]).mean()
        ),
        "formal_phase_source": "predicted_monotonic_labels",
        "official_test_metrics_revealed": bool(allow_test_metrics),
    }
    if allow_test_metrics:
        test = frame[role_column].to_numpy() == evaluation_role
        metrics.update(
            {
                "official_test_frame_accuracy": float(np.mean(predictions[test] == targets[test])),
                "official_test_mean_normalized_boundary_error": float(
                    _phase_boundary_error(predictions[test], targets[test]).mean()
                ),
            }
        )
    return predictions, targets, metrics


def _jittered_pool(
    sequence: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    jitter_fraction: float,
    drop_fraction: float,
) -> np.ndarray:
    length = len(labels)
    bounds = [int(np.flatnonzero(labels == p)[0]) for p in (1, 2)]
    maximum = max(1, int(round(length * jitter_fraction)))
    first = int(np.clip(bounds[0] + rng.integers(-maximum, maximum + 1), 1, length - 2))
    second = int(np.clip(bounds[1] + rng.integers(-maximum, maximum + 1), first + 1, length - 1))
    jittered = np.zeros(length, dtype=np.int64)
    jittered[first:second] = 1
    jittered[second:] = 2
    pooled = []
    for phase in range(3):
        selected = sequence[jittered == phase]
        keep = rng.random(len(selected)) >= drop_fraction
        if np.any(keep):
            selected = selected[keep]
        pooled.append(selected.mean(axis=0) if len(selected) else np.zeros(sequence.shape[1]))
    return np.stack(pooled).astype(np.float32)


def _reference_inputs(
    frame: pd.DataFrame,
    baseline_global: np.ndarray,
    baseline_phase: np.ndarray,
    query_global: np.ndarray,
    query_phase: np.ndarray,
    fit_indices: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], np.ndarray]:
    del query_global  # Query retrieval deliberately uses the frozen baseline view.
    deltas = []
    bases = []
    distances = []
    all_refs: list[list[int]] = []
    open_set = []
    actions = frame.action_type.to_numpy()
    families = frame.event_family.to_numpy()
    qualities = frame.execution_quality.to_numpy(dtype=np.float32)
    fit_actions = actions[fit_indices]
    fit_families = families[fit_indices]
    for index in range(len(frame)):
        candidate_mask = (fit_actions == actions[index]) & (fit_families != families[index])
        candidate_mask &= fit_indices != index
        candidates = fit_indices[candidate_mask].astype(int).tolist()
        if candidates:
            distance = cdist(
                baseline_global[[index]], baseline_global[candidates], metric="cosine"
            )[0]
            order = np.argsort(distance)[:count]
            refs = [candidates[int(i)] for i in order]
            ref_distance = float(np.mean(distance[order]))
        else:
            refs = []
            ref_distance = 1.0
        all_refs.append(refs)
        open_set.append(len(refs) < 3)
        if refs:
            bases.append(float(np.median(qualities[refs])))
            reference = np.median(baseline_phase[refs], axis=0)
        else:
            bases.append(float(np.median(qualities[fit_indices])))
            reference = np.median(baseline_phase[fit_indices], axis=0)
        deltas.append(query_phase[index] - reference)
        distances.append(ref_distance)
    return (
        np.stack(deltas).astype(np.float32),
        np.asarray(bases, dtype=np.float32),
        np.asarray(distances, dtype=np.float32),
        all_refs,
        np.asarray(open_set, dtype=bool),
    )


def _robust_standardize(values: np.ndarray, fit_indices: np.ndarray) -> tuple[np.ndarray, dict]:
    fit = values[fit_indices]
    median = np.median(fit, axis=0)
    mad = np.median(np.abs(fit - median), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(fit, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(fallback > 1e-6, fallback, 1.0))
    transformed = np.clip((values - median) / scale, -12.0, 12.0).astype(np.float32)
    return transformed, {
        "median_abs": float(np.median(np.abs(median))),
        "scale_median": float(np.median(scale)),
        "clipped_fraction": float(np.mean(np.abs(transformed) >= 12.0)),
    }


def prepare_inputs(
    frame: pd.DataFrame,
    variant: str,
    allow_test_metrics: bool,
    split_column: str = "analysis_role",
    backbone: str = "i3d",
    reference_roles: tuple[str, ...] | None = None,
) -> tuple[PreparedInputs, dict]:
    use_visual = backbone == "i3d" and variant in {"temporal_visual", "temporal_visual_dropout"}
    arrays = load_rgb_arrays(
        frame,
        use_augmented_fit=use_visual and split_column == "analysis_role",
        backbone=backbone,
    )
    predicted, targets, phase_metrics = train_or_load_phase_parser(
        frame,
        arrays["baseline_temporal"],
        allow_test_metrics,
        role_column=split_column,
        fit_role="source_fit" if split_column == "source_role" else "fit",
        validation_role="source_validation" if split_column == "source_role" else "validation",
        evaluation_role="source_test" if split_column == "source_role" else "official_test",
        backbone=backbone,
    )
    baseline_phase = np.stack(
        [phase_pool(sequence, labels) for sequence, labels in zip(arrays["baseline_temporal"], predicted)]
    ).astype(np.float32)
    query_phase = baseline_phase.copy()
    if reference_roles is None:
        reference_roles = ("source_fit",) if split_column == "source_role" else ("fit",)
    fit_indices = np.flatnonzero(frame[split_column].isin(reference_roles).to_numpy())
    temporal_enabled = variant in {"temporal", "temporal_visual", "temporal_visual_dropout"}
    if temporal_enabled:
        config = load_v2_contract()["model"]
        rng = np.random.default_rng(int(load_v2_contract()["random"]["master_seed"]))
        for index in fit_indices:
            query_phase[index] = _jittered_pool(
                arrays["temporal"][index],
                predicted[index],
                rng,
                float(config["temporal_boundary_jitter_fraction"]),
                float(config["temporal_frame_drop_fraction"]),
            )
    reference_count = int(load_v2_contract()["score"]["reference_count"])
    delta, base, distance, refs, open_set = _reference_inputs(
        frame,
        arrays["baseline_global"],
        baseline_phase,
        arrays["global"],
        query_phase,
        fit_indices,
        reference_count,
    )
    global_scaled, global_scaling = _robust_standardize(arrays["global"], fit_indices)
    delta_scaled, delta_scaling = _robust_standardize(delta, fit_indices)
    prepared = PreparedInputs(
        global_features=global_scaled,
        rgb_delta=delta_scaled,
        base_quality=base,
        reference_distance=distance,
        references=refs,
        open_set=open_set,
        predicted_phases=predicted,
        phase_targets=targets,
    )
    return prepared, {
        "phase_parser": phase_metrics,
        "global_scaling": global_scaling,
        "delta_scaling": delta_scaling,
        "variant": variant,
        "backbone": backbone,
        "visual_augmented_fit": use_visual,
        "reference_roles": list(reference_roles),
    }


def _model_classes(global_dim: int, phase_dim: int, dropout: float):
    import torch
    import torch.nn as nn

    class GlobalMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(global_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)
            )

        def forward(self, global_features, *_):
            prediction = self.net(global_features).squeeze(-1)
            zeros = torch.zeros((len(prediction), 3), device=prediction.device)
            return prediction, zeros, torch.zeros_like(prediction), torch.ones_like(prediction) * 0.5

    class RelativeRGB(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3 * phase_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)
            )

        def forward(self, _, phase_delta, base):
            residual = self.net(phase_delta.flatten(1)).squeeze(-1)
            prediction = base + residual
            zeros = torch.zeros((len(prediction), 3), device=prediction.device)
            return prediction, zeros, residual, torch.ones_like(prediction) * 0.5

    class PhaseRelative(nn.Module):
        def __init__(self, with_distribution: bool):
            super().__init__()
            self.with_distribution = with_distribution
            self.phase_dropout = nn.Dropout(dropout)
            self.phase_head = nn.Sequential(
                nn.Linear(phase_dim, 96), nn.GELU(), nn.Linear(96, 1)
            )
            self.residual_head = nn.Sequential(
                nn.Linear(3 * phase_dim + global_dim, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
            )
            self.scale_head = nn.Sequential(
                nn.Linear(3 * phase_dim, 128), nn.GELU(), nn.Linear(128, 1)
            )

        def forward(self, global_features, phase_delta, base):
            encoded = self.phase_dropout(phase_delta)
            contributions = self.phase_head(encoded).squeeze(-1)
            fused = torch.cat([encoded.flatten(1), global_features], dim=1)
            residual = torch.tanh(self.residual_head(fused).squeeze(-1))
            prediction = base + contributions.sum(dim=1) + residual
            if self.with_distribution:
                sigma = torch.nn.functional.softplus(self.scale_head(encoded.flatten(1)).squeeze(-1)) + 0.05
            else:
                sigma = torch.ones_like(prediction) * 0.5
            return prediction, contributions, residual, sigma

    return GlobalMLP, RelativeRGB, PhaseRelative


def _judge_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.full((len(frame), 7), np.nan, dtype=np.float32)
    eligible = frame.disagreement_primary_eligible.to_numpy(dtype=bool)
    for index in np.flatnonzero(eligible):
        values = np.asarray(json.loads(frame.iloc[index].judge_scores_json), dtype=np.float32)
        matrix[index, : len(values)] = values
    return matrix, eligible, frame.judge_sample_sd.to_numpy(dtype=np.float32)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    value = float(spearmanr(a, b).statistic)
    return 0.0 if not np.isfinite(value) else value


def train_single_model(
    frame: pd.DataFrame,
    inputs: PreparedInputs,
    model_kind: str,
    seed: int,
    variant: str,
    fit_role: str = "fit",
    validation_role: str = "validation",
    role_column: str = "analysis_role",
    disagreement_rescue: bool = False,
    fit_roles: tuple[str, ...] | None = None,
    fixed_epochs: int | None = None,
) -> dict:
    import torch
    import torch.nn as nn

    contract = load_v2_contract()
    set_seed(seed)
    fit_roles = fit_roles or (fit_role,)
    fit_indices = np.flatnonzero(frame[role_column].isin(fit_roles).to_numpy())
    validation_indices = np.flatnonzero(frame[role_column].to_numpy() == validation_role)
    dropout = (
        float(contract["model"]["phase_feature_dropout"])
        if variant == "temporal_visual_dropout"
        else 0.0
    )
    GlobalMLP, RelativeRGB, PhaseRelative = _model_classes(
        inputs.global_features.shape[1], inputs.rgb_delta.shape[2], dropout
    )
    if model_kind == "global":
        model = GlobalMLP()
    elif model_kind == "relative":
        model = RelativeRGB()
    elif model_kind == "phase_relative":
        model = PhaseRelative(False)
    elif model_kind == "trustdive_d":
        model = PhaseRelative(True)
    else:
        raise ValueError(f"Unknown v2 model kind: {model_kind}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    global_tensor = torch.from_numpy(inputs.global_features).float().to(device)
    phase_tensor = torch.from_numpy(inputs.rgb_delta).float().to(device)
    base_tensor = torch.from_numpy(inputs.base_quality).float().to(device)
    target = torch.from_numpy(frame.execution_quality.to_numpy(dtype=np.float32)).to(device)
    fit = torch.from_numpy(fit_indices).long().to(device)
    validation = torch.from_numpy(validation_indices).long().to(device)
    judge_matrix, judge_eligible, judge_sd = _judge_matrix(frame)
    judges = torch.from_numpy(judge_matrix).float().to(device)
    eligible_fit_np = np.intersect1d(fit_indices, np.flatnonzero(judge_eligible))
    eligible_fit = torch.from_numpy(eligible_fit_np).long().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    patience = 25
    maximum_epochs = int(fixed_epochs) if fixed_epochs is not None else 200
    for epoch in range(maximum_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, contributions, residual, sigma = model(global_tensor, phase_tensor, base_tensor)
        score_loss = nn.functional.huber_loss(prediction[fit], target[fit], delta=1.0)
        loss = score_loss
        if model_kind == "trustdive_d" and len(eligible_fit_np):
            distribution = torch.distributions.StudentT(
                df=float(contract["model"]["student_t_df"]),
                loc=prediction[eligible_fit, None],
                scale=sigma[eligible_fit, None],
            )
            judge_nll = -distribution.log_prob(judges[eligible_fit]).mean()
            loss = loss + float(contract["model"]["judge_nll_weight"]) * judge_nll
            if disagreement_rescue and len(eligible_fit_np) >= 8:
                ordered = eligible_fit_np[np.argsort(judge_sd[eligible_fit_np])]
                half = len(ordered) // 2
                low = torch.from_numpy(ordered[:half]).long().to(device)
                high = torch.from_numpy(ordered[-half:]).long().to(device)
                rank_loss = torch.relu(0.05 - (sigma[high] - sigma[low])).mean()
                loss = loss + float(contract["model"]["disagreement_rank_weight_rescue"]) * rank_loss
        if model_kind in {"phase_relative", "trustdive_d"}:
            loss = loss + float(contract["score"]["residual_l1_weight"]) * residual[fit].abs().mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prediction, _, _, _ = model(global_tensor, phase_tensor, base_tensor)
            val_loss = float(
                nn.functional.huber_loss(
                    val_prediction[validation], target[validation], delta=1.0
                ).cpu()
            )
        if fixed_epochs is not None:
            best_score = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            continue
        if val_loss < best_score - 1e-6:
            best_score = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("v2 model training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction, contributions, residual, sigma = model(global_tensor, phase_tensor, base_tensor)
        ablations = []
        for phase in range(3):
            ablated = phase_tensor.clone()
            ablated[:, phase] = 0.0
            ablated_prediction, _, _, _ = model(global_tensor, ablated, base_tensor)
            ablations.append(ablated_prediction)
    prediction_np = prediction.cpu().numpy()
    contribution_np = contributions.cpu().numpy()
    residual_np = residual.cpu().numpy()
    sigma_np = sigma.cpu().numpy()
    validation_scores = score_metrics(
        frame.dive_score.iloc[validation_indices],
        restore_total_score(prediction_np[validation_indices], frame.difficulty.iloc[validation_indices]),
    )
    eligible_validation = np.intersect1d(validation_indices, np.flatnonzero(judge_eligible))
    disagreement_rho = _safe_spearman(
        sigma_np[eligible_validation], judge_sd[eligible_validation]
    )
    return {
        "model": model,
        "prediction": prediction_np,
        "contributions": contribution_np,
        "residual": residual_np,
        "sigma_judge": sigma_np,
        "ablated_predictions": np.stack([x.cpu().numpy() for x in ablations], axis=1),
        "validation_score": validation_scores,
        "validation_disagreement_spearman": disagreement_rho,
        "validation_disagreement_n": int(len(eligible_validation)),
        "validation_loss": best_score,
        "best_epoch": int(best_epoch),
        "state_dict": best_state,
    }


def _save_validation_predictions(
    frame: pd.DataFrame,
    result: dict,
    inputs: PreparedInputs,
    destination: Path,
) -> None:
    validation = frame.analysis_role == "validation"
    output = frame.loc[
        validation,
        ["clip_uid", "analysis_role", "difficulty", "dive_score", "judge_sample_sd"],
    ].copy()
    indices = np.flatnonzero(validation.to_numpy())
    output["predicted_quality"] = result["prediction"][indices]
    output["predicted_score"] = restore_total_score(output.predicted_quality, output.difficulty)
    output["predicted_judge_sd"] = result["sigma_judge"][indices]
    output["reference_distance"] = inputs.reference_distance[indices]
    output.to_parquet(destination, index=False)


def train_baselines_v2(frame: pd.DataFrame) -> dict:
    contract = load_v2_contract()
    seed = int(contract["random"]["model_seeds"][0])
    inputs, preparation = prepare_inputs(frame, "none", allow_test_metrics=False)
    summaries = {}
    for model_kind in ("global", "relative", "phase_relative"):
        result = train_single_model(frame, inputs, model_kind, seed, "none")
        _save_validation_predictions(
            frame,
            result,
            inputs,
            V2_RESULTS_ROOT / "02_TUNING" / f"baseline_{model_kind}_validation_v2.parquet",
        )
        checkpoint = RUNS_ROOT / "v2_disagreement" / "checkpoints" / f"baseline_{model_kind}_v2.pth"
        import torch

        torch.save(
            {"state_dict": result["state_dict"], "model_kind": model_kind, "seed": seed},
            checkpoint,
        )
        summaries[model_kind] = {
            "validation_score": result["validation_score"],
            "validation_disagreement_spearman": result["validation_disagreement_spearman"],
            "validation_disagreement_n": result["validation_disagreement_n"],
        }
    output = {
        "status": "PASS",
        "official_test_metrics_revealed": False,
        "seed": seed,
        "models": summaries,
        "best_validation_spearman": max(
            value["validation_score"]["spearman"] for value in summaries.values()
        ),
        "preparation": preparation,
    }
    write_json(V2_RESULTS_ROOT / "02_TUNING" / "baseline_summary_v2.json", output)
    return output


def tune_v2(frame: pd.DataFrame) -> dict:
    baseline_path = V2_RESULTS_ROOT / "02_TUNING" / "baseline_summary_v2.json"
    if not baseline_path.exists():
        raise RuntimeError("Run v2 baselines before tuning")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    contract = load_v2_contract()
    seed = int(contract["random"]["model_seeds"][0])
    rows = []
    cached_results: dict[str, tuple[dict, PreparedInputs]] = {}
    for variant in contract["tuning"]["variants"]:
        inputs, preparation = prepare_inputs(frame, str(variant), allow_test_metrics=False)
        result = train_single_model(frame, inputs, "trustdive_d", seed, str(variant))
        cached_results[str(variant)] = (result, inputs)
        row = {
            "variant": str(variant),
            "backbone": "i3d",
            "model_kind": "trustdive_d",
            "rescue": False,
            "validation_spearman": result["validation_score"]["spearman"],
            "validation_relative_l2": result["validation_score"]["relative_l2"],
            "validation_mae": result["validation_score"]["mae"],
            "validation_disagreement_spearman": result["validation_disagreement_spearman"],
            "validation_disagreement_n": result["validation_disagreement_n"],
            "official_test_metrics_revealed": False,
            "visual_augmented_fit": preparation["visual_augmented_fit"],
        }
        rows.append(row)
    trials = pd.DataFrame(rows)
    baseline_rho = float(baseline["best_validation_spearman"])
    score_floor = baseline_rho - float(
        contract["tuning"]["maximum_spearman_drop_from_best_baseline"]
    )
    candidates = trials[trials.validation_spearman >= score_floor]
    if candidates.empty:
        selected_index = int(trials.validation_spearman.idxmax())
    else:
        selected_index = int(
            candidates.sort_values(
                ["validation_disagreement_spearman", "validation_mae"],
                ascending=[False, True],
            ).index[0]
        )
    selected = trials.loc[selected_index].to_dict()
    if float(selected["validation_disagreement_spearman"]) < float(
        contract["tuning"]["disagreement_rescue_threshold"]
    ):
        variant = str(selected["variant"])
        inputs = cached_results[variant][1]
        rescued = train_single_model(
            frame, inputs, "trustdive_d", seed, variant, disagreement_rescue=True
        )
        rescue_row = {
            "variant": f"{variant}_disagreement_rank_rescue",
            "backbone": "i3d",
            "model_kind": "trustdive_d",
            "base_variant": variant,
            "rescue": True,
            "validation_spearman": rescued["validation_score"]["spearman"],
            "validation_relative_l2": rescued["validation_score"]["relative_l2"],
            "validation_mae": rescued["validation_score"]["mae"],
            "validation_disagreement_spearman": rescued[
                "validation_disagreement_spearman"
            ],
            "validation_disagreement_n": rescued["validation_disagreement_n"],
            "official_test_metrics_revealed": False,
            "visual_augmented_fit": variant in {"temporal_visual", "temporal_visual_dropout"},
        }
        trials = pd.concat([trials, pd.DataFrame([rescue_row])], ignore_index=True)
        if (
            rescue_row["validation_spearman"] >= score_floor
            and rescue_row["validation_disagreement_spearman"]
            > float(selected["validation_disagreement_spearman"])
        ):
            selected = rescue_row
            cached_results[rescue_row["variant"]] = (rescued, inputs)
    trials.to_csv(V2_RESULTS_ROOT / "02_TUNING" / "augmentation_trials_v2.csv", index=False)
    selected_variant = str(selected.get("base_variant", selected["variant"]))
    selected_result, selected_inputs = cached_results[str(selected["variant"])]
    _save_validation_predictions(
        frame,
        selected_result,
        selected_inputs,
        V2_RESULTS_ROOT / "02_TUNING" / "selected_validation_predictions_v2.parquet",
    )
    score_rescue_required = bool(
        float(selected["validation_spearman"])
        < float(contract["tuning"]["minimum_validation_spearman"])
        or float(selected["validation_spearman"]) < score_floor
    )
    selection = {
        "status": "SELECTED_WITH_SCORE_WARNING" if score_rescue_required else "SELECTED",
        "selected_variant": selected_variant,
        "selected_backbone": "i3d",
        "disagreement_rank_rescue": bool(selected["rescue"]),
        "selection_record": selected,
        "best_internal_baseline_validation_spearman": baseline_rho,
        "score_noninferiority_floor": score_floor,
        "videomae_rescue_triggered": score_rescue_required,
        "videomae_rescue_status": "DEFERRED_UNTIL_TRIGGER_REVIEW" if score_rescue_required else "NOT_NEEDED",
        "official_test_metrics_revealed": False,
    }
    write_json(V2_RESULTS_ROOT / "02_TUNING" / "selected_config_v2.json", selection)
    return selection


def tune_videomae_rescue(frame: pd.DataFrame) -> dict:
    """Run the one prespecified frozen-VideoMAE rescue without touching test labels."""

    tuning_dir = V2_RESULTS_ROOT / "02_TUNING"
    selection_path = tuning_dir / "selected_config_v2.json"
    trials_path = tuning_dir / "augmentation_trials_v2.csv"
    if not selection_path.exists() or not trials_path.exists():
        raise RuntimeError("Run the I3D v2 tuning stage before VideoMAE rescue")
    existing_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not existing_selection.get("videomae_rescue_triggered", False):
        return existing_selection
    inventory = feature_inventory(frame, backbone="videomae")
    if inventory["base_missing"]:
        raise RuntimeError("The full frozen VideoMAE feature inventory is incomplete")

    contract = load_v2_contract()
    seed = int(contract["random"]["model_seeds"][0])
    inputs, preparation = prepare_inputs(
        frame, "none", allow_test_metrics=False, backbone="videomae"
    )
    model_results: dict[tuple[str, bool], dict] = {}
    rows: list[dict] = []
    for model_kind, disagreement_rescue in (
        ("global", False),
        ("phase_relative", False),
        ("trustdive_d", False),
        ("trustdive_d", True),
    ):
        result = train_single_model(
            frame,
            inputs,
            model_kind,
            seed,
            "none",
            disagreement_rescue=disagreement_rescue,
        )
        model_results[(model_kind, disagreement_rescue)] = result
        name = f"videomae_{model_kind}"
        if disagreement_rescue:
            name += "_disagreement_rank_rescue"
        rows.append(
            {
                "variant": name,
                "base_variant": "none",
                "backbone": "videomae",
                "model_kind": model_kind,
                "rescue": disagreement_rescue,
                "validation_spearman": result["validation_score"]["spearman"],
                "validation_relative_l2": result["validation_score"]["relative_l2"],
                "validation_mae": result["validation_score"]["mae"],
                "validation_disagreement_spearman": result[
                    "validation_disagreement_spearman"
                ],
                "validation_disagreement_n": result["validation_disagreement_n"],
                "official_test_metrics_revealed": False,
                "visual_augmented_fit": False,
                "best_epoch": result["best_epoch"],
            }
        )

    trials = pd.read_csv(trials_path)
    if "backbone" not in trials:
        trials["backbone"] = "i3d"
    if "model_kind" not in trials:
        trials["model_kind"] = "trustdive_d"
    trials = trials[trials.backbone != "videomae"]
    trials = pd.concat([trials, pd.DataFrame(rows)], ignore_index=True, sort=False)

    baseline_summary = json.loads(
        (tuning_dir / "baseline_summary_v2.json").read_text(encoding="utf-8")
    )
    video_baselines = [
        row["validation_spearman"]
        for row in rows
        if row["model_kind"] in {"global", "phase_relative"}
    ]
    best_baseline = max(float(baseline_summary["best_validation_spearman"]), *video_baselines)
    score_floor = best_baseline - float(
        contract["tuning"]["maximum_spearman_drop_from_best_baseline"]
    )
    trust = trials[trials.model_kind == "trustdive_d"].copy()
    eligible = trust[trust.validation_spearman >= score_floor]
    if eligible.empty:
        selected = trust.sort_values(
            ["validation_spearman", "validation_disagreement_spearman"],
            ascending=[False, False],
        ).iloc[0]
    else:
        selected = eligible.sort_values(
            ["validation_disagreement_spearman", "validation_mae"], ascending=[False, True]
        ).iloc[0]

    trials.to_csv(trials_path, index=False)
    selected_backbone = str(selected.backbone)
    selected_rescue = bool(selected.rescue)
    base_variant = selected.get("base_variant", None)
    selected_variant = str(selected.variant if pd.isna(base_variant) else base_variant)
    if selected_backbone == "videomae":
        selected_result = model_results[("trustdive_d", selected_rescue)]
        _save_validation_predictions(
            frame,
            selected_result,
            inputs,
            tuning_dir / "selected_validation_predictions_v2.parquet",
        )
    score_warning = bool(
        float(selected.validation_spearman) < float(contract["tuning"]["minimum_validation_spearman"])
        or float(selected.validation_spearman) < score_floor
    )
    selection = {
        "status": "SELECTED_WITH_SCORE_WARNING" if score_warning else "SELECTED",
        "selected_variant": selected_variant,
        "selected_backbone": selected_backbone,
        "disagreement_rank_rescue": selected_rescue,
        "selection_record": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in selected.to_dict().items()
        },
        "best_internal_baseline_validation_spearman": best_baseline,
        "score_noninferiority_floor": score_floor,
        "videomae_rescue_triggered": True,
        "videomae_rescue_status": "COMPLETED",
        "videomae_preparation": preparation,
        "official_test_metrics_revealed": False,
    }
    write_json(selection_path, selection)
    return selection


def _ensemble_output(
    frame: pd.DataFrame,
    inputs: PreparedInputs,
    model_kind: str,
    variant: str,
    seeds: list[int],
    disagreement_rescue: bool,
    fixed_epochs: int,
) -> tuple[pd.DataFrame, dict]:
    results = []
    for seed in seeds:
        result = train_single_model(
            frame,
            inputs,
            model_kind,
            int(seed),
            variant,
            disagreement_rescue=disagreement_rescue,
            fit_roles=("fit", "validation"),
            fixed_epochs=fixed_epochs,
        )
        results.append(result)
        import torch

        checkpoint = (
            RUNS_ROOT
            / "v2_disagreement"
            / "checkpoints"
            / f"final_{model_kind}_{variant}_seed_{int(seed)}.pth"
        )
        torch.save(
            {
                "state_dict": result["state_dict"],
                "model_kind": model_kind,
                "variant": variant,
                "seed": int(seed),
            },
            checkpoint,
        )
    predictions = np.stack([result["prediction"] for result in results])
    mean_prediction = predictions.mean(axis=0)
    output = frame[
        [
            "clip_uid",
            "feature_key",
            "official_split",
            "analysis_role",
            "source_role",
            "difficulty",
            "dive_score",
            "execution_quality",
            "judge_count",
            "judge_scores_json",
            "judge_label_valid",
            "judge_sample_sd",
            "judge_pairwise_abs",
            "disagreement_primary_eligible",
        ]
    ].copy()
    output["predicted_quality"] = mean_prediction
    output["predicted_score"] = restore_total_score(mean_prediction, frame.difficulty)
    output["sigma_model"] = predictions.std(axis=0, ddof=0)
    output["sigma_judge"] = np.stack([r["sigma_judge"] for r in results]).mean(axis=0)
    contribution = np.stack([r["contributions"] for r in results]).mean(axis=0)
    residual = np.stack([r["residual"] for r in results]).mean(axis=0)
    output["phase_takeoff_contribution"] = contribution[:, 0]
    output["phase_flight_contribution"] = contribution[:, 1]
    output["phase_entry_contribution"] = contribution[:, 2]
    output["residual"] = residual
    output["base_quality"] = inputs.base_quality
    output["reference_distance"] = inputs.reference_distance
    output["open_set"] = inputs.open_set
    output["reference_indices_json"] = [stable for stable in (json.dumps(x) for x in inputs.references)]
    ablated = np.stack([r["ablated_predictions"] for r in results]).mean(axis=0)
    for phase, name in enumerate(("takeoff", "flight", "entry")):
        output[f"ablate_{name}_quality"] = ablated[:, phase]
    output["phase_boundary_error_normalized"] = _phase_boundary_error(
        inputs.predicted_phases, inputs.phase_targets
    )
    calibration = frame.analysis_role.to_numpy() == "calibration"
    radius = split_conformal_radius(
        frame.execution_quality.to_numpy()[calibration], mean_prediction[calibration], 0.90
    )
    output["lower_quality"] = mean_prediction - radius
    output["upper_quality"] = mean_prediction + radius
    risk_features = output[["sigma_model", "sigma_judge", "reference_distance", "open_set"]].astype(float)
    calibration_error = np.abs(
        output.loc[calibration, "predicted_quality"].to_numpy()
        - output.loc[calibration, "execution_quality"].to_numpy()
    )
    high_error = calibration_error >= np.quantile(
        calibration_error, float(load_v2_contract()["model"]["large_error_quantile"])
    )
    if len(np.unique(high_error)) == 2:
        risk_model = LogisticRegression(max_iter=1000, random_state=int(seeds[0]))
        risk_model.fit(risk_features.loc[calibration], high_error.astype(int))
        review_risk = risk_model.predict_proba(risk_features)[:, 1]
        coefficients = risk_model.coef_[0].tolist()
    else:
        review_risk = risk_features.sigma_model.to_numpy()
        coefficients = []
    threshold = float(
        np.quantile(
            review_risk[calibration],
            1.0 - float(load_v2_contract()["model"]["review_fraction"]),
        )
    )
    output["review_risk"] = review_risk
    output["review_recommended"] = (review_risk >= threshold) | output.open_set.to_numpy()
    test = frame.official_split.to_numpy() == "test"
    summary = {
        "model_kind": model_kind,
        "variant": variant,
        "seeds": [int(x) for x in seeds],
        "final_fit_roles": ["fit", "validation"],
        "fixed_epochs": int(fixed_epochs),
        "official_test_n": int(test.sum()),
        "official_test_score": score_metrics(
            frame.dive_score.to_numpy()[test], output.predicted_score.to_numpy()[test]
        ),
        "seed_official_test_scores": [
            score_metrics(
                frame.dive_score.to_numpy()[test],
                restore_total_score(r["prediction"][test], frame.difficulty.to_numpy()[test]),
            )
            for r in results
        ],
        "conformal_radius_quality": radius,
        "review_threshold": threshold,
        "review_model_coefficients": coefficients,
        "official_test_review_fraction": float(output.review_recommended.to_numpy()[test].mean()),
    }
    return output, summary


def train_final_v2(frame: pd.DataFrame) -> dict:
    selection = json.loads(
        (V2_RESULTS_ROOT / "02_TUNING" / "selected_config_v2.json").read_text(encoding="utf-8")
    )
    contract = load_v2_contract()
    if not contract["state"]["frozen"]:
        raise RuntimeError("Freeze the v2 contract before final training")
    variant = str(selection["selected_variant"])
    backbone = str(selection.get("selected_backbone", "i3d"))
    fixed_epochs = max(1, int(round(float(selection["selection_record"].get("best_epoch", 25)))))
    selected_inputs, preparation = prepare_inputs(
        frame,
        variant,
        allow_test_metrics=True,
        backbone=backbone,
        reference_roles=("fit", "validation"),
    )
    baseline_inputs, baseline_preparation = prepare_inputs(
        frame,
        "none",
        allow_test_metrics=True,
        backbone=backbone,
        reference_roles=("fit", "validation"),
    )
    summaries = {}
    for model_kind in ("global", "relative", "phase_relative", "trustdive_d"):
        inputs = selected_inputs if model_kind == "trustdive_d" else baseline_inputs
        output, summary = _ensemble_output(
            frame,
            inputs,
            model_kind,
            variant if model_kind in {"phase_relative", "trustdive_d"} else "none",
            [int(x) for x in contract["random"]["model_seeds"]],
            bool(selection["disagreement_rank_rescue"]) if model_kind == "trustdive_d" else False,
            fixed_epochs,
        )
        destination = V2_RESULTS_ROOT / "03_FINAL" / f"{model_kind}_predictions_v2.parquet"
        output.to_parquet(destination, index=False)
        write_json(V2_RESULTS_ROOT / "03_FINAL" / f"{model_kind}_summary_v2.json", summary)
        summaries[model_kind] = summary
        if model_kind == "trustdive_d":
            output.to_parquet(V2_RESULTS_ROOT / "03_FINAL" / "predictions_v2.parquet", index=False)
    best_baseline = max(
        summaries[k]["official_test_score"]["spearman"]
        for k in ("global", "relative", "phase_relative")
    )
    summaries["best_internal_baseline_test_spearman"] = best_baseline
    summaries["trustdive_score_delta"] = (
        summaries["trustdive_d"]["official_test_score"]["spearman"] - best_baseline
    )
    summaries["preparation"] = preparation
    summaries["baseline_preparation"] = baseline_preparation
    source_inputs, source_preparation = prepare_inputs(
        frame,
        "none",
        allow_test_metrics=True,
        split_column="source_role",
        backbone=backbone,
        reference_roles=("source_fit", "source_validation"),
    )
    source_seed = int(contract["random"]["model_seeds"][0])
    source_result = train_single_model(
        frame,
        source_inputs,
        "trustdive_d",
        source_seed,
        "none",
        fit_role="source_fit",
        validation_role="source_validation",
        role_column="source_role",
        disagreement_rescue=bool(selection["disagreement_rank_rescue"]),
        fit_roles=("source_fit", "source_validation"),
        fixed_epochs=fixed_epochs,
    )
    source_test = frame.source_role.to_numpy() == "source_test"
    fit_actions = set(frame.loc[frame.source_role == "source_fit", "action_type"])
    unseen_action = ~frame.action_type.isin(fit_actions)
    source_output = frame.loc[
        source_test,
        [
            "clip_uid",
            "source_role",
            "event_family",
            "action_type",
            "difficulty",
            "dive_score",
            "execution_quality",
        ],
    ].copy()
    source_indices = np.flatnonzero(source_test)
    source_output["predicted_quality"] = source_result["prediction"][source_indices]
    source_output["predicted_score"] = restore_total_score(
        source_output.predicted_quality, source_output.difficulty
    )
    source_output["sigma_judge"] = source_result["sigma_judge"][source_indices]
    source_output["reference_distance"] = source_inputs.reference_distance[source_indices]
    source_output["open_set"] = source_inputs.open_set[source_indices]
    source_output["unseen_action"] = unseen_action.to_numpy()[source_indices]
    source_output.to_parquet(
        V2_RESULTS_ROOT / "03_FINAL" / "source_isolated_predictions_v2.parquet", index=False
    )
    seen = ~source_output.unseen_action.to_numpy(dtype=bool)
    source_summary = {
        "seed": source_seed,
        "n": int(len(source_output)),
        "seen_action_n": int(seen.sum()),
        "unseen_action_n": int((~seen).sum()),
        "all_metrics": score_metrics(source_output.dive_score, source_output.predicted_score),
        "seen_action_metrics": score_metrics(
            source_output.dive_score.to_numpy()[seen], source_output.predicted_score.to_numpy()[seen]
        )
        if seen.any()
        else None,
        "claim_boundary": "Event-family-isolated robustness; not cross-dataset generalization.",
        "preparation": source_preparation,
    }
    summaries["source_isolated"] = source_summary
    write_json(V2_RESULTS_ROOT / "03_FINAL" / "source_isolated_summary_v2.json", source_summary)
    write_json(V2_RESULTS_ROOT / "03_FINAL" / "final_training_summary_v2.json", summaries)
    return summaries
