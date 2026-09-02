from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import aqa_score_metrics
from .util import sha256_file, write_json
from .v6_data import (
    V6_RESULTS_ROOT,
    V6_RUN_ROOT,
    load_v6_contract,
    load_v6_frame,
    require_v6_audit,
    require_v6_frozen,
)


@dataclass
class V6Assets:
    frame: pd.DataFrame
    global_latent: np.ndarray
    semantic_latent: np.ndarray
    actions: np.ndarray
    teacher_quality: np.ndarray


def load_v6_assets() -> V6Assets:
    summary = V6_RESULTS_ROOT / "01_LATENTS" / "latent_extraction_v6.json"
    if not summary.exists() or json.loads(summary.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Run extract-latents --protocol v6 first")
    frame = load_v6_frame().reset_index(drop=True)
    with np.load(V6_RESULTS_ROOT / "01_LATENTS" / "teacher_latents_v6.npz", allow_pickle=False) as payload:
        uid = payload["clip_uid"].astype(str)
        ordered = frame.set_index("clip_uid").loc[uid].reset_index()
        global_latent = payload["global_latent"].astype(np.float32)
        semantic_latent = payload["semantic_latent"].astype(np.float32)
        actions = payload["action_presence"].astype(np.float32)
    teacher = pd.read_parquet(
        V6_RESULTS_ROOT / "01_LATENTS" / "teacher_predictions_v6.parquet"
    ).set_index("clip_uid").loc[uid]
    return V6Assets(
        ordered,
        global_latent,
        semantic_latent,
        actions,
        teacher.teacher_predicted_quality.to_numpy(dtype=np.float32),
    )


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def build_reference_map_v6(final: bool = False) -> dict:
    require_v6_audit()
    if final:
        require_v6_frozen()
    contract = load_v6_contract()
    assets = load_v6_assets()
    frame = assets.frame
    roles = contract["adapter"]["final_reference_roles" if final else "development_reference_roles"]
    pool = np.flatnonzero(frame.analysis_role.isin(roles).to_numpy())
    actions = frame.action_type.astype(str).to_numpy()
    families = frame.event_family.astype(str).to_numpy()
    normalized = _normalized_rows(assets.global_latent)
    primary = int(contract["data"]["reference_count"])
    alternate = int(contract["data"]["alternate_reference_count"])
    slots = primary + alternate
    refs = np.full((len(frame), slots), -1, dtype=np.int32)
    distances = np.full((len(frame), slots), np.nan, dtype=np.float32)
    valid_count = np.zeros(len(frame), dtype=np.int16)
    q_true = frame.execution_quality.to_numpy(dtype=np.float32)
    ref_dispersion = np.zeros(len(frame), dtype=np.float32)
    for index in range(len(frame)):
        candidates = pool[
            (actions[pool] == actions[index])
            & (families[pool] != families[index])
            & (pool != index)
        ]
        if len(candidates):
            distance = 1.0 - normalized[candidates] @ normalized[index]
            order = np.argsort(distance, kind="stable")[:slots]
            selected = candidates[order]
            selected_distance = distance[order]
            refs[index, : len(selected)] = selected
            distances[index, : len(selected)] = selected_distance
            valid_count[index] = min(len(selected), primary)
            ref_dispersion[index] = float(np.std(q_true[selected[:primary]], ddof=0))
    open_set = valid_count < int(contract["data"]["minimum_valid_references"])
    # Only primary slots contribute to the score. A training reference's own
    # teacher error is a legal reliability signal; no query target is used.
    primary_refs = refs[:, :primary]
    primary_distance = distances[:, :primary]
    weights = np.zeros_like(primary_distance, dtype=np.float32)
    available_distances = primary_distance[np.isfinite(primary_distance)]
    temperature = max(float(np.median(available_distances)), 1e-3)
    for index in range(len(frame)):
        valid = primary_refs[index] >= 0
        if not np.any(valid):
            continue
        ref_index = primary_refs[index, valid]
        ref_error = np.abs(assets.teacher_quality[ref_index] - q_true[ref_index])
        reliability = 1.0 / (1.0 + ref_error + ref_dispersion[index])
        raw = np.exp(-primary_distance[index, valid] / temperature) * reliability
        weights[index, valid] = raw / max(float(raw.sum()), 1e-8)
    suffix = "final" if final else "development"
    path = V6_RESULTS_ROOT / "02_ADAPTER" / f"reference_map_{suffix}_v6.npz"
    np.savez_compressed(
        path,
        references=refs,
        distances=distances,
        weights=weights,
        valid_reference_count=valid_count,
        open_set=open_set,
        pool_indices=pool,
    )
    legal = True
    for index in range(len(frame)):
        for ref in refs[index, : valid_count[index]]:
            legal &= bool(
                ref in pool
                and actions[ref] == actions[index]
                and families[ref] != families[index]
            )
    result = {
        "status": "PASS" if legal else "FAIL",
        "stage": suffix,
        "pool_roles": list(roles),
        "pool_rows": int(len(pool)),
        "rows": int(len(frame)),
        "all_valid_references_legal": bool(legal),
        "open_set_count": int(open_set.sum()),
        "open_set_fraction": float(open_set.mean()),
        "distance_temperature": temperature,
        "reference_map_sha256": sha256_file(path),
    }
    write_json(V6_RESULTS_ROOT / "02_ADAPTER" / f"reference_summary_{suffix}_v6.json", result)
    return result


def load_reference_map_v6(final: bool = False) -> dict[str, np.ndarray]:
    suffix = "final" if final else "development"
    path = V6_RESULTS_ROOT / "02_ADAPTER" / f"reference_map_{suffix}_v6.npz"
    if not path.exists():
        build_reference_map_v6(final=final)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def make_reference_features(
    assets: V6Assets, reference_map: dict[str, np.ndarray]
) -> np.ndarray:
    refs = reference_map["references"][:, :5].astype(int)
    safe_refs = np.maximum(refs, 0)
    gq = np.repeat(assets.global_latent[:, None, :], refs.shape[1], axis=1)
    gr = assets.global_latent[safe_refs]
    qtq = np.repeat(assets.teacher_quality[:, None, None], refs.shape[1], axis=1)
    qtr = assets.teacher_quality[safe_refs, None]
    qr = assets.frame.execution_quality.to_numpy(dtype=np.float32)[safe_refs, None]
    distance = np.nan_to_num(reference_map["distances"][:, :5], nan=2.0)[..., None]
    dispersion = np.std(qr[..., 0], axis=1, keepdims=True)
    dispersion = np.repeat(dispersion[:, None, :], refs.shape[1], axis=1)
    features = np.concatenate(
        (gq, gr, gq - gr, np.abs(gq - gr), qtq, qtr, qr, distance, dispersion),
        axis=2,
    ).astype(np.float32)
    features[refs < 0] = 0.0
    return features


class ReferenceMLP:
    """Small PyTorch adapter kept behind a NumPy-facing interface."""

    def __init__(self, hidden: int, residual_limit: float):
        import torch

        self.hidden = int(hidden)
        self.residual_limit = float(residual_limit)
        self.module = torch.nn.Sequential(
            torch.nn.Linear(1029, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, 1),
        )

    def to(self, device):
        self.module.to(device)
        return self

    def predict_torch(self, x, weights, teacher):
        import torch

        delta = self.residual_limit * torch.tanh(self.module(x).squeeze(-1))
        return teacher + torch.sum(weights * delta, dim=1), delta


def _within_action_rank_loss(pred, target, action_codes):
    import torch

    pairs = []
    for action in torch.unique(action_codes):
        idx = torch.nonzero(action_codes == action, as_tuple=False).flatten()
        if len(idx) >= 2:
            pairs.extend(zip(idx[:-1].tolist(), idx[1:].tolist()))
    if not pairs:
        return pred.new_tensor(0.0)
    left = torch.tensor([a for a, _ in pairs], device=pred.device)
    right = torch.tensor([b for _, b in pairs], device=pred.device)
    sign = torch.sign(target[left] - target[right])
    valid = sign != 0
    if not torch.any(valid):
        return pred.new_tensor(0.0)
    return torch.relu(0.05 - sign[valid] * (pred[left][valid] - pred[right][valid])).mean()


def _fit_mlp(
    features: np.ndarray,
    weights: np.ndarray,
    teacher: np.ndarray,
    targets: np.ndarray,
    actions: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: dict,
    seed: int,
    save_path: Path,
    fixed_epochs: int | None = None,
) -> tuple[ReferenceMLP, dict]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReferenceMLP(config["hidden"], config["residual_limit"]).to(device)
    optimizer = torch.optim.AdamW(
        model.module.parameters(), lr=config["learning_rate"], weight_decay=1e-4
    )
    batch_size = int(load_v6_contract()["adapter"]["batch_size"])
    maximum_epochs = int(fixed_epochs or load_v6_contract()["adapter"]["maximum_epochs"])
    patience = int(load_v6_contract()["adapter"]["patience"])
    rng = np.random.default_rng(seed)
    best = {"mae": math.inf, "epoch": 0, "state": None}
    stale = 0
    for epoch in range(maximum_epochs):
        model.module.train()
        order = rng.permutation(train_indices)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            x = torch.from_numpy(features[index]).to(device)
            w = torch.from_numpy(weights[index]).to(device)
            t = torch.from_numpy(teacher[index]).to(device)
            y = torch.from_numpy(targets[index]).to(device)
            a = torch.from_numpy(actions[index]).to(device)
            pred, delta = model.predict_torch(x, w, t)
            loss = F.smooth_l1_loss(pred, y, beta=0.5)
            loss = loss + 0.05 * _within_action_rank_loss(pred, y, a)
            loss = loss + float(load_v6_contract()["adapter"]["residual_l2_weight"]) * (delta**2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if fixed_epochs is not None:
            best = {
                "mae": float("nan"),
                "epoch": epoch + 1,
                "state": {key: value.detach().cpu().clone() for key, value in model.module.state_dict().items()},
            }
            continue
        validation_pred, _ = predict_mlp(model, features[validation_indices], weights[validation_indices], teacher[validation_indices])
        mae = float(np.mean(np.abs(validation_pred - targets[validation_indices])))
        if mae < best["mae"] - 1e-5:
            best = {
                "mae": mae,
                "epoch": epoch + 1,
                "state": {key: value.detach().cpu().clone() for key, value in model.module.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.module.load_state_dict(best["state"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.module.state_dict(),
            "hidden": model.hidden,
            "residual_limit": model.residual_limit,
            "config": config,
            "best_epoch": best["epoch"],
        },
        save_path,
    )
    return model, {"best_epoch": int(best["epoch"]), "validation_quality_mae": float(best["mae"])}


def predict_mlp(model: ReferenceMLP, features, weights, teacher) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = next(model.module.parameters()).device
    model.module.eval()
    outputs = []
    deltas = []
    with torch.inference_mode():
        for start in range(0, len(features), 512):
            pred, delta = model.predict_torch(
                torch.from_numpy(np.asarray(features[start : start + 512], dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(weights[start : start + 512], dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(teacher[start : start + 512], dtype=np.float32)).to(device),
            )
            outputs.append(pred.cpu().numpy())
            deltas.append(delta.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(deltas)


def _metrics_for_indices(assets: V6Assets, prediction_quality: np.ndarray, indices: np.ndarray) -> dict:
    score = 3.0 * assets.frame.difficulty.to_numpy(dtype=float)[indices] * prediction_quality[indices]
    true = assets.frame.dive_score.to_numpy(dtype=float)[indices]
    return aqa_score_metrics(true, score)


def _candidate_row(name: str, model_type: str, metrics: dict, teacher_metrics: dict, residual: np.ndarray, bound: float | None = None, **extra) -> dict:
    mae_improvement = (float(teacher_metrics["mae"]) - float(metrics["mae"])) / float(teacher_metrics["mae"])
    spearman_drop = float(teacher_metrics["spearman"]) - float(metrics["spearman"])
    bound_ok = True if bound is None else float(np.quantile(np.abs(residual), 0.95)) <= bound + 1e-6
    contract = load_v6_contract()
    eligible = (
        spearman_drop <= float(contract["adapter"]["validation_maximum_spearman_drop"])
        and mae_improvement >= float(contract["adapter"]["validation_minimum_mae_improvement"])
        and bound_ok
    )
    return {
        "name": name,
        "model_type": model_type,
        **{key: value for key, value in metrics.items() if not isinstance(value, str)},
        "teacher_spearman": float(teacher_metrics["spearman"]),
        "teacher_mae": float(teacher_metrics["mae"]),
        "spearman_drop": spearman_drop,
        "mae_improvement_fraction": mae_improvement,
        "residual_abs_p95": float(np.quantile(np.abs(residual), 0.95)),
        "residual_bound": bound,
        "residual_bound_ok": bool(bound_ok),
        "eligible": bool(eligible),
        **extra,
    }


def optimize_adapter_v6() -> dict:
    require_v6_audit()
    build_reference_map_v6(final=False)
    contract = load_v6_contract()
    assets = load_v6_assets()
    refs = load_reference_map_v6(final=False)
    features = make_reference_features(assets, refs)
    weights = refs["weights"].astype(np.float32)
    target = assets.frame.execution_quality.to_numpy(dtype=np.float32)
    fit = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "fit")
    validation = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "validation")
    teacher_metrics = _metrics_for_indices(assets, assets.teacher_quality, validation)
    trials: list[dict] = []
    artifacts: dict[str, str] = {}
    checkpoint_dir = V6_RUN_ROOT / "checkpoints"

    # Global linear calibration.
    linear = LinearRegression().fit(assets.teacher_quality[fit, None], target[fit])
    pred = assets.teacher_quality.copy()
    pred[validation] = linear.predict(assets.teacher_quality[validation, None])
    path = checkpoint_dir / "global_linear_v6.joblib"
    joblib.dump(linear, path)
    artifacts["global_linear"] = str(path)
    trials.append(_candidate_row(
        "global_linear", "linear", _metrics_for_indices(assets, pred, validation),
        teacher_metrics, pred[validation] - assets.teacher_quality[validation],
    ))

    # Same-action reference residual is a parameter-free baseline.
    ref_index = refs["references"][:, :5].astype(int)
    safe = np.maximum(ref_index, 0)
    ref_residual = target[safe] - assets.teacher_quality[safe]
    same_residual = np.sum(weights * ref_residual, axis=1)
    same_pred = assets.teacher_quality + same_residual
    trials.append(_candidate_row(
        "same_action_residual", "reference_residual",
        _metrics_for_indices(assets, same_pred, validation), teacher_metrics,
        same_residual[validation],
    ))

    # Latent Ridge residual candidates.
    ridge_x = np.concatenate((assets.global_latent, assets.teacher_quality[:, None]), axis=1)
    for alpha in contract["adapter"]["ridge_alphas"]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(ridge_x[fit], target[fit] - assets.teacher_quality[fit])
        residual = model.predict(ridge_x)
        prediction = assets.teacher_quality + residual
        name = f"latent_ridge_alpha_{float(alpha):g}"
        path = checkpoint_dir / f"{name}_v6.joblib"
        joblib.dump(model, path)
        artifacts[name] = str(path)
        trials.append(_candidate_row(
            name, "ridge", _metrics_for_indices(assets, prediction, validation),
            teacher_metrics, residual[validation], alpha=float(alpha),
        ))

    action_codes = pd.factorize(assets.frame.action_type.astype(str), sort=True)[0].astype(np.int64)
    for hidden in contract["adapter"]["mlp_hidden_dimensions"]:
        for learning_rate in contract["adapter"]["mlp_learning_rates"]:
            for limit in contract["adapter"]["residual_limits"]:
                name = f"reference_mlp_h{hidden}_lr{learning_rate:g}_d{limit:g}".replace(".", "p")
                path = checkpoint_dir / f"{name}_v6.pt"
                model, fit_info = _fit_mlp(
                    features, weights, assets.teacher_quality, target, action_codes,
                    fit, validation,
                    {"hidden": int(hidden), "learning_rate": float(learning_rate), "residual_limit": float(limit)},
                    int(contract["statistics"]["model_seeds"][0]), path,
                )
                prediction, delta = predict_mlp(model, features, weights, assets.teacher_quality)
                artifacts[name] = str(path)
                trials.append(_candidate_row(
                    name, "mlp", _metrics_for_indices(assets, prediction, validation),
                    teacher_metrics, prediction[validation] - assets.teacher_quality[validation],
                    bound=float(limit), hidden=int(hidden), learning_rate=float(learning_rate),
                    best_epoch=fit_info["best_epoch"], delta_abs_p95=float(np.quantile(np.abs(delta[validation]), 0.95)),
                ))

    table = pd.DataFrame(trials).sort_values(["spearman", "mae"], ascending=[False, True])
    trials_path = V6_RESULTS_ROOT / "02_ADAPTER" / "adapter_trials_v6.csv"
    table.to_csv(trials_path, index=False)
    eligible = table[table.eligible].copy()
    if eligible.empty:
        selected = None
        status = "STOP"
    else:
        best_rho = float(eligible.spearman.max())
        near = eligible[eligible.spearman >= best_rho - float(contract["adapter"]["simple_model_tie_tolerance"])].copy()
        complexity = {"linear": 0, "reference_residual": 1, "ridge": 2, "mlp": 3}
        near["complexity"] = near.model_type.map(complexity)
        selected_raw = near.sort_values(["mae", "complexity", "name"]).iloc[0].to_dict()
        selected = {
            key: (
                None
                if pd.isna(value)
                else value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in selected_raw.items()
        }
        status = "PASS"
    result = {
        "status": status,
        "teacher_validation_metrics": teacher_metrics,
        "candidate_count": int(len(table)),
        "eligible_count": int(len(eligible)),
        "selected": selected,
        "selected_artifact": None if selected is None else artifacts.get(str(selected["name"])),
        "validation_labels_only": True,
        "official_test_labels_accessed": False,
        "trials_sha256": sha256_file(trials_path),
    }
    path = V6_RESULTS_ROOT / "02_ADAPTER" / "selected_adapter_v6.json"
    write_json(path, result)
    if selected is not None:
        result["selected_adapter_sha256"] = sha256_file(Path(result["selected_artifact"]))
        write_json(path, result)
    return result


def load_selected_adapter_v6() -> tuple[dict, object]:
    path = V6_RESULTS_ROOT / "02_ADAPTER" / "selected_adapter_v6.json"
    if not path.exists():
        raise RuntimeError("Run optimize-adapter --protocol v6 first")
    selected = json.loads(path.read_text(encoding="utf-8"))
    if selected.get("status") != "PASS":
        raise RuntimeError("No v6 adapter passed the validation gate")
    artifact = Path(selected["selected_artifact"])
    model_type = selected["selected"]["model_type"]
    if model_type == "mlp":
        import torch

        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        model = ReferenceMLP(payload["hidden"], payload["residual_limit"])
        model.module.load_state_dict(payload["state_dict"])
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return selected, model
    return selected, joblib.load(artifact)


def predict_selected_v6(
    assets: V6Assets,
    reference_map: dict[str, np.ndarray],
    selected: dict,
    model: object,
    global_latent: np.ndarray | None = None,
    teacher_quality: np.ndarray | None = None,
    reference_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    global_latent = assets.global_latent if global_latent is None else global_latent
    teacher_quality = assets.teacher_quality if teacher_quality is None else teacher_quality
    model_type = selected["selected"]["model_type"]
    if model_type == "linear":
        prediction = model.predict(teacher_quality[:, None])
    elif model_type == "ridge":
        x = np.concatenate((global_latent, teacher_quality[:, None]), axis=1)
        prediction = teacher_quality + model.predict(x)
    elif model_type == "reference_residual":
        refs = reference_map["references"][:, :5].astype(int)
        safe = np.maximum(refs, 0)
        residual = assets.frame.execution_quality.to_numpy(dtype=float)[safe] - assets.teacher_quality[safe]
        prediction = teacher_quality + np.sum(reference_map["weights"] * residual, axis=1)
    elif model_type == "mlp":
        features = reference_features if reference_features is not None else make_reference_features(assets, reference_map)
        prediction, _ = predict_mlp(model, features, reference_map["weights"].astype(np.float32), teacher_quality.astype(np.float32))
    else:
        raise ValueError(f"Unsupported adapter type: {model_type}")
    open_set = reference_map["open_set"].astype(bool)
    prediction = np.asarray(prediction, dtype=np.float32)
    prediction[open_set] = teacher_quality[open_set]
    return prediction, prediction - teacher_quality


def _fit_selected_on_indices(
    selected: dict, assets: V6Assets, refs: dict[str, np.ndarray], train: np.ndarray,
    validation: np.ndarray, seed: int, artifact: Path,
) -> object:
    model_type = selected["selected"]["model_type"]
    target = assets.frame.execution_quality.to_numpy(dtype=np.float32)
    if model_type == "linear":
        model = LinearRegression().fit(assets.teacher_quality[train, None], target[train])
        joblib.dump(model, artifact)
        return model
    if model_type == "ridge":
        alpha = float(selected["selected"]["alpha"])
        x = np.concatenate((assets.global_latent, assets.teacher_quality[:, None]), axis=1)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[train], target[train] - assets.teacher_quality[train])
        joblib.dump(model, artifact)
        return model
    if model_type == "reference_residual":
        joblib.dump({"type": "reference_residual"}, artifact)
        return {"type": "reference_residual"}
    features = make_reference_features(assets, refs)
    actions = pd.factorize(assets.frame.action_type.astype(str), sort=True)[0].astype(np.int64)
    config = {
        "hidden": int(selected["selected"]["hidden"]),
        "learning_rate": float(selected["selected"]["learning_rate"]),
        "residual_limit": float(selected["selected"]["residual_bound"]),
    }
    model, _ = _fit_mlp(
        features, refs["weights"].astype(np.float32), assets.teacher_quality, target,
        actions, train, validation, config, seed, artifact,
        fixed_epochs=int(selected["selected"]["best_epoch"]),
    )
    return model


def evaluate_final_v6() -> dict:
    require_v6_frozen()
    selected_record, _ = load_selected_adapter_v6()
    assets = load_v6_assets()
    build_reference_map_v6(final=True)
    refs = load_reference_map_v6(final=True)
    train = np.flatnonzero(assets.frame.analysis_role.isin(("fit", "validation")).to_numpy())
    calibration = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "calibration")
    test = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "official_test")
    predictions = []
    residuals = []
    seeds = load_v6_contract()["statistics"]["model_seeds"]
    for seed in seeds:
        artifact_suffix = ".pt" if selected_record["selected"]["model_type"] == "mlp" else ".joblib"
        artifact = V6_RUN_ROOT / "checkpoints" / f"final_adapter_seed_{seed}_v6{artifact_suffix}"
        model = _fit_selected_on_indices(
            selected_record, assets, refs, train, calibration, int(seed), artifact
        )
        pred, residual = predict_selected_v6(assets, refs, selected_record, model)
        predictions.append(pred)
        residuals.append(residual)
    prediction = np.mean(predictions, axis=0)
    residual = np.mean(residuals, axis=0)
    prediction_sd = np.std(predictions, axis=0, ddof=0)
    score = 3.0 * assets.frame.difficulty.to_numpy(dtype=float) * prediction
    teacher_score = 3.0 * assets.frame.difficulty.to_numpy(dtype=float) * assets.teacher_quality
    open_set = refs["open_set"].astype(bool)
    target_quality = assets.frame.execution_quality.to_numpy(dtype=np.float32)
    # Matched official-test ablations are trained only after the contract is
    # frozen. They share the same train/reference pools as the final adapter.
    linear_ablation = LinearRegression().fit(
        assets.teacher_quality[train, None], target_quality[train]
    )
    linear_quality = linear_ablation.predict(assets.teacher_quality[:, None]).astype(np.float32)
    linear_quality[open_set] = assets.teacher_quality[open_set]
    ref_index = refs["references"][:, :5].astype(int)
    safe_ref = np.maximum(ref_index, 0)
    same_residual = np.sum(
        refs["weights"] * (target_quality[safe_ref] - assets.teacher_quality[safe_ref]), axis=1
    )
    same_quality = assets.teacher_quality + same_residual
    same_quality[open_set] = assets.teacher_quality[open_set]
    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
        np.concatenate((assets.global_latent[train], assets.teacher_quality[train, None]), axis=1),
        target_quality[train] - assets.teacher_quality[train],
    )
    ridge_quality = assets.teacher_quality + ridge_model.predict(
        np.concatenate((assets.global_latent, assets.teacher_quality[:, None]), axis=1)
    )
    ridge_quality[open_set] = assets.teacher_quality[open_set]

    trial_table = pd.read_csv(V6_RESULTS_ROOT / "02_ADAPTER" / "adapter_trials_v6.csv")
    mlp_rows = trial_table[trial_table.model_type == "mlp"].sort_values(["spearman", "mae"], ascending=[False, True])
    unweighted_quality = np.full(len(assets.frame), np.nan, dtype=np.float32)
    if len(mlp_rows):
        best_mlp = mlp_rows.iloc[0]
        feature = make_reference_features(assets, refs)
        action_codes = pd.factorize(assets.frame.action_type.astype(str), sort=True)[0].astype(np.int64)
        artifact = V6_RUN_ROOT / "checkpoints" / "final_unweighted_reference_mlp_v6.pt"
        mlp_model, _ = _fit_mlp(
            feature, refs["weights"].astype(np.float32), assets.teacher_quality,
            target_quality, action_codes, train, calibration,
            {"hidden": int(best_mlp.hidden), "learning_rate": float(best_mlp.learning_rate), "residual_limit": float(best_mlp.residual_bound)},
            int(seeds[0]), artifact, fixed_epochs=int(best_mlp.best_epoch),
        )
        valid = ref_index >= 0
        uniform = valid.astype(np.float32) / np.maximum(valid.sum(axis=1, keepdims=True), 1)
        unweighted_quality, _ = predict_mlp(
            mlp_model, feature, uniform, assets.teacher_quality.astype(np.float32)
        )
        unweighted_quality[open_set] = assets.teacher_quality[open_set]
    output = assets.frame[
        ["clip_uid", "official_split", "analysis_role", "source_role", "event_family", "action_type", "difficulty", "dive_score", "execution_quality", "judge_sample_sd", "disagreement_primary_eligible"]
    ].copy()
    output["teacher_predicted_quality"] = assets.teacher_quality
    output["teacher_predicted_score"] = teacher_score
    output["global_linear_predicted_score"] = 3.0 * output.difficulty * linear_quality
    output["same_action_residual_predicted_score"] = 3.0 * output.difficulty * same_quality
    output["latent_ridge_predicted_score"] = 3.0 * output.difficulty * ridge_quality
    output["unweighted_reference_mlp_predicted_score"] = 3.0 * output.difficulty * unweighted_quality
    output["adapter_predicted_quality"] = prediction
    output["adapter_predicted_score"] = score
    output["adapter_residual"] = residual
    output["ensemble_sd"] = prediction_sd
    output["reference_distance"] = np.nanmean(refs["distances"][:, :5], axis=1)
    output["reference_dispersion"] = np.nanstd(
        assets.frame.execution_quality.to_numpy(dtype=float)[np.maximum(refs["references"][:, :5], 0)], axis=1
    )
    output["valid_reference_count"] = refs["valid_reference_count"]
    output["open_set"] = open_set
    calibration_error = np.abs(
        output.loc[calibration, "adapter_predicted_score"].to_numpy()
        - output.loc[calibration, "dive_score"].to_numpy()
    )
    conformal = float(np.quantile(calibration_error, 0.90, method="higher"))
    local_scale = 1.0 + output.reference_dispersion.to_numpy(dtype=float)
    output["prediction_interval_width"] = 2.0 * conformal * local_scale / max(float(np.median(local_scale[calibration])), 1e-8)
    path = V6_RESULTS_ROOT / "05_FINAL" / "adapter_predictions_v6.parquet"
    output.to_parquet(path, index=False)

    teacher_metrics = aqa_score_metrics(output.loc[test, "dive_score"], output.loc[test, "teacher_predicted_score"])
    adapter_metrics = aqa_score_metrics(output.loc[test, "dive_score"], output.loc[test, "adapter_predicted_score"])
    mae_improvement = (float(teacher_metrics["mae"]) - float(adapter_metrics["mae"])) / float(teacher_metrics["mae"])
    spearman_drop = float(teacher_metrics["spearman"]) - float(adapter_metrics["spearman"])
    contract = load_v6_contract()
    score_gate = (
        spearman_drop <= float(contract["publication_gates"]["maximum_test_spearman_drop"])
        and mae_improvement >= float(contract["publication_gates"]["minimum_test_mae_improvement"])
    )
    comparison_rows = []
    for model_name, column in (
        ("RICA2 deterministic", "teacher_predicted_score"),
        ("global linear calibration", "global_linear_predicted_score"),
        ("same-action reference residual", "same_action_residual_predicted_score"),
        ("256-d latent Ridge", "latent_ridge_predicted_score"),
        ("unweighted reference MLP", "unweighted_reference_mlp_predicted_score"),
        ("TrustDive-ECR", "adapter_predicted_score"),
    ):
        metrics = aqa_score_metrics(output.loc[test, "dive_score"], output.loc[test, column])
        comparison_rows.append({"model": model_name, **{k: v for k, v in metrics.items() if not isinstance(v, str)}})
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = V6_RESULTS_ROOT / "05_FINAL" / "ablation_summary_v6.csv"
    comparison.to_csv(comparison_path, index=False)

    # Event-family five-fold cross-fit labels for the downstream error-risk head.
    development = np.flatnonzero(assets.frame.analysis_role.isin(("fit", "validation")).to_numpy())
    crossfit = np.full(len(assets.frame), np.nan, dtype=np.float32)
    splitter = GroupKFold(n_splits=int(contract["risk"]["crossfit_folds"]))
    groups = assets.frame.event_family.astype(str).to_numpy()[development]
    for fold, (tr, va) in enumerate(splitter.split(development, groups=groups)):
        tr_index = development[tr]
        va_index = development[va]
        artifact_suffix = ".pt" if selected_record["selected"]["model_type"] == "mlp" else ".joblib"
        artifact = V6_RUN_ROOT / "checkpoints" / f"crossfit_{fold}_v6{artifact_suffix}"
        model = _fit_selected_on_indices(
            selected_record, assets, refs, tr_index, va_index,
            int(seeds[fold % len(seeds)]), artifact,
        )
        pred, _ = predict_selected_v6(assets, refs, selected_record, model)
        crossfit[va_index] = pred[va_index]
    crossfit_frame = output[["clip_uid", "analysis_role", "event_family", "difficulty", "dive_score"]].copy()
    crossfit_frame["crossfit_predicted_quality"] = crossfit
    crossfit_frame["crossfit_predicted_score"] = 3.0 * crossfit_frame.difficulty * crossfit
    crossfit_path = V6_RESULTS_ROOT / "05_FINAL" / "crossfit_predictions_v6.parquet"
    crossfit_frame.to_parquet(crossfit_path, index=False)
    source_fit = np.flatnonzero(assets.frame.source_role.to_numpy() == "source_fit")
    source_test = np.flatnonzero(assets.frame.source_role.to_numpy() == "source_test")
    source_artifact_suffix = ".pt" if selected_record["selected"]["model_type"] == "mlp" else ".joblib"
    source_model = _fit_selected_on_indices(
        selected_record, assets, refs, source_fit, source_test,
        int(seeds[0]), V6_RUN_ROOT / "checkpoints" / f"source_heldout_adapter_v6{source_artifact_suffix}",
    )
    source_prediction, _ = predict_selected_v6(assets, refs, selected_record, source_model)
    source_score = 3.0 * assets.frame.difficulty.to_numpy(dtype=float) * source_prediction
    source_metrics = aqa_score_metrics(
        assets.frame.dive_score.to_numpy(dtype=float)[source_test], source_score[source_test]
    )
    result = {
        "status": "PASS" if score_gate else "FAIL",
        "selected_adapter": selected_record["selected"],
        "teacher_test_metrics": teacher_metrics,
        "adapter_test_metrics": adapter_metrics,
        "test_spearman_drop": spearman_drop,
        "test_mae_improvement_fraction": mae_improvement,
        "score_gate": bool(score_gate),
        "ensemble_seeds": list(seeds),
        "conformal_score_half_width": conformal,
        "prediction_sha256": sha256_file(path),
        "comparison_sha256": sha256_file(comparison_path),
        "crossfit_sha256": sha256_file(crossfit_path),
        "adapter_source_family_heldout_metrics": source_metrics,
        "source_family_sensitivity_scope": "adapter-only; the frozen teacher retains its official training protocol",
    }
    write_json(V6_RESULTS_ROOT / "05_FINAL" / "final_score_summary_v6.json", result)
    return result
