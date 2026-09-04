from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

from .config import PROJECT_ROOT, RESULTS_ROOT, RUNS_ROOT, load_contract
from .metrics import aqa_score_metrics
from .util import git_head, set_seed, sha256_file, utc_now, write_json
from .v4_counterfactual import resample_phase_labels
from .v6_modeling import load_v6_assets


V10_CONTRACT_PATH = PROJECT_ROOT / "01_PROTOCOL" / "analysis_contract_v10_feature_baselines.yaml"
V10_RESULTS_ROOT = RESULTS_ROOT / "V10_FEATURE_BASELINES"
V10_RUN_ROOT = RUNS_ROOT / "v10_feature_baselines"
V10_LEDGER_PATH = V10_RUN_ROOT / "v10_gpu_budget_ledger.json"


def load_v10_contract() -> dict:
    return load_contract(V10_CONTRACT_PATH)


def ensure_v10_dirs() -> None:
    for path in (
        V10_RESULTS_ROOT / "00_AUDIT",
        V10_RESULTS_ROOT / "01_SMOKE_TEST",
        V10_RESULTS_ROOT / "02_PILOT",
        V10_RESULTS_ROOT / "03_FINAL",
        V10_RESULTS_ROOT / "04_ANALYSIS",
        V10_RESULTS_ROOT / "05_VERIFY",
        V10_RUN_ROOT / "checkpoints",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _read_ledger() -> dict:
    contract = load_v10_contract()
    if not V10_LEDGER_PATH.exists():
        return {"budget_hours": float(contract["compute"]["gpu_budget_hours"]), "entries": []}
    return json.loads(V10_LEDGER_PATH.read_text(encoding="utf-8"))


def _assert_budget(estimated_hours: float) -> None:
    ledger = _read_ledger()
    used = sum(float(item["elapsed_seconds"]) for item in ledger["entries"] if item.get("used_gpu")) / 3600.0
    if used + estimated_hours > float(ledger["budget_hours"]):
        raise RuntimeError(
            f"GPU budget would be exceeded: used={used:.3f} h, "
            f"estimated={estimated_hours:.3f} h, cap={ledger['budget_hours']:.3f} h"
        )


@contextmanager
def _budget_entry(command: str, estimated_hours: float):
    _assert_budget(estimated_hours)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    started_at = utc_now()
    ok = False
    try:
        yield
        ok = True
    finally:
        elapsed = time.perf_counter() - started
        used_gpu = bool(torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 0)
        peak = int(torch.cuda.max_memory_allocated()) if used_gpu else 0
        ledger = _read_ledger()
        ledger["entries"].append(
            {
                "command": command,
                "started_at": started_at,
                "finished_at": utc_now(),
                "elapsed_seconds": elapsed,
                "used_gpu": used_gpu,
                "peak_vram_bytes": peak,
                "success": ok,
            }
        )
        write_json(V10_LEDGER_PATH, ledger)


@dataclass
class V10Assets:
    frame: pd.DataFrame
    global_latent: np.ndarray
    sequences: np.ndarray
    phase_labels: np.ndarray
    phase_pooled: np.ndarray
    teacher_quality: np.ndarray
    true_quality: np.ndarray
    score_scale: np.ndarray


def phase_pool_sequences(sequences: np.ndarray, labels: np.ndarray) -> np.ndarray:
    sequences = np.asarray(sequences, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int8)
    if sequences.ndim != 3 or labels.shape != sequences.shape[:2]:
        raise ValueError("sequences must be [N,T,D] and labels must be [N,T]")
    pooled = np.empty((len(sequences), 3, sequences.shape[2]), dtype=np.float32)
    fallback = sequences.mean(axis=1)
    for phase in range(3):
        mask = labels == phase
        count = mask.sum(axis=1)
        numerator = (sequences * mask[..., None]).sum(axis=1)
        pooled[:, phase] = numerator / np.maximum(count[:, None], 1)
        pooled[count == 0, phase] = fallback[count == 0]
    return pooled


def _load_phase_labels(length: int, expected_rows: int) -> np.ndarray:
    path = (
        RESULTS_ROOT
        / "V2_DISAGREEMENT"
        / "01_FEATURES"
        / "phase_predictions_videomae_official_v2.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        original = payload["predictions"].astype(np.int8)
    if original.shape[0] != expected_rows:
        raise AssertionError("Frozen phase predictions do not cover all videos")
    return np.stack([resample_phase_labels(item, length) for item in original])


def load_v10_assets() -> V10Assets:
    base = load_v6_assets()
    frame = base.frame.reset_index(drop=True)
    sequence_path = (
        RESULTS_ROOT / "V5_APPLIED_CFPD" / "01_REFERENCES" / "teacher_sequences_v5.npz"
    )
    with np.load(sequence_path, allow_pickle=False) as payload:
        uid = payload["clip_uid"].astype(str)
        sequences = payload["sequence"].astype(np.float32)
    if not np.array_equal(uid, frame.clip_uid.to_numpy(dtype=str)):
        raise AssertionError("Frozen sequence order does not match the v6/v7 frame")
    labels = _load_phase_labels(sequences.shape[1], len(frame))
    true_quality = frame.execution_quality.to_numpy(dtype=np.float32)
    scale = (3.0 * frame.difficulty.to_numpy(dtype=np.float32)).astype(np.float32)
    return V10Assets(
        frame=frame,
        global_latent=base.global_latent.astype(np.float32),
        sequences=sequences,
        phase_labels=labels,
        phase_pooled=phase_pool_sequences(sequences, labels),
        teacher_quality=base.teacher_quality.astype(np.float32),
        true_quality=true_quality,
        score_scale=scale,
    )


def _reference_path(stage: str) -> Path:
    if stage not in {"development", "final"}:
        raise ValueError(f"Unknown reference stage: {stage}")
    return RESULTS_ROOT / "V7_RISK_TASK" / "02_BASELINES" / f"reference_map_{stage}_v7.npz"


def load_reference_map(stage: str) -> dict[str, np.ndarray]:
    with np.load(_reference_path(stage), allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _primary_references(mapping: dict[str, np.ndarray], count: int = 5) -> tuple[np.ndarray, np.ndarray]:
    refs = mapping["references"][:, :count].astype(np.int64)
    valid = refs >= 0
    return refs, valid


def _input_paths(contract: dict) -> dict[str, str]:
    merged = dict(contract["read_only_anchor"]["input_hashes"])
    merged.update(contract["read_only_anchor"]["manuscript_hashes"])
    return merged


def _phase_source_shape() -> list[int]:
    path = (
        RESULTS_ROOT
        / "V2_DISAGREEMENT"
        / "01_FEATURES"
        / "phase_predictions_videomae_official_v2.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        return list(payload["predictions"].shape)


def audit_v10() -> dict:
    ensure_v10_dirs()
    contract = load_v10_contract()
    assets = load_v10_assets()
    counts = assets.frame.analysis_role.value_counts().to_dict()
    hashes = {}
    hash_match = {}
    for relative, expected in _input_paths(contract).items():
        path = PROJECT_ROOT / relative
        hashes[relative] = sha256_file(path) if path.exists() else None
        hash_match[relative] = (
            hashes[relative] is not None
            and hashes[relative].lower() == str(expected).lower()
        )
    development = load_reference_map("development")
    final = load_reference_map("final")
    checks = {
        "all_frozen_hashes_match": all(hash_match.values()),
        "samples": len(assets.frame) == int(contract["data"]["samples"]),
        "fit": int(counts.get("fit", 0)) == int(contract["data"]["fit"]),
        "validation": int(counts.get("validation", 0)) == int(contract["data"]["validation"]),
        "calibration": int(counts.get("calibration", 0)) == int(contract["data"]["calibration"]),
        "official_test": int(counts.get("official_test", 0)) == int(contract["data"]["official_test"]),
        "global_latent_shape": list(assets.global_latent.shape) == list(contract["data"]["global_latent_shape"]),
        "sequence_shape": list(assets.sequences.shape) == list(contract["data"]["temporal_sequence_shape"]),
        "phase_source_shape": _phase_source_shape()
        == list(contract["data"]["predicted_phase_shape"]),
        "development_reference_rows": len(development["references"]) == len(assets.frame),
        "final_reference_rows": len(final["references"]) == len(assets.frame),
        "finite_global_latent": bool(np.isfinite(assets.global_latent).all()),
        "finite_sequences": bool(np.isfinite(assets.sequences).all()),
        "finite_phase_pool": bool(np.isfinite(assets.phase_pooled).all()),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "hash_match": hash_match,
        "observed_hashes": hashes,
        "current_git_head": git_head(PROJECT_ROOT),
        "repository_anchor": contract["read_only_anchor"]["repository_commit"],
        "manuscript_was_not_modified": all(
            hash_match[key] for key in contract["read_only_anchor"]["manuscript_hashes"]
        ),
        "material_passport": "experiment-agent / audit / 2026-09-04 / VERIFIED_DATA / feature_baselines_v10",
    }
    write_json(V10_RESULTS_ROOT / "00_AUDIT" / "audit_v10.json", result)
    return result


def require_v10_audit() -> dict:
    path = V10_RESULTS_ROOT / "00_AUDIT" / "audit_v10.json"
    if not path.exists():
        raise RuntimeError("Run trustdive.feature_baselines audit first")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("v10 audit did not pass")
    return result


class QueryDataset(Dataset):
    def __init__(self, indices: Iterable[int]):
        self.indices = np.asarray(list(indices), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> int:
        return int(self.indices[index])


def masked_reference_median(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("values and valid must have shape [batch,references]")
    counts = valid.sum(dim=1)
    ordered = torch.where(valid, values, torch.full_like(values, float("inf"))).sort(dim=1).values
    safe_counts = counts.clamp_min(1)
    lower = ((safe_counts - 1) // 2).unsqueeze(1)
    upper = (safe_counts // 2).unsqueeze(1)
    lower_value = ordered.gather(1, lower).squeeze(1)
    upper_value = ordered.gather(1, upper).squeeze(1)
    median = 0.5 * (lower_value + upper_value)
    return torch.where(counts > 0, median, torch.zeros_like(median))


class CoReStyleAdapter(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4 * 256 + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        reference: torch.Tensor,
        reference_error: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        q = query[:, None, :].expand_as(reference)
        pair = torch.cat((q, reference, q - reference, torch.abs(q - reference), reference_error[..., None]), dim=-1)
        per_reference = self.network(pair).squeeze(-1)
        return masked_reference_median(per_reference, valid)


class TSAStyleAdapter(nn.Module):
    def __init__(self, hidden: int, dropout: float, heads: int = 2):
        super().__init__()
        self.projection = nn.Linear(1024, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.phase_head = nn.Sequential(
            nn.Linear(4 * hidden + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        reference: torch.Tensor,
        reference_error: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, refs, phases, _ = reference.shape
        q = self.projection(query)[:, None, :, :].expand(batch, refs, phases, -1)
        r = self.projection(reference)
        q_flat = q.reshape(batch * refs, phases, -1)
        r_flat = r.reshape(batch * refs, phases, -1)
        attended, _ = self.attention(q_flat, r_flat, r_flat, need_weights=False)
        attended = attended.reshape(batch, refs, phases, -1)
        ref_error = reference_error[:, :, None, None].expand(batch, refs, phases, 1)
        phase_feature = torch.cat((q, attended, q - attended, torch.abs(q - attended), ref_error), dim=-1)
        phase_residual = self.phase_head(phase_feature).squeeze(-1)
        per_reference = phase_residual.sum(dim=-1)
        return masked_reference_median(per_reference, valid)


def _standardizer(values: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subset = values[indices]
    mean = subset.mean(axis=tuple(range(subset.ndim - 1)), keepdims=False).astype(np.float32)
    std = subset.std(axis=tuple(range(subset.ndim - 1)), keepdims=False).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def _normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def _make_model(name: str, hidden: int, dropout: float, heads: int) -> nn.Module:
    if name == "core_style":
        return CoReStyleAdapter(hidden, dropout)
    if name == "tsa_style":
        return TSAStyleAdapter(hidden, dropout, heads)
    raise ValueError(f"Unknown model: {name}")


def _prepare_inputs(
    assets: V10Assets,
    mapping: dict[str, np.ndarray],
    name: str,
    train_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    refs, valid = _primary_references(mapping)
    safe = np.maximum(refs, 0)
    reference_error = (assets.true_quality[safe] - assets.teacher_quality[safe]).astype(np.float32)
    reference_error[~valid] = 0.0
    if name == "core_style":
        mean, std = _standardizer(assets.global_latent, train_indices)
        query = _normalize(assets.global_latent, mean, std)
        reference = query[safe]
    elif name == "tsa_style":
        mean, std = _standardizer(assets.phase_pooled, train_indices)
        query = _normalize(assets.phase_pooled, mean, std)
        reference = query[safe]
    else:
        raise ValueError(name)
    reference[~valid] = 0.0
    return {
        "query": query,
        "reference": reference,
        "reference_error": reference_error,
        "valid": valid,
        "normalizer_mean": mean,
        "normalizer_std": std,
    }


def _tensor_batch(inputs: dict[str, np.ndarray], indices: torch.Tensor, device: torch.device):
    index = indices.detach().cpu().numpy().astype(int)
    return (
        torch.as_tensor(inputs["query"][index], dtype=torch.float32, device=device),
        torch.as_tensor(inputs["reference"][index], dtype=torch.float32, device=device),
        torch.as_tensor(inputs["reference_error"][index], dtype=torch.float32, device=device),
        torch.as_tensor(inputs["valid"][index], dtype=torch.bool, device=device),
    )


@torch.no_grad()
def _predict_residual(
    model: nn.Module,
    inputs: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    result = np.full(len(inputs["query"]), np.nan, dtype=np.float32)
    loader = DataLoader(QueryDataset(indices), batch_size=batch_size, shuffle=False)
    offset = 0
    for batch in loader:
        query, reference, ref_error, valid = _tensor_batch(inputs, batch, device)
        pred = model(query, reference, ref_error, valid).detach().cpu().numpy()
        selected = batch.numpy().astype(int)
        result[selected] = pred
        offset += len(selected)
    return result


def _score_metrics_for_indices(assets: V10Assets, quality: np.ndarray, indices: np.ndarray) -> dict:
    score = assets.score_scale[indices] * quality[indices]
    return aqa_score_metrics(assets.frame.dive_score.to_numpy(dtype=float)[indices], score)


def _train_one(
    name: str,
    hidden: int,
    learning_rate: float,
    seed: int,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray | None,
    mapping: dict[str, np.ndarray],
    fixed_epochs: int | None = None,
) -> tuple[dict, dict, np.ndarray]:
    contract = load_v10_contract()
    assets = load_v10_assets()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    inputs = _prepare_inputs(assets, mapping, name, train_indices)
    model = _make_model(
        name,
        hidden,
        float(contract["models"]["dropout"]),
        int(contract["models"]["tsa_attention_heads"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.HuberLoss()
    batch_size = int(contract["models"]["batch_size"])
    maximum_epochs = int(fixed_epochs or contract["models"]["maximum_epochs"])
    patience = int(contract["models"]["early_stopping_patience"])
    target_residual = (assets.true_quality - assets.teacher_quality).astype(np.float32)
    lower, upper = np.quantile(
        target_residual[train_indices], contract["models"]["residual_clip_quantiles"]
    ).astype(float)
    best_state = None
    best_epoch = 0
    best_validation_mae = float("inf")
    epochs_without_improvement = 0
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        loader = DataLoader(
            QueryDataset(train_indices),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        losses = []
        for batch in loader:
            query, reference, ref_error, valid = _tensor_batch(inputs, batch, device)
            target = torch.as_tensor(target_residual[batch.numpy().astype(int)], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(query, reference, ref_error, valid)
            loss = criterion(predicted, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if evaluation_indices is not None:
            residual = _predict_residual(model, inputs, evaluation_indices, device, batch_size)
            residual = np.clip(residual, lower, upper)
            quality = assets.teacher_quality.copy()
            quality[evaluation_indices] += residual[evaluation_indices]
            metrics = _score_metrics_for_indices(assets, quality, evaluation_indices)
            row.update({f"validation_{key}": value for key, value in metrics.items() if isinstance(value, float)})
            if float(metrics["mae"]) < best_validation_mae - 1e-8:
                best_validation_mae = float(metrics["mae"])
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if fixed_epochs is None and epochs_without_improvement >= patience:
                history.append(row)
                break
        history.append(row)
    if evaluation_indices is not None:
        if best_state is None:
            raise RuntimeError("No validation checkpoint was produced")
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum_epochs
    all_indices = np.arange(len(assets.frame), dtype=int)
    residual = _predict_residual(model, inputs, all_indices, device, batch_size)
    residual = np.clip(residual, lower, upper)
    quality = assets.teacher_quality + residual
    open_set = mapping["open_set"].astype(bool)
    quality[open_set] = assets.teacher_quality[open_set]
    artifact = {
        "model": name,
        "hidden": hidden,
        "learning_rate": learning_rate,
        "seed": seed,
        "epochs": best_epoch,
        "state_dict": model.state_dict(),
        "normalizer_mean": inputs["normalizer_mean"],
        "normalizer_std": inputs["normalizer_std"],
        "residual_clip": (lower, upper),
    }
    summary = {
        "model": name,
        "hidden": hidden,
        "learning_rate": learning_rate,
        "seed": seed,
        "best_epoch": best_epoch,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "residual_clip_lower": lower,
        "residual_clip_upper": upper,
        "history": history,
    }
    if evaluation_indices is not None:
        summary.update(_score_metrics_for_indices(assets, quality, evaluation_indices))
    return artifact, summary, quality


def smoke_test_v10() -> dict:
    require_v10_audit()
    ensure_v10_dirs()
    contract = load_v10_contract()
    assets = load_v10_assets()
    mapping = load_reference_map("development")
    fit = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "fit")
    validation = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "validation")
    rows = int(contract["compute"]["smoke_test_rows"])
    train = fit[:rows]
    evaluate = validation[: min(rows, len(validation))]
    reports = []
    started = time.perf_counter()
    with _budget_entry("smoke-test", float(contract["compute"]["smoke_estimated_hours"])):
        for name in contract["models"]["names"]:
            artifact, summary, quality = _train_one(
                name,
                hidden=64,
                learning_rate=0.001,
                seed=int(contract["models"]["pilot_seeds"][0]),
                train_indices=train,
                evaluation_indices=evaluate,
                mapping=mapping,
                fixed_epochs=2,
            )
            del artifact
            reports.append(
                {
                    "model": name,
                    "finite_predictions": bool(np.isfinite(quality[evaluate]).all()),
                    "parameter_count": summary["parameter_count"],
                }
            )
    elapsed = time.perf_counter() - started
    peak_mib = int(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0
    # The smoke run already performs full-dataset inference for both models.
    # Extrapolate only by trained model-epochs; multiplying by the dataset size
    # again would double-count the 3,000-query prediction pass.
    smoke_model_epochs = len(contract["models"]["names"]) * 2
    pilot_model_epochs = (
        len(contract["models"]["names"])
        * len(contract["models"]["hidden_dimensions"])
        * len(contract["models"]["learning_rates"])
        * len(contract["models"]["pilot_seeds"])
        * int(contract["models"]["maximum_epochs"])
    )
    final_model_epochs = (
        len(contract["models"]["names"])
        * len(contract["models"]["final_seeds"])
        * int(contract["models"]["maximum_epochs"])
    )
    full_estimate_hours = (
        elapsed / 3600.0
        * (pilot_model_epochs + final_model_epochs)
        / smoke_model_epochs
    )
    checks = {
        "finite_predictions": all(item["finite_predictions"] for item in reports),
        "peak_vram_within_limit": peak_mib <= int(contract["compute"]["maximum_vram_mib"]),
        "projected_gpu_hours_within_budget": full_estimate_hours <= float(contract["compute"]["gpu_budget_hours"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rows": rows,
        "elapsed_seconds": elapsed,
        "peak_vram_mib": peak_mib,
        "projected_full_gpu_hours": full_estimate_hours,
        "models": reports,
        "material_passport": "experiment-agent / smoke-test / 2026-09-04 / VERIFIED / feature_baselines_v10",
    }
    write_json(V10_RESULTS_ROOT / "01_SMOKE_TEST" / "smoke_test_v10.json", result)
    return result


def _require_smoke_pass() -> dict:
    path = V10_RESULTS_ROOT / "01_SMOKE_TEST" / "smoke_test_v10.json"
    if not path.exists():
        raise RuntimeError("Run smoke-test first")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("v10 smoke test did not pass")
    return result


def train_pilot_v10(name: str) -> dict:
    require_v10_audit()
    _require_smoke_pass()
    ensure_v10_dirs()
    contract = load_v10_contract()
    if name not in contract["models"]["names"]:
        raise ValueError(name)
    assets = load_v10_assets()
    mapping = load_reference_map("development")
    fit = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "fit")
    validation = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "validation")
    rows = []
    estimate = float(contract["compute"]["pilot_estimated_hours_per_model"])
    with _budget_entry(f"train --model {name} --stage pilot", estimate):
        for hidden in contract["models"]["hidden_dimensions"]:
            for learning_rate in contract["models"]["learning_rates"]:
                for seed in contract["models"]["pilot_seeds"]:
                    artifact, summary, _ = _train_one(
                        name,
                        int(hidden),
                        float(learning_rate),
                        int(seed),
                        fit,
                        validation,
                        mapping,
                    )
                    checkpoint = (
                        V10_RUN_ROOT
                        / "checkpoints"
                        / f"pilot_{name}_h{hidden}_lr{learning_rate:g}_s{seed}.pt"
                    )
                    torch.save(artifact, checkpoint)
                    history = pd.DataFrame(summary.pop("history"))
                    history.to_csv(checkpoint.with_suffix(".csv"), index=False)
                    rows.append(summary)
    trials = pd.DataFrame(rows)
    teacher_metrics = _score_metrics_for_indices(assets, assets.teacher_quality, validation)
    aggregate = (
        trials.groupby(["model", "hidden", "learning_rate"], as_index=False)
        .agg(
            spearman=("spearman", "mean"),
            spearman_sd=("spearman", "std"),
            mae=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse=("rmse", "mean"),
            relative_l2=("relative_l2", "mean"),
            median_best_epoch=("best_epoch", "median"),
            parameter_count=("parameter_count", "first"),
        )
    )
    aggregate["eligible"] = aggregate.spearman >= (
        float(teacher_metrics["spearman"])
        - float(contract["models"]["maximum_validation_spearman_drop"])
    )
    eligible = aggregate[aggregate.eligible].copy()
    if eligible.empty:
        status = "FAIL"
        selected = None
    else:
        best_mae = float(eligible.mae.min())
        near = eligible[
            eligible.mae <= best_mae + float(contract["models"]["simple_model_mae_tolerance"])
        ]
        selected_row = near.sort_values(
            ["parameter_count", "mae", "hidden", "learning_rate"], kind="stable"
        ).iloc[0]
        selected = selected_row.where(pd.notna(selected_row), None).to_dict()
        status = "PASS"
    trials_path = V10_RESULTS_ROOT / "02_PILOT" / f"pilot_trials_{name}_v10.csv"
    aggregate_path = V10_RESULTS_ROOT / "02_PILOT" / f"pilot_aggregate_{name}_v10.csv"
    trials.to_csv(trials_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    result = {
        "status": status,
        "model": name,
        "teacher_validation_metrics": teacher_metrics,
        "selection_rule": "Spearman drop <= 0.01, then validation MAE; within 0.02 points choose fewer parameters",
        "selected": selected,
        "official_test_used_for_selection": False,
        "pilot_trials_sha256": sha256_file(trials_path),
        "pilot_aggregate_sha256": sha256_file(aggregate_path),
        "material_passport": "experiment-agent / training / 2026-09-04 / VERIFIED_VALIDATION / feature_baselines_v10",
    }
    write_json(V10_RESULTS_ROOT / "02_PILOT" / f"selected_{name}_v10.json", result)
    return result


def _selected_models() -> dict:
    selected = {}
    for name in load_v10_contract()["models"]["names"]:
        path = V10_RESULTS_ROOT / "02_PILOT" / f"selected_{name}_v10.json"
        if not path.exists():
            raise RuntimeError(f"Pilot result missing for {name}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS":
            raise RuntimeError(f"Pilot gate failed for {name}")
        selected[name] = result["selected"]
    return selected


def freeze_contract_v10() -> dict:
    require_v10_audit()
    smoke = _require_smoke_pass()
    selected = _selected_models()
    contract = load_v10_contract()
    if contract["state"]["frozen"]:
        return contract["state"]
    selected_path = V10_RESULTS_ROOT / "02_PILOT" / "selected_models_v10.json"
    write_json(selected_path, selected)
    contract["state"].update(
        {
            "frozen": True,
            "frozen_at": utc_now(),
            "official_test_unlocked": True,
            "selected_models_sha256": sha256_file(selected_path),
            "smoke_test_sha256": sha256_file(
                V10_RESULTS_ROOT / "01_SMOKE_TEST" / "smoke_test_v10.json"
            ),
            "contract_sha256": None,
        }
    )
    canonical = yaml.safe_dump(contract, sort_keys=True, allow_unicode=True).encode("utf-8")
    contract["state"]["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    V10_CONTRACT_PATH.write_text(
        yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    result = dict(contract["state"])
    result["selected_models"] = selected
    result["smoke_status"] = smoke["status"]
    write_json(V10_RESULTS_ROOT / "02_PILOT" / "contract_freeze_v10.json", result)
    return result


def require_v10_frozen() -> dict:
    contract = load_v10_contract()
    if not contract["state"]["frozen"] or not contract["state"]["official_test_unlocked"]:
        raise RuntimeError("Freeze the v10 contract before final training and official-test evaluation")
    return contract


def train_final_v10() -> dict:
    require_v10_audit()
    contract = require_v10_frozen()
    selected = _selected_models()
    assets = load_v10_assets()
    mapping = load_reference_map("final")
    development = np.flatnonzero(assets.frame.analysis_role.isin(("fit", "validation")).to_numpy())
    test = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "official_test")
    output_rows = []
    checkpoints = []
    with _budget_entry("train --model all --stage final", float(contract["compute"]["final_estimated_hours"])):
        for name in contract["models"]["names"]:
            config = selected[name]
            epochs = max(1, int(round(float(config["median_best_epoch"]))))
            for seed in contract["models"]["final_seeds"]:
                artifact, summary, quality = _train_one(
                    name,
                    int(config["hidden"]),
                    float(config["learning_rate"]),
                    int(seed),
                    development,
                    None,
                    mapping,
                    fixed_epochs=epochs,
                )
                checkpoint = V10_RUN_ROOT / "checkpoints" / f"final_{name}_s{seed}.pt"
                torch.save(artifact, checkpoint)
                checkpoints.append({"path": str(checkpoint), "sha256": sha256_file(checkpoint)})
                predicted_score = assets.score_scale[test] * quality[test]
                for index, score in zip(test, predicted_score):
                    output_rows.append(
                        {
                            "clip_uid": assets.frame.loc[index, "clip_uid"],
                            "event_family": assets.frame.loc[index, "event_family"],
                            "action_type": assets.frame.loc[index, "action_type"],
                            "difficulty": float(assets.frame.loc[index, "difficulty"]),
                            "dive_score": float(assets.frame.loc[index, "dive_score"]),
                            "disagreement_primary_eligible": bool(
                                assets.frame.loc[index, "disagreement_primary_eligible"]
                            ),
                            "model": name,
                            "seed": int(seed),
                            "predicted_score": float(score),
                            "open_set": bool(mapping["open_set"][index]),
                            "epochs": epochs,
                        }
                    )
    output = pd.DataFrame(output_rows)
    path = V10_RESULTS_ROOT / "03_FINAL" / "predictions_v10.parquet"
    output.to_parquet(path, index=False)
    result = {
        "status": "PASS",
        "rows": int(len(output)),
        "test_rows_per_model_seed": int(len(test)),
        "models": list(contract["models"]["names"]),
        "seeds": list(contract["models"]["final_seeds"]),
        "official_test_used_for_model_selection": False,
        "predictions_sha256": sha256_file(path),
        "checkpoints": checkpoints,
        "material_passport": "experiment-agent / training / 2026-09-04 / VERIFIED_TEST / feature_baselines_v10",
    }
    write_json(V10_RESULTS_ROOT / "03_FINAL" / "final_training_v10.json", result)
    return result


def _metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    return float(aqa_score_metrics(y_true, y_pred)[metric])


def _cluster_bootstrap_differences(
    frame: pd.DataFrame,
    trustdive: np.ndarray,
    baseline: np.ndarray,
    metrics: list[str],
    iterations: int,
    seed: int,
) -> list[dict]:
    families = frame.event_family.astype(str).to_numpy()
    unique = np.unique(families)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in metrics}
    y = frame.dive_score.to_numpy(dtype=float)
    family_rows = {family: np.flatnonzero(families == family) for family in unique}
    for _ in range(iterations):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([family_rows[family] for family in drawn])
        for metric in metrics:
            samples[metric].append(
                _metric_value(y[indices], trustdive[indices], metric)
                - _metric_value(y[indices], baseline[indices], metric)
            )
    rows = []
    for metric, values in samples.items():
        values = np.asarray(values, dtype=float)
        observed = _metric_value(y, trustdive, metric) - _metric_value(y, baseline, metric)
        rows.append(
            {
                "metric": metric,
                "difference_definition": "TrustDive minus matched-feature baseline",
                "estimate": observed,
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "iterations": iterations,
            }
        )
    return rows


def analyze_v10() -> dict:
    contract = require_v10_frozen()
    prediction_path = V10_RESULTS_ROOT / "03_FINAL" / "predictions_v10.parquet"
    if not prediction_path.exists():
        raise RuntimeError("Run final training first")
    new = pd.read_parquet(prediction_path)
    v7 = pd.read_parquet(
        RESULTS_ROOT / "V7_RISK_TASK" / "03_SCORE" / "predictions_v7.parquet"
    )
    risk = pd.read_parquet(
        RESULTS_ROOT / "V7_RISK_TASK" / "01_RISK_TASK" / "risk_task_manifest_v7.parquet"
    )[["clip_uid", "high_judge_risk"]]
    test = v7[v7.analysis_role == "official_test"].copy().reset_index(drop=True).merge(
        risk, on="clip_uid", how="left", validate="one_to_one"
    )
    ensembles = (
        new.groupby(["clip_uid", "model"], as_index=False)
        .agg(predicted_score=("predicted_score", "mean"), seed_sd=("predicted_score", "std"))
    )
    comparison_rows = []
    prediction_columns = {
        "Deterministic RICA2": test.teacher_predicted_score.to_numpy(dtype=float),
        "TrustDive": test.trustdive_predicted_score.to_numpy(dtype=float),
    }
    for name in contract["models"]["names"]:
        values = ensembles[ensembles.model == name].set_index("clip_uid").loc[test.clip_uid]
        prediction_columns[name] = values.predicted_score.to_numpy(dtype=float)
    y = test.dive_score.to_numpy(dtype=float)
    high = test.high_judge_risk.fillna(False).to_numpy(dtype=bool)
    for name, prediction in prediction_columns.items():
        overall = aqa_score_metrics(y, prediction)
        high_metrics = aqa_score_metrics(y[high], prediction[high])
        comparison_rows.append(
            {
                "model": name,
                **{key: value for key, value in overall.items() if isinstance(value, float)},
                "high_disagreement_n": int(high.sum()),
                "high_disagreement_mae": float(high_metrics["mae"]),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = V10_RESULTS_ROOT / "04_ANALYSIS" / "comparison_v10.csv"
    comparison.to_csv(comparison_path, index=False)

    seed_rows = []
    for (name, seed), group in new.groupby(["model", "seed"]):
        ordered = group.set_index("clip_uid").loc[test.clip_uid]
        metrics = aqa_score_metrics(y, ordered.predicted_score.to_numpy(dtype=float))
        seed_rows.append(
            {"model": name, "seed": int(seed), **{k: v for k, v in metrics.items() if isinstance(v, float)}}
        )
    seed_metrics = pd.DataFrame(seed_rows)
    seed_path = V10_RESULTS_ROOT / "04_ANALYSIS" / "seed_metrics_v10.csv"
    seed_metrics.to_csv(seed_path, index=False)

    bootstrap_rows = []
    metrics = ["mae", "spearman", "rmse"]
    for index, name in enumerate(contract["models"]["names"]):
        rows = _cluster_bootstrap_differences(
            test,
            prediction_columns["TrustDive"],
            prediction_columns[name],
            metrics,
            int(contract["statistics"]["cluster_bootstrap_iterations"]),
            int(contract["statistics"]["bootstrap_seed"]) + index,
        )
        for row in rows:
            row["baseline"] = name
            row["subset"] = "overall"
            bootstrap_rows.append(row)
        high_rows = _cluster_bootstrap_differences(
            test.loc[high].reset_index(drop=True),
            prediction_columns["TrustDive"][high],
            prediction_columns[name][high],
            ["mae"],
            int(contract["statistics"]["cluster_bootstrap_iterations"]),
            int(contract["statistics"]["bootstrap_seed"]) + 100 + index,
        )
        for row in high_rows:
            row["baseline"] = name
            row["subset"] = "high_disagreement"
            bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap_path = V10_RESULTS_ROOT / "04_ANALYSIS" / "paired_bootstrap_v10.csv"
    bootstrap.to_csv(bootstrap_path, index=False)

    lookup = comparison.set_index("model")
    trust = lookup.loc["TrustDive"]
    advantage = {}
    for name in contract["models"]["names"]:
        ci = bootstrap[
            (bootstrap.baseline == name)
            & (bootstrap.metric == "mae")
            & (bootstrap.subset == "overall")
        ].iloc[0]
        high_ci = bootstrap[
            (bootstrap.baseline == name)
            & (bootstrap.metric == "mae")
            & (bootstrap.subset == "high_disagreement")
        ].iloc[0]
        advantage[name] = {
            "trustdive_lower_mae_point_estimate": bool(trust.mae < lookup.loc[name, "mae"]),
            "trustdive_mae_advantage_ci_below_zero": bool(float(ci.ci_high) < 0.0),
            "trustdive_higher_spearman": bool(trust.spearman > lookup.loc[name, "spearman"]),
            "trustdive_lower_high_disagreement_mae": bool(
                trust.high_disagreement_mae < lookup.loc[name, "high_disagreement_mae"]
            ),
            "trustdive_high_disagreement_mae_advantage_ci_below_zero": bool(
                float(high_ci.ci_high) < 0.0
            ),
        }
    if all(value["trustdive_mae_advantage_ci_below_zero"] for value in advantage.values()):
        decision = "CLEAR_SCORING_ADVANTAGE"
    elif any(
        value["trustdive_mae_advantage_ci_below_zero"]
        or value["trustdive_high_disagreement_mae_advantage_ci_below_zero"]
        for value in advantage.values()
    ):
        decision = "PARTIAL_ADVANTAGE"
    elif any(
        value["trustdive_lower_mae_point_estimate"]
        and not value["trustdive_higher_spearman"]
        for value in advantage.values()
    ):
        decision = "METRIC_TRADEOFF"
    elif any(value["trustdive_lower_mae_point_estimate"] for value in advantage.values()):
        decision = "NON_SIGNIFICANT_POINT_ADVANTAGE"
    else:
        decision = "NO_SCORING_ADVANTAGE_REPORT_BEFORE_MANUSCRIPT_EDIT"
    result = {
        "status": "PASS",
        "decision": decision,
        "advantage_by_baseline": advantage,
        "test_rows": int(len(test)),
        "high_disagreement_rows": int(high.sum()),
        "comparison_sha256": sha256_file(comparison_path),
        "seed_metrics_sha256": sha256_file(seed_path),
        "bootstrap_sha256": sha256_file(bootstrap_path),
        "manuscript_modified": False,
        "material_passport": "experiment-agent / analysis / 2026-09-04 / ANALYZED / feature_baselines_v10",
    }
    write_json(V10_RESULTS_ROOT / "04_ANALYSIS" / "analysis_summary_v10.json", result)
    return result


def _legal_references(assets: V10Assets, mapping: dict[str, np.ndarray], roles: set[str]) -> bool:
    refs, valid = _primary_references(mapping)
    actions = assets.frame.action_type.astype(str).to_numpy()
    families = assets.frame.event_family.astype(str).to_numpy()
    analysis_roles = assets.frame.analysis_role.astype(str).to_numpy()
    for query in range(len(refs)):
        for ref in refs[query, valid[query]]:
            if (
                analysis_roles[ref] not in roles
                or actions[ref] != actions[query]
                or families[ref] == families[query]
                or ref == query
            ):
                return False
    return True


def _checkpoint_inference_is_repeatable(
    assets: V10Assets,
    mapping: dict[str, np.ndarray],
    contract: dict,
) -> bool:
    development = np.flatnonzero(
        assets.frame.analysis_role.isin(("fit", "validation")).to_numpy()
    )
    test = np.flatnonzero(assets.frame.analysis_role.to_numpy() == "official_test")[:100]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    selected = _selected_models()
    for name in contract["models"]["names"]:
        seed = int(contract["models"]["final_seeds"][0])
        checkpoint = V10_RUN_ROOT / "checkpoints" / f"final_{name}_s{seed}.pt"
        if not checkpoint.exists():
            return False
        artifact = torch.load(checkpoint, map_location=device, weights_only=False)
        config = selected[name]
        model = _make_model(
            name,
            int(config["hidden"]),
            float(contract["models"]["dropout"]),
            int(contract["models"]["tsa_attention_heads"]),
        ).to(device)
        model.load_state_dict(artifact["state_dict"])
        inputs = _prepare_inputs(assets, mapping, name, development)
        first = _predict_residual(model, inputs, test, device, int(contract["models"]["batch_size"]))
        second = _predict_residual(model, inputs, test, device, int(contract["models"]["batch_size"]))
        if not np.array_equal(first[test], second[test]):
            return False
    return True


def _write_run_manifest(contract: dict) -> Path:
    output_paths = [
        V10_RESULTS_ROOT / "00_AUDIT" / "audit_v10.json",
        V10_RESULTS_ROOT / "01_SMOKE_TEST" / "smoke_test_v10.json",
        V10_RESULTS_ROOT / "02_PILOT" / "selected_models_v10.json",
        V10_RESULTS_ROOT / "02_PILOT" / "contract_freeze_v10.json",
        V10_RESULTS_ROOT / "03_FINAL" / "predictions_v10.parquet",
        V10_RESULTS_ROOT / "03_FINAL" / "final_training_v10.json",
        V10_RESULTS_ROOT / "04_ANALYSIS" / "comparison_v10.csv",
        V10_RESULTS_ROOT / "04_ANALYSIS" / "paired_bootstrap_v10.csv",
        V10_RESULTS_ROOT / "04_ANALYSIS" / "seed_metrics_v10.csv",
        V10_RESULTS_ROOT / "04_ANALYSIS" / "analysis_summary_v10.json",
        V10_RESULTS_ROOT / "RESULT_REPORT_V10.md",
    ]
    ledger = _read_ledger()
    manifest = {
        "protocol": "feature_baselines_v10",
        "material_passport": "experiment-agent / manifest / 2026-09-04 / VERIFIED / feature_baselines_v10",
        "git_commit": git_head(PROJECT_ROOT),
        "contract_path": str(V10_CONTRACT_PATH),
        "contract_state_hash": contract["state"]["contract_sha256"],
        "frozen_input_hashes": {
            key: value.lower() for key, value in _input_paths(contract).items()
        },
        "selected_models": _selected_models(),
        "seeds": contract["models"]["final_seeds"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "gpu_elapsed_seconds": sum(
            float(item["elapsed_seconds"])
            for item in ledger["entries"]
            if item.get("used_gpu")
        ),
        "gpu_budget_hours": ledger["budget_hours"],
        "commands": ledger["entries"],
        "output_hashes": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in output_paths
            if path.exists()
        },
    }
    path = V10_RESULTS_ROOT / "run_manifest_v10.json"
    write_json(path, manifest)
    return path


def verify_v10() -> dict:
    contract = require_v10_frozen()
    audit = audit_v10()
    assets = load_v10_assets()
    development = load_reference_map("development")
    final = load_reference_map("final")
    predictions_path = V10_RESULTS_ROOT / "03_FINAL" / "predictions_v10.parquet"
    analysis_path = V10_RESULTS_ROOT / "04_ANALYSIS" / "analysis_summary_v10.json"
    predictions = pd.read_parquet(predictions_path) if predictions_path.exists() else pd.DataFrame()
    expected_rows = (
        int(contract["data"]["official_test"])
        * len(contract["models"]["names"])
        * len(contract["models"]["final_seeds"])
    )
    test_uids = set(
        assets.frame.loc[assets.frame.analysis_role == "official_test", "clip_uid"].astype(str)
    )
    open_indices = np.flatnonzero(final["open_set"].astype(bool))
    open_test_uids = set(assets.frame.loc[open_indices, "clip_uid"].astype(str)) & test_uids
    open_ok = True
    if not predictions.empty and open_test_uids:
        teacher = assets.frame.assign(
            teacher_score=assets.score_scale * assets.teacher_quality
        ).set_index("clip_uid").teacher_score
        selected = predictions[predictions.clip_uid.isin(open_test_uids)]
        open_ok = bool(
            np.allclose(
                selected.predicted_score.to_numpy(dtype=float),
                teacher.loc[selected.clip_uid].to_numpy(dtype=float),
                atol=1e-6,
            )
        )
    checks = {
        "audit_pass": audit.get("status") == "PASS",
        "contract_frozen": bool(contract["state"]["frozen"]),
        "development_references_legal": _legal_references(assets, development, {"fit"}),
        "final_references_legal": _legal_references(assets, final, {"fit", "validation"}),
        "prediction_rows": len(predictions) == expected_rows,
        "prediction_test_uid_set": set(predictions.clip_uid.astype(str)) == test_uids if not predictions.empty else False,
        "finite_predictions": bool(np.isfinite(predictions.predicted_score).all()) if not predictions.empty else False,
        "checkpoint_inference_repeatable": _checkpoint_inference_is_repeatable(
            assets, final, contract
        ),
        "open_set_fallback": open_ok,
        "analysis_exists": analysis_path.exists(),
        "manuscript_hashes_unchanged": bool(audit.get("manuscript_was_not_modified")),
        "gpu_budget_not_exceeded": sum(
            float(item["elapsed_seconds"])
            for item in _read_ledger()["entries"]
            if item.get("used_gpu")
        )
        / 3600.0
        <= float(contract["compute"]["gpu_budget_hours"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "prediction_sha256": sha256_file(predictions_path) if predictions_path.exists() else None,
        "analysis_sha256": sha256_file(analysis_path) if analysis_path.exists() else None,
        "gpu_ledger": _read_ledger(),
        "material_passport": "experiment-agent / verification / 2026-09-04 / VERIFIED / feature_baselines_v10",
    }
    write_json(V10_RESULTS_ROOT / "05_VERIFY" / "verification_v10.json", result)
    manifest_path = _write_run_manifest(contract)
    result["run_manifest_sha256"] = sha256_file(manifest_path)
    write_json(V10_RESULTS_ROOT / "05_VERIFY" / "verification_v10.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="TrustDive v10 matched-feature baseline benchmark")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    commands.add_parser("smoke-test")
    train = commands.add_parser("train")
    train.add_argument("--model", choices=("core_style", "tsa_style", "all"), required=True)
    train.add_argument("--stage", choices=("pilot", "final"), required=True)
    commands.add_parser("freeze-contract")
    commands.add_parser("analyze")
    commands.add_parser("verify")
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "audit":
        result = audit_v10()
    elif args.command == "smoke-test":
        result = smoke_test_v10()
    elif args.command == "train" and args.stage == "pilot":
        if args.model == "all":
            result = {name: train_pilot_v10(name) for name in load_v10_contract()["models"]["names"]}
        else:
            result = train_pilot_v10(args.model)
    elif args.command == "freeze-contract":
        result = freeze_contract_v10()
    elif args.command == "train" and args.stage == "final":
        if args.model != "all":
            raise ValueError("Final training must use --model all")
        result = train_final_v10()
    elif args.command == "analyze":
        result = analyze_v10()
    elif args.command == "verify":
        result = verify_v10()
    else:
        raise ValueError(vars(args))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()


