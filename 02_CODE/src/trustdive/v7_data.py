from __future__ import annotations

import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROJECT_ROOT, RESULTS_ROOT, RUNS_ROOT, Paths, load_contract
from .util import git_head, sha256_file, write_json
from .v6_data import V6_RESULTS_ROOT
from .v6_modeling import load_v6_assets


V7_CONTRACT_PATH = PROJECT_ROOT / "01_PROTOCOL" / "analysis_contract_v7_risk_task.yaml"
V7_RESULTS_ROOT = RESULTS_ROOT / "V7_RISK_TASK"
V7_RUN_ROOT = RUNS_ROOT / "v7_risk_task"
V7_CACHE_ROOT = PROJECT_ROOT / ".cache" / "v7_risk_task"


def v7_paths() -> Paths:
    return replace(Paths(), contract=V7_CONTRACT_PATH)


def load_v7_contract() -> dict:
    return load_contract(V7_CONTRACT_PATH)


def ensure_v7_dirs() -> None:
    for path in (
        V7_RESULTS_ROOT / "00_AUDIT",
        V7_RESULTS_ROOT / "01_RISK_TASK",
        V7_RESULTS_ROOT / "02_BASELINES",
        V7_RESULTS_ROOT / "03_SCORE",
        V7_RESULTS_ROOT / "04_PHASE_EVIDENCE",
        V7_RESULTS_ROOT / "05_RISK_REVIEW",
        V7_RESULTS_ROOT / "figures_v7" / "source_data",
        V7_RUN_ROOT / "checkpoints",
        V7_CACHE_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_v7_frame() -> pd.DataFrame:
    return load_v6_assets().frame.copy().reset_index(drop=True)


def _changed_since_anchor(anchor: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", anchor, "--"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def _is_protected_historical(path: str) -> bool:
    normalized = path.replace("\\", "/")

    def version_from_name(name: str) -> int | None:
        match = re.search(r"(?:^|[_-])v(\d+)(?:[_\-.]|$)", name.lower())
        return int(match.group(1)) if match else None

    if normalized.startswith("03_RESULTS/"):
        result_root = normalized.split("/", 2)[1]
        version = version_from_name(result_root)
        return version is None or version < 7
    if normalized.startswith("runs/"):
        version = version_from_name(Path(normalized).name)
        return version is None or version < 7
    if normalized.startswith("01_PROTOCOL/"):
        version = version_from_name(Path(normalized).name)
        return version is None or version < 7
    if normalized.startswith("02_CODE/src/trustdive/v"):
        stem = Path(normalized).name
        return any(stem.startswith(f"v{version}_") for version in range(1, 7))
    if normalized.startswith("README_V"):
        version = version_from_name(Path(normalized).name)
        return version is not None and version < 7
    return False


def audit_v7() -> dict:
    ensure_v7_dirs()
    contract = load_v7_contract()
    frame = load_v7_frame()
    anchor = str(contract["read_only_anchor"]["repository_commit"])
    head = git_head(PROJECT_ROOT) or ""
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", anchor, head],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    changed = _changed_since_anchor(anchor)
    protected_changes = [path for path in changed if _is_protected_historical(path)]
    latent_summary = V6_RESULTS_ROOT / "01_LATENTS" / "latent_extraction_v6.json"
    latent_ok = latent_summary.exists() and json.loads(latent_summary.read_text(encoding="utf-8")).get("status") == "PASS"
    counts = frame.analysis_role.value_counts().to_dict()
    invalid = int((~frame.judge_label_valid.astype(bool)).sum())
    checks = {
        "anchor_is_ancestor": ancestor,
        "protected_v1_v6_unchanged": not protected_changes,
        "v6_latent_cache_pass": latent_ok,
        "samples": len(frame) == int(contract["data"]["samples"]),
        "official_train": int((frame.official_split == "train").sum()) == int(contract["data"]["official_train"]),
        "official_test": int((frame.official_split == "test").sum()) == int(contract["data"]["official_test"]),
        "fit": int(counts.get("fit", 0)) == int(contract["data"]["fit"]),
        "validation": int(counts.get("validation", 0)) == int(contract["data"]["validation"]),
        "calibration": int(counts.get("calibration", 0)) == int(contract["data"]["calibration"]),
        "valid_seven_judge": int(frame.disagreement_primary_eligible.sum()) == int(contract["data"]["valid_seven_judge"]),
        "test_seven_judge": int(((frame.analysis_role == "official_test") & frame.disagreement_primary_eligible).sum()) == int(contract["data"]["official_test_seven_judge"]),
        "invalid_judge_arrays": invalid == int(contract["data"]["invalid_judge_arrays"]),
        "event_families": int(frame.event_family.nunique()) == int(contract["data"]["event_families"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_git_head": head,
        "anchor": anchor,
        "protected_changes": protected_changes,
        "v6_latent_summary_sha256": sha256_file(latent_summary) if latent_summary.exists() else None,
        "material_passport": "experiment-agent / audit / 2026-08-20 / VERIFIED_DATA / trustdive_risk_v7",
    }
    write_json(V7_RESULTS_ROOT / "00_AUDIT" / "audit_v7.json", result)
    return result


def require_v7_audit() -> dict:
    path = V7_RESULTS_ROOT / "00_AUDIT" / "audit_v7.json"
    if not path.exists():
        raise RuntimeError("Run audit --protocol v7 first")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("v7 audit did not pass")
    return result


def _proxy_features(global_latent: np.ndarray, teacher_quality: np.ndarray) -> np.ndarray:
    return np.concatenate((global_latent, teacher_quality[:, None]), axis=1).astype(np.float32)


def _fit_proxy(x: np.ndarray, residual: np.ndarray, indices: np.ndarray):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(x[indices], residual[indices])
    return model


def build_risk_task_v7() -> dict:
    """Freeze training-derived review-risk definitions before v7 model selection."""
    require_v7_audit()
    contract = load_v7_contract()
    assets = load_v6_assets()
    frame = assets.frame.copy().reset_index(drop=True)
    x = _proxy_features(assets.global_latent, assets.teacher_quality)
    target_quality = frame.execution_quality.to_numpy(dtype=float)
    residual = target_quality - assets.teacher_quality
    fit = np.flatnonzero(frame.analysis_role.to_numpy() == "fit")
    validation = np.flatnonzero(frame.analysis_role.to_numpy() == "validation")
    calibration = np.flatnonzero(frame.analysis_role.to_numpy() == "calibration")
    proxy = np.full(len(frame), np.nan, dtype=float)
    groups = frame.event_family.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=int(contract["risk_task"]["crossfit_folds"]))
    fold_id = np.full(len(frame), -1, dtype=int)
    for fold, (train_local, held_local) in enumerate(splitter.split(fit, groups=groups[fit])):
        train_index = fit[train_local]
        held_index = fit[held_local]
        model = _fit_proxy(x, residual, train_index)
        proxy[held_index] = assets.teacher_quality[held_index] + model.predict(x[held_index])
        fold_id[held_index] = fold
    fit_model = _fit_proxy(x, residual, fit)
    for indices in (validation, calibration):
        proxy[indices] = assets.teacher_quality[indices] + fit_model.predict(x[indices])
    proxy_score = 3.0 * frame.difficulty.to_numpy(dtype=float) * proxy
    proxy_error = np.abs(proxy_score - frame.dive_score.to_numpy(dtype=float))
    error_threshold = float(np.quantile(proxy_error[fit], float(contract["risk_task"]["error_quantile"])))
    judge_fit = (
        (frame.analysis_role.to_numpy() == "fit")
        & frame.disagreement_primary_eligible.to_numpy(dtype=bool)
    )
    judge_threshold = float(np.quantile(
        frame.loc[judge_fit, "judge_sample_sd"].to_numpy(dtype=float),
        float(contract["risk_task"]["judge_quantile"]),
    ))
    output = frame[[
        "clip_uid", "official_split", "analysis_role", "source_role", "event_family",
        "action_type", "difficulty", "dive_score", "execution_quality", "judge_count",
        "judge_label_valid", "judge_invalid_reason", "judge_sample_sd",
        "disagreement_primary_eligible",
    ]].copy()
    output["judge_risk_threshold"] = judge_threshold
    output["high_judge_risk"] = (
        output.disagreement_primary_eligible.astype(bool)
        & (output.judge_sample_sd >= judge_threshold)
    )
    output["proxy_fold"] = fold_id
    output["proxy_predicted_score"] = proxy_score
    output["proxy_absolute_error"] = proxy_error
    output["error_risk_threshold"] = error_threshold
    output["high_error_risk_proxy"] = np.where(
        np.isfinite(proxy_error), proxy_error >= error_threshold, False
    )
    output["error_proxy_available"] = np.isfinite(proxy_error)
    output.loc[output.analysis_role == "official_test", [
        "proxy_predicted_score", "proxy_absolute_error", "high_error_risk_proxy", "error_proxy_available"
    ]] = [np.nan, np.nan, False, False]
    path = V7_RESULTS_ROOT / "01_RISK_TASK" / "risk_task_manifest_v7.parquet"
    output.to_parquet(path, index=False)
    thresholds = {
        "threshold_state": "FROZEN_TRAINING_THRESHOLDS",
        "judge_sd_threshold_from_fit": judge_threshold,
        "error_threshold_from_fit_oof_score": error_threshold,
        "fit_oof_rows": int(np.isfinite(proxy[fit]).sum()),
        "validation_proxy_rows": int(np.isfinite(proxy[validation]).sum()),
        "calibration_proxy_rows": int(np.isfinite(proxy[calibration]).sum()),
        "test_labels_used_for_thresholds": False,
        "manifest_sha256": sha256_file(path),
    }
    threshold_path = V7_RESULTS_ROOT / "01_RISK_TASK" / "risk_thresholds_v7.json"
    write_json(threshold_path, thresholds)
    result = {
        "status": "PASS",
        "rows": int(len(output)),
        "valid_seven_judge": int(output.disagreement_primary_eligible.sum()),
        "fit_high_judge_rows": int(output.loc[output.analysis_role == "fit", "high_judge_risk"].sum()),
        "fit_high_error_rows": int(output.loc[output.analysis_role == "fit", "high_error_risk_proxy"].sum()),
        **thresholds,
        "threshold_sha256": sha256_file(threshold_path),
        "material_passport": "experiment-agent / etl / 2026-08-20 / VERIFIED_DATA / trustdive_risk_v7",
    }
    write_json(V7_RESULTS_ROOT / "01_RISK_TASK" / "risk_task_summary_v7.json", result)
    return result


def load_risk_manifest_v7() -> pd.DataFrame:
    path = V7_RESULTS_ROOT / "01_RISK_TASK" / "risk_task_manifest_v7.parquet"
    if not path.exists():
        raise RuntimeError("Run build-risk-task --protocol v7 first")
    return pd.read_parquet(path)
