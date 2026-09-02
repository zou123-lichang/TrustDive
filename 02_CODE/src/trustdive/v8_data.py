from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, SplineTransformer, StandardScaler

from .config import PROJECT_ROOT, RESULTS_ROOT, RUNS_ROOT, Paths, load_contract
from .util import git_head, sha256_file, write_json
from .v6_modeling import load_v6_assets
from .v7_data import V7_RESULTS_ROOT
from .v7_modeling import (
    _fit_ridge_artifact,
    _reference_map_for_pool,
    make_ridge_features_v7,
    predict_ridge_artifact,
)


V8_CONTRACT_PATH = PROJECT_ROOT / "01_PROTOCOL" / "analysis_contract_v8_phase_conflict.yaml"
V8_RESULTS_ROOT = RESULTS_ROOT / "V8_PHASE_CONFLICT"
V8_RUN_ROOT = RUNS_ROOT / "v8_phase_conflict"
V8_CACHE_ROOT = PROJECT_ROOT / ".cache" / "v8_phase_conflict"


def v8_paths() -> Paths:
    return replace(Paths(), contract=V8_CONTRACT_PATH)


def load_v8_contract() -> dict:
    return load_contract(V8_CONTRACT_PATH)


def ensure_v8_dirs() -> None:
    for path in (
        V8_RESULTS_ROOT / "00_AUDIT",
        V8_RESULTS_ROOT / "01_CONDITIONAL_RISK",
        V8_RESULTS_ROOT / "02_PHASE_TOKENS",
        V8_RESULTS_ROOT / "03_PILOT",
        V8_RESULTS_ROOT / "04_FINAL",
        V8_RESULTS_ROOT / "05_DUAL_EVIDENCE",
        V8_RESULTS_ROOT / "06_REVIEW",
        V8_RESULTS_ROOT / "figures_v8" / "source_data",
        V8_RUN_ROOT / "checkpoints",
        V8_CACHE_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


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
    if path.startswith("03_RESULTS/") and not path.startswith("03_RESULTS/V8_PHASE_CONFLICT/"):
        return True
    if path.startswith("runs/") and "v8" not in Path(path).name.lower():
        return True
    if path.startswith("01_PROTOCOL/") and "v8" not in Path(path).name.lower():
        return True
    if path.startswith("02_CODE/src/trustdive/v"):
        stem = Path(path).name
        return any(stem.startswith(f"v{version}_") for version in range(1, 8))
    if path.startswith("README_V") and not path.startswith("README_V8"):
        return True
    return False


def audit_v8() -> dict:
    ensure_v8_dirs()
    contract = load_v8_contract()
    assets = load_v6_assets()
    frame = assets.frame
    anchor = str(contract["read_only_anchor"]["repository_commit"])
    head = git_head(PROJECT_ROOT) or ""
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", anchor, head],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    changed = _changed_since_anchor(anchor)
    protected = [path for path in changed if _is_protected_historical(path)]
    required = [
        V7_RESULTS_ROOT / "02_BASELINES" / "selected_model_v7.json",
        V7_RESULTS_ROOT / "02_BASELINES" / "reference_map_development_v7.npz",
        V7_RESULTS_ROOT / "02_BASELINES" / "reference_map_final_v7.npz",
        V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet",
        V7_RESULTS_ROOT / "03_SCORE" / "crossfit_predictions_v7.parquet",
        V7_RESULTS_ROOT / "04_PHASE_EVIDENCE" / "phase_evidence_v7.parquet",
        V7_RESULTS_ROOT / "05_RISK_REVIEW" / "review_priority_v7.parquet",
        PROJECT_ROOT / "03_RESULTS" / "V5_APPLIED_CFPD" / "01_REFERENCES" / "teacher_sequences_v5.npz",
        PROJECT_ROOT / "03_RESULTS" / "V6_EXACT_REVIEW" / "01_LATENTS" / "teacher_latents_v6.npz",
        PROJECT_ROOT / "runs" / "v7_risk_task" / "checkpoints" / "final_plain_ridge_v7.joblib",
    ]
    counts = frame.analysis_role.value_counts().to_dict()
    checks = {
        "anchor_is_ancestor": ancestor,
        "protected_v1_v7_unchanged": not protected,
        "required_caches_exist": all(path.exists() for path in required),
        "samples": len(frame) == int(contract["data"]["samples"]),
        "official_train": int((frame.official_split == "train").sum()) == int(contract["data"]["official_train"]),
        "official_test": int((frame.official_split == "test").sum()) == int(contract["data"]["official_test"]),
        "fit": int(counts.get("fit", 0)) == int(contract["data"]["fit"]),
        "validation": int(counts.get("validation", 0)) == int(contract["data"]["validation"]),
        "calibration": int(counts.get("calibration", 0)) == int(contract["data"]["calibration"]),
        "valid_seven_judge": int(frame.disagreement_primary_eligible.sum()) == int(contract["data"]["valid_seven_judge"]),
        "test_seven_judge": int(((frame.analysis_role == "official_test") & frame.disagreement_primary_eligible).sum()) == int(contract["data"]["official_test_seven_judge"]),
        "invalid_judge_arrays": int((~frame.judge_label_valid.astype(bool)).sum()) == int(contract["data"]["invalid_judge_arrays"]),
        "event_families": int(frame.event_family.nunique()) == int(contract["data"]["event_families"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_git_head": head,
        "anchor": anchor,
        "protected_changes": protected,
        "required_cache_hashes": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in required if path.exists()
        },
        "material_passport": "experiment-agent / audit / 2026-08-21 / VERIFIED_DATA / trustdive_phase_conflict_v8",
    }
    write_json(V8_RESULTS_ROOT / "00_AUDIT" / "audit_v8.json", result)
    return result


def require_v8_audit() -> dict:
    path = V8_RESULTS_ROOT / "00_AUDIT" / "audit_v8.json"
    if not path.exists():
        raise RuntimeError("Run audit --protocol v8 first")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("v8 audit did not pass")
    return result


def _conditional_model(contract: dict):
    numeric_score = make_pipeline(
        SplineTransformer(
            n_knots=int(contract["conditional_disagreement"]["spline_knots"]),
            degree=int(contract["conditional_disagreement"]["spline_degree"]),
            include_bias=False,
        ),
        StandardScaler(),
    )
    preprocess = ColumnTransformer(
        [
            ("score", numeric_score, ["predicted_quality"]),
            ("action", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["action_type"]),
            ("difficulty", RobustScaler(), ["difficulty"]),
        ],
        sparse_threshold=0.0,
    )
    return make_pipeline(preprocess, Ridge(alpha=float(contract["conditional_disagreement"]["ridge_alpha"])))


def _plain_crossfit_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assets = load_v6_assets()
    frame = assets.frame
    selection = json.loads(
        (V7_RESULTS_ROOT / "02_BASELINES" / "selected_model_v7.json").read_text(encoding="utf-8")
    )
    alpha = float(selection["plain_ridge"]["alpha"])
    development = np.flatnonzero(frame.analysis_role.isin(("fit", "validation")).to_numpy())
    groups = frame.event_family.astype(str).to_numpy()
    quality = np.full(len(frame), np.nan, dtype=float)
    distance = np.full(len(frame), np.nan, dtype=float)
    dispersion = np.full(len(frame), np.nan, dtype=float)
    fold_id = np.full(len(frame), -1, dtype=np.int16)
    residual = frame.execution_quality.to_numpy(dtype=float) - assets.teacher_quality
    splitter = GroupKFold(n_splits=int(load_v8_contract()["conditional_disagreement"]["crossfit_folds"]))
    for fold, (train_local, held_local) in enumerate(splitter.split(development, groups=groups[development])):
        train_index = development[train_local]
        held_index = development[held_local]
        mapping = _reference_map_for_pool(assets, train_index)
        features = make_ridge_features_v7(assets, mapping)
        artifact = _fit_ridge_artifact(features, residual, train_index, alpha, None)
        prediction = predict_ridge_artifact(artifact, features, assets.teacher_quality)
        prediction[mapping["open_set"].astype(bool)] = assets.teacher_quality[mapping["open_set"].astype(bool)]
        quality[held_index] = prediction[held_index]
        refs = mapping["references"][:, :5].astype(int)
        valid = refs >= 0
        weights = np.where(valid, mapping["weights"][:, :5], 0.0).astype(float)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        distance[held_index] = np.sum(
            weights[held_index] * np.nan_to_num(mapping["distances"][held_index, :5], nan=2.0), axis=1
        )
        safe = np.maximum(refs, 0)
        ref_q = frame.execution_quality.to_numpy(dtype=float)[safe]
        mean_q = np.sum(weights * ref_q, axis=1, keepdims=True)
        all_dispersion = np.sqrt(np.sum(weights * (ref_q - mean_q) ** 2, axis=1))
        dispersion[held_index] = all_dispersion[held_index]
        fold_id[held_index] = fold
    return quality, distance, dispersion, fold_id


def build_conditional_disagreement_v8() -> dict:
    require_v8_audit()
    ensure_v8_dirs()
    contract = load_v8_contract()
    assets = load_v6_assets()
    frame = assets.frame.copy().reset_index(drop=True)
    v7_prediction = pd.read_parquet(V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet").set_index("clip_uid")
    v7_prediction = v7_prediction.loc[frame.clip_uid].reset_index()
    oof_quality, oof_distance, oof_dispersion, fold_id = _plain_crossfit_predictions()
    scale = 3.0 * frame.difficulty.to_numpy(dtype=float)
    final_plain_score = v7_prediction.plain_ridge_predicted_score.to_numpy(dtype=float)
    deployment_quality = final_plain_score / scale
    development = frame.analysis_role.isin(("fit", "validation")).to_numpy()
    model_quality = deployment_quality.copy()
    model_quality[development] = oof_quality[development]
    if not np.isfinite(model_quality[development]).all():
        raise AssertionError("Plain-Ridge cross-fit predictions are incomplete")

    features = pd.DataFrame({
        "predicted_quality": model_quality,
        "action_type": frame.action_type.astype(str),
        "difficulty": frame.difficulty.to_numpy(dtype=float),
    })
    eligible = frame.disagreement_primary_eligible.to_numpy(dtype=bool)
    fit_judge = np.flatnonzero((frame.analysis_role.to_numpy() == "fit") & eligible)
    target = np.log(frame.judge_sample_sd.to_numpy(dtype=float) + float(contract["conditional_disagreement"]["epsilon"]))
    expected = np.full(len(frame), np.nan, dtype=float)
    judge_fold = np.full(len(frame), -1, dtype=np.int16)
    groups = frame.event_family.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=int(contract["conditional_disagreement"]["crossfit_folds"]))
    for fold, (train_local, held_local) in enumerate(splitter.split(fit_judge, groups=groups[fit_judge])):
        train_index = fit_judge[train_local]
        held_index = fit_judge[held_local]
        model = _conditional_model(contract)
        model.fit(features.iloc[train_index], target[train_index])
        expected[held_index] = model.predict(features.iloc[held_index])
        judge_fold[held_index] = fold
    final_model = _conditional_model(contract)
    final_model.fit(features.iloc[fit_judge], target[fit_judge])
    remaining = np.flatnonzero(~((frame.analysis_role.to_numpy() == "fit") & eligible))
    expected[remaining] = final_model.predict(features.iloc[remaining])
    model_path = V8_RUN_ROOT / "checkpoints" / "conditional_disagreement_v8.joblib"
    joblib.dump(final_model, model_path)

    excess = target - expected
    threshold = float(np.quantile(excess[fit_judge], float(contract["conditional_disagreement"]["high_quantile"])))
    error = np.abs(scale * model_quality - frame.dive_score.to_numpy(dtype=float))
    fit = frame.analysis_role.to_numpy() == "fit"
    error_threshold = float(np.quantile(error[fit], float(contract["error_risk"]["high_quantile"])))

    output = frame[[
        "clip_uid", "official_split", "analysis_role", "source_role", "event_family",
        "action_type", "difficulty", "dive_score", "execution_quality", "judge_count",
        "judge_scores_json", "judge_label_valid", "judge_sample_sd", "disagreement_primary_eligible",
    ]].copy()
    output["model_predicted_quality"] = model_quality
    output["model_predicted_score"] = scale * model_quality
    output["score_crossfit_fold"] = fold_id
    output["judge_crossfit_fold"] = judge_fold
    output["oof_reference_distance"] = oof_distance
    output["oof_reference_dispersion"] = oof_dispersion
    output["expected_log_judge_sd"] = expected
    output["excess_log_judge_sd"] = np.where(eligible, excess, np.nan)
    output["excess_threshold"] = threshold
    output["high_excess_disagreement"] = eligible & (excess >= threshold)
    output["oof_absolute_error"] = np.where(development, error, np.nan)
    output["error_threshold"] = error_threshold
    output["high_error_risk"] = development & (error >= error_threshold)
    test = output.analysis_role == "official_test"
    protected_columns = [
        "judge_sample_sd", "excess_log_judge_sd", "high_excess_disagreement",
        "oof_absolute_error", "high_error_risk",
    ]
    output.loc[test, protected_columns] = [np.nan, np.nan, False, np.nan, False]
    path = V8_RESULTS_ROOT / "01_CONDITIONAL_RISK" / "conditional_disagreement_v8.parquet"
    output.to_parquet(path, index=False)
    thresholds = {
        "state": "TRAINING_THRESHOLDS_FROZEN_TEST_MASKED",
        "excess_log_sd_threshold": threshold,
        "error_score_threshold": error_threshold,
        "fit_judge_rows": int(len(fit_judge)),
        "fit_oof_expected_rows": int(np.isfinite(expected[fit_judge]).sum()),
        "test_targets_persisted": False,
        "conditional_model_sha256": sha256_file(model_path),
        "manifest_sha256": sha256_file(path),
    }
    write_json(V8_RESULTS_ROOT / "01_CONDITIONAL_RISK" / "conditional_thresholds_v8.json", thresholds)
    return {"status": "PASS", **thresholds}


def load_conditional_manifest_v8() -> pd.DataFrame:
    path = V8_RESULTS_ROOT / "01_CONDITIONAL_RISK" / "conditional_disagreement_v8.parquet"
    if not path.exists():
        raise RuntimeError("Run build-conditional-risk --protocol v8 first")
    return pd.read_parquet(path)


def reveal_test_disagreement_v8(manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach held-out targets only after the v8 contract is frozen."""
    freeze = V8_RESULTS_ROOT / "03_PILOT" / "contract_freeze_v8.json"
    if not freeze.exists():
        raise RuntimeError("The v8 contract is not frozen")
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("The v8 contract freeze is invalid")
    assets = load_v6_assets()
    frame = assets.frame.set_index("clip_uid").loc[manifest.clip_uid].reset_index()
    result = manifest.copy()
    test = result.analysis_role == "official_test"
    epsilon = float(load_v8_contract()["conditional_disagreement"]["epsilon"])
    raw_sd = frame.judge_sample_sd.to_numpy(dtype=float)
    eligible = frame.disagreement_primary_eligible.to_numpy(dtype=bool)
    excess = np.log(raw_sd + epsilon) - result.expected_log_judge_sd.to_numpy(dtype=float)
    threshold = result.excess_threshold.to_numpy(dtype=float)
    result.loc[test, "judge_sample_sd"] = raw_sd[test]
    result.loc[test, "excess_log_judge_sd"] = np.where(eligible[test], excess[test], np.nan)
    result.loc[test, "high_excess_disagreement"] = eligible[test] & (excess[test] >= threshold[test])
    return result
