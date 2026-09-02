from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd

from .budget import consumed_hours, gpu_budget_entry, ledger_path
from .features import (
    extract_rgb_augmented_features,
    extract_rgb_features,
    extract_splash_features,
    extract_videomae_features,
)
from .util import git_dirty_count, git_head, sha256_file, write_json
from .v2_analysis import analyze_all_v2
from .v2_data import (
    V2_CONTRACT_PATH,
    V2_RESULTS_ROOT,
    audit_panel,
    build_panel_targets,
    ensure_v2_dirs,
    freeze_v2_contract,
    load_panel_targets,
    load_v2_contract,
    require_contract_frozen,
    require_v2_audit,
    v2_paths,
)
from .v2_modeling import (
    feature_inventory,
    train_baselines_v2,
    train_final_v2,
    tune_v2,
    tune_videomae_rescue,
)


def command_audit_panel_v2() -> dict:
    result = audit_panel()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(20)
    return result


def command_build_panel_targets_v2() -> pd.DataFrame:
    require_v2_audit()
    frame = build_panel_targets(write=True)
    summary = json.loads(
        (V2_RESULTS_ROOT / "00_AUDIT" / "panel_targets_summary_v2.json").read_text(
            encoding="utf-8"
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return frame


def command_extract_features_v2(modalities: list[str]) -> dict:
    require_v2_audit()
    frame = load_panel_targets()
    paths = v2_paths()
    unknown = set(modalities) - {"rgb", "splash"}
    if unknown:
        raise ValueError(f"v2 primary extraction supports rgb and optional splash, not {sorted(unknown)}")
    result: dict = {}
    if "rgb" in modalities:
        with gpu_budget_entry("v2:extract-features:rgb", paths=paths):
            result["rgb"] = extract_rgb_features(frame, paths)
    if "splash" in modalities:
        splash = extract_splash_features(frame, paths)
        result["splash"] = {"rows": int(len(splash)), "valid": int(splash.splash_valid.sum())}
    result["inventory"] = feature_inventory(frame)
    result["gpu_hours_consumed_v2"] = consumed_hours(paths)
    if result["inventory"]["base_missing"]:
        raise RuntimeError("The full v2 RGB feature inventory is incomplete")
    write_json(V2_RESULTS_ROOT / "01_FEATURES" / "feature_extraction_v2.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_train_baselines_v2() -> dict:
    require_v2_audit()
    frame = load_panel_targets()
    inventory = feature_inventory(frame)
    if inventory["base_missing"]:
        raise RuntimeError("Extract all v2 RGB features before baseline training")
    with gpu_budget_entry("v2:train-baselines", paths=v2_paths()):
        result = train_baselines_v2(frame)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_tune_v2() -> dict:
    require_v2_audit()
    frame = load_panel_targets()
    fit_uids = set(frame.loc[frame.analysis_role == "fit", "clip_uid"])
    inventory = feature_inventory(frame, augmented_fit=True)
    if inventory["base_missing"]:
        raise RuntimeError("Extract all v2 RGB features before tuning")
    if inventory["augmented_fit_missing"]:
        with gpu_budget_entry("v2:extract-train-augmentation-view", paths=v2_paths()):
            augmentation = extract_rgb_augmented_features(
                frame, fit_uids, v2_paths(), overwrite=False, seed=20260817
            )
        write_json(V2_RESULTS_ROOT / "01_FEATURES" / "visual_augmentation_v2.json", augmentation)
    selection_path = V2_RESULTS_ROOT / "02_TUNING" / "selected_config_v2.json"
    existing = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.exists()
        else None
    )
    if existing and existing.get("videomae_rescue_status") == "DEFERRED_UNTIL_TRIGGER_REVIEW":
        result = existing
    else:
        with gpu_budget_entry("v2:tune", paths=v2_paths()):
            result = tune_v2(frame)

    if result.get("videomae_rescue_triggered") and result.get("videomae_rescue_status") != "COMPLETED":
        contract = load_v2_contract()
        video_inventory = feature_inventory(frame, backbone="videomae")
        probe_path = V2_RESULTS_ROOT / "01_FEATURES" / "videomae_probe_v2.json"
        if video_inventory["base_present"] < 20:
            probe_keys = set(frame.loc[frame.analysis_role == "fit", "clip_uid"].iloc[:20])
            with gpu_budget_entry("v2:videomae-probe", paths=v2_paths()):
                probe = extract_videomae_features(
                    frame, v2_paths(), only_keys=probe_keys, overwrite=False
                )
            write_json(probe_path, probe)
        else:
            probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {
                "estimated_full_hours": 0.0,
                "peak_vram_gb": 0.0,
                "status": "CACHE_REUSED",
            }
        if (
            float(probe.get("estimated_full_hours", 0.0))
            > float(contract["compute"]["videomae_rescue_budget_hours"])
            or float(probe.get("peak_vram_gb", 0.0))
            > float(contract["compute"]["max_peak_vram_gb"])
        ):
            result["videomae_rescue_status"] = "PROBE_FAILED_BUDGET_OR_VRAM"
            result["videomae_probe"] = probe
            write_json(selection_path, result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return result
        video_inventory = feature_inventory(frame, backbone="videomae")
        if video_inventory["base_missing"]:
            with gpu_budget_entry("v2:videomae-full-extraction", paths=v2_paths()):
                extraction = extract_videomae_features(frame, v2_paths(), overwrite=False)
            write_json(
                V2_RESULTS_ROOT / "01_FEATURES" / "videomae_extraction_v2.json", extraction
            )
        with gpu_budget_entry("v2:videomae-rescue-training", paths=v2_paths()):
            result = tune_videomae_rescue(frame)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_freeze_contract_v2() -> dict:
    result = freeze_v2_contract()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_train_final_v2() -> dict:
    require_contract_frozen()
    frame = load_panel_targets()
    with gpu_budget_entry("v2:train-final", paths=v2_paths()):
        result = train_final_v2(frame)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_analyze_v2(part: str) -> dict:
    if part != "all":
        raise ValueError("The locked v2 analysis runs all prespecified parts together")
    result = analyze_all_v2()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_verify_v2() -> dict:
    ensure_v2_dirs()
    contract = load_v2_contract()
    audit_path = V2_RESULTS_ROOT / "00_AUDIT" / "panel_audit_v2.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    final_path = V2_RESULTS_ROOT / "03_FINAL" / "predictions_v2.parquet"
    analysis_path = V2_RESULTS_ROOT / "04_ANALYSIS" / "analysis_summary_v2.json"
    frozen = bool(contract["state"]["frozen"])
    test_outputs_legal = frozen or not final_path.exists()
    anchors = audit.get("v1_anchor_checks", {}) if audit else {}
    output_hashes = {}
    for path in sorted(V2_RESULTS_ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md"}:
            output_hashes[str(path.relative_to(v2_paths().project))] = sha256_file(path)
    checks = {
        "audit_pass": bool(audit and audit.get("status") == "PASS"),
        "v1_anchors_unchanged": bool(anchors and all(anchors.values())),
        "test_outputs_legal": test_outputs_legal,
        "contract_selection_consistent": (
            not frozen
            or (V2_RESULTS_ROOT / "02_TUNING" / "selected_config_v2.json").exists()
        ),
        "analysis_requires_final": not analysis_path.exists() or final_path.exists(),
        "gpu_budget_respected": consumed_hours(v2_paths())
        <= float(contract["compute"]["gpu_budget_hours"]),
    }
    installed = sorted(
        {
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in distributions()
            if dist.metadata.get("Name")
        },
        key=str.lower,
    )
    environment_lock = v2_paths().code / "environment" / "requirements-lock-v2.txt"
    environment_lock.write_text("\n".join(installed) + "\n", encoding="utf-8")
    try:
        import torch

        accelerator = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - environment report only
        accelerator = {"error": repr(exc)}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "experiment_state": (
            "ANALYZED"
            if analysis_path.exists()
            else "FINAL_TRAINED"
            if final_path.exists()
            else "CONTRACT_FROZEN"
            if frozen
            else "DEVELOPMENT"
        ),
        "git_head": git_head(v2_paths().project),
        "git_dirty_count": git_dirty_count(v2_paths().project),
        "contract_sha256": sha256_file(V2_CONTRACT_PATH),
        "gpu_ledger_sha256": sha256_file(ledger_path(v2_paths()))
        if ledger_path(v2_paths()).exists()
        else None,
        "gpu_hours_consumed_v2": consumed_hours(v2_paths()),
        "output_hashes": output_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "accelerator": accelerator,
        "environment_lock": str(environment_lock.relative_to(v2_paths().project)),
        "environment_packages": len(installed),
    }
    destination = v2_paths().runs / "run_manifest_v2.json"
    write_json(destination, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(21)
    return result
