from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

from .budget import consumed_hours, gpu_budget_entry, ledger_path
from .util import git_dirty_count, git_head, sha256_file, write_json
from .v4_analysis import analyze_v4
from .v4_counterfactual import build_counterfactual_targets_v4
from .v4_data import (
    V4_CONTRACT_PATH,
    V4_RESULTS_ROOT,
    audit_v4,
    ensure_v4_dirs,
    freeze_v4_contract,
    load_v4_contract,
    v4_paths,
)
from .v4_modeling import pilot_cfpd_v4, train_final_cfpd_v4
from .v4_stress import stress_test_v4
from .v4_teacher import (
    export_teacher_v4,
    extract_teacher_features_v4,
    prepare_teacher_assets_v4,
    train_teacher_v4,
)


def _print(result: dict) -> dict:
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_audit_v4() -> dict:
    result = audit_v4()
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(40)
    return result


def command_prepare_teacher_v4() -> dict:
    with gpu_budget_entry("v4:teacher-smoke", estimated_hours=1.0, paths=v4_paths()):
        preparation = prepare_teacher_assets_v4()
        if preparation["status"] != "PASS":
            result = {"status": "STOP", "stage": "teacher_preparation", "preparation": preparation}
        else:
            extraction = extract_teacher_features_v4(limit=100)
            contract = load_v4_contract()
            checks = {
                "extraction": extraction["status"] == "PASS",
                "estimated_full_hours": extraction["estimated_full_hours"]
                <= float(contract["compute"]["feature_extraction_budget_hours"]),
                "peak_vram": extraction["peak_vram_bytes"]
                <= float(contract["external_teacher"]["maximum_peak_vram_gib"]) * 2**30,
            }
            result = {
                "status": "PASS" if all(checks.values()) else "STOP",
                "stage": "teacher_smoke",
                "checks": checks,
                "preparation": preparation,
                "feature_probe": extraction,
            }
        write_json(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_probe_gate_v4.json", result)
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(41)
    return result


def command_train_teacher_v4() -> dict:
    probe = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_probe_gate_v4.json"
    if not probe.exists() or json.loads(probe.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Teacher smoke gate must pass before formal RICA2 training")
    # RICA2 runs in an isolated subprocess, so the outer Python process cannot
    # observe its CUDA allocator. Mark this known GPU subprocess explicitly.
    with gpu_budget_entry(
        "v4:train-teacher", estimated_hours=13.0, paths=v4_paths(), force_gpu=True
    ):
        result = train_teacher_v4()
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(42)
    return result


def command_export_teacher_v4() -> dict:
    with gpu_budget_entry("v4:export-teacher", estimated_hours=0.5, paths=v4_paths()):
        result = export_teacher_v4()
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(43)
    return result


def command_build_counterfactuals_v4() -> dict:
    with gpu_budget_entry("v4:build-counterfactuals", estimated_hours=2.0, paths=v4_paths()):
        result = build_counterfactual_targets_v4()
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(44)
    return result


def command_pilot_v4() -> dict:
    with gpu_budget_entry("v4:pilot-cfpd", estimated_hours=1.0, paths=v4_paths()):
        result = pilot_cfpd_v4()
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(45)
    return result


def command_freeze_contract_v4() -> dict:
    return _print(freeze_v4_contract())


def command_train_final_v4() -> dict:
    with gpu_budget_entry("v4:train-final-cfpd", estimated_hours=1.5, paths=v4_paths()):
        result = train_final_cfpd_v4()
    return _print(result)


def command_stress_test_v4() -> dict:
    with gpu_budget_entry("v4:stress-test", estimated_hours=0.5, paths=v4_paths()):
        result = stress_test_v4()
    return _print(result)


def command_analyze_v4(part: str) -> dict:
    if part != "all":
        raise ValueError("The locked v4 analysis runs all score, trace, stability, and review parts together")
    return _print(analyze_v4())


def _freeze_current_environment(path: Path) -> None:
    installed = sorted(
        {f"{dist.metadata['Name']}=={dist.version}" for dist in distributions() if dist.metadata.get("Name")},
        key=str.lower,
    )
    path.write_text("\n".join(installed) + "\n", encoding="utf-8")


def _freeze_teacher_environment(path: Path) -> None:
    executable = Path(sys.prefix).parent / "trustdive_rica2" / "python.exe"
    if not executable.exists():
        configured = os.environ.get("TRUSTDIVE_RICA2_PYTHON")
        if not configured:
            raise RuntimeError(
                "Set TRUSTDIVE_RICA2_PYTHON to the Python executable in the RICA2 environment"
            )
        executable = Path(configured)
    output = subprocess.check_output([str(executable), "-m", "pip", "freeze"], text=True, encoding="utf-8")
    path.write_text(output.replace("\r\n", "\n"), encoding="utf-8")


def _stage_state() -> tuple[str, bool, list[Path]]:
    teacher = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_gate_v4.json"
    counter = V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "counterfactual_summary_v4.json"
    pilot = V4_RESULTS_ROOT / "03_PILOT" / "selected_config_v4.json"
    final = V4_RESULTS_ROOT / "04_FINAL" / "predictions_v4.parquet"
    stress = V4_RESULTS_ROOT / "05_STRESS" / "trace_stress_v4.parquet"
    analysis = V4_RESULTS_ROOT / "06_ANALYSIS" / "analysis_summary_v4.json"
    sequence = [teacher, counter, pilot]
    labels = ["STOPPED_TEACHER", "STOPPED_COUNTERFACTUAL", "STOPPED_PILOT"]
    for path, label in zip(sequence, labels):
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") in {"STOP", "FAIL"}:
            downstream = sequence[sequence.index(path) + 1 :] + [final, stress, analysis]
            return label, not any(item.exists() for item in downstream), [path]
    required = [teacher, counter, pilot, final, stress, analysis]
    complete = all(path.exists() for path in required)
    return ("COMPLETE" if complete else "IN_PROGRESS"), complete, required


def command_verify_v4() -> dict:
    ensure_v4_dirs()
    contract = load_v4_contract()
    audit_path = V4_RESULTS_ROOT / "00_AUDIT" / "audit_v4.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    anchor_checks = {}
    if audit:
        for key, relative in audit["anchor_paths"].items():
            path = v4_paths().project / relative
            anchor_checks[key] = path.exists() and sha256_file(path) == audit["anchor_hashes"].get(key)

    state, stage_valid, required = _stage_state()
    before = {str(path.relative_to(v4_paths().project)): sha256_file(path) for path in required if path.exists()}
    if state == "COMPLETE":
        analyze_v4()
    after = {str(path.relative_to(v4_paths().project)): sha256_file(path) for path in required if path.exists()}

    environment = v4_paths().code / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    current_lock = environment / "requirements-lock-v4.txt"
    teacher_lock = environment / "requirements-lock-rica2-v4.txt"
    _freeze_current_environment(current_lock)
    _freeze_teacher_environment(teacher_lock)
    budget_ok = consumed_hours(v4_paths()) <= float(contract["compute"]["gpu_budget_hours"])
    checks = {
        "audit_pass": bool(audit and audit.get("status") == "PASS"),
        "v1_v3_anchors_unchanged": bool(anchor_checks and all(anchor_checks.values())),
        "stage_outputs_consistent": stage_valid,
        "analysis_reproducible": before == after,
        "gpu_budget_respected": budget_ok,
        "contract_state_consistent": (state == "COMPLETE" and bool(contract["state"]["frozen"]))
        or (state != "COMPLETE" and not bool(contract["state"]["official_student_test_unlocked"])),
    }
    try:
        import torch

        accelerator = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        accelerator = {"error": repr(exc)}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "experiment_state": state,
        "checks": checks,
        "anchor_checks": anchor_checks,
        "material_passport": "experiment-agent / verification / 2026-08-18 / VERIFIED / trustdive_cfpd_v4",
        "git_head": git_head(v4_paths().project),
        "git_dirty_count": git_dirty_count(v4_paths().project),
        "contract_sha256": sha256_file(V4_CONTRACT_PATH),
        "gpu_hours_consumed_v4": consumed_hours(v4_paths()),
        "gpu_ledger_sha256": sha256_file(ledger_path(v4_paths())) if ledger_path(v4_paths()).exists() else None,
        "environment_lock_sha256": sha256_file(current_lock),
        "teacher_environment_lock_sha256": sha256_file(teacher_lock),
        "output_hashes": after,
        "python": sys.version,
        "platform": platform.platform(),
        "accelerator": accelerator,
    }
    write_json(v4_paths().runs / "run_manifest_v4.json", result)
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(46)
    return result
