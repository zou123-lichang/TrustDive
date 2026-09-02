from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import distributions

from .budget import consumed_hours, gpu_budget_entry, ledger_path
from .util import git_dirty_count, git_head, sha256_file, write_json
from .v3_analysis import analyze_all_v3
from .v3_data import (
    V3_CONTRACT_PATH,
    V3_RESULTS_ROOT,
    audit_trace_v3,
    ensure_v3_dirs,
    freeze_v3_contract,
    load_v3_contract,
    v3_paths,
)
from .v3_modeling import build_teacher_targets_v3, pilot_trace_v3, train_final_trace_v3
from .v3_stress import stress_test_v3


def command_audit_trace_v3() -> dict:
    result = audit_trace_v3()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(30)
    return result


def command_build_teacher_targets_v3() -> dict:
    frame = build_teacher_targets_v3()
    result = json.loads((V3_RESULTS_ROOT / "01_TEACHER" / "teacher_targets_summary_v3.json").read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_pilot_trace_v3() -> dict:
    with gpu_budget_entry("v3:pilot-trace", paths=v3_paths()):
        result = pilot_trace_v3()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(31)
    return result


def command_freeze_contract_v3() -> dict:
    result = freeze_v3_contract()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_train_final_v3() -> dict:
    with gpu_budget_entry("v3:train-final", paths=v3_paths()):
        result = train_final_trace_v3()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_stress_test_v3() -> dict:
    with gpu_budget_entry("v3:stress-test", paths=v3_paths()):
        result = stress_test_v3()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_analyze_v3(part: str) -> dict:
    if part != "all":
        raise ValueError("The locked v3 analysis runs all parts together")
    result = analyze_all_v3()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_verify_v3() -> dict:
    ensure_v3_dirs()
    contract = load_v3_contract()
    audit_path = V3_RESULTS_ROOT / "00_AUDIT" / "trace_audit_v3.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    anchor_checks = {}
    if audit:
        for key, relative in audit["anchor_paths"].items():
            anchor_checks[key] = sha256_file(v3_paths().project / relative) == audit["anchor_hashes"][key]
    required = [
        V3_RESULTS_ROOT / "01_TEACHER" / "teacher_targets_v3.parquet",
        V3_RESULTS_ROOT / "02_PILOT" / "selected_config_v3.json",
        V3_RESULTS_ROOT / "03_FINAL" / "predictions_trace_v3.parquet",
        V3_RESULTS_ROOT / "04_STRESS" / "trace_stress_v3.parquet",
        V3_RESULTS_ROOT / "05_ANALYSIS" / "analysis_summary_v3.json",
    ]
    output_hashes_before = {str(path.relative_to(v3_paths().project)): sha256_file(path) for path in required if path.exists()}
    if all(path.exists() for path in required[-2:]):
        analyze_all_v3()
    output_hashes_after = {str(path.relative_to(v3_paths().project)): sha256_file(path) for path in required if path.exists()}
    environment_lock = v3_paths().code / "environment" / "requirements-lock-v3.txt"
    installed = sorted({f"{dist.metadata['Name']}=={dist.version}" for dist in distributions() if dist.metadata.get("Name")}, key=str.lower)
    environment_lock.write_text("\n".join(installed) + "\n", encoding="utf-8")
    checks = {
        "audit_pass": bool(audit and audit.get("status") == "PASS"),
        "v1_v2_anchors_unchanged": bool(anchor_checks and all(anchor_checks.values())),
        "contract_frozen": bool(contract["state"]["frozen"]),
        "required_outputs_present": all(path.exists() for path in required),
        "analysis_reproducible": output_hashes_before == output_hashes_after,
        "gpu_budget_respected": consumed_hours(v3_paths()) <= float(contract["compute"]["gpu_budget_hours"]),
    }
    try:
        import torch
        accelerator = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as exc:
        accelerator = {"error": repr(exc)}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "anchor_checks": anchor_checks,
        "git_head": git_head(v3_paths().project),
        "git_dirty_count": git_dirty_count(v3_paths().project),
        "contract_sha256": sha256_file(V3_CONTRACT_PATH),
        "gpu_hours_consumed_v3": consumed_hours(v3_paths()),
        "gpu_ledger_sha256": sha256_file(ledger_path(v3_paths())) if ledger_path(v3_paths()).exists() else None,
        "output_hashes": output_hashes_after,
        "python": sys.version,
        "platform": platform.platform(),
        "accelerator": accelerator,
        "environment_lock": str(environment_lock.relative_to(v3_paths().project)),
    }
    write_json(v3_paths().runs / "run_manifest_v3.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(32)
    return result
