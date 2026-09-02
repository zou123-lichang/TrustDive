from __future__ import annotations

import json
import platform
import sys

from .budget import consumed_hours, gpu_budget_entry, ledger_path
from .util import git_dirty_count, git_head, sha256_file, write_json
from .v7_analysis import analyze_risk_v7
from .v7_attribution import build_phase_evidence_v7
from .v7_data import (
    V7_RESULTS_ROOT,
    audit_v7,
    build_risk_task_v7,
    ensure_v7_dirs,
    load_v7_contract,
    v7_paths,
)
from .v7_modeling import train_baselines_v7, train_final_v7


def _print(result: dict) -> dict:
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_audit_v7() -> dict:
    result = audit_v7()
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(2)
    return result


def command_build_risk_task_v7() -> dict:
    return _print(build_risk_task_v7())


def command_train_baselines_v7() -> dict:
    return _print(train_baselines_v7())


def command_train_v7() -> dict:
    return _print(train_final_v7())


def command_build_phase_evidence_v7() -> dict:
    contract = load_v7_contract()
    with gpu_budget_entry(
        "v7:build-phase-evidence",
        estimated_hours=float(contract["compute"]["build_phase_evidence_estimated_hours"]),
        paths=v7_paths(),
        force_gpu=True,
    ):
        result = build_phase_evidence_v7()
    return _print(result)


def command_analyze_risk_v7() -> dict:
    return _print(analyze_risk_v7())


def command_verify_v7() -> dict:
    ensure_v7_dirs()
    contract = load_v7_contract()
    audit = audit_v7()
    required = [
        V7_RESULTS_ROOT / "01_RISK_TASK" / "risk_task_manifest_v7.parquet",
        V7_RESULTS_ROOT / "02_BASELINES" / "selected_model_v7.json",
        V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet",
        V7_RESULTS_ROOT / "03_SCORE" / "crossfit_predictions_v7.parquet",
        V7_RESULTS_ROOT / "04_PHASE_EVIDENCE" / "phase_evidence_v7.parquet",
        V7_RESULTS_ROOT / "05_RISK_REVIEW" / "review_priority_v7.parquet",
        V7_RESULTS_ROOT / "05_RISK_REVIEW" / "analysis_summary_v7.json",
        V7_RESULTS_ROOT / "figures_v7" / "render_status_v7.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    tracked = [
        path for path in V7_RESULTS_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md", ".parquet"}
    ]
    ledger = ledger_path(v7_paths())
    budget = consumed_hours(v7_paths())
    figure_status = None
    figure_path = V7_RESULTS_ROOT / "figures_v7" / "render_status_v7.json"
    if figure_path.exists():
        figure_status = json.loads(figure_path.read_text(encoding="utf-8")).get("status")
    checks = {
        "audit_pass": audit["status"] == "PASS",
        "protected_v1_v6_unchanged": audit["checks"]["protected_v1_v6_unchanged"],
        "required_outputs_exist": not missing,
        "gpu_budget_respected": budget <= float(contract["compute"]["gpu_budget_hours"]),
        "figure_qa_pass": figure_status == "PASS",
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing_outputs": missing,
        "gpu_hours_consumed": budget,
        "gpu_budget_hours": float(contract["compute"]["gpu_budget_hours"]),
        "git_head": git_head(v7_paths().project),
        "git_dirty_count": git_dirty_count(v7_paths().project),
        "contract_sha256": sha256_file(v7_paths().contract),
        "gpu_ledger_sha256": sha256_file(ledger) if ledger.exists() else None,
        "output_hashes": {
            str(path.relative_to(v7_paths().project)): sha256_file(path)
            for path in sorted(tracked)
        },
        "python": sys.version,
        "platform": platform.platform(),
        "material_passport": "experiment-agent / verification / 2026-08-20 / VERIFIED / trustdive_risk_v7",
    }
    write_json(v7_paths().runs / "run_manifest_v7.json", result)
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(6)
    return result

