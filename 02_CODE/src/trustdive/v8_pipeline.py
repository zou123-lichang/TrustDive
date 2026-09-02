from __future__ import annotations

import json
import platform
import sys

from .budget import consumed_hours, gpu_budget_entry, ledger_path
from .util import git_dirty_count, git_head, sha256_file, write_json
from .v8_analysis import analyze_review_v8
from .v8_attribution import build_dual_evidence_v8
from .v8_data import (
    V8_RESULTS_ROOT,
    audit_v8,
    build_conditional_disagreement_v8,
    ensure_v8_dirs,
    load_v8_contract,
    v8_paths,
)
from .v8_modeling import (
    freeze_contract_v8,
    pilot_v8,
    train_baselines_v8,
    train_final_v8,
)
from .v8_tokens import build_phase_tokens_v8


def _print(result: dict) -> dict:
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_audit_v8() -> dict:
    result = audit_v8()
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(2)
    return result


def command_build_conditional_risk_v8() -> dict:
    return _print(build_conditional_disagreement_v8())


def command_build_phase_tokens_v8() -> dict:
    final = (V8_RESULTS_ROOT / "03_PILOT" / "contract_freeze_v8.json").exists()
    contract = load_v8_contract()
    with gpu_budget_entry(
        f"v8:build-phase-tokens:{'final' if final else 'development'}",
        estimated_hours=float(contract["compute"]["phase_token_estimated_hours"]),
        paths=v8_paths(),
        force_gpu=True,
    ):
        result = build_phase_tokens_v8(final=final)
    return _print(result)


def command_train_baselines_v8() -> dict:
    return _print(train_baselines_v8())


def command_pilot_v8() -> dict:
    contract = load_v8_contract()
    with gpu_budget_entry(
        "v8:pilot",
        estimated_hours=float(contract["compute"]["pilot_estimated_hours"]),
        paths=v8_paths(),
        force_gpu=True,
    ):
        result = pilot_v8()
    return _print(result)


def command_freeze_v8() -> dict:
    return _print(freeze_contract_v8())


def command_train_final_v8() -> dict:
    final_tokens = V8_RESULTS_ROOT / "02_PHASE_TOKENS" / "reference_phase_tokens_final_v8.npz"
    if not final_tokens.exists():
        raise RuntimeError("Run build-phase-tokens --protocol v8 again after contract freeze")
    contract = load_v8_contract()
    with gpu_budget_entry(
        "v8:train-final",
        estimated_hours=float(contract["compute"]["final_estimated_hours"]),
        paths=v8_paths(),
        force_gpu=True,
    ):
        result = train_final_v8()
    return _print(result)


def command_build_dual_evidence_v8() -> dict:
    contract = load_v8_contract()
    with gpu_budget_entry(
        "v8:build-dual-evidence",
        estimated_hours=float(contract["compute"]["dual_evidence_estimated_hours"]),
        paths=v8_paths(),
        force_gpu=True,
    ):
        result = build_dual_evidence_v8()
    return _print(result)


def command_analyze_review_v8() -> dict:
    return _print(analyze_review_v8())


def command_verify_v8() -> dict:
    ensure_v8_dirs()
    contract = load_v8_contract()
    audit = audit_v8()
    pilot_path = V8_RESULTS_ROOT / "03_PILOT" / "pilot_gate_v8.json"
    pilot_status = json.loads(pilot_path.read_text(encoding="utf-8")).get("status") if pilot_path.exists() else None
    early_required = [
        V8_RESULTS_ROOT / "01_CONDITIONAL_RISK" / "conditional_disagreement_v8.parquet",
        V8_RESULTS_ROOT / "02_PHASE_TOKENS" / "reference_phase_tokens_development_v8.npz",
        pilot_path,
        V8_RESULTS_ROOT / "03_PILOT" / "pilot_trials_v8.csv",
        V8_RESULTS_ROOT / "RESULTS_DECISION_V8.md",
        V8_RESULTS_ROOT / "figures_v8" / "render_status_v8.json",
    ]
    final_required = [
        V8_RESULTS_ROOT / "02_PHASE_TOKENS" / "reference_phase_tokens_final_v8.npz",
        V8_RESULTS_ROOT / "03_PILOT" / "contract_freeze_v8.json",
        V8_RESULTS_ROOT / "04_FINAL" / "predictions_v8.parquet",
        V8_RESULTS_ROOT / "05_DUAL_EVIDENCE" / "dual_phase_evidence_v8.parquet",
        V8_RESULTS_ROOT / "06_REVIEW" / "review_priority_v8.parquet",
        V8_RESULTS_ROOT / "06_REVIEW" / "analysis_summary_v8.json",
    ]
    required = early_required if pilot_status == "STOP" else early_required + final_required
    missing = [str(path) for path in required if not path.exists()]
    figure_status = None
    figure_path = V8_RESULTS_ROOT / "figures_v8" / "render_status_v8.json"
    if figure_path.exists():
        figure_status = json.loads(figure_path.read_text(encoding="utf-8")).get("status")
    tracked = [
        path for path in V8_RESULTS_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md", ".parquet", ".npz"}
    ]
    budget = consumed_hours(v8_paths())
    checks = {
        "audit_pass": audit["status"] == "PASS",
        "protected_v1_v7_unchanged": audit["checks"]["protected_v1_v7_unchanged"],
        "required_outputs_exist": not missing,
        "gpu_budget_respected": budget <= float(contract["compute"]["gpu_budget_hours"]),
        "figure_workflow_closed": figure_status in {"PASS", "STOPPED_BY_EVIDENCE_GATE"},
        "pilot_terminal_state_valid": pilot_status in {"PASS", "STOP"},
    }
    ledger = ledger_path(v8_paths())
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing_outputs": missing,
        "gpu_hours_consumed": budget,
        "gpu_budget_hours": float(contract["compute"]["gpu_budget_hours"]),
        "pilot_status": pilot_status,
        "stopped_at_pilot": pilot_status == "STOP",
        "git_head": git_head(v8_paths().project),
        "git_dirty_count": git_dirty_count(v8_paths().project),
        "contract_sha256": sha256_file(v8_paths().contract),
        "gpu_ledger_sha256": sha256_file(ledger) if ledger.exists() else None,
        "output_hashes": {
            str(path.relative_to(v8_paths().project)): sha256_file(path) for path in sorted(tracked)
        },
        "python": sys.version,
        "platform": platform.platform(),
        "material_passport": "experiment-agent / verification / 2026-08-21 / VERIFIED / trustdive_phase_conflict_v8",
    }
    write_json(v8_paths().runs / "run_manifest_v8.json", result)
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(6)
    return result
