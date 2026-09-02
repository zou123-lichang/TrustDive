from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .budget import consumed_hours, gpu_budget_entry, ledger_path
from .util import git_dirty_count, git_head, sha256_file, write_json
from .v5_analysis import analyze_v5
from .v5_counterfactual import build_counterfactual_targets_v5, build_references_v5
from .v5_data import (
    V5_RESULTS_ROOT,
    audit_v5,
    ensure_v5_dirs,
    freeze_v5_contract,
    load_v5_contract,
    v5_paths,
)
from .v5_modeling import optimize_baseline_v5, pilot_cfpd_plus_v5, train_final_v5
from .v5_stress import stress_test_v5
from .v4_data import RICA2_ROOT


def _print(result: dict) -> dict:
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_audit_v5() -> dict:
    result = audit_v5()
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(2)
    return result


def _export_teacher_subprocess() -> None:
    summary = V5_RESULTS_ROOT / "01_REFERENCES" / "teacher_export_v5.json"
    if summary.exists() and json.loads(summary.read_text(encoding="utf-8")).get("status") == "PASS":
        return
    conda_exe = Path(
        os.environ.get("CONDA_EXE", "conda")
    )
    command = [
        str(conda_exe),
        "run",
        "-n",
        "trustdive_rica2",
        "python",
        "-m",
        "trustdive.v5_teacher",
    ]
    # RICA2's pinned config intentionally keeps its text-embedding asset path
    # relative to the external repository root.
    subprocess.run(command, cwd=RICA2_ROOT, check=True)


def command_build_references_v5() -> dict:
    contract = load_v5_contract()
    with gpu_budget_entry(
        "v5:export-teacher-and-build-references",
        estimated_hours=float(contract["compute"]["teacher_export_estimated_hours"]),
        paths=v5_paths(),
        force_gpu=True,
    ):
        _export_teacher_subprocess()
        result = build_references_v5()
    if result["status"] != "PASS":
        raise SystemExit(3)
    return _print(result)


def command_optimize_baseline_v5() -> dict:
    contract = load_v5_contract()
    with gpu_budget_entry(
        "v5:optimize-rica2-ar",
        estimated_hours=float(contract["compute"]["baseline_estimated_hours"]),
        paths=v5_paths(),
        force_gpu=True,
    ):
        result = optimize_baseline_v5()
    return _print(result)


def command_build_counterfactuals_v5() -> dict:
    contract = load_v5_contract()
    with gpu_budget_entry(
        "v5:build-counterfactuals",
        estimated_hours=float(contract["compute"]["counterfactual_estimated_hours"]),
        paths=v5_paths(),
        force_gpu=True,
    ):
        result = build_counterfactual_targets_v5()
    if result["status"] != "PASS":
        raise SystemExit(4)
    return _print(result)


def command_pilot_v5() -> dict:
    contract = load_v5_contract()
    with gpu_budget_entry(
        "v5:pilot-cfpd-plus",
        estimated_hours=float(contract["compute"]["pilot_estimated_hours"]),
        paths=v5_paths(),
        force_gpu=True,
    ):
        result = pilot_cfpd_plus_v5()
    return _print(result)


def command_freeze_v5() -> dict:
    return _print(freeze_v5_contract())


def command_train_final_v5() -> dict:
    contract = load_v5_contract()
    with gpu_budget_entry(
        "v5:train-final",
        estimated_hours=float(contract["compute"]["final_estimated_hours"]),
        paths=v5_paths(),
        force_gpu=True,
    ):
        result = train_final_v5()
    return _print(result)


def command_stress_v5() -> dict:
    contract = load_v5_contract()
    with gpu_budget_entry(
        "v5:stress-test",
        estimated_hours=float(contract["compute"]["stress_estimated_hours"]),
        paths=v5_paths(),
        force_gpu=True,
    ):
        result = stress_test_v5()
    return _print(result)


def command_analyze_v5(part: str) -> dict:
    if part != "all":
        raise ValueError("The v5 contract runs score, trace, stability, and review together")
    return _print(analyze_v5())


def _stage_state() -> tuple[str, list[Path]]:
    ordered = [
        ("AUDITED", V5_RESULTS_ROOT / "00_AUDIT" / "audit_v5.json"),
        ("REFERENCES", V5_RESULTS_ROOT / "01_REFERENCES" / "reference_summary_v5.json"),
        ("BASELINE", V5_RESULTS_ROOT / "02_BASELINE" / "selected_baseline_v5.json"),
        ("COUNTERFACTUAL", V5_RESULTS_ROOT / "03_COUNTERFACTUAL" / "counterfactual_summary_v5.json"),
        ("PILOT", V5_RESULTS_ROOT / "04_PILOT" / "selected_cfpd_v5.json"),
        ("FROZEN", V5_RESULTS_ROOT / "04_PILOT" / "contract_freeze_v5.json"),
        ("FINAL", V5_RESULTS_ROOT / "05_FINAL" / "predictions_cfpd_plus_v5.parquet"),
        ("STRESS", V5_RESULTS_ROOT / "06_STRESS" / "trace_stress_v5.parquet"),
        ("ANALYZED", V5_RESULTS_ROOT / "07_ANALYSIS" / "analysis_summary_v5.json"),
    ]
    existing = [path for _, path in ordered if path.exists()]
    state = "NOT_STARTED"
    for label, path in ordered:
        if path.exists():
            state = label
        else:
            break
    stop_path = V5_RESULTS_ROOT / "04_PILOT" / "stop_decision_v5.json"
    if state == "PILOT" and stop_path.exists():
        state = "PILOT_STOP"
        existing.append(stop_path)
    return state, existing


def command_verify_v5() -> dict:
    ensure_v5_dirs()
    contract = load_v5_contract()
    audit = audit_v5()
    state, existing = _stage_state()
    output_hashes = {
        str(path.relative_to(v5_paths().project)): sha256_file(path)
        for path in existing
        if path.is_file()
    }
    required_by_state = {
        "FINAL": [
            V5_RESULTS_ROOT / "05_FINAL" / "baseline_predictions_v5.parquet",
            V5_RESULTS_ROOT / "05_FINAL" / "ablation_summary_v5.csv",
        ],
        "STRESS": [V5_RESULTS_ROOT / "06_STRESS" / "phase_evidence_v5.parquet"],
        "ANALYZED": [
            V5_RESULTS_ROOT / "07_ANALYSIS" / "review_priority_v5.parquet",
            V5_RESULTS_ROOT / "07_ANALYSIS" / "sota_comparison_v5.csv",
        ],
    }
    missing = []
    labels = ["NOT_STARTED", "AUDITED", "REFERENCES", "BASELINE", "COUNTERFACTUAL", "PILOT", "PILOT_STOP", "FROZEN", "FINAL", "STRESS", "ANALYZED"]
    state_index = labels.index(state)
    for label, paths in required_by_state.items():
        if state_index >= labels.index(label):
            missing.extend(str(path) for path in paths if not path.exists())
    budget = consumed_hours(v5_paths())
    result = {
        "status": "PASS"
        if audit["status"] == "PASS"
        and not missing
        and budget <= float(contract["compute"]["gpu_budget_hours"])
        else "FAIL",
        "experiment_state": state,
        "historical_anchors_unchanged": audit["checks"]["historical_anchors_unchanged"],
        "missing_required_outputs": missing,
        "gpu_hours_consumed_v5": budget,
        "gpu_budget_hours": float(contract["compute"]["gpu_budget_hours"]),
        "git_head": git_head(v5_paths().project),
        "git_dirty_count": git_dirty_count(v5_paths().project),
        "contract_sha256": sha256_file(v5_paths().contract),
        "gpu_ledger_sha256": sha256_file(ledger_path(v5_paths()))
        if ledger_path(v5_paths()).exists()
        else None,
        "output_hashes": output_hashes,
        "material_passport": "experiment-agent / verification / 2026-08-18 / VERIFIED / trustdive_cfpd_plus_v5",
    }
    write_json(v5_paths().runs / "run_manifest_v5.json", result)
    return _print(result)
