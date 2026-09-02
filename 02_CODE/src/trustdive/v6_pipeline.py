from __future__ import annotations

import json
from pathlib import Path

from .budget import consumed_hours, gpu_budget_entry, ledger_path
from .util import git_dirty_count, git_head, sha256_file, write_json
from .v6_analysis import analyze_review_v6
from .v6_attribution import build_attributions_v6, pilot_gate_v6
from .v6_data import (
    V6_RESULTS_ROOT,
    audit_v6,
    ensure_v6_dirs,
    freeze_v6_contract,
    load_v6_contract,
    v6_paths,
)
from .v6_latents import extract_teacher_latents_v6
from .v6_modeling import evaluate_final_v6, optimize_adapter_v6


def _print(result: dict) -> dict:
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_audit_v6() -> dict:
    result = audit_v6()
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(2)
    return result


def command_extract_latents_v6() -> dict:
    contract = load_v6_contract()
    with gpu_budget_entry(
        "v6:extract-latents",
        estimated_hours=float(contract["compute"]["latent_extraction_estimated_hours"]),
        paths=v6_paths(), force_gpu=True,
    ):
        result = extract_teacher_latents_v6()
    _print(result)
    if result["status"] != "PASS":
        raise SystemExit(3)
    return result


def command_optimize_adapter_v6() -> dict:
    contract = load_v6_contract()
    with gpu_budget_entry(
        "v6:optimize-adapter",
        estimated_hours=float(contract["compute"]["adapter_optimization_estimated_hours"]),
        paths=v6_paths(), force_gpu=True,
    ):
        result = optimize_adapter_v6()
    return _print(result)


def command_build_attributions_v6() -> dict:
    contract = load_v6_contract()
    with gpu_budget_entry(
        "v6:build-pilot-attributions",
        estimated_hours=float(contract["compute"]["pilot_attribution_estimated_hours"]),
        paths=v6_paths(), force_gpu=True,
    ):
        result = build_attributions_v6(final=False)
    return _print(result)


def command_pilot_v6() -> dict:
    return _print(pilot_gate_v6())


def command_freeze_v6() -> dict:
    return _print(freeze_v6_contract())


def command_evaluate_final_v6() -> dict:
    contract = load_v6_contract()
    with gpu_budget_entry(
        "v6:evaluate-final",
        estimated_hours=float(contract["compute"]["final_evaluation_estimated_hours"]),
        paths=v6_paths(), force_gpu=True,
    ):
        score = evaluate_final_v6()
    result = {"score": score, "attribution": {"status": "LOCKED"}}
    if score["status"] == "PASS":
        with gpu_budget_entry(
            "v6:build-final-attributions",
            estimated_hours=float(contract["compute"]["final_attribution_estimated_hours"]),
            paths=v6_paths(), force_gpu=True,
        ):
            result["attribution"] = build_attributions_v6(final=True)
    return _print(result)


def command_analyze_review_v6() -> dict:
    return _print(analyze_review_v6())


def _experiment_state() -> tuple[str, list[Path]]:
    ordered = [
        ("AUDITED", V6_RESULTS_ROOT / "00_AUDIT" / "audit_v6.json"),
        ("LATENTS", V6_RESULTS_ROOT / "01_LATENTS" / "latent_extraction_v6.json"),
        ("ADAPTER", V6_RESULTS_ROOT / "02_ADAPTER" / "selected_adapter_v6.json"),
        ("ATTRIBUTION", V6_RESULTS_ROOT / "03_ATTRIBUTION" / "pilot_attribution_summary_v6.json"),
        ("PILOT", V6_RESULTS_ROOT / "04_PILOT" / "pilot_gate_v6.json"),
        ("FROZEN", V6_RESULTS_ROOT / "04_PILOT" / "contract_freeze_v6.json"),
        ("FINAL", V6_RESULTS_ROOT / "05_FINAL" / "final_score_summary_v6.json"),
        ("REVIEW", V6_RESULTS_ROOT / "06_REVIEW" / "analysis_summary_v6.json"),
        ("FIGURES", V6_RESULTS_ROOT / "figures_v6" / "render_status_v6.json"),
    ]
    state = "NOT_STARTED"
    files: list[Path] = []
    for label, path in ordered:
        if not path.exists():
            break
        state = label
        files.append(path)
    pilot = V6_RESULTS_ROOT / "04_PILOT" / "pilot_gate_v6.json"
    if pilot.exists() and json.loads(pilot.read_text(encoding="utf-8")).get("status") != "PASS":
        state = "PILOT_STOP"
    final = V6_RESULTS_ROOT / "05_FINAL" / "final_score_summary_v6.json"
    if final.exists() and json.loads(final.read_text(encoding="utf-8")).get("status") != "PASS":
        state = "FINAL_SCORE_STOP"
    return state, files


def command_verify_v6() -> dict:
    ensure_v6_dirs()
    contract = load_v6_contract()
    audit = audit_v6()
    state, files = _experiment_state()
    budget = consumed_hours(v6_paths())
    required = {
        "LATENTS": [V6_RESULTS_ROOT / "01_LATENTS" / "teacher_predictions_v6.parquet", V6_RESULTS_ROOT / "01_LATENTS" / "teacher_latents_v6.npz"],
        "ADAPTER": [V6_RESULTS_ROOT / "02_ADAPTER" / "adapter_trials_v6.csv"],
        "ATTRIBUTION": [V6_RESULTS_ROOT / "03_ATTRIBUTION" / "phase_evidence_pilot_v6.parquet"],
        "FINAL": [V6_RESULTS_ROOT / "05_FINAL" / "adapter_predictions_v6.parquet", V6_RESULTS_ROOT / "05_FINAL" / "crossfit_predictions_v6.parquet"],
        "REVIEW": [V6_RESULTS_ROOT / "06_REVIEW" / "review_priority_v6.parquet", V6_RESULTS_ROOT / "RESULTS_DECISION_V6.md"],
    }
    order = ["NOT_STARTED", "AUDITED", "LATENTS", "ADAPTER", "ATTRIBUTION", "PILOT", "PILOT_STOP", "FROZEN", "FINAL", "FINAL_SCORE_STOP", "REVIEW", "FIGURES"]
    current = order.index(state)
    missing: list[str] = []
    for label, paths in required.items():
        if current >= order.index(label):
            missing.extend(str(path) for path in paths if not path.exists())
    tracked_outputs = [
        path for path in V6_RESULTS_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md"}
    ]
    result = {
        "status": "PASS" if audit["status"] == "PASS" and not missing and budget <= float(contract["compute"]["gpu_budget_hours"]) else "FAIL",
        "experiment_state": state,
        "historical_anchors_unchanged": audit["checks"]["historical_anchors_unchanged"],
        "missing_required_outputs": missing,
        "gpu_hours_consumed_v6": budget,
        "gpu_budget_hours": float(contract["compute"]["gpu_budget_hours"]),
        "git_head": git_head(v6_paths().project),
        "git_dirty_count": git_dirty_count(v6_paths().project),
        "contract_sha256": sha256_file(v6_paths().contract),
        "gpu_ledger_sha256": sha256_file(ledger_path(v6_paths())) if ledger_path(v6_paths()).exists() else None,
        "output_hashes": {str(path.relative_to(v6_paths().project)): sha256_file(path) for path in sorted(tracked_outputs)},
        "material_passport": "experiment-agent / verification / 2026-08-19 / VERIFIED / trustdive_exact_review_v6",
    }
    write_json(v6_paths().runs / "run_manifest_v6.json", result)
    return _print(result)
