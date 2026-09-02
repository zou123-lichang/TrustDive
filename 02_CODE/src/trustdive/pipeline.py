from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .analysis import (
    analyze_panel,
    analyze_score,
    analyze_trace,
    baseline_gate,
    build_trace300,
    pilot_gate,
)
from .budget import consumed_hours, gpu_budget_entry
from .config import Paths, ensure_project_dirs, load_contract
from .data import audit_dataset, build_manifest, load_manifest, save_manifest, select_probe_clips
from .features import (
    I3DExtractor,
    ZipFrameStore,
    extract_pose_features,
    extract_rgb_features,
    extract_splash_features,
    train_pose_model,
)
from .modeling import train_experiment
from .util import git_dirty_count, git_head, sha256_file, utc_now, write_json


def _require_pass(path: Path, label: str) -> dict:
    if not path.exists():
        raise RuntimeError(f"{label} gate has not run: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"{label} gate did not pass: {payload.get('status')}")
    return payload


def command_audit(paths: Paths) -> dict:
    report = audit_dataset(paths, verify_zip=True)
    write_json(paths.results / "00_AUDIT" / "data_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(2)
    return report


def command_build_manifest(paths: Paths) -> pd.DataFrame:
    frame = build_manifest(paths)
    save_manifest(frame, paths.manifest)
    summary = {
        "rows": len(frame),
        "columns": list(frame.columns),
        "official_split": frame.official_split.value_counts().to_dict(),
        "judge_count": {str(k): int(v) for k, v in frame.judge_count.value_counts().to_dict().items()},
        "event_families": int(frame.event_family.nunique()),
        "path": str(paths.manifest),
    }
    write_json(paths.results / "00_AUDIT" / "manifest_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return frame


def command_probe(paths: Paths, clips: int) -> dict:
    contract = load_contract(paths.contract)
    manifest = load_manifest(paths)
    sample = select_probe_clips(manifest, clips, int(contract["random"]["master_seed"]))
    sample.to_csv(paths.results / "01_PROBE" / "probe_clips.csv", index=False)
    decode_start = time.perf_counter()
    with ZipFrameStore(paths.trimmed_zip) as store:
        first = sample.iloc[0]
        repeat_a = np.stack([np.asarray(x) for x in store.load(first.source, first.instance, 8)])
        repeat_b = np.stack([np.asarray(x) for x in store.load(first.source, first.instance, 8)])
        for row in sample.itertuples(index=False):
            store.load(row.source, row.instance, 3)
    decode_seconds = time.perf_counter() - decode_start
    decode_repeat_exact = bool(np.array_equal(repeat_a, repeat_b))

    with gpu_budget_entry("probe", estimated_hours=0.0, paths=paths):
        pose_result = train_pose_model(paths, max_epochs=12, seed=int(contract["random"]["master_seed"]))
        rgb_result = extract_rgb_features(manifest, paths, set(sample.clip_uid), overwrite=True)
        pose_extract_result = extract_pose_features(manifest, paths, set(sample.clip_uid), overwrite=True)
    extract_seconds = rgb_result["elapsed_seconds"] + pose_extract_result["elapsed_seconds"]
    estimated_feature_hours = extract_seconds * (len(manifest) / len(sample)) / 3600.0
    peak_gb = 0.0
    try:
        import torch

        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    except ImportError:
        pass
    pose_best = pose_result["best"]
    gate = contract["gates"]["probe"]
    checks = {
        "decode_repeat_exact": decode_repeat_exact,
        "pck_at_0_1": pose_best["pck_at_0_1"] >= float(gate["pck_at_0_1"]),
        "key_angle_mae": pose_best["key_angle_mae_deg"] <= float(gate["key_angle_mae_deg"]),
        "peak_vram": peak_gb <= float(contract["compute"]["max_peak_vram_gb"]),
        "estimated_feature_time": estimated_feature_hours
        <= float(contract["compute"]["max_estimated_feature_hours"]),
        "probe_budget": consumed_hours(paths) <= float(contract["compute"]["probe_budget_hours"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "clips": len(sample),
        "decode_seconds": decode_seconds,
        "rgb": rgb_result,
        "pose_extract": pose_extract_result,
        "pose_validation": pose_best,
        "pose_validation_source": "PoseDive human/automatic image annotations; not Trace-300",
        "estimated_full_feature_hours": estimated_feature_hours,
        "peak_vram_gb": peak_gb,
        "gpu_hours_consumed_total": consumed_hours(paths),
    }
    write_json(paths.results / "01_PROBE" / "probe_gate.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(3)
    return result


def command_extract_features(paths: Paths, modalities: list[str]) -> dict:
    _require_pass(paths.results / "01_PROBE" / "probe_gate.json", "Probe")
    manifest = load_manifest(paths)
    result = {}
    if "splash" in modalities:
        frame = extract_splash_features(manifest, paths)
        result["splash"] = {"rows": len(frame), "valid": int(frame.splash_valid.sum())}
    gpu_modalities = [x for x in modalities if x in {"rgb", "pose"}]
    if gpu_modalities:
        with gpu_budget_entry(f"extract-features:{','.join(gpu_modalities)}", paths=paths):
            if "rgb" in gpu_modalities:
                result["rgb"] = extract_rgb_features(manifest, paths)
            if "pose" in gpu_modalities:
                result["pose"] = extract_pose_features(manifest, paths)
    result["gpu_hours_consumed_total"] = consumed_hours(paths)
    write_json(paths.runs / "feature_extraction.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_train_baselines(paths: Paths) -> dict:
    _require_pass(paths.results / "01_PROBE" / "probe_gate.json", "Probe")
    manifest = load_manifest(paths)
    contract = load_contract(paths.contract)
    seeds = list(contract["random"]["model_seeds"])
    with gpu_budget_entry("train-baselines", paths=paths):
        _, global_summary = train_experiment(manifest, paths, "global", seeds, "global")
        _, relative_summary = train_experiment(manifest, paths, "relative", seeds, "relative")
    gate = baseline_gate(paths)
    result = {"global": global_summary, "relative": relative_summary, "gate": gate}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if gate["status"] != "PASS":
        raise SystemExit(4)
    return result


def command_train(paths: Paths, stage: str) -> dict:
    manifest = load_manifest(paths)
    contract = load_contract(paths.contract)
    if stage == "pilot":
        _require_pass(paths.results / "02_SCORE" / "baseline_gate.json", "Baseline")
        seeds = [int(contract["random"]["model_seeds"][0])]
        with gpu_budget_entry("train:pilot", paths=paths):
            _, summary = train_experiment(manifest, paths, "trustdive", seeds, "pilot")
        gate = pilot_gate(paths)
        print(json.dumps(gate, indent=2, ensure_ascii=False))
        if gate["status"] != "PASS":
            raise SystemExit(5)
        return {"summary": summary, "gate": gate}
    if stage == "final":
        _require_pass(paths.results / "02_SCORE" / "pilot_gate.json", "Pilot")
        if not bool(contract["state"]["frozen"]):
            raise RuntimeError("Contract must be frozen before final training")
        with gpu_budget_entry("train:final", paths=paths):
            _, summary = train_experiment(
                manifest, paths, "trustdive", list(contract["random"]["model_seeds"]), "final"
            )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
    raise ValueError(f"Unknown training stage: {stage}")


def command_freeze_contract(paths: Paths) -> dict:
    _require_pass(paths.results / "02_SCORE" / "pilot_gate.json", "Pilot")
    contract = load_contract(paths.contract)
    if contract["state"]["frozen"]:
        return contract["state"]
    contract["state"]["frozen"] = True
    contract["state"]["frozen_at"] = utc_now()
    contract["state"]["contract_sha256"] = None
    canonical = yaml.safe_dump(contract, sort_keys=True, allow_unicode=True).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    contract["state"]["contract_sha256"] = digest
    paths.contract.write_text(yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8")
    result = {"frozen": True, "frozen_at": contract["state"]["frozen_at"], "contract_sha256": digest}
    write_json(paths.results / "02_SCORE" / "contract_freeze.json", result)
    print(json.dumps(result, indent=2))
    return result


def command_render_reports(paths: Paths) -> dict:
    # Formal figures remain locked until every prerequisite exists. This
    # command deliberately creates no "pretty" substitute for missing evidence.
    score = analyze_score(paths)
    trace = analyze_trace(paths)
    panel_path = paths.results / "04_PANEL" / "panel_analysis.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8")) if panel_path.exists() else {"status": "NOT_RUN"}
    prerequisites = {
        "score": score.get("status") == "ANALYZED",
        "trace": trace.get("status") == "ANALYZED",
        "panel": panel.get("status") in {"PASS", "FAIL"},
    }
    result = {
        "status": "READY" if all(prerequisites.values()) else "LOCKED",
        "prerequisites": prerequisites,
        "message": "Formal figure rendering is intentionally deferred until evidence gates are complete.",
    }
    write_json(paths.results / "figures" / "render_status.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def command_verify(paths: Paths) -> dict:
    audit_path = paths.results / "00_AUDIT" / "data_audit.json"
    current_audit = audit_dataset(paths, verify_zip=False)
    initial = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    old_projects_unchanged = None
    if initial:
        old_projects_unchanged = initial["old_project_snapshots"] == current_audit["old_project_snapshots"]
    probe_path = paths.results / "01_PROBE" / "probe_gate.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else None
    downstream_paths = [
        paths.results / "02_SCORE" / "baseline_gate.json",
        paths.results / "02_SCORE" / "pilot_gate.json",
        paths.results / "02_SCORE" / "contract_freeze.json",
        paths.results / "02_SCORE" / "predictions.parquet",
        paths.results / "03_TRACE" / "trace300_manifest.csv",
        paths.results / "04_PANEL" / "panel_analysis.json",
    ]
    downstream_absent_after_stop = bool(
        probe and probe.get("status") == "FAIL" and not any(path.exists() for path in downstream_paths)
    )
    output_hashes = {}
    for path in sorted(
        item
        for item in paths.results.rglob("*")
        if item.is_file() and item.suffix.lower() in {".json", ".csv", ".md"}
    ):
        output_hashes[str(path.relative_to(paths.project))] = sha256_file(path)
    contract_hash = sha256_file(paths.contract)
    ledger = paths.runs / "gpu_budget_ledger.json"
    environment_lock = paths.code / "environment" / "requirements-lock.txt"
    verification_checks = {
        "data_audit_pass": current_audit["status"] == "PASS",
        "old_projects_unchanged": old_projects_unchanged is not False,
        "probe_recorded": probe is not None,
        "stop_respected": downstream_absent_after_stop,
        "contract_not_frozen": not bool(load_contract(paths.contract)["state"]["frozen"]),
    }
    result = {
        "status": "PASS" if all(verification_checks.values()) else "FAIL",
        "experiment_state": "STOPPED_AT_STAGE_A" if probe and probe.get("status") == "FAIL" else "IN_PROGRESS",
        "verification_checks": verification_checks,
        "data_audit": current_audit["status"],
        "old_projects_unchanged": old_projects_unchanged,
        "probe_status": None if probe is None else probe.get("status"),
        "downstream_outputs_absent": downstream_absent_after_stop,
        "git_head": git_head(paths.project),
        "git_dirty_count": git_dirty_count(paths.project),
        "gpu_hours_consumed": consumed_hours(paths),
        "input_hashes": None if initial is None else initial.get("input_hashes"),
        "contract_sha256": contract_hash,
        "gpu_ledger_sha256": sha256_file(ledger) if ledger.exists() else None,
        "environment_lock_sha256": sha256_file(environment_lock) if environment_lock.exists() else None,
        "output_hashes": output_hashes,
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(paths.runs / "run_manifest.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(6)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m trustdive.pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--protocol", choices=("v1", "v4", "v5", "v6", "v7", "v8", "v9"), default="v1")
    fit_judge_simulator = commands.add_parser("fit-judge-simulator")
    fit_judge_simulator.add_argument("--protocol", choices=("v9",), default="v9")
    validate_judge_simulator = commands.add_parser("validate-judge-simulator")
    validate_judge_simulator.add_argument("--protocol", choices=("v9",), default="v9")
    generate_conflicts = commands.add_parser("generate-conflicts")
    generate_conflicts.add_argument("--protocol", choices=("v9",), default="v9")
    generate_conflicts.add_argument("--stage", choices=("development", "final"), required=True)
    judge_phase_features = commands.add_parser("build-judge-phase-features")
    judge_phase_features.add_argument("--protocol", choices=("v9",), default="v9")
    conditional_risk = commands.add_parser("build-conditional-risk")
    conditional_risk.add_argument("--protocol", choices=("v8",), default="v8")
    phase_tokens = commands.add_parser("build-phase-tokens")
    phase_tokens.add_argument("--protocol", choices=("v8",), default="v8")
    dual_evidence = commands.add_parser("build-dual-evidence")
    dual_evidence.add_argument("--protocol", choices=("v8",), default="v8")
    risk_task = commands.add_parser("build-risk-task")
    risk_task.add_argument("--protocol", choices=("v7",), default="v7")
    phase_evidence = commands.add_parser("build-phase-evidence")
    phase_evidence.add_argument("--protocol", choices=("v7",), default="v7")
    analyze_risk = commands.add_parser("analyze-risk")
    analyze_risk.add_argument("--protocol", choices=("v7",), default="v7")
    extract_latents = commands.add_parser("extract-latents")
    extract_latents.add_argument("--protocol", choices=("v6",), default="v6")
    optimize_adapter = commands.add_parser("optimize-adapter")
    optimize_adapter.add_argument("--protocol", choices=("v6",), default="v6")
    build_attributions = commands.add_parser("build-attributions")
    build_attributions.add_argument("--protocol", choices=("v6",), default="v6")
    evaluate_final = commands.add_parser("evaluate-final")
    evaluate_final.add_argument("--protocol", choices=("v6",), default="v6")
    analyze_review = commands.add_parser("analyze-review")
    analyze_review.add_argument("--protocol", choices=("v6", "v8", "v9"), default="v6")
    references = commands.add_parser("build-references")
    references.add_argument("--protocol", choices=("v5",), default="v5")
    optimize = commands.add_parser("optimize-baseline")
    optimize.add_argument("--protocol", choices=("v5",), default="v5")
    manifest = commands.add_parser("build-manifest")
    manifest.add_argument("--protocol", choices=("v1",), default="v1")
    audit_panel = commands.add_parser("audit-panel")
    audit_panel.add_argument("--protocol", choices=("v2",), default="v2")
    panel_targets = commands.add_parser("build-panel-targets")
    panel_targets.add_argument("--protocol", choices=("v2",), default="v2")
    probe = commands.add_parser("probe")
    probe.add_argument("--clips", type=int, default=100)
    probe.add_argument("--protocol", choices=("v1",), default="v1")
    extract = commands.add_parser("extract-features")
    extract.add_argument("--modalities", default="rgb,pose,splash")
    extract.add_argument("--protocol", choices=("v1", "v2"), default="v1")
    baselines = commands.add_parser("train-baselines")
    baselines.add_argument("--protocol", choices=("v1", "v2", "v7", "v8", "v9"), default="v1")
    tune = commands.add_parser("tune")
    tune.add_argument("--protocol", choices=("v2",), default="v2")
    train = commands.add_parser("train")
    train.add_argument("--stage", choices=("pilot", "final"), required=False, default=None)
    train.add_argument("--protocol", choices=("v1", "v2", "v4", "v5", "v7", "v8", "v9"), default="v1")
    freeze = commands.add_parser("freeze-contract")
    freeze.add_argument("--protocol", choices=("v1", "v2", "v4", "v5", "v6", "v8", "v9"), default="v1")
    trace = commands.add_parser("build-trace300")
    trace.add_argument("--protocol", choices=("v1",), default="v1")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--part", choices=("score", "trace", "panel", "all"), required=True)
    analyze.add_argument("--protocol", choices=("v1", "v2", "v4", "v5"), default="v1")
    render = commands.add_parser("render-reports")
    render.add_argument("--protocol", choices=("v1", "v2", "v4", "v5", "v6", "v7", "v8", "v9"), default="v1")
    verify = commands.add_parser("verify")
    verify.add_argument("--protocol", choices=("v1", "v2", "v4", "v5", "v6", "v7", "v8", "v9"), default="v1")
    audit_trace = commands.add_parser("audit-trace")
    audit_trace.add_argument("--protocol", choices=("v3",), default="v3")
    teacher = commands.add_parser("build-teacher-targets")
    teacher.add_argument("--protocol", choices=("v3",), default="v3")
    pilot_trace = commands.add_parser("pilot-trace")
    pilot_trace.add_argument("--protocol", choices=("v3",), default="v3")
    stress = commands.add_parser("stress-test")
    stress.add_argument("--protocol", choices=("v3", "v4", "v5", "v9"), default="v3")
    prepare_teacher = commands.add_parser("prepare-teacher")
    prepare_teacher.add_argument("--protocol", choices=("v4",), default="v4")
    train_teacher = commands.add_parser("train-teacher")
    train_teacher.add_argument("--protocol", choices=("v4",), default="v4")
    export_teacher = commands.add_parser("export-teacher")
    export_teacher.add_argument("--protocol", choices=("v4",), default="v4")
    counterfactuals = commands.add_parser("build-counterfactuals")
    counterfactuals.add_argument("--protocol", choices=("v4", "v5"), default="v4")
    pilot_v4 = commands.add_parser("pilot")
    pilot_v4.add_argument("--protocol", choices=("v4", "v5", "v6", "v8", "v9"), default="v4")
    for command in (train, freeze, analyze, render, verify):
        action = next((item for item in command._actions if item.dest == "protocol"), None)
        if action is not None:
            action.choices = tuple(dict.fromkeys((*action.choices, "v3")))
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    paths = Paths()
    ensure_project_dirs(paths)
    protocol = getattr(args, "protocol", "v1")
    if protocol == "v9":
        from .v9_pipeline import (
            command_analyze_review_v9, command_audit_v9, command_build_features_v9,
            command_fit_simulator_v9, command_freeze_v9, command_generate_conflicts_v9,
            command_pilot_v9, command_render_v9, command_stress_v9,
            command_train_baselines_v9, command_train_final_v9,
            command_validate_simulator_v9, command_verify_v9,
        )
        if args.command == "audit": command_audit_v9()
        elif args.command == "fit-judge-simulator": command_fit_simulator_v9()
        elif args.command == "validate-judge-simulator": command_validate_simulator_v9()
        elif args.command == "generate-conflicts": command_generate_conflicts_v9(args.stage)
        elif args.command == "build-judge-phase-features": command_build_features_v9()
        elif args.command == "train-baselines": command_train_baselines_v9()
        elif args.command == "pilot": command_pilot_v9()
        elif args.command == "freeze-contract": command_freeze_v9()
        elif args.command == "train" and args.stage == "final": command_train_final_v9()
        elif args.command == "stress-test": command_stress_v9()
        elif args.command == "analyze-review": command_analyze_review_v9()
        elif args.command == "render-reports": command_render_v9()
        elif args.command == "verify": command_verify_v9()
        else: raise ValueError(f"Command {args.command!r} is not available for protocol v9")
        return
    if protocol == "v8":
        from .v8_pipeline import (
            command_analyze_review_v8,
            command_audit_v8,
            command_build_conditional_risk_v8,
            command_build_dual_evidence_v8,
            command_build_phase_tokens_v8,
            command_freeze_v8,
            command_pilot_v8,
            command_train_baselines_v8,
            command_train_final_v8,
            command_verify_v8,
        )

        if args.command == "audit":
            command_audit_v8()
        elif args.command == "build-conditional-risk":
            command_build_conditional_risk_v8()
        elif args.command == "build-phase-tokens":
            command_build_phase_tokens_v8()
        elif args.command == "train-baselines":
            command_train_baselines_v8()
        elif args.command == "pilot":
            command_pilot_v8()
        elif args.command == "freeze-contract":
            command_freeze_v8()
        elif args.command == "train" and args.stage == "final":
            command_train_final_v8()
        elif args.command == "build-dual-evidence":
            command_build_dual_evidence_v8()
        elif args.command == "analyze-review":
            command_analyze_review_v8()
        elif args.command == "render-reports":
            from .v8_figures import render_reports_v8

            print(json.dumps(render_reports_v8(), indent=2, ensure_ascii=False))
        elif args.command == "verify":
            command_verify_v8()
        else:
            raise ValueError(f"Command {args.command!r} is not available for protocol v8")
        return
    if protocol == "v7":
        from .v7_figures import render_reports_v7
        from .v7_pipeline import (
            command_analyze_risk_v7,
            command_audit_v7,
            command_build_phase_evidence_v7,
            command_build_risk_task_v7,
            command_train_baselines_v7,
            command_train_v7,
            command_verify_v7,
        )

        if args.command == "audit":
            command_audit_v7()
        elif args.command == "build-risk-task":
            command_build_risk_task_v7()
        elif args.command == "train-baselines":
            command_train_baselines_v7()
        elif args.command == "train":
            command_train_v7()
        elif args.command == "build-phase-evidence":
            command_build_phase_evidence_v7()
        elif args.command == "analyze-risk":
            command_analyze_risk_v7()
        elif args.command == "render-reports":
            print(json.dumps(render_reports_v7(), indent=2, ensure_ascii=False))
        elif args.command == "verify":
            command_verify_v7()
        else:
            raise ValueError(f"Command {args.command!r} is not available for protocol v7")
        return
    if args.command == "train" and args.stage is None:
        raise SystemExit("--stage is required for train commands outside protocol v7")
    if protocol == "v6":
        from .v6_figures import render_reports_v6
        from .v6_pipeline import (
            command_analyze_review_v6,
            command_audit_v6,
            command_build_attributions_v6,
            command_evaluate_final_v6,
            command_extract_latents_v6,
            command_freeze_v6,
            command_optimize_adapter_v6,
            command_pilot_v6,
            command_verify_v6,
        )

        if args.command == "audit":
            command_audit_v6()
        elif args.command == "extract-latents":
            command_extract_latents_v6()
        elif args.command == "optimize-adapter":
            command_optimize_adapter_v6()
        elif args.command == "build-attributions":
            command_build_attributions_v6()
        elif args.command == "pilot":
            command_pilot_v6()
        elif args.command == "freeze-contract":
            command_freeze_v6()
        elif args.command == "evaluate-final":
            command_evaluate_final_v6()
        elif args.command == "analyze-review":
            command_analyze_review_v6()
        elif args.command == "render-reports":
            print(json.dumps(render_reports_v6(), indent=2, ensure_ascii=False))
        elif args.command == "verify":
            command_verify_v6()
        else:
            raise ValueError(f"Command {args.command!r} is not available for protocol v6")
        return
    if protocol == "v5":
        from .v5_figures import render_reports_v5
        from .v5_pipeline import (
            command_analyze_v5,
            command_audit_v5,
            command_build_counterfactuals_v5,
            command_build_references_v5,
            command_freeze_v5,
            command_optimize_baseline_v5,
            command_pilot_v5,
            command_stress_v5,
            command_train_final_v5,
            command_verify_v5,
        )

        if args.command == "audit":
            command_audit_v5()
        elif args.command == "build-references":
            command_build_references_v5()
        elif args.command == "optimize-baseline":
            command_optimize_baseline_v5()
        elif args.command == "build-counterfactuals":
            command_build_counterfactuals_v5()
        elif args.command == "pilot":
            command_pilot_v5()
        elif args.command == "freeze-contract":
            command_freeze_v5()
        elif args.command == "train" and args.stage == "final":
            command_train_final_v5()
        elif args.command == "stress-test":
            command_stress_v5()
        elif args.command == "analyze":
            command_analyze_v5(args.part)
        elif args.command == "render-reports":
            print(json.dumps(render_reports_v5(), indent=2, ensure_ascii=False))
        elif args.command == "verify":
            command_verify_v5()
        else:
            raise ValueError(f"Command {args.command!r} is not available for protocol v5")
        return
    if protocol == "v4":
        from .v4_figures import render_reports_v4
        from .v4_pipeline import (
            command_analyze_v4,
            command_audit_v4,
            command_build_counterfactuals_v4,
            command_export_teacher_v4,
            command_freeze_contract_v4,
            command_pilot_v4,
            command_prepare_teacher_v4,
            command_stress_test_v4,
            command_train_final_v4,
            command_train_teacher_v4,
            command_verify_v4,
        )

        if args.command == "audit":
            command_audit_v4()
        elif args.command == "prepare-teacher":
            command_prepare_teacher_v4()
        elif args.command == "train-teacher":
            command_train_teacher_v4()
        elif args.command == "export-teacher":
            command_export_teacher_v4()
        elif args.command == "build-counterfactuals":
            command_build_counterfactuals_v4()
        elif args.command == "pilot":
            command_pilot_v4()
        elif args.command == "freeze-contract":
            command_freeze_contract_v4()
        elif args.command == "train" and args.stage == "final":
            command_train_final_v4()
        elif args.command == "stress-test":
            command_stress_test_v4()
        elif args.command == "analyze":
            command_analyze_v4(args.part)
        elif args.command == "render-reports":
            print(json.dumps(render_reports_v4(), indent=2, ensure_ascii=False))
        elif args.command == "verify":
            command_verify_v4()
        else:
            raise ValueError(f"Command {args.command!r} is not available for protocol v4")
        return
    if protocol == "v3":
        from .v3_figures import render_reports_v3
        from .v3_pipeline import (
            command_analyze_v3,
            command_audit_trace_v3,
            command_build_teacher_targets_v3,
            command_freeze_contract_v3,
            command_pilot_trace_v3,
            command_stress_test_v3,
            command_train_final_v3,
            command_verify_v3,
        )

        if args.command == "audit-trace":
            command_audit_trace_v3()
        elif args.command == "build-teacher-targets":
            command_build_teacher_targets_v3()
        elif args.command == "pilot-trace":
            command_pilot_trace_v3()
        elif args.command == "freeze-contract":
            command_freeze_contract_v3()
        elif args.command == "train" and args.stage == "final":
            command_train_final_v3()
        elif args.command == "stress-test":
            command_stress_test_v3()
        elif args.command == "analyze":
            command_analyze_v3(args.part)
        elif args.command == "render-reports":
            print(json.dumps(render_reports_v3(), indent=2, ensure_ascii=False))
        elif args.command == "verify":
            command_verify_v3()
        else:
            raise ValueError(f"Command {args.command!r} is not available for protocol v3")
        return
    if protocol == "v2":
        from .v2_pipeline import (
            command_analyze_v2,
            command_audit_panel_v2,
            command_build_panel_targets_v2,
            command_extract_features_v2,
            command_freeze_contract_v2,
            command_train_baselines_v2,
            command_train_final_v2,
            command_tune_v2,
            command_verify_v2,
        )

        if args.command == "audit-panel":
            command_audit_panel_v2()
        elif args.command == "build-panel-targets":
            command_build_panel_targets_v2()
        elif args.command == "extract-features":
            modalities = [x.strip() for x in args.modalities.split(",") if x.strip()]
            command_extract_features_v2(modalities)
        elif args.command == "train-baselines":
            command_train_baselines_v2()
        elif args.command == "tune":
            command_tune_v2()
        elif args.command == "freeze-contract":
            command_freeze_contract_v2()
        elif args.command == "train" and args.stage == "final":
            command_train_final_v2()
        elif args.command == "analyze":
            command_analyze_v2(args.part)
        elif args.command == "verify":
            command_verify_v2()
        elif args.command == "render-reports":
            from .v2_figures import render_reports_v2

            render_reports_v2()
        else:
            raise ValueError(f"Command {args.command!r} is not available for protocol v2")
        return
    if args.command == "audit":
        command_audit(paths)
    elif args.command == "build-manifest":
        command_build_manifest(paths)
    elif args.command == "probe":
        command_probe(paths, args.clips)
    elif args.command == "extract-features":
        modalities = [x.strip() for x in args.modalities.split(",") if x.strip()]
        unknown = set(modalities) - {"rgb", "pose", "splash"}
        if unknown:
            raise ValueError(f"Unknown modalities: {sorted(unknown)}")
        command_extract_features(paths, modalities)
    elif args.command == "train-baselines":
        command_train_baselines(paths)
    elif args.command == "train":
        command_train(paths, args.stage)
    elif args.command == "freeze-contract":
        command_freeze_contract(paths)
    elif args.command == "build-trace300":
        frame = build_trace300(load_manifest(paths), paths)
        print(f"Trace manifest: {len(frame)} rows")
    elif args.command == "analyze":
        if args.part == "score":
            result = analyze_score(paths)
        elif args.part == "trace":
            result = analyze_trace(paths)
        elif args.part == "panel":
            result = analyze_panel(load_manifest(paths), paths)
        else:
            raise ValueError("The 'all' analysis part is available only for protocol v2")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "render-reports":
        command_render_reports(paths)
    elif args.command == "verify":
        command_verify(paths)


if __name__ == "__main__":
    main()
