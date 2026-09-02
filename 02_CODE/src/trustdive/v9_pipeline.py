from __future__ import annotations

import json
import platform
import sys

from .budget import consumed_hours, gpu_budget_entry, ledger_path
from .util import git_dirty_count, git_head, sha256_file, write_json
from .v9_analysis import analyze_review_v9, final_decision_v9, stress_test_v9
from .v9_data import V9_RESULTS_ROOT, audit_v9, ensure_v9_dirs, fit_judge_simulator_v9, generate_conflicts_v9, load_v9_contract, v9_paths, validate_judge_simulator_v9
from .v9_features import build_judge_phase_features_v9
from .v9_modeling import freeze_contract_v9, pilot_v9, train_baselines_v9, train_final_v9


def _print(x): print(json.dumps(x,indent=2,ensure_ascii=False)); return x
def command_audit_v9():
    x=audit_v9(); _print(x)
    if x["status"]!="PASS": raise SystemExit(2)
    return x
def command_fit_simulator_v9(): return _print(fit_judge_simulator_v9())
def command_validate_simulator_v9():
    x=validate_judge_simulator_v9(); _print(x)
    if x["status"]!="PASS": raise SystemExit(3)
    return x
def command_generate_conflicts_v9(stage): return _print(generate_conflicts_v9(stage))
def command_build_features_v9():
    stage="final" if (V9_RESULTS_ROOT/"04_PILOT"/"contract_freeze_v9.json").exists() and (V9_RESULTS_ROOT/"02_CONFLICTS"/"synthetic_judge_pairs_final_v9.parquet").exists() else "development"
    return _print(build_judge_phase_features_v9(stage))
def command_train_baselines_v9(): return _print(train_baselines_v9())
def command_pilot_v9():
    with gpu_budget_entry("v9:pilot",float(load_v9_contract()["compute"]["pilot_estimated_hours"]),v9_paths(),force_gpu=True): x=pilot_v9()
    _print(x)
    if x["status"]!="PASS": raise SystemExit(4)
    return x
def command_freeze_v9(): return _print(freeze_contract_v9())
def command_train_final_v9():
    with gpu_budget_entry("v9:final",float(load_v9_contract()["compute"]["final_estimated_hours"]),v9_paths(),force_gpu=True): x=train_final_v9()
    return _print(x)
def command_stress_v9():
    with gpu_budget_entry("v9:stress",float(load_v9_contract()["compute"]["stress_estimated_hours"]),v9_paths(),force_gpu=True): x=stress_test_v9()
    return _print(x)
def command_analyze_review_v9():
    review=analyze_review_v9(); decision=final_decision_v9(); return _print({"review":review,"decision":decision})


def command_render_v9():
    decision_path=V9_RESULTS_ROOT/"07_REVIEW"/"analysis_summary_v9.json"
    decision=json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else None
    if decision is None or decision.get("status")=="STOP":
        pilot_path=V9_RESULTS_ROOT/"04_PILOT"/"pilot_gate_v9.json"; sim_path=V9_RESULTS_ROOT/"01_SIMULATOR"/"simulator_validation_v9.json"
        status={"status":"STOPPED_BY_EVIDENCE_GATE","reason":"formal figures locked because simulator or pilot did not pass",
                "simulator":json.loads(sim_path.read_text(encoding="utf-8")).get("status") if sim_path.exists() else None,
                "pilot":json.loads(pilot_path.read_text(encoding="utf-8")).get("status") if pilot_path.exists() else None,
                "publication_decision":decision.get("publication_decision") if decision else None}
        write_json(V9_RESULTS_ROOT/"figures_v9"/"render_status_v9.json",status); return _print(status)
    from .v9_figures import render_reports_v9
    return _print(render_reports_v9())


def command_verify_v9():
    ensure_v9_dirs(); audit=audit_v9(); contract=load_v9_contract(); sim_path=V9_RESULTS_ROOT/"01_SIMULATOR"/"simulator_validation_v9.json"
    sim=json.loads(sim_path.read_text(encoding="utf-8")) if sim_path.exists() else {}; pilot_path=V9_RESULTS_ROOT/"04_PILOT"/"pilot_gate_v9.json"; pilot=json.loads(pilot_path.read_text(encoding="utf-8")) if pilot_path.exists() else {}
    early=[V9_RESULTS_ROOT/"00_AUDIT"/"audit_v9.json",V9_RESULTS_ROOT/"01_SIMULATOR"/"judge_simulator_v9.json",sim_path]
    if sim.get("status")=="PASS": early += [V9_RESULTS_ROOT/"02_CONFLICTS"/"synthetic_judge_pairs_development_v9.parquet",V9_RESULTS_ROOT/"03_FEATURES"/"judge_phase_features_development_v9.npz",V9_RESULTS_ROOT/"04_PILOT"/"pilot_gate_v9.json"]
    final=[V9_RESULTS_ROOT/"04_PILOT"/"contract_freeze_v9.json",V9_RESULTS_ROOT/"05_FINAL"/"predictions_v9.parquet",V9_RESULTS_ROOT/"06_STRESS"/"stress_results_v9.parquet",V9_RESULTS_ROOT/"07_REVIEW"/"analysis_summary_v9.json"]
    required=early+(final if pilot.get("status")=="PASS" else [])
    fig=V9_RESULTS_ROOT/"figures_v9"/"render_status_v9.json"; required.append(fig)
    missing=[str(p) for p in required if not p.exists()]; figure_status=json.loads(fig.read_text(encoding="utf-8")).get("status") if fig.exists() else None
    checks={"audit_pass":audit["status"]=="PASS","history_protected":audit["checks"]["protected_v1_v8_unchanged"],"required_outputs":not missing,
            "terminal_state":sim.get("status") in {"PASS","STOP"} and (sim.get("status")=="STOP" or pilot.get("status") in {"PASS","STOP"}),
            "figure_closed":figure_status in {"PASS","STOPPED_BY_EVIDENCE_GATE"},"gpu_budget":consumed_hours(v9_paths())<=float(contract["compute"]["gpu_budget_hours"])}
    tracked=[p for p in V9_RESULTS_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".json",".csv",".md",".parquet",".npz"}]
    ledger=ledger_path(v9_paths()); result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"missing":missing,
        "simulator_status":sim.get("status"),"pilot_status":pilot.get("status"),"gpu_hours":consumed_hours(v9_paths()),"git_head":git_head(v9_paths().project),"git_dirty_count":git_dirty_count(v9_paths().project),
        "contract_sha256":sha256_file(v9_paths().contract),"gpu_ledger_sha256":sha256_file(ledger) if ledger.exists() else None,
        "output_hashes":{str(p.relative_to(v9_paths().project)):sha256_file(p) for p in sorted(tracked)},"python":sys.version,"platform":platform.platform(),
        "material_passport":"experiment-agent / verification / 2026-08-21 / VERIFIED / trustdive_judgesim_v9"}
    write_json(v9_paths().runs/"run_manifest_v9.json",result); _print(result)
    if result["status"]!="PASS": raise SystemExit(6)
    return result
