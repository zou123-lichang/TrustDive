from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from .config import PROJECT_ROOT, RESULTS_ROOT, RUNS_ROOT, Paths, load_contract
from .util import git_head, sha256_file, utc_now, write_json
from .v2_data import load_panel_targets


V4_CONTRACT_PATH = PROJECT_ROOT / "01_PROTOCOL" / "analysis_contract_v4_counterfactual.yaml"
V4_RESULTS_ROOT = RESULTS_ROOT / "V4_COUNTERFACTUAL"
V4_RUN_ROOT = RUNS_ROOT / "v4_counterfactual"
V4_CACHE_ROOT = PROJECT_ROOT / ".cache" / "v4_counterfactual"
RICA2_ROOT = PROJECT_ROOT / ".cache" / "external" / "rica2_aqa"
RICA2_DATA_ROOT = PROJECT_ROOT / ".cache" / "rica2_data" / "finediving_v4"
RICA2_FEATURE_ROOT = RICA2_DATA_ROOT / "I3D_features_v4_txc"


def v4_paths() -> Paths:
    return Paths(contract=V4_CONTRACT_PATH)


def ensure_v4_dirs() -> None:
    for path in (
        V4_RESULTS_ROOT / "00_AUDIT",
        V4_RESULTS_ROOT / "01_TEACHER",
        V4_RESULTS_ROOT / "02_COUNTERFACTUAL",
        V4_RESULTS_ROOT / "03_PILOT",
        V4_RESULTS_ROOT / "04_FINAL",
        V4_RESULTS_ROOT / "05_STRESS",
        V4_RESULTS_ROOT / "06_ANALYSIS",
        V4_RESULTS_ROOT / "figures_v4",
        V4_RUN_ROOT / "checkpoints",
        V4_RUN_ROOT / "teacher",
        V4_CACHE_ROOT / "configs",
        RICA2_FEATURE_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_v4_contract() -> dict:
    return load_contract(V4_CONTRACT_PATH)


def load_v4_frame() -> pd.DataFrame:
    return load_panel_targets().copy()


def _anchor_paths(contract: dict) -> dict[str, Path]:
    return {
        key: PROJECT_ROOT / relative
        for key, relative in contract["read_only_anchors"].items()
        if key != "repository_commit"
    }


def audit_v4() -> dict:
    ensure_v4_dirs()
    contract = load_v4_contract()
    frame = load_v4_frame()
    anchors = _anchor_paths(contract)
    anchor_hashes = {key: sha256_file(path) for key, path in anchors.items() if path.exists()}
    weights = RICA2_ROOT / "pre_trained" / "model_rgb.pth"
    trimmed_zip = v4_paths().trimmed_zip
    external_head = git_head(RICA2_ROOT)
    expected_head = str(contract["external_teacher"]["commit"])
    current_head = git_head(PROJECT_ROOT) or ""
    expected_repository = str(contract["read_only_anchors"]["repository_commit"])
    repository_anchor_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_repository, current_head],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    checks = {
        "repository_anchor": repository_anchor_is_ancestor,
        "anchors_exist": all(path.exists() for path in anchors.values()),
        "samples": len(frame) == int(contract["data"]["samples"]),
        "official_train": int((frame.official_split == "train").sum()) == int(contract["data"]["official_train"]),
        "official_test": int((frame.official_split == "test").sum()) == int(contract["data"]["official_test"]),
        "seven_judge_test": int(((frame.official_split == "test") & frame.disagreement_primary_eligible).sum()) == int(contract["data"]["official_test_seven_judge"]),
        "event_families": int(frame.event_family.nunique()) == int(contract["data"]["source_event_families"]),
        "trimmed_zip": trimmed_zip.exists() and trimmed_zip.stat().st_size > 4_000_000_000,
        "rica2_commit": external_head == expected_head,
        "rica2_weights": weights.exists() and sha256_file(weights) == str(contract["external_teacher"]["weights_sha256"]),
        "frames_view": (RICA2_DATA_ROOT / "FINADiving_MTL_256s").exists(),
        "annotations_view": all((RICA2_DATA_ROOT / "Annotations" / name).exists() for name in (
            "FineDiving_fine-grained_annotation.pkl", "FineDiving_coarse_annotation.pkl",
            "train_split.pkl", "test_split.pkl",
        )),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_git_head": current_head,
        "repository_anchor": expected_repository,
        "anchor_paths": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in anchors.items()},
        "anchor_hashes": anchor_hashes,
        "external_teacher_commit": external_head,
        "external_weights_sha256": sha256_file(weights) if weights.exists() else None,
        "material_passport": "experiment-agent / audit / 2026-08-18 / VERIFIED_DATA / trustdive_cfpd_v4",
    }
    write_json(V4_RESULTS_ROOT / "00_AUDIT" / "audit_v4.json", result)
    return result


def require_v4_audit() -> dict:
    path = V4_RESULTS_ROOT / "00_AUDIT" / "audit_v4.json"
    if not path.exists():
        raise RuntimeError("Run audit --protocol v4 first")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("v4 audit did not pass")
    return result


def require_v4_frozen() -> dict:
    contract = load_v4_contract()
    if not contract["state"]["frozen"]:
        raise RuntimeError("Freeze the v4 contract before official student-test outputs")
    return contract


def freeze_v4_contract() -> dict:
    teacher_gate = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_gate_v4.json"
    counterfactual_gate = V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "counterfactual_summary_v4.json"
    selection = V4_RESULTS_ROOT / "03_PILOT" / "selected_config_v4.json"
    for path in (teacher_gate, counterfactual_gate, selection):
        if not path.exists():
            raise RuntimeError(f"Required pre-freeze artifact missing: {path.name}")
    teacher = json.loads(teacher_gate.read_text(encoding="utf-8"))
    counterfactual = json.loads(counterfactual_gate.read_text(encoding="utf-8"))
    selected = json.loads(selection.read_text(encoding="utf-8"))
    if teacher.get("status") != "PASS" or counterfactual.get("status") != "PASS" or selected.get("status") != "PASS":
        raise RuntimeError("Teacher, counterfactual, and pilot gates must all pass before freezing v4")
    contract = load_v4_contract()
    if contract["state"]["frozen"]:
        return contract["state"]
    contract["state"].update(
        {
            "frozen": True,
            "frozen_at": utc_now(),
            "official_student_test_unlocked": True,
            "selected_config_sha256": sha256_file(selection),
            "contract_sha256": None,
        }
    )
    canonical = yaml.safe_dump(contract, sort_keys=True, allow_unicode=True).encode("utf-8")
    contract["state"]["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    V4_CONTRACT_PATH.write_text(yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(V4_RESULTS_ROOT / "03_PILOT" / "contract_freeze_v4.json", contract["state"])
    return contract["state"]
