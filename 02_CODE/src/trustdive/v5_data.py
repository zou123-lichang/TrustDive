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


V5_CONTRACT_PATH = PROJECT_ROOT / "01_PROTOCOL" / "analysis_contract_v5_cfpd_plus.yaml"
V5_RESULTS_ROOT = RESULTS_ROOT / "V5_APPLIED_CFPD"
V5_RUN_ROOT = RUNS_ROOT / "v5_applied_cfpd"
V5_CACHE_ROOT = PROJECT_ROOT / ".cache" / "v5_applied_cfpd"


def v5_paths() -> Paths:
    return Paths(contract=V5_CONTRACT_PATH)


def ensure_v5_dirs() -> None:
    for path in (
        V5_RESULTS_ROOT / "00_AUDIT",
        V5_RESULTS_ROOT / "01_REFERENCES",
        V5_RESULTS_ROOT / "02_BASELINE",
        V5_RESULTS_ROOT / "03_COUNTERFACTUAL",
        V5_RESULTS_ROOT / "04_PILOT",
        V5_RESULTS_ROOT / "05_FINAL",
        V5_RESULTS_ROOT / "06_STRESS",
        V5_RESULTS_ROOT / "07_ANALYSIS",
        V5_RESULTS_ROOT / "figures_v5",
        V5_RUN_ROOT / "checkpoints",
        V5_CACHE_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_v5_contract() -> dict:
    return load_contract(V5_CONTRACT_PATH)


def load_v5_frame() -> pd.DataFrame:
    return load_panel_targets().copy()


def _anchor_paths(contract: dict) -> dict[str, Path]:
    return {
        key: PROJECT_ROOT / relative
        for key, relative in contract["read_only_anchors"].items()
        if key != "repository_commit"
    }


def _git_blob_hash(commit: str, relative: str) -> str | None:
    process = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if process.returncode != 0:
        return None
    return hashlib.sha256(process.stdout).hexdigest()


def _git_content_unchanged(commit: str, path: Path) -> bool:
    relative = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return (
        subprocess.run(
            ["git", "diff", "--quiet", commit, "--", relative],
            cwd=PROJECT_ROOT,
            check=False,
        ).returncode
        == 0
    )


def audit_v5() -> dict:
    ensure_v5_dirs()
    contract = load_v5_contract()
    frame = load_v5_frame()
    anchor_commit = str(contract["read_only_anchors"]["repository_commit"])
    current_head = git_head(PROJECT_ROOT) or ""
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", anchor_commit, current_head],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    anchors = _anchor_paths(contract)
    anchor_hashes = {key: sha256_file(path) for key, path in anchors.items() if path.exists()}
    expected_hashes = {
        key: _git_blob_hash(anchor_commit, str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
        for key, path in anchors.items()
    }
    # Git's own comparison honors working-tree line-ending normalization. Raw
    # SHA-256 values are still recorded, but are not used as a false CRLF gate.
    anchor_match = {
        key: path.exists() and _git_content_unchanged(anchor_commit, path)
        for key, path in anchors.items()
    }
    checkpoint = (
        PROJECT_ROOT
        / "runs"
        / "v4_counterfactual"
        / "teacher"
        / "rica2_final_v4_v4_final"
        / "last.pth.tar"
    )
    checks = {
        "repository_anchor_is_ancestor": ancestor,
        "historical_anchors_unchanged": all(anchor_match.values()),
        "samples": len(frame) == int(contract["data"]["samples"]),
        "official_train": int((frame.official_split == "train").sum())
        == int(contract["data"]["official_train"]),
        "official_test": int((frame.official_split == "test").sum())
        == int(contract["data"]["official_test"]),
        "seven_judge_test": int(
            ((frame.official_split == "test") & frame.disagreement_primary_eligible).sum()
        )
        == int(contract["data"]["official_test_seven_judge"]),
        "event_families": int(frame.event_family.nunique())
        == int(contract["data"]["source_event_families"]),
        "teacher_checkpoint": checkpoint.exists()
        and sha256_file(checkpoint) == str(contract["teacher"]["checkpoint_sha256"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_git_head": current_head,
        "repository_anchor": anchor_commit,
        "anchor_hashes": anchor_hashes,
        "expected_anchor_hashes": expected_hashes,
        "anchor_match": anchor_match,
        "teacher_checkpoint": str(checkpoint),
        "material_passport": "experiment-agent / audit / 2026-08-18 / VERIFIED_DATA / trustdive_cfpd_plus_v5",
    }
    write_json(V5_RESULTS_ROOT / "00_AUDIT" / "audit_v5.json", result)
    return result


def require_v5_audit() -> dict:
    path = V5_RESULTS_ROOT / "00_AUDIT" / "audit_v5.json"
    if not path.exists():
        raise RuntimeError("Run audit --protocol v5 first")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("v5 audit did not pass")
    return result


def require_v5_frozen() -> dict:
    contract = load_v5_contract()
    if not contract["state"]["frozen"] or not contract["state"]["official_test_unlocked"]:
        raise RuntimeError("Freeze the v5 contract before official-test prediction")
    return contract


def freeze_v5_contract() -> dict:
    baseline = V5_RESULTS_ROOT / "02_BASELINE" / "selected_baseline_v5.json"
    counterfactual = V5_RESULTS_ROOT / "03_COUNTERFACTUAL" / "counterfactual_summary_v5.json"
    selected = V5_RESULTS_ROOT / "04_PILOT" / "selected_cfpd_v5.json"
    for path in (baseline, counterfactual, selected):
        if not path.exists():
            raise RuntimeError(f"Required pre-freeze artifact missing: {path.name}")
    selected_result = json.loads(selected.read_text(encoding="utf-8"))
    if selected_result.get("status") != "PASS":
        raise RuntimeError(
            "v5 pilot trace gate did not pass; official-test contract cannot be frozen"
        )
    contract = load_v5_contract()
    if contract["state"]["frozen"]:
        return contract["state"]
    contract["state"].update(
        {
            "frozen": True,
            "frozen_at": utc_now(),
            "official_test_unlocked": True,
            "selected_baseline_sha256": sha256_file(baseline),
            "selected_cfpd_sha256": sha256_file(selected),
            "contract_sha256": None,
        }
    )
    canonical = yaml.safe_dump(contract, sort_keys=True, allow_unicode=True).encode("utf-8")
    contract["state"]["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    V5_CONTRACT_PATH.write_text(
        yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    result = dict(contract["state"])
    write_json(V5_RESULTS_ROOT / "04_PILOT" / "contract_freeze_v5.json", result)
    return result
