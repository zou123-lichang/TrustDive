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


V6_CONTRACT_PATH = PROJECT_ROOT / "01_PROTOCOL" / "analysis_contract_v6_exact_review.yaml"
V6_RESULTS_ROOT = RESULTS_ROOT / "V6_EXACT_REVIEW"
V6_RUN_ROOT = RUNS_ROOT / "v6_exact_review"
V6_CACHE_ROOT = PROJECT_ROOT / ".cache" / "v6_exact_review"


def v6_paths() -> Paths:
    return Paths(contract=V6_CONTRACT_PATH)


def ensure_v6_dirs() -> None:
    for path in (
        V6_RESULTS_ROOT / "00_AUDIT",
        V6_RESULTS_ROOT / "01_LATENTS",
        V6_RESULTS_ROOT / "02_ADAPTER",
        V6_RESULTS_ROOT / "03_ATTRIBUTION",
        V6_RESULTS_ROOT / "04_PILOT",
        V6_RESULTS_ROOT / "05_FINAL",
        V6_RESULTS_ROOT / "06_REVIEW",
        V6_RESULTS_ROOT / "figures_v6",
        V6_RUN_ROOT / "checkpoints",
        V6_CACHE_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_v6_contract() -> dict:
    return load_contract(V6_CONTRACT_PATH)


def load_v6_frame() -> pd.DataFrame:
    return load_panel_targets().copy()


def _git_unchanged(commit: str, path: Path) -> bool:
    relative = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return path.exists() and subprocess.run(
        ["git", "diff", "--quiet", commit, "--", relative],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0


def audit_v6() -> dict:
    ensure_v6_dirs()
    contract = load_v6_contract()
    frame = load_v6_frame()
    anchor = str(contract["read_only_anchors"]["repository_commit"])
    head = git_head(PROJECT_ROOT) or ""
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", anchor, head],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    paths = {
        key: PROJECT_ROOT / value
        for key, value in contract["read_only_anchors"].items()
        if key != "repository_commit"
    }
    unchanged = {key: _git_unchanged(anchor, path) for key, path in paths.items()}
    counts = frame.analysis_role.value_counts().to_dict()
    checkpoint = (
        PROJECT_ROOT / "runs" / "v4_counterfactual" / "teacher" /
        "rica2_final_v4_v4_final" / "last.pth.tar"
    )
    checks = {
        "repository_anchor_is_ancestor": ancestor,
        "historical_anchors_unchanged": all(unchanged.values()),
        "samples": len(frame) == int(contract["data"]["samples"]),
        "official_train": int((frame.official_split == "train").sum()) == int(contract["data"]["official_train"]),
        "official_test": int((frame.official_split == "test").sum()) == int(contract["data"]["official_test"]),
        "fit": int(counts.get("fit", 0)) == int(contract["data"]["fit"]),
        "validation": int(counts.get("validation", 0)) == int(contract["data"]["validation"]),
        "calibration": int(counts.get("calibration", 0)) == int(contract["data"]["calibration"]),
        "seven_judge_test": int(((frame.official_split == "test") & frame.disagreement_primary_eligible).sum()) == int(contract["data"]["official_test_seven_judge"]),
        "event_families": int(frame.event_family.nunique()) == int(contract["data"]["source_event_families"]),
        "teacher_checkpoint": checkpoint.exists() and sha256_file(checkpoint) == str(contract["teacher"]["checkpoint_sha256"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_git_head": head,
        "repository_anchor": anchor,
        "anchor_match": unchanged,
        "anchor_hashes": {key: sha256_file(path) for key, path in paths.items() if path.exists()},
        "teacher_checkpoint": str(checkpoint),
        "material_passport": "experiment-agent / audit / 2026-08-19 / VERIFIED_DATA / trustdive_exact_review_v6",
    }
    write_json(V6_RESULTS_ROOT / "00_AUDIT" / "audit_v6.json", result)
    return result


def require_v6_audit() -> dict:
    path = V6_RESULTS_ROOT / "00_AUDIT" / "audit_v6.json"
    if not path.exists():
        raise RuntimeError("Run audit --protocol v6 first")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("v6 audit did not pass")
    return result


def require_v6_frozen() -> dict:
    contract = load_v6_contract()
    if not contract["state"]["frozen"] or not contract["state"]["official_test_unlocked"]:
        raise RuntimeError("Freeze the v6 contract before official-test evaluation")
    return contract


def freeze_v6_contract() -> dict:
    selected = V6_RESULTS_ROOT / "02_ADAPTER" / "selected_adapter_v6.json"
    attribution = V6_RESULTS_ROOT / "03_ATTRIBUTION" / "pilot_attribution_summary_v6.json"
    pilot = V6_RESULTS_ROOT / "04_PILOT" / "pilot_gate_v6.json"
    for path in (selected, attribution, pilot):
        if not path.exists():
            raise RuntimeError(f"Required pre-freeze artifact missing: {path.name}")
    if json.loads(pilot.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("v6 pilot gate did not pass; official test remains locked")
    contract = load_v6_contract()
    if contract["state"]["frozen"]:
        return contract["state"]
    contract["state"].update({
        "frozen": True,
        "frozen_at": utc_now(),
        "official_test_unlocked": True,
        "selected_adapter_sha256": sha256_file(selected),
        "pilot_attribution_sha256": sha256_file(attribution),
        "contract_sha256": None,
    })
    canonical = yaml.safe_dump(contract, sort_keys=True, allow_unicode=True).encode("utf-8")
    contract["state"]["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    V6_CONTRACT_PATH.write_text(yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8")
    result = dict(contract["state"])
    write_json(V6_RESULTS_ROOT / "04_PILOT" / "contract_freeze_v6.json", result)
    return result
