from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from .config import PROJECT_ROOT, RESULTS_ROOT, RUNS_ROOT, Paths, load_contract
from .util import git_head, sha256_file, utc_now, write_json
from .v2_data import load_panel_targets


V3_CONTRACT_PATH = PROJECT_ROOT / "01_PROTOCOL" / "analysis_contract_v3_trace.yaml"
V3_RESULTS_ROOT = RESULTS_ROOT / "V3_TRACE_FIRST"
V3_RUN_ROOT = RUNS_ROOT / "v3_trace"


def v3_paths() -> Paths:
    return Paths(contract=V3_CONTRACT_PATH)


def ensure_v3_dirs() -> None:
    for path in (
        V3_RESULTS_ROOT / "00_AUDIT",
        V3_RESULTS_ROOT / "01_TEACHER",
        V3_RESULTS_ROOT / "02_PILOT",
        V3_RESULTS_ROOT / "03_FINAL",
        V3_RESULTS_ROOT / "04_STRESS",
        V3_RESULTS_ROOT / "05_ANALYSIS",
        V3_RESULTS_ROOT / "figures_v3",
        V3_RUN_ROOT / "checkpoints",
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_v3_contract() -> dict:
    return load_contract(V3_CONTRACT_PATH)


def load_v3_frame() -> pd.DataFrame:
    return load_panel_targets().copy()


def _anchor_paths(contract: dict) -> dict[str, Path]:
    anchors = contract["read_only_anchors"]
    return {
        key: PROJECT_ROOT / value
        for key, value in anchors.items()
        if key != "repository_commit"
    }


def audit_trace_v3() -> dict:
    ensure_v3_dirs()
    contract = load_v3_contract()
    frame = load_v3_frame()
    anchor_paths = _anchor_paths(contract)
    hashes = {key: sha256_file(path) for key, path in anchor_paths.items()}
    cache = v3_paths().feature_store / "videomae_v2"
    cached = list(cache.glob("*.npz"))
    checks = {
        "repository_anchor": git_head(PROJECT_ROOT) == contract["read_only_anchors"]["repository_commit"],
        "samples": len(frame) == int(contract["data"]["samples"]),
        "official_train": int((frame.official_split == "train").sum()) == int(contract["data"]["official_train"]),
        "official_test": int((frame.official_split == "test").sum()) == int(contract["data"]["official_test"]),
        "seven_judge_test": int(((frame.official_split == "test") & frame.disagreement_primary_eligible).sum()) == int(contract["data"]["official_test_seven_judge"]),
        "event_families": int(frame.event_family.nunique()) == int(contract["data"]["source_event_families"]),
        "videomae_cache_complete": len(cached) == len(frame),
        "v1_v2_anchors_exist": all(path.exists() for path in anchor_paths.values()),
        "v2_contract_frozen": bool(load_contract(anchor_paths["v2_contract"])["state"]["frozen"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "anchor_hashes": hashes,
        "anchor_paths": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in anchor_paths.items()},
        "repository_anchor": contract["read_only_anchors"]["repository_commit"],
        "current_git_head": git_head(PROJECT_ROOT),
        "cached_feature_files": len(cached),
        "cached_feature_bytes": sum(path.stat().st_size for path in cached),
        "material_passport": "experiment-agent / audit / 2026-08-18 / VERIFIED_DATA / trustdive_trace_v3",
    }
    write_json(V3_RESULTS_ROOT / "00_AUDIT" / "trace_audit_v3.json", result)
    return result


def require_v3_audit() -> dict:
    path = V3_RESULTS_ROOT / "00_AUDIT" / "trace_audit_v3.json"
    if not path.exists():
        raise RuntimeError("Run audit-trace --protocol v3 first")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("v3 trace audit did not pass")
    return result


def require_v3_frozen() -> dict:
    contract = load_v3_contract()
    if not contract["state"]["frozen"]:
        raise RuntimeError("Freeze the v3 contract before official-test outputs")
    return contract


def freeze_v3_contract() -> dict:
    selection = V3_RESULTS_ROOT / "02_PILOT" / "selected_config_v3.json"
    if not selection.exists():
        raise RuntimeError("Run pilot-trace --protocol v3 first")
    selected = json.loads(selection.read_text(encoding="utf-8"))
    if selected.get("status") != "PASS":
        raise RuntimeError("The v3 pilot gate did not pass")
    contract = load_v3_contract()
    if contract["state"]["frozen"]:
        return contract["state"]
    contract["state"].update(
        {
            "frozen": True,
            "frozen_at": utc_now(),
            "official_test_unlocked": True,
            "selected_config_sha256": sha256_file(selection),
            "contract_sha256": None,
        }
    )
    canonical = yaml.safe_dump(contract, sort_keys=True, allow_unicode=True).encode("utf-8")
    contract["state"]["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    V3_CONTRACT_PATH.write_text(
        yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    result = dict(contract["state"])
    write_json(V3_RESULTS_ROOT / "02_PILOT" / "contract_freeze_v3.json", result)
    return result
