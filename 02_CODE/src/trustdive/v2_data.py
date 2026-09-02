from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import CONTRACT_PATH, PROJECT_ROOT, RESULTS_ROOT, RUNS_ROOT, Paths, load_contract
from .data import load_manifest
from .splits import add_analysis_roles, source_heldout_roles
from .util import sha256_file, stable_json, utc_now, write_json


V2_CONTRACT_PATH = PROJECT_ROOT / "01_PROTOCOL" / "analysis_contract_v2.yaml"
V2_RESULTS_ROOT = RESULTS_ROOT / "V2_DISAGREEMENT"


def v2_paths() -> Paths:
    return Paths(contract=V2_CONTRACT_PATH)


def ensure_v2_dirs() -> None:
    for path in (
        V2_RESULTS_ROOT / "00_AUDIT",
        V2_RESULTS_ROOT / "01_FEATURES",
        V2_RESULTS_ROOT / "02_TUNING",
        V2_RESULTS_ROOT / "03_FINAL",
        V2_RESULTS_ROOT / "04_ANALYSIS",
        V2_RESULTS_ROOT / "figures",
        RUNS_ROOT / "v2_disagreement" / "checkpoints",
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_v2_contract() -> dict:
    return load_contract(V2_CONTRACT_PATH)


def load_base_manifest() -> pd.DataFrame:
    return load_manifest(Paths())


def judge_values_valid(values: list[float], minimum: float = 0.0, maximum: float = 10.0) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(
        len(array) in {3, 5, 7}
        and np.isfinite(array).all()
        and (array >= minimum).all()
        and (array <= maximum).all()
        and np.isclose(array * 2.0, np.round(array * 2.0), atol=1e-8).all()
    )


def judge_invalid_reason(values: list[float]) -> str:
    array = np.asarray(values, dtype=float)
    reasons: list[str] = []
    if len(array) not in {3, 5, 7}:
        reasons.append("unsupported_panel_size")
    if not np.isfinite(array).all():
        reasons.append("nonfinite")
    if ((array < 0.0) | (array > 10.0)).any():
        reasons.append("outside_0_10")
    if not np.isclose(array * 2.0, np.round(array * 2.0), atol=1e-8).all():
        reasons.append("not_half_point_increment")
    return "valid" if not reasons else ";".join(reasons)


def official_panel_aggregate(values: list[float]) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    if len(ordered) == 7:
        selected = ordered[2:-2]
    elif len(ordered) == 5:
        selected = ordered[1:-1]
    elif len(ordered) == 3:
        selected = ordered
    else:
        raise ValueError("Panel must contain 3, 5, or 7 judges")
    return float(np.mean(selected))


def mean_pairwise_absolute_difference(values: list[float]) -> float:
    pairs = [abs(float(a) - float(b)) for a, b in combinations(values, 2)]
    return float(np.mean(pairs)) if pairs else 0.0


def build_panel_targets(write: bool = True) -> pd.DataFrame:
    ensure_v2_dirs()
    frame = add_analysis_roles(
        load_base_manifest(), int(load_v2_contract()["random"]["master_seed"]), attempts=2000
    )
    frame["source_role"] = source_heldout_roles(
        frame, int(load_v2_contract()["random"]["master_seed"]), attempts=2000
    )
    parsed = frame.judge_scores_json.map(json.loads)
    frame["judge_label_valid"] = parsed.map(judge_values_valid)
    frame["judge_invalid_reason"] = parsed.map(judge_invalid_reason)
    frame["judge_consensus_median"] = parsed.map(lambda x: float(np.median(x)))
    frame["judge_sample_sd"] = parsed.map(
        lambda x: float(np.std(np.asarray(x, dtype=float), ddof=1)) if len(x) > 1 else 0.0
    )
    frame["judge_pairwise_abs"] = parsed.map(mean_pairwise_absolute_difference)
    frame["judge_official_aggregate"] = parsed.map(official_panel_aggregate)
    frame["judge_aggregate_abs_error"] = (
        frame.execution_quality - frame.judge_official_aggregate
    ).abs()
    frame["disagreement_primary_eligible"] = frame.judge_label_valid & (frame.judge_count == 7)
    frame["disagreement_supplemental_eligible"] = frame.judge_label_valid & (frame.judge_count == 5)
    frame["judge_scores_json"] = parsed.map(stable_json)
    if write:
        destination = V2_RESULTS_ROOT / "00_AUDIT" / "panel_targets_v2.parquet"
        frame.to_parquet(destination, index=False)
        summary = {
            "rows": int(len(frame)),
            "official_split": frame.official_split.value_counts().to_dict(),
            "analysis_role": frame.analysis_role.value_counts().to_dict(),
            "source_role": frame.source_role.value_counts().to_dict(),
            "judge_count": {str(k): int(v) for k, v in frame.judge_count.value_counts().to_dict().items()},
            "invalid_judge_arrays": int((~frame.judge_label_valid).sum()),
            "invalid_clips": frame.loc[
                ~frame.judge_label_valid,
                ["clip_uid", "official_split", "judge_count", "judge_scores_json", "judge_invalid_reason"],
            ].to_dict("records"),
            "primary_seven_judge_total": int(frame.disagreement_primary_eligible.sum()),
            "primary_seven_judge_test": int(
                (frame.disagreement_primary_eligible & (frame.official_split == "test")).sum()
            ),
            "five_judge_test": int(
                (frame.disagreement_supplemental_eligible & (frame.official_split == "test")).sum()
            ),
            "official_aggregate_abs_error_quantiles": {
                str(q): float(frame.judge_aggregate_abs_error.quantile(q))
                for q in (0.5, 0.9, 0.95, 0.99, 1.0)
            },
            "path": str(destination),
        }
        write_json(V2_RESULTS_ROOT / "00_AUDIT" / "panel_targets_summary_v2.json", summary)
    return frame


def load_panel_targets() -> pd.DataFrame:
    path = V2_RESULTS_ROOT / "00_AUDIT" / "panel_targets_v2.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Build v2 panel targets first: {path}")
    return pd.read_parquet(path)


def audit_panel() -> dict:
    ensure_v2_dirs()
    contract = load_v2_contract()
    frame = build_panel_targets(write=False)
    anchors = contract["v1_read_only_anchors"]
    anchor_paths = {
        "stage_a_stop_sha256": RESULTS_ROOT / "01_PROBE" / "stage_a_stop.json",
        "stop_decision_sha256": RESULTS_ROOT / "01_PROBE" / "STOP_DECISION.md",
        "v1_contract_sha256": CONTRACT_PATH,
    }
    anchor_checks = {
        name: sha256_file(path) == str(anchors[name]).lower()
        for name, path in anchor_paths.items()
    }
    expected = contract["data"]
    checks = {
        "samples": len(frame) == int(expected["samples"]),
        "official_train": int((frame.official_split == "train").sum()) == int(expected["official_train"]),
        "official_test": int((frame.official_split == "test").sum()) == int(expected["official_test"]),
        "seven_judge": int((frame.judge_count == 7).sum()) == int(expected["judge_panels"]["7"]),
        "seven_judge_test": int(((frame.judge_count == 7) & (frame.official_split == "test")).sum())
        == int(expected["official_test_seven_judge"]),
        "known_invalid_judge_arrays": int((~frame.judge_label_valid).sum()) == 3,
        "event_families": int(frame.event_family.nunique()) == int(expected["source_event_families"]),
        "v1_anchors": all(anchor_checks.values()),
        "v1_contract_unfrozen": not bool(load_contract()["state"]["frozen"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "v1_anchor_checks": anchor_checks,
        "invalid_judge_arrays": frame.loc[
            ~frame.judge_label_valid,
            ["clip_uid", "judge_scores_json", "judge_invalid_reason"],
        ].to_dict("records"),
        "material_passport": "experiment-agent / audit / 2026-08-17 / VERIFIED_DATA / trustdive_disagreement_v2",
    }
    write_json(V2_RESULTS_ROOT / "00_AUDIT" / "panel_audit_v2.json", result)
    return result


def require_v2_audit() -> dict:
    path = V2_RESULTS_ROOT / "00_AUDIT" / "panel_audit_v2.json"
    if not path.exists():
        raise RuntimeError("Run audit-panel --protocol v2 first")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("The v2 panel audit did not pass")
    return result


def require_contract_frozen() -> dict:
    contract = load_v2_contract()
    if not bool(contract["state"]["frozen"]):
        raise RuntimeError("The v2 contract must be frozen before official-test prediction or analysis")
    return contract


def freeze_v2_contract() -> dict:
    contract = load_v2_contract()
    selection_path = V2_RESULTS_ROOT / "02_TUNING" / "selected_config_v2.json"
    if not selection_path.exists():
        raise RuntimeError("Tune v2 on fit/validation data before freezing the contract")
    if contract["state"]["frozen"]:
        return contract["state"]
    contract["state"]["frozen"] = True
    contract["state"]["frozen_at"] = utc_now()
    contract["state"]["official_test_unlocked"] = True
    contract["state"]["contract_sha256"] = None
    canonical = yaml.safe_dump(contract, sort_keys=True, allow_unicode=True).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    contract["state"]["contract_sha256"] = digest
    V2_CONTRACT_PATH.write_text(
        yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    result = {
        "frozen": True,
        "frozen_at": contract["state"]["frozen_at"],
        "official_test_unlocked": True,
        "contract_sha256": digest,
        "selected_config_sha256": sha256_file(selection_path),
    }
    write_json(V2_RESULTS_ROOT / "02_TUNING" / "contract_freeze_v2.json", result)
    return result
