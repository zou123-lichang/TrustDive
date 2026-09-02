from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from .config import Paths, load_contract
from .util import utc_now, write_json


def ledger_path(paths: Paths | None = None) -> Path:
    paths = paths or Paths()
    names = {
        "analysis_contract_v2.yaml": "v2_gpu_budget_ledger.json",
        "analysis_contract_v3_trace.yaml": "v3_gpu_budget_ledger.json",
        "analysis_contract_v4_counterfactual.yaml": "v4_gpu_budget_ledger.json",
        "analysis_contract_v5_cfpd_plus.yaml": "v5_gpu_budget_ledger.json",
        "analysis_contract_v6_exact_review.yaml": "v6_gpu_budget_ledger.json",
        "analysis_contract_v7_risk_task.yaml": "v7_gpu_budget_ledger.json",
        "analysis_contract_v8_phase_conflict.yaml": "v8_gpu_budget_ledger.json",
        "analysis_contract_v9_judge_sim.yaml": "v9_gpu_budget_ledger.json",
    }
    name = names.get(paths.contract.name, "gpu_budget_ledger.json")
    return paths.runs / name


def read_ledger(paths: Paths | None = None) -> dict:
    paths = paths or Paths()
    path = ledger_path(paths)
    if not path.exists():
        return {
            "budget_hours": float(load_contract(paths.contract)["compute"]["gpu_budget_hours"]),
            "entries": [],
        }
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def consumed_hours(paths: Paths | None = None) -> float:
    return sum(float(x["elapsed_seconds"]) for x in read_ledger(paths)["entries"] if x.get("used_gpu")) / 3600.0


def assert_budget(estimated_hours: float = 0.0, paths: Paths | None = None) -> None:
    ledger = read_ledger(paths)
    used = sum(float(x["elapsed_seconds"]) for x in ledger["entries"] if x.get("used_gpu")) / 3600.0
    if used + estimated_hours > float(ledger["budget_hours"]):
        raise RuntimeError(
            f"GPU budget would be exceeded: used={used:.3f} h, estimated={estimated_hours:.3f} h, "
            f"cap={ledger['budget_hours']:.3f} h"
        )


@contextmanager
def gpu_budget_entry(
    command: str,
    estimated_hours: float = 0.0,
    paths: Paths | None = None,
    force_gpu: bool = False,
):
    paths = paths or Paths()
    assert_budget(estimated_hours, paths)
    start = time.perf_counter()
    started_at = utc_now()
    ok = False
    try:
        yield
        ok = True
    finally:
        elapsed = time.perf_counter() - start
        used_gpu = False
        peak_bytes = None
        try:
            import torch

            used_gpu = bool(
                force_gpu or (torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 0)
            )
            peak_bytes = int(torch.cuda.max_memory_allocated()) if used_gpu else 0
        except ImportError:
            pass
        ledger = read_ledger(paths)
        ledger["entries"].append(
            {
                "command": command,
                "started_at": started_at,
                "finished_at": utc_now(),
                "elapsed_seconds": elapsed,
                "used_gpu": used_gpu,
                "peak_vram_bytes": peak_bytes,
                "success": ok,
            }
        )
        write_json(ledger_path(paths), ledger)
