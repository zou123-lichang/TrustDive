from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "02_CODE"
RESULTS_ROOT = PROJECT_ROOT / "03_RESULTS"
RUNS_ROOT = PROJECT_ROOT / "runs"
CONTRACT_PATH = PROJECT_ROOT / "01_PROTOCOL" / "analysis_contract.yaml"


@dataclass(frozen=True)
class Paths:
    project: Path = PROJECT_ROOT
    code: Path = CODE_ROOT
    results: Path = RESULTS_ROOT
    runs: Path = RUNS_ROOT
    contract: Path = CONTRACT_PATH
    fine_diving: Path = Path(
        os.environ.get(
            "TRUSTDIVE_FINE_DIVING",
            str(PROJECT_ROOT / "data" / "FineDiving"),
        )
    )
    pose_dive: Path = Path(
        os.environ.get("TRUSTDIVE_POSE_DIVE", str(PROJECT_ROOT / "data" / "PoseDive"))
    )
    splash: Path = Path(
        os.environ.get("TRUSTDIVE_SPLASH", str(PROJECT_ROOT / "data" / "splash_data"))
    )

    @property
    def manifest(self) -> Path:
        return self.results / "00_AUDIT" / "manifest.parquet"

    @property
    def feature_store(self) -> Path:
        return self.project / "feature_store"

    @property
    def trimmed_zip(self) -> Path:
        return self.fine_diving / "Trimmed_Video_Frames" / "FineDiving_Trimmed_VideoFrames.zip"


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict):
        raise ValueError(f"Contract is not a mapping: {path}")
    return contract


def ensure_project_dirs(paths: Paths | None = None) -> None:
    paths = paths or Paths()
    for path in (
        paths.results / "00_AUDIT",
        paths.results / "01_PROBE",
        paths.results / "02_SCORE",
        paths.results / "03_TRACE",
        paths.results / "04_PANEL",
        paths.results / "figures",
        paths.runs,
        paths.feature_store / "rgb",
        paths.feature_store / "pose",
        paths.feature_store / "splash",
    ):
        path.mkdir(parents=True, exist_ok=True)
