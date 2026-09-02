from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from trustdive.manuscript_figures import (
    OUTPUT_ROOT,
    PHASES,
    SOURCE_ROOT,
    _phase_frame_indices,
)


def test_phase_frame_indices_stay_inside_predicted_phase() -> None:
    labels = np.asarray([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int8)
    frame_count = 80
    sampled = labels[np.linspace(0, len(labels) - 1, frame_count).round().astype(int)]
    selections = _phase_frame_indices(labels, frame_count)

    assert tuple(selections) == PHASES
    for phase_index, phase in enumerate(PHASES):
        assert len(selections[phase]) == 3
        assert selections[phase] == sorted(selections[phase])
        assert all(sampled[index] == phase_index for index in selections[phase])


def test_case_selection_audit_is_deterministic_and_unique() -> None:
    cases = pd.read_csv(SOURCE_ROOT / "case_selection_audit.csv")
    assert cases.case_type.tolist() == [
        "typical_accurate",
        "high_disagreement_gain",
        "scoring_failure",
        "boundary_sensitive",
        "reference_sensitive",
    ]
    assert cases.clip_uid.is_unique
    assert (cases.query_index != cases.reference_index).all()


def test_frame_audit_contains_three_frames_per_phase_and_role() -> None:
    frames = pd.read_csv(SOURCE_ROOT / "frame_selection_audit.csv")
    grouped = frames.groupby(["case_type", "role", "phase"]).size()
    assert set(frames.phase) == set(PHASES)
    assert set(frames.role) == {"query", "reference"}
    assert (grouped == 3).all()
    assert not frames.local_adjustment.astype(bool).any()


def test_figure_contract_and_frozen_counts() -> None:
    contracts = json.loads((OUTPUT_ROOT / "figure_contracts.json").read_text(encoding="utf-8"))
    audit = json.loads((OUTPUT_ROOT / "frozen_input_audit.json").read_text(encoding="utf-8"))
    assert contracts["backend"] == "Python/matplotlib exclusively"
    assert contracts["Figure2"]["size_mm"] == [180, 128]
    assert contracts["Figure3"]["size_mm"] == [180, 140]
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())


def test_explicit_figure_text_is_not_smaller_than_eight_points() -> None:
    source = (OUTPUT_ROOT.parents[2] / "02_CODE" / "src" / "trustdive" / "manuscript_figures.py").read_text(encoding="utf-8")
    explicit_sizes = [float(value) for value in re.findall(r"fontsize\s*=\s*([0-9]+(?:\.[0-9]+)?)", source)]
    label_sizes = [float(value) for value in re.findall(r"labelsize\s*=\s*([0-9]+(?:\.[0-9]+)?)", source)]
    assert explicit_sizes
    assert min(explicit_sizes + label_sizes) >= 8.0
