from __future__ import annotations

import numpy as np
import torch

from trustdive.v2_data import official_panel_aggregate
from trustdive.v9_data import _round_score, _stable_int
from trustdive.v9_modeling import JudgePhaseNet


def test_half_point_range_and_official_aggregate():
    assert _round_score(-3.1) == 0.0
    assert _round_score(10.4) == 10.0
    assert _round_score(7.24) == 7.0
    assert _round_score(7.26) == 7.5
    assert official_panel_aggregate([6, 7, 7.5, 8, 8.5, 9, 10]) == 8.0


def test_virtual_style_hash_is_deterministic():
    a = _stable_int(20260824, "family-1", "phase_bias")
    b = _stable_int(20260824, "family-1", "phase_bias")
    c = _stable_int(20260825, "family-1", "phase_bias")
    assert a == b
    assert a != c


def test_judge_network_is_panel_permutation_invariant_and_slot_equivariant():
    torch.manual_seed(7)
    model = JudgePhaseNet(8, 10, 7, 53, 64, 0.0, include_phase=True).eval()
    judge = torch.randn(4, 7, 8)
    phase = torch.randn(4, 3, 10)
    global_feature = torch.randn(4, 7)
    action = torch.tensor([1, 2, 3, 4])
    permutation = torch.tensor([4, 1, 6, 0, 3, 2, 5])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        panel, slot, kind, phase_out = model(judge, phase, global_feature, action)
        panel_p, slot_p, kind_p, phase_p = model(judge[:, permutation], phase, global_feature, action)
    assert torch.allclose(panel, panel_p, atol=1e-6)
    assert torch.allclose(kind, kind_p, atol=1e-6)
    assert torch.allclose(phase_out, phase_p, atol=1e-6)
    assert torch.allclose(slot, slot_p[:, inverse], atol=1e-6)


def test_matched_pair_differs_only_at_target_slot():
    base = np.array([7.0, 7.5, 8.0, 8.0, 8.5, 8.5, 9.0])
    null = base.copy(); null[3] = 7.5
    anomaly = base.copy(); anomaly[3] = 9.0
    changed = np.flatnonzero(null != anomaly)
    assert changed.tolist() == [3]

