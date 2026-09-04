from __future__ import annotations

import numpy as np
import torch

from trustdive.feature_baselines import (
    CoReStyleAdapter,
    TSAStyleAdapter,
    masked_reference_median,
    parser,
    phase_pool_sequences,
)
from trustdive.metrics import restore_total_score


def test_v10_cli_contract() -> None:
    assert parser().parse_args(["audit"]).command == "audit"
    parsed = parser().parse_args(["train", "--model", "core_style", "--stage", "pilot"])
    assert parsed.model == "core_style"
    assert parsed.stage == "pilot"
    parsed = parser().parse_args(["train", "--model", "all", "--stage", "final"])
    assert parsed.model == "all"


def test_masked_reference_median_is_permutation_invariant() -> None:
    values = torch.tensor([[1.0, 7.0, 3.0, 5.0, 9.0]])
    valid = torch.tensor([[True, True, True, False, False]])
    expected = masked_reference_median(values, valid)
    permutation = torch.tensor([2, 0, 4, 1, 3])
    observed = masked_reference_median(values[:, permutation], valid[:, permutation])
    assert torch.equal(expected, observed)
    assert expected.item() == 3.0


def test_masked_reference_median_returns_zero_for_open_set() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0]])
    valid = torch.tensor([[False, False, False]])
    result = masked_reference_median(values, valid)
    assert torch.equal(result, torch.zeros(1))


def test_phase_pool_sequences_uses_only_phase_members() -> None:
    sequences = np.arange(1 * 6 * 2, dtype=np.float32).reshape(1, 6, 2)
    labels = np.asarray([[0, 0, 1, 1, 2, 2]], dtype=np.int8)
    pooled = phase_pool_sequences(sequences, labels)
    np.testing.assert_allclose(pooled[0, 0], sequences[0, :2].mean(axis=0))
    np.testing.assert_allclose(pooled[0, 1], sequences[0, 2:4].mean(axis=0))
    np.testing.assert_allclose(pooled[0, 2], sequences[0, 4:].mean(axis=0))


def test_total_score_recovery() -> None:
    quality = np.asarray([8.0, 9.0])
    difficulty = np.asarray([3.0, 3.5])
    np.testing.assert_allclose(restore_total_score(quality, difficulty), [72.0, 94.5])


def test_adapter_shapes_and_finite_outputs() -> None:
    valid = torch.ones((2, 5), dtype=torch.bool)
    global_query = torch.randn(2, 256)
    global_reference = torch.randn(2, 5, 256)
    reference_error = torch.randn(2, 5)
    core = CoReStyleAdapter(hidden=64, dropout=0.0)
    core_output = core(global_query, global_reference, reference_error, valid)
    assert core_output.shape == (2,)
    assert torch.isfinite(core_output).all()

    phase_query = torch.randn(2, 3, 1024)
    phase_reference = torch.randn(2, 5, 3, 1024)
    tsa = TSAStyleAdapter(hidden=64, dropout=0.0, heads=2)
    tsa_output = tsa(phase_query, phase_reference, reference_error, valid)
    assert tsa_output.shape == (2,)
    assert torch.isfinite(tsa_output).all()


