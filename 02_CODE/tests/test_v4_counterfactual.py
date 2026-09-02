import numpy as np
import torch

from trustdive.pipeline import parser
from trustdive.v4_counterfactual import (
    exact_three_phase_shapley,
    hybrid_sequence,
    resample_phase_labels,
    shapley_efficiency_error,
)
from trustdive.v4_modeling import _jitter_phase_labels, cfpd_model_class
from trustdive.v4_teacher import i3d_output_to_cache, rica2_raw_output_to_execution_quality
from trustdive.v4_stress import _token_dropout_sequences
from trustdive.v4_analysis import VISIBLE_REVIEW_REASONS
from trustdive.metrics import aqa_score_metrics


def test_exact_shapley_recovers_additive_phase_effects():
    effects = np.asarray([0.5, -1.25, 2.0])
    values = np.asarray([4.0 + sum(effects[index] for index in range(3) if mask & (1 << index)) for mask in range(8)])
    phi = exact_three_phase_shapley(values)
    np.testing.assert_allclose(phi, effects, atol=1e-12)
    np.testing.assert_allclose(shapley_efficiency_error(values, phi), 0.0, atol=1e-12)


def test_rica2_raw_output_is_already_execution_quality():
    raw = torch.tensor([[[7.0], [9.0]], [[4.0], [6.0]]])
    quality = rica2_raw_output_to_execution_quality(raw)
    torch.testing.assert_close(quality, torch.tensor([8.0, 5.0]))


def test_exact_shapley_splits_interaction_symmetrically():
    values = np.zeros(8, dtype=float)
    for mask in range(8):
        values[mask] = float(bool(mask & 1)) + 2.0 * float(bool(mask & 2))
        if (mask & 1) and (mask & 2):
            values[mask] += 3.0
    phi = exact_three_phase_shapley(values)
    np.testing.assert_allclose(phi, [2.5, 3.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(phi.sum(), values[7] - values[0], atol=1e-12)


def test_hybrid_sequence_uses_query_only_for_selected_phases():
    query = np.arange(18, dtype=np.float32).reshape(6, 3)
    reference = 100.0 + np.arange(18, dtype=np.float32).reshape(6, 3)
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    mixed = hybrid_sequence(query, reference, labels, labels, coalition_mask=0b010)
    np.testing.assert_array_equal(mixed[labels == 1], query[labels == 1])
    np.testing.assert_array_equal(mixed[labels != 1], reference[labels != 1])


def test_phase_resampling_produces_three_ordered_nonempty_phases():
    original = np.asarray([0, 0, 1, 1, 1, 2, 2, 2])
    output = resample_phase_labels(original, 9)
    assert output.shape == (9,)
    assert set(output.tolist()) == {0, 1, 2}
    assert np.all(np.diff(output) >= 0)


def test_cfpd_has_no_free_residual_and_reconstructs_exactly():
    Model = cfpd_model_class(input_dim=16, hidden=8, dropout=0.0)
    model = Model().eval()
    pair = torch.randn(5, 5, 3, 16)
    base = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0])
    with torch.no_grad():
        prediction, contribution, per_reference = model(pair, base)
    torch.testing.assert_close(prediction, base + contribution.sum(dim=1), atol=1e-6, rtol=0)
    assert per_reference.shape == (5, 5, 3)


def test_rica2_feature_cache_uses_time_by_channel_disk_layout():
    extracted = np.zeros((2, 1024, 9), dtype=np.float32)
    extracted[0, 17, 3] = 4.5
    cached = i3d_output_to_cache(extracted)
    assert cached.shape == (2, 9, 1024)
    assert cached[0, 3, 17] == np.float16(4.5)


def test_v4_token_dropout_is_deterministic_and_drops_one_of_nine_tokens():
    sequences = np.ones((4, 9, 6), dtype=np.float32)
    first = _token_dropout_sequences(sequences, 20260818)
    second = _token_dropout_sequences(sequences, 20260818)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sum(np.all(first == 0.0, axis=2), axis=1), np.ones(4, dtype=int))


def test_v4_boundary_jitter_is_deterministic_and_preserves_phase_order():
    labels = np.tile(np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2]), (8, 1))
    first = _jitter_phase_labels(labels, 20260817)
    second = _jitter_phase_labels(labels, 20260817)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first, axis=1) >= 0)
    assert all(set(row.tolist()) == {0, 1, 2} for row in first)


def test_v4_cli_contract_is_exposed():
    for command in ("prepare-teacher", "train-teacher", "export-teacher", "build-counterfactuals", "pilot"):
        parsed = parser().parse_args([command, "--protocol", "v4"])
        assert parsed.command == command
        assert parsed.protocol == "v4"
    parsed = parser().parse_args(["train", "--protocol", "v4", "--stage", "final"])
    assert parsed.protocol == "v4"
    parsed = parser().parse_args(["stress-test", "--protocol", "v4"])
    assert parsed.protocol == "v4"


def test_visible_review_reasons_never_expose_seed_disagreement():
    assert all("seed" not in value.lower() and "种子" not in value for value in VISIBLE_REVIEW_REASONS)


def test_v4_relative_l2_matches_rica2_range_normalization():
    target = np.asarray([0.0, 5.0, 10.0])
    prediction = np.asarray([1.0, 4.0, 11.0])
    metrics = aqa_score_metrics(target, prediction)
    np.testing.assert_allclose(metrics["relative_l2"], 1.0)
    assert metrics["relative_l2_variance_ratio_internal"] != metrics["relative_l2"]
