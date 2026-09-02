import numpy as np
import pandas as pd

from trustdive.data import clip_uid, event_family
from trustdive.metrics import execution_quality, restore_total_score, score_metrics
from trustdive.modeling import monotonic_decode, ordered_phase_targets, phase_pool
from trustdive.statistics import holm_adjust, leave_one_judge_consensus, split_conformal_radius


def test_event_family_normalizes_round_and_replay_suffix():
    assert event_family("FINA_Women10m_final_r4") == "FINA_Women10m_final"
    assert event_family("BudapestWomenFinal_2") == "BudapestWomenFinal"
    assert clip_uid("01", 7) == "01::7"


def test_execution_quality_round_trip():
    score = np.asarray([81.18, 88.0, 0.0])
    difficulty = np.asarray([3.3, 3.2, 3.4])
    quality = execution_quality(score, difficulty)
    np.testing.assert_allclose(restore_total_score(quality, difficulty), score)


def test_pose_backbone_accepts_letterboxed_canvas():
    import torch

    from trustdive.features import create_pose_model

    model = create_pose_model(pretrained=False).eval()
    with torch.no_grad():
        heatmaps = model(torch.zeros(1, 3, 192, 256))
    assert heatmaps.shape == (1, 16, 48, 64)


def test_relative_l2_is_zero_for_exact_prediction():
    metrics = score_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    assert metrics["spearman"] == 1.0
    assert metrics["relative_l2"] == 0.0
    assert metrics["mae"] == 0.0


def test_ordered_targets_and_monotonic_decode():
    target = ordered_phase_targets(90, [20, 60], 9)
    assert target.tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2]
    logits = np.full((9, 3), -5.0)
    logits[:2, 0] = 5
    logits[2:6, 1] = 5
    logits[6:, 2] = 5
    assert monotonic_decode(logits).tolist() == target.tolist()


def test_phase_pool_outputs_three_rows():
    sequence = np.arange(18, dtype=np.float32).reshape(9, 2)
    labels = np.repeat([0, 1, 2], 3)
    pooled = phase_pool(sequence, labels)
    assert pooled.shape == (3, 2)
    np.testing.assert_allclose(pooled[0], sequence[:3].mean(axis=0))


def test_seven_judge_leave_one_consensus():
    rows = leave_one_judge_consensus([6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
    assert len(rows) == 7
    assert rows[0]["consensus"] == 7.75
    assert rows[-1]["consensus"] == 7.25


def test_conformal_radius_uses_finite_sample_rank():
    y = np.arange(10, dtype=float)
    prediction = y + np.arange(10) / 10
    radius = split_conformal_radius(y, prediction, 0.9)
    assert np.isclose(radius, 0.9)


def test_holm_adjustment_is_order_preserving():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
