import json

import numpy as np
import pandas as pd
import torch

from trustdive.v2_data import (
    build_panel_targets,
    judge_values_valid,
    mean_pairwise_absolute_difference,
    official_panel_aggregate,
)
from trustdive.v2_modeling import _model_classes, _phase_boundary_error, _robust_standardize


def test_judge_label_validation_and_aggregation():
    assert judge_values_valid([6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
    assert not judge_values_valid([2.5, 4.0, 45.0, 3.5, 5.0, 4.0, 4.0])
    assert not judge_values_valid([7.5, 8.0, 8.9, 8.5, 8.5])
    assert official_panel_aggregate([6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]) == 7.5
    assert official_panel_aggregate([6.0, 6.5, 7.0, 7.5, 8.0]) == 7.0
    assert official_panel_aggregate([6.0, 6.5, 7.0]) == 6.5


def test_pairwise_disagreement_matches_manual_result():
    values = [6.0, 7.0, 8.0]
    assert np.isclose(mean_pairwise_absolute_difference(values), (1.0 + 2.0 + 1.0) / 3.0)


def test_live_v2_panel_targets_preserve_score_rows_and_flag_three_bad_arrays():
    frame = build_panel_targets(write=False)
    assert len(frame) == 3000
    assert (~frame.judge_label_valid).sum() == 3
    assert frame.disagreement_primary_eligible.sum() == 1368
    assert (
        frame.disagreement_primary_eligible & (frame.official_split == "test")
    ).sum() == 325
    assert set(frame.loc[frame.official_split == "test", "analysis_role"]) == {"official_test"}
    bad = frame.loc[~frame.judge_label_valid, "judge_scores_json"].map(json.loads).tolist()
    assert any(45.0 in values for values in bad)
    assert any(55.5 in values for values in bad)
    assert any(8.9 in values for values in bad)


def test_phase_model_contributions_and_residual_reconstruct_score():
    _, _, PhaseRelative = _model_classes(global_dim=4, phase_dim=5, dropout=0.0)
    model = PhaseRelative(True).eval()
    global_features = torch.zeros(3, 4)
    phase = torch.randn(3, 3, 5)
    base = torch.tensor([7.0, 7.5, 8.0])
    with torch.no_grad():
        prediction, contribution, residual, sigma = model(global_features, phase, base)
    torch.testing.assert_close(prediction, base + contribution.sum(dim=1) + residual)
    assert torch.all(residual <= 1.0)
    assert torch.all(residual >= -1.0)
    assert torch.all(sigma > 0.0)


def test_robust_standardization_uses_fit_only():
    values = np.asarray([[0.0], [1.0], [1000.0]], dtype=np.float32)
    transformed, audit = _robust_standardize(values, np.asarray([0, 1]))
    assert transformed[2, 0] == 12.0
    assert audit["clipped_fraction"] > 0.0


def test_phase_boundary_error_is_normalized():
    truth = np.asarray([[0, 0, 1, 1, 2, 2]])
    prediction = np.asarray([[0, 1, 1, 1, 1, 2]])
    error = _phase_boundary_error(prediction, truth)
    assert np.isclose(error[0], (1 / 6 + 1 / 6) / 2)
