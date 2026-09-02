from __future__ import annotations

import numpy as np
import pandas as pd

from trustdive.pipeline import parser
from trustdive.v4_counterfactual import exact_three_phase_shapley
from trustdive.v6_modeling import V6Assets
from trustdive.v7_analysis import _empirical_percentile, _top_fraction
from trustdive.v7_data import _is_protected_historical, load_v7_contract
from trustdive.v7_modeling import (
    _reference_map_for_pool,
    _sample_weights,
    make_ridge_features_v7,
)


def _synthetic_assets() -> V6Assets:
    frame = pd.DataFrame({
        "clip_uid": [f"c{i}" for i in range(8)],
        "action_type": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "event_family": ["f1", "f2", "f3", "f4", "f1", "f2", "f3", "f4"],
        "execution_quality": [5.0, 5.5, 6.0, 6.5, 4.0, 4.5, 5.0, 5.5],
        "analysis_role": ["fit"] * 8,
    })
    latent = np.eye(8, 256, dtype=np.float32)
    teacher = np.asarray([4.8, 5.4, 5.8, 6.4, 4.1, 4.4, 4.8, 5.4], dtype=np.float32)
    return V6Assets(frame, latent, np.zeros((8, 29, 512), np.float32), np.zeros((8, 52), np.float32), teacher)


def test_v7_cli_contract_contains_all_public_commands():
    expected = (
        ["audit", "--protocol", "v7"],
        ["build-risk-task", "--protocol", "v7"],
        ["train-baselines", "--protocol", "v7"],
        ["train", "--protocol", "v7"],
        ["build-phase-evidence", "--protocol", "v7"],
        ["analyze-risk", "--protocol", "v7"],
        ["render-reports", "--protocol", "v7"],
        ["verify", "--protocol", "v7"],
    )
    for argv in expected:
        assert parser().parse_args(argv).protocol == "v7"


def test_review_budget_selects_exact_fraction_and_forces_open_set():
    risk = np.asarray([0.1, 0.9, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0])
    force = np.zeros(10, dtype=bool)
    force[0] = True
    selected = _top_fraction(risk, 0.2, force)
    assert selected.sum() == 2
    assert selected[0]
    assert selected[1]


def test_empirical_percentiles_use_calibration_reference_only():
    reference = np.asarray([1.0, 2.0, 3.0, 4.0])
    values = np.asarray([0.0, 2.0, 5.0])
    np.testing.assert_allclose(_empirical_percentile(reference, values), [0.0, 0.5, 1.0])


def test_reference_map_excludes_query_family_and_action_mismatch():
    assets = _synthetic_assets()
    mapping = _reference_map_for_pool(assets, np.arange(8))
    for query, refs in enumerate(mapping["references"]):
        for ref in refs[refs >= 0]:
            assert assets.frame.loc[query, "action_type"] == assets.frame.loc[ref, "action_type"]
            assert assets.frame.loc[query, "event_family"] != assets.frame.loc[ref, "event_family"]


def test_ridge_features_include_latent_teacher_and_reference_statistics():
    assets = _synthetic_assets()
    mapping = _reference_map_for_pool(assets, np.arange(8))
    features = make_ridge_features_v7(assets, mapping)
    assert features.shape == (8, 262)
    assert np.isfinite(features).all()


def test_risk_weights_do_not_impute_missing_judge_label():
    manifest = pd.DataFrame({
        "high_error_risk_proxy": [False, True, False],
        "high_judge_risk": [True, True, True],
        "disagreement_primary_eligible": [True, False, True],
    })
    weight = _sample_weights(manifest, np.arange(3), 1.0, 1.0)
    np.testing.assert_allclose(weight, [2.0, 2.0, 2.0])


def test_exact_three_phase_shapley_reconstructs_interacting_function():
    coalition = np.asarray([0.0, 1.0, 2.0, 4.0, 3.0, 5.5, 6.5, 10.0])
    phi = exact_three_phase_shapley(coalition[None, None, :])[0, 0]
    assert abs(coalition[0] + phi.sum() - coalition[7]) < 1e-12


def test_historical_protection_detects_v6_but_allows_v7_and_shared_router():
    assert _is_protected_historical("03_RESULTS/V6_EXACT_REVIEW/x.json")
    assert _is_protected_historical("02_CODE/src/trustdive/v6_analysis.py")
    assert not _is_protected_historical("03_RESULTS/V7_RISK_TASK/x.json")
    assert not _is_protected_historical("03_RESULTS/V8_PHASE_CONFLICT/x.json")
    assert not _is_protected_historical("01_PROTOCOL/analysis_contract_v9_judge_sim.yaml")
    assert not _is_protected_historical("runs/run_manifest_v8.json")
    assert not _is_protected_historical("README_V9.md")
    assert not _is_protected_historical("02_CODE/src/trustdive/pipeline.py")


def test_contract_uses_review_risk_not_injury_risk():
    contract = load_v7_contract()
    assert contract["terminology"]["high_risk"] == "model-error or judge-disagreement review risk"
    assert any("injury" in item for item in contract["terminology"]["prohibited_claims"])
