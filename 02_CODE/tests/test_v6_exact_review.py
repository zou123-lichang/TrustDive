import subprocess

import numpy as np
import pandas as pd

from trustdive.budget import ledger_path
from trustdive.config import PROJECT_ROOT
from trustdive.pipeline import parser
from trustdive.v4_counterfactual import exact_three_phase_shapley
from trustdive.v5_teacher import _v5_execution_quality
from trustdive.v6_data import load_v6_contract, v6_paths
from trustdive.v6_modeling import (
    V6Assets,
    make_reference_features,
    predict_selected_v6,
)


def test_v6_cli_contract_is_exposed():
    for command in (
        "audit", "extract-latents", "optimize-adapter", "build-attributions",
        "pilot", "freeze-contract", "evaluate-final", "analyze-review",
        "render-reports", "verify",
    ):
        parsed = parser().parse_args([command, "--protocol", "v6"])
        assert parsed.protocol == "v6"


def test_v6_has_independent_two_hour_gpu_ledger():
    assert ledger_path(v6_paths()).name == "v6_gpu_budget_ledger.json"
    assert load_v6_contract()["compute"]["gpu_budget_hours"] == 2.0


def test_v6_official_test_is_initially_locked():
    state = load_v6_contract()["state"]
    assert state["frozen"] is False
    assert state["official_test_unlocked"] is False


def test_rica2_execution_quality_scale_is_divided_by_three():
    raw = np.asarray([[[18.0]], [[24.0]]], dtype=np.float32)
    scaled = _v5_execution_quality(raw)
    np.testing.assert_allclose(scaled, np.asarray([6.0, 8.0], dtype=np.float32))


def test_exact_shapley_reconstructs_interacting_scorer():
    # f(S)=1 + individual effects + takeoff-flight interaction.
    values = np.zeros((2, 1, 8), dtype=float)
    effects = np.asarray([0.4, -0.2, 0.7])
    for mask in range(8):
        active = np.asarray([(mask >> p) & 1 for p in range(3)])
        values[:, 0, mask] = 1.0 + active @ effects + 0.3 * active[0] * active[1]
    phi = exact_three_phase_shapley(values)
    np.testing.assert_allclose(values[:, :, 0] + phi.sum(axis=2), values[:, :, 7], atol=1e-10)
    np.testing.assert_allclose(phi[0, 0], [0.55, -0.05, 0.7], atol=1e-10)


def _synthetic_assets():
    frame = pd.DataFrame({
        "clip_uid": ["a", "b", "c"],
        "execution_quality": [5.0, 6.0, 7.0],
        "difficulty": [2.0, 2.0, 2.0],
        "action_type": ["x", "x", "x"],
    })
    latent = np.arange(3 * 256, dtype=np.float32).reshape(3, 256) / 1000.0
    return V6Assets(frame, latent, np.zeros((3, 29, 512), np.float32), np.ones((3, 29), np.float32), np.asarray([4.5, 5.5, 6.5], np.float32))


def test_reference_feature_contract_is_1029_dimensions():
    assets = _synthetic_assets()
    refs = {
        "references": np.asarray([[1, 2, 1, 2, 1], [0, 2, 0, 2, 0], [0, 1, 0, 1, 0]], dtype=np.int32),
        "distances": np.full((3, 5), 0.2, dtype=np.float32),
    }
    features = make_reference_features(assets, refs)
    assert features.shape == (3, 5, 1029)
    assert np.isfinite(features).all()


def test_open_set_falls_back_to_deterministic_teacher():
    from sklearn.linear_model import LinearRegression

    assets = _synthetic_assets()
    model = LinearRegression().fit(assets.teacher_quality[:, None], assets.teacher_quality + 1.0)
    refs = {
        "open_set": np.asarray([False, True, False]),
        "references": np.zeros((3, 5), dtype=np.int32),
        "weights": np.full((3, 5), 0.2, dtype=np.float32),
    }
    selected = {"selected": {"model_type": "linear"}}
    prediction, _ = predict_selected_v6(assets, refs, selected, model)
    assert prediction[1] == assets.teacher_quality[1]
    assert prediction[0] != assets.teacher_quality[0]


def test_v1_to_v5_evidence_anchors_are_unchanged_from_v6_anchor():
    anchor = load_v6_contract()["read_only_anchors"]["repository_commit"]
    paths = [
        value for key, value in load_v6_contract()["read_only_anchors"].items()
        if key != "repository_commit"
    ]
    completed = subprocess.run(
        ["git", "diff", "--quiet", anchor, "--", *paths],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert completed.returncode == 0
