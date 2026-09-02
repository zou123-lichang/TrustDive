from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from trustdive.pipeline import parser
from trustdive.v4_counterfactual import exact_three_phase_shapley
from trustdive.v8_analysis import _pareto_select
from trustdive.v8_data import _conditional_model, _is_protected_historical, load_v8_contract
from trustdive.v8_modeling import PhaseConflictNetwork, _pairwise_rank_loss
from trustdive.v8_tokens import TOKEN_NAMES


def test_v8_cli_contract_contains_all_public_commands():
    expected = (
        ["audit", "--protocol", "v8"],
        ["build-conditional-risk", "--protocol", "v8"],
        ["build-phase-tokens", "--protocol", "v8"],
        ["train-baselines", "--protocol", "v8"],
        ["pilot", "--protocol", "v8"],
        ["freeze-contract", "--protocol", "v8"],
        ["train", "--protocol", "v8", "--stage", "final"],
        ["build-dual-evidence", "--protocol", "v8"],
        ["analyze-review", "--protocol", "v8"],
        ["render-reports", "--protocol", "v8"],
        ["verify", "--protocol", "v8"],
    )
    for argv in expected:
        assert parser().parse_args(argv).protocol == "v8"


def test_conditional_model_handles_scores_actions_and_difficulty():
    frame = pd.DataFrame({
        "predicted_quality": np.linspace(3.0, 9.0, 20),
        "action_type": [f"a{i % 4}" for i in range(20)],
        "difficulty": np.linspace(2.0, 4.0, 20),
    })
    target = -0.2 * frame.predicted_quality.to_numpy() + 0.1 * frame.difficulty.to_numpy()
    model = _conditional_model(load_v8_contract())
    model.fit(frame.iloc[:15], target[:15])
    prediction = model.predict(frame.iloc[15:])
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()


def test_pareto_queue_has_exact_budget_and_forces_open_set():
    error = np.linspace(0.0, 1.0, 20)
    dispute = error[::-1]
    force = np.zeros(20, dtype=bool)
    force[[4, 5]] = True
    selected = _pareto_select(error, dispute, 0.20, 0.10, force)
    assert selected.sum() == 4
    assert selected[4] and selected[5]


def test_phase_conflict_network_outputs_all_three_tasks_and_bounded_delta():
    mean = np.zeros((1, 1, 1, len(TOKEN_NAMES)), dtype=np.float32)
    scale = np.ones_like(mean)
    model = PhaseConflictNetwork(len(TOKEN_NAMES), 32, 2, 0.0, mean, scale)
    token = torch.randn(4, 5, 3, len(TOKEN_NAMES))
    weight = torch.full((4, 5), 0.2)
    base = torch.full((4,), 6.0)
    result = model.module(token, weight, base, torch.ones(4, 3))
    assert set(result) == {"quality", "score_delta", "log_excess_scale", "excess_logit", "error_logit"}
    assert torch.all(torch.abs(result["score_delta"]) <= 0.250001)


def test_masked_network_and_shapley_exactly_reconstruct_outputs():
    mean = np.zeros((1, 1, 1, len(TOKEN_NAMES)), dtype=np.float32)
    scale = np.ones_like(mean)
    model = PhaseConflictNetwork(len(TOKEN_NAMES), 32, 2, 0.0, mean, scale)
    model.module.eval()
    token = torch.randn(1, 5, 3, len(TOKEN_NAMES))
    weight = torch.full((1, 5), 0.2)
    values = []
    with torch.inference_mode():
        for mask in range(8):
            phase_mask = torch.tensor([[float(bool(mask & (1 << p))) for p in range(3)]])
            values.append(float(model.module(token, weight, torch.tensor([5.0]), phase_mask)["quality"]))
    coalition = np.asarray(values)
    phi = exact_three_phase_shapley(coalition[None, None])[0, 0]
    assert abs(coalition[0] + phi.sum() - coalition[7]) < 1e-6


def test_student_t_distribution_is_finite_for_seven_judges():
    distribution = torch.distributions.StudentT(
        df=torch.tensor(4.0), loc=torch.tensor([[7.0]]), scale=torch.tensor([[0.5]])
    )
    judges = torch.tensor([[6.5, 7.0, 7.5, 7.0, 6.5, 7.5, 7.0]])
    assert torch.isfinite(-distribution.log_prob(judges).mean())


def test_rank_loss_rewards_correct_excess_order():
    target = torch.tensor([0.0, 1.0, 2.0])
    correct = _pairwise_rank_loss(torch.tensor([0.0, 1.0, 2.0]), target)
    reversed_loss = _pairwise_rank_loss(torch.tensor([2.0, 1.0, 0.0]), target)
    assert correct < reversed_loss


def test_historical_protection_allows_v8_only():
    assert _is_protected_historical("03_RESULTS/V7_RISK_TASK/x.json")
    assert _is_protected_historical("02_CODE/src/trustdive/v7_analysis.py")
    assert not _is_protected_historical("03_RESULTS/V8_PHASE_CONFLICT/x.json")
    assert not _is_protected_historical("02_CODE/src/trustdive/v8_analysis.py")
    assert not _is_protected_historical("02_CODE/src/trustdive/pipeline.py")


def test_contract_prevents_injury_and_psychological_causal_claims():
    prohibited = load_v8_contract()["terminology"]["prohibited_claims"]
    assert any("injury" in item for item in prohibited)
    assert any("psychological" in item for item in prohibited)


def test_contract_has_fixed_three_seeds_and_two_hour_budget():
    contract = load_v8_contract()
    assert contract["statistics"]["model_seeds"] == [20260821, 20260822, 20260823]
    assert contract["compute"]["gpu_budget_hours"] == 2.0
