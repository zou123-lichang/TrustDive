import numpy as np
import torch

from trustdive.budget import ledger_path
from trustdive.pipeline import parser
from trustdive.v5_counterfactual import _ecdf_scaled
from trustdive.v5_data import v5_paths
from trustdive.v5_modeling import _jitter_labels, cfpd_plus_model_class


def test_v5_cli_contract_is_exposed():
    for command in (
        "audit",
        "build-references",
        "optimize-baseline",
        "build-counterfactuals",
        "pilot",
        "freeze-contract",
        "stress-test",
        "analyze",
        "render-reports",
        "verify",
    ):
        arguments = [command, "--protocol", "v5"]
        if command == "analyze":
            arguments.extend(["--part", "all"])
        parsed = parser().parse_args(arguments)
        assert parsed.protocol == "v5"
    parsed = parser().parse_args(["train", "--protocol", "v5", "--stage", "final"])
    assert parsed.stage == "final"


def test_v5_has_independent_gpu_ledger():
    assert ledger_path(v5_paths()).name == "v5_gpu_budget_ledger.json"


def test_v5_matched_model_has_no_hidden_residual():
    Model = cfpd_plus_model_class(input_dim=20, hidden=16, dropout=0.0)
    model = Model().eval()
    pair = torch.randn(7, 5, 3, 20)
    base = torch.linspace(5.0, 8.0, 7)
    with torch.no_grad():
        prediction, contribution, per_reference = model(pair, base)
    torch.testing.assert_close(prediction, base + contribution.sum(dim=1), atol=1e-6, rtol=0)
    assert per_reference.shape == (7, 5, 3)


def test_v5_reliability_scaling_is_bounded_and_monotonic():
    reference = np.asarray([0.0, 0.1, 0.2, 0.3, 2.0])
    values = np.asarray([0.0, 0.1, 0.2, 1.0])
    scaled = _ecdf_scaled(values, reference, 0.99)
    assert np.all((scaled >= 0.0) & (scaled <= 1.0))
    assert np.all(np.diff(scaled) >= 0.0)


def test_v5_boundary_jitter_is_deterministic_and_ordered():
    labels = np.tile(np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2]), (12, 1))
    first = _jitter_labels(labels, 20260817)
    second = _jitter_labels(labels, 20260817)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first, axis=1) >= 0)
    assert all(set(row.tolist()) == {0, 1, 2} for row in first)
