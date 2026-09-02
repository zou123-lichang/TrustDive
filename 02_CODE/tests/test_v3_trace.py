import numpy as np
import torch

from trustdive.pipeline import parser
from trustdive.v3_modeling import _trace_model_class, attribution_coverage
from trustdive.v3_stress import _boundary_variant, _token_drop_variant


def test_trace_student_exact_additive_reconstruction_and_zero_residual():
    Model = _trace_model_class(phase_dim=5, hidden=8, residual_bound=0.0)
    model = Model().eval()
    phase = torch.randn(4, 3, 5)
    base = torch.tensor([6.0, 7.0, 8.0, 9.0])
    with torch.no_grad():
        prediction, contribution, residual, sigma = model(phase, base)
    torch.testing.assert_close(prediction, base + contribution.sum(dim=1), atol=1e-6, rtol=0)
    assert torch.count_nonzero(residual) == 0
    assert torch.all(sigma > 0)


def test_bounded_trace_residual_never_exceeds_contract():
    Model = _trace_model_class(phase_dim=4, hidden=8, residual_bound=0.25)
    model = Model().eval()
    with torch.no_grad():
        _, _, residual, _ = model(torch.randn(30, 3, 4) * 100, torch.zeros(30))
    assert torch.all(residual <= 0.25)
    assert torch.all(residual >= -0.25)


def test_attribution_coverage_formula():
    contributions = np.asarray([[1.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    residual = np.asarray([0.5, 0.0])
    coverage = attribution_coverage(contributions, residual)
    np.testing.assert_allclose(coverage, [0.8, 1.0])


def test_stress_variants_are_deterministic_and_shape_preserving():
    delta = np.ones((2, 3, 4), dtype=np.float32)
    shifted_a = _boundary_variant(delta, 0, 1)
    shifted_b = _boundary_variant(delta, 0, 1)
    np.testing.assert_array_equal(shifted_a, shifted_b)
    labels = np.asarray([[0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 1, 1, 2, 2, 2, 2]])
    dropped = _token_drop_variant(delta, labels, 0)
    assert dropped.shape == delta.shape
    assert np.all(dropped[:, 0] < delta[:, 0])


def test_v3_cli_contract_is_exposed():
    parsed = parser().parse_args(["pilot-trace", "--protocol", "v3"])
    assert parsed.command == "pilot-trace"
    assert parsed.protocol == "v3"
    parsed = parser().parse_args(["train", "--protocol", "v3", "--stage", "final"])
    assert parsed.protocol == "v3"
