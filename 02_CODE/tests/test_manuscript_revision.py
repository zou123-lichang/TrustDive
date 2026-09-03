from __future__ import annotations

import numpy as np
import pandas as pd

from trustdive.manuscript_revision import (
    _annotation_boundaries,
    _cluster_bootstrap_spearman,
    _pairwise_interactions,
    _reference_map_k,
    _transition_positions,
)
from trustdive.v4_counterfactual import exact_three_phase_shapley


def test_annotation_boundaries_follow_finediving_first_and_last_transitions() -> None:
    assert _annotation_boundaries(128, "[35,61,70]") == (35, 70)
    assert _annotation_boundaries(85, "[2,34]") == (2, 34)


def test_transition_positions_require_monotonic_three_phase_labels() -> None:
    assert _transition_positions(np.asarray([0, 0, 1, 1, 2, 2])) == (2, 4)


def test_pairwise_interactions_detect_only_the_injected_pair() -> None:
    values = []
    for mask in range(8):
        x0 = bool(mask & 1)
        x1 = bool(mask & 2)
        x2 = bool(mask & 4)
        values.append(float(x0) + 2.0 * float(x1) + 3.0 * float(x2) + 4.0 * float(x0 and x1))
    interactions = _pairwise_interactions(np.asarray([values]))
    np.testing.assert_allclose(interactions, [[4.0, 0.0, 0.0]], atol=1e-12)


def test_cluster_bootstrap_spearman_preserves_perfect_monotonic_relation() -> None:
    left = np.arange(1.0, 9.0)
    groups = np.repeat(np.asarray(["a", "b", "c", "d"]), 2)
    interval = _cluster_bootstrap_spearman(
        left, 2.0 * left, groups, iterations=100, seed=7
    )
    np.testing.assert_allclose(interval, [1.0, 1.0], atol=1e-12)


def test_shapley_reconstructs_interacting_coalition() -> None:
    coalition = np.asarray([[0.0, 1.0, 2.0, 7.0, 3.0, 4.0, 5.0, 10.0]])
    phi = exact_three_phase_shapley(coalition[:, None, :])[:, 0, :]
    np.testing.assert_allclose(coalition[:, 0] + phi.sum(axis=1), coalition[:, 7])


def test_reference_map_k_preserves_requested_slots_and_three_reference_gate() -> None:
    class Assets:
        teacher_quality = np.asarray([4.8, 5.1, 5.4, 5.7])
        frame = pd.DataFrame({"execution_quality": [5.0, 5.0, 5.5, 6.0]})

    base = {
        "references": np.asarray([[1, 2, 3], [0, 2, -1], [0, 1, 3], [0, 1, 2]]),
        "distances": np.asarray([[0.1, 0.2, 0.3], [0.1, 0.2, np.nan], [0.2, 0.3, 0.4], [0.1, 0.2, 0.5]]),
    }
    mapping = _reference_map_k(Assets(), base, 3)
    assert mapping["references"].shape == (4, 3)
    assert mapping["open_set"].tolist() == [False, True, False, False]
    np.testing.assert_allclose(mapping["weights"].sum(axis=1), 1.0)
