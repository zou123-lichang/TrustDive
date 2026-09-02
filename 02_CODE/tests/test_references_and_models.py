import numpy as np
import pandas as pd
import torch

from trustdive.modeling import _cosine_neighbors, _torch_models


def test_reference_retrieval_excludes_same_family_and_other_action():
    frame = pd.DataFrame(
        {
            "clip_uid": ["q", "same", "good1", "good2", "other"],
            "action_type": ["107b", "107b", "107b", "107b", "207c"],
            "event_family": ["A", "A", "B", "C", "D"],
        }
    )
    features = np.asarray([[1, 0], [1, 0], [0.9, 0.1], [0.8, 0.2], [1, 0]], dtype=float)
    references = _cosine_neighbors(frame, features, np.arange(len(frame)), 0, 5)
    assert references == [2, 3]


def test_trustdive_residual_is_bounded():
    _, _, TrustDive = _torch_models(rgb_dim=4, concept_dim=6)
    model = TrustDive()
    rgb = torch.full((5, 3, 4), 1000.0)
    concepts = torch.zeros((5, 3, 6))
    base = torch.zeros(5)
    _, contributions, residual = model(rgb, concepts, base)
    assert contributions.shape == (5, 3)
    assert torch.all(residual <= 1.0)
    assert torch.all(residual >= -1.0)
