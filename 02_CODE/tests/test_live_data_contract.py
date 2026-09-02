from collections import Counter

from trustdive.config import Paths
from trustdive.data import build_manifest
from trustdive.features import PoseDiveDataset


def test_live_manifest_contract():
    frame = build_manifest(Paths())
    assert len(frame) == 3000
    assert Counter(frame.official_split) == {"train": 2251, "test": 749}
    assert Counter(frame.judge_count) == {7: 1369, 3: 1024, 5: 607}
    assert ((frame.official_split == "test") & (frame.judge_count == 7)).sum() == 325
    assert frame.splash_exists.all()


def test_posedive_topdown_crop_contains_visible_joints():
    dataset = PoseDiveDataset(Paths().pose_dive, "test")
    image, joints, visible = dataset[0]
    assert tuple(image.shape) == (3, 192, 256)
    selected = joints[visible.bool()]
    assert bool(((selected >= 0.0) & (selected <= 1.0)).all())
