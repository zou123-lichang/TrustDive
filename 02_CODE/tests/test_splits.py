import pandas as pd

from trustdive.splits import grouped_partition


def test_grouped_partition_never_splits_event_family():
    rows = []
    for family in range(12):
        for action in ("107b", "207c"):
            rows.append(
                {
                    "event_family": f"F{family}",
                    "action_type": action,
                    "judge_count": 7 if family % 2 else 5,
                    "dive_score": 50 + family,
                }
            )
    frame = pd.DataFrame(rows)
    roles = grouped_partition(frame, (0.7, 0.15, 0.15), ("fit", "validation", "calibration"), 20260815, 200)
    for family, group in frame.assign(role=roles).groupby("event_family"):
        assert group.role.nunique() == 1
    assert set(roles) == {"fit", "validation", "calibration"}
