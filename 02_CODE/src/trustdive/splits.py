from __future__ import annotations

import numpy as np
import pandas as pd


def _distribution_distance(reference: pd.DataFrame, subset: pd.DataFrame) -> float:
    if subset.empty:
        return 1e9
    score = 0.0
    for column in ("action_type", "judge_count"):
        ref = reference[column].value_counts(normalize=True)
        cur = subset[column].value_counts(normalize=True)
        labels = ref.index.union(cur.index)
        score += float(np.abs(ref.reindex(labels, fill_value=0) - cur.reindex(labels, fill_value=0)).sum())
    ref_bins = pd.qcut(reference.dive_score.rank(method="first"), 4, labels=False)
    cuts = np.quantile(reference.dive_score, [0.25, 0.5, 0.75])
    cur_bins = np.digitize(subset.dive_score, cuts)
    score += float(
        np.abs(
            np.bincount(ref_bins, minlength=4) / len(ref_bins)
            - np.bincount(cur_bins, minlength=4) / len(cur_bins)
        ).sum()
    )
    return score


def grouped_partition(
    frame: pd.DataFrame,
    proportions: tuple[float, float, float],
    labels: tuple[str, str, str],
    seed: int,
    attempts: int = 10000,
) -> pd.Series:
    if not np.isclose(sum(proportions), 1.0):
        raise ValueError("Split proportions must sum to one")
    groups = np.asarray(sorted(frame.event_family.unique()))
    n_first = max(1, round(len(groups) * proportions[0]))
    n_second = max(1, round(len(groups) * proportions[1]))
    rng = np.random.default_rng(seed)
    best = None
    best_score = float("inf")
    all_actions = set(frame.action_type)
    for _ in range(attempts):
        order = rng.permutation(groups)
        group_sets = (
            set(order[:n_first]),
            set(order[n_first : n_first + n_second]),
            set(order[n_first + n_second :]),
        )
        subsets = [frame[frame.event_family.isin(group_set)] for group_set in group_sets]
        if any(subset.empty for subset in subsets):
            continue
        missing_from_fit = len((set(subsets[1].action_type) | set(subsets[2].action_type)) - set(subsets[0].action_type))
        score = 1000.0 * missing_from_fit
        score += _distribution_distance(frame, subsets[1]) + _distribution_distance(frame, subsets[2])
        target_sizes = np.asarray(proportions) * len(frame)
        score += float(np.abs(np.asarray([len(x) for x in subsets]) - target_sizes).sum() / len(frame))
        if score < best_score:
            best_score = score
            best = group_sets
            if missing_from_fit == 0 and score < 0.5:
                break
    if best is None:
        raise RuntimeError("Unable to construct grouped split")
    output = pd.Series(index=frame.index, dtype="object")
    for label, group_set in zip(labels, best):
        output.loc[frame.event_family.isin(group_set)] = label
    if output.isna().any():
        raise AssertionError("Grouped split did not assign every row")
    return output


def add_analysis_roles(manifest: pd.DataFrame, seed: int, attempts: int = 10000) -> pd.DataFrame:
    frame = manifest.copy()
    frame["analysis_role"] = "official_test"
    train = frame[frame.official_split == "train"].copy()
    roles = grouped_partition(
        train,
        (0.70, 0.15, 0.15),
        ("fit", "validation", "calibration"),
        seed,
        attempts,
    )
    frame.loc[train.index, "analysis_role"] = roles
    return frame


def source_heldout_roles(manifest: pd.DataFrame, seed: int, attempts: int = 10000) -> pd.Series:
    return grouped_partition(
        manifest,
        (0.80, 0.10, 0.10),
        ("source_fit", "source_validation", "source_test"),
        seed,
        attempts,
    )
