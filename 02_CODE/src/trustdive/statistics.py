from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


def split_conformal_radius(y_true, y_pred, coverage: float = 0.90) -> float:
    residuals = np.sort(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))
    n = len(residuals)
    if n == 0:
        raise ValueError("Calibration set is empty")
    rank = min(n, int(np.ceil((n + 1) * coverage))) - 1
    return float(residuals[rank])


def leave_one_judge_consensus(judges: list[float]) -> list[dict]:
    values = np.asarray(judges, dtype=float)
    if len(values) != 7:
        raise ValueError("Primary panel simulation requires exactly seven judges")
    rows = []
    for index, value in enumerate(values):
        others = np.delete(values, index)
        rows.append({"judge_index": index, "judge_score": float(value), "consensus": float(np.median(others))})
    return rows


def learn_fusion_weight(calibration: pd.DataFrame) -> float:
    required = {"clip_uid", "judge_score", "ai_score", "consensus"}
    if not required <= set(calibration):
        raise ValueError(f"Missing fusion columns: {required - set(calibration)}")
    best_weight = 0.5
    best_error = float("inf")
    for weight in np.linspace(0.0, 1.0, 101):
        fused = weight * calibration.judge_score.to_numpy() + (1.0 - weight) * calibration.ai_score.to_numpy()
        per_clip = pd.DataFrame(
            {"clip_uid": calibration.clip_uid, "error": np.abs(fused - calibration.consensus.to_numpy())}
        ).groupby("clip_uid").error.median()
        error = float(per_clip.median())
        if error < best_error - 1e-12:
            best_error = error
            best_weight = float(weight)
    return best_weight


def panel_rows(predictions: pd.DataFrame, manifest: pd.DataFrame, role: str) -> pd.DataFrame:
    merged = manifest.merge(predictions, on="clip_uid", validate="one_to_one")
    merged = merged[(merged.judge_count == 7) & (merged.prediction_role == role)].copy()
    rows = []
    for row in merged.itertuples(index=False):
        judges = json.loads(row.judge_scores_json)
        for item in leave_one_judge_consensus(judges):
            rows.append(
                {
                    "clip_uid": row.clip_uid,
                    "judge_index": item["judge_index"],
                    "judge_score": item["judge_score"],
                    "consensus": item["consensus"],
                    "ai_score": float(row.predicted_quality),
                    "interval_width": float(row.upper_quality - row.lower_quality),
                }
            )
    return pd.DataFrame(rows)


def sign_flip_permutation(values, iterations: int, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.median(values)))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        signs = rng.choice((-1.0, 1.0), size=len(values))
        extreme += abs(float(np.median(values * signs))) >= observed
    return float((extreme + 1) / (iterations + 1))


def bootstrap_median_ci(values, iterations: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        estimates[index] = np.median(rng.choice(values, size=len(values), replace=True))
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()
