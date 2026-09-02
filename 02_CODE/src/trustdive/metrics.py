from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def execution_quality(score, difficulty):
    score = np.asarray(score, dtype=float)
    difficulty = np.asarray(difficulty, dtype=float)
    if np.any(difficulty <= 0):
        raise ValueError("Difficulty must be positive")
    return score / (3.0 * difficulty)


def restore_total_score(quality, difficulty):
    return 3.0 * np.asarray(difficulty, dtype=float) * np.asarray(quality, dtype=float)


def score_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be same-length vectors")
    rho = float(spearmanr(y_true, y_pred).statistic)
    error = y_pred - y_true
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    relative_l2 = float(np.sum(error**2) / denominator) if denominator > 0 else float("nan")
    return {
        "spearman": rho,
        "relative_l2": relative_l2,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def aqa_score_metrics(y_true, y_pred) -> dict[str, float | str]:
    """FineDiving/RICA2 metrics with the published range-normalized Relative-L2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result = score_metrics(y_true, y_pred)
    result["relative_l2_variance_ratio_internal"] = result.pop("relative_l2")
    score_range = float(np.max(y_true) - np.min(y_true))
    result["relative_l2"] = (
        float(np.mean(((y_pred - y_true) / score_range) ** 2) * 100.0)
        if score_range > 0
        else float("nan")
    )
    result["relative_l2_definition"] = "100 * mean(((prediction-target)/(max(target)-min(target)))^2)"
    return result


def pck(pred_xy, true_xy, visible, threshold: float = 0.1) -> float:
    pred = np.asarray(pred_xy, dtype=float)
    true = np.asarray(true_xy, dtype=float)
    vis = np.asarray(visible, dtype=bool)
    if pred.shape != true.shape or pred.shape[-1] != 2 or vis.shape != pred.shape[:-1]:
        raise ValueError("Invalid PCK shapes")
    scores = []
    for p, t, v in zip(pred, true, vis):
        if not np.any(v):
            continue
        extent = np.ptp(t[v], axis=0)
        scale = max(float(np.max(extent)), 1e-8)
        distance = np.linalg.norm(p[v] - t[v], axis=1)
        scores.extend((distance <= threshold * scale).tolist())
    return float(np.mean(scores)) if scores else float("nan")


def angle_degrees(a, b, c) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    u = a - b
    v = c - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom <= 1e-12:
        return float("nan")
    cosine = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def interval_coverage(y_true, lower, upper) -> float:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))
