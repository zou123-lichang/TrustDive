from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import Paths, load_contract
from .metrics import interval_coverage, score_metrics
from .statistics import (
    bootstrap_median_ci,
    learn_fusion_weight,
    panel_rows,
    sign_flip_permutation,
)
from .util import write_json


def baseline_gate(paths: Paths | None = None) -> dict:
    paths = paths or Paths()
    contract = load_contract(paths.contract)
    path = paths.results / "02_SCORE" / "relative_summary.json"
    if not path.exists():
        return {"status": "NOT_RUN", "reason": str(path)}
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    gate = contract["gates"]["baseline"]
    checks = {
        "spearman": summary["spearman_mean"] >= float(gate["min_spearman"]),
        "seed_sd": summary["spearman_sd"] <= float(gate["max_seed_sd"]),
        "test_n": summary["official_test_n"] == 749,
        "predicted_phases": summary.get("formal_phase_source") == "predicted_monotonic_labels",
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "summary": summary}
    write_json(paths.results / "02_SCORE" / "baseline_gate.json", result)
    return result


def pilot_gate(paths: Paths | None = None) -> dict:
    paths = paths or Paths()
    contract = load_contract(paths.contract)
    relative_path = paths.results / "02_SCORE" / "relative_summary.json"
    pilot_path = paths.results / "02_SCORE" / "pilot_summary.json"
    if not relative_path.exists() or not pilot_path.exists():
        return {"status": "NOT_RUN", "missing": [str(x) for x in (relative_path, pilot_path) if not x.exists()]}
    relative = json.loads(relative_path.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    gate = contract["gates"]["pilot"]
    rel_l2 = float(np.mean([x["relative_l2"] for x in relative["seed_metrics"]]))
    pilot_l2 = float(np.mean([x["relative_l2"] for x in pilot["seed_metrics"]]))
    degradation = (pilot_l2 - rel_l2) / max(rel_l2, 1e-12)
    intervention = pilot.get("intervention_audit", {})
    checks = {
        "delta_spearman": pilot["spearman_mean"] - relative["spearman_mean"] >= float(gate["min_delta_spearman"]),
        "relative_l2_degradation": degradation <= float(gate["max_relative_l2_degradation_fraction"]),
        "trace_coverage": pilot["median_trace_coverage"] >= float(gate["min_median_trace_coverage"]),
        "targeted_deletion": bool(intervention.get("targeted_greater_than_random", False)),
        "splash_stability": float(intervention.get("nonentry_rank_stability_without_splash", -1))
        >= float(gate["min_nonentry_rank_stability_without_splash"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "delta_spearman": pilot["spearman_mean"] - relative["spearman_mean"],
        "relative_l2_degradation_fraction": degradation,
        "pilot": pilot,
    }
    write_json(paths.results / "02_SCORE" / "pilot_gate.json", result)
    return result


def analyze_score(paths: Paths | None = None) -> dict:
    paths = paths or Paths()
    prediction_path = paths.results / "02_SCORE" / "final_predictions.parquet"
    if not prediction_path.exists():
        prediction_path = paths.results / "02_SCORE" / "pilot_predictions.parquet"
    if not prediction_path.exists():
        result = {"status": "NOT_RUN", "reason": "No pilot or final predictions"}
        write_json(paths.results / "02_SCORE" / "score_analysis.json", result)
        return result
    prediction = pd.read_parquet(prediction_path)
    test = prediction[prediction.analysis_role == "official_test"].copy()
    metrics = score_metrics(test.dive_score, test.predicted_score)
    quality_true = test.dive_score.to_numpy() / (3.0 * test.difficulty.to_numpy())
    coverage = interval_coverage(quality_true, test.lower_quality, test.upper_quality)
    result = {
        "status": "ANALYZED",
        "prediction_file": str(prediction_path),
        "n": len(test),
        "metrics": metrics,
        "quality_interval_coverage": coverage,
        "open_set_n": int(test.open_set.sum()),
        "claim_boundary": "Full official test scoring; not cross-dataset generalization.",
    }
    write_json(paths.results / "02_SCORE" / "score_analysis.json", result)
    return result


def _representative_trace_sample(test: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    frame = test.copy()
    frame["score_quartile"] = pd.qcut(frame.dive_score.rank(method="first"), 4, labels=False)
    frame["difficulty_tertile"] = pd.qcut(frame.difficulty.rank(method="first"), 3, labels=False)
    # Greedy marginal-preserving selection: sample proportional numbers from
    # action group and panel-size strata, while retaining score/difficulty labels.
    selected = []
    for _, group in frame.groupby(["action_group", "judge_count"]):
        take = max(1, round(n * len(group) / len(frame)))
        selected.append(group.sample(min(take, len(group)), random_state=seed))
    output = pd.concat(selected).drop_duplicates("clip_uid")
    if len(output) > n:
        output = output.sample(n, random_state=seed)
    elif len(output) < n:
        remaining = frame[~frame.clip_uid.isin(output.clip_uid)].sample(n - len(output), random_state=seed)
        output = pd.concat([output, remaining])
    output = output.copy()
    output["trace_group"] = "representative"
    output["sampling_weight"] = len(frame) / len(output)
    return output


def build_trace300(manifest: pd.DataFrame, paths: Paths | None = None) -> pd.DataFrame:
    paths = paths or Paths()
    gate = pilot_gate(paths)
    if gate.get("status") != "PASS":
        raise RuntimeError("Trace-300 is locked until the pilot gate passes")
    prediction_path = paths.results / "02_SCORE" / "final_predictions.parquet"
    if not prediction_path.exists():
        prediction_path = paths.results / "02_SCORE" / "pilot_predictions.parquet"
    prediction = pd.read_parquet(prediction_path)
    merged = manifest.merge(prediction[["clip_uid", "predicted_quality", "ensemble_sd", "upper_quality", "lower_quality"]], on="clip_uid")
    test = merged[merged.official_split == "test"].copy()
    representative = _representative_trace_sample(test, 240, 20260815)
    remaining = test[~test.clip_uid.isin(representative.clip_uid)].copy()
    remaining["uncertainty"] = remaining.upper_quality - remaining.lower_quality
    uncertainty = remaining.nlargest(30, "uncertainty").copy()
    uncertainty["trace_group"] = "challenge_uncertainty"
    uncertainty["sampling_weight"] = np.nan
    remaining = remaining[~remaining.clip_uid.isin(uncertainty.clip_uid)].copy()
    remaining["panel_disagreement"] = remaining.judge_scores_json.map(lambda x: float(np.std(json.loads(x))))
    remaining["human_ai_disagreement"] = abs(
        remaining.predicted_quality
        - remaining.judge_scores_json.map(lambda x: float(np.median(json.loads(x))))
    )
    disagreement = remaining.nlargest(30, "human_ai_disagreement").copy()
    disagreement["trace_group"] = "challenge_disagreement"
    disagreement["sampling_weight"] = np.nan
    output = pd.concat([representative, uncertainty, disagreement], ignore_index=True)
    output["repeat_annotation"] = False
    repeat_indices = output[output.trace_group == "representative"].sample(60, random_state=20260815).index
    output.loc[repeat_indices, "repeat_annotation"] = True
    destination = paths.results / "03_TRACE" / "trace300_manifest.csv"
    output.to_csv(destination, index=False)
    return output


def analyze_trace(paths: Paths | None = None) -> dict:
    paths = paths or Paths()
    annotation_path = paths.results / "03_TRACE" / "trace300_annotations.csv"
    if not annotation_path.exists():
        result = {
            "status": "PENDING_MANUAL_ANNOTATION",
            "required_file": str(annotation_path),
            "prohibited": "No trace accuracy or expert-utility claim may be made before this file exists.",
        }
        write_json(paths.results / "03_TRACE" / "trace_analysis.json", result)
        return result
    annotations = pd.read_csv(annotation_path)
    required = {"clip_uid", "pose_pck", "angle_mae_deg", "splash_dice", "phase_boundary_error"}
    if not required <= set(annotations):
        raise ValueError(f"Trace annotations missing columns: {required - set(annotations)}")
    result = {
        "status": "ANALYZED",
        "n": len(annotations),
        "pose_pck_mean": float(annotations.pose_pck.mean()),
        "angle_mae_median_deg": float(annotations.angle_mae_deg.median()),
        "splash_dice_median": float(annotations.splash_dice.median()),
        "phase_boundary_error_median": float(annotations.phase_boundary_error.median()),
    }
    write_json(paths.results / "03_TRACE" / "trace_analysis.json", result)
    return result


def analyze_panel(manifest: pd.DataFrame, paths: Paths | None = None) -> dict:
    paths = paths or Paths()
    contract = load_contract(paths.contract)
    prediction_path = paths.results / "02_SCORE" / "final_predictions.parquet"
    if not prediction_path.exists():
        prediction_path = paths.results / "02_SCORE" / "pilot_predictions.parquet"
    if not prediction_path.exists():
        result = {"status": "NOT_RUN", "reason": "No TrustDive predictions"}
        write_json(paths.results / "04_PANEL" / "panel_analysis.json", result)
        return result
    predictions = pd.read_parquet(prediction_path)
    calibration = panel_rows(predictions, manifest, "calibration")
    test = panel_rows(predictions, manifest, "official_test")
    if calibration.empty or test.empty:
        raise RuntimeError("Calibration or official seven-judge test panel is empty")
    weight = learn_fusion_weight(calibration)
    test["fused_score"] = weight * test.judge_score + (1.0 - weight) * test.ai_score
    test["human_error"] = abs(test.judge_score - test.consensus)
    test["ai_error"] = abs(test.ai_score - test.consensus)
    test["fused_error"] = abs(test.fused_score - test.consensus)
    clip = test.groupby("clip_uid")[["human_error", "ai_error", "fused_error", "interval_width"]].median()
    difference = clip.fused_error - clip.human_error
    iterations = int(contract["statistics"]["permutations"])
    bootstrap = int(contract["statistics"]["bootstrap"])
    seed = int(contract["random"]["master_seed"])
    ci = bootstrap_median_ci(difference, bootstrap, seed)
    p_value = sign_flip_permutation(difference, iterations, seed)
    human_median = float(clip.human_error.median())
    fused_median = float(clip.fused_error.median())
    reduction = (human_median - fused_median) / max(human_median, 1e-12)
    accepted = clip.nsmallest(round(0.8 * len(clip)), "interval_width")
    accepted_reduction = (
        float(clip.fused_error.mean()) - float(accepted.fused_error.mean())
    ) / max(float(clip.fused_error.mean()), 1e-12)
    gate = contract["gates"]["panel"]
    checks = {
        "n_325": len(clip) == 325,
        "relative_error_reduction": reduction >= float(gate["min_relative_error_reduction"]),
        "ci_below_zero": ci[1] < 0,
        "permutation_p": p_value < 0.05,
        "selective_reduction": accepted_reduction
        >= float(gate["min_selective_error_reduction_at_80pct_coverage"]),
    }
    test.to_parquet(paths.results / "04_PANEL" / "panel_simulation.parquet", index=False)
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "fusion_weight_human": weight,
        "clips": len(clip),
        "human_median_absolute_error": human_median,
        "fused_median_absolute_error": fused_median,
        "relative_error_reduction": reduction,
        "median_difference": float(np.median(difference)),
        "bootstrap_95_ci": ci,
        "permutation_p": p_value,
        "selective_error_reduction_at_80pct_coverage": accepted_reduction,
        "claim_boundary": "Retrospective judge-panel simulation; no real human-AI interaction.",
    }
    write_json(paths.results / "04_PANEL" / "panel_analysis.json", result)
    return result
