from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t, wilcoxon
from sklearn.metrics import roc_auc_score

from .metrics import interval_coverage, score_metrics
from .statistics import (
    bootstrap_median_ci,
    learn_fusion_weight,
    leave_one_judge_consensus,
    sign_flip_permutation,
)
from .util import write_json
from .v2_data import V2_RESULTS_ROOT, load_panel_targets, load_v2_contract, require_contract_frozen


def _safe_spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else 0.0


def bootstrap_spearman_ci(a, b, iterations: int, seed: int) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float32)
    for index in range(iterations):
        sample = rng.integers(0, len(a), len(a))
        estimates[index] = _safe_spearman(a[sample], b[sample])
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def _student_t_scores(rows: pd.DataFrame, quantile_samples: int, seed: int) -> tuple[float, float]:
    df = float(load_v2_contract()["model"]["student_t_df"])
    nll: list[float] = []
    crps: list[float] = []
    rng = np.random.default_rng(seed)
    for row in rows.itertuples(index=False):
        judges = np.asarray(json.loads(row.judge_scores_json), dtype=float)
        scale = max(float(row.sigma_judge), 1e-5)
        nll.extend((-student_t.logpdf(judges, df=df, loc=row.predicted_quality, scale=scale)).tolist())
        first = student_t.rvs(
            df=df, loc=row.predicted_quality, scale=scale, size=quantile_samples, random_state=rng
        )
        second = student_t.rvs(
            df=df, loc=row.predicted_quality, scale=scale, size=quantile_samples, random_state=rng
        )
        for value in judges:
            crps.append(float(np.mean(np.abs(first - value)) - 0.5 * np.mean(np.abs(first - second))))
    return float(np.mean(nll)), float(np.mean(crps))


def analyze_score_v2(predictions: pd.DataFrame) -> dict:
    test = predictions[predictions.official_split == "test"].copy()
    baseline_summary = json.loads(
        (V2_RESULTS_ROOT / "03_FINAL" / "final_training_summary_v2.json").read_text(encoding="utf-8")
    )
    metrics = score_metrics(test.dive_score, test.predicted_score)
    seed_metrics = baseline_summary["trustdive_d"]["seed_official_test_scores"]
    result = {
        "status": "ANALYZED",
        "n": int(len(test)),
        "metrics": metrics,
        "seed_metrics": seed_metrics,
        "seed_spearman_mean": float(np.mean([x["spearman"] for x in seed_metrics])),
        "seed_spearman_sd": float(np.std([x["spearman"] for x in seed_metrics], ddof=0)),
        "best_internal_baseline_spearman": float(
            baseline_summary["best_internal_baseline_test_spearman"]
        ),
        "spearman_delta_from_best_internal_baseline": float(
            baseline_summary["trustdive_score_delta"]
        ),
        "interval_coverage_90": interval_coverage(
            test.execution_quality, test.lower_quality, test.upper_quality
        ),
        "open_set_n": int(test.open_set.sum()),
        "source_isolated": baseline_summary.get("source_isolated"),
        "claim_boundary": "Official FineDiving split; not cross-dataset generalization.",
    }
    write_json(V2_RESULTS_ROOT / "04_ANALYSIS" / "score_analysis_v2.json", result)
    return result


def analyze_disagreement_v2(predictions: pd.DataFrame) -> dict:
    contract = load_v2_contract()
    seed = int(contract["random"]["master_seed"])
    iterations = int(contract["statistics"]["bootstrap"])
    primary = predictions[
        (predictions.official_split == "test")
        & predictions.disagreement_primary_eligible.astype(bool)
    ].copy()
    rho = _safe_spearman(primary.sigma_judge, primary.judge_sample_sd)
    ci = bootstrap_spearman_ci(
        primary.sigma_judge, primary.judge_sample_sd, iterations, seed
    )
    pairwise_rho = _safe_spearman(primary.sigma_judge, primary.judge_pairwise_abs)
    high = primary.judge_sample_sd >= primary.judge_sample_sd.quantile(0.75)
    auroc = float(roc_auc_score(high.astype(int), primary.sigma_judge))
    nll, crps = _student_t_scores(
        primary, int(contract["statistics"]["crps_quantile_samples"]), seed
    )
    absolute_error = np.abs(primary.predicted_quality - primary.execution_quality)
    model_error_rho = _safe_spearman(primary.sigma_model, absolute_error)
    two_uncertainties_rho = _safe_spearman(primary.sigma_judge, primary.sigma_model)
    supplemental = predictions[
        (predictions.official_split == "test")
        & (predictions.judge_count == 5)
        & predictions.judge_label_valid.astype(bool)
    ]
    result = {
        "status": "ANALYZED",
        "primary_n": int(len(primary)),
        "judge_sd_spearman": rho,
        "judge_sd_bootstrap_95_ci": ci,
        "pairwise_disagreement_spearman": pairwise_rho,
        "high_disagreement_quartile_auroc": auroc,
        "student_t_nll": nll,
        "student_t_crps": crps,
        "model_uncertainty_absolute_error_spearman": model_error_rho,
        "judge_model_uncertainty_spearman": two_uncertainties_rho,
        "supplemental_five_judge_n": int(len(supplemental)),
        "supplemental_five_judge_spearman": _safe_spearman(
            supplemental.sigma_judge, supplemental.judge_sample_sd
        ),
        "claim_boundary": "Predicted panel disagreement is not a claim of judge error.",
    }
    write_json(V2_RESULTS_ROOT / "04_ANALYSIS" / "disagreement_analysis_v2.json", result)
    return result


def analyze_trace_v2(predictions: pd.DataFrame) -> dict:
    test = predictions[predictions.official_split == "test"].copy()
    contribution_columns = [
        "phase_takeoff_contribution",
        "phase_flight_contribution",
        "phase_entry_contribution",
    ]
    reconstruction = (
        test.base_quality
        + test[contribution_columns].sum(axis=1)
        + test.residual
        - test.predicted_quality
    )
    contribution = test[contribution_columns].to_numpy(dtype=float)
    effects = np.column_stack(
        [
            np.abs(test.predicted_quality - test[f"ablate_{phase}_quality"])
            for phase in ("takeoff", "flight", "entry")
        ]
    )
    top = np.argmax(np.abs(contribution), axis=1)
    bottom = np.argmin(np.abs(contribution), axis=1)
    rng = np.random.default_rng(int(load_v2_contract()["random"]["master_seed"]))
    random_phase = np.asarray(
        [rng.choice([phase for phase in range(3) if phase != selected]) for selected in top]
    )
    row = np.arange(len(test))
    targeted = effects[row, top]
    lowest = effects[row, bottom]
    random_effect = effects[row, random_phase]
    difference = targeted - random_effect
    try:
        p_value = float(wilcoxon(difference, alternative="greater").pvalue)
    except ValueError:
        p_value = 1.0
    result = {
        "status": "ANALYZED",
        "n": int(len(test)),
        "reconstruction_max_abs_error": float(np.max(np.abs(reconstruction))),
        "phase_boundary_error_mean_normalized": float(
            test.phase_boundary_error_normalized.mean()
        ),
        "phase_boundary_error_median_normalized": float(
            test.phase_boundary_error_normalized.median()
        ),
        "targeted_deletion_median_change": float(np.median(targeted)),
        "lowest_deletion_median_change": float(np.median(lowest)),
        "random_deletion_median_change": float(np.median(random_effect)),
        "targeted_minus_random_median": float(np.median(difference)),
        "targeted_greater_than_random_wilcoxon_p": p_value,
        "targeted_deletion_supported": bool(np.median(difference) > 0 and p_value < 0.05),
        "claim_boundary": "Model-attributed phase contributions; not observed judge deductions.",
    }
    write_json(V2_RESULTS_ROOT / "04_ANALYSIS" / "trace_analysis_v2.json", result)
    return result


def _panel_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in frame.itertuples(index=False):
        judges = json.loads(item.judge_scores_json)
        for leave_one in leave_one_judge_consensus(judges):
            rows.append(
                {
                    "clip_uid": item.clip_uid,
                    "analysis_role": item.analysis_role,
                    "judge_index": leave_one["judge_index"],
                    "judge_score": leave_one["judge_score"],
                    "consensus": leave_one["consensus"],
                    "ai_score": float(item.predicted_quality),
                    "review_risk": float(item.review_risk),
                    "review_recommended": bool(item.review_recommended),
                    "sigma_judge": float(item.sigma_judge),
                    "sigma_model": float(item.sigma_model),
                }
            )
    return pd.DataFrame(rows)


def _risk_curve(errors: pd.Series, risks: pd.Series) -> tuple[pd.DataFrame, float]:
    order = np.argsort(risks.to_numpy())
    rows = []
    for coverage in np.linspace(0.1, 1.0, 19):
        count = max(1, int(round(coverage * len(order))))
        rows.append({"coverage": float(coverage), "risk": float(errors.iloc[order[:count]].mean())})
    curve = pd.DataFrame(rows)
    aurc = float(
        np.trapezoid(curve.risk, curve.coverage)
        / (curve.coverage.max() - curve.coverage.min())
    )
    return curve, aurc


def analyze_panel_v2(predictions: pd.DataFrame) -> dict:
    contract = load_v2_contract()
    eligible = predictions[
        predictions.disagreement_primary_eligible.astype(bool)
        & predictions.judge_label_valid.astype(bool)
    ].copy()
    calibration = _panel_rows(eligible[eligible.analysis_role == "calibration"])
    test = _panel_rows(eligible[eligible.analysis_role == "official_test"])
    if calibration.empty or len(test.clip_uid.unique()) != 325:
        raise AssertionError("Primary panel simulation requires calibration data and 325 test clips")
    weight = learn_fusion_weight(calibration)
    for frame in (calibration, test):
        frame["fixed_fused_score"] = 0.5 * frame.judge_score + 0.5 * frame.ai_score
        frame["learned_fused_score"] = weight * frame.judge_score + (1.0 - weight) * frame.ai_score
        frame["human_error"] = np.abs(frame.judge_score - frame.consensus)
        frame["ai_error"] = np.abs(frame.ai_score - frame.consensus)
        frame["fixed_fused_error"] = np.abs(frame.fixed_fused_score - frame.consensus)
        frame["learned_fused_error"] = np.abs(frame.learned_fused_score - frame.consensus)
    clip = test.groupby("clip_uid").agg(
        human_error=("human_error", "median"),
        ai_error=("ai_error", "median"),
        fixed_fused_error=("fixed_fused_error", "median"),
        learned_fused_error=("learned_fused_error", "median"),
        review_risk=("review_risk", "median"),
        review_recommended=("review_recommended", "max"),
        sigma_judge=("sigma_judge", "median"),
        sigma_model=("sigma_model", "median"),
    )
    difference = clip.learned_fused_error - clip.human_error
    seed = int(contract["random"]["master_seed"])
    ci = bootstrap_median_ci(difference, int(contract["statistics"]["bootstrap"]), seed)
    p_value = sign_flip_permutation(
        difference, int(contract["statistics"]["permutations"]), seed
    )
    human_median = float(clip.human_error.median())
    fused_median = float(clip.learned_fused_error.median())
    human_mean = float(clip.human_error.mean())
    fused_mean = float(clip.learned_fused_error.mean())
    fusion_reduction = (
        (human_median - fused_median) / human_median if human_median > 0 else None
    )
    fusion_mean_reduction = (
        (human_mean - fused_mean) / human_mean if human_mean > 0 else None
    )
    accepted = clip.nsmallest(round(0.8 * len(clip)), "review_risk")
    selective_reduction = (
        float(clip.learned_fused_error.mean()) - float(accepted.learned_fused_error.mean())
    ) / max(float(clip.learned_fused_error.mean()), 1e-12)
    rng = np.random.default_rng(seed)
    random_risks = []
    for _ in range(1000):
        selected = rng.choice(len(clip), size=len(accepted), replace=False)
        random_risks.append(float(clip.learned_fused_error.iloc[selected].mean()))
    curve, aurc = _risk_curve(clip.learned_fused_error, clip.review_risk)
    curve.to_csv(V2_RESULTS_ROOT / "04_ANALYSIS" / "coverage_risk_curve_v2.csv", index=False)
    primary_disagreement = predictions[
        (predictions.official_split == "test")
        & predictions.disagreement_primary_eligible.astype(bool)
    ].set_index("clip_uid")
    high_disagreement = primary_disagreement.judge_sample_sd >= primary_disagreement.judge_sample_sd.quantile(0.75)
    high_review = float(primary_disagreement.loc[high_disagreement, "review_recommended"].mean())
    low_review = float(primary_disagreement.loc[~high_disagreement, "review_recommended"].mean())
    test.to_parquet(V2_RESULTS_ROOT / "04_ANALYSIS" / "panel_simulation_v2.parquet", index=False)
    result = {
        "status": "ANALYZED",
        "clips": int(len(clip)),
        "fusion_weight_human": float(weight),
        "human_median_absolute_error": human_median,
        "human_mean_absolute_error": human_mean,
        "fixed_fused_median_absolute_error": float(clip.fixed_fused_error.median()),
        "learned_fused_median_absolute_error": fused_median,
        "learned_fused_mean_absolute_error": fused_mean,
        "fusion_relative_error_reduction": fusion_reduction,
        "fusion_mean_relative_error_reduction": fusion_mean_reduction,
        "fusion_relative_reduction_note": (
            "Undefined because the human median absolute error is zero."
            if fusion_reduction is None
            else "Defined against the human median absolute error."
        ),
        "median_error_difference": float(np.median(difference)),
        "bootstrap_95_ci": ci,
        "sign_flip_permutation_p": p_value,
        "selective_error_reduction_at_80pct_coverage": selective_reduction,
        "accepted_80pct_mean_error": float(accepted.learned_fused_error.mean()),
        "random_80pct_mean_error_median": float(np.median(random_risks)),
        "aurc": aurc,
        "actual_review_fraction": float(clip.review_recommended.mean()),
        "high_disagreement_review_rate": high_review,
        "lower_disagreement_review_rate": low_review,
        "claim_boundary": "Retrospective judge-panel simulation; no real human-AI interaction.",
    }
    write_json(V2_RESULTS_ROOT / "04_ANALYSIS" / "panel_analysis_v2.json", result)
    return result


def publication_decision(score: dict, disagreement: dict, trace: dict, panel: dict) -> dict:
    contract = load_v2_contract()["publication_gates"]
    score_delta = float(score["spearman_delta_from_best_internal_baseline"])
    disagreement_positive = disagreement["judge_sd_spearman"] > 0
    disagreement_ci_positive = disagreement["judge_sd_bootstrap_95_ci"][0] > 0
    fusion_reduction = panel["fusion_relative_error_reduction"]
    fusion_positive = panel["median_error_difference"] < 0
    fusion_ci_positive = panel["bootstrap_95_ci"][1] < 0
    selective_positive = panel["selective_error_reduction_at_80pct_coverage"] > 0
    human_endpoints = [disagreement_positive, fusion_positive, selective_positive]
    strong_gate = contract["strong_psychology"]
    strong = bool(
        score_delta >= -float(strong_gate["max_score_spearman_drop"])
        and disagreement_ci_positive
        and fusion_reduction is not None
        and fusion_reduction >= float(strong_gate["min_fusion_relative_reduction"])
        and fusion_ci_positive
        and panel["selective_error_reduction_at_80pct_coverage"]
        >= float(strong_gate["min_selective_reduction_at_80pct_coverage"])
        and trace["targeted_deletion_supported"]
    )
    clear_reverse = bool(
        disagreement["judge_sd_bootstrap_95_ci"][1] < 0
        or (panel["bootstrap_95_ci"][0] > 0 and panel["median_error_difference"] > 0)
        or panel["selective_error_reduction_at_80pct_coverage"] < -0.02
    )
    application = bool(
        score_delta >= -float(contract["stop"]["max_score_spearman_drop"])
        and sum(human_endpoints)
        >= int(contract["application_psychology"]["required_positive_human_centered_endpoints"])
        and not clear_reverse
    )
    sports = bool(score_delta >= -0.03 and trace["targeted_deletion_supported"])
    if strong:
        verdict = "FRONTIERS_PSYCHOLOGY_STRONG_GO"
    elif application:
        verdict = "FRONTIERS_PSYCHOLOGY_APPLICATION_GO"
    elif sports:
        verdict = "SPORTS_TECHNOLOGY_GO"
    else:
        verdict = "NO_GO"
    result = {
        "verdict": verdict,
        "strong_psychology": strong,
        "application_psychology": application,
        "sports_technology": sports,
        "score_delta": score_delta,
        "positive_human_centered_endpoints": int(sum(human_endpoints)),
        "human_centered_endpoint_directions": {
            "disagreement": disagreement_positive,
            "fusion": fusion_positive,
            "selective_review": selective_positive,
        },
        "clear_reverse_endpoint": clear_reverse,
        "material_passport": "experiment-agent / analysis / 2026-08-17 / ANALYZED / trustdive_disagreement_v2",
    }
    write_json(V2_RESULTS_ROOT / "04_ANALYSIS" / "publication_decision_v2.json", result)
    return result


def analyze_all_v2() -> dict:
    require_contract_frozen()
    prediction_path = V2_RESULTS_ROOT / "03_FINAL" / "predictions_v2.parquet"
    if not prediction_path.exists():
        raise RuntimeError("Run final v2 training before analysis")
    predictions = pd.read_parquet(prediction_path)
    score = analyze_score_v2(predictions)
    disagreement = analyze_disagreement_v2(predictions)
    trace = analyze_trace_v2(predictions)
    panel = analyze_panel_v2(predictions)
    decision = publication_decision(score, disagreement, trace, panel)
    output = {
        "score": score,
        "disagreement": disagreement,
        "trace": trace,
        "panel": panel,
        "publication": decision,
    }
    write_json(V2_RESULTS_ROOT / "04_ANALYSIS" / "analysis_summary_v2.json", output)
    return output
