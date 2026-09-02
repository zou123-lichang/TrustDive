from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .metrics import aqa_score_metrics
from .util import sha256_file, write_json
from .v7_analysis import _aurc, _empirical_percentile, _top_fraction
from .v7_data import V7_RESULTS_ROOT
from .v8_data import (
    V8_RESULTS_ROOT,
    load_conditional_manifest_v8,
    load_v8_contract,
    reveal_test_disagreement_v8,
)


def _binary(labels: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
    probability = np.clip(risk, 0.0, 1.0)
    return {
        "auroc": float(roc_auc_score(labels, risk)),
        "auprc": float(average_precision_score(labels, risk)),
        "brier": float(brier_score_loss(labels, probability)),
    }


def _cluster_bootstrap(frame: pd.DataFrame, statistic, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    families = frame.event_family.astype(str).unique()
    groups = {family: frame[frame.event_family.astype(str) == family] for family in families}
    values = np.full(iterations, np.nan, dtype=float)
    for iteration in range(iterations):
        sampled = rng.choice(families, size=len(families), replace=True)
        draw = pd.concat([groups[family] for family in sampled], ignore_index=True)
        try:
            values[iteration] = statistic(draw)
        except (ValueError, ZeroDivisionError):
            continue
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _review_metrics(frame: pd.DataFrame, selected: np.ndarray) -> dict[str, float]:
    selected = np.asarray(selected, dtype=bool)
    accepted = ~selected
    all_mae = float(frame.absolute_error.mean())
    accepted_mae = float(frame.loc[accepted, "absolute_error"].mean())
    high_error = frame.high_error.to_numpy(dtype=bool)
    eligible = frame.disagreement_primary_eligible.to_numpy(dtype=bool)
    high_dispute = frame.high_excess_disagreement.to_numpy(dtype=bool) & eligible
    prevalence = float(high_dispute[eligible].mean()) if eligible.any() else float("nan")
    return {
        "review_fraction": float(selected.mean()),
        "accepted_mae": accepted_mae,
        "accepted_mae_reduction": (all_mae - accepted_mae) / max(all_mae, 1e-8),
        "high_error_recall": float(selected[high_error].mean()) if high_error.any() else float("nan"),
        "high_excess_disagreement_recall": float(selected[high_dispute].mean()) if high_dispute.any() else float("nan"),
        "high_excess_disagreement_enrichment": (
            float(high_dispute[selected & eligible].mean() / max(prevalence, 1e-8))
            if np.any(selected & eligible) else float("nan")
        ),
    }


def _pareto_select(
    error_risk: np.ndarray,
    dispute_risk: np.ndarray,
    total_fraction: float,
    error_fraction: float,
    force: np.ndarray,
) -> np.ndarray:
    n = len(error_risk)
    total = int(np.ceil(total_fraction * n))
    error_count = int(np.ceil(error_fraction * n))
    selected = np.zeros(n, dtype=bool)
    forced = np.flatnonzero(force)
    if len(forced) > total:
        priority = np.maximum(error_risk[forced], dispute_risk[forced])
        forced = forced[np.argsort(-priority, kind="stable")[:total]]
    selected[forced] = True
    error_candidates = np.flatnonzero(~selected)
    need_error = max(min(error_count, total) - int(selected.sum()), 0)
    order = error_candidates[np.argsort(-error_risk[error_candidates], kind="stable")]
    selected[order[:need_error]] = True
    remaining = np.flatnonzero(~selected)
    need = total - int(selected.sum())
    order = remaining[np.argsort(-dispute_risk[remaining], kind="stable")]
    selected[order[:need]] = True
    return selected


def _harmonic(a: float, b: float) -> float:
    return 2.0 * a * b / max(a + b, 1e-8)


def analyze_review_v8() -> dict:
    contract = load_v8_contract()
    prediction = pd.read_parquet(V8_RESULTS_ROOT / "04_FINAL" / "predictions_v8.parquet")
    evidence = pd.read_parquet(V8_RESULTS_ROOT / "05_DUAL_EVIDENCE" / "dual_phase_evidence_v8.parquet")
    manifest = reveal_test_disagreement_v8(load_conditional_manifest_v8())
    frame = prediction.merge(
        manifest[[
            "clip_uid", "judge_sample_sd", "excess_log_judge_sd", "high_excess_disagreement",
            "judge_scores_json", "error_threshold",
        ]],
        on="clip_uid", validate="one_to_one", suffixes=("", "_target"),
    ).merge(
        evidence[[
            "clip_uid", "score_top_phase", "disagreement_top_phase",
            "score_top_match", "disagreement_top_match",
            "score_targeted_effect", "score_random_effect",
            "disagreement_targeted_effect", "disagreement_random_effect",
        ]],
        on="clip_uid", validate="one_to_one",
    )
    v7_review = pd.read_parquet(V7_RESULTS_ROOT / "05_RISK_REVIEW" / "review_priority_v7.parquet").set_index("clip_uid")
    v7_prediction = pd.read_parquet(V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet").set_index("clip_uid")
    frame["v7_disagreement_risk"] = v7_review.loc[frame.clip_uid, "risk_disagreement"].to_numpy(dtype=float)
    frame["v7_combined_risk"] = v7_review.loc[frame.clip_uid, "review_priority"].to_numpy(dtype=float)
    frame["rica2_uncertainty"] = v7_prediction.loc[frame.clip_uid, "teacher_uncertainty"].to_numpy(dtype=float)
    frame["rica2_predicted_score"] = v7_prediction.loc[frame.clip_uid, "teacher_predicted_score"].to_numpy(dtype=float)
    frame["plain_ridge_predicted_score"] = v7_prediction.loc[frame.clip_uid, "plain_ridge_predicted_score"].to_numpy(dtype=float)
    calibration = frame.analysis_role == "calibration"
    test = frame.analysis_role == "official_test"
    frame["risk_error_percentile"] = _empirical_percentile(
        frame.loc[calibration, "risk_error"], frame.risk_error
    )
    frame["risk_disagreement_percentile"] = _empirical_percentile(
        frame.loc[calibration, "risk_excess_disagreement"], frame.risk_excess_disagreement
    )
    frame["negative_score_risk"] = -frame.predicted_quality
    frame["conditional_risk"] = frame.expected_log_judge_sd
    frame["absolute_error"] = np.abs(frame.predicted_score - frame.dive_score)
    frame["high_error"] = frame.absolute_error >= frame.error_threshold_target

    calibration_frame = frame.loc[calibration].copy().reset_index(drop=True)
    calibration_frame["high_error"] = calibration_frame.absolute_error >= calibration_frame.error_threshold_target
    allocations = []
    for error_fraction, dispute_fraction in contract["review"]["allocations"]:
        selected = _pareto_select(
            calibration_frame.risk_error_percentile.to_numpy(),
            calibration_frame.risk_disagreement_percentile.to_numpy(),
            float(contract["review"]["review_fraction"]),
            float(error_fraction),
            calibration_frame.open_set.to_numpy(dtype=bool),
        )
        metrics = _review_metrics(calibration_frame, selected)
        allocations.append({
            "error_fraction": float(error_fraction),
            "dispute_fraction": float(dispute_fraction),
            "harmonic_recall": _harmonic(metrics["high_error_recall"], metrics["high_excess_disagreement_recall"]),
            **metrics,
        })
    allocation_frame = pd.DataFrame(allocations)
    eligible_allocation = allocation_frame[allocation_frame.accepted_mae_reduction >= 0].copy()
    if eligible_allocation.empty:
        eligible_allocation = allocation_frame.copy()
    best_value = eligible_allocation.harmonic_recall.max()
    near = eligible_allocation[eligible_allocation.harmonic_recall >= best_value - 0.01]
    balanced = near[np.isclose(near.error_fraction, 0.10)]
    selected_allocation = balanced.iloc[0] if len(balanced) else near.sort_values(
        ["harmonic_recall", "accepted_mae"], ascending=[False, True]
    ).iloc[0]
    allocation_path = V8_RESULTS_ROOT / "06_REVIEW" / "pareto_allocation_calibration_v8.csv"
    allocation_frame.to_csv(allocation_path, index=False)

    test_frame = frame.loc[test].copy().reset_index(drop=True)
    review_fraction = float(contract["review"]["review_fraction"])
    pareto = _pareto_select(
        test_frame.risk_error_percentile.to_numpy(),
        test_frame.risk_disagreement_percentile.to_numpy(),
        review_fraction,
        float(selected_allocation.error_fraction),
        test_frame.open_set.to_numpy(dtype=bool),
    )
    strategy_masks = {
        "low_score": _top_fraction(test_frame.negative_score_risk.to_numpy(), review_fraction),
        "rica2_uncertainty": _top_fraction(test_frame.rica2_uncertainty.to_numpy(), review_fraction),
        "v7_max_percentile": _top_fraction(test_frame.v7_combined_risk.to_numpy(), review_fraction),
        "error_risk_only": _top_fraction(test_frame.risk_error_percentile.to_numpy(), review_fraction, test_frame.open_set),
        "disagreement_risk_only": _top_fraction(test_frame.risk_disagreement_percentile.to_numpy(), review_fraction, test_frame.open_set),
        "v8_pareto": pareto,
    }
    strategy_metrics = {name: _review_metrics(test_frame, selected) for name, selected in strategy_masks.items()}
    for name, selected in strategy_masks.items():
        strategy_metrics[name]["aurc"] = _aurc(
            test_frame.absolute_error.to_numpy(),
            np.where(selected, 1.0, 0.0) if name == "v8_pareto" else {
                "low_score": test_frame.negative_score_risk,
                "rica2_uncertainty": test_frame.rica2_uncertainty,
                "v7_max_percentile": test_frame.v7_combined_risk,
                "error_risk_only": test_frame.risk_error_percentile,
                "disagreement_risk_only": test_frame.risk_disagreement_percentile,
            }[name],
        )
    test_frame["review_budget20"] = pareto

    panel = test_frame[test_frame.disagreement_primary_eligible.astype(bool)].copy().reset_index(drop=True)
    risks = {
        "negative_predicted_score": panel.negative_score_risk.to_numpy(),
        "score_action_difficulty": panel.conditional_risk.to_numpy(),
        "rica2_uncertainty": panel.rica2_uncertainty.to_numpy(),
        "v7_disagreement": panel.v7_disagreement_risk.to_numpy(),
        "v8_phase_conflict": panel.risk_excess_disagreement.to_numpy(),
    }
    v2_path = V8_RESULTS_ROOT.parent / "V2_DISAGREEMENT" / "03_FINAL" / "predictions_v2.parquet"
    if v2_path.exists():
        v2 = pd.read_parquet(v2_path).set_index("clip_uid")
        risks["v2_student_t_sigma"] = v2.loc[panel.clip_uid, "sigma_judge"].to_numpy(dtype=float)
    disagreement_rows = []
    labels = panel.high_excess_disagreement.to_numpy(dtype=bool)
    for name, risk in risks.items():
        metrics = _binary(labels, _empirical_percentile(risk, risk))
        metrics["continuous_excess_spearman"] = float(spearmanr(risk, panel.excess_log_judge_sd).statistic)
        metrics["raw_sd_spearman"] = float(spearmanr(risk, panel.judge_sample_sd).statistic)
        disagreement_rows.append({"model": name, **metrics})
    disagreement_table = pd.DataFrame(disagreement_rows)
    disagreement_path = V8_RESULTS_ROOT / "06_REVIEW" / "disagreement_comparison_v8.csv"
    disagreement_table.to_csv(disagreement_path, index=False)
    v8_auc = float(disagreement_table.loc[disagreement_table.model == "v8_phase_conflict", "auroc"].iloc[0])
    simple = disagreement_table[disagreement_table.model.isin(("negative_predicted_score", "score_action_difficulty"))]
    best_simple_auc = float(simple.auroc.max())
    auc_gain = v8_auc - best_simple_auc

    high = panel.high_excess_disagreement.to_numpy(dtype=bool)
    teacher_high_mae = float(np.abs(panel.loc[high, "rica2_predicted_score"] - panel.loc[high, "dive_score"]).mean())
    v8_high_mae = float(panel.loc[high, "absolute_error"].mean())
    high_reduction = (teacher_high_mae - v8_high_mae) / max(teacher_high_mae, 1e-8)
    iterations = int(contract["statistics"]["cluster_bootstrap_iterations"])
    seed = int(contract["statistics"]["seed"])
    auc_gain_ci = _cluster_bootstrap(
        panel,
        lambda draw: float(
            roc_auc_score(draw.high_excess_disagreement, draw.risk_excess_disagreement)
            - max(
                roc_auc_score(draw.high_excess_disagreement, draw.negative_score_risk),
                roc_auc_score(draw.high_excess_disagreement, draw.conditional_risk),
            )
        ),
        iterations, seed,
    )
    high_reduction_ci = _cluster_bootstrap(
        panel,
        lambda draw: float(
            (
                np.abs(draw.loc[draw.high_excess_disagreement, "rica2_predicted_score"] - draw.loc[draw.high_excess_disagreement, "dive_score"]).mean()
                - draw.loc[draw.high_excess_disagreement, "absolute_error"].mean()
            )
            / max(np.abs(draw.loc[draw.high_excess_disagreement, "rica2_predicted_score"] - draw.loc[draw.high_excess_disagreement, "dive_score"]).mean(), 1e-8)
        ),
        iterations, seed + 1,
    )
    selective_ci = _cluster_bootstrap(
        test_frame,
        lambda draw: float(
            (draw.absolute_error.mean() - draw.loc[~draw.review_budget20, "absolute_error"].mean())
            / max(draw.absolute_error.mean(), 1e-8)
        ),
        iterations, seed + 2,
    )
    enrichment_ci = _cluster_bootstrap(
        panel,
        lambda draw: float(
            draw.loc[draw.review_budget20, "high_excess_disagreement"].mean()
            / max(draw.high_excess_disagreement.mean(), 1e-8)
        ),
        iterations, seed + 3,
    )
    evidence_test = evidence[(evidence.analysis_role == "official_test") & (~evidence.open_set.astype(bool))].copy()
    score_targeted_ci = _cluster_bootstrap(
        evidence_test,
        lambda draw: float(np.median(draw.score_targeted_effect - draw.score_random_effect)),
        iterations, seed + 4,
    )
    disagreement_targeted_ci = _cluster_bootstrap(
        evidence_test,
        lambda draw: float(np.median(draw.disagreement_targeted_effect - draw.disagreement_random_effect)),
        iterations, seed + 5,
    )

    rng = np.random.default_rng(seed)
    random_rows = []
    for _ in range(int(contract["review"]["random_review_iterations"])):
        selected = np.zeros(len(test_frame), dtype=bool)
        selected[rng.choice(len(test_frame), size=int(pareto.sum()), replace=False)] = True
        random_rows.append(_review_metrics(test_frame, selected)["accepted_mae_reduction"])

    calibration_thresholds = {
        "wide_interval": float(frame.loc[calibration, "prediction_interval_width"].quantile(0.8)),
        "high_error": float(frame.loc[calibration, "risk_error_percentile"].quantile(0.8)),
        "high_disagreement": float(frame.loc[calibration, "risk_disagreement_percentile"].quantile(0.8)),
    }
    reasons = []
    for row in frame.itertuples(index=False):
        items = []
        if bool(row.open_set):
            items.append("insufficient similar references")
        if row.prediction_interval_width >= calibration_thresholds["wide_interval"]:
            items.append("wide prediction interval")
        if row.risk_error_percentile >= calibration_thresholds["high_error"]:
            items.append("high model-error risk")
        if row.risk_disagreement_percentile >= calibration_thresholds["high_disagreement"]:
            items.append("predicted excess judge disagreement")
        if row.disagreement_top_phase != "open_set" and row.risk_disagreement_percentile >= 0.8:
            items.append(f"disagreement evidence concentrated in {row.disagreement_top_phase}")
        reasons.append("; ".join(items) if items else "routine automatic scoring")
    frame["review_reason"] = reasons
    lookup = dict(zip(test_frame.clip_uid, test_frame.review_budget20))
    frame["review_budget20"] = frame.clip_uid.map(lookup).eq(True)
    review_path = V8_RESULTS_ROOT / "06_REVIEW" / "review_priority_v8.parquet"
    frame[[
        "clip_uid", "official_split", "analysis_role", "event_family", "action_type",
        "predicted_score", "risk_error", "risk_excess_disagreement",
        "risk_error_percentile", "risk_disagreement_percentile", "review_budget20",
        "review_reason", "open_set", "score_top_phase", "disagreement_top_phase",
    ]].to_parquet(review_path, index=False)

    score_metrics = aqa_score_metrics(test_frame.dive_score, test_frame.predicted_score)
    base_metrics = aqa_score_metrics(test_frame.dive_score, test_frame.base_predicted_score)
    score_noninferior = (float(base_metrics["spearman"]) - float(score_metrics["spearman"])) <= float(contract["publication"]["maximum_score_spearman_drop"])
    trace_summary = json.loads(
        (V8_RESULTS_ROOT / "05_DUAL_EVIDENCE" / "dual_phase_summary_v8.json").read_text(encoding="utf-8")
    )
    trace_pass = bool(
        trace_summary.get("status") == "PASS"
        and score_targeted_ci[0] > 0
        and disagreement_targeted_ci[0] > 0
    )
    excess_strong = bool(
        v8_auc >= float(contract["publication"]["strong_minimum_excess_auroc"])
        and auc_gain >= float(contract["publication"]["strong_minimum_auroc_gain"])
        and auc_gain_ci[0] > 0
    )
    selected_metrics = strategy_metrics["v8_pareto"]
    review_pass = bool(
        selected_metrics["accepted_mae_reduction"] >= float(contract["publication"]["minimum_selective_mae_reduction"])
        and selective_ci[0] > 0
        and selected_metrics["high_excess_disagreement_enrichment"] >= float(contract["publication"]["minimum_disagreement_enrichment"])
        and enrichment_ci[0] > 1.0
        and selected_metrics["high_error_recall"] > strategy_metrics["rica2_uncertainty"]["high_error_recall"]
        and selected_metrics["high_excess_disagreement_recall"] > strategy_metrics["rica2_uncertainty"]["high_excess_disagreement_recall"]
    )
    high_score_pass = bool(
        high_reduction >= float(contract["publication"]["minimum_high_excess_mae_reduction"])
        and high_reduction_ci[0] > 0
    )
    if score_noninferior and high_score_pass and excess_strong and trace_pass and review_pass:
        decision = "FRONTIERS_PSYCHOLOGY_8_5"
    elif score_noninferior and trace_pass and (
        selected_metrics["accepted_mae_reduction"] > 0 or auc_gain > 0
    ):
        decision = "FRONTIERS_PSYCHOLOGY_APPLICATION"
    elif score_noninferior and trace_pass:
        decision = "SPORTS_TECHNOLOGY"
    else:
        decision = "EXPLORATORY_ONLY"

    result = {
        "status": "ANALYZED",
        "publication_decision": decision,
        "score": {
            "base": base_metrics,
            "v8": score_metrics,
            "score_noninferior": score_noninferior,
            "high_excess_rows": int(high.sum()),
            "rica2_high_excess_mae": teacher_high_mae,
            "v8_high_excess_mae": v8_high_mae,
            "high_excess_mae_reduction": high_reduction,
            "high_excess_mae_reduction_cluster_ci": list(high_reduction_ci),
        },
        "conditional_disagreement": {
            "panel_rows": int(len(panel)),
            "comparison": disagreement_table.to_dict(orient="records"),
            "best_simple_auroc": best_simple_auc,
            "v8_auroc": v8_auc,
            "auroc_gain": auc_gain,
            "auroc_gain_cluster_ci": list(auc_gain_ci),
            "phase_conflict_incremental_value": bool(auc_gain > 0 and auc_gain_ci[0] > 0),
        },
        "dual_evidence": {
            **trace_summary,
            "score_targeted_minus_random_cluster_ci": list(score_targeted_ci),
            "disagreement_targeted_minus_random_cluster_ci": list(disagreement_targeted_ci),
            "trace_pass": trace_pass,
        },
        "selective_review": {
            "selected_allocation": {
                "error_fraction": float(selected_allocation.error_fraction),
                "dispute_fraction": float(selected_allocation.dispute_fraction),
            },
            "strategies": strategy_metrics,
            "selective_reduction_cluster_ci": list(selective_ci),
            "disagreement_enrichment_cluster_ci": list(enrichment_ci),
            "random_review_reduction_interval": [
                float(np.quantile(random_rows, 0.025)), float(np.quantile(random_rows, 0.975))
            ],
            "review_pass": review_pass,
        },
        "fallacy_guardrails": {
            "low_score_confound_tested": True,
            "test_threshold_tuning": False,
            "judge_video_pairs_treated_as_independent": False,
            "reviewed_cases_assumed_corrected": False,
            "causal_psychological_claim": False,
        },
        "output_hashes": {
            "review_priority_v8.parquet": sha256_file(review_path),
            "disagreement_comparison_v8.csv": sha256_file(disagreement_path),
            "pareto_allocation_calibration_v8.csv": sha256_file(allocation_path),
        },
    }
    write_json(V8_RESULTS_ROOT / "06_REVIEW" / "analysis_summary_v8.json", result)
    return result
