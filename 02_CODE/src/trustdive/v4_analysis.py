from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import aqa_score_metrics
from .util import write_json
from .v4_counterfactual import PHASES
from .v4_data import V4_RESULTS_ROOT, load_v4_contract, require_v4_frozen


VISIBLE_REVIEW_REASONS = (
    "相似参考动作不足",
    "预测区间较宽",
    "教师与透明评分存在明显差异",
    "阶段结论对边界扰动较敏感",
    "预测裁判分歧较高",
    "相似参考距离较大",
)


def _safe_spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else 0.0


def _cluster_bootstrap(frame: pd.DataFrame, statistic, iterations: int, seed: int) -> tuple[float, float]:
    families = frame.event_family.astype(str).unique()
    groups = {family: np.flatnonzero(frame.event_family.astype(str).to_numpy() == family) for family in families}
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.choice(families, size=len(families), replace=True)
        indices = np.concatenate([groups[family] for family in sampled])
        values[iteration] = statistic(frame.iloc[indices])
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(finite, (0.025, 0.975)))


def _fit_binary_risk(features: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels).size < 2:
        return None, float(labels.mean())
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260818),
    )
    model.fit(features, labels)
    return model, None


def _predict_binary(model, constant, features: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(features), float(constant), dtype=float)
    return model.predict_proba(features)[:, 1]


def _calibrated_percentile(calibration: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(calibration, dtype=float))
    return np.searchsorted(ordered, np.asarray(values, dtype=float), side="right") / max(len(ordered), 1)


def _write_sota_comparison(
    student_metrics: dict, teacher_metrics: dict, final_training: dict
) -> pd.DataFrame:
    rows = [
        {"method": "MUSDL", "spearman": 0.8978, "relative_l2": 0.3704, "status": "literature", "same_code_reproduced": False, "backbone": "I3D", "extra_pose": False, "true_test_phases": False, "source": "RICA2 ECCV 2024 comparison table"},
        {"method": "TSA", "spearman": 0.9203, "relative_l2": 0.3420, "status": "literature", "same_code_reproduced": False, "backbone": "I3D", "extra_pose": False, "true_test_phases": False, "source": "RICA2 ECCV 2024 comparison table"},
        {"method": "TPT", "spearman": 0.9333, "relative_l2": 0.2877, "status": "literature", "same_code_reproduced": False, "backbone": "I3D", "extra_pose": False, "true_test_phases": False, "source": "RICA2 ECCV 2024 comparison table"},
        {"method": "HP-MCoRe", "spearman": 0.9383, "relative_l2": 0.2690, "status": "literature", "same_code_reproduced": False, "backbone": "I3D + HRNet-W48 pose", "extra_pose": True, "true_test_phases": False, "source": "official HP-MCoRe repository, commit 1bb3212"},
        {"method": "RICA2 stochastic", "spearman": 0.9421, "relative_l2": 0.2600, "status": "literature", "same_code_reproduced": False, "backbone": "I3D", "extra_pose": False, "true_test_phases": False, "source": "RICA2 ECCV 2024 comparison table"},
        {"method": "RICA2 deterministic", "spearman": teacher_metrics["spearman"], "relative_l2": teacher_metrics["relative_l2"], "status": "reproduced_here", "same_code_reproduced": True, "backbone": "I3D", "extra_pose": False, "true_test_phases": False, "source": "official code pinned at 07701b6"},
        {"method": "CFPD without Shapley", "spearman": final_training["no_shapley_ablation_metrics"]["spearman"], "relative_l2": final_training["no_shapley_ablation_metrics"]["relative_l2"], "status": "measured_here", "same_code_reproduced": True, "backbone": "frozen RICA2 I3D", "extra_pose": False, "true_test_phases": False, "source": "v4 ablation"},
        {"method": "CFPD", "spearman": student_metrics["spearman"], "relative_l2": student_metrics["relative_l2"], "status": "measured_here", "same_code_reproduced": True, "backbone": "frozen RICA2 I3D", "extra_pose": False, "true_test_phases": False, "source": "v4 primary"},
    ]
    historical = [
        ("TrustDive-D relative", V4_RESULTS_ROOT.parent / "V2_DISAGREEMENT" / "03_FINAL" / "relative_predictions_v2.parquet"),
        ("TrustDive-Trace v3", V4_RESULTS_ROOT.parent / "V3_TRACE_FIRST" / "03_FINAL" / "predictions_trace_v3.parquet"),
    ]
    for name, path in historical:
        if path.exists():
            frame = pd.read_parquet(path)
            frame = frame.loc[frame.official_split == "test"]
            metrics = aqa_score_metrics(frame.dive_score, frame.predicted_score)
            rows.append(
                {"method": name, "spearman": metrics["spearman"], "relative_l2": metrics["relative_l2"], "status": "historical_internal", "same_code_reproduced": True, "backbone": "VideoMAE", "extra_pose": False, "true_test_phases": False, "source": "preserved v2/v3 output; AQA Relative-L2 recomputed without refitting"}
            )
    output = pd.DataFrame(rows)
    output.to_csv(V4_RESULTS_ROOT / "06_ANALYSIS" / "sota_comparison_v4.csv", index=False)
    return output


def build_review_priority_v4(predictions: pd.DataFrame, stress: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    merged = predictions.merge(
        stress[["clip_uid", "perturbed_top_phase_agreement", "perturbed_contribution_cosine", "perturbed_prediction_change"]],
        on="clip_uid",
        validate="one_to_one",
    )
    merged["conformal_interval_width"] = merged.upper_quality - merged.lower_quality
    error_feature_columns = [
        "conformal_interval_width",
        "teacher_student_abs_difference",
        "reference_distance",
        "reference_dispersion",
        "perturbed_prediction_change",
        "perturbed_top_phase_agreement",
        "open_set",
    ]
    disagreement_feature_columns = [
        "reference_distance",
        "reference_dispersion",
        "perturbed_prediction_change",
        "perturbed_top_phase_agreement",
        "open_set",
    ]
    error_features = merged[error_feature_columns].astype(float).to_numpy()
    disagreement_features = merged[disagreement_feature_columns].astype(float).to_numpy()
    train = merged.analysis_role.isin(("fit", "validation")).to_numpy()
    calibration = (merged.analysis_role == "calibration").to_numpy()
    abs_score_error = np.abs(merged.predicted_score.to_numpy(dtype=float) - merged.dive_score.to_numpy(dtype=float))
    error_threshold = float(np.quantile(abs_score_error[train], 0.75))
    error_model, error_constant = _fit_binary_risk(
        error_features[train], abs_score_error[train] >= error_threshold
    )
    risk_error = _predict_binary(error_model, error_constant, error_features)

    judge_train = train & merged.disagreement_primary_eligible.to_numpy(dtype=bool)
    disagreement_threshold = float(np.quantile(merged.judge_sample_sd.to_numpy(dtype=float)[judge_train], 0.75))
    disagreement_model, disagreement_constant = _fit_binary_risk(
        disagreement_features[judge_train],
        merged.judge_sample_sd.to_numpy(dtype=float)[judge_train] >= disagreement_threshold,
    )
    risk_disagreement = _predict_binary(
        disagreement_model, disagreement_constant, disagreement_features
    )
    error_percentile = _calibrated_percentile(risk_error[calibration], risk_error)
    disagreement_percentile = _calibrated_percentile(risk_disagreement[calibration], risk_disagreement)
    priority = np.maximum(error_percentile, disagreement_percentile)
    threshold = float(np.quantile(priority[calibration], 0.80))
    review = priority >= threshold

    interval_width = merged.conformal_interval_width.to_numpy(dtype=float)
    cutoffs = {
        "interval": float(np.quantile(interval_width[calibration], 0.80)),
        "teacher": float(np.quantile(merged.teacher_student_abs_difference.to_numpy()[calibration], 0.80)),
        "stability": float(np.quantile(merged.perturbed_top_phase_agreement.to_numpy()[calibration], 0.20)),
        "distance": float(np.quantile(merged.reference_distance.to_numpy()[calibration], 0.80)),
    }
    reasons = []
    for index, row in merged.iterrows():
        if bool(row.open_set):
            reason = "相似参考动作不足"
        elif interval_width[index] >= cutoffs["interval"]:
            reason = "预测区间较宽"
        elif float(row.teacher_student_abs_difference) >= cutoffs["teacher"]:
            reason = "教师与透明评分存在明显差异"
        elif float(row.perturbed_top_phase_agreement) <= cutoffs["stability"]:
            reason = "阶段结论对边界扰动较敏感"
        elif disagreement_percentile[index] >= error_percentile[index]:
            reason = "预测裁判分歧较高"
        else:
            reason = "相似参考距离较大"
        reasons.append(reason)
    if not set(reasons).issubset(set(VISIBLE_REVIEW_REASONS)):
        raise AssertionError("A non-contract review reason was generated")
    output = merged.copy()
    output["risk_error"] = risk_error
    output["risk_disagreement"] = risk_disagreement
    output["risk_error_percentile"] = error_percentile
    output["risk_disagreement_percentile"] = disagreement_percentile
    output["review_priority"] = priority
    output["review_flag_20pct"] = review
    output["review_reason"] = reasons
    output.to_parquet(V4_RESULTS_ROOT / "06_ANALYSIS" / "review_priority_v4.parquet", index=False)
    summary = {
        "error_label_threshold_score": error_threshold,
        "disagreement_label_threshold_sd": disagreement_threshold,
        "review_priority_threshold": threshold,
        "calibration_review_fraction": float(review[calibration].mean()),
        "visible_reason_uses_seed_disagreement": False,
        "error_feature_columns": error_feature_columns,
        "disagreement_feature_columns": disagreement_feature_columns,
    }
    return output, summary


def analyze_v4() -> dict:
    require_v4_frozen()
    contract = load_v4_contract()
    prediction_path = V4_RESULTS_ROOT / "04_FINAL" / "predictions_v4.parquet"
    stress_path = V4_RESULTS_ROOT / "05_STRESS" / "trace_stress_v4.parquet"
    if not prediction_path.exists() or not stress_path.exists():
        raise RuntimeError("Run final training and stress testing before v4 analysis")
    predictions = pd.read_parquet(prediction_path)
    stress = pd.read_parquet(stress_path)
    review, review_training = build_review_priority_v4(predictions, stress)
    test = predictions.loc[predictions.official_split == "test"].copy()
    test_metrics = aqa_score_metrics(test.dive_score, test.predicted_score)
    teacher_score = 3.0 * test.difficulty.to_numpy() * test.teacher_predicted_quality.to_numpy()
    teacher_metrics = aqa_score_metrics(test.dive_score, teacher_score)
    score_delta = float(test_metrics["spearman"] - teacher_metrics["spearman"])
    iterations = int(contract["statistics"]["cluster_bootstrap_iterations"])
    seed = int(contract["statistics"]["seed"])
    score_bootstrap_frame = test[["event_family", "dive_score", "predicted_score", "difficulty", "teacher_predicted_quality"]].copy()
    score_delta_ci = _cluster_bootstrap(
        score_bootstrap_frame,
        lambda sample: aqa_score_metrics(sample.dive_score, sample.predicted_score)["spearman"]
        - aqa_score_metrics(sample.dive_score, 3.0 * sample.difficulty * sample.teacher_predicted_quality)["spearman"],
        iterations,
        seed,
    )
    final_training = json.loads((V4_RESULTS_ROOT / "04_FINAL" / "final_training_summary_v4.json").read_text(encoding="utf-8"))
    sota_comparison = _write_sota_comparison(test_metrics, teacher_metrics, final_training)
    score_gates = {
        "minimum_spearman": test_metrics["spearman"] >= float(contract["publication_gates"]["minimum_test_spearman"]),
        "teacher_noninferior_point": score_delta >= -float(contract["publication_gates"]["maximum_teacher_spearman_drop"]),
        "teacher_noninferior_ci": score_delta_ci[0] >= -float(contract["publication_gates"]["maximum_teacher_spearman_drop"]),
        "seed_stability": float(final_training["seed_spearman_sd"]) <= float(contract["publication_gates"]["maximum_seed_spearman_sd"]),
    }

    merged_test = test.merge(stress, on=["clip_uid", "official_split", "analysis_role", "source_role", "event_family"], validate="one_to_one")
    trace_eligible = ~merged_test.open_set.to_numpy(dtype=bool)
    student_phi = merged_test[[f"phase_{phase}_contribution" for phase in PHASES]].to_numpy(dtype=float)
    teacher_phi = merged_test[[f"teacher_phi_{phase}" for phase in PHASES]].to_numpy(dtype=float)
    phase_rho = _safe_spearman(student_phi[trace_eligible].reshape(-1), teacher_phi[trace_eligible].reshape(-1))
    reconstruction = np.abs(
        merged_test.base_quality.to_numpy(dtype=float) + student_phi.sum(axis=1) - merged_test.predicted_quality.to_numpy(dtype=float)
    )
    top_match = float(merged_test.top_phase_matches_teacher.to_numpy(dtype=bool)[trace_eligible].mean())
    targeted_ci = _cluster_bootstrap(
        merged_test[["event_family", "targeted_minus_random"]],
        lambda sample: float(np.median(sample.targeted_minus_random)),
        iterations,
        seed + 1,
    )
    stress_summary = json.loads((V4_RESULTS_ROOT / "05_STRESS" / "stress_summary_v4.json").read_text(encoding="utf-8"))
    trace_gates = {
        "reconstruction": float(reconstruction.max()) <= 1e-6,
        "teacher_phase_fidelity": phase_rho >= float(contract["publication_gates"]["minimum_phase_teacher_spearman"]),
        "top_phase_match": top_match >= float(contract["publication_gates"]["minimum_top_phase_match"]),
        "targeted_intervention": targeted_ci[0] > 0.0,
        "perturbed_top_phase": float(stress_summary["median_perturbed_top_phase_agreement"]) >= float(contract["publication_gates"]["minimum_perturbed_top_phase_agreement"]),
        "contribution_cosine": float(stress_summary["median_contribution_cosine"]) >= float(contract["publication_gates"]["minimum_contribution_cosine"]),
        "cross_seed_kappa": float(stress_summary["cross_seed_fleiss_kappa"]) >= 0.40,
    }

    panel = review.loc[(review.official_split == "test") & review.disagreement_primary_eligible].copy()
    panel["absolute_score_error"] = np.abs(panel.predicted_score - panel.dive_score)
    baseline_mae = float(panel.absolute_score_error.mean())
    accepted = panel.loc[~panel.review_flag_20pct]
    accepted_mae = float(accepted.absolute_score_error.mean())
    error_reduction = float(1.0 - accepted_mae / baseline_mae) if baseline_mae > 0 else 0.0
    high_disagreement = panel.judge_sample_sd >= review_training["disagreement_label_threshold_sd"]
    high_rate = float(panel.loc[high_disagreement, "review_flag_20pct"].mean())
    low_rate = float(panel.loc[~high_disagreement, "review_flag_20pct"].mean())
    enrichment = float(high_rate / low_rate) if low_rate > 0 else float("inf")
    disagreement_rho = _safe_spearman(panel.risk_disagreement, panel.judge_sample_sd)
    ordered = panel.sort_values("review_priority")
    selective_curve = ordered.absolute_score_error.expanding().mean().to_numpy(dtype=float)
    aurc = float(np.mean(selective_curve))
    rng = np.random.default_rng(seed + 20)
    random_reductions = np.empty(int(contract["statistics"]["random_review_iterations"]), dtype=float)
    accepted_count = max(1, int(round(0.80 * len(panel))))
    errors = panel.absolute_score_error.to_numpy(dtype=float)
    for iteration in range(len(random_reductions)):
        kept = rng.choice(len(panel), size=accepted_count, replace=False)
        random_reductions[iteration] = 1.0 - float(errors[kept].mean()) / baseline_mae if baseline_mae > 0 else 0.0
    random_review_interval = tuple(float(value) for value in np.quantile(random_reductions, (0.025, 0.975)))

    def reduction_stat(sample):
        baseline = float(sample.absolute_score_error.mean())
        accepted_sample = sample.loc[~sample.review_flag_20pct]
        if baseline <= 0 or accepted_sample.empty:
            return float("nan")
        return 1.0 - float(accepted_sample.absolute_score_error.mean()) / baseline

    def enrichment_stat(sample):
        high = sample.judge_sample_sd >= review_training["disagreement_label_threshold_sd"]
        if not high.any() or high.all():
            return float("nan")
        high_value = float(sample.loc[high, "review_flag_20pct"].mean())
        low_value = float(sample.loc[~high, "review_flag_20pct"].mean())
        return high_value / low_value if low_value > 0 else float("nan")

    reduction_ci = _cluster_bootstrap(panel, reduction_stat, iterations, seed + 2)
    enrichment_ci = _cluster_bootstrap(panel, enrichment_stat, iterations, seed + 3)
    review_gates = {
        "error_reduction_point": error_reduction >= float(contract["publication_gates"]["minimum_review_error_reduction"]),
        "error_reduction_ci": reduction_ci[0] > 0.0,
        "disagreement_enrichment_point": enrichment >= float(contract["publication_gates"]["minimum_disagreement_enrichment"]),
        "disagreement_enrichment_ci": enrichment_ci[0] > 1.0,
    }

    score_pass = all(score_gates.values())
    trace_pass = all(trace_gates.values())
    review_strong = all(review_gates.values())
    review_application = error_reduction > 0 and reduction_ci[0] > 0
    if score_pass and trace_pass and review_strong:
        decision = "FRONTIERS_PSYCHOLOGY_8_OF_10_PENDING_SOURCE_ISOLATION"
    elif score_pass and trace_pass and review_application:
        decision = "FRONTIERS_PSYCHOLOGY_APPLICATION_GO_PENDING_SOURCE_ISOLATION"
    elif score_pass and trace_pass:
        decision = "SPORTS_TECHNOLOGY_GO_PENDING_SOURCE_ISOLATION"
    else:
        decision = "NO_GO"
    result = {
        "material_passport": "experiment-agent / result / 2026-08-18 / ANALYZED / trustdive_cfpd_v4",
        "decision": decision,
        "score": {
            "student": test_metrics,
            "teacher": teacher_metrics,
            "spearman_delta_teacher": score_delta,
            "spearman_delta_cluster_ci": score_delta_ci,
            "gates": score_gates,
        },
        "trace": {
            "eligible_n": int(trace_eligible.sum()),
            "maximum_reconstruction_error": float(reconstruction.max()),
            "student_teacher_phase_spearman": phase_rho,
            "top_phase_match": top_match,
            "targeted_minus_random_median": float(merged_test.targeted_minus_random.median()),
            "targeted_minus_random_cluster_ci": targeted_ci,
            "stress": stress_summary,
            "gates": trace_gates,
        },
        "review": {
            "panel_n": len(panel),
            "baseline_mae": baseline_mae,
            "accepted_80pct_mae": accepted_mae,
            "error_reduction": error_reduction,
            "error_reduction_cluster_ci": reduction_ci,
            "disagreement_enrichment": enrichment,
            "disagreement_enrichment_cluster_ci": enrichment_ci,
            "predicted_disagreement_spearman": disagreement_rho,
            "aurc": aurc,
            "random_review_reduction_95pct": random_review_interval,
            "selective_better_than_random_fraction": float(np.mean(error_reduction > random_reductions)),
            "visible_reason_counts": {str(key): int(value) for key, value in panel.review_reason.value_counts().items()},
            "gates": review_gates,
            "training": review_training,
        },
        "source_isolated_status": "PENDING_IF_MAIN_GATES_PASS",
        "sota_comparison_rows": len(sota_comparison),
    }
    write_json(V4_RESULTS_ROOT / "06_ANALYSIS" / "analysis_summary_v4.json", result)
    lines = [
        "# TrustDive-CFPD v4 result decision",
        "",
        f"**Decision: {decision}**",
        "",
        f"- Student Spearman: {test_metrics['spearman']:.4f}",
        f"- Teacher Spearman: {teacher_metrics['spearman']:.4f}",
        f"- Phase fidelity Spearman: {phase_rho:.4f}",
        f"- Top-phase match: {top_match:.3%}",
        f"- Selective-review error reduction: {error_reduction:.3%}",
        f"- Disagreement enrichment: {enrichment:.3f}",
        "",
        "Source-isolated robustness is run only if the main score and trace gates pass.",
    ]
    (V4_RESULTS_ROOT / "RESULTS_DECISION_V4.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
