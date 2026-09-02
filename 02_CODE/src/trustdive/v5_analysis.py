from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .metrics import aqa_score_metrics
from .util import sha256_file, write_json
from .v4_analysis import (
    VISIBLE_REVIEW_REASONS,
    _calibrated_percentile,
    _cluster_bootstrap,
    _fit_binary_risk,
    _predict_binary,
    _safe_spearman,
)
from .v4_counterfactual import PHASES
from .v5_data import V5_RESULTS_ROOT, load_v5_contract, require_v5_frozen


def build_review_priority_v5(
    predictions: pd.DataFrame, stress: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    merged = predictions.merge(
        stress[
            [
                "clip_uid",
                "perturbed_top_phase_agreement",
                "perturbed_contribution_cosine",
                "perturbed_prediction_change",
            ]
        ],
        on="clip_uid",
        validate="one_to_one",
    )
    v2_path = (
        V5_RESULTS_ROOT.parent
        / "V2_DISAGREEMENT"
        / "03_FINAL"
        / "predictions_v2.parquet"
    )
    v2 = pd.read_parquet(v2_path)[["clip_uid", "sigma_judge"]]
    merged = merged.merge(v2, on="clip_uid", how="left", validate="one_to_one")
    merged["sigma_judge"] = merged.sigma_judge.fillna(merged.sigma_judge.median())
    merged["teacher_student_abs_difference"] = np.abs(
        merged.teacher_predicted_quality - merged.predicted_quality
    )
    merged["conformal_interval_width"] = merged.upper_quality - merged.lower_quality
    error_columns = [
        "conformal_interval_width",
        "teacher_student_abs_difference",
        "reference_distance",
        "reference_dispersion",
        "perturbed_prediction_change",
        "perturbed_top_phase_agreement",
        "ensemble_quality_sd",
        "open_set",
    ]
    disagreement_columns = [
        "sigma_judge",
        "reference_distance",
        "reference_dispersion",
        "perturbed_prediction_change",
        "perturbed_top_phase_agreement",
        "open_set",
    ]
    error_features = merged[error_columns].astype(float).to_numpy()
    disagreement_features = merged[disagreement_columns].astype(float).to_numpy()
    train = merged.analysis_role.isin(("fit", "validation")).to_numpy()
    calibration = (merged.analysis_role == "calibration").to_numpy()
    absolute_error = np.abs(
        merged.predicted_score.to_numpy(float) - merged.dive_score.to_numpy(float)
    )
    error_threshold = float(np.quantile(absolute_error[train], 0.75))
    error_model, error_constant = _fit_binary_risk(
        error_features[train], absolute_error[train] >= error_threshold
    )
    risk_error = _predict_binary(error_model, error_constant, error_features)

    judge_train = train & merged.disagreement_primary_eligible.to_numpy(bool)
    judge_sd = merged.judge_sample_sd.to_numpy(float)
    disagreement_threshold = float(np.quantile(judge_sd[judge_train], 0.75))
    disagreement_model, disagreement_constant = _fit_binary_risk(
        disagreement_features[judge_train],
        judge_sd[judge_train] >= disagreement_threshold,
    )
    risk_disagreement = _predict_binary(
        disagreement_model, disagreement_constant, disagreement_features
    )
    error_percentile = _calibrated_percentile(risk_error[calibration], risk_error)
    disagreement_percentile = _calibrated_percentile(
        risk_disagreement[calibration], risk_disagreement
    )
    priority = np.maximum(error_percentile, disagreement_percentile)
    review_threshold = float(np.quantile(priority[calibration], 0.80))
    review = priority >= review_threshold

    cutoffs = {
        "interval": float(np.quantile(merged.conformal_interval_width[calibration], 0.80)),
        "teacher": float(
            np.quantile(merged.teacher_student_abs_difference[calibration], 0.80)
        ),
        "stability": float(
            np.quantile(merged.perturbed_top_phase_agreement[calibration], 0.20)
        ),
        "disagreement": float(np.quantile(risk_disagreement[calibration], 0.80)),
    }
    reasons = []
    for position, row in enumerate(merged.itertuples(index=False)):
        if bool(row.open_set):
            reason = "相似参考动作不足"
        elif float(row.conformal_interval_width) >= cutoffs["interval"]:
            reason = "预测区间较宽"
        elif float(row.teacher_student_abs_difference) >= cutoffs["teacher"]:
            reason = "教师与透明评分存在明显差异"
        elif float(row.perturbed_top_phase_agreement) <= cutoffs["stability"]:
            reason = "阶段结论对边界扰动较敏感"
        elif risk_disagreement[position] >= cutoffs["disagreement"]:
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
    output_path = V5_RESULTS_ROOT / "07_ANALYSIS" / "review_priority_v5.parquet"
    output.to_parquet(output_path, index=False)
    summary = {
        "error_label_threshold_score": error_threshold,
        "disagreement_label_threshold_sd": disagreement_threshold,
        "review_priority_threshold": review_threshold,
        "calibration_review_fraction": float(review[calibration].mean()),
        "visible_reason_uses_seed_disagreement": False,
        "error_feature_columns": error_columns,
        "disagreement_feature_columns": disagreement_columns,
        "review_priority_sha256": sha256_file(output_path),
    }
    return output, summary


def _score_delta_ci(frame: pd.DataFrame, left: str, right: str, iterations: int, seed: int):
    return _cluster_bootstrap(
        frame,
        lambda sample: _safe_spearman(sample.dive_score, sample[left])
        - _safe_spearman(sample.dive_score, sample[right]),
        iterations,
        seed,
    )


def _write_sota_table(ablation: pd.DataFrame) -> pd.DataFrame:
    measured = ablation.copy()
    measured["status"] = "measured_here"
    measured["same_code_reproduced"] = True
    measured["source"] = "v5 matched comparison"
    literature = pd.DataFrame(
        [
            {
                "method": "MUSDL",
                "spearman": 0.8978,
                "relative_l2": 0.3704,
                "status": "literature",
                "same_code_reproduced": False,
                "source": "RICA2 ECCV 2024 comparison table",
            },
            {
                "method": "TSA",
                "spearman": 0.9203,
                "relative_l2": 0.3420,
                "status": "literature",
                "same_code_reproduced": False,
                "source": "RICA2 ECCV 2024 comparison table",
            },
            {
                "method": "RICA2 deterministic published",
                "spearman": 0.9402,
                "relative_l2": 0.2621,
                "status": "literature_different_model_selection_protocol",
                "same_code_reproduced": False,
                "source": "RICA2 ECCV 2024",
            },
        ]
    )
    output = pd.concat([literature, measured], ignore_index=True, sort=False)
    output.to_csv(V5_RESULTS_ROOT / "07_ANALYSIS" / "sota_comparison_v5.csv", index=False)
    return output


def analyze_v5() -> dict:
    require_v5_frozen()
    contract = load_v5_contract()
    prediction_path = V5_RESULTS_ROOT / "05_FINAL" / "predictions_cfpd_plus_v5.parquet"
    baseline_path = V5_RESULTS_ROOT / "05_FINAL" / "baseline_predictions_v5.parquet"
    stress_path = V5_RESULTS_ROOT / "06_STRESS" / "trace_stress_v5.parquet"
    for path in (prediction_path, baseline_path, stress_path):
        if not path.exists():
            raise RuntimeError(f"Required v5 result missing: {path.name}")
    predictions = pd.read_parquet(prediction_path)
    baseline = pd.read_parquet(baseline_path)[["clip_uid", "predicted_score"]].rename(
        columns={"predicted_score": "baseline_score"}
    )
    stress = pd.read_parquet(stress_path)
    predictions = predictions.merge(baseline, on="clip_uid", validate="one_to_one")
    review, review_training = build_review_priority_v5(predictions, stress)
    test = review.loc[review.official_split == "test"].copy()
    full_metrics = aqa_score_metrics(test.dive_score, test.predicted_score)
    baseline_metrics = aqa_score_metrics(test.dive_score, test.baseline_score)
    teacher_metrics = aqa_score_metrics(test.dive_score, test.teacher_predicted_score)
    iterations = int(contract["statistics"]["cluster_bootstrap_iterations"])
    seed = int(contract["statistics"]["seed"])
    score_frame = test[
        [
            "event_family",
            "dive_score",
            "predicted_score",
            "baseline_score",
            "teacher_predicted_score",
        ]
    ].copy()
    full_teacher_ci = _score_delta_ci(
        score_frame, "predicted_score", "teacher_predicted_score", iterations, seed
    )
    full_baseline_ci = _score_delta_ci(
        score_frame, "predicted_score", "baseline_score", iterations, seed + 1
    )
    ablation = pd.read_csv(V5_RESULTS_ROOT / "05_FINAL" / "ablation_summary_v5.csv")
    sota = _write_sota_table(ablation)

    trace_test = stress.loc[stress.official_split == "test"].copy()
    contribution = test[[f"phase_{phase}_contribution" for phase in PHASES]].to_numpy()
    teacher_phi = test[[f"teacher_phi_{phase}" for phase in PHASES]].to_numpy()
    reconstruction = np.abs(
        test.predicted_quality.to_numpy()
        - (test.base_quality.to_numpy() + contribution.sum(axis=1))
    )
    phase_rho = _safe_spearman(contribution.reshape(-1), teacher_phi.reshape(-1))
    top_match = float(
        np.mean(np.argmax(np.abs(contribution), axis=1) == np.argmax(np.abs(teacher_phi), axis=1))
    )
    intervention_ci = _cluster_bootstrap(
        trace_test,
        lambda sample: float(np.median(sample.targeted_minus_random)),
        iterations,
        seed + 2,
    )
    stress_summary = json.loads(
        (V5_RESULTS_ROOT / "06_STRESS" / "stress_summary_v5.json").read_text(
            encoding="utf-8"
        )
    )
    trace_pass = bool(
        float(reconstruction.max()) <= 1e-6
        and phase_rho >= float(contract["publication_gates"]["minimum_phase_teacher_spearman"])
        and top_match >= float(contract["publication_gates"]["minimum_top_phase_match"])
        and intervention_ci[0] > 0
    )
    stability_pass = bool(
        stress_summary["median_contribution_cosine"]
        >= float(contract["publication_gates"]["minimum_contribution_cosine"])
        and stress_summary["median_perturbed_top_phase_agreement"] >= 0.75
    )

    panel = review.loc[
        (review.official_split == "test") & review.disagreement_primary_eligible
    ].copy()
    panel["absolute_error"] = np.abs(panel.predicted_score - panel.dive_score)
    accepted = ~panel.review_flag_20pct
    all_mae = float(panel.absolute_error.mean())
    accepted_mae = float(panel.loc[accepted, "absolute_error"].mean())
    accepted_reduction = (all_mae - accepted_mae) / all_mae if all_mae > 0 else 0.0
    high_threshold = float(
        np.quantile(
            review.loc[
                review.analysis_role.isin(("fit", "validation"))
                & review.disagreement_primary_eligible,
                "judge_sample_sd",
            ],
            0.75,
        )
    )
    high = panel.judge_sample_sd >= high_threshold
    population_rate = float(high.mean())
    reviewed_rate = float(high[panel.review_flag_20pct].mean()) if panel.review_flag_20pct.any() else 0.0
    enrichment = reviewed_rate / population_rate if population_rate > 0 else 0.0
    disagreement_rho = _safe_spearman(panel.risk_disagreement, panel.judge_sample_sd)
    rng = np.random.default_rng(seed)
    random_reductions = []
    review_count = int(panel.review_flag_20pct.sum())
    for _ in range(int(contract["statistics"]["random_review_iterations"])):
        reviewed = rng.choice(len(panel), size=review_count, replace=False)
        keep = np.ones(len(panel), dtype=bool)
        keep[reviewed] = False
        random_mae = float(panel.absolute_error.to_numpy()[keep].mean())
        random_reductions.append((all_mae - random_mae) / all_mae if all_mae > 0 else 0.0)

    review_bootstrap = panel[
        ["event_family", "absolute_error", "review_flag_20pct", "judge_sample_sd"]
    ].copy()
    review_reduction_ci = _cluster_bootstrap(
        review_bootstrap,
        lambda sample: (
            float(sample.absolute_error.mean())
            - float(sample.loc[~sample.review_flag_20pct, "absolute_error"].mean())
        )
        / float(sample.absolute_error.mean())
        if float(sample.absolute_error.mean()) > 0
        and (~sample.review_flag_20pct).any()
        else 0.0,
        iterations,
        seed + 3,
    )
    review_pass = accepted_reduction >= float(
        contract["publication_gates"]["minimum_review_error_reduction"]
    )
    enrichment_pass = enrichment >= float(
        contract["publication_gates"]["minimum_disagreement_enrichment"]
    )

    full_teacher_delta = full_metrics["spearman"] - teacher_metrics["spearman"]
    full_baseline_delta = full_metrics["spearman"] - baseline_metrics["spearman"]
    strong = bool(
        full_teacher_delta > 0
        and full_baseline_delta
        >= -float(contract["publication_gates"]["maximum_drop_from_adapted_baseline"])
        and trace_pass
        and stability_pass
        and review_pass
        and enrichment_pass
    )
    application = bool(
        full_metrics["spearman"]
        >= teacher_metrics["spearman"]
        - float(
            contract["publication_gates"]["application_maximum_drop_from_strong_baseline"]
        )
        and trace_pass
        and stability_pass
        and review_pass
    )
    sports = bool(
        full_metrics["spearman"]
        >= teacher_metrics["spearman"]
        - float(
            contract["publication_gates"]["application_maximum_drop_from_strong_baseline"]
        )
        and trace_pass
        and stability_pass
    )
    decision = (
        "FRONTIERS_PSYCHOLOGY_STRONG"
        if strong
        else "FRONTIERS_PSYCHOLOGY_APPLICATION"
        if application
        else "SPORTS_TECHNOLOGY"
        if sports
        else "NO_GO"
    )
    result = {
        "status": "ANALYZED",
        "decision": decision,
        "score": {
            "teacher": teacher_metrics,
            "adapted_baseline": baseline_metrics,
            "cfpd_plus": full_metrics,
            "cfpd_minus_teacher_spearman": full_teacher_delta,
            "cfpd_minus_teacher_cluster_ci": full_teacher_ci,
            "cfpd_minus_adapted_baseline_spearman": full_baseline_delta,
            "cfpd_minus_adapted_baseline_cluster_ci": full_baseline_ci,
            "literature_comparison_rows": len(sota),
        },
        "trace": {
            "reconstruction_max_abs_error": float(reconstruction.max()),
            "student_teacher_phase_spearman": phase_rho,
            "top_phase_match": top_match,
            "targeted_minus_random_cluster_ci": intervention_ci,
            "faithfulness_pass": trace_pass,
            "stress": stress_summary,
            "stability_pass": stability_pass,
        },
        "review": {
            "n": len(panel),
            "all_automatic_mae": all_mae,
            "accepted_80pct_mae": accepted_mae,
            "accepted_error_reduction": accepted_reduction,
            "accepted_error_reduction_cluster_ci": review_reduction_ci,
            "disagreement_spearman": disagreement_rho,
            "high_disagreement_threshold": high_threshold,
            "disagreement_enrichment": enrichment,
            "random_review_reduction_95pct": tuple(
                float(value) for value in np.quantile(random_reductions, (0.025, 0.975))
            ),
            "review_pass": review_pass,
            "enrichment_pass": enrichment_pass,
            "training": review_training,
            "claim_boundary": "Retrospective review prioritization; no judge viewed AI output.",
        },
        "publication": {
            "frontiers_psychology_strong": strong,
            "frontiers_psychology_application": application,
            "sports_technology": sports,
            "material_passport": "experiment-agent / analysis / 2026-08-18 / ANALYZED / trustdive_cfpd_plus_v5",
        },
    }
    write_json(V5_RESULTS_ROOT / "07_ANALYSIS" / "analysis_summary_v5.json", result)
    return result
