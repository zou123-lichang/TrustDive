from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .util import sha256_file, write_json
from .v6_data import V6_RESULTS_ROOT, load_v6_contract, require_v6_frozen


def _empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(reference, np.asarray(values, dtype=float), side="right") / max(len(reference), 1)


def _aurc(error: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(risk)
    cumulative = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
    return float(np.trapz(cumulative, dx=1.0 / max(len(error) - 1, 1)))


def _cluster_bootstrap(
    frame: pd.DataFrame, statistic, iterations: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    families = frame.event_family.astype(str).unique()
    values = np.empty(iterations, dtype=float)
    grouped = {family: frame[frame.event_family.astype(str) == family] for family in families}
    for iteration in range(iterations):
        sampled = rng.choice(families, size=len(families), replace=True)
        draw = pd.concat([grouped[family] for family in sampled], ignore_index=True)
        values[iteration] = statistic(draw)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def analyze_review_v6() -> dict:
    require_v6_frozen()
    contract = load_v6_contract()
    score_path = V6_RESULTS_ROOT / "05_FINAL" / "final_score_summary_v6.json"
    attribution_path = V6_RESULTS_ROOT / "05_FINAL" / "final_attribution_summary_v6.json"
    for path in (score_path, attribution_path):
        if not path.exists():
            raise RuntimeError(f"Required final result missing: {path.name}")
    score_summary = json.loads(score_path.read_text(encoding="utf-8"))
    attribution_summary = json.loads(attribution_path.read_text(encoding="utf-8"))
    if score_summary.get("status") != "PASS" or attribution_summary.get("status") != "PASS":
        raise RuntimeError("Score and final attribution gates must pass before review analysis")

    prediction = pd.read_parquet(
        V6_RESULTS_ROOT / "05_FINAL" / "adapter_predictions_v6.parquet"
    )
    phase = pd.read_parquet(
        V6_RESULTS_ROOT / "05_FINAL" / "phase_evidence_final_v6.parquet"
    )
    crossfit = pd.read_parquet(
        V6_RESULTS_ROOT / "05_FINAL" / "crossfit_predictions_v6.parquet"
    )[["clip_uid", "crossfit_predicted_score"]]
    frame = prediction.merge(phase.drop(columns=[
        column for column in ("official_split", "analysis_role", "event_family", "action_type", "difficulty", "dive_score", "open_set")
        if column in phase.columns
    ]), on="clip_uid", validate="one_to_one").merge(crossfit, on="clip_uid", validate="one_to_one")
    frame["teacher_adapter_gap"] = np.abs(frame.adapter_predicted_score - frame.teacher_predicted_score)
    frame["phase_instability"] = 1.0 - frame.phase_stability_cosine.fillna(0.0)
    frame["phase_abs_sum"] = frame[["phi_takeoff", "phi_flight", "phi_entry"]].abs().sum(axis=1).fillna(0.0)
    phase_abs = frame[["phi_takeoff", "phi_flight", "phi_entry"]].abs().fillna(0.0).to_numpy()
    frame["phase_dominance"] = np.max(phase_abs, axis=1) / np.maximum(np.sum(phase_abs, axis=1), 1e-8)

    error_features = [
        "teacher_adapter_gap", "reference_distance", "reference_dispersion",
        "prediction_interval_width", "phase_instability", "open_set",
    ]
    disagreement_features = [
        "phase_abs_sum", "phase_dominance", "phase_instability",
        "reference_distance", "reference_dispersion", "open_set",
    ]
    development = frame.analysis_role.isin(("fit", "validation")) & frame.crossfit_predicted_score.notna()
    calibration = frame.analysis_role == "calibration"
    test = frame.analysis_role == "official_test"
    crossfit_error = np.abs(
        frame.loc[development, "crossfit_predicted_score"].to_numpy()
        - frame.loc[development, "dive_score"].to_numpy()
    )
    error_cut = float(np.quantile(crossfit_error, float(contract["risk"]["high_error_quantile"])))
    error_label = (crossfit_error >= error_cut).astype(int)
    error_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=int(contract["statistics"]["seed"])))
    error_model.fit(frame.loc[development, error_features].astype(float), error_label)
    frame["risk_error"] = error_model.predict_proba(frame[error_features].astype(float))[:, 1]

    judge_train = development & frame.disagreement_primary_eligible.astype(bool)
    disagreement_cut = float(np.quantile(
        frame.loc[judge_train, "judge_sample_sd"], float(contract["risk"]["high_disagreement_quantile"])
    ))
    disagreement_label = (frame.loc[judge_train, "judge_sample_sd"].to_numpy() >= disagreement_cut).astype(int)
    disagreement_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=int(contract["statistics"]["seed"])))
    disagreement_model.fit(frame.loc[judge_train, disagreement_features].astype(float), disagreement_label)
    frame["risk_disagreement"] = disagreement_model.predict_proba(frame[disagreement_features].astype(float))[:, 1]

    frame["risk_error_percentile"] = _empirical_percentile(
        frame.loc[calibration, "risk_error"], frame.risk_error
    )
    frame["risk_disagreement_percentile"] = _empirical_percentile(
        frame.loc[calibration, "risk_disagreement"], frame.risk_disagreement
    )
    frame["review_priority"] = frame[["risk_error_percentile", "risk_disagreement_percentile"]].max(axis=1)
    threshold = float(np.quantile(
        frame.loc[calibration, "review_priority"], 1.0 - float(contract["risk"]["review_fraction"]), method="higher"
    ))
    frame["recommend_review"] = frame.review_priority >= threshold
    frame.loc[frame.open_set.astype(bool), "recommend_review"] = True
    reason = []
    for row in frame.itertuples(index=False):
        reasons = []
        if bool(row.open_set):
            reasons.append("insufficient similar references")
        if row.prediction_interval_width >= frame.loc[calibration, "prediction_interval_width"].quantile(0.8):
            reasons.append("wide prediction interval")
        if row.teacher_adapter_gap >= frame.loc[calibration, "teacher_adapter_gap"].quantile(0.8):
            reasons.append("teacher-adapter difference")
        if row.phase_instability >= frame.loc[calibration, "phase_instability"].quantile(0.8):
            reasons.append("phase evidence instability")
        if row.risk_disagreement_percentile >= 0.8:
            reasons.append("predicted high judge disagreement")
        reason.append("; ".join(reasons) if reasons else "routine automatic scoring")
    frame["review_reason"] = reason

    eligible = frame[test & frame.disagreement_primary_eligible.astype(bool)].copy()
    if len(eligible) != int(contract["data"]["official_test_seven_judge"]):
        raise AssertionError("Seven-judge test cohort mismatch")
    eligible["absolute_error"] = np.abs(eligible.adapter_predicted_score - eligible.dive_score)
    accepted = ~eligible.recommend_review.astype(bool)
    all_mae = float(eligible.absolute_error.mean())
    accepted_mae = float(eligible.loc[accepted, "absolute_error"].mean())
    error_reduction = (all_mae - accepted_mae) / all_mae
    high_disagreement = eligible.judge_sample_sd >= float(np.quantile(eligible.judge_sample_sd, 0.75))
    reviewed = eligible.recommend_review.astype(bool)
    enrichment = float(high_disagreement[reviewed].mean() / max(high_disagreement.mean(), 1e-8)) if reviewed.any() else 0.0
    disagreement_rho = float(spearmanr(eligible.risk_disagreement, eligible.judge_sample_sd).statistic)
    high_auroc = float(roc_auc_score(high_disagreement.astype(int), eligible.risk_disagreement))
    aurc = _aurc(eligible.absolute_error.to_numpy(), eligible.review_priority.to_numpy())

    iterations = int(contract["statistics"]["cluster_bootstrap_iterations"])
    seed = int(contract["statistics"]["seed"])
    def reduction_stat(draw: pd.DataFrame) -> float:
        baseline = float(draw.absolute_error.mean())
        automatic = draw.loc[~draw.recommend_review.astype(bool), "absolute_error"]
        return (baseline - float(automatic.mean())) / baseline if len(automatic) and baseline > 0 else float("nan")
    def enrichment_stat(draw: pd.DataFrame) -> float:
        high = draw.judge_sample_sd >= float(np.quantile(draw.judge_sample_sd, 0.75))
        selected = draw.recommend_review.astype(bool)
        return float(high[selected].mean() / max(high.mean(), 1e-8)) if selected.any() else float("nan")
    reduction_ci = _cluster_bootstrap(eligible, reduction_stat, iterations, seed)
    enrichment_ci = _cluster_bootstrap(eligible, enrichment_stat, iterations, seed + 1)

    rng = np.random.default_rng(seed)
    review_count = int(reviewed.sum())
    random_reductions = []
    for _ in range(int(contract["statistics"]["random_review_iterations"])):
        selected = np.zeros(len(eligible), dtype=bool)
        if review_count:
            selected[rng.choice(len(eligible), size=review_count, replace=False)] = True
        random_mae = float(eligible.loc[~selected, "absolute_error"].mean())
        random_reductions.append((all_mae - random_mae) / all_mae)
    random_reductions = np.asarray(random_reductions)
    better_than_random = float(np.mean(error_reduction > random_reductions))

    output_columns = [
        "clip_uid", "official_split", "analysis_role", "event_family", "action_type",
        "risk_error", "risk_disagreement", "review_priority", "recommend_review",
        "review_reason", "open_set", "phase_instability", "teacher_adapter_gap",
        "prediction_interval_width", "judge_sample_sd", "disagreement_primary_eligible",
        "adapter_predicted_score", "dive_score",
    ]
    output_path = V6_RESULTS_ROOT / "06_REVIEW" / "review_priority_v6.parquet"
    frame[output_columns].to_parquet(output_path, index=False)

    score_pass = score_summary.get("status") == "PASS"
    trace_pass = attribution_summary.get("status") == "PASS"
    review_pass = (
        error_reduction >= float(contract["publication_gates"]["minimum_review_error_reduction"])
        and reduction_ci[0] > 0.0
    )
    strong_disagreement = (
        enrichment >= float(contract["publication_gates"]["minimum_disagreement_enrichment"])
        and enrichment_ci[0] > 1.0
    )
    if not score_pass or not trace_pass:
        decision = "NO_GO"
    elif review_pass and strong_disagreement:
        decision = "FRONTIERS_PSYCHOLOGY_STRONG_GO"
    elif review_pass:
        decision = "FRONTIERS_PSYCHOLOGY_APPLICATION_GO"
    else:
        decision = "SPORTS_TECHNOLOGY_GO"
    result = {
        "status": "PASS",
        "publication_decision": decision,
        "score_gate": score_pass,
        "attribution_gate": trace_pass,
        "review_threshold_from_calibration": threshold,
        "test_review_fraction": float(reviewed.mean()),
        "automatic_coverage": float(accepted.mean()),
        "all_automatic_mae": all_mae,
        "accepted_mae": accepted_mae,
        "accepted_mae_reduction_fraction": error_reduction,
        "accepted_mae_reduction_cluster_ci": list(reduction_ci),
        "review_better_than_random_probability": better_than_random,
        "aurc": aurc,
        "judge_disagreement_spearman": disagreement_rho,
        "high_disagreement_auroc": high_auroc,
        "high_disagreement_enrichment": enrichment,
        "high_disagreement_enrichment_cluster_ci": list(enrichment_ci),
        "psychology_review_gate": review_pass,
        "strong_disagreement_gate": strong_disagreement,
        "random_seed_is_not_an_external_reason": True,
        "review_priority_sha256": sha256_file(output_path),
    }
    summary_path = V6_RESULTS_ROOT / "06_REVIEW" / "analysis_summary_v6.json"
    write_json(summary_path, result)
    decision_path = V6_RESULTS_ROOT / "RESULTS_DECISION_V6.md"
    decision_path.write_text(
        "# TrustDive-ECR v6 decision\n\n"
        f"**{decision}.** Final claims remain bounded to transparent review support.\n\n"
        f"Automatic-accept MAE change: {error_reduction:.1%}; high-disagreement enrichment: {enrichment:.2f}x.\n",
        encoding="utf-8",
    )
    return result
