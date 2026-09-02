from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import aqa_score_metrics
from .util import sha256_file, write_json
from .v7_data import V7_RESULTS_ROOT, load_risk_manifest_v7, load_v7_contract


def _empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(reference, np.asarray(values, dtype=float), side="right") / max(len(reference), 1)


def _top_fraction(values: np.ndarray, fraction: float, force: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    selected = np.zeros(len(values), dtype=bool)
    count = int(np.ceil(float(fraction) * len(values)))
    force = np.zeros(len(values), dtype=bool) if force is None else np.asarray(force, dtype=bool)
    forced_indices = np.flatnonzero(force)
    if len(forced_indices) >= count:
        order = forced_indices[np.argsort(-values[forced_indices], kind="stable")]
        selected[order[:count]] = True
        return selected
    selected[forced_indices] = True
    candidates = np.flatnonzero(~selected)
    order = candidates[np.argsort(-values[candidates], kind="stable")]
    selected[order[: count - int(selected.sum())]] = True
    return selected


def _aurc(error: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(np.asarray(risk, dtype=float), kind="stable")
    accepted = np.cumsum(np.asarray(error, dtype=float)[order]) / np.arange(1, len(order) + 1)
    return float(np.trapz(accepted, dx=1.0 / max(len(order) - 1, 1)))


def _cluster_bootstrap(frame: pd.DataFrame, statistic, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    families = frame.event_family.astype(str).unique()
    grouped = {family: frame[frame.event_family.astype(str) == family] for family in families}
    values = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.choice(families, size=len(families), replace=True)
        draw = pd.concat([grouped[family] for family in sampled], ignore_index=True)
        values[iteration] = statistic(draw)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _fit_probability(features: pd.DataFrame, labels: np.ndarray, seed: int):
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float(labels[0]) if len(labels) else 0.0
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    model.fit(features, labels)
    return model


def _predict_probability(model, features: pd.DataFrame) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(features), model, dtype=float)
    return model.predict_proba(features)[:, 1]


def _binary_metrics(label: np.ndarray, probability: np.ndarray) -> dict:
    label = np.asarray(label, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    if len(np.unique(label)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
    return {
        "auroc": float(roc_auc_score(label, probability)),
        "auprc": float(average_precision_score(label, probability)),
        "brier": float(brier_score_loss(label, probability)),
    }


def _strategy_metrics(frame: pd.DataFrame, risk: np.ndarray, fraction: float, force: np.ndarray | None = None) -> dict:
    selected = _top_fraction(risk, fraction, force)
    accepted = ~selected
    all_mae = float(frame.absolute_error.mean())
    accepted_mae = float(frame.loc[accepted, "absolute_error"].mean())
    high_error = frame.high_error_risk.to_numpy(dtype=bool)
    return {
        "review_fraction": float(selected.mean()),
        "accepted_mae": accepted_mae,
        "accepted_mae_reduction": (all_mae - accepted_mae) / max(all_mae, 1e-8),
        "high_error_recall_at_budget": float(selected[high_error].mean()) if high_error.any() else float("nan"),
        "high_error_enrichment": float(high_error[selected].mean() / max(high_error.mean(), 1e-8)) if selected.any() else float("nan"),
        "aurc": _aurc(frame.absolute_error.to_numpy(), risk),
        "selected": selected,
    }


def analyze_risk_v7() -> dict:
    contract = load_v7_contract()
    prediction_path = V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet"
    crossfit_path = V7_RESULTS_ROOT / "03_SCORE" / "crossfit_predictions_v7.parquet"
    phase_path = V7_RESULTS_ROOT / "04_PHASE_EVIDENCE" / "phase_evidence_v7.parquet"
    for path in (prediction_path, crossfit_path, phase_path):
        if not path.exists():
            raise RuntimeError(f"Required v7 artifact missing: {path.name}")
    prediction = pd.read_parquet(prediction_path)
    crossfit = pd.read_parquet(crossfit_path)
    phase = pd.read_parquet(phase_path)
    manifest = load_risk_manifest_v7()
    frame = prediction.merge(
        crossfit.drop(columns=[column for column in ("analysis_role", "event_family", "difficulty", "dive_score") if column in crossfit.columns]),
        on="clip_uid", validate="one_to_one",
    ).merge(
        phase.drop(columns=[column for column in ("official_split", "analysis_role", "event_family", "action_type", "difficulty", "dive_score", "open_set") if column in phase.columns]),
        on="clip_uid", validate="one_to_one",
    ).merge(
        manifest[["clip_uid", "high_judge_risk", "judge_risk_threshold", "error_risk_threshold"]],
        on="clip_uid", validate="one_to_one",
    )
    development = frame.analysis_role.isin(("fit", "validation"))
    calibration = frame.analysis_role == "calibration"
    test = frame.analysis_role == "official_test"
    frame["error_gap_feature"] = frame.teacher_adapter_gap
    frame.loc[development, "error_gap_feature"] = frame.loc[development, "oof_teacher_adapter_gap"]
    frame["risk_reference_distance"] = frame.reference_distance
    frame["risk_reference_dispersion"] = frame.reference_dispersion
    frame.loc[development, "risk_reference_distance"] = frame.loc[development, "oof_reference_distance"]
    frame.loc[development, "risk_reference_dispersion"] = frame.loc[development, "oof_reference_dispersion"]
    phase_values = frame[["phi_takeoff", "phi_flight", "phi_entry"]].abs().fillna(0.0).to_numpy()
    frame["phase_abs_sum"] = phase_values.sum(axis=1)
    frame["phase_dominance"] = phase_values.max(axis=1) / np.maximum(phase_values.sum(axis=1), 1e-8)
    frame["phase_boundary_blur"] = 1.0 - frame.phase_boundary_cosine.fillna(0.0)
    error_features = [
        "error_gap_feature", "risk_reference_distance", "risk_reference_dispersion",
        "prediction_interval_width", "open_set",
    ]
    disagreement_features = [
        "phase_abs_sum", "phase_dominance", "phase_boundary_blur",
        "risk_reference_distance", "risk_reference_dispersion", "prediction_interval_width", "open_set",
    ]
    error_truth_development = (
        np.abs(frame.loc[development, "oof_predicted_score"] - frame.loc[development, "dive_score"])
        >= frame.loc[development, "error_risk_threshold"]
    ).to_numpy(dtype=bool)
    seed = int(contract["statistics"]["seed"])
    error_model = _fit_probability(frame.loc[development, error_features], error_truth_development, seed)
    judge_train = development & frame.disagreement_primary_eligible.astype(bool)
    judge_model = _fit_probability(
        frame.loc[judge_train, disagreement_features],
        frame.loc[judge_train, "high_judge_risk"].to_numpy(dtype=bool),
        seed + 1,
    )
    frame["risk_error"] = _predict_probability(error_model, frame[error_features])
    frame["risk_disagreement"] = _predict_probability(judge_model, frame[disagreement_features])
    frame["risk_error_percentile"] = _empirical_percentile(frame.loc[calibration, "risk_error"], frame.risk_error)
    frame["risk_disagreement_percentile"] = _empirical_percentile(frame.loc[calibration, "risk_disagreement"], frame.risk_disagreement)
    frame["review_priority"] = frame[["risk_error_percentile", "risk_disagreement_percentile"]].max(axis=1)
    calibration_threshold = float(np.quantile(
        frame.loc[calibration, "review_priority"],
        1.0 - float(contract["risk_task"]["review_fraction"]),
        method="higher",
    ))
    frame["calibrated_review_flag"] = frame.review_priority >= calibration_threshold
    frame.loc[frame.open_set.astype(bool), "calibrated_review_flag"] = True

    test_frame = frame.loc[test].copy().reset_index(drop=True)
    test_frame["absolute_error"] = np.abs(test_frame.trustdive_predicted_score - test_frame.dive_score)
    test_frame["plain_absolute_error"] = np.abs(test_frame.plain_ridge_predicted_score - test_frame.dive_score)
    test_frame["teacher_absolute_error"] = np.abs(test_frame.teacher_predicted_score - test_frame.dive_score)
    test_frame["high_error_risk"] = test_frame.absolute_error >= test_frame.error_risk_threshold
    review_fraction = float(contract["risk_task"]["review_fraction"])
    strategies = {
        "rica2_uncertainty": _strategy_metrics(test_frame, test_frame.teacher_uncertainty.to_numpy(), review_fraction),
        "error_risk": _strategy_metrics(test_frame, test_frame.risk_error_percentile.to_numpy(), review_fraction),
        "disagreement_risk": _strategy_metrics(test_frame, test_frame.risk_disagreement_percentile.to_numpy(), review_fraction),
        "trustdive_combined": _strategy_metrics(
            test_frame, test_frame.review_priority.to_numpy(), review_fraction, test_frame.open_set.to_numpy(dtype=bool)
        ),
    }
    combined_selected = strategies["trustdive_combined"].pop("selected")
    for key, value in strategies.items():
        if "selected" in value:
            value.pop("selected")
    test_frame["review_budget20"] = combined_selected

    panel = test_frame[test_frame.disagreement_primary_eligible.astype(bool)].copy().reset_index(drop=True)
    panel["high_judge_risk"] = panel.high_judge_risk.astype(bool)
    panel_strategies = {}
    panel_risks = {
        "rica2_uncertainty": panel.teacher_uncertainty.to_numpy(),
        "error_risk": panel.risk_error_percentile.to_numpy(),
        "disagreement_risk": panel.risk_disagreement_percentile.to_numpy(),
        "trustdive_combined": panel.review_priority.to_numpy(),
    }
    for name, risk in panel_risks.items():
        selected = _top_fraction(risk, review_fraction, panel.open_set.to_numpy(dtype=bool) if name == "trustdive_combined" else None)
        panel_strategies[name] = {
            "high_judge_recall_at_budget": float(selected[panel.high_judge_risk].mean()),
            "high_judge_enrichment": float(panel.loc[selected, "high_judge_risk"].mean() / max(panel.high_judge_risk.mean(), 1e-8)),
        }
        if name == "trustdive_combined":
            panel["review_panel_budget20"] = selected

    error_binary = _binary_metrics(test_frame.high_error_risk, test_frame.risk_error)
    judge_binary = _binary_metrics(panel.high_judge_risk, panel.risk_disagreement)
    error_binary["spearman_with_absolute_error"] = float(spearmanr(test_frame.risk_error, test_frame.absolute_error).statistic)
    judge_binary["spearman_with_judge_sd"] = float(spearmanr(panel.risk_disagreement, panel.judge_sample_sd).statistic)

    high = panel.high_judge_risk.to_numpy(dtype=bool)
    teacher_high = float(panel.loc[high, "teacher_absolute_error"].mean())
    teacher_low = float(panel.loc[~high, "teacher_absolute_error"].mean())
    ours_high = float(panel.loc[high, "absolute_error"].mean())
    ours_low = float(panel.loc[~high, "absolute_error"].mean())
    plain_high = float(panel.loc[high, "plain_absolute_error"].mean())
    plain_low = float(panel.loc[~high, "plain_absolute_error"].mean())
    teacher_gap = teacher_high - teacher_low
    ours_gap = ours_high - ours_low
    high_delta = ours_high - teacher_high
    high_reduction = (teacher_high - ours_high) / max(teacher_high, 1e-8)
    directed_gain = (teacher_high - ours_high) - (teacher_low - ours_low)
    iterations = int(contract["statistics"]["cluster_bootstrap_iterations"])
    high_delta_ci = _cluster_bootstrap(
        panel,
        lambda draw: float((draw.loc[draw.high_judge_risk, "absolute_error"] - draw.loc[draw.high_judge_risk, "teacher_absolute_error"]).mean()),
        iterations, seed,
    )
    gain_ci = _cluster_bootstrap(
        panel,
        lambda draw: float(
            (draw.loc[draw.high_judge_risk, "teacher_absolute_error"].mean() - draw.loc[draw.high_judge_risk, "absolute_error"].mean())
            - (draw.loc[~draw.high_judge_risk, "teacher_absolute_error"].mean() - draw.loc[~draw.high_judge_risk, "absolute_error"].mean())
        ),
        iterations, seed + 1,
    )
    risk_weighting_delta_ci = _cluster_bootstrap(
        panel,
        lambda draw: float((draw.loc[draw.high_judge_risk, "absolute_error"] - draw.loc[draw.high_judge_risk, "plain_absolute_error"]).mean()),
        iterations, seed + 5,
    )
    selective_reduction = float(strategies["trustdive_combined"]["accepted_mae_reduction"])
    selective_ci = _cluster_bootstrap(
        test_frame,
        lambda draw: float((draw.absolute_error.mean() - draw.loc[~draw.review_budget20, "absolute_error"].mean()) / max(draw.absolute_error.mean(), 1e-8)),
        iterations, seed + 2,
    )
    phase_test = phase[(phase.analysis_role == "official_test") & (~phase.open_set.astype(bool))].copy()
    phase_match_ci = _cluster_bootstrap(
        phase_test,
        lambda draw: float(draw.top_phase_intervention_match.mean()),
        iterations, seed + 3,
    )
    targeted_ci = _cluster_bootstrap(
        phase_test,
        lambda draw: float(np.median(draw.targeted_intervention_effect - draw.random_intervention_effect)),
        iterations, seed + 4,
    )

    rng = np.random.default_rng(seed)
    random_reductions = []
    count = int(combined_selected.sum())
    for _ in range(int(contract["risk_task"]["random_review_iterations"])):
        random_selected = np.zeros(len(test_frame), dtype=bool)
        random_selected[rng.choice(len(test_frame), size=count, replace=False)] = True
        random_reductions.append(
            (test_frame.absolute_error.mean() - test_frame.loc[~random_selected, "absolute_error"].mean())
            / max(test_frame.absolute_error.mean(), 1e-8)
        )
    random_reductions = np.asarray(random_reductions, dtype=float)

    reason = []
    calibration_rows = frame.loc[calibration]
    for row in frame.itertuples(index=False):
        reasons = []
        if bool(row.open_set):
            reasons.append("insufficient similar references")
        if row.prediction_interval_width >= calibration_rows.prediction_interval_width.quantile(0.8):
            reasons.append("wide prediction interval")
        if row.error_gap_feature >= calibration_rows.error_gap_feature.quantile(0.8):
            reasons.append("teacher-adapter difference")
        if row.phase_boundary_blur >= calibration_rows.phase_boundary_blur.quantile(0.8):
            reasons.append("phase boundary ambiguity")
        if row.risk_disagreement_percentile >= 0.8:
            reasons.append("predicted high judge disagreement")
        reason.append("; ".join(reasons) if reasons else "routine automatic scoring")
    frame["review_reason"] = reason
    budget_lookup = dict(zip(test_frame.clip_uid, test_frame.review_budget20))
    panel_lookup = dict(zip(panel.clip_uid, panel.review_panel_budget20))
    frame["review_budget20"] = frame.clip_uid.map(budget_lookup).eq(True)
    frame["review_panel_budget20"] = frame.clip_uid.map(panel_lookup).eq(True)
    output_columns = [
        "clip_uid", "official_split", "analysis_role", "event_family", "action_type",
        "risk_error", "risk_disagreement", "risk_error_percentile", "risk_disagreement_percentile",
        "review_priority", "calibrated_review_flag", "review_budget20", "review_panel_budget20",
        "review_reason", "open_set", "prediction_interval_width", "judge_sample_sd",
        "disagreement_primary_eligible", "high_judge_risk", "trustdive_predicted_score",
        "teacher_predicted_score", "dive_score",
    ]
    review_path = V7_RESULTS_ROOT / "05_RISK_REVIEW" / "review_priority_v7.parquet"
    frame[output_columns].to_parquet(review_path, index=False)

    teacher_metrics = aqa_score_metrics(test_frame.dive_score, test_frame.teacher_predicted_score)
    ours_metrics = aqa_score_metrics(test_frame.dive_score, test_frame.trustdive_predicted_score)
    spearman_drop = float(teacher_metrics["spearman"] - ours_metrics["spearman"])
    score_enhanced = bool(ours_metrics["spearman"] > teacher_metrics["spearman"] and ours_metrics["mae"] < teacher_metrics["mae"])
    risk_directed = bool(
        high_reduction >= float(contract["publication"]["minimum_high_risk_mae_reduction"])
        and directed_gain > 0
    )
    trace_pass = bool(
        json.loads((V7_RESULTS_ROOT / "04_PHASE_EVIDENCE" / "phase_evidence_summary_v7.json").read_text(encoding="utf-8"))["status"] == "PASS"
        and targeted_ci[0] > 0
        and phase_match_ci[0] > float(contract["attribution"]["chance_top_phase_match"])
    )
    risk_better = bool(
        strategies["trustdive_combined"]["aurc"] < strategies["rica2_uncertainty"]["aurc"]
        or strategies["trustdive_combined"]["high_error_recall_at_budget"] > strategies["rica2_uncertainty"]["high_error_recall_at_budget"]
        or panel_strategies["trustdive_combined"]["high_judge_recall_at_budget"] > panel_strategies["rica2_uncertainty"]["high_judge_recall_at_budget"]
    )
    selective_pass = bool(
        selective_reduction >= float(contract["publication"]["minimum_selective_mae_reduction"])
        and selective_ci[0] > 0
    )
    transparent_noninferior = spearman_drop <= float(contract["publication"]["maximum_transparent_spearman_drop"])
    if risk_directed and trace_pass and selective_pass and risk_better:
        decision = "FRONTIERS_PSYCHOLOGY_STRONG"
    elif transparent_noninferior and trace_pass and risk_better:
        decision = "FRONTIERS_PSYCHOLOGY_APPLICATION"
    elif transparent_noninferior and trace_pass:
        decision = "SPORTS_TECHNOLOGY"
    else:
        decision = "EXPLORATORY_ONLY"

    result = {
        "status": "ANALYZED",
        "publication_decision": decision,
        "score": {
            "teacher": teacher_metrics,
            "trustdive": ours_metrics,
            "spearman_drop": spearman_drop,
            "score_enhanced": score_enhanced,
        },
        "high_judge_risk_scoring": {
            "test_panel_rows": int(len(panel)),
            "high_risk_rows": int(high.sum()),
            "teacher_high_mae": teacher_high,
            "teacher_low_mae": teacher_low,
            "trustdive_high_mae": ours_high,
            "trustdive_low_mae": ours_low,
            "plain_ridge_high_mae": plain_high,
            "plain_ridge_low_mae": plain_low,
            "risk_weighted_minus_plain_high_mae": ours_high - plain_high,
            "risk_weighted_minus_plain_high_mae_cluster_ci": list(risk_weighting_delta_ci),
            "risk_weighting_improves_high_risk_mae": ours_high < plain_high,
            "teacher_risk_gap": teacher_gap,
            "trustdive_risk_gap": ours_gap,
            "trustdive_minus_teacher_high_mae": high_delta,
            "high_mae_reduction_fraction": high_reduction,
            "risk_directed_gain": directed_gain,
            "high_mae_delta_cluster_ci": list(high_delta_ci),
            "risk_directed_gain_cluster_ci": list(gain_ci),
            "rica2_error_is_higher_on_high_disagreement": teacher_gap > 0,
            "risk_directed_gate": risk_directed,
        },
        "risk_detection": {"error": error_binary, "judge_disagreement": judge_binary},
        "review_strategies": strategies,
        "panel_review_strategies": panel_strategies,
        "selective_review": {
            "accepted_mae_reduction": selective_reduction,
            "cluster_ci": list(selective_ci),
            "better_than_random_probability": float(np.mean(selective_reduction > random_reductions)),
            "selective_gate": selective_pass,
            "risk_better_than_rica2_uncertainty": risk_better,
        },
        "phase_evidence": {
            "top_phase_match_cluster_ci": list(phase_match_ci),
            "targeted_minus_random_cluster_ci": list(targeted_ci),
            "trace_gate": trace_pass,
        },
        "claim_boundary": (
            (
                "RICA2 high-disagreement failure wording is supported, but risk weighting did not beat the plain latent-reference Ridge on high-risk MAE; attribute score gains to latent reference adaptation."
                if ours_high >= plain_high else
                "RICA2 high-disagreement failure wording and risk-directed weighting are both supported."
            ) if teacher_gap > 0 else
            "Do not claim that RICA2 is more error-prone on high-disagreement dives; emphasize auditable evidence instead."
        ),
        "review_priority_sha256": sha256_file(review_path),
        "material_passport": "experiment-agent / analysis / 2026-08-20 / ANALYZED / trustdive_risk_v7",
    }
    summary_path = V7_RESULTS_ROOT / "05_RISK_REVIEW" / "analysis_summary_v7.json"
    write_json(summary_path, result)
    decision_text = (
        "# TrustDive-Risk v7 decision\n\n"
        f"**{decision}.**\n\n"
        f"High-disagreement MAE change: {high_reduction:.1%}; selective accepted-MAE change: {selective_reduction:.1%}.\n\n"
        f"{result['claim_boundary']}\n"
    )
    (V7_RESULTS_ROOT / "RESULTS_DECISION_V7.md").write_text(decision_text, encoding="utf-8")
    return result
