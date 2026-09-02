from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from .metrics import score_metrics
from .util import write_json
from .v3_data import V3_RESULTS_ROOT, load_v3_contract, require_v3_frozen
from .v3_modeling import PHASES, safe_spearman


def _cluster_bootstrap(frame: pd.DataFrame, statistic, iterations: int, seed: int) -> tuple[float, float]:
    clusters = frame.event_family.drop_duplicates().to_numpy()
    grouped = {cluster: frame.loc[frame.event_family == cluster] for cluster in clusters}
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        bootstrap = pd.concat([grouped[item] for item in sampled], ignore_index=True)
        values[index] = statistic(bootstrap)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return (float("nan"), float("nan"))
    return tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))


def _fleiss_kappa(assignments: np.ndarray, categories: int = 3) -> float:
    assignments = np.asarray(assignments, dtype=int)
    counts = np.stack([(assignments == category).sum(axis=1) for category in range(categories)], axis=1)
    raters = assignments.shape[1]
    if raters < 2:
        return float("nan")
    agreement = ((counts**2).sum(axis=1) - raters) / (raters * (raters - 1))
    marginals = counts.sum(axis=0) / counts.sum()
    expected = float((marginals**2).sum())
    return float((agreement.mean() - expected) / (1.0 - expected)) if expected < 1 else 1.0


def _score_analysis(predictions: pd.DataFrame) -> dict:
    test = predictions.official_split == "test"
    current = score_metrics(predictions.loc[test, "dive_score"], predictions.loc[test, "predicted_score"])
    baseline_root = V3_RESULTS_ROOT.parent / "V2_DISAGREEMENT" / "03_FINAL"
    baseline = {}
    for name in ("global", "relative", "phase_relative", "trustdive_d"):
        payload = json.loads((baseline_root / f"{name}_summary_v2.json").read_text(encoding="utf-8"))
        baseline[name] = payload["official_test_score"]
    best_name = max(baseline, key=lambda key: float(baseline[key]["spearman"]))
    best_rho = float(baseline[best_name]["spearman"])
    source_path = V3_RESULTS_ROOT / "03_FINAL" / "source_isolated_predictions_v3.parquet"
    source = None
    if source_path.exists():
        source_rows = pd.read_parquet(source_path)
        source = score_metrics(source_rows.dive_score, source_rows.predicted_score)
    return {
        "official_test": current,
        "baselines": baseline,
        "best_internal_baseline": best_name,
        "best_internal_baseline_spearman": best_rho,
        "spearman_delta": float(current["spearman"] - best_rho),
        "noninferior": bool(current["spearman"] >= best_rho - float(load_v3_contract()["publication_gates"]["maximum_test_spearman_drop"])),
        "source_isolated": source,
        "source_no_clear_reverse": bool(source is None or source["spearman"] > 0),
    }


def _trace_analysis(predictions: pd.DataFrame, stress: pd.DataFrame) -> dict:
    test = predictions.loc[predictions.official_split == "test"].copy()
    contribution_columns = [f"phase_{phase}_contribution" for phase in PHASES]
    contributions = test[contribution_columns].to_numpy()
    reconstruction = test.base_quality + contributions.sum(axis=1) + test.residual
    top = np.argmax(np.abs(contributions), axis=1)
    low = np.argmin(np.abs(contributions), axis=1)
    ablated = test[[f"ablate_{phase}_quality" for phase in PHASES]].to_numpy()
    kept = test[[f"keep_{phase}_quality" for phase in PHASES]].to_numpy()
    row = np.arange(len(test))
    targeted_effect = np.abs(test.predicted_quality.to_numpy() - ablated[row, top])
    lowest_effect = np.abs(test.predicted_quality.to_numpy() - ablated[row, low])
    rng = np.random.default_rng(20260817)
    random_phase = np.array([rng.choice([p for p in range(3) if p != top[i]]) for i in row])
    random_effect = np.abs(test.predicted_quality.to_numpy() - ablated[row, random_phase])
    actual_top = np.argmax(np.abs(test.predicted_quality.to_numpy()[:, None] - ablated), axis=1)
    fidelity = pd.DataFrame(
        {
            "event_family": test.event_family.to_numpy(),
            "target_minus_random": targeted_effect - random_effect,
        }
    )
    iterations = int(load_v3_contract()["statistics"]["cluster_bootstrap_iterations"])
    deletion_ci = _cluster_bootstrap(
        fidelity, lambda data: float(data.target_minus_random.mean()), iterations, 20260817
    )
    return {
        "reconstruction_max_abs_error": float(np.max(np.abs(reconstruction - test.predicted_quality))),
        "median_attribution_coverage": float(test.attribution_coverage.median()),
        "fraction_coverage_ge_0_70": float((test.attribution_coverage >= 0.70).mean()),
        "median_absolute_residual": float(test.residual.abs().median()),
        "targeted_deletion_mean": float(targeted_effect.mean()),
        "random_deletion_mean": float(random_effect.mean()),
        "lowest_deletion_mean": float(lowest_effect.mean()),
        "target_minus_random_mean": float((targeted_effect - random_effect).mean()),
        "target_minus_random_cluster_ci": list(deletion_ci),
        "top_phase_intervention_match": float(np.mean(top == actual_top)),
        "sufficiency_error_mean": float(np.abs(test.predicted_quality.to_numpy() - kept[row, top]).mean()),
        "boundary_error_mean": float(test.phase_boundary_error_normalized.mean()),
        "faithfulness_pass": bool(deletion_ci[0] > 0 and np.mean(top == actual_top) >= 0.80),
        "interpretation_boundary": "Model faithfulness only; not a judge's psychological deduction.",
    }


def _stability_analysis(predictions: pd.DataFrame, stress: pd.DataFrame) -> dict:
    test_ids = set(predictions.loc[predictions.official_split == "test", "clip_uid"])
    baseline = stress.loc[(stress.perturbation_kind == "baseline") & stress.clip_uid.isin(test_ids)].set_index("clip_uid")
    regular = stress.loc[
        stress.perturbation_kind.isin(["boundary", "token_drop", "noise"]) & stress.clip_uid.isin(test_ids)
    ].copy()
    base_contrib = baseline[[f"{phase}_contribution" for phase in PHASES]]
    changed_contrib = regular[[f"{phase}_contribution" for phase in PHASES]].to_numpy()
    reference_contrib = base_contrib.loc[regular.clip_uid].to_numpy()
    numerator = np.sum(changed_contrib * reference_contrib, axis=1)
    denominator = np.linalg.norm(changed_contrib, axis=1) * np.linalg.norm(reference_contrib, axis=1)
    cosine = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 1e-12)
    top_agreement = np.mean(regular.top_phase.to_numpy() == baseline.loc[regular.clip_uid, "top_phase"].to_numpy())
    prediction_change = np.abs(regular.predicted_quality.to_numpy() - baseline.loc[regular.clip_uid, "predicted_quality"].to_numpy())
    seed_rows = stress.loc[(stress.perturbation_kind == "model_seed") & stress.clip_uid.isin(test_ids)].copy()
    seed_map = {phase: index for index, phase in enumerate(PHASES)}
    seed_assignments = seed_rows.assign(code=seed_rows.top_phase.map(seed_map)).pivot(
        index="clip_uid", columns="perturbation_name", values="code"
    ).to_numpy(dtype=int)
    kappa = _fleiss_kappa(seed_assignments)

    counter = stress.loc[(stress.perturbation_kind == "counterfactual") & stress.clip_uid.isin(test_ids)].copy()
    counter["baseline_prediction"] = baseline.loc[counter.clip_uid, "predicted_quality"].to_numpy()
    for phase in PHASES:
        counter[f"baseline_{phase}"] = baseline.loc[counter.clip_uid, f"{phase}_contribution"].to_numpy()
    target_change, non_target_drift = [], []
    for item in counter.itertuples(index=False):
        target = PHASES.index(item.target_phase)
        new = np.array([item.takeoff_contribution, item.flight_contribution, item.entry_contribution])
        old = np.array([getattr(item, f"baseline_{phase}") for phase in PHASES])
        target_change.append(new[target] - old[target])
        non_target_drift.extend(np.abs(np.delete(new - old, target)).tolist())
    prediction_change_cf = counter.predicted_quality.to_numpy() - counter.baseline_prediction.to_numpy()
    correlation = safe_spearman(prediction_change_cf, np.asarray(target_change))
    non_target = float(np.median(non_target_drift)) if non_target_drift else float("nan")
    gates = load_v3_contract()["stress_gates"]
    checks = {
        "top_phase_agreement": float(top_agreement) >= float(gates["minimum_top_phase_agreement"]),
        "contribution_cosine": float(np.median(cosine)) >= float(gates["minimum_contribution_cosine"]),
        "prediction_change": float(np.median(prediction_change)) <= float(gates["maximum_median_prediction_change"]),
        "seed_kappa": kappa >= float(gates["minimum_seed_fleiss_kappa"]),
        "counterfactual_correlation": correlation >= float(gates["minimum_counterfactual_correlation"]),
        "non_target_drift": non_target <= float(gates["maximum_non_target_median_drift"]),
    }
    return {
        "top_phase_agreement": float(top_agreement),
        "median_contribution_cosine": float(np.median(cosine)),
        "median_prediction_change": float(np.median(prediction_change)),
        "cross_seed_fleiss_kappa": float(kappa),
        "counterfactual_prediction_target_contribution_spearman": float(correlation),
        "counterfactual_non_target_median_drift": non_target,
        "checks": checks,
        "stability_pass": bool(all(checks.values())),
    }


def _review_analysis(predictions: pd.DataFrame) -> dict:
    panel = predictions.loc[
        (predictions.official_split == "test") & predictions.disagreement_primary_eligible
    ].copy()
    iterations = int(load_v3_contract()["statistics"]["cluster_bootstrap_iterations"])
    rho = safe_spearman(panel.sigma_judge, panel.judge_sample_sd)
    rho_ci = _cluster_bootstrap(
        panel,
        lambda data: safe_spearman(data.sigma_judge.to_numpy(), data.judge_sample_sd.to_numpy()),
        iterations,
        20260818,
    )
    threshold = float(panel.judge_sample_sd.quantile(0.75))
    high = panel.judge_sample_sd >= threshold
    auroc = float(roc_auc_score(high.astype(int), panel.risk_disagreement))
    flagged = panel.review_recommended.astype(bool)
    flagged_high = float(flagged[high].mean())
    flagged_low = float(flagged[~high].mean())
    enrichment = flagged_high / max(flagged_low, 1e-12)

    def enrichment_stat(data: pd.DataFrame) -> float:
        local_high = data.judge_sample_sd >= threshold
        return float(data.loc[local_high, "review_recommended"].mean()) / max(
            float(data.loc[~local_high, "review_recommended"].mean()), 1e-12
        )

    enrichment_ci = _cluster_bootstrap(panel, enrichment_stat, iterations, 20260819)
    error = np.abs(panel.predicted_quality - panel.execution_quality)
    accepted = ~flagged
    reduction = 1.0 - float(error[accepted].mean() / error.mean())

    def reduction_stat(data: pd.DataFrame) -> float:
        local_error = np.abs(data.predicted_quality - data.execution_quality)
        keep = ~data.review_recommended.astype(bool)
        return 1.0 - float(local_error[keep].mean() / local_error.mean()) if keep.any() else float("nan")

    reduction_ci = _cluster_bootstrap(panel, reduction_stat, iterations, 20260820)
    rng = np.random.default_rng(20260817)
    review_count = int(flagged.sum())
    random_reduction = []
    for _ in range(int(load_v3_contract()["statistics"]["random_review_iterations"])):
        random_flag = np.zeros(len(panel), dtype=bool)
        random_flag[rng.choice(len(panel), size=review_count, replace=False)] = True
        random_reduction.append(1.0 - float(error[~random_flag].mean() / error.mean()))
    sorted_panel = panel.sort_values("review_priority")
    coverage = np.arange(1, len(sorted_panel) + 1) / len(sorted_panel)
    cumulative_risk = np.cumsum(np.abs(sorted_panel.predicted_quality - sorted_panel.execution_quality)) / np.arange(1, len(sorted_panel) + 1)
    aurc = float(np.trapz(cumulative_risk, coverage))
    gates = load_v3_contract()["publication_gates"]
    return {
        "n": int(len(panel)),
        "disagreement_spearman": rho,
        "disagreement_cluster_ci": list(rho_ci),
        "high_disagreement_auroc": auroc,
        "review_fraction": float(flagged.mean()),
        "high_disagreement_flag_rate": flagged_high,
        "lower_disagreement_flag_rate": flagged_low,
        "enrichment_ratio": enrichment,
        "enrichment_cluster_ci": list(enrichment_ci),
        "accepted_error_reduction": reduction,
        "accepted_error_reduction_cluster_ci": list(reduction_ci),
        "random_review_reduction_95pct": [float(value) for value in np.quantile(random_reduction, [0.025, 0.975])],
        "aurc": aurc,
        "selective_review_pass": bool(reduction >= float(gates["minimum_selective_error_reduction"]) and reduction_ci[0] > 0),
        "disagreement_enrichment_pass": bool(enrichment >= float(gates["minimum_disagreement_enrichment_ratio"]) and enrichment_ci[0] > 1),
        "v2_direct_fusion_negative_result": {
            "relative_error_change": 0.0814,
            "status": "PRESERVED_NOT_RECOMPUTED",
        },
    }


def analyze_all_v3() -> dict:
    require_v3_frozen()
    prediction_path = V3_RESULTS_ROOT / "03_FINAL" / "predictions_trace_v3.parquet"
    stress_path = V3_RESULTS_ROOT / "04_STRESS" / "trace_stress_v3.parquet"
    if not prediction_path.exists() or not stress_path.exists():
        raise RuntimeError("Run final training and stress-test before v3 analysis")
    predictions = pd.read_parquet(prediction_path)
    stress = pd.read_parquet(stress_path)
    score = _score_analysis(predictions)
    trace = _trace_analysis(predictions, stress)
    stability = _stability_analysis(predictions, stress)
    review = _review_analysis(predictions)
    psychology = bool(
        score["noninferior"] and score["source_no_clear_reverse"] and trace["faithfulness_pass"]
        and stability["stability_pass"] and review["selective_review_pass"]
        and review["disagreement_enrichment_pass"]
    )
    sports = bool(score["noninferior"] and trace["faithfulness_pass"] and stability["stability_pass"])
    decision = "FRONTIERS_PSYCHOLOGY_APPLICATION_GO" if psychology else "SPORTS_TECHNOLOGY_GO" if sports else "NO_GO"
    result = {
        "status": "ANALYZED",
        "score": score,
        "trace": trace,
        "stability": stability,
        "review": review,
        "publication_decision": decision,
        "claim": (
            "TrustDive-Trace preserves score-ranking performance while providing additive, "
            "intervention-consistent phase evidence and prioritizing difficult or disputed actions "
            "for transparent human review."
            if psychology
            else None
        ),
    }
    write_json(V3_RESULTS_ROOT / "05_ANALYSIS" / "analysis_summary_v3.json", result)
    markdown = [
        "# TrustDive-Trace v3 result decision", "", f"**Decision: {decision}**", "",
        f"- Test Spearman: {score['official_test']['spearman']:.4f}; delta vs best internal baseline: {score['spearman_delta']:+.4f}.",
        f"- Attribution coverage median: {trace['median_attribution_coverage']:.3f}; targeted-minus-random deletion CI: {trace['target_minus_random_cluster_ci']}.",
        f"- Stability pass: {stability['stability_pass']}; top-phase agreement: {stability['top_phase_agreement']:.3f}; seed kappa: {stability['cross_seed_fleiss_kappa']:.3f}.",
        f"- Selective error reduction: {review['accepted_error_reduction']:.2%}; disagreement enrichment: {review['enrichment_ratio']:.2f}x.",
        "", "The preserved v2 direct-fusion result remains negative (+8.14% mean error) and was not recomputed.",
    ]
    (V3_RESULTS_ROOT / "RESULTS_DECISION_V3.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return result
