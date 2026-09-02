from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import ZipFrameStore
from .util import write_json
from .v3_data import V3_RESULTS_ROOT, v3_paths
from .v3_modeling import PHASES


COLORS = {"takeoff": "#4C78A8", "flight": "#F2A65A", "entry": "#59A14F", "risk": "#B55D60", "neutral": "#73777B"}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save(fig, stem: Path) -> dict:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {suffix: str(stem.with_suffix(suffix)) for suffix in (".svg", ".pdf", ".tiff", ".png")}


def _panel_label(axis, label: str) -> None:
    axis.text(-0.12, 1.04, label, transform=axis.transAxes, weight="bold", fontsize=9, va="bottom")


def _figure1(folder: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.axis("off")
    boxes = [
        ("Frozen teacher", "score target"),
        ("Transparent student", "takeoff + flight + entry"),
        ("Faithfulness tests", "delete · keep · perturb"),
        ("Dual risk", "error · disagreement"),
        ("Human review", "top 20% priority"),
    ]
    x = np.linspace(0.03, 0.83, len(boxes))
    for index, ((title, subtitle), left) in enumerate(zip(boxes, x)):
        color = COLORS[PHASES[index - 1]] if 1 <= index <= 3 else COLORS["neutral"]
        rect = mpl.patches.FancyBboxPatch(
            (left, 0.35), 0.14, 0.34, boxstyle="round,pad=0.015", facecolor="white", edgecolor=color, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(left + 0.07, 0.57, title, ha="center", va="center", weight="bold")
        ax.text(left + 0.07, 0.44, subtitle, ha="center", va="center", fontsize=6, color="#555555")
        if index < len(boxes) - 1:
            ax.annotate("", xy=(x[index + 1] - 0.01, 0.52), xytext=(left + 0.15, 0.52), arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.text(0.5, 0.92, "TrustDive-Trace: score-preserving evidence for selective review", ha="center", weight="bold", fontsize=10)
    ax.text(0.5, 0.12, "All phase values are model-attributed contributions; they are not reconstructed judge deductions.", ha="center", color="#555555")
    return _save(fig, folder / "figure1_system")


def _figure2(folder: Path, analysis: dict) -> dict:
    trials = pd.read_csv(V3_RESULTS_ROOT / "02_PILOT" / "pilot_trials_v3.csv")
    trials.to_csv(folder / "figure2_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for residual, marker in ((0.0, "o"), (0.25, "s")):
        rows = trials.loc[trials.residual_bound == residual]
        axes[0].scatter(rows.median_coverage, rows.validation_spearman, marker=marker, s=45, label=f"residual ≤ {residual:.2f}", alpha=0.85)
        for row in rows.itertuples(index=False):
            axes[0].text(row.median_coverage + 0.006, row.validation_spearman, f"α={row.alpha:g}", fontsize=5)
    axes[0].axhline(0.7251999, color=COLORS["risk"], linestyle="--", linewidth=0.9, label="pilot floor")
    axes[0].axvline(0.80, color="#777777", linestyle=":", linewidth=0.9)
    axes[0].set(xlabel="Median attribution coverage", ylabel="Validation Spearman", title="Transparency–performance Pareto")
    axes[0].legend(fontsize=6)
    _panel_label(axes[0], "a")
    baseline = analysis["score"]["baselines"]
    labels = list(baseline) + ["v3"]
    values = [baseline[name]["spearman"] for name in baseline] + [analysis["score"]["official_test"]["spearman"]]
    axes[1].barh(labels, values, color=[COLORS["neutral"]] * len(baseline) + [COLORS["takeoff"]])
    axes[1].axvline(analysis["score"]["best_internal_baseline_spearman"] - 0.03, color=COLORS["risk"], linestyle="--", linewidth=0.9)
    axes[1].set(xlabel="Official-test Spearman", title="Score-ranking non-inferiority")
    axes[1].set_xlim(max(0, min(values) - 0.08), min(1, max(values) + 0.03))
    _panel_label(axes[1], "b")
    fig.tight_layout()
    return _save(fig, folder / "figure2_pareto_score")


def _figure3(folder: Path, analysis: dict) -> dict:
    trace = analysis["trace"]
    stability = analysis["stability"]
    source = pd.DataFrame(
        {
            "metric": ["Targeted deletion", "Random deletion", "Lowest deletion"],
            "value": [trace["targeted_deletion_mean"], trace["random_deletion_mean"], trace["lowest_deletion_mean"]],
        }
    )
    source.to_csv(folder / "figure3_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    axes[0].bar(source.metric, source.value, color=[COLORS["takeoff"], "#B7BDC2", "#D9DCDE"])
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(ylabel="Absolute prediction change", title="Intervention fidelity")
    _panel_label(axes[0], "a")
    metrics = ["Top phase", "Cosine", "Seed κ", "CF corr."]
    values = [stability["top_phase_agreement"], stability["median_contribution_cosine"], stability["cross_seed_fleiss_kappa"], stability["counterfactual_prediction_target_contribution_spearman"]]
    thresholds = [0.75, 0.90, 0.40, 0.90]
    x = np.arange(len(values))
    axes[1].bar(x, values, color=[COLORS[phase] for phase in PHASES] + [COLORS["neutral"]])
    axes[1].scatter(x, thresholds, marker="_", s=180, linewidth=2, color=COLORS["risk"], label="prespecified gate")
    axes[1].set_xticks(x, metrics)
    axes[1].set_ylim(min(-0.05, min(values) - 0.1), 1.05)
    axes[1].set(title="Attribution stability", ylabel="Agreement / correlation")
    axes[1].legend(fontsize=6)
    _panel_label(axes[1], "b")
    fig.tight_layout()
    return _save(fig, folder / "figure3_faithfulness_stability")


def _figure4(folder: Path, predictions: pd.DataFrame, analysis: dict) -> dict:
    panel = predictions.loc[(predictions.official_split == "test") & predictions.disagreement_primary_eligible].copy()
    panel["absolute_error"] = np.abs(panel.predicted_quality - panel.execution_quality)
    ordered = panel.sort_values("review_priority")
    ordered["coverage"] = np.arange(1, len(ordered) + 1) / len(ordered)
    ordered["selective_mae"] = ordered.absolute_error.expanding().mean()
    ordered[["clip_uid", "coverage", "selective_mae", "review_priority", "judge_sample_sd", "review_recommended"]].to_csv(folder / "figure4_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    axes[0].plot(ordered.coverage, ordered.selective_mae, color=COLORS["risk"], linewidth=1.8)
    axes[0].axvline(0.8, color="#777777", linestyle="--", linewidth=0.9)
    axes[0].set(xlabel="Automatically accepted coverage", ylabel="Mean absolute error", title="Coverage–risk trade-off")
    _panel_label(axes[0], "a")
    high = panel.judge_sample_sd >= panel.judge_sample_sd.quantile(0.75)
    rates = [panel.loc[high, "review_recommended"].mean(), panel.loc[~high, "review_recommended"].mean()]
    axes[1].bar(["High disagreement", "Other"], rates, color=[COLORS["risk"], "#B7BDC2"])
    axes[1].set(ylabel="Review recommendation rate", ylim=(0, max(rates) * 1.25 + 0.01), title=f"Disagreement enrichment: {analysis['review']['enrichment_ratio']:.2f}×")
    _panel_label(axes[1], "b")
    fig.tight_layout()
    return _save(fig, folder / "figure4_selective_review")


def _figure5(folder: Path, predictions: pd.DataFrame) -> dict:
    test = predictions.loc[predictions.official_split == "test"].copy()
    test["absolute_error"] = np.abs(test.predicted_quality - test.execution_quality)
    ordered = test.sort_values("absolute_error")
    selected = [
        ("Low error", ordered.iloc[max(0, int(0.10 * len(ordered)))]),
        ("Median error", ordered.iloc[int(0.50 * len(ordered))]),
        ("High error", ordered.iloc[min(len(ordered) - 1, int(0.90 * len(ordered)))]),
        ("High review risk", test.sort_values("review_priority").iloc[-1]),
    ]
    pd.DataFrame([{"case": label, "clip_uid": row.clip_uid, "absolute_error": row.absolute_error, "review_priority": row.review_priority} for label, row in selected]).to_csv(folder / "figure5_source.csv", index=False)
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.0), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    with ZipFrameStore(v3_paths().trimmed_zip) as store:
        for row_index, (label, row) in enumerate(selected):
            frames = store.load(row.source, int(row.instance), 3)
            for phase_index, (phase, image) in enumerate(zip(PHASES, frames)):
                axes[row_index, phase_index].imshow(image)
                axes[row_index, phase_index].axis("off")
                if row_index == 0:
                    axes[row_index, phase_index].set_title(phase.capitalize(), color=COLORS[phase], weight="bold")
            contributions = [row[f"phase_{phase}_contribution"] for phase in PHASES]
            axes[row_index, 3].barh(PHASES, contributions, color=[COLORS[phase] for phase in PHASES])
            axes[row_index, 3].axvline(0, color="#555555", linewidth=0.7)
            axes[row_index, 3].set_title(f"{label}\nerror={row.absolute_error:.2f}, risk={row.review_priority:.2f}", fontsize=7)
            axes[row_index, 3].set_xlabel("Quality contribution")
    fig.suptitle("Mechanically selected cases: real phase frames and model-attributed contributions", y=0.995, fontsize=10, weight="bold")
    fig.tight_layout()
    return _save(fig, folder / "figure5_cases")


def _figure6(folder: Path, analysis: dict) -> dict:
    official = analysis["score"]["official_test"]
    source = analysis["score"]["source_isolated"]
    rows = pd.DataFrame(
        {
            "split": ["Official", "Event-family isolated"],
            "spearman": [official["spearman"], source["spearman"] if source else np.nan],
            "mae": [official["mae"], source["mae"] if source else np.nan],
        }
    )
    rows.to_csv(folder / "figure6_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    axes[0].bar(rows.split, rows.spearman, color=[COLORS["takeoff"], COLORS["neutral"]])
    axes[0].set(ylabel="Spearman", title="Score ranking", ylim=(0, 1))
    axes[0].tick_params(axis="x", rotation=10)
    _panel_label(axes[0], "a")
    axes[1].bar(rows.split, rows.mae, color=[COLORS["takeoff"], COLORS["neutral"]])
    axes[1].set(ylabel="Total-score MAE", title="Absolute error")
    axes[1].tick_params(axis="x", rotation=10)
    _panel_label(axes[1], "b")
    fig.tight_layout()
    return _save(fig, folder / "figure6_source_robustness")


def render_reports_v3() -> dict:
    _style()
    folder = V3_RESULTS_ROOT / "figures_v3"
    analysis_path = V3_RESULTS_ROOT / "05_ANALYSIS" / "analysis_summary_v3.json"
    if not analysis_path.exists():
        raise RuntimeError("Run v3 analysis before rendering reports")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    decision = analysis["publication_decision"]
    if decision == "NO_GO":
        result = {"status": "LOCKED", "reason": "Formal figures are disabled by the v3 No-Go gate"}
        write_json(folder / "render_status_v3.json", result)
        return result
    predictions = pd.read_parquet(V3_RESULTS_ROOT / "03_FINAL" / "predictions_trace_v3.parquet")
    contract = {
        "core_conclusion": "Transparent phase attribution remains score-competitive, intervention-consistent, and useful for selective review.",
        "evidence_chain": ["system", "score-versus-coverage", "faithfulness-and-stability", "review-triage", "mechanical-cases", "source-isolation"],
        "archetype": "quantitative grid plus image plate",
        "backend": "Python/matplotlib only",
        "exports": ["SVG editable text", "PDF Type 42", "600 dpi TIFF", "PNG preview", "source CSV"],
        "review_risks": ["attribution is model fidelity, not judge cognition", "cases selected mechanically", "no failed pose overlays"],
    }
    write_json(folder / "figure_contract_v3.json", contract)
    outputs = {
        "figure1": _figure1(folder),
        "figure2": _figure2(folder, analysis),
        "figure3": _figure3(folder, analysis),
        "figure4": _figure4(folder, predictions, analysis),
        "figure5": _figure5(folder, predictions),
        "figure6": _figure6(folder, analysis),
    }
    result = {"status": "RENDERED", "decision": decision, "outputs": outputs, "qa": {"backend_exclusive": True, "editable_vector": True, "raster_dpi": 600, "failed_pose_not_used": True, "source_data_present": True}}
    write_json(folder / "render_status_v3.json", result)
    return result
