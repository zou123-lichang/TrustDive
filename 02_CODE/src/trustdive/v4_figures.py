from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import ZipFrameStore
from .util import write_json
from .v4_counterfactual import PHASES
from .v4_data import V4_RESULTS_ROOT, v4_paths


COLORS = {
    "teacher": "#5F6B73",
    "student": "#3F7CAC",
    "takeoff": "#4C78A8",
    "flight": "#F2A65A",
    "entry": "#59A14F",
    "risk": "#B55D60",
    "neutral": "#B8BEC2",
}


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
    outputs = {}
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": 600}),
        (".png", {"dpi": 180}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[suffix.lstrip(".")] = str(path)
    plt.close(fig)
    return outputs


def _panel(axis, label: str) -> None:
    axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="bottom")


def _figure1(folder: Path) -> dict:
    fig, axis = plt.subplots(figsize=(7.2, 2.35))
    axis.axis("off")
    nodes = [
        ("Strong teacher", "RICA² score model"),
        ("8 hybrids", "query/reference phases"),
        ("Exact Shapley", "counterfactual targets"),
        ("CFPD student", "3 additive outputs"),
        ("Selective review", "error or disagreement"),
    ]
    positions = np.linspace(0.025, 0.825, len(nodes))
    colors = [COLORS["teacher"], COLORS["takeoff"], COLORS["flight"], COLORS["student"], COLORS["risk"]]
    for index, ((title, subtitle), x, color) in enumerate(zip(nodes, positions, colors)):
        patch = mpl.patches.FancyBboxPatch(
            (x, 0.34), 0.15, 0.35, boxstyle="round,pad=0.012", facecolor="white", edgecolor=color, linewidth=1.5
        )
        axis.add_patch(patch)
        axis.text(x + 0.075, 0.57, title, ha="center", va="center", fontweight="bold")
        axis.text(x + 0.075, 0.43, subtitle, ha="center", va="center", fontsize=6, color="#555555")
        if index < len(nodes) - 1:
            axis.annotate("", xy=(positions[index + 1] - 0.01, 0.515), xytext=(x + 0.16, 0.515), arrowprops={"arrowstyle": "->", "color": "#555555"})
    axis.text(0.5, 0.91, "Counterfactual Phase Distillation for traceable diving assessment", ha="center", fontsize=10, fontweight="bold")
    axis.text(0.5, 0.12, "Phase values are model-attributed evidence, not reconstructed judge deductions.", ha="center", color="#555555")
    return _save(fig, folder / "figure1_cfpd_system")


def _figure2(folder: Path, predictions: pd.DataFrame, analysis: dict) -> dict:
    test = predictions.loc[predictions.official_split == "test"].copy()
    teacher_score = 3.0 * test.difficulty * test.teacher_predicted_quality
    source = test[["clip_uid", "event_family", "dive_score", "predicted_score"]].copy()
    source["teacher_score"] = teacher_score
    source.to_csv(folder / "figure2_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    axes[0].scatter(test.dive_score, teacher_score, s=9, alpha=0.38, color=COLORS["teacher"], label="RICA² teacher")
    axes[0].scatter(test.dive_score, test.predicted_score, s=9, alpha=0.38, color=COLORS["student"], label="CFPD student")
    limits = [float(min(test.dive_score.min(), teacher_score.min(), test.predicted_score.min())), float(max(test.dive_score.max(), teacher_score.max(), test.predicted_score.max()))]
    axes[0].plot(limits, limits, linestyle="--", linewidth=0.8, color="#777777")
    axes[0].set(xlabel="Official score", ylabel="Predicted score", title="Complete official test set (n=749)")
    axes[0].legend(fontsize=6)
    _panel(axes[0], "a")
    labels = ["Teacher", "CFPD"]
    values = [analysis["score"]["teacher"]["spearman"], analysis["score"]["student"]["spearman"]]
    axes[1].barh(labels, values, color=[COLORS["teacher"], COLORS["student"]])
    axes[1].axvline(values[0] - 0.03, linestyle="--", linewidth=0.9, color=COLORS["risk"], label="non-inferiority floor")
    axes[1].set(xlabel="Spearman correlation", title="Score-ranking performance", xlim=(max(0, min(values) - 0.08), 1.0))
    axes[1].legend(fontsize=6)
    _panel(axes[1], "b")
    fig.tight_layout()
    return _save(fig, folder / "figure2_score_performance")


def _figure3(folder: Path, predictions: pd.DataFrame, stress: pd.DataFrame, analysis: dict) -> dict:
    test = predictions.loc[(predictions.official_split == "test") & ~predictions.open_set].copy()
    student = test[[f"phase_{phase}_contribution" for phase in PHASES]].to_numpy().reshape(-1)
    teacher = test[[f"teacher_phi_{phase}" for phase in PHASES]].to_numpy().reshape(-1)
    source = pd.DataFrame({"student_contribution": student, "teacher_shapley": teacher, "phase": np.tile(PHASES, len(test))})
    source.to_csv(folder / "figure3_source.csv", index=False)
    stress_test = stress.loc[stress.official_split == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    for phase in PHASES:
        selected = source.phase == phase
        axes[0].scatter(source.loc[selected, "teacher_shapley"], source.loc[selected, "student_contribution"], s=8, alpha=0.35, color=COLORS[phase], label=phase)
    axes[0].axline((0, 0), slope=1, linestyle="--", linewidth=0.8, color="#777777")
    axes[0].set(xlabel="Teacher exact-Shapley effect", ylabel="Student phase contribution", title=f"Phase fidelity, ρ={analysis['trace']['student_teacher_phase_spearman']:.2f}")
    axes[0].legend(fontsize=6)
    _panel(axes[0], "a")
    effect = [float(stress_test.targeted_teacher_effect.median()), float(stress_test.random_teacher_effect.median())]
    axes[1].bar(["Highest CFPD phase", "Random phase"], effect, color=[COLORS["student"], COLORS["neutral"]])
    axes[1].set(ylabel="Median teacher score change", title="Independent counterfactual intervention")
    axes[1].tick_params(axis="x", rotation=8)
    _panel(axes[1], "b")
    fig.tight_layout()
    return _save(fig, folder / "figure3_counterfactual_fidelity")


def _figure4(folder: Path, analysis: dict) -> dict:
    stress = analysis["trace"]["stress"]
    rows = pd.DataFrame(
        {
            "metric": ["Top phase", "Contribution cosine", "Seed κ", "Teacher-stage match"],
            "value": [stress["median_perturbed_top_phase_agreement"], stress["median_contribution_cosine"], stress["cross_seed_fleiss_kappa"], stress["top_phase_teacher_match"]],
            "threshold": [0.75, 0.90, 0.40, 0.80],
        }
    )
    rows.to_csv(folder / "figure4_source.csv", index=False)
    fig, axis = plt.subplots(figsize=(7.2, 2.55))
    x = np.arange(len(rows))
    axis.bar(x, rows.value, color=[COLORS[phase] for phase in PHASES] + [COLORS["student"]])
    axis.scatter(x, rows.threshold, marker="_", s=220, linewidth=2, color=COLORS["risk"], label="prespecified gate")
    axis.set_xticks(x, rows.metric)
    axis.set(ylabel="Agreement / similarity", title="Attribution stability under boundary, token, noise and seed perturbations", ylim=(min(-0.05, rows.value.min() - 0.1), 1.05))
    axis.legend(fontsize=6)
    fig.tight_layout()
    return _save(fig, folder / "figure4_stability")


def _figure5(folder: Path, review: pd.DataFrame, analysis: dict) -> dict:
    panel = review.loc[(review.official_split == "test") & review.disagreement_primary_eligible].copy()
    panel["absolute_error"] = np.abs(panel.predicted_score - panel.dive_score)
    ordered = panel.sort_values("review_priority")
    ordered["coverage"] = np.arange(1, len(ordered) + 1) / len(ordered)
    ordered["selective_mae"] = ordered.absolute_error.expanding().mean()
    ordered[["clip_uid", "coverage", "selective_mae", "review_priority", "judge_sample_sd", "review_flag_20pct", "review_reason"]].to_csv(folder / "figure5_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    axes[0].plot(ordered.coverage, ordered.selective_mae, color=COLORS["risk"], linewidth=1.8)
    axes[0].axvline(0.8, linestyle="--", linewidth=0.9, color="#777777")
    axes[0].set(xlabel="Automatically accepted coverage", ylabel="Mean absolute score error", title="Coverage–risk trade-off")
    _panel(axes[0], "a")
    high = panel.judge_sample_sd >= analysis["review"]["training"]["disagreement_label_threshold_sd"]
    rates = [panel.loc[high, "review_flag_20pct"].mean(), panel.loc[~high, "review_flag_20pct"].mean()]
    axes[1].bar(["High disagreement", "Other"], rates, color=[COLORS["risk"], COLORS["neutral"]])
    axes[1].set(ylabel="Review selection rate", title=f"Disagreement enrichment: {analysis['review']['disagreement_enrichment']:.2f}×")
    _panel(axes[1], "b")
    fig.tight_layout()
    return _save(fig, folder / "figure5_selective_review")


def _figure6(folder: Path, review: pd.DataFrame) -> dict:
    test = review.loc[review.official_split == "test"].copy()
    test["absolute_error"] = np.abs(test.predicted_score - test.dive_score)
    ordered = test.sort_values("absolute_error")
    selected = [
        ("Low error", ordered.iloc[int(0.10 * (len(ordered) - 1))]),
        ("Median error", ordered.iloc[int(0.50 * (len(ordered) - 1))]),
        ("High error", ordered.iloc[int(0.90 * (len(ordered) - 1))]),
        ("High review risk", test.sort_values("review_priority").iloc[-1]),
    ]
    pd.DataFrame([{"case": label, "clip_uid": row.clip_uid, "absolute_error": row.absolute_error, "review_priority": row.review_priority} for label, row in selected]).to_csv(folder / "figure6_source.csv", index=False)
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.0), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    with np.load(V4_RESULTS_ROOT / "02_COUNTERFACTUAL" / "reference_map_v4.npz", allow_pickle=False) as payload:
        phase_labels = payload["phase_labels"].astype(np.int8)
    with np.load(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_sequences_v4.npz", allow_pickle=False) as payload:
        sequence_clip_uids = payload["clip_uid"].astype(str)
    clip_to_index = {str(value): index for index, value in enumerate(sequence_clip_uids)}
    with ZipFrameStore(v4_paths().trimmed_zip) as store:
        for row_index, (label, row) in enumerate(selected):
            images = store.load(row.source, int(row.instance), 9)
            for phase_index, phase in enumerate(PHASES):
                positions = np.flatnonzero(phase_labels[clip_to_index[str(row.clip_uid)]] == phase_index)
                image = images[int(positions[len(positions) // 2])]
                axes[row_index, phase_index].imshow(image)
                axes[row_index, phase_index].axis("off")
                if row_index == 0:
                    axes[row_index, phase_index].set_title(phase.capitalize(), color=COLORS[phase], fontweight="bold")
            values = [row[f"phase_{phase}_contribution"] for phase in PHASES]
            axes[row_index, 3].barh(PHASES, values, color=[COLORS[phase] for phase in PHASES])
            axes[row_index, 3].axvline(0, color="#555555", linewidth=0.7)
            axes[row_index, 3].set_title(f"{label}\nerror={row.absolute_error:.2f}; risk={row.review_priority:.2f}", fontsize=7)
            axes[row_index, 3].set_xlabel("Quality contribution")
    fig.suptitle("Mechanically selected real phase frames and CFPD evidence", y=0.995, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, folder / "figure6_cases")


def render_reports_v4() -> dict:
    _style()
    folder = V4_RESULTS_ROOT / "figures_v4"
    analysis_path = V4_RESULTS_ROOT / "06_ANALYSIS" / "analysis_summary_v4.json"
    if not analysis_path.exists():
        raise RuntimeError("Run v4 analysis before rendering reports")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis["decision"] == "NO_GO":
        result = {"status": "LOCKED", "reason": "Formal figures are disabled by the v4 No-Go gate"}
        write_json(folder / "render_status_v4.json", result)
        return result
    predictions = pd.read_parquet(V4_RESULTS_ROOT / "04_FINAL" / "predictions_v4.parquet")
    stress = pd.read_parquet(V4_RESULTS_ROOT / "05_STRESS" / "trace_stress_v4.parquet")
    review = pd.read_parquet(V4_RESULTS_ROOT / "06_ANALYSIS" / "review_priority_v4.parquet")
    figure_contract = {
        "core_conclusion": "CFPD retains teacher-level score ranking while making phase evidence independently counterfactual, stable, and usable for selective review.",
        "evidence_chain": ["system", "score", "counterfactual fidelity", "stability", "selective review", "mechanical cases"],
        "archetype": "schematic-led composite plus quantitative grid and image plate",
        "backend": "Python/matplotlib only",
        "exports": ["SVG editable text", "PDF Type 42", "600 dpi TIFF", "PNG preview", "source CSV"],
        "review_risks": ["model evidence is not judge cognition", "cases are mechanically selected", "no failed pose overlay", "official test n=749; panel n=325"],
    }
    write_json(folder / "figure_contract_v4.json", figure_contract)
    outputs = {
        "figure1": _figure1(folder),
        "figure2": _figure2(folder, predictions, analysis),
        "figure3": _figure3(folder, predictions, stress, analysis),
        "figure4": _figure4(folder, analysis),
        "figure5": _figure5(folder, review, analysis),
        "figure6": _figure6(folder, review),
    }
    result = {
        "status": "RENDERED",
        "decision": analysis["decision"],
        "outputs": outputs,
        "qa": {"python_only": True, "editable_vector": True, "raster_dpi": 600, "source_data_present": True, "failed_pose_not_used": True},
    }
    write_json(folder / "render_status_v4.json", result)
    return result
