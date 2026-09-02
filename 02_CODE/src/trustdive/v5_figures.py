from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .features import ZipFrameStore
from .util import write_json
from .v4_counterfactual import PHASES
from .v5_data import V5_RESULTS_ROOT, load_v5_frame, v5_paths


COLORS = {
    "teacher": "#69757C",
    "baseline": "#8EA6B4",
    "full": "#2E6F9E",
    "takeoff": "#4C78A8",
    "flight": "#E39C44",
    "entry": "#59A14F",
    "risk": "#B55D60",
    "neutral": "#C1C7CA",
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
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def _figure1(folder: Path) -> dict:
    fig, axis = plt.subplots(figsize=(7.2, 2.45))
    axis.axis("off")
    nodes = [
        ("Frozen RICA²", "score anchor"),
        ("RICA²-AR", "same-action references"),
        ("8 hybrids", "exact phase Shapley"),
        ("CFPD+", "weighted + consistent"),
        ("Review queue", "error or disagreement"),
    ]
    positions = np.linspace(0.02, 0.82, len(nodes))
    colors = [COLORS["teacher"], COLORS["baseline"], COLORS["flight"], COLORS["full"], COLORS["risk"]]
    for index, ((title, subtitle), x, color) in enumerate(zip(nodes, positions, colors)):
        patch = mpl.patches.FancyBboxPatch(
            (x, 0.34),
            0.15,
            0.35,
            boxstyle="round,pad=0.012",
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
        )
        axis.add_patch(patch)
        axis.text(x + 0.075, 0.57, title, ha="center", va="center", fontweight="bold")
        axis.text(x + 0.075, 0.43, subtitle, ha="center", va="center", fontsize=6, color="#555555")
        if index < len(nodes) - 1:
            axis.annotate(
                "",
                xy=(positions[index + 1] - 0.01, 0.515),
                xytext=(x + 0.16, 0.515),
                arrowprops={"arrowstyle": "->", "color": "#555555"},
            )
    axis.text(
        0.5,
        0.91,
        "Strong-score adaptation with counterfactually grounded phase evidence",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.12,
        "Outputs support retrospective review; phase values are not reconstructed judge deductions.",
        ha="center",
        color="#555555",
    )
    return _save(fig, folder / "figure1_system")


def _trace_value(frame: pd.DataFrame) -> float:
    contribution = frame[[f"phase_{phase}_contribution" for phase in PHASES]].to_numpy()
    teacher = frame[[f"teacher_phi_{phase}" for phase in PHASES]].to_numpy()
    from scipy.stats import spearmanr

    return float(spearmanr(contribution.reshape(-1), teacher.reshape(-1)).statistic)


def _figure2(folder: Path, predictions: pd.DataFrame, analysis: dict) -> dict:
    test = predictions.loc[predictions.official_split == "test"].copy()
    baseline = pd.read_parquet(V5_RESULTS_ROOT / "05_FINAL" / "baseline_predictions_v5.parquet")
    baseline = baseline.loc[baseline.official_split == "test"].set_index("clip_uid").loc[test.clip_uid]
    source = pd.DataFrame(
        {
            "clip_uid": test.clip_uid,
            "official_score": test.dive_score,
            "teacher_score": test.teacher_predicted_score,
            "adapted_baseline_score": baseline.predicted_score.to_numpy(),
            "cfpd_plus_score": test.predicted_score,
        }
    )
    source.to_csv(folder / "figure2_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].scatter(test.dive_score, test.teacher_predicted_score, s=8, alpha=0.28, color=COLORS["teacher"], label="RICA²")
    axes[0].scatter(test.dive_score, test.predicted_score, s=8, alpha=0.34, color=COLORS["full"], label="CFPD+")
    limits = [float(min(test.dive_score.min(), test.predicted_score.min())), float(max(test.dive_score.max(), test.predicted_score.max()))]
    axes[0].plot(limits, limits, "--", linewidth=0.8, color="#777777")
    axes[0].set(xlabel="Official score", ylabel="Predicted score", title="Official test set (n=749)")
    axes[0].legend(fontsize=6)
    _panel(axes[0], "a")
    methods = ["RICA²", "RICA²-AR", "CFPD+"]
    score = [
        analysis["score"]["teacher"]["spearman"],
        analysis["score"]["adapted_baseline"]["spearman"],
        analysis["score"]["cfpd_plus"]["spearman"],
    ]
    trace = [0.0, _trace_value(baseline), _trace_value(test)]
    for label, x, y, color in zip(methods, score, trace, [COLORS["teacher"], COLORS["baseline"], COLORS["full"]]):
        axes[1].scatter(x, y, s=55, color=color)
        axes[1].text(x + 0.002, y + 0.01, label, fontsize=6)
    axes[1].set(xlabel="Score Spearman", ylabel="Phase–Shapley Spearman", title="Score–traceability trade-off")
    _panel(axes[1], "b")
    fig.tight_layout()
    return _save(fig, folder / "figure2_performance_pareto")


def _phase_image(store, row, phase_labels: np.ndarray, phase: int):
    images = store.load(row.source, int(row.instance), len(phase_labels))
    positions = np.flatnonzero(phase_labels == phase)
    return images[int(positions[len(positions) // 2])]


def _figure3(folder: Path, review: pd.DataFrame) -> dict:
    test = review.loc[review.official_split == "test"].copy()
    test["absolute_error"] = np.abs(test.predicted_score - test.dive_score)
    row = test.sort_values("absolute_error").iloc[len(test) // 2]
    frame = load_v5_frame()
    with np.load(V5_RESULTS_ROOT / "01_REFERENCES" / "reference_map_v5.npz", allow_pickle=False) as payload:
        refs = payload["references"].astype(int)
        labels = payload["phase_labels"].astype(np.int8)
    query_index = int(frame.index[frame.clip_uid == row.clip_uid][0])
    reference_index = int(refs[query_index, 0])
    reference = frame.iloc[reference_index]
    pd.DataFrame(
        [{"query_clip_uid": row.clip_uid, "reference_clip_uid": reference.clip_uid, "selection": "median absolute error"}]
    ).to_csv(folder / "figure3_source.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.75), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    with ZipFrameStore(v5_paths().trimmed_zip) as store:
        for column, phase in enumerate(PHASES):
            axes[0, column].imshow(_phase_image(store, row, labels[query_index], column))
            axes[1, column].imshow(_phase_image(store, reference, labels[reference_index], column))
            axes[0, column].set_title(phase.capitalize(), color=COLORS[phase], fontweight="bold")
            axes[0, column].axis("off")
            axes[1, column].axis("off")
    axes[0, 0].text(-0.08, 0.5, "Query", transform=axes[0, 0].transAxes, rotation=90, va="center", fontweight="bold")
    axes[1, 0].text(-0.08, 0.5, "Reference", transform=axes[1, 0].transAxes, rotation=90, va="center", fontweight="bold")
    values = [row[f"phase_{phase}_contribution"] for phase in PHASES]
    axes[0, 3].barh(PHASES, values, color=[COLORS[phase] for phase in PHASES])
    axes[0, 3].axvline(0, color="#555555", linewidth=0.7)
    axes[0, 3].set(xlabel="Quality contribution", title="Additive phase evidence")
    axes[1, 3].axis("off")
    axes[1, 3].text(0.0, 0.80, f"Official score: {row.dive_score:.2f}")
    axes[1, 3].text(0.0, 0.63, f"Predicted score: {row.predicted_score:.2f}")
    axes[1, 3].text(0.0, 0.46, f"Review priority: {row.review_priority:.2f}")
    axes[1, 3].text(0.0, 0.29, f"Reason: {row.review_reason}", wrap=True)
    fig.suptitle("Mechanically selected query–reference phase comparison", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, folder / "figure3_query_reference")


def _figure4(folder: Path) -> dict:
    pair = pd.read_parquet(V5_RESULTS_ROOT / "03_COUNTERFACTUAL" / "counterfactual_targets_v5.parquet")
    query = pair.groupby("query_index").coalition_7.median().sort_values()
    selected_index = int(query.index[len(query) // 2])
    selected = pair.loc[pair.query_index == selected_index]
    coalition = np.asarray([selected[f"coalition_{mask}"].median() for mask in range(8)])
    phi = np.asarray([selected[f"phi_{phase}"].median() for phase in PHASES])
    pd.DataFrame({"mask": np.arange(8), "teacher_quality": coalition}).to_csv(folder / "figure4_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    axes[0].plot(np.arange(8), coalition, marker="o", color=COLORS["teacher"], linewidth=1.2)
    axes[0].set(xlabel="Phase coalition mask (0–7)", ylabel="Teacher quality", title="All eight counterfactual hybrids")
    _panel(axes[0], "a")
    axes[1].bar(PHASES, phi, color=[COLORS[phase] for phase in PHASES])
    axes[1].axhline(0, color="#555555", linewidth=0.7)
    axes[1].set(ylabel="Exact Shapley effect", title="Teacher phase target")
    _panel(axes[1], "b")
    fig.tight_layout()
    return _save(fig, folder / "figure4_counterfactual_shapley")


def _figure5(folder: Path, analysis: dict) -> dict:
    stress = analysis["trace"]["stress"]
    source = pd.DataFrame(
        {
            "metric": ["Top-phase agreement", "Contribution cosine", "Seed kappa", "Teacher-stage match"],
            "value": [stress["median_perturbed_top_phase_agreement"], stress["median_contribution_cosine"], stress["cross_seed_fleiss_kappa"], stress["top_phase_teacher_match"]],
            "threshold": [0.75, 0.90, 0.40, 0.75],
        }
    )
    source.to_csv(folder / "figure5_source.csv", index=False)
    fig, axis = plt.subplots(figsize=(7.2, 2.55))
    x = np.arange(len(source))
    axis.bar(x, source.value, color=[COLORS["takeoff"], COLORS["flight"], COLORS["entry"], COLORS["full"]])
    axis.scatter(x, source.threshold, marker="_", s=220, linewidth=2, color=COLORS["risk"], label="Prespecified target")
    axis.set_xticks(x, source.metric)
    axis.set(ylabel="Agreement / similarity", title="Phase evidence stability", ylim=(min(-0.05, source.value.min() - 0.1), 1.05))
    axis.legend(fontsize=6)
    fig.tight_layout()
    return _save(fig, folder / "figure5_stability")


def _figure6(folder: Path, review: pd.DataFrame, analysis: dict) -> dict:
    panel = review.loc[(review.official_split == "test") & review.disagreement_primary_eligible].copy()
    panel["absolute_error"] = np.abs(panel.predicted_score - panel.dive_score)
    ordered = panel.sort_values("review_priority")
    ordered["coverage"] = np.arange(1, len(ordered) + 1) / len(ordered)
    ordered["selective_mae"] = ordered.absolute_error.expanding().mean()
    ordered[["clip_uid", "coverage", "selective_mae", "review_priority", "judge_sample_sd", "review_flag_20pct"]].to_csv(folder / "figure6_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    axes[0].plot(ordered.coverage, ordered.selective_mae, color=COLORS["risk"], linewidth=1.8)
    axes[0].axvline(0.8, linestyle="--", linewidth=0.9, color="#777777")
    axes[0].set(xlabel="Automatically accepted coverage", ylabel="Mean absolute score error", title="Coverage–risk trade-off")
    _panel(axes[0], "a")
    threshold = analysis["review"]["high_disagreement_threshold"]
    high = panel.judge_sample_sd >= threshold
    rates = [panel.loc[high, "review_flag_20pct"].mean(), panel.loc[~high, "review_flag_20pct"].mean()]
    axes[1].bar(["High disagreement", "Other"], rates, color=[COLORS["risk"], COLORS["neutral"]])
    axes[1].set(ylabel="Review selection rate", title=f"Disagreement enrichment: {analysis['review']['disagreement_enrichment']:.2f}x")
    _panel(axes[1], "b")
    fig.tight_layout()
    return _save(fig, folder / "figure6_selective_review")


def _qa_exports(outputs: dict) -> dict:
    checks = {}
    for figure, bundle in outputs.items():
        figure_checks = {key: Path(path).exists() and Path(path).stat().st_size > 1000 for key, path in bundle.items()}
        with Image.open(bundle["tiff"]) as image:
            figure_checks["tiff_resolution"] = image.width > 1000 and image.height > 500
        svg_text = Path(bundle["svg"]).read_text(encoding="utf-8", errors="ignore")
        figure_checks["svg_editable_text"] = "<text" in svg_text
        checks[figure] = figure_checks
    return {"checks": checks, "pass": all(all(item.values()) for item in checks.values())}


def render_reports_v5() -> dict:
    _style()
    folder = V5_RESULTS_ROOT / "figures_v5"
    folder.mkdir(parents=True, exist_ok=True)
    analysis_path = V5_RESULTS_ROOT / "07_ANALYSIS" / "analysis_summary_v5.json"
    if not analysis_path.exists():
        raise RuntimeError("Run analyze --protocol v5 --part all before rendering")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis["decision"] == "NO_GO":
        result = {"status": "LOCKED", "reason": "Formal figures are disabled by the v5 No-Go decision"}
        write_json(folder / "render_status_v5.json", result)
        return result
    figure_contract = {
        "core_conclusion": "CFPD+ preserves useful score ranking while adding counterfactually grounded phase evidence and selective-review prioritization.",
        "evidence_chain": ["system", "score and trace Pareto", "real query-reference phases", "exact counterfactuals", "stability", "review allocation"],
        "archetype": "schematic-led composite plus image plate and quantitative validation",
        "backend": "Python/matplotlib only",
        "journal_export": {"width_inches": 7.2, "editable_text": True, "formats": ["SVG", "PDF", "600 dpi TIFF", "PNG"], "source_data": True},
        "review_risks": ["model evidence is not judge cognition", "official split shares source families", "case selection is mechanical", "failed pose route is not used"],
    }
    write_json(folder / "figure_contract_v5.json", figure_contract)
    predictions = pd.read_parquet(V5_RESULTS_ROOT / "05_FINAL" / "predictions_cfpd_plus_v5.parquet")
    review = pd.read_parquet(V5_RESULTS_ROOT / "07_ANALYSIS" / "review_priority_v5.parquet")
    outputs = {
        "figure1": _figure1(folder),
        "figure2": _figure2(folder, predictions, analysis),
        "figure3": _figure3(folder, review),
        "figure4": _figure4(folder),
        "figure5": _figure5(folder, analysis),
        "figure6": _figure6(folder, review, analysis),
    }
    qa = _qa_exports(outputs)
    result = {"status": "RENDERED" if qa["pass"] else "QA_FAIL", "decision": analysis["decision"], "outputs": outputs, "qa": qa}
    write_json(folder / "render_status_v5.json", result)
    return result
