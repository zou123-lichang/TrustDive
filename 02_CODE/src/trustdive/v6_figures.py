from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths
from .features import ZipFrameStore
from .util import sha256_file, write_json
from .v6_data import V6_RESULTS_ROOT, load_v6_frame
from .v6_modeling import load_reference_map_v6


PALETTE = {
    "teacher": "#7A869A",
    "ecr": "#266A8A",
    "takeoff": "#4C78A8",
    "flight": "#F2A65A",
    "entry": "#2A9D8F",
    "risk": "#C9574F",
    "neutral": "#D9E1E8",
}


def _configure():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def _save(fig, folder: Path, stem: str) -> dict:
    outputs = {}
    for suffix, kwargs in (
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
    ):
        path = folder / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[suffix] = {"path": str(path), "sha256": sha256_file(path)}
    return outputs


def _panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def _figure_method(folder: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.1, 2.6))
    ax.axis("off")
    boxes = [
        (0.02, "Frozen RICA²\nscore + latents", PALETTE["teacher"]),
        (0.22, "Same-action\ntraining references", PALETTE["neutral"]),
        (0.42, "Bounded latent\nreference adapter", PALETTE["ecr"]),
        (0.62, "8 phase\ncoalitions", PALETTE["flight"]),
        (0.80, "Exact Shapley\n+ review priority", PALETTE["entry"]),
    ]
    for x, text, color in boxes:
        patch = FancyBboxPatch((x, 0.35), 0.16, 0.30, boxstyle="round,pad=0.02", fc=color, ec="white", lw=1.2, alpha=0.92)
        ax.add_patch(patch)
        ax.text(x + 0.08, 0.50, text, ha="center", va="center", color="white" if color != PALETTE["neutral"] else "#263238", fontsize=7.2, fontweight="bold")
    for x in (0.18, 0.38, 0.58, 0.78):
        ax.add_patch(FancyArrowPatch((x, 0.50), (x + 0.035, 0.50), arrowstyle="-|>", mutation_scale=12, color="#556270", lw=1.2))
    ax.text(0.70, 0.20, "takeoff", color=PALETTE["takeoff"], ha="center", fontweight="bold")
    ax.text(0.76, 0.20, "+ flight", color=PALETTE["flight"], ha="center", fontweight="bold")
    ax.text(0.83, 0.20, "+ entry", color=PALETTE["entry"], ha="center", fontweight="bold")
    ax.text(0.5, 0.86, "TrustDive-ECR: score-preserving adaptation and exact phase evidence", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def _figure_performance(folder: Path, prediction: pd.DataFrame, summary: dict):
    import matplotlib.pyplot as plt

    test = prediction[prediction.analysis_role == "official_test"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [1.45, 0.62, 0.62]})
    axes[0].scatter(test.dive_score, test.teacher_predicted_score, s=10, alpha=0.30, color=PALETTE["teacher"], label="RICA²")
    axes[0].scatter(test.dive_score, test.adapter_predicted_score, s=10, alpha=0.45, color=PALETTE["ecr"], label="TrustDive-ECR")
    limits = [float(test.dive_score.min()), float(test.dive_score.max())]
    axes[0].plot(limits, limits, ls="--", lw=0.8, color="#222222")
    axes[0].set(xlabel="Official score", ylabel="Predicted score")
    axes[0].legend(loc="upper left")
    _panel_label(axes[0], "a")
    rho = [summary["teacher_test_metrics"]["spearman"], summary["adapter_test_metrics"]["spearman"]]
    mae = [summary["teacher_test_metrics"]["mae"], summary["adapter_test_metrics"]["mae"]]
    colors = [PALETTE["teacher"], PALETTE["ecr"]]
    axes[1].bar([0, 1], rho, color=colors)
    axes[1].set_ylim(max(0.0, min(rho) - 0.05), min(1.0, max(rho) + 0.03))
    axes[1].set_xticks([0, 1], ["RICA²", "ECR"], rotation=25)
    axes[1].set_ylabel("Spearman ρ")
    _panel_label(axes[1], "b")
    axes[2].bar([0, 1], mae, color=colors)
    axes[2].set_xticks([0, 1], ["RICA²", "ECR"], rotation=25)
    axes[2].set_ylabel("MAE (score points)")
    _panel_label(axes[2], "c")
    fig.tight_layout()
    return fig, test


def _select_case(prediction: pd.DataFrame, review: pd.DataFrame) -> pd.Series:
    merged = prediction.merge(review[["clip_uid", "review_priority"]], on="clip_uid")
    test = merged[(merged.analysis_role == "official_test") & (~merged.open_set.astype(bool))].copy()
    test["error"] = np.abs(test.adapter_predicted_score - test.dive_score)
    target = float(test.error.median())
    return test.iloc[int(np.argmin(np.abs(test.error.to_numpy() - target)))]


def _figure_case(folder: Path, prediction: pd.DataFrame, evidence: pd.DataFrame, review: pd.DataFrame):
    import matplotlib.pyplot as plt

    frame = load_v6_frame().reset_index(drop=True)
    case = _select_case(prediction, review)
    index = int(frame.index[frame.clip_uid == case.clip_uid][0])
    refs = load_reference_map_v6(final=True)["references"]
    ref_index = int(refs[index, 0])
    query = frame.iloc[index]
    reference = frame.iloc[ref_index]
    with ZipFrameStore(Paths().trimmed_zip) as store:
        query_frames = store.load(query.source, int(query.instance), 3)
        ref_frames = store.load(reference.source, int(reference.instance), 3)
    fig = plt.figure(figsize=(7.1, 3.4))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15], hspace=0.08, wspace=0.08)
    for row, (frames, label) in enumerate(((query_frames, "Query"), (ref_frames, "Training reference"))):
        for col, image in enumerate(frames):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(image)
            ax.axis("off")
            if row == 0:
                ax.set_title(("Takeoff", "Flight", "Entry")[col], color=(PALETTE["takeoff"], PALETTE["flight"], PALETTE["entry"])[col], fontweight="bold")
            if col == 0:
                ax.text(-0.08, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=7, fontweight="bold")
    row = evidence.set_index("clip_uid").loc[case.clip_uid]
    ax = fig.add_subplot(grid[:, 3])
    values = [row.reference_baseline, row.phi_takeoff, row.phi_flight, row.phi_entry]
    labels = ["Reference\nbaseline", "Takeoff", "Flight", "Entry"]
    colors = [PALETTE["neutral"], PALETTE["takeoff"], PALETTE["flight"], PALETTE["entry"]]
    bottoms = [0.0]
    for value in values[:-1]:
        bottoms.append(bottoms[-1] + value)
    ax.bar(np.arange(4), values, bottom=bottoms, color=colors, edgecolor="white")
    ax.axhline(row.predicted_quality, color="#222222", lw=1, ls="--")
    ax.set_xticks(np.arange(4), labels, rotation=25, ha="right")
    ax.set_ylabel("Execution-quality contribution")
    ax.set_title("Exact additive decomposition", fontsize=8, fontweight="bold")
    fig.tight_layout()
    source = pd.DataFrame({"component": labels, "value": values, "clip_uid": case.clip_uid})
    return fig, source


def _figure_coalitions(evidence: pd.DataFrame):
    import matplotlib.pyplot as plt

    closed = evidence[~evidence.open_set.astype(bool)].copy()
    closed["spread"] = closed[[f"coalition_{i}" for i in range(8)]].max(axis=1) - closed[[f"coalition_{i}" for i in range(8)]].min(axis=1)
    cases = closed.sort_values("spread").iloc[np.linspace(0, len(closed) - 1, 4).round().astype(int)]
    matrix = cases[[f"coalition_{i}" for i in range(8)]].to_numpy()
    fig, ax = plt.subplots(figsize=(7.1, 2.6))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(8), [f"{i:03b}" for i in range(8)])
    ax.set_xlabel("Phase coalition (takeoff–flight–entry bits)")
    ax.set_yticks(range(4), ["low spread", "mid-low", "mid-high", "high spread"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if matrix[i, j] > np.nanmedian(matrix) else "#1D2D3A")
    fig.colorbar(image, ax=ax, label="Final scorer output")
    _panel_label(ax, "a")
    fig.tight_layout()
    return fig, cases


def _figure_fidelity(evidence: pd.DataFrame):
    import matplotlib.pyplot as plt

    closed = evidence[~evidence.open_set.astype(bool)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    axes[0].scatter(closed.random_intervention_effect, closed.targeted_intervention_effect, s=12, alpha=0.45, color=PALETTE["ecr"])
    limit = float(max(closed.random_intervention_effect.max(), closed.targeted_intervention_effect.max()))
    axes[0].plot([0, limit], [0, limit], ls="--", color="#333333", lw=0.8)
    axes[0].set(xlabel="Random-phase intervention", ylabel="Top-attribution intervention")
    _panel_label(axes[0], "a")
    axes[1].hist(closed.phase_stability_cosine, bins=20, color=PALETTE["flight"], edgecolor="white")
    axes[1].axvline(0.90, ls="--", color=PALETTE["risk"], lw=1)
    axes[1].set(xlabel="Contribution cosine under perturbations", ylabel="Videos")
    _panel_label(axes[1], "b")
    fig.tight_layout()
    return fig, closed


def _figure_review(review: pd.DataFrame):
    import matplotlib.pyplot as plt

    test = review[(review.analysis_role == "official_test") & review.disagreement_primary_eligible.astype(bool)].copy()
    test["error"] = np.abs(test.adapter_predicted_score - test.dive_score)
    order = np.argsort(test.review_priority.to_numpy())
    coverage = np.arange(1, len(test) + 1) / len(test)
    risk = np.cumsum(test.error.to_numpy()[order]) / np.arange(1, len(test) + 1)
    high = test.judge_sample_sd >= test.judge_sample_sd.quantile(0.75)
    reviewed = test.recommend_review.astype(bool)
    enrichment = high[reviewed].mean() / high.mean() if reviewed.any() else 0.0
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    axes[0].plot(coverage, risk, color=PALETTE["ecr"], lw=1.8)
    axes[0].axvline(0.8, color=PALETTE["risk"], ls="--", lw=1)
    axes[0].set(xlabel="Automatic coverage", ylabel="Accepted-sample MAE")
    _panel_label(axes[0], "a")
    axes[1].bar([0, 1], [1.0, enrichment], color=[PALETTE["neutral"], PALETTE["risk"]])
    axes[1].axhline(1.0, color="#333333", lw=0.8)
    axes[1].set_xticks([0, 1], ["Random review", "TrustDive-ECR"])
    axes[1].set_ylabel("High-disagreement enrichment")
    _panel_label(axes[1], "b")
    fig.tight_layout()
    source = pd.DataFrame({"coverage": coverage, "accepted_mae": risk})
    return fig, source


def render_reports_v6() -> dict:
    analysis_path = V6_RESULTS_ROOT / "06_REVIEW" / "analysis_summary_v6.json"
    if not analysis_path.exists():
        pilot_path = V6_RESULTS_ROOT / "04_PILOT" / "pilot_gate_v6.json"
        pilot = json.loads(pilot_path.read_text(encoding="utf-8")) if pilot_path.exists() else None
        reason = (
            "Validation attribution gate failed; official evaluation and figures remain locked"
            if pilot and pilot.get("status") != "PASS"
            else "Review analysis has not completed"
        )
        result = {"status": "LOCKED", "reason": reason}
        write_json(V6_RESULTS_ROOT / "figures_v6" / "render_status_v6.json", result)
        return result
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("publication_decision") == "NO_GO":
        result = {"status": "LOCKED", "reason": "Evidence gates did not authorize formal figures"}
        write_json(V6_RESULTS_ROOT / "figures_v6" / "render_status_v6.json", result)
        return result
    _configure()
    import matplotlib.pyplot as plt

    folder = V6_RESULTS_ROOT / "figures_v6"
    source_folder = folder / "source_data"
    source_folder.mkdir(parents=True, exist_ok=True)
    prediction = pd.read_parquet(V6_RESULTS_ROOT / "05_FINAL" / "adapter_predictions_v6.parquet")
    evidence = pd.read_parquet(V6_RESULTS_ROOT / "05_FINAL" / "phase_evidence_final_v6.parquet")
    review = pd.read_parquet(V6_RESULTS_ROOT / "06_REVIEW" / "review_priority_v6.parquet")
    score_summary = json.loads((V6_RESULTS_ROOT / "05_FINAL" / "final_score_summary_v6.json").read_text(encoding="utf-8"))

    contract = {
        "core_conclusion": "The adapted final scorer preserves ranking while yielding exact, stable phase evidence that prioritizes difficult performances for review.",
        "archetype": "asymmetric mixed-modality figure",
        "target": "Frontiers in Psychology - Performance Science",
        "backend": "Python/matplotlib only",
        "final_width_mm": 180,
        "exports": ["SVG", "PDF", "600 dpi TIFF", "PNG preview"],
        "review_risks": ["no causal or judge-mental-state claim", "no hand-picked favorable case", "open-set cases identified"],
    }
    write_json(folder / "figure_contract_v6.json", contract)
    figures = {}
    fig = _figure_method(folder); figures["figure_1_method"] = _save(fig, folder, "figure_1_method_v6"); plt.close(fig)
    fig, source = _figure_performance(folder, prediction, score_summary); source.to_csv(source_folder / "figure_2_source.csv", index=False); figures["figure_2_performance"] = _save(fig, folder, "figure_2_performance_v6"); plt.close(fig)
    fig, source = _figure_case(folder, prediction, evidence, review); source.to_csv(source_folder / "figure_3_source.csv", index=False); figures["figure_3_case"] = _save(fig, folder, "figure_3_phase_case_v6"); plt.close(fig)
    fig, source = _figure_coalitions(evidence); source.to_csv(source_folder / "figure_4_source.csv", index=False); figures["figure_4_coalitions"] = _save(fig, folder, "figure_4_coalitions_v6"); plt.close(fig)
    fig, source = _figure_fidelity(evidence); source.to_csv(source_folder / "figure_5_source.csv", index=False); figures["figure_5_fidelity"] = _save(fig, folder, "figure_5_fidelity_v6"); plt.close(fig)
    fig, source = _figure_review(review); source.to_csv(source_folder / "figure_6_source.csv", index=False); figures["figure_6_review"] = _save(fig, folder, "figure_6_review_v6"); plt.close(fig)
    result = {"status": "PASS", "figures": figures, "figure_contract_sha256": sha256_file(folder / "figure_contract_v6.json")}
    write_json(folder / "render_status_v6.json", result)
    return result
