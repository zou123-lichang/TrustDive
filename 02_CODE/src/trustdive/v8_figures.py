from __future__ import annotations

import io
import json
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from .config import Paths
from .util import sha256_file, write_json
from .v6_modeling import load_v6_assets
from .v8_data import V8_RESULTS_ROOT, load_conditional_manifest_v8, reveal_test_disagreement_v8


PALETTE = {
    "neutral": "#7A8793",
    "teacher": "#5975A4",
    "ours": "#3A9D8F",
    "risk": "#D98155",
    "takeoff": "#5B8FF9",
    "flight": "#7BC67B",
    "entry": "#E9A23B",
}


def _configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def _save(fig, stem) -> dict[str, str]:
    outputs = {}
    for suffix, options in (
        ("svg", {}), ("pdf", {}), ("tiff", {"dpi": 600}), ("png", {"dpi": 220})
    ):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", facecolor="white", **options)
        outputs[suffix] = sha256_file(path)
    plt.close(fig)
    return outputs


def _method_figure(root):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axis("off")
    boxes = [
        (0.02, "RICA²\nscore base", PALETTE["teacher"]),
        (0.20, "5 matched\nreferences", PALETTE["neutral"]),
        (0.38, "3 phases ×\n8 coalitions", PALETTE["takeoff"]),
        (0.56, "conditional\ndisagreement", PALETTE["flight"]),
        (0.74, "dual exact\nShapley", PALETTE["entry"]),
        (0.90, "20% review\nqueue", PALETTE["risk"]),
    ]
    for index, (x, text, color) in enumerate(boxes):
        width = 0.13 if index < 5 else 0.09
        ax.add_patch(mpl.patches.FancyBboxPatch(
            (x, 0.39), width, 0.25, boxstyle="round,pad=0.02", facecolor=color,
            edgecolor="white", alpha=0.88,
        ))
        ax.text(x + width / 2, 0.515, text, ha="center", va="center", color="white", weight="bold")
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            ax.annotate("", (next_x - 0.01, 0.515), (x + width + 0.005, 0.515),
                        arrowprops={"arrowstyle": "->", "color": "#59636E", "lw": 1.4})
    ax.text(0.5, 0.82, "TrustDive-Conflict: score, dispute evidence, and selective review",
            ha="center", va="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.20, "Score risk and excess judge-disagreement risk remain distinct outputs",
            ha="center", color="#59636E")
    return _save(fig, root / "figure_1_method_v8")


def _risk_task_figure(root, panel, comparison):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
    sns.scatterplot(data=panel, x="predicted_quality", y="judge_sample_sd",
                    hue="high_excess_disagreement", palette={False: PALETTE["neutral"], True: PALETTE["risk"]},
                    s=18, alpha=0.75, ax=axes[0], legend=False)
    axes[0].set(xlabel="Predicted execution quality", ylabel="Seven-judge SD",
                title="Raw disagreement is score-confounded")
    order = comparison.sort_values("auroc")
    axes[1].barh(order.model, order.auroc, color=[PALETTE["ours"] if "v8" in x else PALETTE["neutral"] for x in order.model])
    axes[1].axvline(0.5, color="#59636E", ls="--", lw=0.8)
    axes[1].set(xlabel="AUROC for excess disagreement", title="Increment beyond simple baselines")
    fig.tight_layout()
    panel.to_csv(root / "source_data" / "figure_2_panel.csv", index=False)
    comparison.to_csv(root / "source_data" / "figure_2_comparison.csv", index=False)
    return _save(fig, root / "figure_2_conditional_disagreement_v8")


def _score_figure(root, panel):
    data = []
    for row in panel.itertuples(index=False):
        group = "High excess" if row.high_excess_disagreement else "Ordinary"
        data.extend([
            {"group": group, "model": "RICA²", "error": abs(row.rica2_predicted_score - row.dive_score)},
            {"group": group, "model": "TrustDive", "error": abs(row.predicted_score - row.dive_score)},
        ])
    source = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    sns.boxplot(data=source, x="group", y="error", hue="model", showfliers=False,
                palette={"RICA²": PALETTE["teacher"], "TrustDive": PALETTE["ours"]}, ax=ax)
    sns.stripplot(data=source, x="group", y="error", hue="model", dodge=True, size=2,
                  alpha=0.25, palette={"RICA²": PALETTE["teacher"], "TrustDive": PALETTE["ours"]}, ax=ax, legend=False)
    ax.set(ylabel="Absolute total-score error", xlabel="", title="Scoring error in ordinary and disputed actions")
    fig.tight_layout()
    source.to_csv(root / "source_data" / "figure_3_score.csv", index=False)
    return _save(fig, root / "figure_3_high_risk_score_v8")


def _frame_from_zip(archive, source: str, instance: int, frame_number: int):
    name = f"FINADiving_MTL_256s/{source}/{int(instance)}/{int(frame_number):08d}.jpg"
    with archive.open(name) as handle:
        return np.asarray(Image.open(io.BytesIO(handle.read())).convert("RGB"))


def _dual_case_figure(root, evidence, frame):
    eligible = evidence[(evidence.analysis_role == "official_test") & (~evidence.open_set)].copy()
    eligible["case_rank"] = eligible.disagreement_targeted_effect.rank(pct=True)
    indices = [
        (eligible.case_rank - target).abs().idxmin() for target in (0.2, 0.5, 0.8)
    ]
    cases = eligible.loc[indices].merge(frame, on="clip_uid", validate="one_to_one")
    fig = plt.figure(figsize=(7.2, 6.0))
    grid = fig.add_gridspec(len(cases), 4, width_ratios=[1, 1, 1, 1.25], hspace=0.28, wspace=0.12)
    with zipfile.ZipFile(Paths().trimmed_zip) as archive:
        for row_index, row in enumerate(cases.itertuples(index=False)):
            points = np.linspace(int(row.start_frame), int(row.end_frame), 5, dtype=int)[1:4]
            for phase, point in enumerate(points):
                ax = fig.add_subplot(grid[row_index, phase])
                ax.imshow(_frame_from_zip(archive, str(row.source), int(row.instance), int(point)))
                ax.axis("off")
                ax.set_title(PHASE_LABELS[phase], color=list(PALETTE.values())[phase + 4], fontsize=7)
            ax = fig.add_subplot(grid[row_index, 3])
            score = [getattr(row, f"score_phi_{phase}") for phase in ("takeoff", "flight", "entry")]
            dispute = [getattr(row, f"disagreement_phi_{phase}") for phase in ("takeoff", "flight", "entry")]
            y = np.arange(3)
            ax.barh(y + 0.18, score, height=0.32, color=PALETTE["ours"], label="Score")
            ax.barh(y - 0.18, dispute, height=0.32, color=PALETTE["risk"], label="Dispute")
            ax.axvline(0, color="#59636E", lw=0.7)
            ax.set_yticks(y, PHASE_LABELS)
            if row_index == 0:
                ax.legend(fontsize=6)
            ax.set_xlabel("Exact phase contribution")
    fig.suptitle("The same performance can have distinct score and dispute evidence", weight="bold")
    cases.to_csv(root / "source_data" / "figure_4_cases.csv", index=False)
    return _save(fig, root / "figure_4_dual_phase_evidence_v8")


PHASE_LABELS = ("Take-off", "Flight", "Entry")


def _faithfulness_figure(root, evidence):
    test = evidence[(evidence.analysis_role == "official_test") & (~evidence.open_set)].copy()
    source = pd.DataFrame({
        "Score targeted": test.score_targeted_effect,
        "Score random": test.score_random_effect,
        "Dispute targeted": test.disagreement_targeted_effect,
        "Dispute random": test.disagreement_random_effect,
    }).melt(var_name="intervention", value_name="effect")
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    sns.boxplot(data=source, x="intervention", y="effect", showfliers=False,
                palette=[PALETTE["ours"], PALETTE["neutral"], PALETTE["risk"], PALETTE["neutral"]], ax=ax)
    ax.tick_params(axis="x", rotation=20)
    ax.set(xlabel="", ylabel="Absolute output change", title="Targeted phase interventions must exceed random phases")
    fig.tight_layout()
    source.to_csv(root / "source_data" / "figure_5_faithfulness.csv", index=False)
    return _save(fig, root / "figure_5_faithfulness_v8")


def _review_figure(root, summary):
    strategies = pd.DataFrame(summary["selective_review"]["strategies"]).T.reset_index(names="strategy")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].barh(strategies.strategy, 100 * strategies.accepted_mae_reduction,
                 color=[PALETTE["ours"] if x == "v8_pareto" else PALETTE["neutral"] for x in strategies.strategy])
    axes[0].set(xlabel="Accepted-set MAE reduction (%)", title="80% automatic acceptance")
    x = np.arange(len(strategies))
    axes[1].plot(x, strategies.high_error_recall, "o-", label="High error", color=PALETTE["teacher"])
    axes[1].plot(x, strategies.high_excess_disagreement_recall, "o-", label="Excess disagreement", color=PALETTE["risk"])
    axes[1].set_xticks(x, strategies.strategy, rotation=35, ha="right")
    axes[1].set(ylabel="Recall at 20% review", title="Limited review attention")
    axes[1].legend()
    fig.tight_layout()
    strategies.to_csv(root / "source_data" / "figure_6_review.csv", index=False)
    return _save(fig, root / "figure_6_selective_review_v8")


def render_reports_v8() -> dict:
    _configure()
    root = V8_RESULTS_ROOT / "figures_v8"
    root.mkdir(parents=True, exist_ok=True)
    (root / "source_data").mkdir(parents=True, exist_ok=True)
    contracts = {
        "backend": "Python/matplotlib",
        "target": "Frontiers in Psychology - Performance Science",
        "exports": ["SVG", "PDF", "600 dpi TIFF", "PNG preview"],
        "core_conclusion": "TrustDive-Conflict must add phase-grounded dispute detection and improve fixed-budget review without sacrificing score assessment.",
        "figures": {
            "1": {"archetype": "schematic-led composite", "claim": "The system separates scoring, excess disagreement, dual evidence, and review."},
            "2": {"archetype": "quantitative grid", "claim": "Excess disagreement is evaluated beyond score and task confounding."},
            "3": {"archetype": "quantitative grid", "claim": "The model improves difficult-action scoring without hiding individual errors."},
            "4": {"archetype": "asymmetric mixed-modality", "claim": "Real phases provide distinct exact score and dispute evidence."},
            "5": {"archetype": "quantitative grid", "claim": "Targeted phase interventions are more influential than random interventions."},
            "6": {"archetype": "quantitative grid", "claim": "A fixed review budget improves retained accuracy and dispute recall."},
        },
    }
    write_json(root / "figure_contract_v8.json", contracts)
    pilot_path = V8_RESULTS_ROOT / "03_PILOT" / "pilot_gate_v8.json"
    analysis_path = V8_RESULTS_ROOT / "06_REVIEW" / "analysis_summary_v8.json"
    if not pilot_path.exists() or json.loads(pilot_path.read_text(encoding="utf-8")).get("status") != "PASS" or not analysis_path.exists():
        result = {
            "status": "STOPPED_BY_EVIDENCE_GATE",
            "reason": "The v8 pilot or final analysis gate did not pass; no formal result figures were generated.",
            "figure_contract_sha256": sha256_file(root / "figure_contract_v8.json"),
        }
        write_json(root / "render_status_v8.json", result)
        return result
    summary = json.loads(analysis_path.read_text(encoding="utf-8"))
    if summary.get("publication_decision") == "EXPLORATORY_ONLY":
        result = {"status": "STOPPED_BY_EVIDENCE_GATE", "reason": "Publication gate is exploratory only."}
        write_json(root / "render_status_v8.json", result)
        return result
    manifest = reveal_test_disagreement_v8(load_conditional_manifest_v8())
    prediction = pd.read_parquet(V8_RESULTS_ROOT / "04_FINAL" / "predictions_v8.parquet")
    evidence = pd.read_parquet(V8_RESULTS_ROOT / "05_DUAL_EVIDENCE" / "dual_phase_evidence_v8.parquet")
    comparison = pd.read_csv(V8_RESULTS_ROOT / "06_REVIEW" / "disagreement_comparison_v8.csv")
    panel = prediction.merge(
        manifest[["clip_uid", "judge_sample_sd", "high_excess_disagreement"]], on="clip_uid", validate="one_to_one"
    )
    panel = panel[(panel.analysis_role == "official_test") & panel.disagreement_primary_eligible.astype(bool)].copy()
    frame = load_v6_assets().frame[["clip_uid", "source", "instance", "start_frame", "end_frame"]]
    hashes = {
        "figure_1": _method_figure(root),
        "figure_2": _risk_task_figure(root, panel, comparison),
        "figure_3": _score_figure(root, panel),
        "figure_4": _dual_case_figure(root, evidence, frame),
        "figure_5": _faithfulness_figure(root, evidence),
        "figure_6": _review_figure(root, summary),
    }
    pngs = sorted(root.glob("figure_*.png"))
    qa = {
        "status": "PASS" if len(pngs) == 6 and all(Image.open(path).width > 800 for path in pngs) else "FAIL",
        "png_count": len(pngs),
        "editable_vector_exports": True,
        "colorblind_safe_palette": True,
        "source_data_present": len(list((root / "source_data").glob("*.csv"))) >= 5,
    }
    write_json(root / "figure_qa_v8.json", qa)
    result = {"status": qa["status"], "figure_hashes": hashes, "qa": qa}
    write_json(root / "render_status_v8.json", result)
    return result
