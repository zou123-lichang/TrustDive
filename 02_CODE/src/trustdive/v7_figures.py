from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import Paths
from .features import ZipFrameStore
from .util import sha256_file, write_json
from .v7_data import V7_RESULTS_ROOT, load_risk_manifest_v7, load_v7_frame
from .v7_modeling import load_reference_map_v7


COLORS = {
    "teacher": "#7A869A",
    "ours": "#246B8E",
    "takeoff": "#4C78A8",
    "flight": "#F2A65A",
    "entry": "#2A9D8F",
    "risk": "#C9574F",
    "safe": "#8FB996",
    "neutral": "#DCE3E8",
    "dark": "#263238",
}


def _configure() -> None:
    import matplotlib as mpl

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


def _save(fig, folder: Path, stem: str) -> dict:
    outputs = {}
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("tiff", {"dpi": 600}), ("png", {"dpi": 300})):
        path = folder / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[suffix] = {"path": str(path), "sha256": sha256_file(path)}
    return outputs


def _panel(ax, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def _figure1_method():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.axis("off")
    nodes = [
        (0.02, "Frozen RICA²\nscore + latents", COLORS["teacher"]),
        (0.21, "Same-action\ntraining references", COLORS["neutral"]),
        (0.40, "Risk-balanced\nlatent adapter", COLORS["ours"]),
        (0.59, "8 phase\ncounterfactuals", COLORS["flight"]),
        (0.78, "Dual-risk\nreview queue", COLORS["risk"]),
    ]
    for x, text, color in nodes:
        patch = FancyBboxPatch((x, 0.38), 0.16, 0.28, boxstyle="round,pad=0.02", fc=color, ec="white", lw=1.2)
        ax.add_patch(patch)
        ax.text(x + 0.08, 0.52, text, ha="center", va="center", fontsize=7.2, fontweight="bold", color="white" if color != COLORS["neutral"] else COLORS["dark"])
    for x in (0.18, 0.37, 0.56, 0.75):
        ax.add_patch(FancyArrowPatch((x, 0.52), (x + 0.03, 0.52), arrowstyle="-|>", mutation_scale=11, lw=1.1, color="#5C6770"))
    ax.text(0.50, 0.84, "TrustDive-Risk: accurate scoring, exact phase evidence, selective review", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.665, 0.24, "takeoff", color=COLORS["takeoff"], ha="center", fontweight="bold")
    ax.text(0.725, 0.24, "+ flight", color=COLORS["flight"], ha="center", fontweight="bold")
    ax.text(0.785, 0.24, "+ entry", color=COLORS["entry"], ha="center", fontweight="bold")
    ax.text(0.50, 0.08, "High risk = scoring-error or judge-disagreement review risk (not injury risk)", ha="center", color=COLORS["dark"])
    fig.tight_layout()
    return fig


def _figure2_performance(prediction: pd.DataFrame, review: pd.DataFrame, summary: dict):
    import matplotlib.pyplot as plt

    test = prediction[prediction.analysis_role == "official_test"].copy()
    panel = review[(review.analysis_role == "official_test") & review.disagreement_primary_eligible.astype(bool)].copy()
    panel["teacher_error"] = np.abs(panel.teacher_predicted_score - panel.dive_score)
    panel["ours_error"] = np.abs(panel.trustdive_predicted_score - panel.dive_score)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.35, 0.85, 0.8]})
    axes[0].scatter(test.dive_score, test.teacher_predicted_score, s=9, alpha=0.25, color=COLORS["teacher"], label="RICA²")
    axes[0].scatter(test.dive_score, test.trustdive_predicted_score, s=9, alpha=0.38, color=COLORS["ours"], label="TrustDive-Risk")
    limits = [float(test.dive_score.min()), float(test.dive_score.max())]
    axes[0].plot(limits, limits, "--", color=COLORS["dark"], lw=0.8)
    axes[0].set(xlabel="Official score", ylabel="Predicted score")
    axes[0].legend(loc="upper left")
    _panel(axes[0], "a")
    high = panel.high_judge_risk.astype(bool)
    values = np.array([
        [panel.loc[~high, "teacher_error"].mean(), panel.loc[high, "teacher_error"].mean()],
        [panel.loc[~high, "ours_error"].mean(), panel.loc[high, "ours_error"].mean()],
    ])
    x = np.arange(2)
    axes[1].bar(x - 0.18, values[0], 0.36, color=COLORS["teacher"], label="RICA²")
    axes[1].bar(x + 0.18, values[1], 0.36, color=COLORS["ours"], label="TrustDive")
    axes[1].set_xticks(x, ["Lower\ndisagreement", "High\ndisagreement"])
    axes[1].set_ylabel("MAE (score points)")
    _panel(axes[1], "b")
    gain = summary["high_judge_risk_scoring"]
    bars = [gain["teacher_risk_gap"], gain["trustdive_risk_gap"], gain["risk_directed_gain"]]
    axes[2].bar(range(3), bars, color=[COLORS["teacher"], COLORS["ours"], COLORS["safe"]])
    axes[2].axhline(0, color=COLORS["dark"], lw=0.8)
    axes[2].set_xticks(range(3), ["RICA²\nrisk gap", "TrustDive\nrisk gap", "Directed\ngain"], rotation=20)
    axes[2].set_ylabel("Score points")
    _panel(axes[2], "c")
    fig.tight_layout()
    source = panel[["clip_uid", "high_judge_risk", "teacher_error", "ours_error"]].copy()
    return fig, source


def _waterfall(ax, row: pd.Series) -> None:
    values = [row.reference_baseline, row.phi_takeoff, row.phi_flight, row.phi_entry]
    labels = ["Reference", "Takeoff", "Flight", "Entry"]
    colors = [COLORS["neutral"], COLORS["takeoff"], COLORS["flight"], COLORS["entry"]]
    running = 0.0
    for index, (value, label, color) in enumerate(zip(values, labels, colors)):
        bottom = running if index else 0.0
        ax.bar(index, value, bottom=bottom, color=color, edgecolor="white")
        running += value
    ax.axhline(row.predicted_quality, color=COLORS["dark"], ls="--", lw=0.9)
    ax.set_xticks(range(4), labels, rotation=25, ha="right")
    ax.set_ylabel("Execution-quality score")


def _select_corrected_case(prediction: pd.DataFrame, review: pd.DataFrame) -> pd.Series:
    merged = prediction.merge(review[["clip_uid", "high_judge_risk", "review_priority"]], on="clip_uid")
    test = merged[(merged.analysis_role == "official_test") & (~merged.open_set.astype(bool))].copy()
    test["gain"] = np.abs(test.teacher_predicted_score - test.dive_score) - np.abs(test.trustdive_predicted_score - test.dive_score)
    candidates = test[test.high_judge_risk.astype(bool)]
    return (candidates if len(candidates) else test).sort_values("gain", ascending=False).iloc[0]


def _figure3_case(prediction: pd.DataFrame, evidence: pd.DataFrame, review: pd.DataFrame):
    import matplotlib.pyplot as plt

    frame = load_v7_frame()
    case = _select_corrected_case(prediction, review)
    query_index = int(frame.index[frame.clip_uid == case.clip_uid][0])
    mapping = load_reference_map_v7(final=True)
    ref_index = int(mapping["references"][query_index, 0])
    query, reference = frame.iloc[query_index], frame.iloc[ref_index]
    with ZipFrameStore(Paths().trimmed_zip) as store:
        query_frames = store.load(query.source, int(query.instance), 3)
        reference_frames = store.load(reference.source, int(reference.instance), 3)
    fig = plt.figure(figsize=(7.2, 3.6), layout="constrained")
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.2], hspace=0.06, wspace=0.08)
    for row_index, (images, label) in enumerate(((query_frames, "High-risk query"), (reference_frames, "Training reference"))):
        for col, image in enumerate(images):
            ax = fig.add_subplot(grid[row_index, col])
            ax.imshow(image)
            ax.axis("off")
            if row_index == 0:
                ax.set_title(("Takeoff", "Flight", "Entry")[col], color=(COLORS["takeoff"], COLORS["flight"], COLORS["entry"])[col], fontweight="bold")
            if col == 0:
                ax.text(-0.08, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="right", fontweight="bold")
    evidence_row = evidence.set_index("clip_uid").loc[case.clip_uid]
    ax = fig.add_subplot(grid[:, 3])
    _waterfall(ax, evidence_row)
    ax.set_title(f"RICA² {case.teacher_predicted_score:.1f} → TrustDive {case.trustdive_predicted_score:.1f}\nOfficial {case.dive_score:.1f}", fontsize=8, fontweight="bold")
    source = pd.DataFrame({
        "clip_uid": [case.clip_uid] * 4,
        "component": ["reference", "takeoff", "flight", "entry"],
        "value": [evidence_row.reference_baseline, evidence_row.phi_takeoff, evidence_row.phi_flight, evidence_row.phi_entry],
    })
    return fig, source


def _figure4_counterfactual(evidence: pd.DataFrame):
    import matplotlib.pyplot as plt

    test = evidence[(evidence.analysis_role == "official_test") & (~evidence.open_set.astype(bool))].copy()
    test["spread"] = test[[f"coalition_{i}" for i in range(8)]].max(axis=1) - test[[f"coalition_{i}" for i in range(8)]].min(axis=1)
    chosen = test.sort_values("spread").iloc[np.linspace(0, len(test) - 1, 5).round().astype(int)]
    matrix = chosen[[f"coalition_{i}" for i in range(8)]].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.3, 0.9]})
    image = axes[0].imshow(matrix, aspect="auto", cmap="Blues")
    axes[0].set_xticks(range(8), [f"{mask:03b}" for mask in range(8)])
    axes[0].set_yticks(range(5), ["lowest", "low", "median", "high", "highest"])
    axes[0].set(xlabel="Takeoff–flight–entry coalition", ylabel="Counterfactual spread")
    fig.colorbar(image, ax=axes[0], label="Final scorer output")
    _panel(axes[0], "a")
    axes[1].scatter(test.random_intervention_effect, test.targeted_intervention_effect, s=11, alpha=0.4, color=COLORS["ours"])
    limit = float(max(test.random_intervention_effect.max(), test.targeted_intervention_effect.max()))
    axes[1].plot([0, limit], [0, limit], "--", color=COLORS["dark"], lw=0.8)
    axes[1].set(xlabel="Random-phase effect", ylabel="Top-attribution phase effect")
    _panel(axes[1], "b")
    fig.tight_layout()
    return fig, test[["clip_uid", "top_phase", "actual_max_intervention_phase", "targeted_intervention_effect", "random_intervention_effect", "top_phase_intervention_match"]]


def _coverage_curve(error: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(risk, kind="stable")
    coverage = np.arange(1, len(order) + 1) / len(order)
    accepted_mae = np.cumsum(error[order]) / np.arange(1, len(order) + 1)
    return coverage, accepted_mae


def _figure5_risk(prediction: pd.DataFrame, review: pd.DataFrame, manifest: pd.DataFrame, summary: dict):
    import matplotlib.pyplot as plt

    test = review[review.analysis_role == "official_test"].merge(
        manifest[["clip_uid", "error_risk_threshold"]], on="clip_uid"
    )
    test["error"] = np.abs(test.trustdive_predicted_score - test.dive_score)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for label, risk, color in (
        ("RICA² uncertainty", prediction.loc[prediction.analysis_role == "official_test", "teacher_uncertainty"].to_numpy(), COLORS["teacher"]),
        ("TrustDive combined", test.review_priority.to_numpy(), COLORS["ours"]),
    ):
        coverage, mae = _coverage_curve(test.error.to_numpy(), risk)
        axes[0].plot(coverage, mae, lw=1.7, label=label, color=color)
    axes[0].axvline(0.8, color=COLORS["risk"], ls="--", lw=0.9)
    axes[0].set(xlabel="Automatic coverage", ylabel="Accepted-sample MAE")
    axes[0].legend()
    _panel(axes[0], "a")
    names = ["RICA²\nuncertainty", "Error\nrisk", "Disagreement\nrisk", "Combined"]
    strategy_keys = ["rica2_uncertainty", "error_risk", "disagreement_risk", "trustdive_combined"]
    reductions = [summary["review_strategies"][key]["accepted_mae_reduction"] * 100 for key in strategy_keys]
    axes[1].bar(range(4), reductions, color=[COLORS["teacher"], COLORS["takeoff"], COLORS["flight"], COLORS["ours"]])
    axes[1].axhline(0, color=COLORS["dark"], lw=0.8)
    axes[1].set_xticks(range(4), names, rotation=20)
    axes[1].set_ylabel("Accepted-MAE reduction (%)")
    _panel(axes[1], "b")
    enrichment = [summary["panel_review_strategies"][key]["high_judge_enrichment"] for key in strategy_keys]
    axes[2].bar(range(4), enrichment, color=[COLORS["teacher"], COLORS["takeoff"], COLORS["flight"], COLORS["ours"]])
    axes[2].axhline(1, color=COLORS["dark"], lw=0.8)
    axes[2].set_xticks(range(4), names, rotation=20)
    axes[2].set_ylabel("High-disagreement enrichment")
    _panel(axes[2], "c")
    fig.tight_layout()
    source = pd.DataFrame({"strategy": strategy_keys, "accepted_mae_reduction_pct": reductions, "high_judge_enrichment": enrichment})
    return fig, source


def _unique_cases(prediction: pd.DataFrame, review: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    merged = prediction.merge(review[["clip_uid", "review_priority", "high_judge_risk", "review_reason"]], on="clip_uid")
    test = merged[(merged.analysis_role == "official_test") & (~merged.open_set.astype(bool))].copy()
    test["ours_error"] = np.abs(test.trustdive_predicted_score - test.dive_score)
    test["gain"] = np.abs(test.teacher_predicted_score - test.dive_score) - test.ours_error
    candidates = [
        ("Low-risk accurate", test.sort_values(["review_priority", "ours_error"]).iloc[0]),
        ("Corrected", test.sort_values("gain", ascending=False).iloc[0]),
        ("High error (not flagged)", test.sort_values("ours_error", ascending=False).iloc[0]),
        ("High disagreement", test[test.high_judge_risk.astype(bool)].sort_values("review_priority", ascending=False).iloc[0]),
    ]
    used = set()
    output = []
    for label, row in candidates:
        if row.clip_uid in used:
            pool = test[~test.clip_uid.isin(used)].sort_values("review_priority", ascending=False)
            row = pool.iloc[0]
        used.add(row.clip_uid)
        output.append((label, row))
    return output


def _figure6_cases(prediction: pd.DataFrame, evidence: pd.DataFrame, review: pd.DataFrame):
    import matplotlib.pyplot as plt

    frame = load_v7_frame().set_index("clip_uid")
    cases = _unique_cases(prediction, review)
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
    source_rows = []
    with ZipFrameStore(Paths().trimmed_zip) as store:
        for row_index, (label, case) in enumerate(cases):
            meta = frame.loc[case.clip_uid]
            images = store.load(meta.source, int(meta.instance), 3)
            for col, image in enumerate(images):
                axes[row_index, col].imshow(image)
                axes[row_index, col].axis("off")
                if row_index == 0:
                    axes[row_index, col].set_title(("Takeoff", "Flight", "Entry")[col], color=(COLORS["takeoff"], COLORS["flight"], COLORS["entry"])[col], fontweight="bold")
            ax = axes[row_index, 3]
            evidence_row = evidence.set_index("clip_uid").loc[case.clip_uid]
            contributions = [evidence_row.phi_takeoff, evidence_row.phi_flight, evidence_row.phi_entry]
            ax.barh([0, 1, 2], contributions, color=[COLORS["takeoff"], COLORS["flight"], COLORS["entry"]])
            ax.axvline(0, color=COLORS["dark"], lw=0.7)
            ax.set_yticks([0, 1, 2], ["Takeoff", "Flight", "Entry"])
            ax.set_title(label, loc="left", fontweight="bold")
            ax.text(0.02, -0.28, f"Official {case.dive_score:.1f} | Pred. {case.trustdive_predicted_score:.1f}\nRisk {case.review_priority:.2f}: {case.review_reason}", transform=ax.transAxes, fontsize=6, va="top", wrap=True)
            source_rows.append({"case": label, "clip_uid": case.clip_uid, "score": case.dive_score, "prediction": case.trustdive_predicted_score, "risk": case.review_priority, "reason": case.review_reason})
    fig.tight_layout()
    return fig, pd.DataFrame(source_rows)


def _qa_exports(figures: dict) -> dict:
    checks = {}
    for name, bundle in figures.items():
        png = Path(bundle["png"]["path"])
        tiff = Path(bundle["tiff"]["path"])
        with Image.open(png) as image:
            gray = np.asarray(image.convert("L"), dtype=float)
            png_ok = image.width > 1000 and image.height > 300 and float(gray.std()) > 5.0
        with Image.open(tiff) as image:
            tiff_ok = image.info.get("dpi", (0, 0))[0] >= 590 and image.width > 2000
        checks[name] = {
            "png_nonblank_and_readable": bool(png_ok),
            "tiff_approximately_600dpi": bool(tiff_ok),
            "pdf_exists": Path(bundle["pdf"]["path"]).stat().st_size > 0,
            "svg_exists": Path(bundle["svg"]["path"]).stat().st_size > 0,
        }
    return {"status": "PASS" if all(all(item.values()) for item in checks.values()) else "FAIL", "checks": checks}


def render_reports_v7() -> dict:
    analysis_path = V7_RESULTS_ROOT / "05_RISK_REVIEW" / "analysis_summary_v7.json"
    if not analysis_path.exists():
        raise RuntimeError("Run analyze-risk --protocol v7 first")
    _configure()
    import matplotlib.pyplot as plt

    folder = V7_RESULTS_ROOT / "figures_v7"
    source_folder = folder / "source_data"
    prediction = pd.read_parquet(V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet")
    evidence = pd.read_parquet(V7_RESULTS_ROOT / "04_PHASE_EVIDENCE" / "phase_evidence_v7.parquet")
    review = pd.read_parquet(V7_RESULTS_ROOT / "05_RISK_REVIEW" / "review_priority_v7.parquet")
    manifest = load_risk_manifest_v7()
    summary = json.loads(analysis_path.read_text(encoding="utf-8"))
    contract = {
        "core_conclusion": "TrustDive-Risk targets adjudication-sensitive dives with exact phase evidence and a selective-review queue.",
        "evidence_chain": ["method", "risk-stratified score", "real phase case", "counterfactual fidelity", "review allocation", "mechanical cases"],
        "archetype": "asymmetric mixed-modality figure set",
        "backend": "Python/matplotlib exclusively",
        "final_width_mm": 183,
        "exports": ["SVG", "PDF", "600 dpi TIFF", "PNG preview"],
        "review_risks": ["risk is not injury risk", "model evidence is not judge cognition", "cases are mechanically selected", "source data exported"],
    }
    write_json(folder / "figure_contract_v7.json", contract)
    figures = {}
    fig = _figure1_method(); figures["figure_1_method"] = _save(fig, folder, "figure_1_method_v7"); plt.close(fig)
    fig, source = _figure2_performance(prediction, review, summary); source.to_csv(source_folder / "figure_2_source.csv", index=False); figures["figure_2_performance"] = _save(fig, folder, "figure_2_risk_performance_v7"); plt.close(fig)
    fig, source = _figure3_case(prediction, evidence, review); source.to_csv(source_folder / "figure_3_source.csv", index=False); figures["figure_3_case"] = _save(fig, folder, "figure_3_phase_case_v7"); plt.close(fig)
    fig, source = _figure4_counterfactual(evidence); source.to_csv(source_folder / "figure_4_source.csv", index=False); figures["figure_4_counterfactual"] = _save(fig, folder, "figure_4_counterfactual_v7"); plt.close(fig)
    fig, source = _figure5_risk(prediction, review, manifest, summary); source.to_csv(source_folder / "figure_5_source.csv", index=False); figures["figure_5_review"] = _save(fig, folder, "figure_5_selective_review_v7"); plt.close(fig)
    fig, source = _figure6_cases(prediction, evidence, review); source.to_csv(source_folder / "figure_6_source.csv", index=False); figures["figure_6_cases"] = _save(fig, folder, "figure_6_cases_v7"); plt.close(fig)
    qa = _qa_exports(figures)
    qa["backend_exclusive"] = "Python"
    qa["source_data_files"] = sorted(str(path) for path in source_folder.glob("*.csv"))
    write_json(folder / "figure_qa_v7.json", qa)
    result = {
        "status": qa["status"],
        "figures": figures,
        "figure_contract_sha256": sha256_file(folder / "figure_contract_v7.json"),
        "figure_qa_sha256": sha256_file(folder / "figure_qa_v7.json"),
    }
    write_json(folder / "render_status_v7.json", result)
    return result
