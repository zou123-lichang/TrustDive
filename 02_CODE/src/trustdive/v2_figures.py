from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .features import ZipFrameStore
from .util import write_json
from .v2_data import V2_RESULTS_ROOT, load_panel_targets, require_contract_frozen, v2_paths


FIGURE_DIR = V2_RESULTS_ROOT / "figures_v2"
SOURCE_DIR = FIGURE_DIR / "source_data"

COLORS = {
    "navy": "#2F4B7C",
    "blue": "#4C78A8",
    "sky": "#72B7B2",
    "orange": "#F2A65A",
    "red": "#D95F5F",
    "gray": "#7A7A7A",
    "light": "#E8EDF3",
    "dark": "#252525",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, stem: str) -> list[str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600}),
        ("png", {"dpi": 220}),
    ):
        destination = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(destination, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(str(destination))
    plt.close(fig)
    return paths


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8)


def _figure1_pipeline() -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    ax.axis("off")
    boxes = [
        (0.02, "Diving\nvideo", COLORS["light"]),
        (0.20, "Predicted\nphases", "#DDEBF2"),
        (0.38, "Relative score +\nphase contributions", "#DCE9E4"),
        (0.60, "Judge disagreement\n$\\sigma_{judge}$", "#FBE7CF"),
        (0.78, "Model uncertainty\n$\\sigma_{model}$", "#F6DCDC"),
    ]
    for x, label, color in boxes:
        patch = FancyBboxPatch(
            (x, 0.52), 0.15, 0.28, boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.9, edgecolor=COLORS["dark"], facecolor=color
        )
        ax.add_patch(patch)
        ax.text(x + 0.075, 0.66, label, ha="center", va="center", fontsize=7)
    for x in (0.17, 0.35, 0.55, 0.75):
        ax.add_patch(FancyArrowPatch((x, 0.66), (x + 0.03, 0.66), arrowstyle="-|>", mutation_scale=10))
    review = FancyBboxPatch(
        (0.37, 0.10), 0.28, 0.20, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2, edgecolor=COLORS["navy"], facecolor="white"
    )
    ax.add_patch(review)
    ax.text(0.51, 0.20, "Selective review recommendation\n20% calibrated workload", ha="center", va="center", color=COLORS["navy"])
    ax.add_patch(FancyArrowPatch((0.855, 0.51), (0.64, 0.28), arrowstyle="-|>", mutation_scale=10, connectionstyle="arc3,rad=-0.15"))
    ax.text(0.02, 0.92, "TrustDive-D", fontsize=12, fontweight="bold", color=COLORS["navy"])
    ax.text(0.20, 0.92, "traceable score, two uncertainty sources, calibrated deferral", fontsize=8, color=COLORS["gray"])
    return _save(fig, "Figure1_system_overview")


def _figure2_score(predictions: pd.DataFrame, summary: dict) -> list[str]:
    test = predictions[predictions.official_split == "test"].copy()
    models = []
    for model in ("global", "relative", "phase_relative", "trustdive_d"):
        item = summary[model]
        models.append(
            {
                "model": model.replace("_", " ").title(),
                "spearman": item["official_test_score"]["spearman"],
                "seed_sd": float(
                    np.std([x["spearman"] for x in item["seed_official_test_scores"]])
                ),
            }
        )
    models_frame = pd.DataFrame(models)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    test[["clip_uid", "dive_score", "predicted_score", "difficulty"]].to_csv(
        SOURCE_DIR / "Figure2a_score_scatter.csv", index=False
    )
    models_frame.to_csv(SOURCE_DIR / "Figure2b_model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.scatter(test.dive_score, test.predicted_score, s=10, alpha=0.45, color=COLORS["blue"], linewidths=0)
    lower = min(test.dive_score.min(), test.predicted_score.min())
    upper = max(test.dive_score.max(), test.predicted_score.max())
    ax.plot([lower, upper], [lower, upper], "--", color=COLORS["gray"], lw=0.9)
    rho = summary["trustdive_d"]["official_test_score"]["spearman"]
    ax.text(0.04, 0.95, f"$\\rho$ = {rho:.3f}\nn = {len(test)} clips", transform=ax.transAxes, va="top")
    ax.set(xlabel="Official total score", ylabel="Predicted total score")
    _panel_label(ax, "a")

    ax = axes[1]
    order = np.arange(len(models_frame))
    colors = [COLORS["gray"], COLORS["sky"], COLORS["orange"], COLORS["navy"]]
    ax.barh(order, models_frame.spearman, xerr=models_frame.seed_sd, color=colors, height=0.62, capsize=2)
    ax.set_yticks(order, models_frame.model)
    ax.invert_yaxis()
    ax.set(xlabel="Spearman correlation", xlim=(0, 1))
    for y, value in zip(order, models_frame.spearman):
        ax.text(value + 0.015, y, f"{value:.3f}", va="center")
    _panel_label(ax, "b")
    fig.suptitle("Official-test scoring performance", x=0.06, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "Figure2_scoring_performance")


def _figure3_disagreement(predictions: pd.DataFrame, analysis: dict) -> list[str]:
    primary = predictions[
        (predictions.official_split == "test") & predictions.disagreement_primary_eligible.astype(bool)
    ].copy()
    threshold = primary.judge_sample_sd.quantile(0.75)
    primary["panel_group"] = np.where(primary.judge_sample_sd >= threshold, "High disagreement", "Lower disagreement")
    primary[["clip_uid", "judge_sample_sd", "sigma_judge", "sigma_model", "panel_group"]].to_csv(
        SOURCE_DIR / "Figure3_disagreement.csv", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    ax.scatter(primary.judge_sample_sd, primary.sigma_judge, s=14, alpha=0.55, color=COLORS["orange"], linewidths=0)
    ax.set(xlabel="Observed judge SD", ylabel="Predicted $\\sigma_{judge}$")
    ci = analysis["judge_sd_bootstrap_95_ci"]
    ax.text(0.04, 0.95, f"$\\rho$ = {analysis['judge_sd_spearman']:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]", transform=ax.transAxes, va="top")
    _panel_label(ax, "a")
    ax = axes[1]
    groups = [
        primary.loc[primary.panel_group == "Lower disagreement", "sigma_judge"],
        primary.loc[primary.panel_group == "High disagreement", "sigma_judge"],
    ]
    violin = ax.violinplot(groups, positions=[0, 1], showmeans=False, showmedians=True, widths=0.72)
    for body, color in zip(violin["bodies"], [COLORS["sky"], COLORS["orange"]]):
        body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(0.75)
    jitter = np.random.default_rng(20260817)
    for index, values in enumerate(groups):
        ax.scatter(index + jitter.normal(0, 0.045, len(values)), values, s=7, alpha=0.25, color=COLORS["dark"], linewidths=0)
    ax.set_xticks([0, 1], ["Lower\n75%", "Highest\nquartile"])
    ax.set_ylabel("Predicted $\\sigma_{judge}$")
    _panel_label(ax, "b")
    fig.suptitle("Judge disagreement is modeled separately from model uncertainty", x=0.06, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "Figure3_disagreement_modeling")


def _figure4_selective(panel: pd.DataFrame, curve: pd.DataFrame, analysis: dict) -> list[str]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(SOURCE_DIR / "Figure4a_coverage_risk.csv", index=False)
    clip = panel.groupby("clip_uid").agg(
        human_error=("human_error", "median"),
        learned_fused_error=("learned_fused_error", "median"),
        review_risk=("review_risk", "median"),
        review_recommended=("review_recommended", "max"),
    ).reset_index()
    clip.to_csv(SOURCE_DIR / "Figure4b_panel_errors.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    ax.plot(curve.coverage, curve.risk, marker="o", ms=3, lw=1.5, color=COLORS["navy"])
    ax.axvline(0.8, ls="--", lw=0.9, color=COLORS["orange"])
    ax.set(xlabel="Automatic acceptance coverage", ylabel="Mean fused error", xlim=(0.08, 1.02))
    _panel_label(ax, "a")
    ax = axes[1]
    rng = np.random.default_rng(20260817)
    x0 = rng.normal(-0.08, 0.025, len(clip)); x1 = rng.normal(1.08, 0.025, len(clip))
    for left, right, y0, y1 in zip(x0, x1, clip.human_error, clip.learned_fused_error):
        ax.plot([left, right], [y0, y1], color=COLORS["light"], lw=0.45, zorder=0)
    ax.scatter(x0, clip.human_error, s=8, alpha=0.45, color=COLORS["gray"])
    ax.scatter(x1, clip.learned_fused_error, s=8, alpha=0.45, color=COLORS["navy"])
    ax.set_xticks([0, 1], ["Judge", "Judge + AI"])
    ax.set_ylabel("Absolute deviation from panel consensus")
    ci = analysis["bootstrap_95_ci"]
    ax.text(
        0.03,
        0.95,
        f"Median change = {analysis['median_error_difference']:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
        transform=ax.transAxes,
        va="top",
    )
    _panel_label(ax, "b")
    fig.suptitle("Selective review and retrospective judge-panel simulation", x=0.06, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "Figure4_selective_review_panel")


def _case_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions[predictions.official_split == "test"].copy()
    test["absolute_error"] = np.abs(test.predicted_score - test.dive_score)
    ordered = test.sort_values("absolute_error")
    selectors = {
        "Low error": ordered.iloc[max(0, round(0.10 * (len(ordered) - 1)))],
        "Median error": ordered.iloc[round(0.50 * (len(ordered) - 1))],
        "High uncertainty": test.sort_values("review_risk").iloc[-1],
    }
    rows = []
    used = set()
    for label, row in selectors.items():
        if row.clip_uid in used:
            row = test[~test.clip_uid.isin(used)].sort_values("review_risk").iloc[-1]
        used.add(row.clip_uid)
        item = row.to_dict(); item["case_label"] = label; rows.append(item)
    return pd.DataFrame(rows)


def _figure5_cases(predictions: pd.DataFrame, metadata: pd.DataFrame) -> list[str]:
    cases = _case_rows(predictions).merge(
        metadata[["clip_uid", "source", "instance", "frame_count", "transition_frames_json"]],
        on="clip_uid", how="left"
    )
    cases.to_csv(SOURCE_DIR / "Figure5_mechanical_cases.csv", index=False)
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 6.3), gridspec_kw={"width_ratios": [1, 1, 1, 1.18]})
    phase_names = ["Take-off", "Flight", "Entry"]
    with ZipFrameStore(v2_paths().trimmed_zip) as store:
        for row_index, row in enumerate(cases.itertuples(index=False)):
            names = store.frame_names(row.source, int(row.instance))
            transitions = json.loads(row.transition_frames_json)
            boundary1, boundary2 = int(transitions[0]), int(transitions[1])
            indices = [max(0, boundary1 // 2), (boundary1 + boundary2) // 2, min(len(names) - 1, (boundary2 + len(names) - 1) // 2)]
            for phase, (axis, frame_index) in enumerate(zip(axes[row_index, :3], indices)):
                with store.archive.open(names[int(np.clip(frame_index, 0, len(names) - 1))]) as handle:
                    from PIL import Image
                    import io

                    image = Image.open(io.BytesIO(handle.read())).convert("RGB")
                axis.imshow(image)
                axis.set_xticks([]); axis.set_yticks([])
                axis.set_title(phase_names[phase], fontsize=7, color=[COLORS["blue"], COLORS["orange"], COLORS["red"]][phase])
                for spine in axis.spines.values():
                    spine.set_visible(True); spine.set_linewidth(1.2); spine.set_edgecolor([COLORS["blue"], COLORS["orange"], COLORS["red"]][phase])
            axis = axes[row_index, 3]
            values = [row.phase_takeoff_contribution, row.phase_flight_contribution, row.phase_entry_contribution, row.residual]
            labels = ["Take-off", "Flight", "Entry", "Residual"]
            colors = [COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["gray"]]
            axis.barh(np.arange(4), values, color=colors, height=0.62)
            axis.axvline(0, color=COLORS["dark"], lw=0.7)
            axis.set_yticks(np.arange(4), labels); axis.invert_yaxis(); axis.set_xlabel("Attributed quality points")
            axis.text(0.02, 1.07, f"{row.case_label}: official {row.dive_score:.1f}, predicted {row.predicted_score:.1f}", transform=axis.transAxes, fontsize=7, fontweight="bold")
    fig.suptitle("Mechanically selected examples: real phase frames and additive evidence", x=0.05, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "Figure5_traceable_cases")


def _figure6_panel(panel: pd.DataFrame, analysis: dict) -> list[str]:
    clip = panel.groupby("clip_uid").agg(
        human_error=("human_error", "median"),
        ai_error=("ai_error", "median"),
        fixed_fused_error=("fixed_fused_error", "median"),
        learned_fused_error=("learned_fused_error", "median"),
    ).reset_index()
    clip["learned_minus_human"] = clip.learned_fused_error - clip.human_error
    clip.to_csv(SOURCE_DIR / "Figure6_panel_simulation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    values = [clip.human_error, clip.ai_error, clip.fixed_fused_error, clip.learned_fused_error]
    positions = np.arange(4)
    violin = ax.violinplot(values, positions=positions, showmedians=True, widths=0.72)
    colors = [COLORS["gray"], COLORS["orange"], COLORS["sky"], COLORS["navy"]]
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(0.72)
    rng = np.random.default_rng(20260817)
    for position, series, color in zip(positions, values, colors):
        ax.scatter(
            position + rng.normal(0, 0.04, len(series)), series, s=6, alpha=0.18,
            color=COLORS["dark"], linewidths=0
        )
    ax.set_xticks(positions, ["Judge", "AI", "50:50", "Learned\nfusion"])
    ax.set_ylabel("Absolute deviation from six-judge consensus")
    _panel_label(ax, "a")
    ax = axes[1]
    difference = clip.learned_minus_human.to_numpy()
    ax.hist(difference, bins=24, color=COLORS["navy"], alpha=0.82, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=COLORS["dark"], lw=0.9, ls="--")
    ax.axvline(np.median(difference), color=COLORS["red"], lw=1.5)
    ci = analysis["bootstrap_95_ci"]
    ax.text(
        0.96, 0.95,
        f"Median change = {analysis['median_error_difference']:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]\nMean error increase = {-analysis['fusion_mean_relative_error_reduction']*100:.1f}%",
        transform=ax.transAxes, ha="right", va="top"
    )
    ax.set(xlabel="Learned-fusion error minus judge error", ylabel="Number of clips")
    _panel_label(ax, "b")
    fig.suptitle("Direct score fusion does not improve retrospective panel consensus", x=0.06, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "Figure6_judge_panel_simulation")


def _qa(paths: list[str], contracts: dict) -> dict:
    from PIL import Image

    checks = {}
    for value in paths:
        path = Path(value)
        record = {"exists": path.exists(), "nonempty": path.exists() and path.stat().st_size > 0}
        if path.suffix.lower() in {".png", ".tiff"} and path.exists():
            with Image.open(path) as image:
                record["pixel_size"] = list(image.size)
                record["mode"] = image.mode
        if path.suffix.lower() == ".svg" and path.exists():
            text = path.read_text(encoding="utf-8")
            record["editable_text_present"] = "<text" in text
        checks[str(path.relative_to(v2_paths().project))] = record
    passed = all(item.get("exists") and item.get("nonempty") for item in checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "backend": "Python/matplotlib only",
        "contracts": contracts,
        "checks": checks,
        "notes": [
            "All quantitative panels have CSV source data.",
            "Case frames are mechanically selected from real FineDiving clips.",
            "No pose skeleton is shown because the prior pose-quality gate failed.",
        ],
    }
    write_json(FIGURE_DIR / "figure_qa_v2.json", result)
    return result


def render_reports_v2() -> dict:
    require_contract_frozen()
    _style()
    prediction_path = V2_RESULTS_ROOT / "03_FINAL" / "predictions_v2.parquet"
    analysis_path = V2_RESULTS_ROOT / "04_ANALYSIS" / "analysis_summary_v2.json"
    panel_path = V2_RESULTS_ROOT / "04_ANALYSIS" / "panel_simulation_v2.parquet"
    if not prediction_path.exists() or not analysis_path.exists() or not panel_path.exists():
        raise RuntimeError("Run frozen final training and v2 analysis before rendering reports")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True); SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(prediction_path)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    final_summary = json.loads(
        (V2_RESULTS_ROOT / "03_FINAL" / "final_training_summary_v2.json").read_text(encoding="utf-8")
    )
    panel = pd.read_parquet(panel_path)
    curve = pd.read_csv(V2_RESULTS_ROOT / "04_ANALYSIS" / "coverage_risk_curve_v2.csv")
    contracts = {
        "Figure1": "The system separates phase evidence, judge disagreement, and model uncertainty before calibrated review.",
        "Figure2": "TrustDive-D preserves official-test scoring performance relative to internal baselines.",
        "Figure3": "Predicted judge disagreement tracks observed seven-judge variability.",
        "Figure4": "Calibrated review and judge-AI fusion alter retrospective consensus error.",
        "Figure5": "Additive phase contributions can be inspected on mechanically selected real examples.",
        "Figure6": "Direct score fusion increases retrospective consensus error and is not supported.",
        "archetype": "schematic-led composite plus quantitative grids and image plate",
        "export": "double-column 7.2-inch figures; editable SVG/PDF; 600 dpi TIFF; PNG preview",
    }
    write_json(FIGURE_DIR / "figure_contracts_v2.json", contracts)
    exported: list[str] = []
    exported += _figure1_pipeline()
    exported += _figure2_score(predictions, final_summary)
    exported += _figure3_disagreement(predictions, analysis["disagreement"])
    exported += _figure4_selective(panel, curve, analysis["panel"])
    exported += _figure5_cases(predictions, load_panel_targets())
    exported += _figure6_panel(panel, analysis["panel"])
    qa = _qa(exported, contracts)
    result = {"status": qa["status"], "figures": len(exported) // 4, "exports": exported, "qa": qa}
    write_json(FIGURE_DIR / "render_summary_v2.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
