from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from .config import PROJECT_ROOT, Paths
from .features import ZipFrameStore
from .util import sha256_file, write_json
from .v4_counterfactual import PHASES, exact_three_phase_shapley
from .v5_counterfactual import _phase_labels
from .v6_attribution import _load_sequences
from .v6_modeling import load_v6_assets
from .v7_attribution import _coalitions
from .v7_data import V7_RESULTS_ROOT, load_v7_frame
from .v7_modeling import load_final_artifact_v7, load_reference_map_v7


MANUSCRIPT_ROOT = PROJECT_ROOT / "04_MANUSCRIPT" / "01_FRONTIERS_PSYCHOLOGY_WORKING"
OUTPUT_ROOT = MANUSCRIPT_ROOT / "figures_final"
SOURCE_ROOT = OUTPUT_ROOT / "source_data"
CAPTION_ROOT = OUTPUT_ROOT / "captions"

PREDICTION_PATH = V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet"
EVIDENCE_PATH = V7_RESULTS_ROOT / "04_PHASE_EVIDENCE" / "phase_evidence_v7.parquet"
REVIEW_PATH = V7_RESULTS_ROOT / "05_RISK_REVIEW" / "review_priority_v7.parquet"
SUMMARY_PATH = V7_RESULTS_ROOT / "05_RISK_REVIEW" / "analysis_summary_v7.json"
ABLATION_PATH = V7_RESULTS_ROOT / "03_SCORE" / "ablation_summary_v7.csv"
REVISION_ROOT = PROJECT_ROOT / "03_RESULTS" / "MANUSCRIPT_REVISION_2026_09_03"
COMPONENT_PATH = REVISION_ROOT / "01_COMPONENT_ABLATION" / "component_metrics.csv"
SHAPLEY_AUDIT_PATH = REVISION_ROOT / "03_SHAPLEY_AUDIT" / "shapley_additional_metrics.parquet"
PHASE_CACHE = (
    PROJECT_ROOT
    / "03_RESULTS"
    / "V2_DISAGREEMENT"
    / "01_FEATURES"
    / "phase_predictions_videomae_official_v2.npz"
)

EXPECTED_HASHES = {
    PROJECT_ROOT / "03_RESULTS" / "00_AUDIT" / "manifest.parquet": "f416e4569da03df23e0c6062fa110a31421b142e91ef5c86b6c8dafffbdb3092",
    PREDICTION_PATH: "8bfd401e4edf6a8d36f73e1d687c1fb702517cc33e5d51be99cb68839cdae043",
    EVIDENCE_PATH: "9bb7d3dab55ff5c5b9092f424b3127bc566d716fdea2ee66a5d38c2a92e81068",
    REVIEW_PATH: "392bd612ee17cf802c637c97e0da2f0b1bed78410eca4b24d00f6de9ba032cc1",
    SUMMARY_PATH: "3bf06a9a01ccecab485dd8dc06c960ac995362bea3b4cf9fe24bbc48f5f8f2b5",
}

COLORS = {
    "teacher": "#8A949E",
    "ours": "#244F6B",
    "takeoff": "#4C78A8",
    "flight": "#F2A65A",
    "entry": "#2A9D8F",
    "gain": "#3E8E6B",
    "failure": "#C85C5C",
    "ink": "#24313A",
    "mid": "#66717A",
    "light": "#E9EEF2",
    "paper": "#FFFFFF",
}
PHASE_COLORS = [COLORS[phase] for phase in PHASES]
MM = 1 / 25.4
SEED = 20260820


@dataclass
class FigureData:
    frame: pd.DataFrame
    prediction: pd.DataFrame
    evidence: pd.DataFrame
    review: pd.DataFrame
    summary: dict
    phase_labels: np.ndarray
    reference_map: dict[str, np.ndarray]


def _ensure_dirs() -> None:
    for folder in (OUTPUT_ROOT, SOURCE_ROOT, CAPTION_ROOT):
        folder.mkdir(parents=True, exist_ok=True)


def _configure() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _panel(ax, label: str, x: float = -0.13, y: float = 1.06) -> None:
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom")


def _load_data() -> FigureData:
    frame = load_v7_frame().reset_index(drop=True)
    prediction = pd.read_parquet(PREDICTION_PATH)
    evidence = pd.read_parquet(EVIDENCE_PATH)
    review = pd.read_parquet(REVIEW_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    with np.load(PHASE_CACHE, allow_pickle=False) as payload:
        labels = payload["predictions"].astype(np.int8)
    if labels.shape != (3000, 8):
        raise AssertionError(f"Unexpected phase-label cache shape: {labels.shape}")
    mapping = load_reference_map_v7(final=True)
    return FigureData(frame, prediction, evidence, review, summary, labels, mapping)


def _audit_frozen(data: FigureData) -> dict:
    checks: dict[str, bool] = {}
    for path, expected in EXPECTED_HASHES.items():
        checks[f"hash::{path.name}"] = path.exists() and sha256_file(path) == expected
    test = data.prediction.analysis_role.eq("official_test")
    closed = data.evidence.analysis_role.eq("official_test") & ~data.evidence.open_set.astype(bool)
    panel = data.review.analysis_role.eq("official_test") & data.review.disagreement_primary_eligible.astype(bool)
    high = panel & data.review.high_judge_risk.astype(bool)
    checks.update(
        {
            "rows_total_3000": len(data.frame) == 3000,
            "official_test_749": int(test.sum()) == 749,
            "closed_set_735": int(closed.sum()) == 735,
            "seven_judge_test_325": int(panel.sum()) == 325,
            "high_disagreement_94": int(high.sum()) == 94,
            "open_set_14": int((test & data.prediction.open_set.astype(bool)).sum()) == 14,
            "phase_cache_alignment": data.frame.clip_uid.tolist() == data.evidence.clip_uid.tolist(),
        }
    )
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    write_json(OUTPUT_ROOT / "frozen_input_audit.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("Frozen manuscript input audit failed")
    return result


def _merged(data: FigureData) -> pd.DataFrame:
    pred = data.prediction.copy()
    pred["teacher_error"] = np.abs(pred.teacher_predicted_score - pred.dive_score)
    pred["ours_error"] = np.abs(pred.trustdive_predicted_score - pred.dive_score)
    pred["error_gain"] = pred.teacher_error - pred.ours_error
    evidence_cols = [
        "clip_uid",
        "reference_baseline",
        "predicted_quality",
        "phi_takeoff",
        "phi_flight",
        "phi_entry",
        "top_phase",
        "actual_max_intervention_phase",
        "top_phase_intervention_match",
        "targeted_intervention_effect",
        "random_intervention_effect",
        "phase_boundary_cosine",
        "phase_boundary_top_match",
        "reference_change_cosine",
    ] + [f"coalition_{i}" for i in range(8)]
    review_cols = ["clip_uid", "high_judge_risk", "review_priority", "review_reason"]
    return pred.merge(data.evidence[evidence_cols], on="clip_uid").merge(data.review[review_cols], on="clip_uid")


def _valid_visual_indices(data: FigureData) -> set[int]:
    valid: set[int] = set()
    refs = data.reference_map["references"][:, :5].astype(int)
    for index in range(len(data.frame)):
        if bool(data.reference_map["open_set"][index]):
            continue
        ref = next((int(item) for item in refs[index] if item >= 0), -1)
        if ref < 0:
            continue
        if set(np.unique(data.phase_labels[index])) == {0, 1, 2} and set(np.unique(data.phase_labels[ref])) == {0, 1, 2}:
            valid.add(index)
    return valid


def _pick_case(
    candidates: pd.DataFrame,
    valid_uids: set[str],
    used: set[str],
    column: str,
    quantile: float,
) -> pd.Series:
    pool = candidates[candidates.clip_uid.isin(valid_uids) & ~candidates.clip_uid.isin(used)].copy()
    if pool.empty:
        raise RuntimeError("No legal mechanically selected case remained")
    target = float(pool[column].quantile(quantile))
    pool["_distance_to_selection_target"] = np.abs(pool[column] - target)
    return pool.sort_values(["_distance_to_selection_target", "clip_uid"], kind="stable").iloc[0]


def _select_cases(data: FigureData) -> pd.DataFrame:
    merged = _merged(data)
    test = merged[(merged.analysis_role == "official_test") & ~merged.open_set.astype(bool)].copy()
    index_by_uid = pd.Series(np.arange(len(data.frame)), index=data.frame.clip_uid)
    valid_indices = _valid_visual_indices(data)
    valid_uids = set(data.frame.iloc[sorted(valid_indices)].clip_uid)
    used: set[str] = set()
    rows: list[dict] = []
    specifications = [
        (
            "typical_accurate",
            "Typical accurate",
            test[(test.error_gain >= 0) & (test.ours_error <= test.ours_error.median())],
            "ours_error",
            0.50,
            "Accurate reference-supported case nearest the median error of the better-performing half",
        ),
        (
            "high_disagreement_gain",
            "High-disagreement improvement",
            test[test.high_judge_risk.astype(bool) & (test.error_gain > 0)],
            "error_gain",
            0.50,
            "High-disagreement case nearest the median positive paired improvement",
        ),
        (
            "scoring_failure",
            "Scoring failure",
            test[test.error_gain < 0],
            "ours_error",
            0.90,
            "Worsened case nearest the 90th percentile TrustDive absolute error",
        ),
        (
            "boundary_sensitive",
            "Boundary-sensitive evidence",
            test[np.isfinite(test.phase_boundary_cosine)],
            "phase_boundary_cosine",
            0.10,
            "Case nearest the 10th percentile boundary-shift cosine similarity",
        ),
        (
            "reference_sensitive",
            "Reference-sensitive evidence",
            test[np.isfinite(test.reference_change_cosine)],
            "reference_change_cosine",
            0.10,
            "Case nearest the 10th percentile alternate-reference cosine similarity",
        ),
    ]
    refs = data.reference_map["references"][:, :5].astype(int)
    distances = data.reference_map["distances"][:, :5].astype(float)
    for order, (case_type, label, pool, column, quantile, rule) in enumerate(specifications):
        row = _pick_case(pool, valid_uids, used, column, quantile)
        used.add(str(row.clip_uid))
        query_index = int(index_by_uid.loc[row.clip_uid])
        reference_slots = [int(value) for value in refs[query_index] if value >= 0]
        reference_index = reference_slots[0]
        rows.append(
            {
                "case_order": order,
                "case_type": case_type,
                "case_label": label,
                "selection_rule": rule,
                "selection_metric": column,
                "selection_value": float(row[column]),
                "clip_uid": row.clip_uid,
                "query_index": query_index,
                "reference_clip_uid": data.frame.iloc[reference_index].clip_uid,
                "reference_index": reference_index,
                "reference_distance": float(distances[query_index, 0]),
                "official_score": float(row.dive_score),
                "teacher_score": float(row.teacher_predicted_score),
                "trustdive_score": float(row.trustdive_predicted_score),
                "ours_error": float(row.ours_error),
                "error_gain": float(row.error_gain),
                "high_disagreement": bool(row.high_judge_risk),
                "reference_baseline": float(row.reference_baseline),
                "phi_takeoff": float(row.phi_takeoff),
                "phi_flight": float(row.phi_flight),
                "phi_entry": float(row.phi_entry),
                "top_phase": row.top_phase,
                "phase_boundary_cosine": float(row.phase_boundary_cosine),
                "reference_change_cosine": float(row.reference_change_cosine),
            }
        )
    return pd.DataFrame(rows)


def _phase_frame_indices(labels: np.ndarray, frame_count: int) -> dict[str, list[int]]:
    if frame_count < 1:
        raise ValueError("Cannot select frames from an empty clip")
    sampled = labels[np.linspace(0, len(labels) - 1, frame_count).round().astype(int)]
    result: dict[str, list[int]] = {}
    for phase_index, phase in enumerate(PHASES):
        available = np.flatnonzero(sampled == phase_index)
        if not len(available):
            raise ValueError(f"Predicted phase absent: {phase}")
        positions = np.quantile(available, [0.25, 0.50, 0.75]).round().astype(int)
        result[phase] = positions.tolist()
    return result


def _frame_audit(data: FigureData, cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    with ZipFrameStore(Paths().trimmed_zip) as store:
        for case in cases.itertuples(index=False):
            for role, index in (("query", int(case.query_index)), ("reference", int(case.reference_index))):
                meta = data.frame.iloc[index]
                names = store.frame_names(meta.source, int(meta.instance))
                selections = _phase_frame_indices(data.phase_labels[index], len(names))
                for phase, indices in selections.items():
                    for within_phase, frame_index in enumerate(indices):
                        rows.append(
                            {
                                "case_type": case.case_type,
                                "role": role,
                                "clip_uid": meta.clip_uid,
                                "phase": phase,
                                "within_phase_order": within_phase,
                                "frame_index_in_trimmed_clip": frame_index,
                                "archive_member": names[frame_index],
                                "global_adjustment": "none",
                                "local_adjustment": False,
                            }
                        )
    return pd.DataFrame(rows)


def _cluster_bootstrap_mean(values: np.ndarray, groups: np.ndarray, iterations: int = 10_000) -> tuple[float, float]:
    unique = np.unique(groups.astype(str))
    group_sum = np.asarray([values[groups.astype(str) == item].sum() for item in unique], dtype=float)
    group_n = np.asarray([(groups.astype(str) == item).sum() for item in unique], dtype=float)
    rng = np.random.default_rng(SEED)
    draws = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(unique), len(unique))
        draws[iteration] = group_sum[sampled].sum() / group_n[sampled].sum()
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def _build_figure2_source(data: FigureData) -> None:
    merged = _merged(data)
    test = merged[merged.analysis_role == "official_test"].copy()
    test["paired_mae_difference"] = test.ours_error - test.teacher_error
    panel = test[test.disagreement_primary_eligible.astype(bool)]
    high = panel[panel.high_judge_risk.astype(bool)]
    rows = []
    for label, subset in (("All official test", test), ("Seven-judge test", panel), ("High disagreement", high)):
        values = subset.paired_mae_difference.to_numpy(dtype=float)
        lower, upper = _cluster_bootstrap_mean(values, subset.event_family.to_numpy())
        rows.append(
            {
                "subset": label,
                "n": len(subset),
                "trustdive_minus_rica2_mae": float(values.mean()),
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    test.to_csv(SOURCE_ROOT / "Figure2_video_level.csv", index=False)
    pd.DataFrame(rows).to_csv(SOURCE_ROOT / "Figure2_effects.csv", index=False)


def _build_figure4_source(data: FigureData, cases: pd.DataFrame) -> None:
    evidence = data.evidence[(data.evidence.analysis_role == "official_test") & ~data.evidence.open_set.astype(bool)].copy()
    evidence["targeted_minus_random"] = evidence.targeted_intervention_effect - evidence.random_intervention_effect
    audit = pd.read_parquet(SHAPLEY_AUDIT_PATH)
    evidence = evidence.merge(
        audit[["clip_uid", "targeted_effect", "mean_nonselected_effect", "strongest_nonselected_effect", "loo_reconstruction_error"]],
        on="clip_uid",
        how="inner",
        validate="one_to_one",
    )
    evidence.to_csv(SOURCE_ROOT / "Figure4_video_level.csv", index=False)
    selected_uid = cases.loc[cases.case_type == "high_disagreement_gain", "clip_uid"].iloc[0]
    selected = evidence[evidence.clip_uid == selected_uid].copy()
    selected.to_csv(SOURCE_ROOT / "Figure4_hero_case.csv", index=False)
    confusion = pd.crosstab(evidence.top_phase, evidence.actual_max_intervention_phase).reindex(index=PHASES, columns=PHASES, fill_value=0)
    confusion.rename_axis("attributed_phase").reset_index().to_csv(SOURCE_ROOT / "Figure4_phase_confusion.csv", index=False)


def _selected_perturbations(data: FigureData, cases: pd.DataFrame) -> pd.DataFrame:
    selected = cases[cases.case_type.isin(["boundary_sensitive", "reference_sensitive"])].copy()
    assets = load_v6_assets()
    mapping = load_reference_map_v7(final=True)
    artifact = load_final_artifact_v7()
    sequences, actions = _load_sequences()
    labels = _phase_labels(sequences.shape[1])
    persisted = data.evidence.set_index("clip_uid")
    rows: list[dict] = []
    for case in selected.itertuples(index=False):
        index = int(case.query_index)
        variants = ["boundary_left", "boundary_right"] if case.case_type == "boundary_sensitive" else ["reference_replace"]
        original_phi = persisted.loc[case.clip_uid, ["phi_takeoff", "phi_flight", "phi_entry"]].to_numpy(dtype=float)
        for phase, value in zip(PHASES, original_phi):
            rows.append({"case_type": case.case_type, "clip_uid": case.clip_uid, "variant": "original", "phase": phase, "contribution": value})
        original_coalition = _coalitions(np.asarray([index]), "original", assets, mapping, artifact, sequences, actions, labels)[0]
        persisted_quality = float(persisted.loc[case.clip_uid, "predicted_quality"])
        endpoint_difference = abs(float(original_coalition[7]) - persisted_quality)
        if endpoint_difference > 1e-3:
            raise AssertionError(f"Frozen scorer path mismatch for {case.clip_uid}: {endpoint_difference}")
        for variant in variants:
            coalition = _coalitions(np.asarray([index]), variant, assets, mapping, artifact, sequences, actions, labels)
            phi = exact_three_phase_shapley(coalition[:, None, :])[0, 0]
            for phase, value in zip(PHASES, phi):
                rows.append({"case_type": case.case_type, "clip_uid": case.clip_uid, "variant": variant, "phase": phase, "contribution": float(value)})
    return pd.DataFrame(rows)


def _build_supplement_sources(data: FigureData) -> None:
    frame = data.frame.copy()
    frame.to_csv(SOURCE_ROOT / "FigureS1_dataset_overview.csv", index=False)
    pd.read_csv(COMPONENT_PATH).to_csv(SOURCE_ROOT / "FigureS2_ablation.csv", index=False)
    data.prediction[["clip_uid", "analysis_role", "valid_reference_count", "open_set", "reference_distance", "reference_dispersion"]].to_csv(
        SOURCE_ROOT / "FigureS3_reference_coverage.csv", index=False
    )
    test = data.review[data.review.analysis_role == "official_test"].merge(
        data.prediction[["clip_uid", "teacher_uncertainty"]], on="clip_uid"
    )
    test["absolute_error"] = np.abs(test.trustdive_predicted_score - test.dive_score)
    curves = []
    for strategy, risk in (("RICA2 uncertainty", test.teacher_uncertainty.to_numpy()), ("Combined exploratory priority", test.review_priority.to_numpy())):
        order = np.argsort(risk, kind="stable")
        coverage = np.arange(1, len(order) + 1) / len(order)
        mae = np.cumsum(test.absolute_error.to_numpy()[order]) / np.arange(1, len(order) + 1)
        curves.append(pd.DataFrame({"strategy": strategy, "coverage": coverage, "accepted_mae": mae}))
    pd.concat(curves, ignore_index=True).to_csv(SOURCE_ROOT / "FigureS4_selective_review.csv", index=False)


def _write_contracts() -> None:
    contracts = {
        "Figure1": {"conclusion": "TrustDive combines bounded latent calibration with exact reference-conditioned phase evidence.", "archetype": "schematic-led composite", "size_mm": [180, 105]},
        "Figure2": {"conclusion": "Latent calibration reduces score error overall and in high-disagreement performances.", "archetype": "quantitative grid", "size_mm": [180, 128]},
        "Figure3": {"conclusion": "Enlarged real query phases and matched training references make the score decomposition inspectable in a typical and a high-disagreement performance.", "archetype": "image plate + quant", "size_mm": [180, 140]},
        "Figure4": {"conclusion": "Exact phase attributions identify a larger direct intervention than either nonselected phase.", "archetype": "asymmetric mixed-modality figure", "size_mm": [180, 150]},
        "Figure5": {"conclusion": "Phase evidence is informative but only moderately stable to boundary and reference changes.", "archetype": "asymmetric mixed-modality figure", "size_mm": [180, 135]},
        "backend": "Python/matplotlib exclusively",
        "exports": ["SVG", "PDF", "600 dpi TIFF", "300 dpi PNG"],
        "image_integrity": "Uniform letterboxing only; no local retouching or unvalidated pose overlays.",
        "review_risks": ["model attribution is not judge cognition", "high disagreement is not abnormal judging", "feature-level coalitions are not generated videos"],
    }
    write_json(OUTPUT_ROOT / "figure_contracts.json", contracts)


def build_source_data() -> dict:
    _ensure_dirs()
    data = _load_data()
    audit = _audit_frozen(data)
    cases = _select_cases(data)
    cases.to_csv(SOURCE_ROOT / "case_selection_audit.csv", index=False)
    _frame_audit(data, cases).to_csv(SOURCE_ROOT / "frame_selection_audit.csv", index=False)
    _build_figure2_source(data)
    _build_figure4_source(data, cases)
    stability = data.evidence[(data.evidence.analysis_role == "official_test") & ~data.evidence.open_set.astype(bool)].copy()
    stability.to_csv(SOURCE_ROOT / "Figure5_stability.csv", index=False)
    perturbations = _selected_perturbations(data, cases)
    perturbations.to_csv(SOURCE_ROOT / "Figure5_selected_perturbations.csv", index=False)
    _build_supplement_sources(data)
    _write_contracts()
    outputs = sorted(SOURCE_ROOT.glob("*.csv"))
    result = {
        "status": "PASS",
        "audit": audit,
        "case_count": len(cases),
        "source_data": {path.name: sha256_file(path) for path in outputs},
    }
    write_json(OUTPUT_ROOT / "source_data_manifest.json", result)
    return result


def _read_member(store: ZipFrameStore, member: str) -> Image.Image:
    import io

    with store.archive.open(member) as handle:
        return Image.open(io.BytesIO(handle.read())).convert("RGB")


def _letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#F5F7F8")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def _phase_tile(store: ZipFrameStore, meta: pd.Series, labels: np.ndarray, phase: str, width: int = 112, height: int = 102) -> Image.Image:
    names = store.frame_names(meta.source, int(meta.instance))
    indices = _phase_frame_indices(labels, len(names))[phase]
    frames = [_letterbox(_read_member(store, names[index]), (width, height)) for index in indices]
    # A compact overlapping strip keeps individual frames legible at journal size.
    offset_px = max(16, round(width * 0.20))
    canvas = Image.new("RGB", (width + offset_px * 2, height + 6), "white")
    draw = ImageDraw.Draw(canvas)
    for offset in reversed(range(3)):
        x = offset * offset_px
        canvas.paste(frames[offset], (x, 6))
        draw.rectangle((x, 6, x + width - 1, height + 5), outline="white", width=2)
    draw.rectangle((0, 0, canvas.width - 1, 5), fill=COLORS[phase])
    return canvas


def _phase_center(store: ZipFrameStore, meta: pd.Series, labels: np.ndarray, phase: str, size: tuple[int, int] = (160, 112)) -> Image.Image:
    names = store.frame_names(meta.source, int(meta.instance))
    index = _phase_frame_indices(labels, len(names))[phase][1]
    return _letterbox(_read_member(store, names[index]), size)


def _motion_focused_phase_frame(
    store: ZipFrameStore,
    meta: pd.Series,
    labels: np.ndarray,
    phase: str,
    size: tuple[int, int] = (280, 190),
) -> Image.Image:
    """Return the phase midpoint with one deterministic, sequence-level crop.

    The crop center is estimated from change across the frozen 25/50/75% phase
    frames.  This is a display-only global crop: it does not add pose labels,
    alter pixels locally, or affect any model output.
    """

    names = store.frame_names(meta.source, int(meta.instance))
    indices = _phase_frame_indices(labels, len(names))[phase]
    frames = [_read_member(store, names[index]) for index in indices]
    arrays = np.stack([np.asarray(frame.convert("L"), dtype=np.float32) for frame in frames])
    motion = arrays.max(axis=0) - arrays.min(axis=0)
    threshold = float(np.quantile(motion, 0.88))
    weights = np.clip(motion - threshold, 0, None)
    height, width = motion.shape
    if float(weights.sum()) > 1e-6:
        yy, xx = np.mgrid[:height, :width]
        center_x = float((weights * xx).sum() / weights.sum())
        center_y = float((weights * yy).sum() / weights.sum())
    else:
        center_x, center_y = width / 2, height / 2

    target_aspect = size[0] / size[1]
    crop_height = min(height * 0.82, width * 0.62 / target_aspect)
    crop_width = crop_height * target_aspect
    left = min(max(center_x - crop_width / 2, 0), width - crop_width)
    top = min(max(center_y - crop_height / 2, 0), height - crop_height)
    box = (
        int(round(left)),
        int(round(top)),
        int(round(left + crop_width)),
        int(round(top + crop_height)),
    )
    focused = frames[1].crop(box)
    return ImageOps.fit(focused, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _show_image(ax, image: Image.Image, border: str | None = None, linewidth: float = 1.4) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(border is not None)
        if border is not None:
            spine.set_color(border)
            spine.set_linewidth(linewidth)


def _save(fig, stem: str) -> dict:
    paths: dict[str, dict] = {}
    for suffix, dpi in (("svg", None), ("pdf", None), ("tiff", 600), ("png", 300)):
        path = OUTPUT_ROOT / f"{stem}.{suffix}"
        kwargs = {"facecolor": "white"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths[suffix] = {"path": str(path), "sha256": sha256_file(path)}
    return paths


def _quality_metrics(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    from scipy.stats import spearmanr

    return float(spearmanr(observed, predicted).statistic), float(np.mean(np.abs(predicted - observed)))


def render_figure2() -> dict:
    import matplotlib.pyplot as plt

    _configure()
    data = pd.read_csv(SOURCE_ROOT / "Figure2_video_level.csv")
    effects = pd.read_csv(SOURCE_ROOT / "Figure2_effects.csv")
    fig = plt.figure(figsize=(180 * MM, 128 * MM), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.22, 0.88], wspace=0.24, hspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharex=ax_a, sharey=ax_a)
    limits = [float(data.dive_score.min()), float(data.dive_score.max())]
    for ax, column, title, color, label in (
        (ax_a, "teacher_predicted_score", "RICA²", COLORS["teacher"], "A"),
        (ax_b, "trustdive_predicted_score", "TrustDive", COLORS["ours"], "B"),
    ):
        ax.scatter(data.dive_score, data[column], s=10, alpha=0.38, color=color, edgecolors="none", rasterized=True)
        ax.plot(limits, limits, "--", color=COLORS["ink"], lw=1.2)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Official score")
        ax.set_ylabel("Predicted score")
        rho, mae = _quality_metrics(data.dive_score.to_numpy(), data[column].to_numpy())
        ax.text(
            0.03,
            0.97,
            f"{title}\nρ = {rho:.3f}   MAE = {mae:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 1.8},
        )
        _panel(ax, label, x=-0.18, y=1.08)
    ax_c = fig.add_subplot(grid[1, 0])
    rng = np.random.default_rng(SEED)
    groups = [data, data[data.high_judge_risk.astype(bool)]]
    labels = ["All test\n($n$=749)", "High disagreement\n($n$=94)"]
    values = [group.teacher_error.to_numpy() - group.ours_error.to_numpy() for group in groups]
    parts = ax_c.violinplot(values, positions=[0, 1], widths=0.72, showextrema=False)
    for body, color in zip(parts["bodies"], [COLORS["ours"], COLORS["flight"]]):
        body.set_facecolor(color)
        body.set_alpha(0.24)
        body.set_edgecolor(color)
    for pos, vals, color in zip([0, 1], values, [COLORS["ours"], COLORS["flight"]]):
        jitter = rng.normal(pos, 0.055, len(vals))
        ax_c.scatter(jitter, vals, s=7, alpha=0.22, color=color, edgecolors="none", rasterized=True)
        ax_c.plot([pos - 0.18, pos + 0.18], [np.median(vals)] * 2, color=COLORS["ink"], lw=2.2)
    ax_c.axhline(0, color=COLORS["ink"], ls="--", lw=1.2)
    ax_c.set_xticks([0, 1], labels)
    ax_c.set_ylabel("Absolute-error reduction\n(RICA² − TrustDive; score points)")
    ax_c.set_ylim(-8.6, 8.8)
    ax_c.text(
        0.98,
        0.95,
        "positive = lower TrustDive error",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        color=COLORS["ours"],
        fontweight="bold",
        fontsize=8,
    )
    ax_c.tick_params(axis="x", labelsize=8)
    _panel(ax_c, "C", x=-0.10, y=1.08)
    ax_d = fig.add_subplot(grid[1, 1])
    y = np.arange(len(effects))[::-1]
    improvement = -effects.trustdive_minus_rica2_mae.to_numpy(dtype=float)
    lower = -effects.ci_upper.to_numpy(dtype=float)
    upper = -effects.ci_lower.to_numpy(dtype=float)
    ax_d.errorbar(
        improvement,
        y,
        xerr=[improvement - lower, upper - improvement],
        fmt="o",
        color=COLORS["ours"],
        ecolor=COLORS["ours"],
        capsize=3,
        markersize=5,
    )
    ax_d.axvline(0, color=COLORS["ink"], ls="--", lw=1.2)
    short_names = {
        "All official test": "All test",
        "Seven-judge test": "Seven-judge",
        "High disagreement": "High disagreement",
    }
    ax_d.set_yticks(y, [f"{short_names.get(row.subset, row.subset)}  ($n$={row.n})" for row in effects.itertuples(index=False)])
    ax_d.set_xlabel("MAE reduction (RICA² − TrustDive)")
    ax_d.set_xlim(min(-0.25, float(lower.min()) - 0.15), float(upper.max()) + 0.35)
    for x_value, y_value in zip(improvement, y):
        ax_d.text(
            x_value + 0.08,
            y_value,
            f"{x_value:.2f}",
            va="center",
            ha="left",
            color=COLORS["ours"],
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.4},
        )
    _panel(ax_d, "D", x=-0.10, y=1.08)
    return _save(fig, "Figure2_scoring_performance")


def _case_rows() -> pd.DataFrame:
    return pd.read_csv(SOURCE_ROOT / "case_selection_audit.csv").sort_values("case_order")


def render_figure3() -> dict:
    import matplotlib.pyplot as plt

    _configure()
    data = _load_data()
    cases = _case_rows().query("case_type in ['typical_accurate','high_disagreement_gain']")
    fig = plt.figure(figsize=(180 * MM, 140 * MM))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.975, bottom=0.035)
    outer = fig.add_gridspec(2, 1, hspace=0.14)
    with ZipFrameStore(Paths().trimmed_zip) as store:
        for row_index, case in enumerate(cases.itertuples(index=False)):
            edge = COLORS["gain"] if case.case_type in {"typical_accurate", "high_disagreement_gain"} else COLORS["failure"]
            row_grid = outer[row_index].subgridspec(
                2,
                5,
                height_ratios=[1.70, 0.78],
                width_ratios=[1.16, 1.08, 1.08, 1.08, 1.10],
                hspace=0.06,
                wspace=0.08,
            )
            ax_text = fig.add_subplot(row_grid[:, 0])
            ax_text.axis("off")
            display_label = {
                "typical_accurate": "Typical accurate",
                "high_disagreement_gain": "High-disagreement\nimprovement",
            }[case.case_type]
            ax_text.text(0, 0.96, display_label, fontweight="bold", color=edge, va="top", fontsize=8.5, linespacing=1.05)
            direction = "reduced" if case.error_gain >= 0 else "increased"
            score_y = [0.67, 0.54, 0.41]
            for y_position, score_label, score_value in zip(
                score_y,
                ["Official", "RICA²", "TrustDive"],
                [case.official_score, case.teacher_score, case.trustdive_score],
            ):
                ax_text.text(0, y_position, score_label, va="top", fontsize=8)
                ax_text.text(0.98, y_position, f"{score_value:.1f}", va="top", ha="right", fontsize=8, family="monospace")
            ax_text.text(0, 0.18, f"|error| {direction}\nby {abs(case.error_gain):.1f} points", color=edge, va="bottom", fontsize=8, fontweight="bold")
            if row_index == 0:
                _panel(ax_text, "A", x=-0.10, y=1.02)
            q_meta = data.frame.iloc[int(case.query_index)]
            r_meta = data.frame.iloc[int(case.reference_index)]
            for phase_index, phase in enumerate(PHASES):
                q_ax = fig.add_subplot(row_grid[0, 1 + phase_index])
                query_image = _motion_focused_phase_frame(
                    store,
                    q_meta,
                    data.phase_labels[int(case.query_index)],
                    phase,
                    size=(300, 205),
                )
                _show_image(q_ax, query_image, PHASE_COLORS[phase_index], linewidth=2.0)
                q_ax.set_title(phase.capitalize(), color=PHASE_COLORS[phase_index], fontweight="bold", pad=2)
                if phase_index == 0:
                    q_ax.text(0.02, 0.96, "QUERY", transform=q_ax.transAxes, va="top", ha="left", color="white", fontweight="bold", fontsize=8, bbox={"facecolor": "black", "alpha": 0.62, "pad": 1.5, "edgecolor": "none"})

                r_ax = fig.add_subplot(row_grid[1, 1 + phase_index])
                reference_image = _motion_focused_phase_frame(
                    store,
                    r_meta,
                    data.phase_labels[int(case.reference_index)],
                    phase,
                    size=(300, 118),
                )
                _show_image(r_ax, reference_image, PHASE_COLORS[phase_index], linewidth=1.5)
                if phase_index == 0:
                    r_ax.text(0.02, 0.94, "REFERENCE", transform=r_ax.transAxes, va="top", ha="left", color="white", fontweight="bold", fontsize=8, bbox={"facecolor": "black", "alpha": 0.62, "pad": 1.3, "edgecolor": "none"})
            ax_bar = fig.add_subplot(row_grid[:, 4])
            values = np.asarray([case.phi_takeoff, case.phi_flight, case.phi_entry])
            x_positions = np.arange(3)
            ax_bar.bar(x_positions, values, color=PHASE_COLORS, width=0.62)
            ax_bar.axhline(0, color=COLORS["ink"], lw=1.0)
            ax_bar.set_xticks(x_positions, ["T", "F", "E"])
            ax_bar.tick_params(axis="x", labelsize=8, pad=2)
            limit = max(0.18, float(np.max(np.abs(values))) * 1.70)
            ax_bar.set_ylim(-limit, limit)
            for idx, value in enumerate(values):
                offset = limit * 0.055
                ax_bar.text(idx, value + (offset if value >= 0 else -offset), f"{value:+.2f}", va="bottom" if value >= 0 else "top", ha="center", fontsize=8, color=COLORS["ink"])
            ax_bar.text(0.03, 0.97, f"Highest: {case.top_phase}", transform=ax_bar.transAxes, va="top", ha="left", fontsize=8, fontweight="bold")
    return _save(fig, "Figure3_phase_case_matrix")


def _coalition_tile(store: ZipFrameStore, q_meta: pd.Series, r_meta: pd.Series, q_labels: np.ndarray, r_labels: np.ndarray, mask: int) -> Image.Image:
    size = (150, 104)
    gap = 4
    canvas = Image.new("RGB", (size[0] * 3 + gap * 2, size[1] + 20), "white")
    draw = ImageDraw.Draw(canvas)
    for phase_index, phase in enumerate(PHASES):
        use_query = bool(mask & (1 << phase_index))
        meta, labels = (q_meta, q_labels) if use_query else (r_meta, r_labels)
        image = _phase_center(store, meta, labels, phase, size=size)
        x = phase_index * (size[0] + gap)
        canvas.paste(image, (x, 20))
        draw.rectangle((x, 20, x + size[0] - 1, 20 + size[1] - 1), outline=COLORS[phase], width=4)
        draw.text((x + 5, 3), f"{phase[0].upper()}:{'Q' if use_query else 'R'}", fill=COLORS[phase])
    return canvas


def render_figure4() -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _configure()
    data = _load_data()
    cases = _case_rows()
    case = cases[cases.case_type == "high_disagreement_gain"].iloc[0]
    hero = pd.read_csv(SOURCE_ROOT / "Figure4_hero_case.csv").iloc[0]
    video = pd.read_csv(SOURCE_ROOT / "Figure4_video_level.csv")
    confusion = pd.read_csv(SOURCE_ROOT / "Figure4_phase_confusion.csv").set_index("attributed_phase").loc[PHASES, PHASES]
    fig = plt.figure(figsize=(180 * MM, 150 * MM), layout="constrained")
    outer = fig.add_gridspec(3, 4, height_ratios=[0.22, 0.22, 1.35], hspace=0.14, wspace=0.25)
    q_meta = data.frame.iloc[int(case.query_index)]
    r_meta = data.frame.iloc[int(case.reference_index)]
    with ZipFrameStore(Paths().trimmed_zip) as store:
        for mask in range(8):
            ax = fig.add_subplot(outer[mask // 4, mask % 4])
            tile = _coalition_tile(store, q_meta, r_meta, data.phase_labels[int(case.query_index)], data.phase_labels[int(case.reference_index)], mask)
            _show_image(ax, tile, COLORS["light"], 0.8)
            ax.set_title(f"{mask:03b}   F = {hero[f'coalition_{mask}']:.2f}", fontsize=8, pad=2)
            if mask == 0:
                _panel(ax, "A", x=-0.14, y=1.10)
    ax_b = fig.add_subplot(outer[2, 0])
    components = [hero.reference_baseline, hero.phi_takeoff, hero.phi_flight, hero.phi_entry]
    labels = ["Baseline", "Takeoff", "Flight", "Entry"]
    colors = [COLORS["light"], *PHASE_COLORS]
    running = float(components[0])
    ax_b.scatter([0], [running], s=45, color=COLORS["teacher"], zorder=4)
    ax_b.text(0, running + 0.035, f"{running:.2f}", ha="center", fontsize=8)
    levels = [running]
    for index, (value, color) in enumerate(zip(components[1:], colors[1:]), start=1):
        next_level = running + float(value)
        ax_b.bar(index, value, bottom=min(running, next_level), color=color, edgecolor="white", width=0.65)
        ax_b.plot([index - 0.55, index - 0.32], [running, running], color=COLORS["mid"], lw=1.0)
        ax_b.text(index, (running + next_level) / 2, f"{value:+.2f}", ha="center", va="center", fontsize=8)
        running = next_level
        levels.append(running)
    ax_b.axhline(hero.predicted_quality, color=COLORS["ink"], ls="--", lw=1.2)
    ax_b.set_xticks(range(4), labels, rotation=28, ha="right")
    ax_b.set_ylabel("Execution-quality score")
    margin = max(0.18, (max(levels) - min(levels)) * 0.28)
    ax_b.set_ylim(min(levels) - margin, max(levels) + margin)
    ax_b.set_title("Exact additive reconstruction", loc="left", fontweight="bold", fontsize=8, pad=5)
    _panel(ax_b, "B", y=1.12)
    ax_c = fig.add_subplot(outer[2, 1:3])
    rng = np.random.default_rng(SEED)
    values = [
        video.mean_nonselected_effect.to_numpy(),
        video.strongest_nonselected_effect.to_numpy(),
        video.targeted_effect.to_numpy(),
    ]
    positions = [0, 1, 2]
    parts = ax_c.violinplot(values, positions=positions, widths=0.72, showextrema=False)
    for body, color in zip(parts["bodies"], [COLORS["teacher"], COLORS["mid"], COLORS["ours"]]):
        body.set_facecolor(color)
        body.set_alpha(0.25)
        body.set_edgecolor(color)
    subset = rng.choice(len(video), size=min(320, len(video)), replace=False)
    for pos, vals, color in zip(positions, values, [COLORS["teacher"], COLORS["mid"], COLORS["ours"]]):
        ax_c.scatter(rng.normal(pos, 0.05, len(subset)), vals[subset], s=6, alpha=0.25, color=color, edgecolors="none", rasterized=True)
        ax_c.plot([pos - 0.18, pos + 0.18], [np.median(vals)] * 2, color=COLORS["ink"], lw=2.2)
    delta_mean = float(np.median(video.targeted_effect - video.mean_nonselected_effect))
    delta_strong = float(np.median(video.targeted_effect - video.strongest_nonselected_effect))
    ax_c.set_xticks(positions, ["Mean of\nnonselected", "Strongest\nnonselected", "Highest\nattribution"])
    ax_c.set_ylabel("Absolute replacement effect")
    ax_c.set_title(
        f"Highest minus mean = {delta_mean:.3f}; minus strongest = {delta_strong:.3f}",
        loc="left",
        fontweight="bold",
        fontsize=8,
        pad=5,
    )
    _panel(ax_c, "C", x=-0.08, y=1.12)
    ax_d = fig.add_subplot(outer[2, 3])
    cmap = LinearSegmentedColormap.from_list("trust", ["#F4F7F9", COLORS["ours"]])
    matrix = confusion.to_numpy(dtype=int)
    ax_d.imshow(matrix, cmap=cmap, aspect="auto")
    threshold = matrix.max() * 0.5
    for row in range(3):
        for col in range(3):
            ax_d.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > threshold else COLORS["ink"], fontweight="bold")
    ax_d.set_xticks(range(3), ["Takeoff", "Flight", "Entry"], rotation=35, ha="right")
    ax_d.set_yticks(range(3), ["Takeoff", "Flight", "Entry"])
    ax_d.set_xlabel("Largest intervention")
    ax_d.set_ylabel("Highest attribution")
    ax_d.set_title("Match = 90.61%", loc="left", fontweight="bold", fontsize=8, pad=5)
    _panel(ax_d, "D", y=1.12)
    return _save(fig, "Figure4_counterfactual_fidelity")


def _bootstrap_phase_mean(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = []
    for phase in PHASES:
        subset = frame[frame.top_phase == phase]
        values = subset[value].to_numpy(dtype=float)
        lo, hi = _cluster_bootstrap_mean(values, subset.event_family.to_numpy(), iterations=4000)
        rows.append({"phase": phase, "n": len(subset), "mean": float(np.mean(values)), "ci_lower": lo, "ci_upper": hi})
    return pd.DataFrame(rows)


def render_figure5() -> dict:
    import matplotlib.pyplot as plt

    _configure()
    data = _load_data()
    stability = pd.read_csv(SOURCE_ROOT / "Figure5_stability.csv")
    perturb = pd.read_csv(SOURCE_ROOT / "Figure5_selected_perturbations.csv")
    cases = _case_rows().set_index("case_type")
    fig = plt.figure(figsize=(180 * MM, 135 * MM), layout="constrained")
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.12], hspace=0.34, wspace=0.42)
    ax_a = fig.add_subplot(grid[0, :2])
    counts = stability.top_phase.value_counts().reindex(PHASES, fill_value=0)
    shares = counts / counts.sum() * 100
    bars = ax_a.bar(PHASES, shares, color=PHASE_COLORS, width=0.68)
    for bar, count, share in zip(bars, counts, shares):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{share:.1f}%\n$n$={count}", ha="center", va="bottom", fontsize=8)
    ax_a.set_ylabel("Highest-contribution phase (%)")
    ax_a.set_ylim(0, max(shares) * 1.22)
    _panel(ax_a, "A")
    ax_b = fig.add_subplot(grid[0, 2:4])
    distributions = [stability.phase_boundary_cosine.dropna().to_numpy(), stability.reference_change_cosine.dropna().to_numpy()]
    parts = ax_b.violinplot(distributions, positions=[0, 1], widths=0.72, showextrema=False)
    for body, color in zip(parts["bodies"], [COLORS["flight"], COLORS["takeoff"]]):
        body.set_facecolor(color)
        body.set_alpha(0.28)
        body.set_edgecolor(color)
    for pos, values in enumerate(distributions):
        ax_b.boxplot(values, positions=[pos], widths=0.20, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "edgecolor": COLORS["ink"]}, medianprops={"color": COLORS["ink"], "linewidth": 1.8}, whiskerprops={"color": COLORS["ink"]}, capprops={"color": COLORS["ink"]})
    ax_b.set_xticks([0, 1], ["Boundary\nshift", "Alternate\nreferences"])
    ax_b.set_ylabel("Contribution-vector cosine")
    ax_b.set_ylim(-1.02, 1.04)
    ax_b.axhline(0, color=COLORS["mid"], ls="--", lw=1.0)
    _panel(ax_b, "B")
    ax_c = fig.add_subplot(grid[0, 4:])
    phase_stats = _bootstrap_phase_mean(stability, "phase_boundary_top_match")
    x = np.arange(3)
    ax_c.errorbar(x, phase_stats["mean"], yerr=[phase_stats["mean"] - phase_stats.ci_lower, phase_stats.ci_upper - phase_stats["mean"]], fmt="none", ecolor=COLORS["mid"], capsize=4)
    ax_c.scatter(x, phase_stats["mean"], c=PHASE_COLORS, s=42, zorder=4)
    ax_c.set_xticks(x, [item.title() for item in PHASES])
    ax_c.set_ylabel("Highest-phase agreement")
    ax_c.set_ylim(0, 1.02)
    for idx, row in enumerate(phase_stats.itertuples(index=False)):
        ax_c.text(idx, row.mean + 0.08, f"$n$={row.n}", ha="center", fontsize=8)
    _panel(ax_c, "C")
    data_frame = data.frame.set_index("clip_uid")
    fig.text(0.015, 0.435, "(D)  Boundary-sensitive case", ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=COLORS["ink"])
    fig.text(0.515, 0.435, "(E)  Reference-sensitive case", ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=COLORS["ink"])
    with ZipFrameStore(Paths().trimmed_zip) as store:
        for panel_index, case_type in enumerate(("boundary_sensitive", "reference_sensitive")):
            sub = grid[1, panel_index * 3 : (panel_index + 1) * 3].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.08)
            case = cases.loc[case_type]
            meta = data_frame.loc[case.clip_uid]
            labels = data.phase_labels[int(case.query_index)]
            for phase_index, phase in enumerate(PHASES):
                ax = fig.add_subplot(sub[0, phase_index])
                _show_image(ax, _phase_tile(store, meta, labels, phase, width=90, height=84), PHASE_COLORS[phase_index])
                ax.set_title(phase.title(), color=PHASE_COLORS[phase_index], fontweight="bold", fontsize=8)
            ax = fig.add_subplot(sub[0, 3])
            subset = perturb[perturb.case_type == case_type]
            variants = subset.variant.drop_duplicates().tolist()
            positions = np.arange(3)
            offsets = np.linspace(-0.22, 0.22, len(variants))
            variant_colors = {
                "original": COLORS["ours"],
                "boundary_left": COLORS["takeoff"],
                "boundary_right": COLORS["flight"],
                "alternate_reference": COLORS["entry"],
            }
            for offset, variant in zip(offsets, variants):
                values = subset[subset.variant == variant].set_index("phase").loc[list(PHASES), "contribution"].to_numpy()
                ax.barh(
                    positions + offset,
                    values,
                    height=0.20,
                    label=variant.replace("boundary_", "boundary ").replace("alternate_reference", "alternate refs").replace("reference_replace", "alternate refs"),
                    color=variant_colors.get(variant, COLORS["mid"]),
                    alpha=1 if variant == "original" else 0.62,
                )
            ax.axvline(0, color=COLORS["ink"], lw=1.0)
            ax.set_yticks(positions, [item.title() for item in PHASES])
            ax.invert_yaxis()
            ax.set_xlabel("Contribution")
            ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=1, handlelength=1.0, labelspacing=0.16, borderaxespad=0.2)
    return _save(fig, "Figure5_stability_boundaries")


def render_figure1() -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    _configure()
    data = _load_data()
    cases = _case_rows().set_index("case_type")
    case = cases.loc["high_disagreement_gain"]
    frame = data.frame
    refs = [int(value) for value in data.reference_map["references"][int(case.query_index), :5] if value >= 0]
    fig, ax = plt.subplots(figsize=(180 * MM, 105 * MM))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    headings = [
        (0.02, "1  Query performance"),
        (0.265, "2  Latent-calibrated score"),
        (0.550, "3  Exact phase evidence"),
        (0.795, "4  Review-oriented record"),
    ]
    for x, heading in headings:
        ax.text(x, 0.955, heading, fontsize=8.0, fontweight="bold", va="top", color=COLORS["ink"])

    def rounded_box(x: float, y: float, width: float, height: float, edge: str, face: str) -> FancyBboxPatch:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.010,rounding_size=0.014",
            fc=face,
            ec=edge,
            lw=1.6,
        )
        ax.add_patch(patch)
        return patch

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = COLORS["mid"]) -> None:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.6, color=color))

    with ZipFrameStore(Paths().trimmed_zip) as store:
        q_meta = frame.iloc[int(case.query_index)]
        for phase_index, phase in enumerate(PHASES):
            image_ax = fig.add_axes([0.022 + phase_index * 0.070, 0.52, 0.064, 0.27])
            _show_image(image_ax, _phase_center(store, q_meta, data.phase_labels[int(case.query_index)], phase, size=(120, 150)), PHASE_COLORS[phase_index], 1.8)
            image_ax.set_title(phase.title(), color=PHASE_COLORS[phase_index], fontsize=8, fontweight="bold", pad=2)
        ax.text(0.118, 0.455, f"Official score  {case.official_score:.1f}", ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(0.118, 0.405, "Predicted phases define\nthe evidence units", ha="center", va="top", fontsize=8, color=COLORS["mid"])
        for slot, ref in enumerate(refs):
            ref_meta = frame.iloc[ref]
            ref_ax = fig.add_axes([0.282 + slot * 0.036, 0.20, 0.032, 0.15])
            _show_image(ref_ax, _phase_center(store, ref_meta, data.phase_labels[ref], "flight", size=(80, 120)), COLORS["light"], 0.9)

    rounded_box(0.285, 0.64, 0.190, 0.16, COLORS["teacher"], ImageColorLight(COLORS["teacher"]))
    ax.text(0.380, 0.744, "Frozen RICA²", ha="center", va="center", fontweight="bold", fontsize=8)
    ax.text(0.380, 0.683, f"global score  {case.teacher_score:.1f}", ha="center", va="center", fontsize=8)
    rounded_box(0.285, 0.415, 0.190, 0.16, COLORS["ours"], ImageColorLight(COLORS["ours"]))
    ax.text(0.380, 0.519, "Bounded latent calibrator", ha="center", va="center", fontweight="bold", fontsize=8)
    ax.text(0.380, 0.457, f"correction  →  {case.trustdive_score:.1f}", ha="center", va="center", fontsize=8)
    ax.text(0.380, 0.360, "Five same-action, cross-family\ntraining references", ha="center", va="top", fontsize=8, color=COLORS["mid"])

    rounded_box(0.540, 0.39, 0.205, 0.42, COLORS["flight"], ImageColorLight(COLORS["flight"]))
    ax.text(0.642, 0.758, "Eight Q/R coalitions", ha="center", va="center", fontweight="bold", fontsize=8)
    for mask in range(8):
        row, col = divmod(mask, 4)
        x0 = 0.558 + col * 0.044
        y0 = 0.625 - row * 0.105
        ax.text(x0 + 0.019, y0 + 0.055, f"{mask:03b}", ha="center", va="bottom", fontsize=8, color=COLORS["mid"])
        for phase_index, phase in enumerate(PHASES):
            is_query = bool(mask & (1 << phase_index))
            ax.add_patch(
                Rectangle(
                    (x0 + phase_index * 0.013, y0),
                    0.011,
                    0.045,
                    fc=PHASE_COLORS[phase_index] if is_query else "white",
                    ec=PHASE_COLORS[phase_index],
                    lw=0.8,
                )
            )
    ax.text(0.642, 0.430, "Exact three-phase Shapley\non the final scorer", ha="center", va="center", fontsize=8)

    rounded_box(0.790, 0.43, 0.185, 0.38, COLORS["entry"], ImageColorLight(COLORS["entry"]))
    ax.text(0.882, 0.756, f"Final score  {case.trustdive_score:.1f}", ha="center", va="center", fontweight="bold", fontsize=8.0)
    contribution_values = [case.phi_takeoff, case.phi_flight, case.phi_entry]
    for index, (phase, value, color) in enumerate(zip(PHASES, contribution_values, PHASE_COLORS)):
        y0 = 0.675 - index * 0.072
        ax.text(0.807, y0, phase.title(), ha="left", va="center", fontsize=8, color=color, fontweight="bold")
        ax.add_patch(Rectangle((0.866, y0 - 0.012), min(abs(value), 0.48) * 0.13, 0.024, fc=color, ec="none", alpha=0.9))
        ax.text(0.954, y0, f"{value:+.2f}", ha="right", va="center", fontsize=8)
    ax.text(0.882, 0.468, "Reference-supported", ha="center", va="center", fontsize=8, color=COLORS["mid"])

    arrow((0.225, 0.705), (0.282, 0.705))
    arrow((0.380, 0.635), (0.380, 0.580), COLORS["ours"])
    arrow((0.478, 0.495), (0.535, 0.565), COLORS["ours"])
    arrow((0.748, 0.600), (0.787, 0.600), COLORS["flight"])
    ax.plot([0.285, 0.475], [0.095, 0.095], color=COLORS["ours"], lw=3)
    ax.text(0.380, 0.070, "score calibration", ha="center", va="top", color=COLORS["ours"], fontsize=8, fontweight="bold")
    ax.plot([0.540, 0.975], [0.095, 0.095], color=COLORS["flight"], lw=3)
    ax.text(0.758, 0.070, "counterfactual evidence path", ha="center", va="top", color=COLORS["flight"], fontsize=8, fontweight="bold")
    return _save(fig, "Figure1_method_overview")


def ImageColorLight(color: str) -> str:
    rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(round(channel * 0.18 + 255 * 0.82) for channel in rgb)
    return "#" + "".join(f"{value:02X}" for value in mixed)


def render_supplement() -> dict:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import to_rgba

    _configure()
    outputs: dict[str, dict] = {}
    overview = pd.read_csv(SOURCE_ROOT / "FigureS1_dataset_overview.csv")
    fig, axes = plt.subplots(3, 1, figsize=(180 * MM, 155 * MM), layout="constrained")
    order = sorted(overview.action_group.unique())
    counts = overview.action_group.value_counts().reindex(order)
    axes[0].bar(np.arange(len(order)), counts, color=[PHASE_COLORS[index % 3] for index in range(len(order))], alpha=0.75)
    axes[0].set_xticks(np.arange(len(order)), [f"Group {item}" for item in order])
    axes[0].set_ylabel("Video count")
    _panel(axes[0], "A", x=-0.07)
    sns.boxplot(data=overview, x="action_group", y="execution_quality", hue="action_group", palette="pastel", legend=False, ax=axes[1])
    axes[1].set(xlabel="Action group", ylabel="Execution-quality score")
    _panel(axes[1], "B", x=-0.07)
    judge = overview.groupby("judge_count").size().reindex([3, 5, 7], fill_value=0)
    axes[2].bar(judge.index.astype(str), judge.values, color=[COLORS["takeoff"], COLORS["flight"], COLORS["entry"]], alpha=0.78)
    axes[2].set(xlabel="Available judge scores", ylabel="Video count")
    _panel(axes[2], "C", x=-0.07)
    outputs["FigureS1"] = _save(fig, "FigureS1_dataset_overview")
    plt.close(fig)
    ablation = pd.read_csv(SOURCE_ROOT / "FigureS2_ablation.csv")
    model_labels = {
        "frozen_teacher": "Frozen RICA²",
        "score_only_linear": "Score-only linear",
        "latent_only_ridge": "Latent-only Ridge",
        "reference_only_ridge": "Reference-only Ridge",
        "full_latent_reference_ridge": "Full latent-reference Ridge",
        "prespecified_trustdive": "TrustDive (prespecified)",
    }
    ablation["display_model"] = ablation.model.map(model_labels).fillna(ablation.model)
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 82 * MM), layout="constrained")
    y = np.arange(len(ablation))
    alphas = np.linspace(0.45, 1, len(ablation))
    ablation_colors = [to_rgba(COLORS["teacher"] if index == 0 else COLORS["ours"], alpha=float(alphas[index])) for index in range(len(ablation))]
    axes[0].hlines(y, 0.826, ablation.spearman, color=ablation_colors, lw=1.6)
    axes[0].scatter(ablation.spearman, y, color=ablation_colors, s=34, zorder=3)
    axes[0].axvline(float(ablation.iloc[0].spearman), color=COLORS["teacher"], ls="--", lw=1.0)
    axes[0].set_yticks(y, ablation.display_model)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Spearman ρ")
    axes[0].set_xlim(0.826, 0.837)
    for row in ablation.itertuples():
        axes[0].text(float(row.spearman) + 0.00015, list(ablation.model).index(row.model), f"{row.spearman:.4f}", va="center", fontsize=8)
    _panel(axes[0], "A")
    axes[1].hlines(y, 5.64, ablation.mae, color=ablation_colors, lw=1.6)
    axes[1].scatter(ablation.mae, y, color=ablation_colors, s=34, zorder=3)
    axes[1].axvline(float(ablation.iloc[0].mae), color=COLORS["teacher"], ls="--", lw=1.0)
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MAE (score points)")
    axes[1].set_xlim(5.64, 6.22)
    for row in ablation.itertuples():
        axes[1].text(float(row.mae) + 0.008, list(ablation.model).index(row.model), f"{row.mae:.3f}", va="center", fontsize=8)
    _panel(axes[1], "B")
    outputs["FigureS2"] = _save(fig, "FigureS2_scoring_ablation")
    plt.close(fig)
    refs = pd.read_csv(SOURCE_ROOT / "FigureS3_reference_coverage.csv")
    test_refs = refs[refs.analysis_role == "official_test"]
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 80 * MM), layout="constrained")
    ref_counts = test_refs.valid_reference_count.value_counts().sort_index()
    axes[0].bar(ref_counts.index.astype(str), ref_counts.values, color=COLORS["ours"])
    axes[0].set(xlabel="Legal same-action references", ylabel="Official-test videos")
    _panel(axes[0], "A")
    axes[1].scatter(test_refs.reference_distance, test_refs.reference_dispersion, c=test_refs.open_set.map({False: COLORS["ours"], True: COLORS["failure"]}), s=12, alpha=0.45)
    axes[1].set(xlabel="Mean reference distance", ylabel="Reference-score dispersion")
    axes[1].set_title("14 reference-sparse videos fall back to RICA²", loc="left")
    _panel(axes[1], "B")
    outputs["FigureS3"] = _save(fig, "FigureS3_reference_coverage")
    plt.close(fig)
    review = pd.read_csv(SOURCE_ROOT / "FigureS4_selective_review.csv")
    fig, ax = plt.subplots(figsize=(120 * MM, 82 * MM), layout="constrained")
    for strategy, subset in review.groupby("strategy"):
        ax.plot(subset.coverage, subset.accepted_mae, label=strategy, color=COLORS["teacher"] if "RICA2" in strategy else COLORS["ours"])
    ax.axvline(0.8, color=COLORS["failure"], ls="--", lw=1.2)
    ax.set(xlabel="Automatic coverage", ylabel="Accepted-video MAE")
    ax.legend()
    _panel(ax, "A", x=-0.09)
    outputs["FigureS4"] = _save(fig, "FigureS4_exploratory_review")
    plt.close(fig)
    outputs["FigureS5"] = render_figure3_variant_supplement()
    return outputs


def render_figure3_variant_supplement() -> dict:
    import matplotlib.pyplot as plt

    data = _load_data()
    cases = _case_rows()
    fig = plt.figure(figsize=(180 * MM, 150 * MM), layout="constrained")
    grid = fig.add_gridspec(len(cases), 5, width_ratios=[0.95, 1, 1, 1, 1.25], hspace=0.15, wspace=0.08)
    with ZipFrameStore(Paths().trimmed_zip) as store:
        for row_index, case in enumerate(cases.itertuples(index=False)):
            meta = data.frame.iloc[int(case.query_index)]
            ax_text = fig.add_subplot(grid[row_index, 0])
            ax_text.axis("off")
            ax_text.text(0, 0.85, case.case_label, fontweight="bold", va="top", wrap=True)
            ax_text.text(0, 0.35, f"Official {case.official_score:.1f}\nPredicted {case.trustdive_score:.1f}", va="top")
            for phase_index, phase in enumerate(PHASES):
                ax = fig.add_subplot(grid[row_index, 1 + phase_index])
                _show_image(ax, _phase_tile(store, meta, data.phase_labels[int(case.query_index)], phase), PHASE_COLORS[phase_index])
            ax = fig.add_subplot(grid[row_index, 4])
            values = [case.phi_takeoff, case.phi_flight, case.phi_entry]
            ax.barh(range(3), values, color=PHASE_COLORS)
            ax.axvline(0, color=COLORS["ink"], lw=1)
            ax.set_yticks(range(3), [item.title() for item in PHASES])
            ax.invert_yaxis()
            ax.set_xlabel("Contribution")
    return _save(fig, "FigureS5_additional_cases")


CAPTIONS = {
    "Figure1": "Overview of TrustDive. A deterministic RICA² scoring foundation undergoes bounded latent calibration, while five same-action references from different event families define a fixed comparison neighborhood. The final deployed scorer is evaluated on eight feature-level query/reference phase coalitions to obtain exact model-attributed contributions for takeoff, flight, and entry. Example frames visualize the returned evidence; quantitative results are reported in subsequent figures.",
    "Figure2": "Latent-calibrated score assessment. (A,B) Official and predicted total scores for RICA² and the prespecified TrustDive model on all 749 official-test videos; dashed lines indicate identity. (C) Video-level reduction in absolute error, with all videos and the high-disagreement subset shown on the same scale; positive values indicate lower TrustDive error. Points denote videos and horizontal lines denote medians. (D) Mean MAE reduction with 95% event-family-clustered bootstrap intervals. High disagreement was defined by the frozen fit-set threshold and does not denote abnormal judging.",
    "Figure3": "Mechanically selected real-video examples. The two full-width case cards show a typical accurate performance and a high-disagreement performance for which the calibrated scorer reduced score error. Enlarged query images show the midpoint of each predicted takeoff, flight, and entry phase after one deterministic sequence-level crop; the smaller lower row shows the nearest displayed legal training reference. Right panels report exact model-attributed phase contributions in execution-quality points. Crop centers were derived from movement across the frozen 25%, 50%, and 75% phase frames, without pose inference or local retouching. Cases were selected by prespecified performance strata rather than visual appeal.",
    "Figure4": "Counterfactual fidelity of phase evidence. (A) All eight feature-level coalitions for one mechanically selected high-disagreement case; Q and R indicate phases from the query and reference. These tiles illustrate feature provenance and are not generated videos. (B) Exact Shapley reconstruction of the final execution-quality prediction. (C) Absolute score effect of the highest-contribution phase compared with the mean and strongest of the two nonselected phases across 735 reference-supported videos. (D) Agreement between the highest attributed phase and the phase with the largest direct intervention effect.",
    "Figure5": "Distribution and stability boundaries of phase evidence. (A) Highest-contribution phase across 735 reference-supported test videos. (B) Contribution-vector cosine similarity after one-token boundary shifts and alternate-reference replacement. (C) Highest-phase agreement under boundary shifts, stratified by the original highest-contribution phase; points and 95% clustered bootstrap intervals are shown. (D,E) Mechanically selected boundary-sensitive and reference-sensitive cases with deterministic perturbation results. These analyses characterize model-evidence stability, not human judging processes.",
    "FigureS1": "FineDiving dataset overview. (A) Number of videos in each action group. (B) Distribution of execution-quality scores by action group. (C) Number of videos with three, five, or seven available judge scores.",
    "FigureS2": "Component scoring ablation. (A) Spearman correlation and (B) mean absolute error for the deterministic RICA² foundation, score-only calibration, latent-only Ridge, reference-only Ridge, full latent-reference Ridge, and the prespecified risk-weighted TrustDive model. Latent calibration explains most of the error reduction; reference statistics add only a small incremental scoring effect, and risk weighting does not improve the full ordinary Ridge model.",
    "FigureS3": "Reference coverage in the official test set. (A) Number of legal same-action, cross-family training references. (B) Reference distance and score dispersion; 14 reference-sparse videos had fewer than three legal references and therefore fell back to deterministic RICA² without reference-conditioned phase evidence.",
    "FigureS4": "Exploratory selective-review analysis. Coverage–risk curves compare deterministic RICA² uncertainty with the combined review priority. The vertical line marks 80% automatic coverage. The combined strategy did not establish a confirmatory review benefit and is reported as exploratory.",
    "FigureS5": "Additional mechanically selected success, failure, and evidence-sensitivity examples. Each phase tile contains frames at 25%, 50%, and 75% of the predicted phase; bars show model-attributed phase contributions.",
}

ALT_TEXT = {
    "Figure1": "Workflow diagram with real takeoff, flight, and entry frames, five reference thumbnails, a frozen RICA² scoring block, a reference adapter, eight phase coalitions, and a phase-evidence output card.",
    "Figure2": "A two-by-two grid shows matched score scatterplots above a full-size video-level error-reduction distribution and effect-size forest plot. Positive reductions indicate lower TrustDive error, with the largest mean reduction in high-disagreement videos.",
    "Figure3": "Two full-width case cards show enlarged real query takeoff, flight, and entry frames, a smaller matched training-reference strip, score estimates, and phase-contribution bars for a typical accurate case and a high-disagreement improvement case.",
    "Figure4": "Eight query-reference phase combinations are paired with a Shapley waterfall, intervention-effect distributions, and a phase-matching confusion matrix.",
    "Figure5": "Phase-frequency bars, stability distributions, boundary-agreement estimates, and two real-video failure-boundary cases show that phase evidence is useful but not invariant.",
    "FigureS1": "Three panels summarize FineDiving action-group counts, execution-quality distributions, and the numbers of videos with three, five, or seven judge scores.",
    "FigureS2": "Horizontal bars compare Spearman correlation and mean absolute error across six component scoring baselines and ablations.",
    "FigureS3": "A reference-count bar chart and distance-versus-dispersion scatterplot identify 14 reference-sparse official-test videos.",
    "FigureS4": "Two coverage–risk curves show an exploratory and statistically limited selective-review comparison at 80 percent automatic coverage.",
    "FigureS5": "Five rows of real takeoff, flight, and entry frame strips accompany phase-contribution bars for mechanically selected success, failure, and sensitivity cases.",
}


def _write_captions() -> None:
    write_json(CAPTION_ROOT / "captions_and_alt_text.json", {"captions": CAPTIONS, "alt_text": ALT_TEXT})
    lines = ["% Auto-generated figure captions. Verify wording before submission."]
    for key in ("Figure1", "Figure2", "Figure3", "Figure4", "Figure5", "FigureS1", "FigureS2", "FigureS3", "FigureS4", "FigureS5"):
        escaped = CAPTIONS[key].replace("%", "\\%")
        lines.append(f"% {key}\n{escaped}\n")
    (CAPTION_ROOT / "figure_captions.txt").write_text("\n".join(lines), encoding="utf-8")


def render(figures: str) -> dict:
    _ensure_dirs()
    if not (OUTPUT_ROOT / "source_data_manifest.json").exists():
        build_source_data()
    requested = [item.strip().lower() for item in figures.split(",") if item.strip()]
    functions = {"1": render_figure1, "2": render_figure2, "3": render_figure3, "4": render_figure4, "5": render_figure5}
    outputs: dict[str, dict] = {}
    for item in requested:
        if item == "supplement":
            outputs.update(render_supplement())
        elif item in functions:
            outputs[f"Figure{item}"] = functions[item]()
        else:
            raise ValueError(f"Unknown figure selector: {item}")
    _write_captions()
    manifest_path = OUTPUT_ROOT / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(outputs)
    manifest["input_hashes"] = {str(path.relative_to(PROJECT_ROOT)): expected for path, expected in EXPECTED_HASHES.items()}
    manifest["case_selection_sha256"] = sha256_file(SOURCE_ROOT / "case_selection_audit.csv")
    manifest["frame_selection_sha256"] = sha256_file(SOURCE_ROOT / "frame_selection_audit.csv")
    manifest["case_selection"] = pd.read_csv(SOURCE_ROOT / "case_selection_audit.csv").to_dict(orient="records")
    manifest["frame_selection"] = pd.read_csv(SOURCE_ROOT / "frame_selection_audit.csv").to_dict(orient="records")
    manifest["backend"] = "Python/matplotlib exclusively"
    write_json(manifest_path, manifest)
    return outputs


def qa() -> dict:
    _ensure_dirs()
    expected = [
        "Figure1_method_overview",
        "Figure2_scoring_performance",
        "Figure3_phase_case_matrix",
        "Figure4_counterfactual_fidelity",
        "Figure5_stability_boundaries",
        "FigureS1_dataset_overview",
        "FigureS2_scoring_ablation",
        "FigureS3_reference_coverage",
        "FigureS4_exploratory_review",
        "FigureS5_additional_cases",
    ]
    expected_main_pixels = {
        "Figure1_method_overview": (4251, 2480),
        "Figure2_scoring_performance": (4251, 3024),
        "Figure3_phase_case_matrix": (4251, 3307),
        "Figure4_counterfactual_fidelity": (4251, 3543),
        "Figure5_stability_boundaries": (4251, 3188),
    }
    grayscale_root = OUTPUT_ROOT / "qa_grayscale"
    grayscale_root.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict[str, bool]] = {}
    for stem in expected:
        bundle: dict[str, bool] = {}
        for suffix in ("svg", "pdf", "tiff", "png"):
            path = OUTPUT_ROOT / f"{stem}.{suffix}"
            bundle[f"{suffix}_exists"] = path.exists() and path.stat().st_size > 1000
        png = OUTPUT_ROOT / f"{stem}.png"
        tiff = OUTPUT_ROOT / f"{stem}.tiff"
        svg = OUTPUT_ROOT / f"{stem}.svg"
        if png.exists():
            with Image.open(png) as image:
                gray = np.asarray(image.convert("L"), dtype=float)
                bundle["png_readable"] = image.width >= 1400 and image.height >= 600 and float(gray.std()) > 8
                Image.fromarray(gray.astype(np.uint8)).save(grayscale_root / f"{stem}_grayscale.png")
        if tiff.exists():
            with Image.open(tiff) as image:
                dpi = image.info.get("dpi", (0, 0))[0]
                bundle["tiff_600dpi"] = dpi >= 590 and image.width >= 2800
                if stem in expected_main_pixels:
                    expected_width, expected_height = expected_main_pixels[stem]
                    bundle["final_dimensions_match"] = abs(image.width - expected_width) <= 2 and abs(image.height - expected_height) <= 2
        if svg.exists():
            text = svg.read_text(encoding="utf-8", errors="ignore")
            bundle["svg_editable_text"] = "<text" in text
        checks[stem] = bundle
    source_files = list(SOURCE_ROOT.glob("*.csv"))
    result = {
        "status": "PASS" if all(all(group.values()) for group in checks.values()) and len(source_files) >= 10 else "FAIL",
        "checks": checks,
        "source_data_count": len(source_files),
        "source_data_hashes": {path.name: sha256_file(path) for path in source_files},
        "case_selection_deterministic_sha256": sha256_file(SOURCE_ROOT / "case_selection_audit.csv") if (SOURCE_ROOT / "case_selection_audit.csv").exists() else None,
        "backend_exclusive": "Python",
    }
    write_json(OUTPUT_ROOT / "figure_qa.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build publication figures for the TrustDive manuscript")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build-source-data")
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--figures", required=True, help="Comma-separated 1-5 or supplement")
    subparsers.add_parser("qa")
    args = parser.parse_args(argv)
    if args.command == "build-source-data":
        result = build_source_data()
    elif args.command == "render":
        result = render(args.figures)
    else:
        result = qa()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
