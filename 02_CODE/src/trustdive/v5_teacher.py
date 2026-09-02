from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .metrics import aqa_score_metrics
from .util import sha256_file, write_json
from .v4_teacher import (
    V4_CACHE_ROOT,
    _add_rica2_path,
    _checkpoint_folder,
    _load_rica2_model,
    _make_feature_dataset,
    _require_teacher_environment,
    rica2_raw_output_to_execution_quality,
)
from .v5_data import (
    V5_RESULTS_ROOT,
    load_v5_contract,
    load_v5_frame,
    require_v5_audit,
)


def _v5_execution_quality(raw_output):
    """Convert the official RICA2 output to Q = S / (3 * DD).

    The frozen v4 model was trained with ``three_judge_score_scaling: false``;
    consequently its raw output is S/DD.  The official v4 evaluation restores
    S and subsequently divides by ``3 * DD`` when it creates the execution
    quality anchor.  Counterfactual inference must therefore divide the raw
    output by three as well.
    """
    return rica2_raw_output_to_execution_quality(raw_output) / 3.0


def _make_rica2_inference_deterministic(model) -> dict:
    """Disable an upstream eval-time Dropout construction bug, without editing it.

    RICA2's ConvEncoder creates ``nn.Dropout(self.conv_dropout)`` inside
    ``forward``.  Such a newly-created module remains in training mode even
    after ``model.eval()``, making nominally deterministic inference random.
    Setting the stored probability to zero is the smallest reversible runtime
    correction and is essential before computing counterfactual Shapley values.
    """
    backbone = model.module.backbone
    original = float(backbone.conv_dropout)
    backbone.conv_dropout = 0.0
    return {
        "upstream_eval_dropout_probability": original,
        "v5_inference_dropout_probability": float(backbone.conv_dropout),
        "external_source_modified": False,
    }


def export_teacher_assets_v5(overwrite: bool = False) -> dict:
    """Export the frozen v4 RICA2 anchor for v5 without changing v4 artifacts."""
    require_v5_audit()
    _require_teacher_environment()
    output_dir = V5_RESULTS_ROOT / "01_REFERENCES"
    summary_path = output_dir / "teacher_export_v5.json"
    required = (
        output_dir / "teacher_counterfactual_v5.pt",
        output_dir / "teacher_sequences_v5.npz",
        output_dir / "teacher_predictions_v5.parquet",
    )
    if not overwrite and summary_path.exists() and all(path.exists() for path in required):
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            return existing

    _add_rica2_path()
    import torch
    from libs.utils.preprocessing import multi_label_multi_class_one_hot_encode

    config_path = V4_CACHE_ROOT / "configs" / "rica2_test_v4.yaml"
    checkpoint_path = _checkpoint_folder("final") / "last.pth.tar"
    model, cfg = _load_rica2_model(config_path, checkpoint_path)
    deterministic_patch = _make_rica2_inference_deterministic(model)

    class CounterfactualWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, sequence, action_presence):
            features = sequence.permute(0, 2, 1)
            masks = torch.ones(
                (features.shape[0], 1, features.shape[-1]),
                dtype=torch.bool,
                device=features.device,
            )
            output, _ = self.wrapped(features, masks, action_presence)
            return _v5_execution_quality(output["all_sample_outputs"])

    dataset = _make_feature_dataset("all_split.pkl")
    first_item = dataset[0]
    first_action = multi_label_multi_class_one_hot_encode(
        first_item["actions_present"], int(cfg["model"]["num_phases"])
    )
    wrapper = CounterfactualWrapper(model.module).cuda().eval()
    dummy_sequence = torch.stack(
        [first_item["feats"].transpose(0, 1), first_item["feats"].transpose(0, 1)]
    ).to(dtype=torch.float32, device="cuda")
    dummy_actions = torch.stack([first_action, first_action]).to(
        dtype=torch.float32, device="cuda"
    )
    with torch.inference_mode():
        direct_first = wrapper(dummy_sequence, dummy_actions)
        direct_second = wrapper(dummy_sequence, dummy_actions)
    repeat_difference = float(
        torch.max(torch.abs(direct_first - direct_second)).detach().cpu()
    )
    traced = torch.jit.trace(
        wrapper, (dummy_sequence, dummy_actions), strict=False, check_trace=True
    )
    with torch.inference_mode():
        trace_difference = float(
            torch.max(torch.abs(traced(dummy_sequence, dummy_actions) - direct_first))
            .detach()
            .cpu()
        )
    trace_path = output_dir / "teacher_counterfactual_v5.pt"
    traced.save(str(trace_path))

    frame = load_v5_frame().set_index("clip_uid")
    sequences: list[np.ndarray] = []
    predictions: list[float] = []
    actions: list[np.ndarray] = []
    clip_uids: list[str] = []
    batch_size = 128
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            items = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            feats = torch.stack([item["feats"] for item in items]).cuda()
            masks = torch.ones(
                (len(items), 1, feats.shape[-1]), dtype=torch.bool, device="cuda"
            )
            gt_actions = torch.stack(
                [
                    multi_label_multi_class_one_hot_encode(
                        item["actions_present"], int(cfg["model"]["num_phases"])
                    )
                    for item in items
                ]
            ).cuda()
            output, _ = model(
                feats, masks, gt_actions, [item["video_id"] for item in items]
            )
            quality = (
                _v5_execution_quality(output["all_sample_outputs"])
                .detach()
                .cpu()
                .numpy()
            )
            for item, value, action_vector in zip(
                items, quality, gt_actions.detach().cpu().numpy()
            ):
                source, instance = item["video_id"]
                clip_uid = f"{source}::{int(instance)}"
                clip_uids.append(clip_uid)
                sequences.append(item["feats"].transpose(0, 1).numpy().astype(np.float16))
                predictions.append(float(value))
                actions.append(action_vector.astype(np.float32))
            print(f"V5 teacher export: {min(start + batch_size, len(dataset))}/{len(dataset)}", flush=True)

    sequence_path = output_dir / "teacher_sequences_v5.npz"
    np.savez_compressed(
        sequence_path,
        clip_uid=np.asarray(clip_uids),
        sequence=np.stack(sequences),
        action_presence=np.stack(actions),
    )
    output = frame.loc[clip_uids].reset_index()[
        [
            "clip_uid",
            "feature_key",
            "source",
            "instance",
            "official_split",
            "analysis_role",
            "source_role",
            "event_family",
            "action_type",
            "difficulty",
            "dive_score",
            "execution_quality",
        ]
    ].copy()
    output["teacher_predicted_quality"] = np.asarray(predictions, dtype=np.float32)
    output["teacher_predicted_score"] = (
        3.0 * output.difficulty * output.teacher_predicted_quality
    )
    output["teacher_uncertainty"] = 0.0
    prediction_path = output_dir / "teacher_predictions_v5.parquet"
    output.to_parquet(prediction_path, index=False)

    old_test = pd.read_parquet(
        V5_RESULTS_ROOT.parent
        / "V4_COUNTERFACTUAL"
        / "01_TEACHER"
        / "teacher_test_predictions_v4.parquet"
    )
    old_column = next(
        column
        for column in ("teacher_predicted_quality", "predicted_quality")
        if column in old_test.columns
    )
    current_test = output.loc[
        output.official_split == "test", ["clip_uid", "teacher_predicted_quality"]
    ].rename(columns={"teacher_predicted_quality": "v5_quality"})
    matched = current_test.merge(
        old_test[["clip_uid", old_column]].rename(columns={old_column: "v4_quality"}),
        on="clip_uid",
        how="inner",
    )
    difference = np.abs(
        matched["v5_quality"].to_numpy() - matched["v4_quality"].to_numpy()
    )
    test_mask = output.official_split == "test"
    deterministic_metrics = aqa_score_metrics(
        output.loc[test_mask, "dive_score"],
        output.loc[test_mask, "teacher_predicted_score"],
    )
    historical_spearman = float(load_v5_contract()["teacher"]["measured_test_spearman"])
    rank_agreement = float(
        matched["v5_quality"].corr(matched["v4_quality"], method="spearman")
    )
    # Exact value equality with the historical export is neither expected nor
    # desirable: the historical upstream path used stochastic eval-time
    # dropout.  We instead require deterministic repeatability and preserved
    # score-ranking behavior within the predeclared 0.03 protocol tolerance.
    ranking_preserved = (
        deterministic_metrics["spearman"] >= historical_spearman - 0.03
        and rank_agreement >= 0.95
    )
    checks = {
        "rows": len(output) == 3000,
        "old_test_rows_matched": len(matched) == 749,
        "repeat_inference_exact": repeat_difference <= 1e-7,
        "torchscript_matches_direct": trace_difference <= 1e-5,
        "historical_ranking_preserved": bool(ranking_preserved),
        "finite_predictions": bool(
            np.isfinite(output.teacher_predicted_quality.to_numpy()).all()
        ),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rows": len(output),
        "sequence_shape": list(np.stack(sequences).shape),
        "old_test_rows_matched": len(matched),
        "deterministic_inference_patch": deterministic_patch,
        "execution_quality_scale_divisor": 3.0,
        "repeat_inference_maximum_difference": repeat_difference,
        "torchscript_direct_maximum_difference": trace_difference,
        "deterministic_official_test_metrics": deterministic_metrics,
        "historical_official_test_spearman": historical_spearman,
        "deterministic_to_historical_rank_spearman": rank_agreement,
        "historical_prediction_comparison_note": (
            "Values need not be identical because v4 retained the upstream "
            "eval-time Dropout behavior; v5 disables it before Shapley inference."
        ),
        "mean_old_test_prediction_difference": float(difference.mean()),
        "maximum_old_test_prediction_difference": float(difference.max()),
        "prediction_sha256": sha256_file(prediction_path),
        "sequence_sha256": sha256_file(sequence_path),
        "counterfactual_model_sha256": sha256_file(trace_path),
    }
    write_json(summary_path, result)
    return result


if __name__ == "__main__":
    print(json.dumps(export_teacher_assets_v5(), indent=2, ensure_ascii=False))
