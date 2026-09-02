from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import PROJECT_ROOT
from .metrics import aqa_score_metrics
from .util import sha256_file, write_json
from .v4_data import (
    RICA2_DATA_ROOT,
    RICA2_FEATURE_ROOT,
    RICA2_ROOT,
    V4_CACHE_ROOT,
    V4_RESULTS_ROOT,
    V4_RUN_ROOT,
    ensure_v4_dirs,
    load_v4_contract,
    load_v4_frame,
    require_v4_audit,
)


PHASES = ("takeoff", "flight", "entry")


def _require_teacher_environment() -> None:
    if Path(sys.prefix).name.lower() != "trustdive_rica2":
        raise RuntimeError(
            "RICA2 teacher commands must run in the isolated trustdive_rica2 environment"
        )


def _add_rica2_path() -> None:
    value = str(RICA2_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)


def _video_ids(frame: pd.DataFrame) -> list[tuple[str, int]]:
    return [(str(row.source), int(row.instance)) for row in frame.itertuples(index=False)]


def _dump_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def i3d_output_to_cache(features: np.ndarray) -> np.ndarray:
    """Convert RICA2 I3D output (B, C, T) to its on-disk (B, T, C) format."""
    value = np.asarray(features)
    if value.ndim != 3:
        raise ValueError("RICA2 I3D output must have shape (batch, channels, time)")
    return value.transpose(0, 2, 1).astype(np.float16)


def rica2_raw_output_to_execution_quality(raw_output):
    """Collapse deterministic/stochastic samples to Q = S / (3 * DD).

    FineDiving's RICA2 dataset divides the total score by difficulty and by
    the three-judge scaling factor before training. Its postprocessor later
    multiplies the raw network output by 3 and DD only to restore the official
    total score. Counterfactual scoring operates before that restoration.
    """
    return raw_output.mean(1).reshape(-1)


def prepare_teacher_assets_v4() -> dict:
    require_v4_audit()
    _require_teacher_environment()
    ensure_v4_dirs()
    frame = load_v4_frame()
    license_path = RICA2_ROOT / "LICENSE"
    source_root = Path(
        os.environ.get(
            "TRUSTDIVE_FINE_DIVING",
            str(PROJECT_ROOT / "data" / "FineDiving"),
        )
    )
    annotation_root = RICA2_DATA_ROOT / "Annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (source_root / "Annotations" / "FineDiving_fine-grained_annotation.pkl", annotation_root / "FineDiving_fine-grained_annotation.pkl"),
        (source_root / "Annotations" / "FineDiving_coarse_annotation.pkl", annotation_root / "FineDiving_coarse_annotation.pkl"),
        (source_root / "train_test_split" / "train_split.pkl", annotation_root / "train_split.pkl"),
        (source_root / "train_test_split" / "test_split.pkl", annotation_root / "test_split.pkl"),
    ):
        if not target.exists() or sha256_file(source) != sha256_file(target):
            shutil.copy2(source, target)

    split_specs = {
        "fit_split.pkl": frame.analysis_role == "fit",
        "validation_split.pkl": frame.analysis_role == "validation",
        "calibration_split.pkl": frame.analysis_role == "calibration",
        "fit_validation_split.pkl": frame.analysis_role.isin(("fit", "validation")),
        "all_split.pkl": np.ones(len(frame), dtype=bool),
        "source_fit_split.pkl": frame.source_role == "source_fit",
        "source_validation_split.pkl": frame.source_role == "source_validation",
        "source_test_split.pkl": frame.source_role == "source_test",
    }
    split_counts = {}
    for name, mask in split_specs.items():
        ids = _video_ids(frame.loc[np.asarray(mask, dtype=bool)])
        _dump_pickle(annotation_root / name, ids)
        split_counts[name] = len(ids)

    official_patch_checks = {
        "windows_path_patch": "os.path.basename(image_list[0])" in (RICA2_ROOT / "libs" / "datasets" / "finediving.py").read_text(encoding="utf-8"),
        "last_checkpoint_patch": "file_name='last.pth.tar'" in (RICA2_ROOT / "train.py").read_text(encoding="utf-8"),
    }
    checks = {
        "environment": Path(sys.prefix).name.lower() == "trustdive_rica2",
        "external_commit": subprocess.check_output(["git", "-C", str(RICA2_ROOT), "rev-parse", "HEAD"], text=True).strip() == load_v4_contract()["external_teacher"]["commit"],
        "weights": sha256_file(RICA2_ROOT / "pre_trained" / "model_rgb.pth") == load_v4_contract()["external_teacher"]["weights_sha256"],
        "split_partition": split_counts["fit_split.pkl"] + split_counts["validation_split.pkl"] + split_counts["calibration_split.pkl"] + 749 == len(frame),
        "patches": all(official_patch_checks.values()),
        "license_present": license_path.exists(),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "patch_checks": official_patch_checks,
        "split_counts": split_counts,
        "data_root": str(RICA2_DATA_ROOT),
        "feature_root": str(RICA2_FEATURE_ROOT),
        "external_commit": load_v4_contract()["external_teacher"]["commit"],
        "weights_sha256": sha256_file(RICA2_ROOT / "pre_trained" / "model_rgb.pth"),
        "license": {
            "path": str(license_path),
            "sha256": sha256_file(license_path) if license_path.exists() else None,
            "identifier": "MIT" if license_path.exists() and "MIT License" in license_path.read_text(encoding="utf-8") else "UNVERIFIED",
        },
    }
    write_json(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_preparation_v4.json", result)
    return result


def _make_dataset(split_file: str):
    _add_rica2_path()
    from libs.datasets.finediving import FineDiving

    return FineDiving(
        False,
        "test",
        str(RICA2_DATA_ROOT),
        "FINADiving_MTL_256s",
        "FineDiving_fine-grained_annotation.pkl",
        "FineDiving_coarse_annotation.pkl",
        "fit_split.pkl",
        split_file,
        10,
        16,
        96,
        9,
        [112, 200],
        112,
        True,
        False,
        False,
        "",
    )


def extract_teacher_features_v4(limit: int | None = None) -> dict:
    _require_teacher_environment()
    preparation = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_preparation_v4.json"
    if not preparation.exists() or json.loads(preparation.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Run prepare-teacher --protocol v4 first")
    _add_rica2_path()
    import torch
    from libs.modeling.backbones import I3D_feat_extractor

    dataset = _make_dataset("all_split.pkl")
    maximum = min(len(dataset), int(limit)) if limit is not None else len(dataset)
    with (RICA2_DATA_ROOT / "Annotations" / "all_split.pkl").open("rb") as handle:
        all_video_ids = pickle.load(handle)
    if len(all_video_ids) != len(dataset):
        raise AssertionError("RICA2 all-split IDs do not match the feature dataset")
    model = I3D_feat_extractor(str(RICA2_ROOT / "pre_trained" / "model_rgb.pth"), False).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    completed = 0
    batch_size = 8
    for start in range(0, maximum, batch_size):
        stop = min(start + batch_size, maximum)
        missing_indices = []
        for index in range(start, stop):
            source, instance = all_video_ids[index]
            destination = RICA2_FEATURE_ROOT / f"{source}-abr-{int(instance)}.npz"
            if not destination.exists():
                missing_indices.append(index)
        missing = [dataset[index] for index in missing_indices]
        if missing:
            frames = torch.stack([item["window_frames"] for item in missing]).cuda(non_blocking=True)
            with torch.inference_mode():
                features = model(frames, [item["video_id"] for item in missing])
            # RICA2's cached-feature loader transposes each stored array from
            # (T, C) to the model-facing (C, T). The official I3D extractor
            # returns (B, C, T), so persist (T, C) here.
            features = i3d_output_to_cache(features.detach().cpu().numpy())
            for item, feature in zip(missing, features):
                source, instance = item["video_id"]
                np.savez_compressed(
                    RICA2_FEATURE_ROOT / f"{source}-abr-{int(instance)}.npz",
                    feats=feature,
                )
        completed += stop - start
        if completed % 100 == 0 or completed == maximum:
            print(f"I3D feature extraction: {completed}/{maximum}", flush=True)
    elapsed = time.perf_counter() - started
    feature_files = list(RICA2_FEATURE_ROOT.glob("*.npz"))
    requested_complete = all(
        (RICA2_FEATURE_ROOT / f"{source}-abr-{int(instance)}.npz").exists()
        for source, instance in all_video_ids[:maximum]
    )
    result = {
        "status": "PASS" if requested_complete else "FAIL",
        "requested": maximum,
        "cached_files": len(feature_files),
        "elapsed_seconds": elapsed,
        "estimated_full_hours": elapsed / max(maximum, 1) * len(dataset) / 3600.0,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "feature_shape": [9, 1024],
    }
    suffix = "probe" if limit is not None else "full"
    write_json(V4_RESULTS_ROOT / "01_TEACHER" / f"feature_extraction_{suffix}_v4.json", result)
    return result


def _teacher_config(
    train_file: str,
    test_file: str,
    output_folder: Path,
    epochs: int = 350,
    warmup_epochs: int = 3,
) -> dict:
    source = RICA2_ROOT / "configs" / "fine" / "deter_fine_diving_text_data_query.yaml"
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg["dataset"]["data_root"] = str(RICA2_DATA_ROOT)
    cfg["dataset"]["use_feats"] = True
    cfg["dataset"]["feat_dir"] = RICA2_FEATURE_ROOT.name
    cfg["dataset"]["train_datafile"] = train_file
    cfg["dataset"]["test_datafile"] = test_file
    # Keep the published deterministic configuration's batch sizes. The
    # official optimizer does not scale its learning rate with batch size.
    cfg["loader"].update({"train_batch_size": 8, "test_batch_size": 16, "num_workers": 2})
    cfg["opt"]["epochs"] = int(epochs)
    cfg["opt"]["warmup_epochs"] = int(warmup_epochs)
    cfg["output_folder"] = str(output_folder)
    return cfg


def _write_config(name: str, cfg: dict) -> Path:
    path = V4_CACHE_ROOT / "configs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _run_external(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=RICA2_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"External RICA2 command failed with exit code {returncode}: {' '.join(command)}")


def _checkpoint_folder(label: str) -> Path:
    return V4_RUN_ROOT / "teacher" / f"rica2_{label}_v4_v4_{label}"


def _load_checkpoint_epoch(path: Path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["epoch"])


def _metric_frame(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    video_ids = [item for batch in payload["video_ids"] for item in batch]
    pred = np.concatenate([np.atleast_1d(item) for item in payload["pred_scores"]]).astype(float)
    true = np.concatenate([np.atleast_1d(item) for item in payload["gt_scores"]]).astype(float)
    rows = []
    for video_id, prediction, target in zip(video_ids, pred, true):
        source, instance = video_id
        rows.append(
            {
                "clip_uid": f"{source}::{int(instance)}",
                "teacher_predicted_score": float(prediction),
                "teacher_target_score": float(target),
            }
        )
    return pd.DataFrame(rows)


def refresh_teacher_gate_metrics_v4() -> dict:
    """Refresh metric labels from frozen predictions without rerunning test inference.

    This is deliberately separate from RICA2 evaluation: the official 749-video
    test set is inferred once, while corrected reporting code may be applied to
    the immutable exported predictions.
    """
    gate_path = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_gate_v4.json"
    prediction_path = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_test_predictions_v4.parquet"
    if not gate_path.exists() or not prediction_path.exists():
        raise RuntimeError("Teacher gate and frozen test predictions are required for metric refresh")
    result = json.loads(gate_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(prediction_path)
    test = predictions.official_split.astype(str) == "test"
    metrics = aqa_score_metrics(
        predictions.loc[test, "dive_score"],
        predictions.loc[test, "teacher_predicted_score"],
    )
    contract = load_v4_contract()
    result["official_test_metrics"] = metrics
    result["checks"]["official_test_n"] = int(test.sum()) == int(contract["data"]["official_test"])
    result["checks"]["minimum_spearman"] = (
        metrics["spearman"] >= float(contract["external_teacher"]["minimum_test_spearman"])
    )
    result["checks"]["published_protocol_gap"] = (
        abs(metrics["spearman"] - float(contract["external_teacher"]["published_test_spearman"]))
        <= float(contract["external_teacher"]["maximum_published_protocol_gap"])
    )
    result["status"] = "PASS" if all(result["checks"].values()) else "STOP"
    result["metric_refresh"] = {
        "source": str(prediction_path),
        "reran_official_test_inference": False,
        "definition": "official AQA Relative-L2 plus MAE/RMSE",
    }
    write_json(gate_path, result)
    return result


def train_teacher_v4() -> dict:
    _require_teacher_environment()
    contract = load_v4_contract()
    if len(list(RICA2_FEATURE_ROOT.glob("*.npz"))) != int(contract["data"]["samples"]):
        extract_teacher_features_v4()

    # Idempotent resume: once official predictions exist, never evaluate the
    # 749-video test set again. Only refresh the reporting metrics from the
    # frozen prediction file.
    existing_gate = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_gate_v4.json"
    existing_predictions = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_test_predictions_v4.parquet"
    if existing_gate.exists() and existing_predictions.exists():
        return refresh_teacher_gate_metrics_v4()

    pilot_folder = _checkpoint_folder("selection")
    pilot_checkpoint = pilot_folder / "srcc_best.pth.tar"
    if not pilot_checkpoint.exists():
        pilot_cfg = _write_config(
            "rica2_selection_v4.yaml",
            _teacher_config("fit_validation_split.pkl", "calibration_split.pkl", V4_RUN_ROOT / "teacher", 350, 3),
        )
        _run_external(
            [sys.executable, "-u", "train.py", str(pilot_cfg), "--output", "v4_selection", "--data_root", str(RICA2_DATA_ROOT)],
            V4_RESULTS_ROOT / "01_TEACHER" / "rica2_selection_training_v4.log",
        )
    if not pilot_checkpoint.exists():
        raise RuntimeError("RICA2 internal selection run did not produce srcc_best.pth.tar")
    selected_total_epochs = _load_checkpoint_epoch(pilot_checkpoint)
    refit_total_epochs = max(selected_total_epochs, 4)

    final_folder = _checkpoint_folder("final")
    final_checkpoint = final_folder / "last.pth.tar"
    if not final_checkpoint.exists():
        final_cfg = _write_config(
            "rica2_final_v4.yaml",
            _teacher_config("train_split.pkl", "calibration_split.pkl", V4_RUN_ROOT / "teacher", refit_total_epochs - 3, 3),
        )
        _run_external(
            [sys.executable, "-u", "train.py", str(final_cfg), "--output", "v4_final", "--data_root", str(RICA2_DATA_ROOT)],
            V4_RESULTS_ROOT / "01_TEACHER" / "rica2_final_training_v4.log",
        )
    if not final_checkpoint.exists():
        raise RuntimeError("RICA2 final refit did not produce last.pth.tar")

    test_cfg = _write_config(
        "rica2_test_v4.yaml",
        _teacher_config("fit_validation_split.pkl", "test_split.pkl", V4_RUN_ROOT / "teacher", 1, 0),
    )
    before = set(final_folder.glob("epoch_-01_*_outputs.pkl"))
    _run_external(
        [sys.executable, "-u", "eval.py", str(test_cfg), "--ckpt", str(final_checkpoint), "--data_root", str(RICA2_DATA_ROOT), "--test_batch_size", "16"],
        V4_RESULTS_ROOT / "01_TEACHER" / "rica2_official_test_v4.log",
    )
    after = set(final_folder.glob("epoch_-01_*_outputs.pkl"))
    output_files = sorted(after - before) or sorted(after)
    if not output_files:
        raise RuntimeError("RICA2 evaluation did not export predictions")
    predictions = _metric_frame(output_files[-1])
    frame = load_v4_frame()[["clip_uid", "difficulty", "dive_score", "official_split"]]
    predictions = frame.merge(predictions, on="clip_uid", how="inner", validate="one_to_one")
    test = predictions.official_split == "test"
    metrics = aqa_score_metrics(
        predictions.loc[test, "dive_score"], predictions.loc[test, "teacher_predicted_score"]
    )
    predictions["teacher_predicted_quality"] = predictions.teacher_predicted_score / (3.0 * predictions.difficulty)
    predictions.to_parquet(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_test_predictions_v4.parquet", index=False)
    peak_limit = float(contract["external_teacher"]["maximum_peak_vram_gib"]) * 2**30
    feature_summary_path = V4_RESULTS_ROOT / "01_TEACHER" / "feature_extraction_full_v4.json"
    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8")) if feature_summary_path.exists() else {}
    checks = {
        "official_test_n": int(test.sum()) == int(contract["data"]["official_test"]),
        "minimum_spearman": metrics["spearman"] >= float(contract["external_teacher"]["minimum_test_spearman"]),
        "published_protocol_gap": abs(
            metrics["spearman"] - float(contract["external_teacher"]["published_test_spearman"])
        )
        <= float(contract["external_teacher"]["maximum_published_protocol_gap"]),
        "peak_vram": float(feature_summary.get("peak_vram_bytes", 0)) <= peak_limit,
        "fixed_epoch_refit": _load_checkpoint_epoch(final_checkpoint) == refit_total_epochs,
        "finite_predictions": bool(np.isfinite(predictions.teacher_predicted_score).all()),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "STOP",
        "checks": checks,
        "official_test_metrics": metrics,
        "selected_total_epochs": selected_total_epochs,
        "refit_total_epochs": refit_total_epochs,
        "pilot_checkpoint": str(pilot_checkpoint),
        "pilot_checkpoint_sha256": sha256_file(pilot_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "test_prediction_source": str(output_files[-1]),
        "official_test_labels_used_for_model_selection": False,
        "published_protocol_spearman": float(contract["external_teacher"]["published_test_spearman"]),
    }
    write_json(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_gate_v4.json", result)
    return result


def _load_rica2_model(config_path: Path, checkpoint_path: Path):
    _add_rica2_path()
    import torch
    import torch.nn as nn
    from libs.core import load_config
    from libs.modeling import make_meta_arch

    cfg = load_config(str(config_path))
    cfg["dataset"]["data_root"] = str(RICA2_DATA_ROOT)
    model = nn.DataParallel(make_meta_arch(cfg["model_name"], **cfg["model"])).cuda()
    payload = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, cfg


def export_teacher_v4() -> dict:
    _require_teacher_environment()
    gate_path = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_gate_v4.json"
    if not gate_path.exists() or json.loads(gate_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("RICA2 teacher gate must pass before export")
    _add_rica2_path()
    import torch
    from libs.utils.preprocessing import multi_label_multi_class_one_hot_encode

    config_path = V4_CACHE_ROOT / "configs" / "rica2_test_v4.yaml"
    checkpoint_path = _checkpoint_folder("final") / "last.pth.tar"
    model, cfg = _load_rica2_model(config_path, checkpoint_path)

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
            return rica2_raw_output_to_execution_quality(output["all_sample_outputs"])

    # Trace the underlying module rather than DataParallel's scatter/gather
    # wrapper; the exported graph is single-device and deterministic.
    wrapper = CounterfactualWrapper(model.module).cuda().eval()
    dummy_sequence = torch.zeros((2, 9, 1024), dtype=torch.float32, device="cuda")
    dummy_actions = torch.zeros((2, int(cfg["model"]["num_phases"])), dtype=torch.float32, device="cuda")
    traced = torch.jit.trace(wrapper, (dummy_sequence, dummy_actions), strict=False)
    trace_path = V4_RESULTS_ROOT / "01_TEACHER" / "teacher_counterfactual_v4.pt"
    traced.save(str(trace_path))
    dataset = _make_feature_dataset("all_split.pkl")
    frame = load_v4_frame().set_index("clip_uid")
    sequences = []
    predictions = []
    actions = []
    clip_uids = []
    batch_size = 128
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            items = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            feats = torch.stack([item["feats"] for item in items]).cuda()
            masks = torch.ones((len(items), 1, feats.shape[-1]), dtype=torch.bool, device="cuda")
            gt_actions = torch.stack(
                [multi_label_multi_class_one_hot_encode(item["actions_present"], int(cfg["model"]["num_phases"])) for item in items]
            ).cuda()
            output, _ = model(feats, masks, gt_actions, [item["video_id"] for item in items])
            quality = (
                rica2_raw_output_to_execution_quality(output["all_sample_outputs"])
                .detach()
                .cpu()
                .numpy()
            )
            for item, value, action_vector in zip(items, quality, gt_actions.detach().cpu().numpy()):
                source, instance = item["video_id"]
                clip_uid = f"{source}::{int(instance)}"
                clip_uids.append(clip_uid)
                sequences.append(item["feats"].transpose(0, 1).numpy().astype(np.float16))
                predictions.append(float(value))
                actions.append(action_vector.astype(np.float32))
            print(f"Teacher export: {min(start + batch_size, len(dataset))}/{len(dataset)}", flush=True)
    np.savez_compressed(
        V4_RESULTS_ROOT / "01_TEACHER" / "teacher_sequences_v4.npz",
        clip_uid=np.asarray(clip_uids),
        sequence=np.stack(sequences),
        action_presence=np.stack(actions),
    )
    output = frame.loc[clip_uids].reset_index()[
        ["clip_uid", "feature_key", "official_split", "analysis_role", "source_role", "event_family", "action_type", "difficulty", "dive_score", "execution_quality"]
    ].copy()
    output["teacher_predicted_quality"] = np.asarray(predictions, dtype=np.float32)
    output["teacher_predicted_score"] = 3.0 * output.difficulty * output.teacher_predicted_quality
    output["teacher_uncertainty"] = 0.0
    output.to_parquet(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_predictions_v4.parquet", index=False)
    result = {
        "status": "PASS",
        "rows": len(output),
        "sequence_shape": list(np.stack(sequences).shape),
        "prediction_sha256": sha256_file(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_predictions_v4.parquet"),
        "sequence_sha256": sha256_file(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_sequences_v4.npz"),
        "counterfactual_model_sha256": sha256_file(trace_path),
    }
    write_json(V4_RESULTS_ROOT / "01_TEACHER" / "teacher_export_v4.json", result)
    return result


def _make_feature_dataset(split_file: str):
    _add_rica2_path()
    from libs.datasets.finediving import FineDiving

    return FineDiving(
        False,
        "test",
        str(RICA2_DATA_ROOT),
        "FINADiving_MTL_256s",
        "FineDiving_fine-grained_annotation.pkl",
        "FineDiving_coarse_annotation.pkl",
        "fit_split.pkl",
        split_file,
        10,
        16,
        96,
        9,
        [112, 200],
        112,
        True,
        False,
        True,
        RICA2_FEATURE_ROOT.name,
    )


def load_teacher_for_counterfactual():
    config_path = V4_CACHE_ROOT / "configs" / "rica2_test_v4.yaml"
    checkpoint_path = _checkpoint_folder("final") / "last.pth.tar"
    return _load_rica2_model(config_path, checkpoint_path)


def predict_teacher_features(model, sequences: np.ndarray, action_presence: np.ndarray, batch_size: int = 256) -> np.ndarray:
    import torch

    values = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(np.asarray(sequences[start : start + batch_size], dtype=np.float32)).permute(0, 2, 1).cuda()
            actions = torch.from_numpy(np.asarray(action_presence[start : start + batch_size], dtype=np.float32)).cuda()
            masks = torch.ones((len(batch), 1, batch.shape[-1]), dtype=torch.bool, device="cuda")
            output, _ = model(batch, masks, actions)
            values.append(
                rica2_raw_output_to_execution_quality(output["all_sample_outputs"])
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(values).astype(np.float32)
