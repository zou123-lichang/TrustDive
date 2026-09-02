from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .util import sha256_file, write_json
from .v4_teacher import (
    V4_CACHE_ROOT,
    _add_rica2_path,
    _checkpoint_folder,
    _load_rica2_model,
    _require_teacher_environment,
)
from .v5_data import V5_RESULTS_ROOT
from .v5_teacher import _make_rica2_inference_deterministic, _v5_execution_quality
from .v6_data import V6_RESULTS_ROOT, load_v6_contract, load_v6_frame, require_v6_audit


def extract_teacher_latents_v6(overwrite: bool = False) -> dict:
    """Export the deterministic teacher's actual semantic and scoring latents.

    The source checkpoint and the v5 input sequences remain read-only. The
    exported TorchScript module is the only scorer used for all v6 hybrids.
    """
    require_v6_audit()
    _require_teacher_environment()
    folder = V6_RESULTS_ROOT / "01_LATENTS"
    summary_path = folder / "latent_extraction_v6.json"
    latent_path = folder / "teacher_latents_v6.npz"
    prediction_path = folder / "teacher_predictions_v6.parquet"
    trace_path = folder / "teacher_latent_counterfactual_v6.pt"
    if not overwrite and summary_path.exists() and all(
        path.exists() for path in (latent_path, prediction_path, trace_path)
    ):
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            return existing

    _add_rica2_path()
    import torch

    config_path = V4_CACHE_ROOT / "configs" / "rica2_test_v4.yaml"
    checkpoint_path = _checkpoint_folder("final") / "last.pth.tar"
    model, _ = _load_rica2_model(config_path, checkpoint_path)
    deterministic_patch = _make_rica2_inference_deterministic(model)

    class LatentWrapper(torch.nn.Module):
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
            x, masks = self.wrapped.backbone(features, masks)
            if self.wrapped.neck is not None:
                semantic, _ = self.wrapped.neck(x, masks, action_presence)
            else:
                semantic = x.permute(0, 2, 1)
            phase_scoring = self.wrapped.quality_score_head.common_base(semantic)
            global_samples, _ = self.wrapped.quality_score_head.phase_composer.process_phases(
                phase_scoring, None, 1, action_presence.detach()
            )
            raw = self.wrapped.quality_score_head.scoring_head(global_samples).permute(1, 0, 2)
            quality = _v5_execution_quality(raw)
            return quality, global_samples.squeeze(0), semantic

    with np.load(
        V5_RESULTS_ROOT / "01_REFERENCES" / "teacher_sequences_v5.npz",
        allow_pickle=False,
    ) as payload:
        clip_uid = payload["clip_uid"].astype(str)
        sequences = payload["sequence"].astype(np.float32)
        actions = payload["action_presence"].astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = LatentWrapper(model.module).to(device).eval()
    dummy_sequence = torch.from_numpy(sequences[:2]).to(device)
    dummy_actions = torch.from_numpy(actions[:2]).to(device)
    with torch.inference_mode():
        first = wrapper(dummy_sequence, dummy_actions)
        second = wrapper(dummy_sequence, dummy_actions)
    repeat_difference = max(
        float(torch.max(torch.abs(a - b)).detach().cpu()) for a, b in zip(first, second)
    )
    traced = torch.jit.trace(
        wrapper, (dummy_sequence, dummy_actions), strict=False, check_trace=True
    )
    with torch.inference_mode():
        traced_first = traced(dummy_sequence, dummy_actions)
    trace_difference = max(
        float(torch.max(torch.abs(a - b)).detach().cpu())
        for a, b in zip(first, traced_first)
    )
    traced.save(str(trace_path))

    quality_parts: list[np.ndarray] = []
    global_parts: list[np.ndarray] = []
    semantic_parts: list[np.ndarray] = []
    batch_size = 256
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            stop = min(start + batch_size, len(sequences))
            output = traced(
                torch.from_numpy(sequences[start:stop]).to(device),
                torch.from_numpy(actions[start:stop]).to(device),
            )
            quality_parts.append(output[0].detach().cpu().numpy().reshape(-1))
            global_parts.append(output[1].detach().cpu().numpy())
            semantic_parts.append(output[2].detach().cpu().numpy().astype(np.float16))
            print(f"V6 latent extraction: {stop}/{len(sequences)}", flush=True)
    quality = np.concatenate(quality_parts).astype(np.float32)
    global_latent = np.concatenate(global_parts).astype(np.float32)
    semantic_latent = np.concatenate(semantic_parts).astype(np.float16)
    np.savez_compressed(
        latent_path,
        clip_uid=clip_uid,
        global_latent=global_latent,
        semantic_latent=semantic_latent,
        action_presence=actions.astype(np.float32),
    )

    frame = load_v6_frame().set_index("clip_uid").loc[clip_uid].reset_index()
    output = frame[
        [
            "clip_uid", "feature_key", "official_split", "analysis_role", "source_role",
            "event_family", "action_type", "difficulty", "dive_score", "execution_quality",
        ]
    ].copy()
    output["teacher_predicted_quality"] = quality
    output["teacher_predicted_score"] = 3.0 * output.difficulty * quality
    output.to_parquet(prediction_path, index=False)

    v5 = pd.read_parquet(
        V5_RESULTS_ROOT / "01_REFERENCES" / "teacher_predictions_v5.parquet"
    ).set_index("clip_uid").loc[clip_uid]
    v5_difference = float(
        np.max(np.abs(quality - v5.teacher_predicted_quality.to_numpy(dtype=float)))
    )
    contract = load_v6_contract()
    checks = {
        "rows": len(output) == int(contract["data"]["samples"]),
        "repeat_inference_exact": repeat_difference <= 1e-7,
        "torchscript_matches_direct": trace_difference <= 1e-5,
        "v5_teacher_scores_match": v5_difference <= 1e-5,
        "global_latent_shape": list(global_latent.shape) == [3000, 256],
        "semantic_latent_shape": list(semantic_latent.shape) == [3000, 29, 512],
        "finite": bool(
            np.isfinite(quality).all()
            and np.isfinite(global_latent).all()
            and np.isfinite(semantic_latent).all()
        ),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "deterministic_inference_patch": deterministic_patch,
        "execution_quality_scale_divisor": 3.0,
        "repeat_inference_maximum_difference": repeat_difference,
        "torchscript_direct_maximum_difference": trace_difference,
        "v5_teacher_maximum_difference": v5_difference,
        "global_latent_shape": list(global_latent.shape),
        "semantic_latent_shape": list(semantic_latent.shape),
        "official_test_labels_accessed": False,
        "latent_sha256": sha256_file(latent_path),
        "prediction_sha256": sha256_file(prediction_path),
        "counterfactual_model_sha256": sha256_file(trace_path),
    }
    write_json(summary_path, result)
    return result


if __name__ == "__main__":
    print(json.dumps(extract_teacher_latents_v6(), indent=2, ensure_ascii=False))
