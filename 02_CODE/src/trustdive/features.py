from __future__ import annotations

import io
import json
import math
import pickle
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import spearmanr
from tqdm import tqdm

from .config import Paths, load_contract
from .metrics import angle_degrees, pck
from .util import set_seed, write_json


MPII_16 = {
    "right_ankle": 0,
    "right_knee": 1,
    "right_hip": 2,
    "left_hip": 3,
    "left_knee": 4,
    "left_ankle": 5,
    "pelvis": 6,
    "thorax": 7,
    "upper_neck": 8,
    "head_top": 9,
    "right_wrist": 10,
    "right_elbow": 11,
    "right_shoulder": 12,
    "left_shoulder": 13,
    "left_elbow": 14,
    "left_wrist": 15,
}


class ZipFrameStore:
    def __init__(self, path: Path):
        self.path = path
        self.archive = zipfile.ZipFile(path)
        self.members: dict[tuple[str, int], list[str]] = {}
        for name in self.archive.namelist():
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = name.replace("\\", "/").split("/")
            if len(parts) < 4:
                continue
            try:
                key = (parts[1], int(parts[2]))
            except ValueError:
                continue
            self.members.setdefault(key, []).append(name)
        for names in self.members.values():
            names.sort()

    def close(self) -> None:
        self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def frame_names(self, source: str, instance: int) -> list[str]:
        names = self.members.get((str(source), int(instance)))
        if not names:
            raise KeyError(f"No trimmed frames for {(source, instance)}")
        return names

    @staticmethod
    def sample_indices(length: int, count: int) -> np.ndarray:
        if length <= 0:
            raise ValueError("Cannot sample an empty clip")
        return np.linspace(0, length - 1, count).round().astype(int)

    def load(self, source: str, instance: int, count: int) -> list[Image.Image]:
        names = self.frame_names(source, instance)
        indices = self.sample_indices(len(names), count)
        frames = []
        for index in indices:
            with self.archive.open(names[int(index)]) as handle:
                frames.append(Image.open(io.BytesIO(handle.read())).convert("RGB"))
        return frames


def _augment_rgb_frames(frames: list[Image.Image], seed: int) -> list[Image.Image]:
    """Create one deterministic, mild broadcast-video augmentation view."""

    rng = np.random.default_rng(seed)
    brightness = float(rng.uniform(0.90, 1.10))
    contrast = float(rng.uniform(0.90, 1.10))
    color = float(rng.uniform(0.90, 1.10))
    blur_radius = float(rng.uniform(0.0, 0.45))
    quality = int(rng.integers(82, 96))
    output: list[Image.Image] = []
    for frame in frames:
        image = ImageEnhance.Brightness(frame).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(color)
        if blur_radius > 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        output.append(Image.open(buffer).convert("RGB"))
    return output


def _frame_tensor(
    frames: list[Image.Image],
    size: tuple[int, int] = (224, 224),
    augmentation_seed: int | None = None,
):
    import torch

    if augmentation_seed is not None:
        frames = _augment_rgb_frames(frames, augmentation_seed)
    arrays = []
    for image in frames:
        width, height = image.size
        scale = max(size[0] / width, size[1] / height)
        resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
        left = (resized.width - size[0]) // 2
        top = (resized.height - size[1]) // 2
        crop = resized.crop((left, top, left + size[0], top + size[1]))
        arrays.append(np.asarray(crop, dtype=np.float32) / 255.0)
    tensor = torch.from_numpy(np.stack(arrays)).permute(3, 0, 1, 2)
    mean = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
    std = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
    return (tensor - mean) / std


class I3DExtractor:
    def __init__(self, device: str | None = None):
        import torch

        try:
            from pytorchvideo.models.hub import i3d_r50
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("pytorchvideo I3D is unavailable; install the GPU extras") from exc
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = i3d_r50(pretrained=True).eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def __call__(
        self, frames: list[Image.Image], augmentation_seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        video = _frame_tensor(frames, augmentation_seed=augmentation_seed).unsqueeze(0).to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            features = video
            for block in self.model.blocks[:-1]:
                features = block(features)
            temporal = features.mean(dim=(-1, -2)).squeeze(0).transpose(0, 1)
            global_feature = temporal.mean(dim=0)
        return temporal.float().cpu().numpy(), global_feature.float().cpu().numpy()


class VideoMAEExtractor:
    """Frozen VideoMAE-Base rescue encoder with temporal token pooling."""

    MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = VideoMAEImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = VideoMAEModel.from_pretrained(self.MODEL_ID).eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.frames = int(self.model.config.num_frames)
        self.temporal_tokens = self.frames // int(self.model.config.tubelet_size)

    def __call__(self, frames: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        inputs = self.processor(frames, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            hidden = self.model(pixel_values=inputs).last_hidden_state
            tokens = hidden.shape[1]
            if tokens % self.temporal_tokens == 1:
                hidden = hidden[:, 1:]
                tokens -= 1
            if tokens % self.temporal_tokens:
                raise AssertionError(
                    f"VideoMAE token count {tokens} is not divisible by {self.temporal_tokens}"
                )
            spatial_tokens = tokens // self.temporal_tokens
            temporal = hidden.reshape(1, self.temporal_tokens, spatial_tokens, -1).mean(dim=2)[0]
            global_feature = temporal.mean(dim=0)
        return temporal.float().cpu().numpy(), global_feature.float().cpu().numpy()


def extract_videomae_features(
    manifest: pd.DataFrame,
    paths: Paths | None = None,
    only_keys: set[str] | None = None,
    overwrite: bool = False,
    maximum_clips: int | None = None,
) -> dict:
    """Extract the prespecified frozen VideoMAE rescue representation."""

    import torch

    paths = paths or Paths()
    output = paths.feature_store / "videomae_v2"
    output.mkdir(parents=True, exist_ok=True)
    extractor = VideoMAEExtractor()
    processed = skipped = 0
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    selected = manifest
    if only_keys is not None:
        selected = selected[selected.clip_uid.isin(only_keys)]
    if maximum_clips is not None:
        selected = selected.iloc[: int(maximum_clips)]
    with ZipFrameStore(paths.trimmed_zip) as store:
        for row in tqdm(selected.itertuples(index=False), total=len(selected), desc="VideoMAE-v2"):
            destination = output / f"{row.feature_key}.npz"
            if destination.exists() and not overwrite:
                skipped += 1
                continue
            frames = store.load(row.source, row.instance, extractor.frames)
            temporal, global_feature = extractor(frames)
            np.savez_compressed(
                destination,
                clip_uid=row.clip_uid,
                model_id=extractor.MODEL_ID,
                temporal=temporal.astype(np.float16),
                global_feature=global_feature.astype(np.float16),
            )
            processed += 1
    elapsed = time.perf_counter() - started
    return {
        "processed": processed,
        "skipped": skipped,
        "eligible": int(len(selected)),
        "elapsed_seconds": elapsed,
        "seconds_per_processed_clip": elapsed / max(processed, 1),
        "estimated_full_hours": (elapsed / max(processed, 1)) * len(manifest) / 3600.0,
        "peak_vram_gb": (
            float(torch.cuda.max_memory_allocated()) / (1024**3) if torch.cuda.is_available() else 0.0
        ),
        "model_id": extractor.MODEL_ID,
        "temporal_tokens": extractor.temporal_tokens,
    }


def extract_rgb_features(
    manifest: pd.DataFrame,
    paths: Paths | None = None,
    only_keys: set[str] | None = None,
    overwrite: bool = False,
) -> dict:
    paths = paths or Paths()
    contract = load_contract(paths.contract)
    output = paths.feature_store / "rgb"
    output.mkdir(parents=True, exist_ok=True)
    frame_count = int(contract["features"]["rgb_frames_per_clip"])
    extractor = I3DExtractor()
    processed = skipped = 0
    started = time.perf_counter()
    with ZipFrameStore(paths.trimmed_zip) as store:
        for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="I3D"):
            if only_keys is not None and row.clip_uid not in only_keys:
                continue
            destination = output / f"{row.feature_key}.npz"
            if destination.exists() and not overwrite:
                skipped += 1
                continue
            frames = store.load(row.source, row.instance, frame_count)
            temporal, global_feature = extractor(frames)
            np.savez_compressed(
                destination,
                clip_uid=row.clip_uid,
                temporal=temporal.astype(np.float16),
                global_feature=global_feature.astype(np.float16),
            )
            processed += 1
    return {"processed": processed, "skipped": skipped, "elapsed_seconds": time.perf_counter() - started}


def extract_rgb_augmented_features(
    manifest: pd.DataFrame,
    fit_clip_uids: set[str],
    paths: Paths | None = None,
    overwrite: bool = False,
    seed: int = 20260817,
) -> dict:
    """Cache one train-only visual augmentation view for v2 tuning."""

    paths = paths or Paths()
    contract = load_contract(paths.contract)
    output = paths.feature_store / "rgb_aug_v2"
    output.mkdir(parents=True, exist_ok=True)
    frame_count = int(contract["features"]["rgb_frames_per_clip"])
    extractor = I3DExtractor()
    processed = skipped = 0
    started = time.perf_counter()
    with ZipFrameStore(paths.trimmed_zip) as store:
        selected = manifest[manifest.clip_uid.isin(fit_clip_uids)]
        for row in tqdm(selected.itertuples(index=False), total=len(selected), desc="I3D-v2-aug"):
            destination = output / f"{row.feature_key}.npz"
            if destination.exists() and not overwrite:
                skipped += 1
                continue
            frames = store.load(row.source, row.instance, frame_count)
            clip_seed = int(row.feature_key[:8], 16) ^ int(seed)
            temporal, global_feature = extractor(frames, augmentation_seed=clip_seed)
            np.savez_compressed(
                destination,
                clip_uid=row.clip_uid,
                temporal=temporal.astype(np.float16),
                global_feature=global_feature.astype(np.float16),
                augmentation_seed=clip_seed,
            )
            processed += 1
    return {
        "processed": processed,
        "skipped": skipped,
        "eligible": int(len(selected)),
        "elapsed_seconds": time.perf_counter() - started,
        "augmentation": "brightness_contrast_color_blur_jpeg_no_flip",
    }


def _pose_letterbox(
    image: Image.Image,
    width: int = 256,
    height: int = 192,
) -> tuple[Image.Image, float, int, int]:
    """Resize without distorting image-plane geometry and pad to an HRNet-safe canvas."""

    scale = min(width / image.width, height / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, (left, top))
    return canvas, scale, left, top


def _image_to_pose_tensor(image: Image.Image, width: int = 256, height: int = 192):
    import torch

    canvas, _, _, _ = _pose_letterbox(image, width, height)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def _square_crop(
    image: Image.Image,
    center_x: float,
    center_y: float,
    side: float,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    side = max(float(side), 8.0)
    left = float(center_x) - side / 2.0
    top = float(center_y) - side / 2.0
    right = left + side
    bottom = top + side
    left_i, top_i, right_i, bottom_i = (round(x) for x in (left, top, right, bottom))
    return image.crop((left_i, top_i, right_i, bottom_i)), (
        float(left_i),
        float(top_i),
        float(right_i),
        float(bottom_i),
    )


def _annotation_pose_tensor(image: Image.Image, row: dict):
    center_x, center_y = (float(x) for x in row["center"])
    side = 200.0 * float(row["scale"])
    crop, box = _square_crop(image, center_x, center_y, side)
    canvas, resize_scale, pad_left, pad_top = _pose_letterbox(crop)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    import torch

    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std, box, resize_scale, pad_left, pad_top


class PoseDiveDataset:
    def __init__(self, root: Path, split: str):
        import torch

        self.torch = torch
        annotation = root / "annotations" / f"pose_finediv_{split}.json"
        with annotation.open("r", encoding="utf-8") as handle:
            self.rows = json.load(handle)
        self.image_root = root / "images"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.image_root / row["image"]).convert("RGB")
        joints = np.asarray(row["joints"], dtype=np.float32)
        tensor, box, resize_scale, pad_left, pad_top = _annotation_pose_tensor(image, row)
        joints[:, 0] = ((joints[:, 0] - box[0]) * resize_scale + pad_left) / 256.0
        joints[:, 1] = ((joints[:, 1] - box[1]) * resize_scale + pad_top) / 192.0
        visible = np.asarray(row["joints_vis"], dtype=np.float32)
        return tensor, self.torch.from_numpy(joints), self.torch.from_numpy(visible)


def soft_argmax_2d(logits):
    import torch

    batch, joints, height, width = logits.shape
    probability = torch.softmax(logits.flatten(2), dim=-1).reshape(batch, joints, height, width)
    x_axis = torch.linspace(0.0, 1.0, width, device=logits.device)
    y_axis = torch.linspace(0.0, 1.0, height, device=logits.device)
    x = (probability.sum(dim=2) * x_axis).sum(dim=-1)
    y = (probability.sum(dim=3) * y_axis).sum(dim=-1)
    confidence = probability.flatten(2).amax(dim=-1)
    return torch.stack([x, y], dim=-1), confidence


def create_pose_model(model_name: str = "hrnet_w18_small_v2.ms_in1k", pretrained: bool = True):
    import torch
    import torch.nn as nn
    import timm

    class PoseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
            channels = self.backbone.feature_info.channels()[-1]
            self.head = nn.Conv2d(channels, 16, kernel_size=1)

        def forward(self, x):
            features = self.backbone(x)[-1]
            heatmaps = self.head(features)
            return nn.functional.interpolate(heatmaps, size=(48, 64), mode="bilinear", align_corners=False)

    return PoseModel()


def _pose_epoch(model, loader, device, optimizer=None):
    import torch

    training = optimizer is not None
    model.train(training)
    losses = []
    predictions = []
    targets = []
    visibility = []
    for images, joints, visible in loader:
        images = images.to(device)
        joints = joints.to(device)
        visible = visible.to(device)
        with torch.set_grad_enabled(training):
            coordinates, _ = soft_argmax_2d(model(images))
            squared = ((coordinates - joints) ** 2).sum(dim=-1)
            loss = (squared * visible).sum() / visible.sum().clamp_min(1.0)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        predictions.append(coordinates.detach().cpu().numpy())
        targets.append(joints.detach().cpu().numpy())
        visibility.append(visible.detach().cpu().numpy())
    pred_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    visible_array = np.concatenate(visibility)
    pred_concepts = pose_concepts(pred_array, np.ones(pred_array.shape[:2], dtype=np.float32))
    target_concepts = pose_concepts(target_array, np.ones(target_array.shape[:2], dtype=np.float32))
    key_angle_mae = float(np.nanmean(np.abs(pred_concepts[:, :2] - target_concepts[:, :2])))
    return {
        "loss": float(np.mean(losses)),
        "pck_at_0_1": pck(pred_array, target_array, visible_array, 0.1),
        "key_angle_mae_deg": key_angle_mae,
    }


def train_pose_model(paths: Paths | None = None, max_epochs: int = 12, seed: int = 20260815) -> dict:
    import torch
    from torch.utils.data import DataLoader

    paths = paths or Paths()
    set_seed(seed)
    destination = paths.feature_store / "pose" / "pose_model.pth"
    train_data = PoseDiveDataset(paths.pose_dive, "train")
    test_data = PoseDiveDataset(paths.pose_dive, "test")
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_pose_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best = {"pck_at_0_1": -np.inf}
    history = []
    for epoch in range(max_epochs):
        train_metrics = _pose_epoch(model, train_loader, device, optimizer)
        test_metrics = _pose_epoch(model, test_loader, device)
        row = {"epoch": epoch + 1, "train": train_metrics, "test": test_metrics}
        history.append(row)
        if test_metrics["pck_at_0_1"] > best["pck_at_0_1"]:
            best = {**test_metrics, "epoch": epoch + 1}
            torch.save({"model": model.state_dict(), "model_name": "hrnet_w18_small_v2.ms_in1k"}, destination)
    result = {"best": best, "history": history, "checkpoint": str(destination)}
    write_json(paths.results / "01_PROBE" / "pose_training.json", result)
    return result


def load_pose_model(checkpoint: Path, device=None):
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = create_pose_model(payload.get("model_name", "hrnet_w18_small_v2.ms_in1k"), pretrained=False)
    model.load_state_dict(payload["model"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model.eval().to(device), device


class PersonDetector:
    """Frozen COCO person detector used only to provide top-down pose crops."""

    def __init__(self, device, threshold: float = 0.25):
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        self.torch = torch
        self.device = device
        self.threshold = float(threshold)
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).eval().to(device)

    def __call__(self, images: list[Image.Image]) -> tuple[list[tuple[float, float, float, float]], np.ndarray]:
        tensors = []
        for image in images:
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensors.append(self.torch.from_numpy(array).permute(2, 0, 1).to(self.device))
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(tensors), 5):
                outputs.extend(self.model(tensors[start : start + 5]))
        boxes = []
        scores = []
        for image, output in zip(images, outputs):
            labels = output["labels"].detach().cpu().numpy()
            confidence = output["scores"].detach().cpu().numpy()
            candidates = np.flatnonzero((labels == 1) & (confidence >= self.threshold))
            if len(candidates):
                chosen = int(candidates[np.argmax(confidence[candidates])])
                x1, y1, x2, y2 = output["boxes"][chosen].detach().cpu().numpy().tolist()
                side = 1.35 * max(x2 - x1, y2 - y1)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                boxes.append((center_x - side / 2, center_y - side / 2, center_x + side / 2, center_y + side / 2))
                scores.append(float(confidence[chosen]))
            else:
                side = float(max(image.size))
                center_x, center_y = image.width / 2.0, image.height / 2.0
                boxes.append((center_x - side / 2, center_y - side / 2, center_x + side / 2, center_y + side / 2))
                scores.append(0.0)
        return boxes, np.asarray(scores, dtype=np.float32)


def _detected_pose_tensor(image: Image.Image, box: tuple[float, float, float, float]):
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    side = max(box[2] - box[0], box[3] - box[1])
    crop, crop_box = _square_crop(image, center_x, center_y, side)
    canvas, resize_scale, pad_left, pad_top = _pose_letterbox(crop)
    return _image_to_pose_tensor(canvas), crop_box, resize_scale, pad_left, pad_top


def _pose_to_image_coordinates(
    normalized_xy: np.ndarray,
    crop_box: tuple[float, float, float, float],
    resize_scale: float,
    pad_left: int,
    pad_top: int,
) -> np.ndarray:
    output = np.asarray(normalized_xy, dtype=np.float32).copy()
    output[:, 0] = (output[:, 0] * 256.0 - pad_left) / resize_scale + crop_box[0]
    output[:, 1] = (output[:, 1] * 192.0 - pad_top) / resize_scale + crop_box[1]
    return output


def pose_concepts(coordinates: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    k = MPII_16
    output = []
    for xy, conf in zip(coordinates, confidence):
        torso = max(np.linalg.norm(xy[k["thorax"]] - xy[k["pelvis"]]), 1e-6)
        knee = np.nanmean(
            [
                angle_degrees(xy[k["right_hip"]], xy[k["right_knee"]], xy[k["right_ankle"]]),
                angle_degrees(xy[k["left_hip"]], xy[k["left_knee"]], xy[k["left_ankle"]]),
            ]
        )
        hip = np.nanmean(
            [
                angle_degrees(xy[k["right_shoulder"]], xy[k["right_hip"]], xy[k["right_knee"]]),
                angle_degrees(xy[k["left_shoulder"]], xy[k["left_hip"]], xy[k["left_knee"]]),
            ]
        )
        axis = xy[k["thorax"]] - xy[k["pelvis"]]
        verticality = abs(math.degrees(math.atan2(float(axis[0]), float(-axis[1]))))
        ankle_separation = np.linalg.norm(xy[k["right_ankle"]] - xy[k["left_ankle"]]) / torso
        compactness = np.mean(
            [np.linalg.norm(xy[index] - xy[k["pelvis"]]) / torso for index in (0, 5, 10, 15)]
        )
        output.append([knee, hip, verticality, ankle_separation, compactness, float(np.mean(conf))])
    return np.asarray(output, dtype=np.float32)


def extract_pose_features(
    manifest: pd.DataFrame,
    paths: Paths | None = None,
    only_keys: set[str] | None = None,
    overwrite: bool = False,
) -> dict:
    import torch

    paths = paths or Paths()
    checkpoint = paths.feature_store / "pose" / "pose_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Pose checkpoint missing: {checkpoint}")
    model, device = load_pose_model(checkpoint)
    detector = PersonDetector(device)
    count = int(load_contract(paths.contract)["features"]["pose_frames_per_clip"])
    processed = skipped = 0
    started = time.perf_counter()
    with ZipFrameStore(paths.trimmed_zip) as store:
        for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Pose"):
            if only_keys is not None and row.clip_uid not in only_keys:
                continue
            destination = paths.feature_store / "pose" / f"{row.feature_key}.npz"
            if destination.exists() and not overwrite:
                skipped += 1
                continue
            frames = store.load(row.source, row.instance, count)
            boxes, detector_scores = detector(frames)
            prepared = [_detected_pose_tensor(frame, box) for frame, box in zip(frames, boxes)]
            batch = torch.stack([item[0] for item in prepared]).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                coordinates, confidence = soft_argmax_2d(model(batch))
            xy = coordinates.float().cpu().numpy()
            conf = confidence.float().cpu().numpy()
            concepts = pose_concepts(xy, conf)
            detector_valid = detector_scores > detector.threshold
            concepts[~detector_valid] = np.nan
            image_xy = np.stack(
                [
                    _pose_to_image_coordinates(point, crop_box, scale, pad_left, pad_top)
                    for point, (_, crop_box, scale, pad_left, pad_top) in zip(xy, prepared)
                ]
            )
            np.savez_compressed(
                destination,
                clip_uid=row.clip_uid,
                coordinates=xy.astype(np.float32),
                confidence=conf.astype(np.float32),
                concepts=concepts,
                image_coordinates=image_xy.astype(np.float32),
                person_boxes=np.asarray(boxes, dtype=np.float32),
                detector_confidence=detector_scores,
                detector_valid=detector_valid,
            )
            processed += 1
    validity = []
    if processed:
        for row in manifest.itertuples(index=False):
            if only_keys is not None and row.clip_uid not in only_keys:
                continue
            with np.load(paths.feature_store / "pose" / f"{row.feature_key}.npz") as payload:
                validity.extend(payload["detector_valid"].tolist())
    return {
        "processed": processed,
        "skipped": skipped,
        "elapsed_seconds": time.perf_counter() - started,
        "person_detection_rate": float(np.mean(validity)) if validity else float("nan"),
    }


def splash_features_from_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    series = payload.get("splash", [])
    values = np.asarray([np.nan if value is None else float(value) for value in series], dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return {
            "splash_valid": False,
            "splash_peak": np.nan,
            "splash_auc": np.nan,
            "splash_duration": 0,
            "splash_expansion": np.nan,
            "splash_peak_index": -1,
        }
    filled = np.where(valid, values, 0.0)
    valid_indices = np.flatnonzero(valid)
    peak_index = int(np.nanargmax(values))
    first = int(valid_indices[0])
    before_peak = filled[first : peak_index + 1]
    expansion = float((before_peak[-1] - before_peak[0]) / max(1, len(before_peak) - 1))
    return {
        "splash_valid": True,
        "splash_peak": float(np.nanmax(values)),
        "splash_auc": float(np.nansum(values)),
        "splash_duration": int(valid.sum()),
        "splash_expansion": expansion,
        "splash_peak_index": peak_index,
    }


def extract_splash_features(manifest: pd.DataFrame, paths: Paths | None = None) -> pd.DataFrame:
    paths = paths or Paths()
    rows = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Splash"):
        features = splash_features_from_pickle(Path(row.splash_path))
        rows.append({"clip_uid": row.clip_uid, "feature_key": row.feature_key, **features})
    frame = pd.DataFrame(rows)
    destination = paths.feature_store / "splash" / "splash_features.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return frame


def load_splash_features(paths: Paths | None = None) -> pd.DataFrame:
    paths = paths or Paths()
    path = paths.feature_store / "splash" / "splash_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Splash features missing: {path}")
    return pd.read_parquet(path)
