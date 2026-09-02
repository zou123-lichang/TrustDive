from __future__ import annotations

import json
import pickle
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Paths, load_contract
from .util import feature_key, git_dirty_count, git_head, sha256_file, stable_json


def load_pickle(path: Path):
    # FineDiving ships trusted local NumPy pickles. Never use this helper on
    # untrusted downloaded pickle files.
    with path.open("rb") as handle:
        return pickle.load(handle)


def event_family(source: str) -> str:
    value = re.sub(r"_r\d+$", "", source, flags=re.IGNORECASE)
    value = re.sub(r"_\d+$", "", value)
    return value


def clip_uid(source: str, instance: int) -> str:
    return f"{source}::{int(instance)}"


def _fine_paths(paths: Paths) -> tuple[Path, Path, Path, Path]:
    annotations = paths.fine_diving / "Annotations"
    splits = paths.fine_diving / "train_test_split"
    return (
        annotations / "FineDiving_coarse_annotation.pkl",
        annotations / "FineDiving_fine-grained_annotation.pkl",
        splits / "train_split.pkl",
        splits / "test_split.pkl",
    )


def build_manifest(paths: Paths | None = None) -> pd.DataFrame:
    paths = paths or Paths()
    coarse_path, fine_path, train_path, test_path = _fine_paths(paths)
    coarse = load_pickle(coarse_path)
    fine = load_pickle(fine_path)
    train_keys = set(map(tuple, load_pickle(train_path)))
    test_keys = set(map(tuple, load_pickle(test_path)))

    if set(coarse) != set(fine):
        raise AssertionError("Coarse and fine annotations do not have identical clip keys")
    if train_keys & test_keys:
        raise AssertionError("Official train/test clip keys overlap")
    if train_keys | test_keys != set(coarse):
        raise AssertionError("Official train/test keys do not cover all annotations")

    rows: list[dict] = []
    for source, instance in sorted(coarse, key=lambda x: (str(x[0]), int(x[1]))):
        c = coarse[(source, instance)]
        f = fine[(source, instance)]
        if str(c["action_type"]) != str(f["action_type"]):
            raise AssertionError(f"Action mismatch for {(source, instance)}")
        if not np.isclose(float(c["dive_score"]), float(f["dive_score"])):
            raise AssertionError(f"Score mismatch for {(source, instance)}")
        judges = [float(x) for x in c["judge_scores"]]
        difficulty = float(c["difficulty"])
        score = float(c["dive_score"])
        splash_path = paths.splash / str(source) / f"{int(instance)}.pkl"
        rows.append(
            {
                "clip_uid": clip_uid(str(source), int(instance)),
                "feature_key": feature_key(str(source), int(instance)),
                "source": str(source),
                "instance": int(instance),
                "event_family": event_family(str(source)),
                "official_split": "train" if (source, instance) in train_keys else "test",
                "action_type": str(c["action_type"]),
                "action_group": str(c["action_type"])[0],
                "difficulty": difficulty,
                "dive_score": score,
                "execution_quality": score / (3.0 * difficulty),
                "judge_count": len(judges),
                "judge_scores_json": stable_json(judges),
                "start_frame": int(c["start_frame"]),
                "end_frame": int(c["end_frame"]),
                "frame_count": int(len(f["frames_labels"])),
                "transition_frames_json": stable_json(np.asarray(f["steps_transit_frames"], dtype=int)),
                "frame_labels_json": stable_json(np.asarray(f["frames_labels"], dtype=int)),
                "subactions_json": stable_json({str(k): str(v) for k, v in f["sub-action_types"].items()}),
                "splash_path": str(splash_path),
                "splash_exists": splash_path.exists(),
            }
        )
    return pd.DataFrame(rows)


def _zip_clip_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            parts = Path(name).parts
            if len(parts) < 4 or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                keys.add((parts[1], int(parts[2])))
            except (ValueError, IndexError):
                continue
    return keys


def audit_dataset(paths: Paths | None = None, verify_zip: bool = True) -> dict:
    paths = paths or Paths()
    contract = load_contract(paths.contract)
    manifest = build_manifest(paths)
    train = manifest[manifest.official_split == "train"]
    test = manifest[manifest.official_split == "test"]
    train_sources = set(train.source)
    test_sources = set(test.source)

    pose_images = len(list((paths.pose_dive / "images").glob("*.jpg")))
    pose_json_counts: dict[str, int] = {}
    for json_path in sorted((paths.pose_dive / "annotations").glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            pose_json_counts[json_path.name] = len(json.load(handle))

    splash_files = len(list(paths.splash.glob("*/*.pkl")))
    zip_keys = _zip_clip_keys(paths.trimmed_zip) if verify_zip else set()
    annotation_keys = set(zip(manifest.source, manifest.instance))
    hashes = {}
    for path in _fine_paths(paths):
        hashes[str(path)] = sha256_file(path)

    actual = {
        "samples": len(manifest),
        "official_train": len(train),
        "official_test": len(test),
        "action_types": int(manifest.action_type.nunique()),
        "difficulty_types": int(manifest.difficulty.nunique()),
        "judge_panels": {str(k): int(v) for k, v in Counter(manifest.judge_count).items()},
        "test_seven_judge": int(((manifest.official_split == "test") & (manifest.judge_count == 7)).sum()),
        "train_sources": len(train_sources),
        "test_sources": len(test_sources),
        "overlapping_sources": len(train_sources & test_sources),
        "event_families": int(manifest.event_family.nunique()),
        "pose_images": pose_images,
        "pose_json_counts": pose_json_counts,
        "splash_files": splash_files,
        "splash_missing": int((~manifest.splash_exists).sum()),
        "zip_clip_keys": len(zip_keys) if verify_zip else None,
        "zip_missing_annotations": len(annotation_keys - zip_keys) if verify_zip else None,
        "zip_extra_clips": len(zip_keys - annotation_keys) if verify_zip else None,
    }
    expected = contract["data_assertions"]
    mismatches = {}
    for key, expected_value in expected.items():
        if key == "judge_panels":
            if actual[key] != expected_value:
                mismatches[key] = {"expected": expected_value, "actual": actual[key]}
        elif actual.get(key) != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": actual.get(key)}
    if verify_zip and annotation_keys - zip_keys:
        mismatches["zip_missing_annotations"] = sorted(map(str, annotation_keys - zip_keys))[:20]

    old_projects = []
    for name in (
        "ACL_Energy_Redistribution_CURRENT",
        "ACL_Fatigue_Readiness_CURRENT",
        "ACL_Multijoint_Fatigue_CURRENT",
    ):
        root = paths.project.parent / name
        old_projects.append(
            {"path": str(root), "head": git_head(root), "dirty_count": git_dirty_count(root)}
        )
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "actual": actual,
        "expected_mismatches": mismatches,
        "input_hashes": hashes,
        "old_project_snapshots": old_projects,
        "notes": [
            "PoseDive JSON annotations are image-level pose labels, not per-video FineDiving skeleton tracks.",
            "Splash PKLs are model-derived predictions and are not treated as ground-truth masks.",
        ],
    }


def save_manifest(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["official_split", "source", "instance"]).to_parquet(path, index=False)


def load_manifest(paths: Paths | None = None) -> pd.DataFrame:
    paths = paths or Paths()
    if not paths.manifest.exists():
        raise FileNotFoundError(f"Manifest missing; run build-manifest first: {paths.manifest}")
    return pd.read_parquet(paths.manifest)


def parse_json_column(value: str) -> list:
    return json.loads(value)


def select_probe_clips(manifest: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n > len(manifest):
        raise ValueError(f"Requested {n} clips from {len(manifest)}")
    # Deterministic stratified sample over split, panel size, and score quartile.
    frame = manifest.copy()
    frame["score_bin"] = pd.qcut(frame.dive_score.rank(method="first"), 4, labels=False)
    groups = frame.groupby(["official_split", "judge_count", "score_bin"], group_keys=False)
    sampled = groups.apply(
        lambda g: g.sample(max(1, round(n * len(g) / len(frame))), random_state=seed),
        include_groups=False,
    )
    sampled = sampled.drop_duplicates("clip_uid")
    if len(sampled) < n:
        remaining = frame[~frame.clip_uid.isin(sampled.clip_uid)].sample(n - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, remaining], ignore_index=True)
    return sampled.sort_values("clip_uid").head(n).reset_index(drop=True)
