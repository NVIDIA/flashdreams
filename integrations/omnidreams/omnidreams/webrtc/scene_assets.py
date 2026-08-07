# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OmniDreams WebRTC scene discovery, extraction, and layout preparation."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from collections.abc import Set as AbstractSet
from pathlib import Path, PurePosixPath

from filelock import FileLock
from loguru import logger
from omnidreams.scenes import (
    SCENE_CLIPGT_DIRNAME,
    SCENE_FRAME_SUFFIXES,
    SCENE_FRAMES_DIRNAME,
    SCENE_IMAGE_SUFFIXES,
    SCENE_VARIANT_DEFAULT,
    hf_hub_download_scene,
    hf_scenes_repo_id,
    prompt_variant_for_scene_variant,
    scene_variant_suffix,
    scenes_cache_root,
)


def _choose_existing_asset(
    directory: Path,
    *,
    exact_name: str | None = None,
    fallback_stems: tuple[str, ...] = (),
    fallback_prefixes: tuple[str, ...] = (),
    allowed_suffixes: AbstractSet[str] | None = None,
    preferred_stems: tuple[str, ...] = (),
) -> Path | None:
    if not directory.is_dir():
        return None

    if exact_name is not None:
        exact_path = directory / exact_name
        if exact_path.is_file() and (
            allowed_suffixes is None or exact_path.suffix.lower() in allowed_suffixes
        ):
            return exact_path

    candidates = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
            continue
        if (
            path.stem in preferred_stems
            or path.stem in fallback_stems
            or any(path.stem.startswith(f"{prefix}-") for prefix in fallback_prefixes)
        ):
            candidates.append(path)

    if not candidates:
        return None

    preferred_order = {stem: index for index, stem in enumerate(preferred_stems)}
    return sorted(
        candidates,
        key=lambda path: (
            preferred_order.get(path.stem, len(preferred_order)),
            path.name,
        ),
    )[0]


def _camera_name_candidates(camera_name: str) -> tuple[str, ...]:
    underscore = camera_name.replace(":", "_")
    colon = camera_name.replace("_", ":")
    return tuple(dict.fromkeys((camera_name, underscore, colon)))


def _first_frame_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), path.name) if stem.isdigit() else (2**63 - 1, path.name)


def _resolve_first_frame(clipgt_dir: Path, camera_name: str) -> Path | None:
    frames_root = clipgt_dir / SCENE_FRAMES_DIRNAME
    if not frames_root.is_dir():
        return None
    candidate_dirs = [
        frames_root / name
        for name in _camera_name_candidates(camera_name)
        if (frames_root / name).is_dir()
    ]
    if not candidate_dirs:
        candidate_dirs = [
            path for path in sorted(frames_root.iterdir()) if path.is_dir()
        ]
    for directory in candidate_dirs:
        frames = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SCENE_FRAME_SUFFIXES
        ]
        if frames:
            return sorted(frames, key=_first_frame_sort_key)[0]
    return None


def resolve_scene_assets(
    scene_dir: Path,
    *,
    prompt_filename: str,
    clipgt_dirname: str,
    camera_name: str = "camera_front_wide_120fov",
    variant: str = SCENE_VARIANT_DEFAULT,
) -> tuple[Path, Path, Path]:
    """Resolve the ClipGT root, first frame, and prompt for a scene."""
    missing_assets = []
    clipgt_dir = scene_dir / clipgt_dirname
    if not clipgt_dir.is_dir():
        missing_assets.append(str(clipgt_dir))
        resolved_clipgt_dir = None
    else:
        resolved_clipgt_dir = clipgt_dir

    first_frame_path = (
        None
        if resolved_clipgt_dir is None
        else _resolve_first_frame(resolved_clipgt_dir, camera_name)
    )
    if first_frame_path is None and resolved_clipgt_dir is not None:
        first_frame_path = _choose_existing_asset(
            resolved_clipgt_dir,
            fallback_stems=("first_image_1",),
            allowed_suffixes=SCENE_IMAGE_SUFFIXES,
            preferred_stems=("first_image",),
        )
    if first_frame_path is None:
        missing_assets.append(
            f"frames/<camera>/*.jpeg or first_image.* under {resolved_clipgt_dir}/"
        )

    weather_prompt_stem = f"prompt{prompt_variant_for_scene_variant(variant)}"
    prompt_path = (
        None
        if resolved_clipgt_dir is None
        else _choose_existing_asset(
            resolved_clipgt_dir,
            fallback_stems=("prompt1", "prompt2", "prompt3", "prompt"),
            allowed_suffixes={".txt"},
            preferred_stems=(weather_prompt_stem, "prompt"),
        )
    )
    if prompt_path is None:
        missing_assets.append(f"{prompt_filename} under {resolved_clipgt_dir}/")

    if missing_assets:
        raise FileNotFoundError(
            "Missing Omnidreams WebRTC scene assets: " + ", ".join(missing_assets)
        )

    assert resolved_clipgt_dir is not None
    assert first_frame_path is not None
    assert prompt_path is not None
    return resolved_clipgt_dir, first_frame_path, prompt_path


def _safe_extract_zip(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_file() or destination.is_symlink():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(source) as zf:
        for member in zf.infolist():
            member_path = PurePosixPath(member.filename)
            if (
                member_path.is_absolute()
                or not member_path.parts
                or any(part in {"", ".", ".."} for part in member_path.parts)
            ):
                raise ValueError(
                    f"Unsafe archive member in {source}: {member.filename}"
                )
            target = destination / Path(*member_path.parts)
            target_resolved = target.resolve()
            if destination_root != target_resolved and destination_root not in (
                target_resolved.parents
            ):
                raise ValueError(
                    f"Archive member escapes destination: {member.filename}"
                )
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def extract_local_scene(
    scene_dir: Path,
    *,
    scene_uuid: str | None,
    variant: str = SCENE_VARIANT_DEFAULT,
    clipgt_dirname: str,
) -> Path:
    """Extract a local scene archive into the WebRTC scene layout."""
    if scene_uuid is None:
        return scene_dir

    scene_uuid = scene_uuid.strip()
    assert scene_uuid, "scene_uuid must be non-empty when provided."
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene_dir does not exist: {scene_dir}")

    suffix = scene_variant_suffix(variant)
    expected_names = (
        f"clipgt-{scene_uuid}{suffix}.usdz",
        f"{scene_uuid}{suffix}.usdz",
    )
    archive_path = _choose_existing_asset(scene_dir, exact_name=expected_names[0]) or (
        _choose_existing_asset(scene_dir, exact_name=expected_names[1])
    )
    if archive_path is None:
        archive_path = _choose_existing_asset(
            scene_dir,
            fallback_prefixes=(
                f"clipgt-{scene_uuid}{suffix}",
                f"{scene_uuid}{suffix}",
                f"clipgt-{scene_uuid}",
                scene_uuid,
            ),
            allowed_suffixes={".usdz"},
            preferred_stems=(
                f"clipgt-{scene_uuid}{suffix}",
                f"{scene_uuid}{suffix}",
                f"clipgt-{scene_uuid}",
                scene_uuid,
            ),
        )
    if archive_path is None:
        raise FileNotFoundError(
            "scene_uuid is set but no local USDZ archive was found in "
            f"{scene_dir}. Expected one of: {', '.join(expected_names)}."
        )

    normalized_scene_dir = scene_dir / f"{scene_uuid}{suffix}"
    _safe_extract_zip(archive_path, normalized_scene_dir / clipgt_dirname)
    return normalized_scene_dir


def ensure_hf_scene_synced(
    scene_uuid: str,
    *,
    variant: str = SCENE_VARIANT_DEFAULT,
    clipgt_dirname: str = SCENE_CLIPGT_DIRNAME,
) -> Path:
    """Download and extract a Hugging Face scene into the WebRTC cache."""
    scene_uuid = scene_uuid.strip()
    assert scene_uuid, "scene_uuid must be set."
    suffix = scene_variant_suffix(variant)
    cache_root = scenes_cache_root()
    scene_dir = cache_root / f"{scene_uuid}{suffix}"
    lock_path = cache_root / ".locks" / f"{scene_uuid}{suffix}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path)):
        archive_path = hf_hub_download_scene(scene_uuid, variant)
        _safe_extract_zip(archive_path, scene_dir / clipgt_dirname)

    logger.info(
        "Synced Omnidreams WebRTC scene {} (variant {}) from Hugging Face ({}) to {}",
        scene_uuid,
        variant,
        hf_scenes_repo_id(),
        scene_dir,
    )
    return scene_dir


def _link_or_copy_file(source: Path, target: Path) -> None:
    try:
        os.symlink(source, target)
        return
    except OSError:
        pass

    try:
        os.link(source, target)
        return
    except OSError:
        shutil.copy2(source, target)


def prepare_clipgt_dir(
    clipgt_dir: Path,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Normalize supported ClipGT parquet layouts for the existing loader."""

    def has_prefixed_parquets(path: Path) -> bool:
        return any(path.glob("*.calibration_estimate.parquet"))

    def has_unprefixed_parquets(path: Path) -> bool:
        return (path / "calibration_estimate.parquet").exists()

    if has_prefixed_parquets(clipgt_dir):
        return clipgt_dir, None

    parquet_source_dir: Path | None = None
    if has_unprefixed_parquets(clipgt_dir):
        parquet_source_dir = clipgt_dir
    else:
        for candidate in (child for child in clipgt_dir.iterdir() if child.is_dir()):
            if has_prefixed_parquets(candidate):
                return candidate, None
            if has_unprefixed_parquets(candidate):
                parquet_source_dir = candidate
                break

    if parquet_source_dir is None:
        return clipgt_dir, None

    temp_dir = tempfile.TemporaryDirectory(prefix="omnidreams-clipgt-")
    staged = Path(temp_dir.name)
    for source in parquet_source_dir.glob("*.parquet"):
        target = staged / f"clip.{source.name}"
        _link_or_copy_file(source.resolve(), target)
    return staged, temp_dir
