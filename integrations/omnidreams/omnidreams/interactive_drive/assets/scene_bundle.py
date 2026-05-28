# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from pathlib import Path

from omnidreams.interactive_drive.scene_loader import (
    load_scene_bundle as load_scene_bundle,
)
from omnidreams.scenes import variant_from_stem


def canonicalize_camera_name(name: str) -> str:
    return name.strip().lower().replace(":", "_").replace("-", "_")


def _discover_prompts(scene_root: Path) -> dict[str, str]:
    # Same underscore-required convention as
    # ``scene_loader._discover_prompts`` (used for zipped USDZ scenes);
    # this helper covers the unpacked-directory case.
    prompts: dict[str, str] = {}
    for prompt_path in sorted(scene_root.glob("prompt*.txt")):
        variant = variant_from_stem(prompt_path.stem, "prompt")
        if variant is None:
            continue
        prompts[variant] = prompt_path.read_text(encoding="utf-8").strip()
    if "default" not in prompts and prompts:
        first_key = sorted(prompts.keys())[0]
        prompts["default"] = prompts[first_key]
    return prompts


def _discover_first_frames(scene_root: Path) -> dict[str, Path]:
    first_frames: dict[str, Path] = {}
    for image_path in sorted(scene_root.glob("first_image*.png")):
        variant = variant_from_stem(image_path.stem, "first_image")
        if variant is None:
            continue
        first_frames[variant] = image_path
    if "default" not in first_frames and first_frames:
        first_key = sorted(first_frames.keys())[0]
        first_frames["default"] = first_frames[first_key]
    return first_frames


__all__ = [
    "canonicalize_camera_name",
    "load_scene_bundle",
]
