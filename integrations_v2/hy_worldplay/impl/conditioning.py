# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HY-WorldPlay input resolution for the reusable Cam2V application."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cam2v import Cam2VConditioning

from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.core.io.download import download_to_cache

DEFAULT_PROMPT = (
    "First-person view walking around ancient Athens, with Greek "
    "architecture and marble structures"
)
"""Default HY-WorldPlay text prompt."""

EXAMPLE_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/Tencent-Hunyuan/HY-WorldPlay/main/assets"
)
"""Base URL for the upstream sample assets."""

EXAMPLE_DATA_DIR_LOCAL = default_flashdreams_cache_dir() / "example_data/hy_worldplay"
"""Local cache directory for the sample assets."""

_EXAMPLE_IMAGE_FILENAME = "test.png"
_REFERENCE_WIDTH = 1920
_REFERENCE_HEIGHT = 1080
_REFERENCE_INTRINSICS = np.array(
    [969.6969696969696, 969.6969696969696, 960.0, 540.0],
    dtype=np.float32,
)
_DEFAULT_WORLD_SCALE = 2.5
"""Normalize Cam2V's 0.2-unit latent-frame stride to HY's 0.08-unit stride."""


def resolve_hy_worldplay_conditioning(
    values: Mapping[str, Any],
) -> Cam2VConditioning:
    """Resolve first-frame, prompt, calibration, and translation inputs."""
    image_path = _optional_path(values.get("image_path"))
    prompt_path = _optional_path(values.get("prompt_path"))
    prompt = _nonempty_text(values.get("prompt"))

    if _as_bool(values.get("example_data", False)):
        example_idx = int(values.get("example_idx", 0))
        if example_idx != 0:
            raise ValueError("HY-WorldPlay only provides example_idx=0.")
        image_path = image_path or _ensure_example_image()

    first_frame_path = _require_existing_path(image_path, label="image_path")
    if not prompt and prompt_path is not None:
        prompt = _read_first_line(
            _require_existing_path(prompt_path, label="prompt_path")
        )
    if not prompt:
        prompt = DEFAULT_PROMPT

    pixel_height = int(values.get("pixel_height", 704))
    pixel_width = int(values.get("pixel_width", 1280))
    intrinsic_path = _optional_path(values.get("intrinsic_path"))
    intrinsics = (
        _default_intrinsics(pixel_height=pixel_height, pixel_width=pixel_width)
        if intrinsic_path is None
        else _load_intrinsics(
            _require_existing_path(intrinsic_path, label="intrinsic_path")
        )
    )
    world_scale_value = values.get("world_scale")
    world_scale = (
        _DEFAULT_WORLD_SCALE
        if world_scale_value is None or world_scale_value == ""
        else float(world_scale_value)
    )
    if world_scale <= 0:
        raise ValueError("HY-WorldPlay world_scale must be > 0.")

    return Cam2VConditioning(
        prompt=prompt,
        first_frame_path=first_frame_path,
        base_intrinsics=intrinsics,
        world_scale=world_scale,
    )


def _ensure_example_image() -> Path:
    distributed = torch.distributed.is_initialized()
    if not distributed or torch.distributed.get_rank() == 0:
        download_to_cache(
            f"{EXAMPLE_DATA_BASE_URL}/img/{_EXAMPLE_IMAGE_FILENAME}",
            cache_dir=EXAMPLE_DATA_DIR_LOCAL,
            filename=_EXAMPLE_IMAGE_FILENAME,
        )
    if distributed:
        torch.distributed.barrier()
    return EXAMPLE_DATA_DIR_LOCAL / _EXAMPLE_IMAGE_FILENAME


def _default_intrinsics(*, pixel_height: int, pixel_width: int) -> torch.Tensor:
    scale = np.array(
        [
            pixel_width / _REFERENCE_WIDTH,
            pixel_height / _REFERENCE_HEIGHT,
            pixel_width / _REFERENCE_WIDTH,
            pixel_height / _REFERENCE_HEIGHT,
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(_REFERENCE_INTRINSICS * scale)


def _load_intrinsics(path: Path) -> torch.Tensor:
    intrinsics = np.asarray(np.load(path), dtype=np.float32)
    if intrinsics.ndim == 2 and intrinsics.shape == (3, 3):
        intrinsics = intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]]
    elif intrinsics.ndim == 2 and intrinsics.shape[0] > 0 and intrinsics.shape[1] == 4:
        intrinsics = intrinsics[0]
    if intrinsics.shape != (4,):
        raise ValueError(
            "HY-WorldPlay intrinsics must have shape [4], [T, 4], or [3, 3], "
            f"got {tuple(intrinsics.shape)}."
        )
    return torch.from_numpy(np.ascontiguousarray(intrinsics))


def _optional_path(value: str | Path | None) -> Path | None:
    return None if value is None or value == "" else Path(value)


def _require_existing_path(path: Path | None, *, label: str) -> Path:
    if path is None:
        raise ValueError(f"HY-WorldPlay Cam2V requires {label}.")
    if not path.exists():
        raise FileNotFoundError(f"HY-WorldPlay Cam2V missing {label}: {path}")
    return path


def _read_first_line(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return _nonempty_text(lines[0]) if lines else ""


def _nonempty_text(value: object) -> str:
    return "" if value is None else " ".join(str(value).split())


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


__all__ = ["resolve_hy_worldplay_conditioning"]
