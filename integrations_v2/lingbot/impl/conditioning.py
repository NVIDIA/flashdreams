# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve Lingbot assets directly into the shared Cam2V contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cam2v import Cam2VConditioning

from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.core.io.download import download_to_cache
from lingbot.impl.encoder.utils import preprocess_example_poses

_EXAMPLE_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/Robbyant/lingbot-world-v2/main/examples"
)
_EXAMPLE_DATA_DIR = default_flashdreams_cache_dir() / "example_data/lingbot_world"
_EXAMPLE_DATA_INDICES = frozenset(range(6))
_EXAMPLE_PROMPT_INDICES = frozenset({0, 1, 2, 5})
_INTRINSICS_REFERENCE_HEIGHT = 480
_INTRINSICS_REFERENCE_WIDTH = 832
_DEFAULT_PIXEL_HEIGHT = 464
_DEFAULT_PIXEL_WIDTH = 832


def resolve_lingbot_conditioning(
    values: Mapping[str, Any],
    *,
    transformer_len_t: int = 3,
) -> Cam2VConditioning:
    """Resolve application arguments without the legacy input-mapping runtime."""
    example_idx = int(values.get("example_idx", 0))
    if example_idx not in _EXAMPLE_DATA_INDICES:
        raise ValueError(
            f"Lingbot example_idx must be one of {sorted(_EXAMPLE_DATA_INDICES)}."
        )

    image_path = _optional_path(values.get("image_path"))
    pose_path = _optional_path(values.get("pose_path"))
    intrinsic_path = _optional_path(values.get("intrinsic_path"))
    prompt_path = _optional_path(values.get("prompt_path"))
    prompt = _nonempty_text(values.get("prompt"))
    world_scale = _optional_float(values.get("world_scale"))

    if _as_bool(values.get("example_data", False)):
        example_dir = _ensure_example_data(example_idx)
        image_path = image_path or example_dir / "image.jpg"
        pose_path = pose_path or example_dir / "poses.npy"
        intrinsic_path = intrinsic_path or example_dir / "intrinsics.npy"
        if (
            not prompt
            and prompt_path is None
            and example_idx in _EXAMPLE_PROMPT_INDICES
        ):
            prompt_path = example_dir / "prompt.txt"

    first_frame_path = _require_existing_path(image_path, label="image_path")
    intrinsics_path = _require_existing_path(
        intrinsic_path,
        label="intrinsic_path",
    )
    if not prompt and prompt_path is not None:
        prompt = _read_first_line(
            _require_existing_path(prompt_path, label="prompt_path")
        )
    camera_poses = None
    if pose_path is not None:
        poses_path = _require_existing_path(pose_path, label="pose_path")
        camera_poses, inferred_world_scale = preprocess_example_poses(
            np.asarray(np.load(poses_path)),
            transformer_len_t=transformer_len_t,
        )
        if world_scale is None:
            world_scale = inferred_world_scale
    elif world_scale is None:
        raise ValueError("Lingbot Cam2V requires pose_path to infer world_scale.")

    return Cam2VConditioning(
        prompt=prompt,
        first_frame_path=first_frame_path,
        base_intrinsics=_load_base_intrinsics(
            intrinsics_path,
            pixel_height=int(values.get("pixel_height", _DEFAULT_PIXEL_HEIGHT)),
            pixel_width=int(values.get("pixel_width", _DEFAULT_PIXEL_WIDTH)),
        ),
        world_scale=world_scale,
        camera_poses=(
            None
            if camera_poses is None
            else torch.from_numpy(np.ascontiguousarray(camera_poses))
        ),
    )


def _ensure_example_data(example_idx: int) -> Path:
    dirname = f"{example_idx:02d}"
    cache_dir = _EXAMPLE_DATA_DIR / dirname
    filenames = ["image.jpg", "poses.npy", "intrinsics.npy"]
    if example_idx in _EXAMPLE_PROMPT_INDICES:
        filenames.append("prompt.txt")

    distributed = torch.distributed.is_initialized()
    if not distributed or torch.distributed.get_rank() == 0:
        for filename in filenames:
            download_to_cache(
                f"{_EXAMPLE_DATA_BASE_URL}/{dirname}/{filename}",
                cache_dir=cache_dir,
                filename=filename,
            )
    if distributed:
        torch.distributed.barrier()
    return cache_dir


def _load_base_intrinsics(
    path: Path,
    *,
    pixel_height: int,
    pixel_width: int,
) -> torch.Tensor:
    intrinsics = np.asarray(np.load(path), dtype=np.float32)
    if intrinsics.ndim == 1:
        intrinsics = intrinsics[None, :]
    if intrinsics.ndim != 2 or intrinsics.shape[0] == 0 or intrinsics.shape[1] != 4:
        raise ValueError(
            f"Lingbot intrinsics must have shape [T, 4], got {tuple(intrinsics.shape)}."
        )
    scale = np.array(
        [
            pixel_width / _INTRINSICS_REFERENCE_WIDTH,
            pixel_height / _INTRINSICS_REFERENCE_HEIGHT,
            pixel_width / _INTRINSICS_REFERENCE_WIDTH,
            pixel_height / _INTRINSICS_REFERENCE_HEIGHT,
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(np.ascontiguousarray(intrinsics[0] * scale))


def _optional_path(value: str | Path | None) -> Path | None:
    return None if value is None or value == "" else Path(value)


def _require_existing_path(path: Path | None, *, label: str) -> Path:
    if path is None:
        raise ValueError(f"Lingbot Cam2V requires {label}.")
    if not path.exists():
        raise FileNotFoundError(f"Lingbot Cam2V missing {label}: {path}")
    return path


def _read_first_line(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return _nonempty_text(lines[0]) if lines else ""


def _nonempty_text(value: object) -> str:
    return "" if value is None else " ".join(str(value).split())


def _optional_float(value: str | int | float | None) -> float | None:
    return None if value is None or value == "" else float(value)


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


__all__ = ["resolve_lingbot_conditioning"]
