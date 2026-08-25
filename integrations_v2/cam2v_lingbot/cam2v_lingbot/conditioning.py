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
_TEMPORAL_COMPRESSION_RATIO = 4
_TRANSFORMER_CHUNK_FRAMES = 3


def resolve_lingbot_conditioning(values: Mapping[str, Any]) -> Cam2VConditioning:
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
        if not prompt and prompt_path is None and example_idx in _EXAMPLE_PROMPT_INDICES:
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
    if world_scale is None:
        poses_path = _require_existing_path(pose_path, label="pose_path")
        world_scale = _infer_world_scale(poses_path)

    return Cam2VConditioning(
        prompt=prompt,
        first_frame_path=first_frame_path,
        base_intrinsics=_load_base_intrinsics(
            intrinsics_path,
            pixel_height=int(values.get("pixel_height", _DEFAULT_PIXEL_HEIGHT)),
            pixel_width=int(values.get("pixel_width", _DEFAULT_PIXEL_WIDTH)),
        ),
        world_scale=world_scale,
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
            "Lingbot intrinsics must have shape [T, 4], got "
            f"{tuple(intrinsics.shape)}."
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


def _infer_world_scale(path: Path) -> float:
    """Return the legacy pose normalizer without constructing a camera trace."""
    poses = np.asarray(np.load(path), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"Lingbot poses must have shape [T, 4, 4], got {tuple(poses.shape)}."
        )

    raw_frame_count = poses.shape[0]
    compatible_frame_count = (
        (raw_frame_count - 1) // _TEMPORAL_COMPRESSION_RATIO
    ) * _TEMPORAL_COMPRESSION_RATIO + 1
    encoded_frame_count = (
        (compatible_frame_count - 1) // _TEMPORAL_COMPRESSION_RATIO
    ) + 1
    if encoded_frame_count < _TRANSFORMER_CHUNK_FRAMES:
        minimum = (
            (_TRANSFORMER_CHUNK_FRAMES - 1) * _TEMPORAL_COMPRESSION_RATIO + 1
        )
        raise ValueError(
            f"Expected at least {minimum} poses to infer world scale, "
            f"got {raw_frame_count}."
        )
    encoded_frame_count -= encoded_frame_count % _TRANSFORMER_CHUNK_FRAMES

    source_indices = np.arange(compatible_frame_count, dtype=np.float64)
    target_indices = np.linspace(
        0,
        compatible_frame_count - 1,
        encoded_frame_count,
    )
    translations = poses[:compatible_frame_count, :3, 3]
    encoded_translations = np.stack(
        [
            np.interp(target_indices, source_indices, translations[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )
    step_distances = np.linalg.norm(np.diff(encoded_translations, axis=0), axis=1)
    return float(step_distances.max(initial=0.0))


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
