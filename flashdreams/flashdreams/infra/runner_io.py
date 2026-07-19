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

"""Runner-facing image, video, prompt, and stats I/O helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import torch

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
"""Image filename suffixes treated as still images by runner helpers."""

DEFAULT_RUNNER_INSTALL_HINT = (
    "Install the runner extras: pip install 'flashdreams[runners]'."
)
"""Default install hint for optional runner I/O dependencies."""

ResizeInterpolation: TypeAlias = Literal[
    "default", "nearest", "linear", "area", "cubic", "lanczos4"
]
"""OpenCV resize interpolation names accepted by runner image/video helpers."""

VideoTensorLayout: TypeAlias = Literal["thwc", "tchw", "bcthw"]
"""Tensor layouts accepted by ``write_video_tensor``."""

InputAssetValidator: TypeAlias = Callable[[Path], object]
"""Validator signature for runner input assets downloaded to local cache."""


def ensure_output_dir(output_dir: Path) -> Path:
    """Create and return a runner output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def runner_artifact_path(output_dir: Path, runner_name: str, suffix: str) -> Path:
    """Return the conventional artifact path for a runner output."""
    return output_dir / f"{runner_name}.{suffix.lstrip('.')}"


def runner_stats_path(output_dir: Path, runner_name: str) -> Path:
    """Return the conventional per-runner stats JSON path."""
    return output_dir / f"stats_{runner_name}.json"


def write_runner_stats(
    output_dir: Path,
    runner_name: str,
    stats: Any,
    *,
    indent: int = 2,
) -> Path:
    """Write runner stats JSON and return the output path."""
    path = runner_stats_path(output_dir, runner_name)
    path.write_text(json.dumps(stats, indent=indent))
    return path


def resolve_prompt_value(value: str | Path) -> str:
    """Resolve an inline prompt or first non-empty line of a prompt file."""
    if isinstance(value, Path):
        lines = [ln.strip() for ln in value.read_text().splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"prompt file {value} has no non-empty lines")
        return lines[0]
    if not value:
        raise ValueError("--prompt must be a non-empty string or a path to a .txt file")
    return value


def resolve_input_path(
    value: str | Path,
    *,
    cache_dir: Path,
    filename: str | None = None,
    validator: InputAssetValidator | None = None,
) -> Path:
    """Resolve a local input path or download an HTTP(S) asset into cache."""
    if isinstance(value, Path):
        return value
    if not value.startswith(("http://", "https://")):
        return Path(value)
    return _download_to_cache(
        value,
        cache_dir=cache_dir,
        filename=filename,
        validator=validator,
    )


def _download_to_cache(
    url: str,
    *,
    cache_dir: Path,
    filename: str | None,
    validator: InputAssetValidator | None,
) -> Path:
    """Download an asset through the core downloader without importing it eagerly."""
    from flashdreams.core.io.download import download_to_cache  # noqa: PLC0415

    return download_to_cache(
        url,
        cache_dir=cache_dir,
        filename=filename,
        validator=validator,
    )


def read_image_rgb(
    path: str | Path,
    *,
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT,
) -> np.ndarray:
    """Read an RGB image as ``[H, W, 3]``."""
    media = _import_mediapy("Loading images", install_hint=install_hint)
    return media.read_image(str(path))[..., :3]


def read_video_rgb(
    path: str | Path,
    *,
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT,
) -> np.ndarray:
    """Read an RGB video as ``[T, H, W, 3]``."""
    media = _import_mediapy("Loading videos", install_hint=install_hint)
    return media.read_video(str(path))[..., :3]


def read_video_fps(
    path: str | Path,
    *,
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT,
) -> float:
    """Read a video's frame rate from ``mediapy`` metadata."""
    media = _import_mediapy("Probing video metadata", install_hint=install_hint)
    return float(media.VideoMetadata.from_path(str(path)).fps)


def read_first_frame_rgb(
    path: str | Path,
    *,
    image_suffixes: set[str] | frozenset[str] = IMAGE_SUFFIXES,
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT,
) -> np.ndarray:
    """Read an image or the first frame of a video as ``[H, W, 3]``."""
    path = Path(path)
    media = _import_mediapy("Loading first-frame assets", install_hint=install_hint)
    if path.suffix.lower() in image_suffixes:
        return media.read_image(str(path))[..., :3]
    video = media.read_video(str(path))
    if video.shape[0] == 0:
        raise ValueError(f"video has no frames: {path}")
    return video[0, ..., :3]


def resize_rgb_image(
    image: np.ndarray,
    *,
    pixel_height: int,
    pixel_width: int,
    interpolation: ResizeInterpolation = "default",
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT,
) -> np.ndarray:
    """Resize an RGB image with OpenCV's ``(width, height)`` convention."""
    cv2 = _import_cv2("Resizing images", install_hint=install_hint)
    resize_kwargs = _cv2_resize_kwargs(cv2, interpolation)
    return cv2.resize(image, (pixel_width, pixel_height), **resize_kwargs)


def resize_rgb_video(
    video: np.ndarray,
    *,
    pixel_height: int,
    pixel_width: int,
    interpolation: ResizeInterpolation = "default",
    skip_if_matching: bool = True,
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT,
) -> np.ndarray:
    """Resize an RGB video frame-by-frame with OpenCV."""
    if skip_if_matching and video.shape[1:3] == (pixel_height, pixel_width):
        return video
    return np.stack(
        [
            resize_rgb_image(
                frame,
                pixel_height=pixel_height,
                pixel_width=pixel_width,
                interpolation=interpolation,
                install_hint=install_hint,
            )
            for frame in video
        ],
        axis=0,
    )


def rgb_image_to_normalized_tensor(
    image: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert an RGB image to ``[1, C, H, W]`` in ``[-1, 1]``."""
    tensor = torch.from_numpy(image).to(device=device, dtype=dtype) / 127.5 - 1.0
    return tensor.permute(2, 0, 1).unsqueeze(0)


def rgb_video_to_normalized_tensor(
    video: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert an RGB video to ``[T, C, H, W]`` in ``[-1, 1]``."""
    tensor = torch.from_numpy(video).to(device=device, dtype=dtype) / 127.5 - 1.0
    return tensor.permute(0, 3, 1, 2)


def load_first_frame_tensor(
    path: str | Path,
    *,
    pixel_height: int,
    pixel_width: int,
    device: torch.device,
    dtype: torch.dtype,
    allow_video: bool = False,
    interpolation: ResizeInterpolation = "default",
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT,
) -> torch.Tensor:
    """Load, resize, and normalize a first-frame asset."""
    if allow_video:
        image = read_first_frame_rgb(path, install_hint=install_hint)
    else:
        image = read_image_rgb(path, install_hint=install_hint)
    image = resize_rgb_image(
        image,
        pixel_height=pixel_height,
        pixel_width=pixel_width,
        interpolation=interpolation,
        install_hint=install_hint,
    )
    return rgb_image_to_normalized_tensor(image, device=device, dtype=dtype)


def load_video_tensor(
    path: str | Path,
    *,
    pixel_height: int,
    pixel_width: int,
    device: torch.device,
    dtype: torch.dtype,
    interpolation: ResizeInterpolation = "default",
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT,
) -> torch.Tensor:
    """Load, resize, and normalize an RGB video."""
    video = read_video_rgb(path, install_hint=install_hint)
    video = resize_rgb_video(
        video,
        pixel_height=pixel_height,
        pixel_width=pixel_width,
        interpolation=interpolation,
        install_hint=install_hint,
    )
    return rgb_video_to_normalized_tensor(video, device=device, dtype=dtype)


def video_tensor_to_uint8(
    video: torch.Tensor,
    *,
    layout: VideoTensorLayout,
) -> np.ndarray:
    """Convert a ``[-1, 1]`` video tensor to uint8 ``[T, H, W, C]`` frames."""
    if layout == "thwc":
        canvas = video
    elif layout == "tchw":
        canvas = video.permute(0, 2, 3, 1)
    elif layout == "bcthw":
        if video.shape[0] != 1:
            raise ValueError(
                "layout='bcthw' expects a single batch element; "
                f"got {tuple(video.shape)}"
            )
        canvas = video[0].permute(1, 2, 3, 0)
    else:
        raise ValueError(f"Unsupported video tensor layout: {layout!r}")

    arr = (canvas.detach().cpu().float().numpy() + 1.0) / 2.0
    return (arr * 255).clip(0, 255).astype("uint8")


def write_video_tensor(
    video: torch.Tensor,
    path: str | Path,
    *,
    fps: int | float,
    layout: VideoTensorLayout,
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT,
) -> Path:
    """Write a ``[-1, 1]`` video tensor as an MP4 and return the path."""
    media = _import_mediapy("Writing videos", install_hint=install_hint)
    path = Path(path)
    media.write_video(str(path), video_tensor_to_uint8(video, layout=layout), fps=fps)
    return path


def _import_mediapy(action: str, *, install_hint: str) -> Any:
    """Import ``mediapy`` with a runner-specific dependency hint."""
    try:
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(f"{action} needs mediapy. {install_hint}") from exc
    return media


def _import_cv2(action: str, *, install_hint: str) -> Any:
    """Import ``cv2`` with a runner-specific dependency hint."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(f"{action} needs opencv. {install_hint}") from exc
    return cv2


def _cv2_resize_kwargs(cv2: Any, interpolation: ResizeInterpolation) -> dict[str, int]:
    """Return OpenCV resize kwargs for the requested interpolation."""
    if interpolation == "default":
        return {}
    interpolation_values = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    return {"interpolation": interpolation_values[interpolation]}
