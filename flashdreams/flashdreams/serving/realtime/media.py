# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral realtime media conversion helpers."""

from __future__ import annotations

import importlib
import io
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import torch

FrameLayout = Literal["hwc", "chw", "thwc", "tchw", "bvtchw"]
ValueRange = Literal["minus_one_one", "zero_one", "uint8"]


def _as_numpy(value: object, *, sync_device: bool) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise TypeError("Expected a numpy array or torch tensor.") from exc

    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a numpy array or torch tensor.")

    tensor = value.detach()
    if tensor.device.type != "cpu":
        if not sync_device:
            raise ValueError(
                "Converting a non-CPU tensor requires sync_device=True so the "
                "host/device synchronization is explicit at the call site."
            )
        tensor = tensor.cpu()
    if tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.numpy()


def _scale_rgb(array: np.ndarray, *, value_range: ValueRange) -> np.ndarray:
    if value_range not in ("minus_one_one", "zero_one", "uint8"):
        raise ValueError(f"Unsupported value_range={value_range!r}.")
    if array.dtype == np.uint8:
        if value_range != "uint8":
            raise ValueError("uint8 inputs require value_range='uint8'.")
        return np.ascontiguousarray(array)

    values = array.astype(np.float32, copy=False)
    if value_range == "minus_one_one":
        values = (values + 1.0) / 2.0 * 255.0
    elif value_range == "zero_one":
        values = values * 255.0
    return np.ascontiguousarray(values.clip(0, 255).astype(np.uint8))


def rgb_frame_to_uint8(
    frame: object,
    *,
    layout: Literal["hwc", "chw"] = "hwc",
    value_range: ValueRange = "uint8",
    sync_device: bool = True,
) -> np.ndarray:
    """Convert one RGB frame to contiguous ``HWC`` uint8 host memory."""
    array = _as_numpy(frame, sync_device=sync_device)
    if layout == "hwc":
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"Expected HWC RGB frame with shape [H, W, 3], got {array.shape}"
            )
        return _scale_rgb(array, value_range=value_range)
    if layout == "chw":
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError(
                f"Expected CHW RGB frame with shape [3, H, W], got {array.shape}"
            )
        return _scale_rgb(np.transpose(array, (1, 2, 0)), value_range=value_range)
    raise ValueError(f"Unsupported layout={layout!r}.")


def rgb_array_to_uint8_frames(
    data: object,
    *,
    layout: FrameLayout,
    value_range: ValueRange = "minus_one_one",
    sync_device: bool = True,
) -> list[np.ndarray]:
    """Convert a tensor/array video chunk to ``HWC`` uint8 RGB frames."""
    array = _as_numpy(data, sync_device=sync_device)
    if layout == "hwc" or layout == "chw":
        return [
            rgb_frame_to_uint8(
                array,
                layout=layout,
                value_range=value_range,
                sync_device=sync_device,
            )
        ]
    if layout == "thwc":
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(
                f"Expected THWC RGB chunk with shape [T, H, W, 3], got {array.shape}"
            )
        frames = array
    elif layout == "tchw":
        if array.ndim != 4 or array.shape[1] != 3:
            raise ValueError(
                f"Expected TCHW RGB chunk with shape [T, 3, H, W], got {array.shape}"
            )
        frames = np.transpose(array, (0, 2, 3, 1))
    elif layout == "bvtchw":
        if (
            array.ndim != 6
            or array.shape[0] != 1
            or array.shape[1] != 1
            or array.shape[3] != 3
        ):
            raise ValueError(
                "Expected single-batch single-view video chunk "
                f"[1, 1, T, 3, H, W], got {array.shape}"
            )
        frames = np.transpose(array[0, 0], (0, 2, 3, 1))
    else:
        raise ValueError(f"Unsupported layout={layout!r}.")

    return [_scale_rgb(frame, value_range=value_range) for frame in frames]


def tensor_chunk_to_rgb_frames(
    video_chunk: torch.Tensor,
    *,
    sync_device: bool = True,
) -> list[np.ndarray]:
    """Convert common model output tensor layouts to RGB uint8 frames."""
    value_range: ValueRange = (
        "minus_one_one" if video_chunk.is_floating_point() else "uint8"
    )
    if video_chunk.ndim == 4:
        if video_chunk.shape[-1] == 3 and video_chunk.shape[1] != 3:
            return rgb_array_to_uint8_frames(
                video_chunk,
                layout="thwc",
                value_range=value_range,
                sync_device=sync_device,
            )
        if video_chunk.shape[1] != 3:
            raise ValueError(
                "Expected video chunk [T, C, H, W] or [T, H, W, C] with 3 RGB "
                f"channels, got {tuple(video_chunk.shape)}"
            )
        return rgb_array_to_uint8_frames(
            video_chunk,
            layout="tchw",
            value_range=value_range,
            sync_device=sync_device,
        )
    if video_chunk.ndim == 6:
        return rgb_array_to_uint8_frames(
            video_chunk,
            layout="bvtchw",
            value_range=value_range,
            sync_device=sync_device,
        )
    raise ValueError(
        "Expected video chunk [T, C, H, W], [T, H, W, C], "
        "or [1, 1, T, 3, H, W], "
        f"got {tuple(video_chunk.shape)}"
    )


def encode_rgb_frame_to_jpeg(
    frame: object,
    *,
    quality: int = 85,
    layout: Literal["hwc", "chw"] = "hwc",
    value_range: ValueRange = "uint8",
    sync_device: bool = True,
) -> bytes:
    """JPEG-encode one RGB frame.

    Pillow is imported lazily so importing realtime serving modules does not
    force image-codec dependencies into non-JPEG transports.
    """
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")

    image_module = importlib.import_module("PIL.Image")
    image_from_array = getattr(image_module, "fromarray")
    rgb_uint8 = rgb_frame_to_uint8(
        frame,
        layout=layout,
        value_range=value_range,
        sync_device=sync_device,
    )
    output = io.BytesIO()
    image_from_array(rgb_uint8).save(output, format="JPEG", quality=quality)
    return output.getvalue()
