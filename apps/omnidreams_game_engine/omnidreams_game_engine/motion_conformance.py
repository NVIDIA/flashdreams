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

"""Optical-flow conformance checks for game-engine driving video."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional

_FLOW_WIDTH = 160
_FLOW_HEIGHT = 88
_MIN_COMPONENT_PX = 0.15
_STRONG_COMPONENT_PX = 0.40
_MIN_GENERATED_COMPONENT_PX = 0.10
_WEAK_RATIO = 0.20
_MIN_USABLE_PAIRS = 3


@dataclass(frozen=True)
class MotionConformanceResult:
    """Chunk-level agreement between conditioning and generated camera motion."""

    mismatched: bool
    """Whether the generated chunk confidently disagrees with conditioning."""

    axis: str
    """Dominant tested motion axis, or ``"none"`` when evidence is insufficient."""

    condition_component_px: float
    """Median signed conditioning motion on the tested axis."""

    generated_component_px: float
    """Median signed generated motion on the tested axis."""

    usable_pairs: int
    """Number of frame pairs carrying sufficient conditioning motion."""

    elapsed_ms: float
    """Wall time spent measuring the chunk."""

    def as_metrics(self) -> dict[str, float | str | bool]:
        """Return JSON-compatible diagnostic values."""
        return {
            "mismatched": self.mismatched,
            "axis": self.axis,
            "condition_component_px": self.condition_component_px,
            "generated_component_px": self.generated_component_px,
            "usable_pairs": float(self.usable_pairs),
            "elapsed_ms": self.elapsed_ms,
        }


def compare_motion(
    condition_frames: Sequence[object],
    generated_frames: Sequence[object],
    *,
    yaw_delta_rad: float,
    longitudinal_delta_m: float,
) -> MotionConformanceResult:
    """Compare dominant generated motion with the aligned condition video."""
    started_at = time.perf_counter()
    condition = _flow_signatures(condition_frames)
    generated = _flow_signatures(generated_frames)
    axis = _dominant_axis(yaw_delta_rad, longitudinal_delta_m)
    if axis == "none" or len(condition) != len(generated):
        return _result(False, "none", 0.0, 0.0, 0, started_at)

    component_index = 0 if axis == "turn" else 1
    usable = [
        index
        for index, signature in enumerate(condition)
        if abs(signature[component_index]) >= _MIN_COMPONENT_PX
    ]
    if len(usable) < _MIN_USABLE_PAIRS:
        return _result(False, "none", 0.0, 0.0, len(usable), started_at)

    condition_component = float(
        np.median([condition[index][component_index] for index in usable])
    )
    generated_component = float(
        np.median([generated[index][component_index] for index in usable])
    )
    opposite = (
        condition_component * generated_component < 0.0
        and abs(generated_component) >= _MIN_GENERATED_COMPONENT_PX
    )
    too_weak = abs(condition_component) >= _STRONG_COMPONENT_PX and abs(
        generated_component
    ) < _WEAK_RATIO * abs(condition_component)
    return _result(
        opposite or too_weak,
        axis,
        condition_component,
        generated_component,
        len(usable),
        started_at,
    )


def _dominant_axis(yaw_delta_rad: float, longitudinal_delta_m: float) -> str:
    turn_strength = abs(float(yaw_delta_rad)) / math.radians(1.0)
    longitudinal_strength = abs(float(longitudinal_delta_m)) / 0.15
    if max(turn_strength, longitudinal_strength) < 1.0:
        return "none"
    return "turn" if turn_strength >= longitudinal_strength else "longitudinal"


def _flow_signatures(frames: Sequence[object]) -> list[tuple[float, float]]:
    gray_frames = [_gray_frame(frame) for frame in frames]
    signatures: list[tuple[float, float]] = []
    for previous, current in zip(gray_frames[:-1], gray_frames[1:], strict=True):
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        flow = flow[_FLOW_HEIGHT // 3 :]
        height, width = flow.shape[:2]
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        xx -= (width - 1) * 0.5
        yy += _FLOW_HEIGHT // 3
        yy -= (_FLOW_HEIGHT - 1) * 0.5
        radius = np.sqrt(xx * xx + yy * yy)
        valid = radius > 0.15 * min(_FLOW_WIDTH, _FLOW_HEIGHT)
        horizontal = float(np.median(flow[..., 0]))
        radial = float(
            np.median(
                (flow[..., 0][valid] * xx[valid] + flow[..., 1][valid] * yy[valid])
                / radius[valid]
            )
        )
        signatures.append((horizontal, radial))
    return signatures


def _gray_frame(frame: object) -> np.ndarray:
    to_tensor = getattr(frame, "to_cuda_tensor", None)
    if callable(to_tensor):
        tensor = to_tensor()
        if torch.is_tensor(tensor):
            source_event = getattr(frame, "to_cuda_event", lambda: None)()
            if tensor.is_cuda and source_event is not None:
                torch.cuda.current_stream(tensor.device).wait_event(source_event)
            if tensor.ndim != 3 or tensor.shape[-1] not in (3, 4):
                raise ValueError(
                    f"motion frame must be HWC RGB(A), got {tuple(tensor.shape)}"
                )
            rgb = tensor[..., :3].permute(2, 0, 1).unsqueeze(0).float()
            resized = torch_functional.interpolate(
                rgb,
                size=(_FLOW_HEIGHT, _FLOW_WIDTH),
                mode="area",
            )[0]
            gray = resized[0] * 0.299 + resized[1] * 0.587 + resized[2] * 0.114
            return gray.round().clamp(0, 255).to(torch.uint8).cpu().numpy()

    value: Any = frame
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"motion frame must be HWC RGB(A), got {array.shape}")
    if array.dtype != np.uint8:
        scale = 255.0 if np.issubdtype(array.dtype, np.floating) else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    resized = cv2.resize(
        np.ascontiguousarray(array[..., :3]),
        (_FLOW_WIDTH, _FLOW_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)


def _result(
    mismatched: bool,
    axis: str,
    condition_component: float,
    generated_component: float,
    usable_pairs: int,
    started_at: float,
) -> MotionConformanceResult:
    return MotionConformanceResult(
        mismatched=mismatched,
        axis=axis,
        condition_component_px=condition_component,
        generated_component_px=generated_component,
        usable_pairs=usable_pairs,
        elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
    )


__all__ = ["MotionConformanceResult", "compare_motion"]
