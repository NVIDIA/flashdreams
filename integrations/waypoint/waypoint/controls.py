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

"""Waypoint's public per-action control representation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from waypoint.spec import WAYPOINT_1_5, WaypointModelSpec


def load_controls_from_file(path: Path) -> tuple["WaypointControl", ...]:
    """Load a versioned JSON sequence of per-action controller inputs.

    The file format is ``{"schema_version": 1, "actions": [...]}``.
    Each action may specify ``buttons`` (integer array), ``mouse_dx``,
    ``mouse_dy``, and ``scroll_wheel``; omitted values use the neutral control.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"control timeline does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(
            f"control timeline is not valid JSON: {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("control timeline root must be an object")
    unexpected = set(payload) - {"schema_version", "actions"}
    if unexpected:
        raise ValueError(
            f"control timeline has unsupported fields: {sorted(unexpected)}"
        )
    if payload.get("schema_version") != 1:
        raise ValueError("control timeline schema_version must be 1")
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("control timeline actions must be a non-empty array")
    return tuple(
        _parse_control_action(action, index) for index, action in enumerate(actions)
    )


def _parse_control_action(payload: Any, index: int) -> "WaypointControl":
    if not isinstance(payload, dict):
        raise ValueError(f"control action {index} must be an object")
    allowed = {"buttons", "mouse_dx", "mouse_dy", "scroll_wheel"}
    unexpected = set(payload) - allowed
    if unexpected:
        raise ValueError(
            f"control action {index} has unsupported fields: {sorted(unexpected)}"
        )
    buttons_value = payload.get("buttons", [])
    if not isinstance(buttons_value, list) or any(
        not isinstance(button, int) or isinstance(button, bool)
        for button in buttons_value
    ):
        raise ValueError(f"control action {index} buttons must be an integer array")
    buttons = cast(list[int], buttons_value)
    invalid_buttons = sorted(
        button for button in buttons if not 0 <= button < WAYPOINT_1_5.n_buttons
    )
    if invalid_buttons:
        raise ValueError(
            f"control action {index} button IDs must be in "
            f"[0, {WAYPOINT_1_5.n_buttons}), got {invalid_buttons}"
        )
    mouse_dx = _finite_number(payload.get("mouse_dx", 0.0), "mouse_dx", index)
    mouse_dy = _finite_number(payload.get("mouse_dy", 0.0), "mouse_dy", index)
    scroll_wheel = payload.get("scroll_wheel", 0)
    if (
        not isinstance(scroll_wheel, int)
        or isinstance(scroll_wheel, bool)
        or scroll_wheel not in (-1, 0, 1)
    ):
        raise ValueError(
            f"control action {index} scroll_wheel must be one of -1, 0, or 1"
        )
    return WaypointControl(
        buttons=frozenset(buttons),
        mouse_dx=mouse_dx,
        mouse_dy=mouse_dy,
        scroll_wheel=scroll_wheel,
    )


def _finite_number(value: Any, name: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"control action {index} {name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"control action {index} {name} must be a finite number")
    return value


@dataclass(frozen=True, kw_only=True)
class WaypointControl:
    """Keyboard and mouse state that conditions one autoregressive action."""

    buttons: frozenset[int] = field(default_factory=frozenset)
    """Pressed button identifiers in the fixed 256-entry control vocabulary."""
    mouse_dx: float = 0.0
    """Mouse displacement along the horizontal axis."""
    mouse_dy: float = 0.0
    """Mouse displacement along the vertical axis."""
    scroll_wheel: int = 0
    """Ternary wheel direction: ``-1``, ``0``, or ``1``."""


def make_control_context(
    control: WaypointControl,
    *,
    frame_index: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    spec: WaypointModelSpec = WAYPOINT_1_5,
) -> dict[str, Tensor]:
    """Convert one public control event into model-ready per-action tensors.

    Args:
        control: Keyboard and mouse event for one autoregressive action.
        frame_index: Zero-based latent-frame index in the rollout.
        batch_size: Number of identical control contexts to construct.
        dtype: Floating-point dtype for continuous control tensors.
        device: Target device; ``None`` keeps tensors on the current default device.
        spec: Static checkpoint contract that defines control dimensions.

    Returns:
        Pre-network button, mouse, scroll, and latent-frame timestamp tensors.

    Raises:
        ValueError: A frame index, batch size, scroll value, or button ID is invalid.
    """
    if frame_index < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame_index}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if control.scroll_wheel not in (-1, 0, 1):
        raise ValueError(
            f"scroll_wheel must be one of -1, 0, or 1, got {control.scroll_wheel}"
        )

    invalid_buttons = sorted(
        button for button in control.buttons if not 0 <= button < spec.n_buttons
    )
    if invalid_buttons:
        raise ValueError(
            f"button IDs must be in [0, {spec.n_buttons}), got {invalid_buttons}"
        )

    resolved_device = torch.device(device) if device is not None else None
    buttons = torch.zeros(
        (batch_size, 1, spec.n_buttons), dtype=dtype, device=resolved_device
    )
    if control.buttons:
        buttons[..., sorted(control.buttons)] = 1

    mouse = (
        torch.tensor(
            (control.mouse_dx, control.mouse_dy), dtype=dtype, device=resolved_device
        )
        .view(1, 1, 2)
        .expand(batch_size, -1, -1)
        .clone()
    )
    scroll = torch.full(
        (batch_size, 1, 1), control.scroll_wheel, dtype=dtype, device=resolved_device
    )
    frame_idx = torch.full(
        (batch_size, 1), frame_index, dtype=torch.long, device=resolved_device
    )
    frame_timestamp = torch.full(
        (batch_size, 1),
        frame_index * spec.frame_timestamp_stride,
        dtype=torch.long,
        device=resolved_device,
    )
    return {
        "button": buttons,
        "mouse": mouse,
        "scroll": scroll,
        "frame_idx": frame_idx,
        "frame_timestamp": frame_timestamp,
    }
