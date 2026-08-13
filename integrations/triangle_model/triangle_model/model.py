# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic runtime and session used to validate demo plumbing."""

from __future__ import annotations

from typing import cast

import torch
from flashdreams.runtime import InferenceInput, StepRequest, StepResult
from triangle_app import DEFAULT_TRIANGLE_COLOR, TriangleScenario


class TriangleRuntime:
    """Stateless runtime that creates isolated synthetic sessions."""

    def start_session(self, inputs: InferenceInput) -> TriangleSession:
        return TriangleSession(_scenario_from_inputs(inputs))

    def close(self) -> None:
        return None


class TriangleSession:
    def __init__(self, scenario: TriangleScenario) -> None:
        self._scenario = scenario
        self._step_index = 0
        self._closed = False

    def next_step_request(self) -> StepRequest | None:
        if self._closed or self._step_index >= self._scenario.total_frames:
            return None
        return StepRequest(
            step_index=self._step_index,
            metadata={"input_frame_count": 1},
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        if self._closed:
            raise RuntimeError("Cannot step a closed synthetic session.")
        index = self._step_index
        self._step_index += 1
        return StepResult.from_video_chunk(
            step_index=index,
            video_chunk=_triangle_frame(
                self._scenario,
                index,
                color=_color(inputs.step.get("color", DEFAULT_TRIANGLE_COLOR)),
            ).unsqueeze(0),
            layout="tchw",
            metadata={"generator": "triangle-model"},
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if inputs is not None:
            scenario = _scenario_from_inputs(inputs)
            if scenario != self._scenario:
                raise ValueError("Create a new session to change triangle geometry.")
        self._step_index = 0
        self._closed = False

    def close(self) -> None:
        self._closed = True


def _scenario_from_inputs(inputs: InferenceInput) -> TriangleScenario:
    values = inputs.global_conditioning
    return TriangleScenario(
        width=int(values["width"]),
        height=int(values["height"]),
        fps=int(values["fps"]),
        total_frames=int(values["total_frames"]),
    )


def _triangle_frame(
    scenario: TriangleScenario,
    step_index: int,
    *,
    color: tuple[int, int, int],
) -> torch.Tensor:
    progress = step_index / max(1, scenario.total_frames - 1)
    center_x = round(scenario.width * (0.2 + 0.6 * progress))
    top = scenario.height // 4
    bottom = 3 * scenario.height // 4
    half_width = max(1, scenario.width // 8)

    y = torch.arange(scenario.height).view(-1, 1)
    x = torch.arange(scenario.width).view(1, -1)
    width_at_y = (y - top) * half_width / max(1, bottom - top)
    mask = (y >= top) & (y <= bottom) & ((x - center_x).abs() <= width_at_y)

    frame = torch.zeros((scenario.height, scenario.width, 3), dtype=torch.uint8)
    frame[mask] = torch.tensor(color, dtype=torch.uint8)
    return frame.permute(2, 0, 1).contiguous()


def _color(value: object) -> tuple[int, int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(type(channel) is not int for channel in value)
    ):
        raise TypeError("Triangle color must contain three integer channels.")
    color = cast(tuple[int, int, int], value)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError("Triangle color channels must be in [0, 255].")
    return color


__all__ = [
    "TriangleRuntime",
    "TriangleSession",
]
