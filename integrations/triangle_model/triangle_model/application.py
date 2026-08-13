# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete triangle application."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import cast

import torch
from flashdreams.runtime import InferenceConfig, InferenceInput, StepRequest, StepResult
from triangle_app import DEFAULT_TRIANGLE_COLOR, TriangleApp, TriangleScenario


class TriangleModel(TriangleApp):
    model_id = "triangle-model"

    def create_runtime(self, config: InferenceConfig) -> TriangleRuntime:
        self.validate_config(config)
        return TriangleRuntime()


class TriangleRuntime:
    def start_session(self, inputs: InferenceInput) -> TriangleSession:
        return TriangleSession(
            TriangleScenario(
                width=int(inputs.global_conditioning["width"]),
                height=int(inputs.global_conditioning["height"]),
                fps=int(inputs.global_conditioning["fps"]),
                total_frames=int(inputs.global_conditioning["total_frames"]),
            )
        )

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
            raise RuntimeError("Cannot step a closed triangle session.")
        if self._step_index >= self._scenario.total_frames:
            raise RuntimeError("Triangle session is complete.")
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
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._step_index = 0
        self._closed = False

    def close(self) -> None:
        self._closed = True


def create_app(args: Sequence[str]) -> TriangleModel:
    parser = argparse.ArgumentParser(prog="triangle-model")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--total-frames", type=int, default=180)
    parser.add_argument("--title", default="FlashDreams · Triangle Model")
    parsed = parser.parse_args(list(args))
    return TriangleModel(
        application_name="triangle-model",
        description="Synthetic triangle application.",
        width=parsed.width,
        height=parsed.height,
        fps=parsed.fps,
        total_frames=parsed.total_frames,
        title=parsed.title,
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


__all__ = ["TriangleModel", "create_app"]
