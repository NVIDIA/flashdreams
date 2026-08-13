# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete triangle application."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from flashdreams.runtime import InferenceConfig
from triangle_app import (
    TriangleApp,
    TriangleInferenceRequest,
    TriangleScenario,
)


class TriangleModel(TriangleApp):
    model_id = "triangle-model"

    def load_checkpoint(self, config: InferenceConfig) -> None:
        del config

    def run_inference(self, request: TriangleInferenceRequest) -> torch.Tensor:
        return _triangle_frame(
            request.scenario,
            request.step_index,
            color=request.color,
        )


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


__all__ = ["TriangleModel", "create_app"]
