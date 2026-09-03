# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only Action2V application for runtime and input development."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.runtime_v2.session_desc import SessionDesc

from .application import Action2VApplication, Action2VApplicationDefaults
from .input import ActionSnapshot
from .session import Action2VStep


class DummyAction2VModelSession:
    """Render solid-color chunks from model-neutral action snapshots."""

    def __init__(self, seed_frames: Tensor) -> None:
        self._seed_frames = seed_frames

    @property
    def seed_frames(self) -> Tensor:
        """Return the neutral four-frame seed chunk."""
        return self._seed_frames

    def step(self, step_index: int, action: Any) -> Action2VStep:
        """Render one four-frame color chunk from an action snapshot."""
        if not isinstance(action, ActionSnapshot):
            raise TypeError("Dummy Action2V actions must be ActionSnapshot values.")
        height, width = self._seed_frames.shape[-2:]
        frames = torch.empty((4, 3, height, width), dtype=torch.float32)
        forward = 0.8 if "W" in action.keys else -0.8
        button = 0.8 if action.mouse_buttons else -0.8
        pointer = max(-1.0, min(1.0, action.mouse_dx * 8.0 + action.wheel_y * 0.2))
        frames[:, 0].fill_(forward)
        frames[:, 1].fill_(button)
        frames[:, 2].fill_(pointer)
        frames.add_(min(step_index, 10) * 0.01).clamp_(-1.0, 1.0)
        return Action2VStep(frames=frames, metrics={"dummy_step": step_index})

    def reset(self) -> None:
        """Reset the stateless dummy model session."""

    def close(self) -> None:
        """Release the stateless dummy model session."""


def _pipeline_factory(seed: int, device: torch.device, profile: bool) -> object:
    del seed, device, profile
    return object()


def _seed_loader(path: Path, session_desc: SessionDesc) -> Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"dummy seed does not exist: {path}")
    return torch.zeros(
        (4, 3, session_desc.video_height, session_desc.video_width),
        dtype=torch.float32,
    )


def _action_mapper(session_desc: SessionDesc, sensitivity: float):
    del session_desc

    def map_snapshot(snapshot: ActionSnapshot) -> ActionSnapshot:
        return ActionSnapshot(
            keys=snapshot.keys,
            mouse_buttons=snapshot.mouse_buttons,
            mouse_dx=snapshot.mouse_dx * sensitivity,
            mouse_dy=snapshot.mouse_dy * sensitivity,
            wheel_x=snapshot.wheel_x,
            wheel_y=snapshot.wheel_y,
        )

    return map_snapshot


def _model_session_builder(
    pipeline: Any,
    pipeline_lock: threading.Lock,
    session_desc: SessionDesc,
    seed_frames: Tensor,
    seed: int,
) -> DummyAction2VModelSession:
    del pipeline, pipeline_lock, session_desc, seed
    return DummyAction2VModelSession(seed_frames)


DUMMY_ACTION2V_DEFAULTS = Action2VApplicationDefaults(
    slug="action2v-dummy",
    pipeline_factory=_pipeline_factory,
    seed_loader=_seed_loader,
    action_mapper_factory=_action_mapper,
    model_session_builder=_model_session_builder,
    pixel_width=320,
    pixel_height=180,
    fps=60,
    device="cpu",
    metadata={"model": "action2v-dummy", "frames_per_action": 4},
)
"""Defaults for the CPU-only action-to-video demonstration."""


def create_app() -> IApplication:
    """Return the CPU-only Action2V demonstration application."""
    return Action2VApplication(defaults=DUMMY_ACTION2V_DEFAULTS)


__all__ = ["DUMMY_ACTION2V_DEFAULTS", "DummyAction2VModelSession", "create_app"]
