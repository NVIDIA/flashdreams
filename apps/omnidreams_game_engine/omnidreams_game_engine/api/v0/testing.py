# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU stand-ins for tests and for engine development without a GPU.

:class:`FakePipeline` has the pipeline methods ``WorldModelRollout`` calls
and returns the condition as the "generated" video; :class:`PoseConditionRenderer`
fills each condition frame from its pose. With :class:`SeedSceneProvider`
they let an external engine run the whole protocol against a host on any
machine. Not part of the stable API (see ``api/STABILITY.md``).
"""

from __future__ import annotations

import io
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from omnidreams_game_engine.api.v0.external import SeedFrame
from omnidreams_game_engine.api.v0.messages import Hello
from omnidreams_game_engine.types import (
    CameraCalibration,
    ConditionBatch,
    SceneDefinition,
    TrajectoryChunk,
)


class FakePipeline:
    """The pipeline surface of ``WorldModelRollout``, on the CPU."""

    def __init__(
        self,
        plan: Sequence[int] = (5, 8),
        *,
        generate_s: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.plan = tuple(int(n) for n in plan)
        self.generate_s = float(generate_s)
        self.device = torch.device(device)
        self.caches = 0
        self.generated: list[tuple[int, tuple[int, ...]]] = []
        """``(autoregressive_index, input shape)`` of every generate call."""
        self.inputs: list[Tensor] = []

    def initialize_cache(self, **kwargs: Any) -> dict[str, Any]:
        self.caches += 1
        image = kwargs.get("image")
        mean = float(image.float().mean()) if isinstance(image, Tensor) else 0.0
        return {"cache": self.caches, "seed_mean": mean, "text": kwargs.get("text")}

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        return self.plan[min(int(autoregressive_index), len(self.plan) - 1)]

    def generate(
        self, *, autoregressive_index: int, cache: Any, input: Tensor
    ) -> Tensor:
        del cache
        if self.generate_s > 0.0:
            time.sleep(self.generate_s)
        self.generated.append((int(autoregressive_index), tuple(input.shape)))
        self.inputs.append(input.detach().clone())
        return input.detach().clone()

    def finalize(self, *, autoregressive_index: int, cache: Any) -> dict[str, float]:
        del autoregressive_index, cache
        return {"fake_generate_ms": self.generate_s * 1000.0}


@dataclass
class PoseConditionRenderer:
    """CPU ``ConditionRenderer``: each frame is filled from its pose and time."""

    width: int = 8
    height: int = 4
    scenes: list[str] = field(default_factory=list)
    trajectories: list[TrajectoryChunk] = field(default_factory=list)
    closed: bool = False

    def load_scene(self, scene: SceneDefinition) -> None:
        self.scenes.append(scene.scene_id)

    def render(self, trajectory: TrajectoryChunk) -> ConditionBatch:
        self.trajectories.append(trajectory)
        count = len(trajectory.timestamps_us)
        values = np.zeros((count, 3), dtype=np.float32)
        values[:, 0] = np.tanh(trajectory.rig_poses_world[:, 0, 3] / 100.0)
        values[:, 1] = np.tanh(trajectory.rig_poses_world[:, 1, 3] / 100.0)
        values[:, 2] = ((trajectory.timestamps_us // 1000) % 1000) / 500.0 - 1.0
        frames = torch.from_numpy(values)[:, :, None, None].expand(
            count, 3, self.height, self.width
        )
        return ConditionBatch(
            hdmap_bvtchw=frames.to(torch.bfloat16).unsqueeze(0).unsqueeze(0).clone()
        )

    def close(self) -> None:
        self.closed = True


def minimal_scene(
    *, width: int = 8, height: int = 4, scene_id: str = "external-test"
) -> SceneDefinition:
    """A scene without geometry for CPU sessions."""
    calibration = CameraCalibration(
        clipgt_name="camera:front:wide:120fov",
        logical_name="camera_front_wide_120fov",
        width=width,
        height=height,
        cx=width / 2.0,
        cy=height / 2.0,
        polynomial=np.zeros(6, dtype=np.float32),
        is_backward_polynomial=False,
        linear_cde=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )
    return SceneDefinition(
        scene_path=Path(f"{scene_id}.usdz"),
        scene_id=scene_id,
        metadata={},
        selected_camera=calibration,
        initial_rig_to_world=np.eye(4, dtype=np.float32),
        initial_timestamp_us=1_000_000_000,
        initial_yaw_rad=0.0,
        initial_speed_mps=0.0,
        initial_rgb=np.zeros((height, width, 3), dtype=np.uint8),
        prompt="a city street",
        line_layers=(),
        triangle_layers=(),
    )


def decode_seed_image(seed: SeedFrame) -> np.ndarray:
    """The seed image as ``HxWx3`` uint8 (``raw``, ``png`` or ``jpeg``)."""
    if seed.encoding == "raw":
        if seed.width is None or seed.height is None:
            raise ValueError("A raw seed needs width and height")
        return (
            np.frombuffer(seed.image, dtype=np.uint8)
            .reshape(seed.height, seed.width, 3)
            .copy()
        )
    from PIL import Image

    with Image.open(io.BytesIO(seed.image)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


@dataclass
class SeedSceneProvider:
    """``SceneProvider`` that puts the seed image and pose into a template scene."""

    template: SceneDefinition = field(default_factory=minimal_scene)
    seeds: list[SeedFrame] = field(default_factory=list)

    def scene_for_seed(self, hello: Hello, seed: SeedFrame) -> SceneDefinition:
        del hello
        self.seeds.append(seed)
        image = decode_seed_image(seed)
        return replace(
            self.template,
            initial_rgb=image,
            initial_rig_to_world=np.asarray(seed.state.rig_to_world, dtype=np.float32),
            initial_yaw_rad=float(seed.state.vehicle.yaw_rad),
            initial_speed_mps=float(seed.state.vehicle.speed_mps),
        )


__all__ = [
    "FakePipeline",
    "PoseConditionRenderer",
    "SeedSceneProvider",
    "decode_seed_image",
    "minimal_scene",
]
