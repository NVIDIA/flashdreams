# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct OmniDreams pipeline bridge for a model-thread game rollout."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from omnidreams_game_engine.engine import EngineStep
from omnidreams_game_engine.types import DriverCommand, SceneDefinition


class VideoPostprocessor(Protocol):
    """Optional session-local generated-video transform."""

    def __call__(self, video: Tensor) -> Tensor:
        ...


class RolloutEngine(Protocol):
    """Engine operations needed by one direct model rollout."""

    @property
    def is_running(self) -> bool: ...

    @property
    def current_game_frame(self) -> object: ...

    def step(self, commands: tuple[DriverCommand, ...]) -> EngineStep: ...

    def submit_text(self, value: str) -> object: ...

    def close(self) -> None: ...


EngineFactory = Callable[[], RolloutEngine]


@dataclass(frozen=True, slots=True)
class WorldModelStep:
    """Generated video plus the engine data that produced it."""

    video_bvtchw: Tensor
    engine: EngineStep
    metrics: Mapping[str, float | int]


class WorldModelRollout:
    """Own one session's game engine and autoregressive model cache."""

    def __init__(
        self,
        *,
        pipeline: Any,
        scene: SceneDefinition,
        engine_factory: EngineFactory,
        postprocess: VideoPostprocessor | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.scene = scene
        self._engine_factory = engine_factory
        self._postprocess = postprocess
        self.engine = engine_factory()
        self.cache = self._new_cache()
        self._closed = False

    @property
    def is_running(self) -> bool:
        return not self._closed and self.engine.is_running

    def frame_count(self, autoregressive_index: int) -> int:
        """Return the pipeline's authoritative output count for one step."""
        count = int(self.pipeline.get_num_frames(autoregressive_index))
        if count <= 0:
            raise ValueError("The pipeline returned a non-positive frame count")
        return count

    def step(
        self,
        *,
        autoregressive_index: int,
        commands: tuple[DriverCommand, ...],
    ) -> WorldModelStep:
        """Simulate, condition, generate, and finalize one block directly."""
        if self._closed:
            raise RuntimeError("WorldModelRollout is closed")
        expected = self.frame_count(autoregressive_index)
        if len(commands) != expected:
            raise ValueError(f"Expected {expected} commands, got {len(commands)}")
        engine_step = self.engine.step(commands)
        with torch.no_grad():
            video = self.pipeline.generate(
                autoregressive_index=autoregressive_index,
                cache=self.cache,
                hdmap=engine_step.condition.hdmap_bvtchw,
            )
            metrics = self.pipeline.finalize(
                autoregressive_index=autoregressive_index,
                cache=self.cache,
            )
        if self._postprocess is not None:
            video = self._postprocess(video)
        if video.ndim != 6 or tuple(video.shape[:2]) != (1, 1):
            raise ValueError(
                "The game requires single-batch, single-view BVTCHW video; got "
                f"{tuple(video.shape)}"
            )
        if int(video.shape[2]) != expected:
            raise ValueError("Generated video does not align with the engine step")
        return WorldModelStep(
            video_bvtchw=video.detach(),
            engine=engine_step,
            metrics=dict(metrics or {}),
        )

    def reset(self) -> None:
        """Recreate all mutable rollout state while retaining model weights."""
        if self._closed:
            raise RuntimeError("WorldModelRollout is closed")
        self.engine.close()
        self.engine = self._engine_factory()
        self.cache = self._new_cache()

    def close(self) -> None:
        """Release all session-local resources."""
        if self._closed:
            return
        self._closed = True
        self.cache = None
        self.engine.close()

    def _new_cache(self) -> Any:
        return self.pipeline.initialize_cache(
            text=[[self.scene.prompt]],
            image=_initial_image_tensor(
                self.scene.initial_rgb,
                device=self.pipeline.device,
            ),
            view_names=[self.scene.selected_camera.logical_name],
        )


def _initial_image_tensor(image: object, *, device: torch.device | str) -> Tensor:
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[..., :3])
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).unsqueeze(0).unsqueeze(2)
    return tensor.to(device=device, dtype=torch.bfloat16) / 127.5 - 1.0
