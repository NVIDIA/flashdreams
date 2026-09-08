# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct OmniDreams pipeline bridge for a model-thread game rollout."""

from __future__ import annotations

import time
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

    def __call__(self, video: Tensor) -> Tensor: ...


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
class _WorldModelStepTrace:
    """Monotonic lifecycle boundaries for one generated chunk."""

    engine_step_started_ns: int
    engine_step_returned_ns: int
    generate_started_ns: int
    generate_returned_ns: int
    cache_finalize_returned_ns: int
    rollout_step_returned_ns: int


@dataclass(frozen=True, slots=True)
class WorldModelStep:
    """Generated video plus the engine data that produced it."""

    video_bvtchw: Tensor
    engine: EngineStep
    metrics: Mapping[str, float | int]
    _trace: _WorldModelStepTrace | None = None


class WorldModelRollout:
    """Own one session's game engine and autoregressive model cache."""

    def __init__(
        self,
        *,
        pipeline: Any,
        scene: SceneDefinition,
        engine_factory: EngineFactory,
        postprocess: VideoPostprocessor | None = None,
        trace_chunk_lifecycle: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.scene = scene
        self._engine_factory = engine_factory
        self._postprocess = postprocess
        self._trace_chunk_lifecycle = trace_chunk_lifecycle
        self.engine = engine_factory()
        self.cache = self._new_cache()
        self._attach_live_edit(None)
        self._closed = False

    @property
    def is_running(self) -> bool:
        return not self._closed and self.engine.is_running

    def frame_count(self, autoregressive_index: int) -> int:
        """Return the pipeline's authoritative output count for one step."""
        count = int(self.pipeline.get_num_output_frames(autoregressive_index))
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

        rollout_wall_started = time.perf_counter()
        rollout_cpu_started = time.thread_time()
        engine_step_started_ns = (
            time.monotonic_ns() if self._trace_chunk_lifecycle else None
        )
        engine_wall_started = time.perf_counter()
        engine_cpu_started = time.thread_time()
        engine_step = self.engine.step(commands)
        engine_step_returned_ns = (
            time.monotonic_ns() if self._trace_chunk_lifecycle else None
        )
        engine_wall_ms = (time.perf_counter() - engine_wall_started) * 1000.0
        engine_cpu_ms = (time.thread_time() - engine_cpu_started) * 1000.0

        pipeline_wall_started = time.perf_counter()
        pipeline_cpu_started = time.thread_time()
        live_edit = getattr(self.engine, "live_edit", None)
        prepare_model_step = getattr(live_edit, "prepare_model_step", None)
        if callable(prepare_model_step):
            prepare_model_step(
                self.pipeline,
                self.engine,
                engine_step,
                autoregressive_index,
            )
        generate_started_ns = (
            time.monotonic_ns() if self._trace_chunk_lifecycle else None
        )
        with torch.no_grad():
            video = self.pipeline.generate(
                autoregressive_index=autoregressive_index,
                cache=self.cache,
                input=engine_step.condition.hdmap_bvtchw,
            )
            generate_returned_ns = (
                time.monotonic_ns() if self._trace_chunk_lifecycle else None
            )
            metrics = self.pipeline.finalize(
                autoregressive_index=autoregressive_index,
                cache=self.cache,
            )
        cache_finalize_returned_ns = (
            time.monotonic_ns() if self._trace_chunk_lifecycle else None
        )
        pipeline_wall_ms = (time.perf_counter() - pipeline_wall_started) * 1000.0
        pipeline_cpu_ms = (time.thread_time() - pipeline_cpu_started) * 1000.0

        postprocess_wall_started = time.perf_counter()
        postprocess_cpu_started = time.thread_time()
        if self._postprocess is not None:
            video = self._postprocess(video)
        else:
            live_edit = getattr(self.engine, "live_edit", None)
            live_edit_postprocess = getattr(live_edit, "postprocess_video", None)
            if callable(live_edit_postprocess):
                video = live_edit_postprocess(video, engine_step)
        postprocess_wall_ms = (time.perf_counter() - postprocess_wall_started) * 1000.0
        postprocess_cpu_ms = (time.thread_time() - postprocess_cpu_started) * 1000.0
        if video.ndim != 6 or tuple(video.shape[:2]) != (1, 1):
            raise ValueError(
                "The game requires single-batch, single-view BVTCHW video; got "
                f"{tuple(video.shape)}"
            )
        if int(video.shape[2]) != expected:
            raise ValueError("Generated video does not align with the engine step")
        step_metrics = dict(metrics or {})
        step_metrics.update(engine_step.metrics)
        step_metrics.update(
            {
                "engine_wall_ms": engine_wall_ms,
                "engine_cpu_ms": engine_cpu_ms,
                "pipeline_wall_ms": pipeline_wall_ms,
                "pipeline_cpu_ms": pipeline_cpu_ms,
                "postprocess_wall_ms": postprocess_wall_ms,
                "postprocess_cpu_ms": postprocess_cpu_ms,
                "rollout_wall_ms": (time.perf_counter() - rollout_wall_started)
                * 1000.0,
                "rollout_cpu_ms": (time.thread_time() - rollout_cpu_started) * 1000.0,
            }
        )
        rollout_step_returned_ns = (
            time.monotonic_ns() if self._trace_chunk_lifecycle else None
        )
        trace = None
        if engine_step_started_ns is not None:
            assert (
                engine_step_returned_ns is not None
                and generate_started_ns is not None
                and generate_returned_ns is not None
                and cache_finalize_returned_ns is not None
                and rollout_step_returned_ns is not None
            )
            trace = _WorldModelStepTrace(
                engine_step_started_ns=engine_step_started_ns,
                engine_step_returned_ns=engine_step_returned_ns,
                generate_started_ns=generate_started_ns,
                generate_returned_ns=generate_returned_ns,
                cache_finalize_returned_ns=cache_finalize_returned_ns,
                rollout_step_returned_ns=rollout_step_returned_ns,
            )
        return WorldModelStep(
            video_bvtchw=video.detach(),
            engine=engine_step,
            metrics=step_metrics,
            _trace=trace,
        )

    def reset(self) -> None:
        """Recreate all mutable rollout state while retaining model weights."""
        if self._closed:
            raise RuntimeError("WorldModelRollout is closed")
        previous_live_edit = getattr(self.engine, "live_edit", None)
        self.engine.close()
        self.engine = self._engine_factory()
        self.cache = self._new_cache()
        self._attach_live_edit(previous_live_edit)

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

    def _attach_live_edit(self, previous: Any | None) -> None:
        live_edit = getattr(self.engine, "live_edit", None)
        if live_edit is None:
            return
        adopt_model_state = getattr(live_edit, "adopt_model_state", None)
        if previous is not None and callable(adopt_model_state):
            adopt_model_state(
                previous,
                self.pipeline,
                self.cache,
                self.scene.prompt,
            )
            return
        attach_model = getattr(live_edit, "attach_model", None)
        if callable(attach_model):
            attach_model(self.pipeline)
        style = getattr(live_edit, "style", None)
        if style is None:
            return
        style.attach_v2(
            self.pipeline,
            self.cache,
            self.scene.prompt,
            seconds_per_chunk=float(self.frame_count(1)) / 30.0,
        )


def _initial_image_tensor(image: object, *, device: torch.device | str) -> Tensor:
    array = np.asarray(image, dtype=np.uint8)[..., :3].copy(order="C")
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).unsqueeze(0).unsqueeze(2)
    return tensor.to(device=device, dtype=torch.bfloat16) / 127.5 - 1.0
