# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waypoint V2 session and autoregressive model loop."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from waypoint import WAYPOINT_1_5, WaypointControl
from waypoint.pipeline import WaypointInferencePipeline

from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from waypoint_v2.control_events import ControlEventAdapter


@dataclass(slots=True)
class WaypointModelState:
    """Mutable state isolated to one Waypoint model loop."""

    pipeline: WaypointInferencePipeline
    pipeline_lock: threading.Lock
    session_desc: SessionDesc
    seed_frames: Tensor
    seed_pixels: Tensor
    seed: int
    controls: tuple[WaypointControl, ...] | None
    control_events: ControlEventAdapter
    cache: Any | None
    initial_rng_state: Tensor
    rng_state: Tensor
    seed_emitted: bool = False
    controls_generated: int = 0


class WaypointModelLoop(IModelLoop[WaypointModelState]):
    """Emit the established seed, then four frames for every model action."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Generate the seed result or one controlled autoregressive action."""
        state = self.state
        if step_index == 0:
            if state.seed_emitted or state.controls_generated:
                raise RuntimeError("Waypoint seed step is out of sequence")
            state.seed_emitted = True
            return [
                StepResult(
                    step_index=0,
                    output=state.seed_frames,
                    frame_count=WAYPOINT_1_5.frames_per_action,
                    output_layout=state.session_desc.output_layout,
                    metrics={"autoregressive_index": 0, "seed_frames": 4},
                )
            ]

        expected_index = state.controls_generated + 1
        if step_index != expected_index:
            raise RuntimeError(
                f"Waypoint action step is out of sequence: expected {expected_index}, "
                f"got {step_index}"
            )
        control = self._control_for_step(events)
        cache = self._require_cache()

        with state.pipeline_lock:
            rng = state.pipeline.diffusion_model.rng
            if rng is None:
                raise RuntimeError("Waypoint pipeline must have a deterministic seed")
            rng.set_state(state.rng_state)
            video = state.pipeline.generate(step_index, cache, control)
            stats = state.pipeline.finalize(step_index, cache)
            state.rng_state = rng.get_state()

        output = _presentation_frames(video, state.session_desc)
        state.controls_generated += 1
        metrics: dict[str, float | int] = dict(stats or {})
        metrics.setdefault("autoregressive_index", step_index)
        metrics.setdefault("generated_frames", output.shape[0])
        return [
            StepResult(
                step_index=step_index,
                output=output,
                frame_count=output.shape[0],
                output_layout=state.session_desc.output_layout,
                metrics=metrics,
            )
        ]

    def is_finished(self) -> bool:
        """Finish after the seed and every file-driven control; live mode persists."""
        controls = self.state.controls
        return (
            controls is not None
            and self.state.seed_emitted
            and self.state.controls_generated >= len(controls)
        )

    def reset(self) -> None:
        """Re-establish the seed cache, RNG stream, and live-input state."""
        state = self.state
        state.control_events.reset()
        state.seed_emitted = False
        state.controls_generated = 0
        state.rng_state = state.initial_rng_state.clone()
        with state.pipeline_lock:
            # Drop the old cache before allocating its replacement so reset does not
            # temporarily retain two full KV caches on the GPU.
            state.cache = None
            state.cache = state.pipeline.initialize_cache(seed_pixels=state.seed_pixels)

    def close(self) -> None:
        """Release the rollout cache and transient control state."""
        self.state.cache = None
        self.state.control_events.reset()

    def _control_for_step(self, events: UserInputEvents) -> WaypointControl:
        state = self.state
        if state.controls is None:
            return state.control_events.consume(events)
        return state.controls[state.controls_generated]

    def _require_cache(self) -> Any:
        cache = self.state.cache
        if cache is None:
            raise RuntimeError("Waypoint session cache is closed")
        return cache


class WaypointSession(ISession):
    """One image-established Waypoint rollout with isolated cache and controls."""

    def __init__(
        self,
        *,
        pipeline: WaypointInferencePipeline,
        pipeline_lock: threading.Lock,
        session_desc: SessionDesc,
        seed_frames: Tensor,
        seed: int,
        controls: tuple[WaypointControl, ...] | None,
        mouse_sensitivity: float,
    ) -> None:
        """Create an uninitialized Waypoint session.

        Args:
            pipeline: Model modules shared by sessions from one application.
            pipeline_lock: Lock protecting the shared model RNG while a session
                restores and advances its own RNG state.
            session_desc: Declared TCHW presentation contract.
            seed_frames: Four normalized TCHW frames establishing the world.
            seed: Fixed per-session diffusion seed.
            controls: Finite file-driven actions, or ``None`` for live input.
            mouse_sensitivity: Multiplier used by the live input adapter.

        Raises:
            ValueError: The layout or seed-frame shape does not match the session.
        """
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                "Waypoint only produces tchw output, got "
                f"{session_desc.output_layout.value}."
            )
        expected_shape = (
            WAYPOINT_1_5.frames_per_action,
            3,
            session_desc.video_height,
            session_desc.video_width,
        )
        if tuple(seed_frames.shape) != expected_shape:
            raise ValueError(
                f"seed_frames must have shape {expected_shape}, got "
                f"{tuple(seed_frames.shape)}"
            )
        self._pipeline = pipeline
        self._pipeline_lock = pipeline_lock
        self._session_desc = session_desc
        self._seed_frames = seed_frames
        self._seed = seed
        self._controls = controls
        self._mouse_sensitivity = mouse_sensitivity
        self._state: WaypointModelState | None = None

    def init(self) -> None:
        """Move the seed to the model device, establish cache, and register the loop."""
        dtype = self._pipeline.diffusion_model.dtype
        seed_frames = self._seed_frames.to(
            device=self._pipeline.device, dtype=dtype
        ).contiguous()
        seed_pixels = _seed_pixels(seed_frames)
        generator = torch.Generator(device=self._pipeline.device).manual_seed(
            self._seed
        )
        initial_rng_state = generator.get_state()
        with self._pipeline_lock:
            cache = self._pipeline.initialize_cache(seed_pixels=seed_pixels)
        state = WaypointModelState(
            pipeline=self._pipeline,
            pipeline_lock=self._pipeline_lock,
            session_desc=self._session_desc,
            seed_frames=seed_frames,
            seed_pixels=seed_pixels,
            seed=self._seed,
            controls=self._controls,
            control_events=ControlEventAdapter(
                video_width=self._session_desc.video_width,
                video_height=self._session_desc.video_height,
                mouse_sensitivity=self._mouse_sensitivity,
            ),
            cache=cache,
            initial_rng_state=initial_rng_state,
            rng_state=initial_rng_state.clone(),
        )
        self._state = state
        self.register_model_loop(WaypointModelLoop, state=state)

    @property
    def session_desc(self) -> SessionDesc:
        """Return the presentation contract accepted by the application."""
        return self._session_desc

    def close(self) -> None:
        """Drop session-owned tensors that retain rollout state."""
        if self._state is not None:
            self._state.cache = None
            self._state.control_events.reset()


def _seed_pixels(seed_frames: Tensor) -> Tensor:
    return seed_frames.add(1.0).mul(0.5).unsqueeze(0)


def _presentation_frames(video: Tensor, session_desc: SessionDesc) -> Tensor:
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError(
            "Waypoint pipeline output must have [1, T, C, H, W] layout, got "
            f"{tuple(video.shape)}"
        )
    output = video[0]
    expected = (WAYPOINT_1_5.frames_per_action, 3)
    if tuple(output.shape[:2]) != expected:
        raise ValueError(
            f"Waypoint pipeline must emit {expected[0]} RGB frames, got "
            f"{tuple(output.shape)}"
        )
    target_size = (session_desc.video_height, session_desc.video_width)
    if tuple(output.shape[-2:]) != target_size:
        raise ValueError(
            "Waypoint pipeline output spatial size must match the session: expected "
            f"{target_size[1]}x{target_size[0]}, got "
            f"{output.shape[-1]}x{output.shape[-2]}"
        )
    return output.contiguous()


__all__ = ["WaypointModelLoop", "WaypointModelState", "WaypointSession"]
