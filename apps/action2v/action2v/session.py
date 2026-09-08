# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session and model loop shared by action-to-video applications."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.api_v2.loop import IModelLoop, IUILoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.blit_model_output_to_screen_loop import (
    BlitModelOutputToScreenLoop,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .input import ActionEventAccumulator, ActionSnapshot

ActionMapper = Callable[[ActionSnapshot], Any]
"""Map a model-neutral snapshot to one integration-owned action."""


@dataclass(slots=True)
class Action2VModelState:
    """Mutable rollout state owned exclusively by the model thread."""

    pipeline: Any
    """Application-owned, loaded model pipeline."""

    pipeline_lock: threading.Lock
    """Lock protecting the shared pipeline RNG during one generation step."""

    session_desc: SessionDesc
    """Output shape, layout, and rates accepted for this session."""

    seed_frames: Tensor
    """Frames that establish the initial displayed world state."""

    seed_pixels: Tensor
    """Batched seed frames in the pipeline's ``[0, 1]`` input domain."""

    initial_rng_state: Tensor
    """Deterministic RNG state restored when the rollout resets."""

    rng_state: Tensor
    """Session-local RNG state swapped into the shared pipeline per step."""

    cache: Any | None
    """Session-local autoregressive model cache."""

    total_blocks: int
    """Number of model actions generated before the rollout completes."""

    action_mapper: ActionMapper
    """Integration hook mapping shared snapshots to model actions."""

    event_accumulator: ActionEventAccumulator = field(
        default_factory=ActionEventAccumulator
    )
    """Persistent live keyboard and mouse state for this rollout."""

    seed_emitted: bool = False
    """Whether the initial world-state frames have been published."""

    actions_generated: int = 0
    """Number of actions generated after the seed."""


class Action2VModelLoop(IModelLoop[Action2VModelState]):
    """Emit the seed world state and one video chunk for each action."""

    def __init__(self, *, ui_loop: IUILoop[Any] | None, reset_key: str) -> None:
        self._ui_loop = ui_loop
        self._reset_key = reset_key

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Apply live input and produce one ordered video result."""
        state = self.state
        if self._ui_loop is not None and _reset_key_pressed(events, self._reset_key):
            ui_loop = self._ui_loop
            invoke_async(
                ui_loop,
                lambda _: ui_loop.request_new_session(ui_loop.session_desc),
            )
        if step_index == 0:
            if state.seed_emitted or state.actions_generated:
                raise RuntimeError("Action2V seed step is out of sequence")
            state.event_accumulator.consume(events)
            state.seed_emitted = True
            frames = _validate_frames(state.seed_frames, state.session_desc)
            return [
                StepResult(
                    step_index=0,
                    output=frames.detach(),
                    frame_count=frames.shape[0],
                    output_layout=state.session_desc.output_layout,
                    metrics={
                        "autoregressive_index": 0,
                        "seed_frames": frames.shape[0],
                    },
                )
            ]

        expected_index = state.actions_generated + 1
        if step_index != expected_index:
            raise RuntimeError(
                f"Action2V action step is out of sequence: expected {expected_index}, "
                f"got {step_index}"
            )
        action = state.action_mapper(state.event_accumulator.consume(events))
        cache = _require_cache(state)
        with state.pipeline_lock:
            rng = _require_rng(state.pipeline)
            rng.set_state(state.rng_state)
            video = state.pipeline.generate(
                autoregressive_index=step_index,
                cache=cache,
                input=action,
            )
            stats = state.pipeline.finalize(
                autoregressive_index=step_index,
                cache=cache,
            )
            state.rng_state = rng.get_state()

        frames = _presentation_frames(video, state.session_desc)
        state.actions_generated += 1
        metrics: dict[str, float | int] = dict(stats or {})
        metrics.setdefault("autoregressive_index", step_index)
        metrics.setdefault("generated_frames", frames.shape[0])
        return [
            StepResult(
                step_index=step_index,
                output=frames,
                frame_count=frames.shape[0],
                output_layout=state.session_desc.output_layout,
                metrics=metrics,
            )
        ]

    def is_finished(self) -> bool:
        """Return whether this rollout generated its requested blocks."""
        return self.state.actions_generated >= self.state.total_blocks

    def reset(self) -> None:
        """Restore the model cache, RNG, and live input to their seed state."""
        state = self.state
        state.event_accumulator.reset()
        state.seed_emitted = False
        state.actions_generated = 0
        state.rng_state = state.initial_rng_state.clone()
        with state.pipeline_lock:
            state.cache = None
            state.cache = state.pipeline.initialize_cache(seed_pixels=state.seed_pixels)

    def close(self) -> None:
        """Release session-owned model state and transient user input."""
        self.state.cache = None
        self.state.event_accumulator.reset()


class Action2VSession(ISession):
    """One action-conditioned rollout sharing its application's loaded model."""

    def __init__(
        self,
        *,
        pipeline: Any,
        pipeline_lock: threading.Lock,
        session_desc: SessionDesc,
        seed_frames: Tensor,
        seed: int,
        action_mapper: ActionMapper,
        total_blocks: int,
        use_ui: bool = True,
        reset_key: str = "T",
    ) -> None:
        if total_blocks <= 0:
            raise ValueError("Action2V total_blocks must be > 0.")
        reset_key = _normalize_reset_key(reset_key)
        self._pipeline = pipeline
        self._pipeline_lock = pipeline_lock
        self._session_desc = session_desc
        self._seed_frames = seed_frames
        self._seed = seed
        self._action_mapper = action_mapper
        self._total_blocks = total_blocks
        self._use_ui = use_ui
        self._reset_key = reset_key
        self._state: Action2VModelState | None = None

    @property
    def session_desc(self) -> SessionDesc:
        """Return the output contract accepted by this session."""
        return self._session_desc

    def init(self) -> None:
        """Initialize standard pipeline state and register shared loops."""
        dtype = self._pipeline.diffusion_model.dtype
        seed_frames = self._seed_frames.to(
            device=self._pipeline.device,
            dtype=dtype,
        ).contiguous()
        _validate_frames(seed_frames, self._session_desc)
        seed_pixels = seed_frames.add(1.0).mul(0.5).unsqueeze(0)
        initial_rng_state = (
            torch.Generator(device=self._pipeline.device)
            .manual_seed(self._seed)
            .get_state()
        )
        with self._pipeline_lock:
            cache = self._pipeline.initialize_cache(seed_pixels=seed_pixels)
        ui_loop = None
        if self._use_ui:
            ui_loop = self.register_ui_loop(BlitModelOutputToScreenLoop)
            invoke_async(ui_loop, lambda _: ui_loop.request_hide_cursor(True))
            invoke_async(ui_loop, lambda _: ui_loop.request_lock_cursor_to_window(True))
        state = Action2VModelState(
            pipeline=self._pipeline,
            pipeline_lock=self._pipeline_lock,
            session_desc=self._session_desc,
            seed_frames=seed_frames,
            seed_pixels=seed_pixels,
            initial_rng_state=initial_rng_state,
            rng_state=initial_rng_state.clone(),
            cache=cache,
            total_blocks=self._total_blocks,
            action_mapper=self._action_mapper,
        )
        self._state = state
        self.register_model_loop(
            Action2VModelLoop,
            state=state,
            ui_loop=ui_loop,
            reset_key=self._reset_key,
        )

    def close(self) -> None:
        """Release session-owned tensors while retaining the application model."""
        if self._state is not None:
            self._state.cache = None
            self._state.event_accumulator.reset()
            self._state = None


def _reset_key_pressed(events: UserInputEvents, reset_key: str) -> bool:
    """Return whether an event batch presses the configured reset key."""
    return any(
        isinstance(event, KeyboardUserInputEvent)
        and event.state is KeyboardInputState.PRESSED
        and event.key.casefold() == reset_key.casefold()
        for event in events.get_events()
    )


def _normalize_reset_key(key: str) -> str:
    """Return one normalized ASCII letter for the reset binding."""
    if len(key) != 1 or not key.isascii() or not key.isalpha():
        raise ValueError("reset key must be one ASCII letter (a-zA-Z).")
    return key.upper()


def _require_cache(state: Action2VModelState) -> Any:
    if state.cache is None:
        raise RuntimeError("Action2V session cache is closed")
    return state.cache


def _require_rng(pipeline: Any) -> torch.Generator:
    rng = pipeline.diffusion_model.rng
    if rng is None:
        raise RuntimeError("Action2V pipeline config must set diffusion_model.seed.")
    return rng


def _presentation_frames(video: Tensor, session_desc: SessionDesc) -> Tensor:
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError(
            "Action2V pipeline output must have [1, T, C, H, W] layout, got "
            f"{tuple(video.shape)}"
        )
    return _validate_frames(video[0], session_desc).detach()


def _validate_frames(frames: Tensor, session_desc: SessionDesc) -> Tensor:
    if session_desc.output_layout is not VideoTensorLayout.tchw:
        raise ValueError("Action2V currently requires tchw output.")
    if frames.ndim != 4 or frames.shape[1] not in (1, 3, 4):
        raise ValueError(
            "Action2V frames must have [T, C, H, W] layout with 1, 3, or 4 "
            f"channels, got {tuple(frames.shape)}"
        )
    expected_size = (session_desc.video_height, session_desc.video_width)
    if tuple(frames.shape[-2:]) != expected_size:
        raise ValueError(
            "Action2V frame size must match the session: expected "
            f"{expected_size[1]}x{expected_size[0]}, got "
            f"{frames.shape[-1]}x{frames.shape[-2]}"
        )
    return frames.contiguous()


__all__ = [
    "Action2VModelLoop",
    "Action2VModelState",
    "Action2VSession",
    "ActionMapper",
]
