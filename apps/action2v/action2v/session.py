# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session and model loop shared by action-to-video applications."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import Tensor

from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.blit_model_output_to_screen_loop import (
    BlitModelOutputToScreenLoop,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .input import ActionEventAccumulator, ActionSnapshot

ActionMapper = Callable[[ActionSnapshot], Any]
"""Map a model-neutral snapshot to one integration-owned action."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Action2VStep:
    """Generated frames and metrics returned by an integration model session."""

    frames: Tensor
    """Generated frames in the session's declared output layout."""

    metrics: Mapping[str, float | int] = field(default_factory=dict)
    """Numeric measurements associated with this generation step."""


class Action2VModelSession(Protocol):
    """Integration-owned mutable model state for one action-to-video rollout."""

    @property
    def seed_frames(self) -> Tensor:
        """Return the frames that establish the initial displayed world state."""
        ...

    def step(self, step_index: int, action: Any) -> Action2VStep:
        """Generate one action-conditioned video chunk."""
        ...

    def reset(self) -> None:
        """Rebuild model state for the original seed and random stream."""
        ...

    def close(self) -> None:
        """Release session-owned model state."""
        ...


Action2VModelSessionFactory = Callable[[], Action2VModelSession]
"""Create integration-owned model state when a v2 session initializes."""


@dataclass(slots=True)
class Action2VModelState:
    """Mutable rollout state owned exclusively by the model thread."""

    model_session: Action2VModelSession
    """Integration-owned cache, RNG, and generation behavior."""

    session_desc: SessionDesc
    """Output shape, layout, and rates accepted for this session."""

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

    def __getattr__(self, name: str) -> Any:
        """Expose integration model-session state for compatibility and debugging."""
        return getattr(self.model_session, name)


class Action2VModelLoop(IModelLoop[Action2VModelState]):
    """Emit the seed world state and one video chunk for each action."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Apply live input and produce one ordered video result."""
        state = self.state
        if step_index == 0:
            if state.seed_emitted or state.actions_generated:
                raise RuntimeError("Action2V seed step is out of sequence")
            state.event_accumulator.consume(events)
            state.seed_emitted = True
            frames = _validate_frames(
                state.model_session.seed_frames, state.session_desc
            )
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
        snapshot = state.event_accumulator.consume(events)
        action = state.action_mapper(snapshot)

        generated = state.model_session.step(step_index, action)
        frames = _validate_frames(generated.frames, state.session_desc)
        state.actions_generated += 1
        metrics = dict(generated.metrics)
        metrics.setdefault("autoregressive_index", step_index)
        metrics.setdefault("generated_frames", frames.shape[0])
        return [
            StepResult(
                step_index=step_index,
                output=frames.detach(),
                frame_count=frames.shape[0],
                output_layout=state.session_desc.output_layout,
                metrics=metrics,
            )
        ]

    def is_finished(self) -> bool:
        """Keep the live model loop running until the runtime stops it."""
        return False

    def reset(self) -> None:
        """Restore the model session and live-input accumulator to their seed state."""
        state = self.state
        state.model_session.reset()
        state.event_accumulator.reset()
        state.seed_emitted = False
        state.actions_generated = 0

    def close(self) -> None:
        """Release model state and transient user input."""
        self.state.model_session.close()
        self.state.event_accumulator.reset()


class Action2VSession(ISession):
    """One action-conditioned rollout with integration-owned model state."""

    def __init__(
        self,
        *,
        model_session_factory: Action2VModelSessionFactory,
        session_desc: SessionDesc,
        action_mapper: ActionMapper,
    ) -> None:
        self._model_session_factory = model_session_factory
        self._session_desc = session_desc
        self._action_mapper = action_mapper
        self._model_session: Action2VModelSession | None = None

    @property
    def session_desc(self) -> SessionDesc:
        """Return the output contract accepted by this session."""
        return self._session_desc

    def init(self) -> None:
        """Create integration model state and register the shared model loop."""
        model_session = self._model_session_factory()
        self._model_session = model_session
        ui_loop = self.register_ui_loop(BlitModelOutputToScreenLoop)
        invoke_async(ui_loop, lambda _: ui_loop.request_hide_cursor(True))
        invoke_async(ui_loop, lambda _: ui_loop.request_lock_cursor_to_window(True))
        self.register_model_loop(
            Action2VModelLoop,
            state=Action2VModelState(
                model_session=model_session,
                session_desc=self._session_desc,
                action_mapper=self._action_mapper,
            ),
        )

    def close(self) -> None:
        """Release integration model state retained by this session."""
        if self._model_session is not None:
            self._model_session.close()
            self._model_session = None


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
    "Action2VModelSession",
    "Action2VModelSessionFactory",
    "Action2VModelState",
    "Action2VSession",
    "Action2VStep",
    "ActionMapper",
]
