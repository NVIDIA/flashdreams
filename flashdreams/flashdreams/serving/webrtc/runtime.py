# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime contracts for shared WebRTC demo serving."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.inputs import UserInputSchema
from flashdreams.runtime.interfaces import InferenceSession
from flashdreams.runtime.mapping import InputMapping
from flashdreams.serving.realtime.input import PoseSegment


class WebRTCRuntimeConfig(Protocol):
    """Config fields consumed by the shared WebRTC session manager."""

    video_width: int
    video_height: int
    warmup_chunks: int
    warmup_timeout_s: float


class WebRTCGenerationRuntime(Protocol):
    """Generation lifecycle for one shared WebRTC session.

    Integrations keep their model-specific state, checkpoints, conditioning,
    and cache logic inside their concrete runtime. The shared manager only
    needs this lifecycle and chunk-generation surface.

    By default, ``peek_next_chunk_num_frames`` and
    ``peek_steady_chunk_num_frames`` are used for both input sampling and
    output queue sizing. Runtimes whose model input clock differs from their
    output video clock may also implement these optional methods:

    - ``peek_input_fps() -> float`` for the control/input sampling clock.
    - ``peek_next_input_num_frames() -> int`` for the length of ``frame_times``.
    - ``peek_steady_output_num_frames() -> int`` for video queue sizing.
    """

    async def initialize(self) -> None: ...

    async def reset_for_new_session(self) -> None: ...

    def peek_steady_chunk_num_frames(self) -> int: ...

    def peek_next_chunk_num_frames(self) -> int: ...

    async def generate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> VideoStepResult: ...

    async def close(self) -> None: ...


class WebRTCEventRuntime(Protocol):
    """Optional runtime capability for model-specific data-channel events."""

    def trigger_event(
        self, *, event_id: str, state: str = "trigger"
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...


class WebRTCInferenceSessionRuntime(Protocol):
    """Optional runtime capability for driving an ``InferenceSession``.

    A runtime implementing this opts into the manager's session branch, where
    raw key and text events are canonicalized and mapped into per-step
    ``InferenceInput`` instead of being handed to ``generate_chunk`` as
    pre-integrated pose segments. The transport keeps owning event
    timestamping and input-window selection; the model only declares its
    mapping and consumes model-facing inputs.

    Runtimes on this branch do not need ``generate_chunk`` or ``trigger_event``:
    camera control arrives as mapped step inputs, and text events arrive as a
    session-global conditioning update in the same payload.
    """

    async def start_inference_session(self) -> InferenceSession: ...

    @property
    def input_mapping(self) -> InputMapping: ...

    @property
    def input_canonicalizer(self) -> InputCanonicalizer: ...

    @property
    def input_source_schema(self) -> UserInputSchema: ...


class WebRTCServerLifecycle(Protocol):
    """Distributed worker lifecycle used by the shared WebRTC serve loop."""

    def send_exit_signal(self) -> None: ...

    def wait_for_termination(self) -> None: ...


class WebRTCSessionRuntime(WebRTCGenerationRuntime, WebRTCServerLifecycle, Protocol):
    """Complete runtime contract consumed by the shared session manager."""
