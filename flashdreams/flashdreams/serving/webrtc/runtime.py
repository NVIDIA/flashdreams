# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime contracts for shared WebRTC demo serving."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

from flashdreams.runtime.types import StepResult
from flashdreams.serving.realtime.input import PoseSegment


class WebRTCRuntimeConfig(Protocol):
    """Config fields consumed by the shared WebRTC session manager."""

    video_width: int
    video_height: int
    warmup_chunks: int
    warmup_timeout_s: float


class WebRTCServerLifecycle(Protocol):
    """Distributed worker lifecycle used by the shared WebRTC serve loop."""

    def send_exit_signal(self) -> None: ...

    def wait_for_termination(self) -> None: ...


class WebRTCSessionRuntime(WebRTCServerLifecycle, Protocol):
    """Complete runtime contract consumed by the shared session manager.

    Integrations keep their model-specific state, checkpoints, conditioning,
    and cache logic inside their concrete runtime. The shared manager only
    needs this lifecycle and chunk-generation surface.
    """

    async def initialize(self) -> None: ...

    async def reset_for_new_session(self, *, session_input: Any = None) -> None: ...

    def peek_input_fps(self) -> float: ...

    def peek_next_input_num_frames(self) -> int: ...

    def peek_steady_output_num_frames(self) -> int: ...

    async def generate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> StepResult: ...

    async def close(self) -> None: ...

class WebRTCEventRuntime(Protocol):
    """Optional runtime capability for model-specific data-channel events."""

    def trigger_event(
        self, *, event_id: str, state: str = "trigger"
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...
