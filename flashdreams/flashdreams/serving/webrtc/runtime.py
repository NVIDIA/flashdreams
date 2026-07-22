# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime contracts for shared WebRTC demo serving."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from flashdreams.serving.realtime.input import PoseSegment


@dataclass(slots=True)
class WebRTCStepResult:
    """One generated chunk handed back by a WebRTC model runtime."""

    chunk_index: int
    # Number of output video frames in ``video_chunk``.
    num_frames: int
    video_chunk: torch.Tensor
    stats: dict[str, float] | None


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
    ) -> WebRTCStepResult: ...

    async def close(self) -> None: ...


class WebRTCEventRuntime(Protocol):
    """Optional runtime capability for model-specific data-channel events."""

    def trigger_event(
        self, *, event_id: str, state: str = "trigger"
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...


class WebRTCServerLifecycle(Protocol):
    """Distributed worker lifecycle used by the shared WebRTC serve loop."""

    def send_exit_signal(self) -> None: ...

    def wait_for_termination(self) -> None: ...


class WebRTCSessionRuntime(WebRTCGenerationRuntime, WebRTCServerLifecycle, Protocol):
    """Complete runtime contract consumed by the shared session manager."""
