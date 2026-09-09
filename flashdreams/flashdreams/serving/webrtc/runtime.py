# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime contracts for shared WebRTC demo serving."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Any, Protocol

import torch

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.video_output import VideoStepResult, infer_video_num_frames
from flashdreams.serving.realtime.input import PoseSegment


class WebRTCStepResult(VideoStepResult):
    """One generated chunk handed back by a WebRTC model runtime."""


def make_webrtc_step_result(
    *,
    chunk_index: int,
    video_chunk: torch.Tensor | None,
    layout: VideoTensorLayout,
    stats: dict[str, float] | None = None,
    sync_device: torch.device | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    num_frames: int | None = None,
) -> WebRTCStepResult:
    """Package a generated chunk for WebRTC without forcing a host copy.

    ``video_chunk`` may be ``None`` in latent/token-streaming mode, where the
    server skips the VAE decode and sends the latent (via ``metadata``) instead.
    In that case ``num_frames`` must be supplied explicitly (it cannot be
    inferred from a missing tensor); it is available from the runtime's
    ``peek_next_chunk_num_frames`` (decoder config, not decoder state).
    """
    if sync_device is not None:
        device = torch.device(sync_device)
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()

    if video_chunk is None:
        if num_frames is None:
            raise ValueError(
                "make_webrtc_step_result: num_frames is required when "
                "video_chunk is None (latent/token mode)."
            )
        resolved_frames = num_frames
    else:
        resolved_frames = (
            num_frames
            if num_frames is not None
            else infer_video_num_frames(video_chunk, layout=layout)
        )

    return WebRTCStepResult(
        chunk_index=chunk_index,
        num_frames=resolved_frames,
        video_chunk=video_chunk.detach() if video_chunk is not None else None,
        stats=stats,
        layout=layout,
        metadata=dict(metadata or {}),
    )


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
