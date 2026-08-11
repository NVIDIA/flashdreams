# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime contracts and thread-affine execution for shared WebRTC serving."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from enum import IntEnum
from typing import Any, Generic, Protocol, TypeVar

import torch
import torch.distributed as dist

from flashdreams.core.distributed.rank_orchestration import (
    RankCoordinator,
    distributed_op,
)
from flashdreams.runtime.types import StepRequest, StepResult
from flashdreams.runtime.worker import ThreadAffineRuntimeWorker
from flashdreams.serving.webrtc.encoders import (
    EncoderBackend,
    VideoEncoder,
    select_encoder,
)


class WebRTCControlSignal(IntEnum):
    """Rank-orchestration signals shared by WebRTC runtimes."""

    INITIALIZE = 0
    RESET_SESSION = 1
    ACTION_STEP = 2
    CLOSE = 3
    EVENT = 4
    SESSION_STEP = 5
    SESSION_CLOSE = 6
    EXIT = 99


class WebRTCRuntimeConfig(Protocol):
    """Config fields consumed by the shared WebRTC session manager."""

    video_width: int
    video_height: int
    warmup_chunks: int
    warmup_timeout_s: float


class ThreadAffineWebRTCRuntimeConfig(WebRTCRuntimeConfig, Protocol):
    """Configuration consumed by the shared runtime execution layer."""

    device: str
    fps: int
    encoder_backend: EncoderBackend
    encoder_bitrate_bps: int
    encoder_gop: int


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

    def next_step_request(self) -> StepRequest: ...

    def peek_steady_output_num_frames(self) -> int: ...

    async def step(
        self,
        *,
        request: StepRequest,
        segments: list[Any],
        frame_times: list[float],
    ) -> StepResult: ...

    async def close(self) -> None: ...


class WebRTCEventRuntime(Protocol):
    """Optional runtime capability for model-specific data-channel events."""

    def trigger_event(
        self, *, event_id: str, state: str = "trigger"
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...


_ConfigT = TypeVar("_ConfigT", bound=ThreadAffineWebRTCRuntimeConfig)
_SessionInputT = TypeVar("_SessionInputT")


class ThreadAffineDistributedWebRTCRuntime(
    ABC,
    Generic[_ConfigT, _SessionInputT],
):
    """Coordinate one thread-affine, distributed WebRTC model runtime.

    Subclasses own model construction, rollout state, conditioning, and chunk
    generation. This base owns the identical async-to-thread dispatch, rank
    signaling, step ordering, and video-encoder lifecycle used by integrations.
    """

    MASTER_RANK = 0

    def __init__(
        self,
        *,
        config: _ConfigT,
        runtime_error_type: type[RuntimeError],
        thread_name: str,
    ) -> None:
        self.config = config
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()
        self._runtime_error_type = runtime_error_type
        self._device = self._resolve_device(config.device)
        self._closed = False
        self._video_encoder: VideoEncoder | None = None
        self._worker = ThreadAffineRuntimeWorker(
            device=self._device,
            thread_name=thread_name,
        )
        self._step_lock = asyncio.Lock()
        self.rank_coordinator = RankCoordinator(
            device=self._device,
            signal_type=WebRTCControlSignal,
            is_master=self.is_master,
            master_rank=self.MASTER_RANK,
        )
        self.rank_coordinator.register_distributed_ops(self)

    @staticmethod
    def _resolve_device(device_spec: str | torch.device) -> torch.device:
        device = torch.device(device_spec)
        if device.type == "cuda" and device.index is None:
            device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cuda:0"
            )
        return device

    @property
    def is_master(self) -> bool:
        return self.rank == self.MASTER_RANK

    @property
    def video_encoder(self) -> VideoEncoder:
        """Return the encoder selected during runtime initialization."""
        if self._video_encoder is None:
            raise self._runtime_error(
                "Video encoder is not initialized; call runtime.initialize() first."
            )
        return self._video_encoder

    def wait_for_termination(self) -> None:
        self.rank_coordinator.worker_loop(exit_signal=WebRTCControlSignal.EXIT)

    def send_exit_signal(self) -> None:
        if self.is_master:
            self.rank_coordinator.send_exit(exit_signal=WebRTCControlSignal.EXIT)

    async def initialize(self) -> None:
        if self._is_runtime_initialized():
            return
        await self._worker.call(self._initialize_sync_all_ranks)

    async def reset_for_new_session(
        self, session_input: _SessionInputT | None = None
    ) -> None:
        self._require_open_and_initialized()
        await self._worker.call(self._reset_rollout_sync_all_ranks, session_input)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._worker.call(self._close_sync_all_ranks)
        finally:
            await self._worker.close()

    async def step(
        self,
        *,
        request: StepRequest,
        segments: list[Any],
        frame_times: list[float],
    ) -> StepResult:
        self._require_open_and_initialized(session=True)
        expected_step = self._runtime_step_index()
        if request.step_index != expected_step:
            raise self._runtime_error(
                f"Expected request step {expected_step}, got {request.step_index}."
            )

        async with self._step_lock:
            self._require_open_and_initialized(session=True)
            return await self._worker.call(
                self._generate_chunk_sync_all_ranks,
                segments,
                frame_times,
            )

    def peek_input_fps(self) -> float:
        return float(self.config.fps)

    def next_step_request(self) -> StepRequest:
        self._require_open_and_initialized()
        return StepRequest(
            step_index=self._runtime_step_index(),
            metadata={"input_frame_count": self._next_input_frame_count()},
        )

    def peek_steady_output_num_frames(self) -> int:
        self._require_open_and_initialized()
        return self._steady_output_frame_count()

    def _runtime_error(self, message: str) -> RuntimeError:
        return self._runtime_error_type(message)

    def _require_open_and_initialized(self, *, session: bool = False) -> None:
        if self._closed:
            noun = "Session" if session else "Runtime"
            raise self._runtime_error(f"{noun} is closed.")
        if not self._is_runtime_initialized():
            raise self._runtime_error("Runtime is not initialized.")

    def _initialize_video_encoder_sync(self) -> None:
        """Select the master rank's encoder on the model runtime thread."""
        if not self.is_master:
            return
        if self._video_encoder is not None:
            self._video_encoder.close()
            self._video_encoder = None

        backend = self.config.encoder_backend
        if self._device.type != "cuda" and backend == "auto":
            backend = "default"
        if self._device.type != "cuda" and backend == "nvenc":
            raise self._runtime_error(
                "encoder_backend='nvenc' requires a CUDA runtime device."
            )
        gpu_id = self._device.index if self._device.index is not None else 0
        self._video_encoder = select_encoder(
            backend=backend,
            width=self.config.video_width,
            height=self.config.video_height,
            fps=self.config.fps,
            bitrate=self.config.encoder_bitrate_bps,
            gpu_id=gpu_id,
            gop=self.config.encoder_gop,
        )

    def _close_video_encoder_sync(self) -> None:
        if self._video_encoder is not None:
            self._video_encoder.close()
            self._video_encoder = None

    @distributed_op(WebRTCControlSignal.INITIALIZE)
    def _initialize_sync_all_ranks(self) -> None:
        self._initialize_sync()

    @distributed_op(WebRTCControlSignal.RESET_SESSION)
    def _reset_rollout_sync_all_ranks(
        self, session_input: _SessionInputT | None = None
    ) -> None:
        self._reset_rollout_sync(session_input=session_input)

    @distributed_op(WebRTCControlSignal.ACTION_STEP)
    def _generate_chunk_sync_all_ranks(
        self,
        segments: list[Any],
        frame_times: list[float],
    ) -> StepResult:
        return self._generate_one_chunk_sync(segments=segments, frame_times=frame_times)

    @distributed_op(WebRTCControlSignal.CLOSE)
    def _close_sync_all_ranks(self) -> None:
        try:
            self._close_sync()
        finally:
            self._close_video_encoder_sync()

    @abstractmethod
    def _is_runtime_initialized(self) -> bool: ...

    @abstractmethod
    def _runtime_step_index(self) -> int: ...

    @abstractmethod
    def _next_input_frame_count(self) -> int: ...

    @abstractmethod
    def _steady_output_frame_count(self) -> int: ...

    @abstractmethod
    def _initialize_sync(self) -> None: ...

    @abstractmethod
    def _reset_rollout_sync(
        self, session_input: _SessionInputT | None = None
    ) -> None: ...

    @abstractmethod
    def _generate_one_chunk_sync(
        self,
        *,
        segments: list[Any],
        frame_times: list[float],
    ) -> StepResult: ...

    @abstractmethod
    def _close_sync(self) -> None: ...
