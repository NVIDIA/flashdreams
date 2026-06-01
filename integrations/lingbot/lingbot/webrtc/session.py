# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from loguru import logger

from flashdreams.core.distributed.rank_orchestration import (
    RankCoordinator,
    distributed_op,
)
from flashdreams.infra.config import derive_config
from flashdreams.serving.webrtc.controls import (
    CameraPoseIntegrator,
    KeyboardResampler,
    PoseSegment,
)
from flashdreams.serving.webrtc.media import BufferedVideoTrack
from flashdreams.serving.webrtc.server import SessionBusyError
from flashdreams.serving.webrtc.warmup import (
    run_loopback_warmup_session,
    wait_for_ice_gathering_complete,
)
from lingbot.config import PIPELINE_CONFIGS
from lingbot.encoder.camctrl import CamCtrlInput
from lingbot.encoder.utils import (
    get_Ks_transformed,
    preprocess_example_poses,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CLIENT_LIVENESS_TIMEOUT_S = 10.0
_CLIENT_LIVENESS_CHECK_INTERVAL_S = 1.0


class LingbotRuntimeError(RuntimeError):
    """Raised when the Lingbot runtime is used incorrectly."""


class LingbotControlSignal(IntEnum):
    INITIALIZE = 0
    RESET_SESSION = 1
    ACTION_STEP = 2
    CLOSE = 3
    EXIT = 4


@dataclass(slots=True)
class LingbotRuntimeConfig:
    config_name: str = "lingbot-world-fast-taehv-window15-sink3"
    compile_network: bool = True
    seed: int = 42
    context_parallel_size: int = 1
    device: str = "cuda:0"
    video_height: int = 464
    video_width: int = 832
    world_scale: float | None = None
    warmup_chunks: int = 10
    warmup_timeout_s: float = 600.0

    example_data_dir: Path = REPO_ROOT / "assets/example_data/lingbot_world"
    first_frame_filename: str = "image.jpg"
    intrinsics_filename: str = "intrinsics.npy"
    poses_filename: str = "poses.npy"
    prompt_filename: str = "prompt.txt"


@dataclass(slots=True)
class LingbotStepResult:
    chunk_index: int
    num_frames: int
    video_chunk: torch.Tensor
    stats: dict[str, float] | None


class LingbotInferenceRuntime:
    """Single-session Lingbot runtime with action-bound chunk generation."""

    def __init__(self, config: LingbotRuntimeConfig | None = None) -> None:
        self.config = config or LingbotRuntimeConfig()
        self.MASTER_RANK = 0
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

        control_device = torch.device(self.config.device)
        if control_device.type == "cuda" and control_device.index is None:
            control_device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cuda:0"
            )

        self.pose_integrator = CameraPoseIntegrator()
        self.autoregressive_index = 0

        self._device: torch.device | None = None
        self._pipeline: Any | None = None
        self._cache: Any | None = None
        self._base_intrinsics: torch.Tensor | None = None
        self._first_frames: torch.Tensor | None = None
        self._prompt: str | None = None
        self._world_scale = 1.0
        self._closed = False

        self._step_lock = asyncio.Lock()
        self.rank_coordinator = RankCoordinator(
            device=control_device,
            signal_type=LingbotControlSignal,
            is_master=self.is_master,
            master_rank=self.MASTER_RANK,
        )
        self.rank_coordinator.register_distributed_ops(self)

    @property
    def is_master(self) -> bool:
        return self.rank == self.MASTER_RANK

    def wait_for_termination(self) -> None:
        self.rank_coordinator.worker_loop(exit_signal=LingbotControlSignal.EXIT)

    def send_exit_signal(self) -> None:
        if self.is_master:
            self.rank_coordinator.send_exit(exit_signal=LingbotControlSignal.EXIT)

    async def initialize(self) -> None:
        if self._pipeline is not None:
            return
        await asyncio.to_thread(self._initialize_sync_all_ranks)

    async def reset_for_new_session(self) -> None:
        if self._closed:
            raise LingbotRuntimeError("Runtime is closed.")
        if self._pipeline is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        await asyncio.to_thread(self._reset_rollout_sync_all_ranks)

    async def close(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._close_sync_all_ranks)

    async def generate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> LingbotStepResult:
        """Generate one autoregressive chunk from a piecewise-constant timeline.

        Args:
            segments: Piecewise-constant keyboard-state segments
                covering the chunk's virtual-time window; produced by
                :meth:`KeyboardResampler.sample_chunk`.
            frame_times: Virtual times at which to sample the camera
                pose; must have length equal to
                :meth:`peek_next_chunk_num_frames` at call time.

        Returns:
            :class:`LingbotStepResult` carrying the produced video chunk
            and the post-generation pipeline stats.

        Raises:
            LingbotRuntimeError: Runtime is closed or not initialized.
        """
        if self._closed:
            raise LingbotRuntimeError("Session is closed.")
        if self._pipeline is None or self._cache is None:
            raise LingbotRuntimeError("Runtime is not initialized.")

        async with self._step_lock:
            if self._closed:
                raise LingbotRuntimeError("Session is closed.")
            return await asyncio.to_thread(
                self._generate_chunk_sync_all_ranks, segments, frame_times
            )

    def peek_next_chunk_num_frames(self) -> int:
        """Return the number of frames the next chunk's pipeline call will emit.

        Master-only read with no distributed broadcast; safe to call from
        the master rank's asyncio event loop to size the resampler's
        per-chunk request.
        """
        if self._pipeline is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        return int(self._pipeline.get_num_output_frames(self.autoregressive_index))

    # Arbitrary index well past the AR-step transient; for the Wan/lingbot
    # pipelines used here the per-step count is constant for any index
    # ``>= 1`` (only AR 0 emits fewer frames due to causal first-frame
    # padding). Picking a large number is a robust way to ask "what is
    # the steady-state chunk size?" without leaning on the exact
    # boundary of that transient.
    _STEADY_STATE_AR_PROBE_INDEX: int = 1000

    def peek_steady_chunk_num_frames(self) -> int:
        """Return the steady-state per-chunk frame count.

        AR step 0 emits *fewer* frames than every subsequent step
        because of the decoder's causal first-frame padding (e.g. AR 0
        → 9 frames vs AR ≥ 1 → 12 frames for the current config). The
        video track's bounded queue must be sized to the *steady-state*
        chunk size so that the producer is not forced to block on the
        very next chunk after the AR-0 transient. Probing at a large AR
        index returns that steady-state value directly.

        Master-only read with no distributed broadcast.
        """
        if self._pipeline is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        return int(
            self._pipeline.get_num_output_frames(self._STEADY_STATE_AR_PROBE_INDEX)
        )

    @distributed_op(LingbotControlSignal.INITIALIZE)
    def _initialize_sync_all_ranks(self) -> None:
        self._initialize_sync()

    @distributed_op(LingbotControlSignal.RESET_SESSION)
    def _reset_rollout_sync_all_ranks(self) -> None:
        self._reset_rollout_sync()

    @distributed_op(LingbotControlSignal.ACTION_STEP)
    def _generate_chunk_sync_all_ranks(
        self,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> LingbotStepResult:
        return self._generate_one_chunk_sync(segments=segments, frame_times=frame_times)

    @distributed_op(LingbotControlSignal.CLOSE)
    def _close_sync_all_ranks(self) -> None:
        self._close_sync()

    def _initialize_sync(self) -> None:
        if self._pipeline is not None:
            return

        data_dir = self.config.example_data_dir
        first_frame_path = data_dir / self.config.first_frame_filename
        intrinsics_path = data_dir / self.config.intrinsics_filename
        poses_path = data_dir / self.config.poses_filename
        prompt_path = data_dir / self.config.prompt_filename

        missing_paths = [
            str(path)
            for path in (first_frame_path, intrinsics_path, prompt_path)
            if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "Missing Lingbot example assets: " + ", ".join(missing_paths)
            )

        if self.config.config_name not in PIPELINE_CONFIGS:
            supported = ", ".join(sorted(PIPELINE_CONFIGS))
            raise ValueError(
                f"Unknown config_name={self.config.config_name!r}. Supported: {supported}"
            )

        self._device = torch.device(self.config.device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Lingbot runtime.")

        image_bgr = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read first frame from {first_frame_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Bicubic to match the upstream Lingbot World demo / generate_fast.py
        # (which uses ``F.interpolate(mode='bicubic')`` over the ``[-1, 1]``
        # tensor); bilinear here would give a different first-frame VAE latent.
        image_rgb = cv2.resize(
            image_rgb,
            (self.config.video_width, self.config.video_height),
            interpolation=cv2.INTER_CUBIC,
        )
        first_frame_t = (
            torch.from_numpy(image_rgb).to(device=self._device, dtype=torch.bfloat16)
            / 127.5
            - 1.0
        )
        # Lingbot's shipped configs pin ``batch_shape=()`` (single-rollout
        # layout), so the pipeline expects the first frame in shape
        # ``[T=1, C, H, W]``; the leading ``unsqueeze(0)`` lifts ``[C, H, W]``
        # to that ``T=1`` axis the I2V encoder pads/slices against.
        first_frames_t = first_frame_t.permute(2, 0, 1).unsqueeze(0)

        intrinsics_np = np.load(intrinsics_path)
        if intrinsics_np.ndim == 1:
            base_intrinsics = intrinsics_np
        else:
            base_intrinsics = intrinsics_np[0]
        if base_intrinsics.shape != (4,):
            raise ValueError(
                f"Expected base intrinsics shape (4,), got {base_intrinsics.shape}"
            )
        # The provided intrinsics are stored at the original 480x832 image
        # size; rescale them to the inference resolution so Plücker rays
        # land on the right pixel centers.
        base_intrinsics_t = torch.from_numpy(base_intrinsics).to(
            device=self._device, dtype=torch.float32
        )
        base_intrinsics_t = get_Ks_transformed(
            base_intrinsics_t.view(1, 4),
            height_org=480,
            width_org=832,
            height_resize=self.config.video_height,
            width_resize=self.config.video_width,
            height_final=self.config.video_height,
            width_final=self.config.video_width,
        ).view(4)
        self._base_intrinsics = base_intrinsics_t

        with prompt_path.open("r", encoding="utf-8") as handle:
            prompt = handle.readline().strip()
        if not prompt:
            raise ValueError("Prompt file is empty.")

        if self.config.world_scale is not None:
            self._world_scale = float(self.config.world_scale)
        elif poses_path.exists():
            # Match upstream: world-scale normalizer is computed on the
            # encoded-length poses, not the raw stream. The returned
            # ``poses`` array (per-pixel-frame cadence) is unused here —
            # webrtc generates poses live via :class:`CameraPoseIntegrator`.
            _, self._world_scale = preprocess_example_poses(np.load(poses_path))
            if self._world_scale <= 0:
                self._world_scale = 1.0

        rollout_seed = (
            self.config.seed + self.rank
            if self.config.context_parallel_size > 1
            else self.config.seed
        )
        pipeline_config = derive_config(
            base_config=PIPELINE_CONFIGS[self.config.config_name],
            enable_sync_and_profile=True,
            diffusion_model=dict(
                seed=rollout_seed,
                transformer=dict(compile_network=self.config.compile_network),
            ),
        )
        self._pipeline = pipeline_config.setup().to(device=self._device)
        self._first_frames = first_frames_t
        self._prompt = prompt
        self._reset_rollout_sync()

    def _reset_rollout_sync(self) -> None:
        if self._pipeline is None:
            raise LingbotRuntimeError("Runtime pipeline is not initialized.")
        if self._first_frames is None or self._prompt is None:
            raise LingbotRuntimeError("Runtime input state is not initialized.")

        if self._cache is not None:
            del self._cache
            self._cache = None

        self.pose_integrator = CameraPoseIntegrator()
        self.autoregressive_index = 0
        self._cache = self._pipeline.initialize_cache(
            text=[self._prompt],
            image=self._first_frames,
        )

    def _close_sync(self) -> None:
        cache = self._cache
        pipeline = self._pipeline
        self._cache = None
        self._pipeline = None
        self._base_intrinsics = None
        self._first_frames = None
        self._prompt = None

        if cache is not None:
            del cache
        if pipeline is not None:
            del pipeline

        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(device=self._device)
            torch.cuda.empty_cache()

    def _generate_one_chunk_sync(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> LingbotStepResult:
        if (
            self._pipeline is None
            or self._cache is None
            or self._base_intrinsics is None
        ):
            raise LingbotRuntimeError("Runtime is not initialized.")
        if self._device is None:
            raise LingbotRuntimeError("Runtime device is not initialized.")

        num_frames = int(
            self._pipeline.get_num_output_frames(self.autoregressive_index)
        )
        if len(frame_times) != num_frames:
            raise LingbotRuntimeError(
                f"Expected {num_frames} frame_times for "
                f"chunk={self.autoregressive_index}, got {len(frame_times)}."
            )
        if not segments:
            raise LingbotRuntimeError(
                f"Chunk={self.autoregressive_index} received empty segments."
            )
        poses = self.pose_integrator.integrate_chunk(
            segments=segments, frame_times=frame_times
        )
        poses_t = torch.from_numpy(poses).to(device=self._device, dtype=torch.float32)
        poses_t = poses_t.view(num_frames, 4, 4)
        intrinsics_t = self._base_intrinsics.view(1, 4).repeat(num_frames, 1)

        camctrl_input = CamCtrlInput(
            intrinsics=intrinsics_t,
            poses=poses_t,
            world_scale=self._world_scale,
        )
        video_chunk = self._pipeline.generate(
            autoregressive_index=self.autoregressive_index,
            cache=self._cache,
            input=camctrl_input,
        )
        stats = self._pipeline.finalize(self.autoregressive_index, self._cache)

        result = LingbotStepResult(
            chunk_index=self.autoregressive_index,
            num_frames=num_frames,
            video_chunk=video_chunk.detach().cpu(),
            stats=stats,
        )
        self.autoregressive_index += 1
        return result


@dataclass(slots=True)
class _ManagedLingbotSession:
    runtime: LingbotInferenceRuntime
    video_track: BufferedVideoTrack
    peer_connection: Any
    resampler: KeyboardResampler
    """Per-session sparse-edge resampler; produces the per-frame keyboard
    states consumed by :meth:`LingbotInferenceRuntime.generate_chunk`."""

    control_channel: Any | None = None
    generation_task: asyncio.Task[Any] | None = None
    """Long-running coroutine that wallclock-aligns chunk generation;
    started after the data channel opens and cancelled on ``close``."""

    first_action_received: asyncio.Event = field(default_factory=asyncio.Event)
    """Set by :meth:`_handle_datachannel_message` the first time a valid
    ``keydown``/``keyup`` event lands in the resampler. The generation
    worker blocks on this before kicking off chunk 0 so the server stays
    completely idle until the user actually interacts — matching the
    "no video until first action" behaviour the old pull-driven worker
    used to give. After the wait the worker re-anchors the resampler's
    virtual clock to ``loop.time()`` so chunk 0's window starts at the
    moment of first interaction, not at data-channel open time."""

    pending_action_arrivals: deque[float] = field(default_factory=deque)
    """Accepted control-edge arrival times not yet reported in latency telemetry."""

    last_client_message_at: float = 0.0
    liveness_task: asyncio.Task[Any] | None = None
    """Watchdog that closes the session when browser heartbeats stop."""

    closed: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        current_task = asyncio.current_task()
        if (
            self.liveness_task is not None
            and self.liveness_task is not current_task
            and not self.liveness_task.done()
        ):
            self.liveness_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.liveness_task
        self.liveness_task = None

        if (
            self.generation_task is not None
            and self.generation_task is not current_task
            and not self.generation_task.done()
        ):
            self.generation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.generation_task
        self.generation_task = None

        await self.video_track.close()
        await self.peer_connection.close()


class LingbotWebRTCSessionManager:
    """Owns one active WebRTC session and forwards actions into Lingbot runtime."""

    def __init__(
        self,
        *,
        runtime_config: LingbotRuntimeConfig | None = None,
        fps: int = 16,
        client_liveness_timeout_s: float = DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if client_liveness_timeout_s <= 0:
            raise ValueError("client_liveness_timeout_s must be > 0")
        self.runtime_config = runtime_config or LingbotRuntimeConfig()
        self.fps = fps
        self.client_liveness_timeout_s = client_liveness_timeout_s
        self._runtime = LingbotInferenceRuntime(config=self.runtime_config)
        self._runtime_ready = False
        self._warmup_complete = False
        self._active_session: _ManagedLingbotSession | None = None
        self._preload_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()

    def has_active_session(self) -> bool:
        return self._active_session is not None and not self._active_session.closed

    def is_runtime_ready(self) -> bool:
        return self._runtime_ready

    async def preload_runtime(self) -> None:
        async with self._preload_lock:
            if not self._runtime_ready:
                await self._runtime.initialize()
                self._runtime_ready = True
            if not self._warmup_complete:
                await self._run_loopback_warmup_session(
                    num_chunks=self.runtime_config.warmup_chunks
                )
                self._warmup_complete = True

    async def create_answer(self, *, offer_sdp: str, offer_type: str) -> dict[str, str]:
        if not self._runtime_ready or not self._warmup_complete:
            await self.preload_runtime()

        async with self._session_lock:
            if self._active_session is not None and not self._active_session.closed:
                raise SessionBusyError("A Lingbot session is already active.")

            return await self._create_answer_with_runtime_ready_locked(
                offer_sdp=offer_sdp,
                offer_type=offer_type,
            )

    async def _create_answer_with_runtime_ready_locked(
        self,
        *,
        offer_sdp: str,
        offer_type: str,
        rtc_configuration: RTCConfiguration | None = None,
        enable_liveness_watchdog: bool = True,
    ) -> dict[str, str]:
        if self._active_session is not None and not self._active_session.closed:
            raise SessionBusyError("A Lingbot session is already active.")
        if not self._runtime_ready:
            raise LingbotRuntimeError("Runtime is not initialized.")

        await self._runtime.reset_for_new_session()

        peer_connection = RTCPeerConnection(rtc_configuration)
        # Bounded queue sized to one *steady-state* chunk: ``put``
        # blocks only when the queue holds a full steady-state chunk
        # already, which throttles the producer to the consumer's
        # drain rate.
        #
        # Important: AR step 0 emits fewer frames than every
        # subsequent step (e.g. 9 vs 12 here) due to the decoder's
        # causal first-frame padding. Sizing the queue to AR 0 would
        # force the producer to block 3 times on *every* steady-state
        # chunk, leaving < 1 chunk of buffer at gen-start and
        # producing a once-per-chunk ~60 ms playback stall. We
        # therefore size to the steady-state count.
        num_frames = self._runtime.peek_steady_chunk_num_frames()
        video_track = BufferedVideoTrack(fps=self.fps, maxsize=num_frames)
        peer_connection.addTrack(video_track)
        # Start the resampler's virtual clock at 0; the real anchor
        # is set inside the ``on_datachannel`` handler so chunk 0's
        # window starts at the moment input can actually arrive.
        # Anchoring earlier (at offer time) would make the first few
        # chunks integrate over an empty pre-channel window.
        resampler = KeyboardResampler(fps=self.fps, start_v=0.0)
        loop = asyncio.get_running_loop()
        managed_session = _ManagedLingbotSession(
            runtime=self._runtime,
            video_track=video_track,
            peer_connection=peer_connection,
            resampler=resampler,
            last_client_message_at=loop.time(),
        )
        self._active_session = managed_session
        if enable_liveness_watchdog:
            managed_session.liveness_task = asyncio.create_task(
                self._client_liveness_watchdog(managed_session=managed_session)
            )

        @peer_connection.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            managed_session.control_channel = channel
            # Belt-and-braces reset of the resampler at channel
            # open: the resampler is freshly constructed in
            # ``create_answer`` so this is normally a no-op, but
            # clearing here guarantees a clean event log even if
            # the resampler lifecycle ever changes. The real
            # virtual-clock anchor happens inside
            # ``_generation_worker`` once the first keyboard event
            # arrives so chunk 0's window starts at the moment of
            # first interaction, not at data-channel open.
            channel_open_v = asyncio.get_running_loop().time()
            managed_session.resampler.reset(start_v=channel_open_v)

            @channel.on("message")
            def on_message(message: Any) -> None:
                asyncio.create_task(
                    self._handle_datachannel_message(
                        managed_session=managed_session,
                        raw_message=message,
                    )
                )

            # Spawn the generation worker once the data channel has
            # been wired up so ``chunk_done`` notifications have a
            # channel to land on. The worker is per-session and
            # cancelled in :meth:`_ManagedLingbotSession.close`.
            managed_session.generation_task = asyncio.create_task(
                self._generation_worker(managed_session=managed_session)
            )

            @channel.on("close")
            def on_close() -> None:
                logger.info("Control data channel closed; closing active session.")
                asyncio.create_task(self.close_active_session())

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if peer_connection.connectionState in {
                "failed",
                "disconnected",
                "closed",
            }:
                await self.close_active_session()

        try:
            offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
            await peer_connection.setRemoteDescription(offer)
            answer = await peer_connection.createAnswer()
            await peer_connection.setLocalDescription(answer)
            await wait_for_ice_gathering_complete(peer_connection)
            local_description = peer_connection.localDescription
            if local_description is None:
                raise RuntimeError("Peer connection did not produce local description.")
            return {"sdp": local_description.sdp, "type": local_description.type}
        except Exception:
            logger.exception("WebRTC negotiation failed while creating an answer.")
            await managed_session.close()
            self._active_session = None
            raise

    async def _run_loopback_warmup_session(self, *, num_chunks: int) -> None:
        if not self._runtime_ready:
            raise LingbotRuntimeError("Runtime is not initialized.")
        await run_loopback_warmup_session(
            num_chunks=num_chunks,
            warmup_timeout_s=self.runtime_config.warmup_timeout_s,
            create_answer=self._create_loopback_warmup_answer,
            close_active_session=self.close_active_session,
            label="Lingbot WebRTC",
            logger=logger,
        )

    async def _create_loopback_warmup_answer(
        self, *, offer_sdp: str, offer_type: str
    ) -> dict[str, str]:
        async with self._session_lock:
            return await self._create_answer_with_runtime_ready_locked(
                offer_sdp=offer_sdp,
                offer_type=offer_type,
                rtc_configuration=RTCConfiguration(iceServers=[]),
                enable_liveness_watchdog=False,
            )

    async def close_active_session(self) -> None:
        async with self._session_lock:
            if self._active_session is None:
                return
            active_session = self._active_session
            self._active_session = None
            await active_session.close()

    async def _client_liveness_watchdog(
        self, *, managed_session: _ManagedLingbotSession
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            while not managed_session.closed:
                elapsed_s = loop.time() - managed_session.last_client_message_at
                if elapsed_s >= self.client_liveness_timeout_s:
                    logger.warning(
                        "No client heartbeat/control message for {:.1f}s; "
                        "closing active session.",
                        elapsed_s,
                    )
                    await self.close_active_session()
                    return
                await asyncio.sleep(
                    min(
                        _CLIENT_LIVENESS_CHECK_INTERVAL_S,
                        self.client_liveness_timeout_s - elapsed_s,
                    )
                )
        except asyncio.CancelledError:
            raise

    async def shutdown(self) -> None:
        await self.close_active_session()
        await self._runtime.close()
        self._runtime_ready = False
        self._warmup_complete = False

    def wait_for_termination(self) -> None:
        self._runtime.wait_for_termination()

    def send_exit_signal(self) -> None:
        self._runtime.send_exit_signal()

    async def _handle_datachannel_message(
        self,
        *,
        managed_session: _ManagedLingbotSession,
        raw_message: Any,
    ) -> None:
        channel = managed_session.control_channel
        if channel is None or managed_session.closed:
            return
        managed_session.last_client_message_at = asyncio.get_running_loop().time()

        if not isinstance(raw_message, str):
            self._send_json(
                channel, {"type": "error", "message": "Expected text payload."}
            )
            return

        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            self._send_json(
                channel, {"type": "error", "message": "Invalid JSON payload."}
            )
            return

        if not isinstance(payload, dict):
            self._send_json(
                channel, {"type": "error", "message": "Payload must be a JSON object."}
            )
            return
        message_type = str(payload.get("type", "")).strip().lower()
        if message_type == "heartbeat":
            return
        if message_type == "disconnect":
            logger.info("Client requested disconnect; closing active session.")
            await self.close_active_session()
            return
        if message_type != "action":
            self._send_json(
                channel,
                {
                    "type": "error",
                    "message": "Unsupported message type, expected "
                    "'action', 'heartbeat', or 'disconnect'.",
                },
            )
            return

        action_payload = payload.get("action", payload)
        if not isinstance(action_payload, dict):
            self._send_json(
                channel, {"type": "error", "message": "'action' must be an object."}
            )
            return

        event = str(action_payload.get("event", "")).strip().lower()
        # ``step`` payloads were previously emitted by an older browser
        # client on every ``chunk_done`` round trip. The server-side
        # generation worker now drives the pipeline directly, so they
        # are accepted silently as no-ops to avoid breaking older clients.
        if event == "step":
            logger.debug("Ignoring legacy 'step' control payload.")
            return
        if event not in ("keydown", "keyup"):
            self._send_json(
                channel,
                {
                    "type": "error",
                    "message": f"Unsupported event={event!r}; "
                    "expected 'keydown' or 'keyup'.",
                },
            )
            return
        key = str(action_payload.get("key", "")).strip()
        if not key:
            self._send_json(
                channel,
                {
                    "type": "error",
                    "message": "Action payload must include non-empty 'key'.",
                },
            )
            return

        # Stamp arrival on the same monotonic clock that seeds the
        # resampler's ``next_chunk_start_v`` so virtual-time comparisons
        # in :meth:`KeyboardResampler.sample_chunk` are well-defined.
        arrival_t = asyncio.get_running_loop().time()
        managed_session.resampler.on_edge(arrival_t=arrival_t, event=event, key=key)
        managed_session.pending_action_arrivals.append(arrival_t)
        logger.info(
            "Logged control event={} key={} arrival_t={:.3f} log_size={}",
            event,
            key,
            arrival_t,
            managed_session.resampler.event_log_size(),
        )
        # Releases the generation worker, which blocks on this event
        # until the user actually interacts. Idempotent: ``Event.set``
        # is a no-op once already set.
        managed_session.first_action_received.set()

    async def _generation_worker(
        self, *, managed_session: _ManagedLingbotSession
    ) -> None:
        """Drive back-to-back chunk generation aligned to the resampler clock.

        Sits idle until the first keyboard event arrives, then drives
        the chunk loop. Each iteration waits for wallclock to catch up
        to the *end* of the next chunk's virtual window
        (``V_{N+1} = V_N + num_frames * dt``), samples the chunk's
        piecewise-constant timeline, hands segments and frame times to
        the runtime, and pushes the generated frames into the video
        track. Triggering at the window end (instead of the window's
        last frame time) guarantees every keyboard edge whose
        ``arrival_t`` falls inside the chunk has a chance to land in
        the timeline before sampling. The track's bounded queue then
        paces the loop to playback via backpressure on
        :meth:`BufferedVideoTrack.enqueue_chunk`.
        """
        loop = asyncio.get_running_loop()
        runtime = managed_session.runtime
        resampler = managed_session.resampler
        video_track = managed_session.video_track

        # Stay idle until the user actually interacts. Generating
        # eagerly would burn GPU cycles producing a still scene the
        # viewer never sees (chunks would sit in the queue but recv
        # blocks anyway until aiortc requests a frame). Once an event
        # arrives we re-anchor the resampler's virtual clock to ``now``
        # so chunk 0's window starts at the moment of first interaction,
        # not at data-channel open. ``on_edge`` already journalled the
        # triggering event with ``arrival_t < now``, so the resampler's
        # drain path folds it into ``carried_state`` and chunk 0's
        # segments reflect the held-key state from frame 0.
        logger.info("Generation worker idle; waiting for first action.")
        try:
            await managed_session.first_action_received.wait()
        except asyncio.CancelledError:
            logger.info("Generation worker cancelled before first action.")
            raise
        if managed_session.closed:
            return
        resampler.next_chunk_start_v = loop.time()
        logger.info(
            "First action received; starting generation at start_v={:.3f}",
            resampler.next_chunk_start_v,
        )
        try:
            while not managed_session.closed:
                try:
                    num_frames = runtime.peek_next_chunk_num_frames()
                except LingbotRuntimeError:
                    logger.exception("Runtime not ready; stopping generation worker.")
                    return
                # Trigger when wallclock reaches the chunk's window end
                # (= the next chunk's start virtual time). Earlier
                # triggers truncate the chunk's last dt of events;
                # later triggers just add idle slack between chunks.
                chunk_duration = num_frames * resampler.dt
                trigger_wall = resampler.next_chunk_start_v + chunk_duration
                delay = trigger_wall - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                if managed_session.closed:
                    break

                # Catch the virtual clock up to wall if it has fallen
                # more than one chunk behind. Bootstrap (first chunk's
                # CUDA-graph warmup) and transient stalls both push
                # ``next_chunk_start_v`` arbitrarily far behind
                # ``loop.time()``; without rewinding, every subsequent
                # chunk samples a stale window of events and end-to-end
                # latency stays pinned to the worst-case stall forever.
                # Skipping ahead drops *tap* events that fell entirely
                # in the gap (matching the "stalls are fine" stance);
                # held-key continuity is still preserved because
                # ``KeyboardResampler.sample_chunk`` folds every event
                # below the new window start into ``carried_state``.
                now = loop.time()
                lag = now - (resampler.next_chunk_start_v + chunk_duration)
                if lag > chunk_duration:
                    skipped_to = now - chunk_duration
                    logger.warning(
                        "Resampler virtual clock lagging wall by {:.3f}s; "
                        "skipping next_chunk_start_v {:.3f} -> {:.3f} to "
                        "track wall and keep input-to-pixel latency bounded.",
                        lag,
                        resampler.next_chunk_start_v,
                        skipped_to,
                    )
                    resampler.next_chunk_start_v = skipped_to

                t_before_gen = loop.time()
                segments, frame_times = resampler.sample_chunk(num_frames)
                chunk_end_v = resampler.next_chunk_start_v
                consumed_action_arrivals: list[float] = []
                while (
                    managed_session.pending_action_arrivals
                    and managed_session.pending_action_arrivals[0] <= chunk_end_v
                ):
                    consumed_action_arrivals.append(
                        managed_session.pending_action_arrivals.popleft()
                    )
                try:
                    result = await runtime.generate_chunk(
                        segments=segments, frame_times=frame_times
                    )
                except Exception as exc:
                    logger.exception("Chunk generation failed.")
                    channel = managed_session.control_channel
                    if channel is not None:
                        self._send_json(channel, {"type": "error", "message": str(exc)})
                    continue
                t_after_gen = loop.time()
                enqueued = await video_track.enqueue_chunk(result.video_chunk)
                t_after_enqueue = loop.time()

                gen_ms = (t_after_gen - t_before_gen) * 1e3
                enqueue_ms = (t_after_enqueue - t_after_gen) * 1e3
                play_ms = result.num_frames * 1000.0 / video_track.fps
                # Diagnostic: how far behind wall the resampler's
                # virtual clock is at the END of this chunk. In steady
                # state this should hover around one ``chunk_duration``
                # (the worker triggers at chunk_end_v, then spends
                # ``gen_ms + enqueue_ms`` advancing wall); a value that
                # keeps growing indicates the catch-up branch isn't
                # firing and end-to-end latency will degrade.
                lag_ms = (t_after_enqueue - resampler.next_chunk_start_v) * 1e3
                control_latency_ms = (
                    (t_after_enqueue - consumed_action_arrivals[0]) * 1e3
                    if consumed_action_arrivals
                    else None
                )
                logger.debug(
                    "Chunk done chunk={} num_frames={} segments={} "
                    "enqueued={} gen_ms={:.1f} enqueue_ms={:.1f} play_ms={:.1f} "
                    "queue_depth={} next_v={:.3f} wall={:.3f} lag_ms={:.1f} "
                    "control_latency_ms={} consumed_actions={}",
                    result.chunk_index,
                    result.num_frames,
                    len(segments),
                    enqueued,
                    gen_ms,
                    enqueue_ms,
                    play_ms,
                    video_track.qsize(),
                    resampler.next_chunk_start_v,
                    t_after_enqueue,
                    lag_ms,
                    (
                        f"{control_latency_ms:.1f}"
                        if control_latency_ms is not None
                        else "n/a"
                    ),
                    len(consumed_action_arrivals),
                )

                channel = managed_session.control_channel
                if channel is not None:
                    payload: dict[str, Any] = {
                        "type": "chunk_done",
                        "chunk_index": result.chunk_index,
                        "num_frames": result.num_frames,
                        "enqueued_frames": enqueued,
                        "fps": video_track.fps,
                        "resolution": {
                            "width": self.runtime_config.video_width,
                            "height": self.runtime_config.video_height,
                        },
                        "model": self.runtime_config.config_name,
                        "gen_ms": round(gen_ms, 1),
                        "enqueue_ms": round(enqueue_ms, 1),
                        "play_ms": round(play_ms, 1),
                        "queue_depth": video_track.qsize(),
                        "lag_ms": round(lag_ms, 1),
                    }
                    if control_latency_ms is not None:
                        payload["latency_ms"] = round(control_latency_ms, 1)
                        payload["control_latency_ms"] = round(control_latency_ms, 1)
                        payload["consumed_actions"] = len(consumed_action_arrivals)
                    self._send_json(channel, payload)
        except asyncio.CancelledError:
            logger.info("Generation worker cancelled.")
            raise

    @staticmethod
    def _send_json(channel: Any, payload: dict[str, Any]) -> None:
        try:
            channel.send(json.dumps(payload))
        except Exception:
            # If the data channel is closing we just drop the message.
            return
