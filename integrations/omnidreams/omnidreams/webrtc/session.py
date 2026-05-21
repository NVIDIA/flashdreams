# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, TypeVar

import cv2
import numpy as np
import torch
import torch.distributed as dist
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from loguru import logger
from omnidreams.conditioning.conditioning_wrapper import (
    AV_POSITIVE_PROMPT,
    OmnidreamsConditioningState,
    OmnidreamsConditioningWrapper,
    TextPrompt,
)
from omnidreams.conditioning.renderer import load_and_attach_ludus_scene
from omnidreams.conditioning.world_scenario.data_loaders import load_scene
from omnidreams.conditioning.world_scenario.settings import SETTINGS
from omnidreams.config import OMNIDREAMS_CONFIGS
from omnidreams.transformer import CosmosTransformerConfig

from flashdreams.core.distributed.rank_orchestration import (
    RankCoordinator,
    distributed_op,
)
from flashdreams.serving.webrtc.controls import (
    WSAD_SUPPORTED_KEYS,
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

REPO_ROOT = Path(__file__).resolve().parents[4]
_T = TypeVar("_T")
DEFAULT_CLIENT_LIVENESS_TIMEOUT_S = 10.0
_CLIENT_LIVENESS_CHECK_INTERVAL_S = 1.0


def _summarize_sdp_candidates(sdp: str) -> str:
    candidates = [
        line.removeprefix("a=candidate:")
        for line in sdp.splitlines()
        if line.startswith("a=candidate:")
    ]
    if not candidates:
        return "0 candidates"

    protocols: dict[str, int] = {}
    addresses: set[str] = set()
    endpoints: list[str] = []
    for candidate in candidates:
        parts = candidate.split()
        if len(parts) >= 5:
            protocols[parts[2].lower()] = protocols.get(parts[2].lower(), 0) + 1
            addresses.add(parts[4])
        if len(parts) >= 6:
            endpoints.append(f"{parts[2].lower()}://{parts[4]}:{parts[5]}")
    protocol_summary = ",".join(
        f"{key}={value}" for key, value in sorted(protocols.items())
    )
    address_summary = ",".join(sorted(addresses)[:8])
    if len(addresses) > 8:
        address_summary += f",+{len(addresses) - 8} more"
    endpoint_summary = ",".join(endpoints[:12])
    if len(endpoints) > 12:
        endpoint_summary += f",+{len(endpoints) - 12} more"
    return (
        f"{len(candidates)} candidates protocols=[{protocol_summary}] "
        f"addresses=[{address_summary}] endpoints=[{endpoint_summary}]"
    )


class OmnidreamsRuntimeError(RuntimeError):
    """Raised when the Omnidreams WebRTC runtime is used incorrectly."""


class OmnidreamsControlSignal(IntEnum):
    INITIALIZE = 0
    RESET_SESSION = 1
    ACTION_STEP = 2
    CLOSE = 3
    EXIT = 4


@dataclass(slots=True)
class OmnidreamsRuntimeConfig:
    pipeline_config_name: str = (
        "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"
    )
    scene_dir: Path = (
        REPO_ROOT
        / "assets"
        / "example_data"
        / "omnidreams-webrtc"
        / "0d404ff7-2b66-498c-b047-1ed8cded60d4"
    )
    seed: int | None = 42
    device: str = "cuda:0"
    video_height: int = 704
    video_width: int = 1280
    fps: int = 30
    camera_name: str = "camera_front_wide_120fov"
    first_frame_filename: str = "first_frame.jpeg"
    prompt_filename: str = "prompt.txt"
    clipgt_dirname: str = "clipgt"
    move_speed_per_s: float = 6.0
    rotate_speed_rad_per_s: float = float(np.deg2rad(35.0))
    warmup_chunks: int = 10
    warmup_timeout_s: float = 600.0
    debug_serve_hdmaps: bool = False


@dataclass(slots=True)
class OmnidreamsStepResult:
    chunk_index: int
    num_frames: int
    video_chunk: torch.Tensor
    stats: dict[str, float] | None


class OmnidreamsInferenceRuntime:
    """Single-scene, single-view Omnidreams runtime for WebRTC control."""

    def __init__(self, config: OmnidreamsRuntimeConfig | None = None) -> None:
        self.config = config or OmnidreamsRuntimeConfig()
        self.MASTER_RANK = 0
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

        control_device = torch.device(self.config.device)
        if control_device.type == "cuda" and control_device.index is None:
            control_device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cuda:0"
            )

        self.pose_integrator = CameraPoseIntegrator(
            move_speed_per_s=self.config.move_speed_per_s,
            rotate_speed_rad_per_s=self.config.rotate_speed_rad_per_s,
            coordinate_system="FLU",
        )
        self.autoregressive_index = 0

        self._device: torch.device | None = None
        self._wrapper: OmnidreamsConditioningWrapper | None = None
        self._state: OmnidreamsConditioningState | None = None
        self._renderer: Any | None = None
        self._scene_data: Any | None = None
        self._initial_rgb_frames: torch.Tensor | None = None
        self._text_prompts: list[TextPrompt] | None = None
        self._camera_to_rig: torch.Tensor | None = None
        self._initial_ego_pose: np.ndarray | None = None
        self._next_timestamp_us: int = 0
        self._closed = False
        self._clipgt_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        # Keep every blocking runtime call on the same OS thread. This is not
        # for throughput: Omnidreams uses CUDA graph capture/replay through
        # torch.compile/cuDNN, and the captured state appears to depend on
        # thread-local CUDA/cuDNN context. Replacing this with
        # ``asyncio.to_thread`` lets the default executor move initialize,
        # warmup, and generation calls across workers; that was observed to
        # fail after a few chunks with
        # CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED followed by
        # cudaErrorStreamCaptureInvalidated during capture_end.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="omnidreams-webrtc-runtime",
        )

        self._step_lock = asyncio.Lock()
        self.rank_coordinator = RankCoordinator(
            device=control_device,
            signal_type=OmnidreamsControlSignal,
            is_master=self.is_master,
            master_rank=self.MASTER_RANK,
        )
        self.rank_coordinator.register_distributed_ops(self)

    @property
    def is_master(self) -> bool:
        return self.rank == self.MASTER_RANK

    def wait_for_termination(self) -> None:
        self.rank_coordinator.worker_loop(exit_signal=OmnidreamsControlSignal.EXIT)

    def send_exit_signal(self) -> None:
        if self.is_master:
            self.rank_coordinator.send_exit(exit_signal=OmnidreamsControlSignal.EXIT)

    async def initialize(self) -> None:
        if self._wrapper is not None:
            return
        await self._run_on_runtime_thread(self._initialize_sync_all_ranks)

    async def reset_for_new_session(self) -> None:
        if self._closed:
            raise OmnidreamsRuntimeError("Runtime is closed.")
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        await self._run_on_runtime_thread(self._reset_rollout_sync_all_ranks)

    async def close(self) -> None:
        self._closed = True
        try:
            await self._run_on_runtime_thread(self._close_sync_all_ranks)
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)

    async def generate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> OmnidreamsStepResult:
        if self._closed:
            raise OmnidreamsRuntimeError("Session is closed.")
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")

        async with self._step_lock:
            if self._closed:
                raise OmnidreamsRuntimeError("Session is closed.")
            return await self._run_on_runtime_thread(
                self._generate_chunk_sync_all_ranks,
                segments,
                frame_times,
            )

    async def _run_on_runtime_thread(
        self,
        func: Callable[..., _T],
        *args: Any,
    ) -> _T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._runtime_thread_entry,
            func,
            args,
        )

    def _runtime_thread_entry(
        self,
        func: Callable[..., _T],
        args: tuple[Any, ...],
    ) -> _T:
        device = self._device
        if device is None:
            device = torch.device(self.config.device)
            if device.type == "cuda" and device.index is None:
                device = torch.device(
                    f"cuda:{torch.cuda.current_device()}"
                    if torch.cuda.is_available()
                    else "cuda:0"
                )
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return func(*args)

    def peek_next_chunk_num_frames(self) -> int:
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        if self._state is None:
            return int(self._wrapper.initial_frame_chunk_size)
        return int(self._wrapper.frame_chunk_size)

    def peek_steady_chunk_num_frames(self) -> int:
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        return int(self._wrapper.frame_chunk_size)

    @distributed_op(OmnidreamsControlSignal.INITIALIZE)
    def _initialize_sync_all_ranks(self) -> None:
        self._initialize_sync()

    @distributed_op(OmnidreamsControlSignal.RESET_SESSION)
    def _reset_rollout_sync_all_ranks(self) -> None:
        self._reset_rollout_sync()

    @distributed_op(OmnidreamsControlSignal.ACTION_STEP)
    def _generate_chunk_sync_all_ranks(
        self,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> OmnidreamsStepResult:
        return self._generate_one_chunk_sync(segments=segments, frame_times=frame_times)

    @distributed_op(OmnidreamsControlSignal.CLOSE)
    def _close_sync_all_ranks(self) -> None:
        self._close_sync()

    def _initialize_sync(self) -> None:
        if self._wrapper is not None:
            return

        cfg = self.config
        scene_dir = cfg.scene_dir
        clipgt_dir = scene_dir / cfg.clipgt_dirname
        first_frame_path = scene_dir / cfg.first_frame_filename
        prompt_path = scene_dir / cfg.prompt_filename
        missing_paths = [
            str(path)
            for path in (clipgt_dir, first_frame_path, prompt_path)
            if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "Missing Omnidreams WebRTC scene assets: " + ", ".join(missing_paths)
            )
        if cfg.pipeline_config_name not in OMNIDREAMS_CONFIGS:
            supported = ", ".join(sorted(OMNIDREAMS_CONFIGS))
            raise ValueError(
                f"Unknown pipeline_config_name={cfg.pipeline_config_name!r}. "
                f"Supported: {supported}"
            )

        pipeline_cfg = OMNIDREAMS_CONFIGS[cfg.pipeline_config_name]
        transformer_cfg = pipeline_cfg.diffusion_model.transformer
        if not isinstance(transformer_cfg, CosmosTransformerConfig):
            raise TypeError(
                "Omnidreams WebRTC requires a CosmosTransformerConfig pipeline."
            )
        if transformer_cfg.num_views != 1:
            raise ValueError(
                "Omnidreams WebRTC v1 only supports single-view configs; "
                f"{cfg.pipeline_config_name!r} has num_views={transformer_cfg.num_views}."
            )

        self._device = torch.device(cfg.device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Omnidreams WebRTC runtime.")

        image_bgr = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read first frame from {first_frame_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(
            image_rgb,
            (cfg.video_width, cfg.video_height),
            interpolation=cv2.INTER_CUBIC,
        )
        self._initial_rgb_frames = (
            torch.from_numpy(image_rgb)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self._device, dtype=torch.uint8)
        )

        prompt = prompt_path.read_text(encoding="utf-8").strip() or AV_POSITIVE_PROMPT
        self._text_prompts = [TextPrompt(positive=prompt)]

        loadable_clipgt_dir = self._prepare_clipgt_dir(clipgt_dir)
        scene_data = load_scene(
            loadable_clipgt_dir,
            camera_names=[cfg.camera_name],
            max_frames=-1,
            input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
            resize_resolution_hw=(cfg.video_height, cfg.video_width),
        )
        scene_data = load_and_attach_ludus_scene(
            loadable_clipgt_dir,
            scene_data,
            device=self._device,
        )
        if not scene_data.ego_poses:
            raise ValueError(f"Scene {loadable_clipgt_dir} has no ego poses.")
        if cfg.camera_name not in scene_data.camera_models:
            raise ValueError(
                f"Camera {cfg.camera_name!r} was not loaded from {loadable_clipgt_dir}."
            )
        if cfg.camera_name not in scene_data.camera_extrinsics:
            raise ValueError(
                f"Camera {cfg.camera_name!r} has no extrinsics in {loadable_clipgt_dir}."
            )

        self._wrapper = OmnidreamsConditioningWrapper(
            pipeline_config_name=cfg.pipeline_config_name,
            resolution_wh=(cfg.video_width, cfg.video_height),
            seed_for_every_rollout=cfg.seed,
            device=self._device,
        )
        self._scene_data = scene_data
        self._renderer = self._wrapper.create_renderer(scene_data, [cfg.camera_name])
        self._camera_to_rig = torch.as_tensor(
            scene_data.camera_extrinsics[cfg.camera_name],
            device=self._device,
            dtype=torch.float32,
        )
        self._initial_ego_pose = scene_data.ego_poses[0].transformation_matrix
        self._next_timestamp_us = int(scene_data.ego_poses[0].timestamp)
        self._reset_rollout_sync()

    def _prepare_clipgt_dir(self, clipgt_dir: Path) -> Path:
        if list(clipgt_dir.glob("*.calibration_estimate.parquet")):
            return clipgt_dir
        if not (clipgt_dir / "calibration_estimate.parquet").exists():
            return clipgt_dir

        self._clipgt_temp_dir = tempfile.TemporaryDirectory(prefix="omnidreams-clipgt-")
        staged = Path(self._clipgt_temp_dir.name)
        for source in clipgt_dir.glob("*.parquet"):
            target = staged / f"clip.{source.name}"
            os.symlink(source.resolve(), target)
        return staged

    def _reset_rollout_sync(self) -> None:
        if self._wrapper is None or self._renderer is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        if self._initial_ego_pose is None or self._scene_data is None:
            raise OmnidreamsRuntimeError("Scene state is not initialized.")

        if self._state is not None and self._state.pipeline_cache is not None:
            del self._state.pipeline_cache
        self._state = None
        self.pose_integrator = CameraPoseIntegrator(
            move_speed_per_s=self.config.move_speed_per_s,
            rotate_speed_rad_per_s=self.config.rotate_speed_rad_per_s,
            coordinate_system="FLU",
        )
        self.pose_integrator.reset(self._initial_ego_pose)
        self.autoregressive_index = 0
        self._next_timestamp_us = int(self._scene_data.ego_poses[0].timestamp)
        self._wrapper.set_rollout_seed(self.config.seed)

    def _close_sync(self) -> None:
        state = self._state
        wrapper = self._wrapper
        self._state = None
        self._wrapper = None
        self._renderer = None
        self._scene_data = None
        self._initial_rgb_frames = None
        self._text_prompts = None
        self._camera_to_rig = None
        self._initial_ego_pose = None

        if state is not None and wrapper is not None:
            wrapper.cleanup(state)
        if wrapper is not None:
            del wrapper
        if self._clipgt_temp_dir is not None:
            self._clipgt_temp_dir.cleanup()
            self._clipgt_temp_dir = None

        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(device=self._device)
            torch.cuda.empty_cache()

    def _generate_one_chunk_sync(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> OmnidreamsStepResult:
        if (
            self._wrapper is None
            or self._renderer is None
            or self._initial_rgb_frames is None
            or self._text_prompts is None
            or self._camera_to_rig is None
        ):
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        if self._device is None:
            raise OmnidreamsRuntimeError("Runtime device is not initialized.")

        num_frames = self.peek_next_chunk_num_frames()
        if len(frame_times) != num_frames:
            raise OmnidreamsRuntimeError(
                f"Expected {num_frames} frame_times for chunk={self.autoregressive_index}, "
                f"got {len(frame_times)}."
            )
        if not segments:
            raise OmnidreamsRuntimeError(
                f"Chunk={self.autoregressive_index} received empty segments."
            )

        ego_poses = self.pose_integrator.integrate_chunk(
            segments=segments, frame_times=frame_times
        )
        ego_poses_t = torch.from_numpy(ego_poses).to(
            device=self._device, dtype=torch.float32
        )
        camera_poses = torch.einsum("nij,jk->nik", ego_poses_t, self._camera_to_rig)
        frame_timestamps_us = self._consume_timestamps(num_frames)

        camera_names = [self.config.camera_name]
        camera_poses_per_view = {self.config.camera_name: camera_poses}
        serve_hdmaps = self.config.debug_serve_hdmaps
        if self._state is None:
            output = self._wrapper.start_generation(
                text_prompts=self._text_prompts,
                initial_rgb_frames=self._initial_rgb_frames,
                renderer=self._renderer,
                camera_names=camera_names,
                camera_poses_per_view=camera_poses_per_view,
                frame_timestamps_us=frame_timestamps_us,
                skip_video_generation=serve_hdmaps,
            )
            self._state = output.state
        else:
            output = self._wrapper.continue_generation(
                state=self._state,
                camera_names=camera_names,
                camera_poses_per_view=camera_poses_per_view,
                frame_timestamps_us=frame_timestamps_us,
                skip_video_generation=serve_hdmaps,
            )
            self._state = output.state

        if self._state.pipeline_cache is not None:
            self._wrapper.finalize_block_generation(
                self._state.pipeline_cache,
                output.finalization_state,
            )

        if serve_hdmaps:
            video_chunk = output.condition_frames
        elif output.rgb_frames is None:
            raise OmnidreamsRuntimeError("Omnidreams WebRTC received no RGB frames.")
        else:
            video_chunk = output.rgb_frames

        result = OmnidreamsStepResult(
            chunk_index=self.autoregressive_index,
            num_frames=int(video_chunk.shape[2]),
            video_chunk=video_chunk.detach().cpu(),
            stats=None,
        )
        self.autoregressive_index += 1
        return result

    def _consume_timestamps(self, num_frames: int) -> list[int]:
        step_us = int(round(1_000_000 / self.config.fps))
        timestamps = [self._next_timestamp_us + i * step_us for i in range(num_frames)]
        self._next_timestamp_us += num_frames * step_us
        return timestamps


@dataclass(slots=True)
class _ManagedOmnidreamsSession:
    runtime: OmnidreamsInferenceRuntime
    video_track: BufferedVideoTrack
    peer_connection: Any
    resampler: KeyboardResampler
    control_channel: Any | None = None
    generation_task: asyncio.Task[Any] | None = None
    first_action_received: asyncio.Event = field(default_factory=asyncio.Event)
    pending_action_arrivals: deque[float] = field(default_factory=deque)
    last_client_message_at: float = 0.0
    liveness_task: asyncio.Task[Any] | None = None
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


class OmnidreamsWebRTCSessionManager:
    """Owns one active WebRTC session and forwards WSAD actions."""

    def __init__(
        self,
        *,
        runtime_config: OmnidreamsRuntimeConfig | None = None,
        client_liveness_timeout_s: float = DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    ) -> None:
        if client_liveness_timeout_s <= 0:
            raise ValueError("client_liveness_timeout_s must be > 0")
        self.runtime_config = runtime_config or OmnidreamsRuntimeConfig()
        self.fps = self.runtime_config.fps
        self.client_liveness_timeout_s = client_liveness_timeout_s
        self._runtime = OmnidreamsInferenceRuntime(config=self.runtime_config)
        self._runtime_ready = False
        self._warmup_complete = False
        self._active_session: _ManagedOmnidreamsSession | None = None
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
                raise SessionBusyError("An Omnidreams session is already active.")

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
    ) -> dict[str, str]:
        if self._active_session is not None and not self._active_session.closed:
            raise SessionBusyError("An Omnidreams session is already active.")
        if not self._runtime_ready:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")

        await self._runtime.reset_for_new_session()

        peer_connection = RTCPeerConnection(rtc_configuration)
        num_frames = self._runtime.peek_steady_chunk_num_frames()
        video_track = BufferedVideoTrack(fps=self.fps, maxsize=num_frames)
        peer_connection.addTrack(video_track)
        resampler = KeyboardResampler(
            fps=self.fps,
            start_v=0.0,
            supported_keys=WSAD_SUPPORTED_KEYS,
        )
        loop = asyncio.get_running_loop()
        managed_session = _ManagedOmnidreamsSession(
            runtime=self._runtime,
            video_track=video_track,
            peer_connection=peer_connection,
            resampler=resampler,
            last_client_message_at=loop.time(),
        )
        self._active_session = managed_session
        managed_session.liveness_task = asyncio.create_task(
            self._client_liveness_watchdog(managed_session=managed_session)
        )

        @peer_connection.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            managed_session.control_channel = channel
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

            managed_session.generation_task = asyncio.create_task(
                self._generation_worker(managed_session=managed_session)
            )

            @channel.on("close")
            def on_close() -> None:
                logger.info("Control data channel closed; closing active session.")
                asyncio.create_task(self.close_active_session())

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.info(
                "Peer connection state changed: {}",
                peer_connection.connectionState,
            )
            if peer_connection.connectionState in {
                "failed",
                "disconnected",
                "closed",
            }:
                await self.close_active_session()

        @peer_connection.on("iceconnectionstatechange")
        def on_iceconnectionstatechange() -> None:
            logger.info(
                "Peer ICE connection state changed: {}",
                peer_connection.iceConnectionState,
            )

        @peer_connection.on("icegatheringstatechange")
        def on_icegatheringstatechange() -> None:
            logger.debug(
                "Peer ICE gathering state changed: {}",
                peer_connection.iceGatheringState,
            )

        try:
            offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
            logger.info(
                "Received WebRTC offer with {}.",
                _summarize_sdp_candidates(offer_sdp),
            )
            await peer_connection.setRemoteDescription(offer)
            answer = await peer_connection.createAnswer()
            await peer_connection.setLocalDescription(answer)
            await wait_for_ice_gathering_complete(peer_connection)
            local_description = peer_connection.localDescription
            if local_description is None:
                raise RuntimeError("Peer connection did not produce local description.")
            logger.info(
                "Created WebRTC answer with {}.",
                _summarize_sdp_candidates(local_description.sdp),
            )
            return {"sdp": local_description.sdp, "type": local_description.type}
        except Exception:
            logger.exception("WebRTC negotiation failed while creating an answer.")
            await managed_session.close()
            self._active_session = None
            raise

    async def _run_loopback_warmup_session(self, *, num_chunks: int) -> None:
        if not self._runtime_ready:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        await run_loopback_warmup_session(
            num_chunks=num_chunks,
            warmup_timeout_s=self.runtime_config.warmup_timeout_s,
            create_answer=self._create_loopback_warmup_answer,
            close_active_session=self.close_active_session,
            label="Omnidreams WebRTC",
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
            )

    async def close_active_session(self) -> None:
        async with self._session_lock:
            if self._active_session is None:
                return
            active_session = self._active_session
            self._active_session = None
            await active_session.close()

    async def _client_liveness_watchdog(
        self, *, managed_session: _ManagedOmnidreamsSession
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
        managed_session: _ManagedOmnidreamsSession,
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
        if event == "step":
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

        arrival_t = asyncio.get_running_loop().time()
        managed_session.resampler.on_edge(arrival_t=arrival_t, event=event, key=key)
        managed_session.pending_action_arrivals.append(arrival_t)
        managed_session.first_action_received.set()

    async def _generation_worker(
        self, *, managed_session: _ManagedOmnidreamsSession
    ) -> None:
        loop = asyncio.get_running_loop()
        runtime = managed_session.runtime
        resampler = managed_session.resampler
        video_track = managed_session.video_track

        logger.info("Generation worker idle; waiting for first WSAD action.")
        try:
            await managed_session.first_action_received.wait()
        except asyncio.CancelledError:
            logger.info("Generation worker cancelled before first action.")
            raise
        if managed_session.closed:
            return
        resampler.next_chunk_start_v = loop.time()

        try:
            while not managed_session.closed:
                try:
                    num_frames = runtime.peek_next_chunk_num_frames()
                except OmnidreamsRuntimeError:
                    logger.exception("Runtime not ready; stopping generation worker.")
                    return
                chunk_duration = num_frames * resampler.dt
                trigger_wall = resampler.next_chunk_start_v + chunk_duration
                delay = trigger_wall - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                if managed_session.closed:
                    break

                now = loop.time()
                lag = now - (resampler.next_chunk_start_v + chunk_duration)
                if lag > chunk_duration:
                    resampler.next_chunk_start_v = now - chunk_duration

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
                    logger.exception("Chunk generation failed; closing session.")
                    channel = managed_session.control_channel
                    if channel is not None:
                        self._send_json(channel, {"type": "error", "message": str(exc)})
                    await self.close_active_session()
                    return
                t_after_gen = loop.time()
                enqueued = await video_track.enqueue_chunk(result.video_chunk)
                t_after_enqueue = loop.time()

                gen_ms = (t_after_gen - t_before_gen) * 1e3
                enqueue_ms = (t_after_enqueue - t_after_gen) * 1e3
                play_ms = result.num_frames * 1000.0 / video_track.fps
                lag_ms = (t_after_enqueue - resampler.next_chunk_start_v) * 1e3
                control_latency_ms = (
                    (t_after_enqueue - consumed_action_arrivals[0]) * 1e3
                    if consumed_action_arrivals
                    else None
                )
                logger.info(
                    "Chunk done chunk={} num_frames={} segments={} "
                    "enqueued={} gen_ms={:.1f} enqueue_ms={:.1f} play_ms={:.1f} "
                    "queue_depth={} lag_ms={:.1f}",
                    result.chunk_index,
                    result.num_frames,
                    len(segments),
                    enqueued,
                    gen_ms,
                    enqueue_ms,
                    play_ms,
                    video_track.qsize(),
                    lag_ms,
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
                        "model": self.runtime_config.pipeline_config_name,
                        "stream": (
                            "hdmap" if self.runtime_config.debug_serve_hdmaps else "rgb"
                        ),
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
            return
