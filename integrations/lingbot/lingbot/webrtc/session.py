from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from flashdreams.recipes.lingbot_world.config import LINGBOT_WORLD_CONFIG_BUILDERS
from flashdreams.recipes.lingbot_world.encoder.camctrl import CamCtrlInput
from flashdreams.recipes.lingbot_world.encoder.utils import compute_relative_poses
from lingbot.webrtc.controls import CameraPoseIntegrator, KeyboardState
from lingbot.webrtc.media import LingbotVideoTrack

REPO_ROOT = Path(__file__).resolve().parents[4]
LOGGER = logging.getLogger(__name__)


class LingbotRuntimeError(RuntimeError):
    """Raised when the Lingbot runtime is used incorrectly."""


class SessionBusyError(RuntimeError):
    """Raised when a second peer tries to open a session."""


@dataclass(slots=True)
class LingbotRuntimeConfig:
    config_name: str = "LingBot-World-Fast"
    compile_network: bool = True
    seed: int = 42
    device: str = "cuda:0"
    video_height: int = 464
    video_width: int = 832
    world_scale: float | None = None

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

        self.keyboard_state = KeyboardState()
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

    async def initialize(self) -> None:
        if self._pipeline is not None:
            return
        await asyncio.to_thread(self._initialize_sync)

    async def reset_for_new_session(self) -> None:
        if self._closed:
            raise LingbotRuntimeError("Runtime is closed.")
        if self._pipeline is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        await asyncio.to_thread(self._reset_rollout_sync)

    async def close(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._close_sync)

    async def apply_actions_and_generate(
        self, actions: list[dict[str, Any]]
    ) -> LingbotStepResult:
        if self._closed:
            raise LingbotRuntimeError("Session is closed.")
        if self._pipeline is None or self._cache is None:
            raise LingbotRuntimeError("Runtime is not initialized.")

        for action in actions:
            event = str(action.get("event", "keydown")).strip().lower()
            if event == "step":
                LOGGER.info(
                    "Received step event with active_keys=%s",
                    sorted(self.keyboard_state.snapshot()),
                )
                continue
            key = str(action.get("key", "")).strip()
            if not key:
                raise LingbotRuntimeError(
                    "Action payload must include non-empty 'key' for keydown/keyup."
                )

            applied = self.keyboard_state.apply_event(event=event, key=key)
            if not applied:
                raise LingbotRuntimeError(
                    f"Unsupported action payload: event={event!r}, key={key!r}."
                )
            LOGGER.info(
                "Applied control event=%s key=%s active_keys=%s effective_keys=%s",
                event,
                key,
                sorted(self.keyboard_state.snapshot()),
                sorted(self.keyboard_state.resolved_effective_keys()),
            )

        async with self._step_lock:
            if self._closed:
                raise LingbotRuntimeError("Session is closed.")
            return await asyncio.to_thread(self._generate_one_chunk_sync)

    async def apply_action_and_generate(
        self, action: dict[str, Any]
    ) -> LingbotStepResult:
        """Backward-compatible single-action wrapper."""
        return await self.apply_actions_and_generate([action])

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

        if self.config.config_name not in LINGBOT_WORLD_CONFIG_BUILDERS:
            supported = ", ".join(sorted(LINGBOT_WORLD_CONFIG_BUILDERS))
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
        image_rgb = cv2.resize(
            image_rgb,
            (self.config.video_width, self.config.video_height),
            interpolation=cv2.INTER_LINEAR,
        )
        first_frame_t = (
            torch.from_numpy(image_rgb).to(device=self._device, dtype=torch.bfloat16)
            / 127.5
            - 1.0
        )
        first_frame_t = first_frame_t.permute(2, 0, 1).unsqueeze(0)
        first_frames_t = first_frame_t.unsqueeze(0).unsqueeze(0)  # [1, 1, 1, C, H, W]

        intrinsics_np = np.load(intrinsics_path)
        if intrinsics_np.ndim == 1:
            base_intrinsics = intrinsics_np
        else:
            base_intrinsics = intrinsics_np[0]
        if base_intrinsics.shape != (4,):
            raise ValueError(
                f"Expected base intrinsics shape (4,), got {base_intrinsics.shape}"
            )
        self._base_intrinsics = torch.from_numpy(base_intrinsics).to(
            device=self._device,
            dtype=torch.float32,
        )

        with prompt_path.open("r", encoding="utf-8") as handle:
            prompt = handle.readline().strip()
        if not prompt:
            raise ValueError("Prompt file is empty.")

        if self.config.world_scale is not None:
            self._world_scale = float(self.config.world_scale)
        elif poses_path.exists():
            poses = np.load(poses_path)
            poses_t = torch.from_numpy(poses).to(
                device=self._device, dtype=torch.float32
            )
            _, trans_normalizer = compute_relative_poses(poses_t, framewise=True)
            if isinstance(trans_normalizer, torch.Tensor):
                self._world_scale = float(trans_normalizer.item())
            else:
                self._world_scale = float(trans_normalizer)
            if self._world_scale <= 0:
                self._world_scale = 1.0

        builder = LINGBOT_WORLD_CONFIG_BUILDERS[self.config.config_name]
        self._pipeline = (
            builder(
                cp_size=1,
                compile_network=self.config.compile_network,
                seed=self.config.seed,
                enable_sync_and_profile=True,
            )
            .setup()
            .to(device=self._device)
        )
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

        self.keyboard_state = KeyboardState()
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

    def _generate_one_chunk_sync(self) -> LingbotStepResult:
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
        pressed_keys = self.keyboard_state.snapshot()
        effective_keys = self.keyboard_state.resolved_effective_keys()
        poses = self.pose_integrator.next_pose_chunk(
            num_frames=num_frames,
            pressed_keys=effective_keys,
        )
        first_pose = poses[0]
        last_pose = poses[-1]
        first_translation = first_pose[:3, 3].tolist()
        last_translation = last_pose[:3, 3].tolist()
        first_heading_y = float(np.arctan2(first_pose[0, 2], first_pose[0, 0]))
        last_heading_y = float(np.arctan2(last_pose[0, 2], last_pose[0, 0]))
        LOGGER.info(
            "Rendering chunk=%s num_frames=%s keys=%s effective_keys=%s first_xyz=%s last_xyz=%s first_heading_y=%.5f last_heading_y=%.5f",
            self.autoregressive_index,
            num_frames,
            sorted(pressed_keys),
            sorted(effective_keys),
            [round(float(x), 5) for x in first_translation],
            [round(float(x), 5) for x in last_translation],
            first_heading_y,
            last_heading_y,
        )
        LOGGER.info(
            "Chunk=%s first_pose=%s",
            self.autoregressive_index,
            np.array2string(first_pose, precision=4, suppress_small=True),
        )
        LOGGER.info(
            "Chunk=%s last_pose=%s",
            self.autoregressive_index,
            np.array2string(last_pose, precision=4, suppress_small=True),
        )
        poses_t = torch.from_numpy(poses).to(device=self._device, dtype=torch.float32)
        poses_t = poses_t.view(1, 1, num_frames, 4, 4)
        intrinsics_t = self._base_intrinsics.view(1, 1, 1, 4).repeat(
            1, 1, num_frames, 1
        )

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
    video_track: LingbotVideoTrack
    peer_connection: Any
    control_channel: Any | None = None
    action_task: asyncio.Task[Any] | None = None
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        if self.action_task is not None and not self.action_task.done():
            self.action_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.action_task
            self.action_task = None
        self.pending_actions.clear()

        await self.video_track.close()
        await self.peer_connection.close()


class LingbotWebRTCSessionManager:
    """Owns one active WebRTC session and forwards actions into Lingbot runtime."""

    def __init__(
        self,
        *,
        runtime_config: LingbotRuntimeConfig | None = None,
        fps: int = 16,
    ) -> None:
        self.runtime_config = runtime_config or LingbotRuntimeConfig()
        self.fps = fps
        self._runtime = LingbotInferenceRuntime(config=self.runtime_config)
        self._runtime_ready = False
        self._active_session: _ManagedLingbotSession | None = None
        self._session_lock = asyncio.Lock()

    def has_active_session(self) -> bool:
        return self._active_session is not None and not self._active_session.closed

    def is_runtime_ready(self) -> bool:
        return self._runtime_ready

    async def preload_runtime(self) -> None:
        if self._runtime_ready:
            return
        await self._runtime.initialize()
        self._runtime_ready = True

    async def create_answer(self, *, offer_sdp: str, offer_type: str) -> dict[str, str]:
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "aiortc is required for WebRTC signaling. Install aiortc dependency."
            ) from exc

        async with self._session_lock:
            if self._active_session is not None and not self._active_session.closed:
                raise SessionBusyError("A Lingbot session is already active.")

            if not self._runtime_ready:
                await self._runtime.initialize()
                self._runtime_ready = True
            await self._runtime.reset_for_new_session()

            peer_connection = RTCPeerConnection()
            video_track = LingbotVideoTrack(fps=self.fps)
            peer_connection.addTrack(video_track)
            managed_session = _ManagedLingbotSession(
                runtime=self._runtime,
                video_track=video_track,
                peer_connection=peer_connection,
            )
            self._active_session = managed_session

            @peer_connection.on("datachannel")
            def on_datachannel(channel: Any) -> None:
                managed_session.control_channel = channel

                @channel.on("message")
                def on_message(message: Any) -> None:
                    asyncio.create_task(
                        self._handle_datachannel_message(
                            managed_session=managed_session,
                            raw_message=message,
                        )
                    )

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
                local_description = peer_connection.localDescription
                if local_description is None:
                    raise RuntimeError(
                        "Peer connection did not produce local description."
                    )
                return {"sdp": local_description.sdp, "type": local_description.type}
            except Exception:
                LOGGER.exception("WebRTC negotiation failed while creating an answer.")
                await managed_session.close()
                self._active_session = None
                raise

    async def close_active_session(self) -> None:
        async with self._session_lock:
            if self._active_session is None:
                return
            active_session = self._active_session
            self._active_session = None
            await active_session.close()

    async def shutdown(self) -> None:
        await self.close_active_session()
        await self._runtime.close()
        self._runtime_ready = False

    async def _handle_datachannel_message(
        self,
        *,
        managed_session: _ManagedLingbotSession,
        raw_message: Any,
    ) -> None:
        channel = managed_session.control_channel
        if channel is None or managed_session.closed:
            return

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
        if payload.get("type") != "action":
            self._send_json(
                channel,
                {
                    "type": "error",
                    "message": "Unsupported message type, expected 'action'.",
                },
            )
            return

        action_payload = payload.get("action", payload)
        if not isinstance(action_payload, dict):
            self._send_json(
                channel, {"type": "error", "message": "'action' must be an object."}
            )
            return

        LOGGER.info("Incoming control payload: %s", action_payload)
        managed_session.pending_actions.append(action_payload)
        LOGGER.info(
            "Queued control payload count=%s latest=%s",
            len(managed_session.pending_actions),
            action_payload,
        )
        self._start_next_action_step(managed_session)

    async def _run_action_step(
        self,
        *,
        managed_session: _ManagedLingbotSession,
        action_payloads: list[dict[str, Any]],
    ) -> None:
        channel = managed_session.control_channel
        if channel is None or managed_session.closed:
            return

        LOGGER.info("Starting action step with payloads: %s", action_payloads)
        try:
            step_result = await managed_session.runtime.apply_actions_and_generate(
                action_payloads
            )
            enqueued_frames = await managed_session.video_track.enqueue_chunk(
                step_result.video_chunk
            )
            LOGGER.info(
                "Finished action step chunk=%s num_frames=%s enqueued_frames=%s",
                step_result.chunk_index,
                step_result.num_frames,
                enqueued_frames,
            )
            self._send_json(
                channel,
                {
                    "type": "chunk_done",
                    "chunk_index": step_result.chunk_index,
                    "num_frames": step_result.num_frames,
                    "enqueued_frames": enqueued_frames,
                },
            )
        except Exception as exc:
            LOGGER.exception("Action-bound Lingbot inference step failed.")
            self._send_json(channel, {"type": "error", "message": str(exc)})
        finally:
            managed_session.action_task = None
            self._start_next_action_step(managed_session)

    def _start_next_action_step(self, managed_session: _ManagedLingbotSession) -> None:
        if managed_session.closed:
            return
        if (
            managed_session.action_task is not None
            and not managed_session.action_task.done()
        ):
            return
        if not managed_session.pending_actions:
            return
        action_payloads = managed_session.pending_actions
        managed_session.pending_actions = []
        managed_session.action_task = asyncio.create_task(
            self._run_action_step(
                managed_session=managed_session,
                action_payloads=action_payloads,
            )
        )

    @staticmethod
    def _send_json(channel: Any, payload: dict[str, Any]) -> None:
        try:
            channel.send(json.dumps(payload))
        except Exception:
            # If the data channel is closing we just drop the message.
            return
