"""gRPC server for bbox-conditioned video generation.

This server implements the WorldModelService defined in video_model.proto.
It maintains session state for each client and handles:
- Session initialization with static HD map and initial frame
- Video chunk generation with dynamic objects and camera trajectories
"""

from __future__ import annotations

import argparse
import atexit
import gc
import os
import threading
import time
import traceback
import uuid
import warnings
from concurrent import futures
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable

import grpc
import numpy as np
import torch
import torch.distributed as dist
from alpadreams.conditioning.conditioning_wrapper import (
    AV_POSITIVE_PROMPT,
    AlpadreamsConditioningState,
    AlpadreamsConditioningWrapper,
    TextPrompt,
)
from alpadreams.conditioning.renderer import LudusRenderer
from alpadreams.conditioning.world_scenario.data_types import SceneData
from alpadreams.conditioning.world_scenario.ftheta import FThetaCamera
from alpadreams.grpc.profiling_server import (
    get_profiler,
    init_profiler,
    profiling_context,
)
from alpadreams.grpc.protos import common_pb2, video_model_pb2, video_model_pb2_grpc
from alpadreams.grpc.session_recorder import SessionRecorder
from alpadreams.grpc.utils import (
    camera_spec_to_ftheta,
    compute_camera_poses_from_rig,
    decode_image,
    dynamic_state_to_object_info,
    encode_image,
    get_external_ip,
    load_static_world_from_zip_bytes,
    parse_rig_to_camera,
    proto_to_dict,
    trajectory_to_camera_poses,
)
from loguru import logger
from ludus_renderer import nvjpeg

from flashdreams.core.distributed import init as distributed_init
from flashdreams.core.distributed.context_parallel import (
    cat_outputs_cp_object_list,
    split_inputs_cp_object_list,
)

VERBOSE = False

RESOLUTION_MAP: dict[str, tuple[int, int]] = {
    "480p": (832, 480),
    "720p": (1280, 720),
    "704p": (1280, 704),
}


def resolve_num_frames_per_block(
    *,
    n_cameras: int,
    encode_with_pixel_shuffle: bool,
    num_frames_per_block: int | None,
) -> int:
    """Resolve and validate temporal block size against available checkpoints."""
    if n_cameras not in (1, 4):
        raise ValueError(
            f"Only n_cameras in {{1, 4}} is supported by current checkpoints, got {n_cameras}"
        )

    if num_frames_per_block is None:
        # len_t = num_frames_per_block // 4. Any multi-view or pixel-shuffle path
        # currently requires len_t=4, i.e. num_frames_per_block=16.
        if n_cameras > 1 or encode_with_pixel_shuffle:
            return 16
        return 12

    if num_frames_per_block % 4 != 0:
        raise ValueError("num_frames_per_block must be divisible by 4.")

    len_t = num_frames_per_block // 4
    if n_cameras > 1 and len_t != 4:
        raise ValueError(
            "Multi-view checkpoints require len_t=4 "
            f"(got len_t={len_t} from num_frames_per_block={num_frames_per_block}). "
            "Use --num_frames_per_block 16."
        )
    if n_cameras == 1 and encode_with_pixel_shuffle and len_t != 4:
        raise ValueError(
            "Single-view pixel-shuffle checkpoints require len_t=4 "
            f"(got len_t={len_t} from num_frames_per_block={num_frames_per_block}). "
            "Use --num_frames_per_block 16."
        )
    if n_cameras == 1 and not encode_with_pixel_shuffle and len_t not in (2, 3):
        raise ValueError(
            "Single-view VAE-encoding checkpoints require len_t in {2, 3} "
            f"(got len_t={len_t} from num_frames_per_block={num_frames_per_block}). "
            "Use --num_frames_per_block 8 or 12."
        )
    return num_frames_per_block


class ControlSignal(IntEnum):
    START = 0
    VIDEO_CHUNK = 1
    CLOSE = 2
    EXIT = 3
    FINALIZE_KV = 4
    INVALID = -1


signal_counter: int = 0


def send_signal(signal: ControlSignal, device: torch.device) -> None:
    global signal_counter

    encoded_signal = torch.tensor(
        [signal_counter, signal.value], dtype=torch.int64, device=device
    )
    if dist.is_initialized():
        dist.broadcast(encoded_signal, src=0)

    if VERBOSE:
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.debug(
            f"[Rank {rank}] broadcast signal (#{signal_counter}) -> step={int(encoded_signal[1].item())}"
        )

    signal_counter += 1


def recv_signal(device: torch.device) -> ControlSignal:
    global signal_counter

    packet = torch.tensor([signal_counter, 0], dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.broadcast(packet, src=0)
    else:
        raise RuntimeError(
            "Single-GPU mode, receiving signal on non-master rank should not happen"
        )

    assert int(packet[0].item()) == signal_counter, (
        f"Error: signal counter mismatch: {int(packet[0].item())} != {signal_counter}"
    )

    if VERBOSE:
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.debug(
            f"[Rank {rank}] received signal (#{signal_counter}) -> step={int(packet[1].item())}"
        )

    signal_counter += 1

    return ControlSignal(int(packet[1].item()))


def sync_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if dist.is_initialized():
        kwargs_list = [kwargs]
        dist.broadcast_object_list(kwargs_list, src=0)
        return kwargs_list[0]
    else:
        return kwargs


# decorator to capture exceptions and print stack trace
def capture_exceptions(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")  # ty:ignore[unresolved-attribute]

            # print stack trace
            traceback.print_exc()
            raise

    return wrapper


def build_alpadreams_conditioning_wrapper(
    rank: int,
    n_cameras: int,
    local_attn_size: int,
    sink_size: int,
    device: torch.device,
    cp_size: int = 1,
    seed_for_every_rollout: int | None = None,
    resolution: str = "704p",
    encode_with_pixel_shuffle: bool = False,
    denoising_step_list: list[int] | None = None,
    num_frames_per_block: int = 12,
    compile_net: bool = True,
    use_cuda_graphs: bool = True,
    s3_credential_path: str = "credentials/s3_checkpoint.secret",
    upsampler: str = "none",
    kv_cache_on_side_stream: bool = False,
    no_tae: bool = False,
) -> AlpadreamsConditioningWrapper:
    logger.info(
        f"[Rank {rank}] Initializing WorldModelService with {n_cameras} cameras on device {device}"
    )

    if denoising_step_list is None:
        denoising_step_list = [1000, 500]

    api = AlpadreamsConditioningWrapper(
        n_cameras=n_cameras,
        resolution_wh=RESOLUTION_MAP[resolution],
        local_attn_size=local_attn_size,
        sink_size=sink_size,
        cp_size=cp_size,
        denoising_step_list=denoising_step_list,
        num_frames_per_block=num_frames_per_block,
        compile_net=compile_net,
        seed_for_every_rollout=seed_for_every_rollout,
        encode_with_pixel_shuffle=encode_with_pixel_shuffle,
        no_tae=no_tae,
        upsampler=upsampler,
        use_cuda_graphs=use_cuda_graphs,
        kv_cache_on_side_stream=kv_cache_on_side_stream,
        s3_credential_path=s3_credential_path,
        device=device,
    )
    if cp_size > 1:
        logger.info(
            "Context-parallel server orchestration enabled; view split/gather is handled inside flashdreams pipeline."
        )

    return api


class WorldModelEngine:
    def __init__(
        self,
        device: torch.device | str = torch.device("cuda:0"),
        # image encoding related
        output_format: str = "png",
        jpeg_quality: int = 90,
        # model related
        n_cameras: int = 1,
        local_attn_size: int | None = None,
        sink_size: int | None = None,
        context_parallel_size: int = 1,
        seed_for_every_rollout: int | None = None,
        resolution: str = "704p",
        encode_with_pixel_shuffle: bool = False,
        denoising_step_list: list[int] | None = None,
        num_frames_per_block: int | None = None,
        compile_net: bool = True,
        use_cuda_graphs: bool = True,
        s3_credential_path: str = "credentials/s3_checkpoint.secret",
        upsampler: str = "none",
        kv_cache_on_side_stream: bool = False,
        no_tae: bool = False,
    ):
        # determine rank
        self.MASTER_RANK = 0
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

        # Set default values if not provided
        local_attn_size = 8 if local_attn_size is None else local_attn_size
        sink_size = 0 if sink_size is None else sink_size
        num_frames_per_block = resolve_num_frames_per_block(
            n_cameras=n_cameras,
            encode_with_pixel_shuffle=encode_with_pixel_shuffle,
            num_frames_per_block=num_frames_per_block,
        )
        len_t = num_frames_per_block // 4
        logger.info(
            f"KV cache window: local_attn_size={local_attn_size}, sink_size={sink_size}"
        )
        logger.info(
            f"Temporal chunk config: num_frames_per_block={num_frames_per_block}, len_t={len_t}"
        )

        # Save configurations
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"CUDA device is required, got {self.device}")
        if self.device.index is None:
            self.device = torch.device("cuda:0")
        self.n_cameras = n_cameras
        self.seed_for_every_rollout_default = seed_for_every_rollout

        # Load the video generation model (SV when n_cameras=1, MV when n_cameras>1)
        self.api = build_alpadreams_conditioning_wrapper(
            rank=self.rank,
            n_cameras=n_cameras,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            device=self.device,
            cp_size=context_parallel_size,
            seed_for_every_rollout=seed_for_every_rollout,
            resolution=resolution,
            encode_with_pixel_shuffle=encode_with_pixel_shuffle,
            denoising_step_list=denoising_step_list,
            num_frames_per_block=num_frames_per_block,
            compile_net=compile_net,
            use_cuda_graphs=use_cuda_graphs,
            s3_credential_path=s3_credential_path,
            upsampler=upsampler,
            kv_cache_on_side_stream=kv_cache_on_side_stream,
            no_tae=no_tae,
        )
        logger.info("WorldModelEngine initialized successfully")

        # Store encoding configuration
        self.output_format = output_format.lower()
        self.jpeg_quality = jpeg_quality
        logger.info(
            f"Output format: {output_format}"
            + (f" (quality={jpeg_quality})" if output_format == "jpeg" else "")
        )

        # Session storage: session_id -> SessionState
        self.sessions: dict[str, SessionState] = {}

    @property
    def is_master(self) -> bool:
        return self.rank == self.MASTER_RANK

    def _set_rollout_seed_for_next_generation(self, seed: int | None) -> None:
        """Set seed used by the underlying model for next start_generation call."""
        self.api.set_rollout_seed(seed)

    def _cleanup_session(self, session_id: str) -> None:
        """
        Clean up a session and its associated resources.

        Closes the recorder if one exists and removes the session from storage.
        This is idempotent - safe to call multiple times.

        Args:
            session_id: The session ID to clean up.
        """

        # Remove session from storage if it exists
        session = self.sessions.get(session_id)
        if session is not None:
            del self.sessions[session_id]
            logger.info(f"Removed session {session_id} from storage")

    def wait_for_termination(self) -> None:
        """
        Worker loop for non-master ranks.

        Waits for control signals from master rank and executes corresponding operations.
        Supported signals:
        - START: Initialize a new session
        - VIDEO_CHUNK: Generate video frames
        - FINALIZE_KV: Update KV cache (deferred from VIDEO_CHUNK for latency hiding)
        - CLOSE: Clean up session resources
        - EXIT: Terminate the worker loop
        """
        control = ControlSignal.INVALID

        while True:
            control = recv_signal(self.device)
            match control:
                case ControlSignal.START:
                    self.open_session_on_all_ranks()
                case ControlSignal.VIDEO_CHUNK:
                    self.render_video_chunk_all_ranks()
                case ControlSignal.FINALIZE_KV:
                    self.finalize_kv_cache_all_ranks()
                case ControlSignal.CLOSE:
                    self.close_session_all_ranks()
                case ControlSignal.EXIT:
                    break
                case _:
                    raise ValueError(
                        f"[Rank {self.rank}] received invalid control signal: {control}"
                    )

    def open_session_on_all_ranks(self, **kwargs) -> None:
        """
        Start a new generation session.
        """

        # --- Start the session ---------------------------------------------
        if self.is_master:
            send_signal(ControlSignal.START, self.device)
        else:
            assert not kwargs, (
                f"[rank {self.rank}] kwargs is not empty on non-master rank"
            )
        # Distribute data to let all ranks
        kwargs = sync_kwargs(kwargs)
        assert kwargs, f"[rank {self.rank}] kwargs is empty"

        session_id = kwargs.get("session_id")
        initial_frames_list = kwargs.get("initial_frames_list")
        camera_names = kwargs.get("camera_names")
        camera_models_from_client = kwargs.get("camera_models_from_client")
        rig_to_camera_transforms = kwargs.get("rig_to_camera_transforms")
        text_prompts = kwargs.get("text_prompts")
        hdmap_parquets = kwargs.get("hdmap_parquets")
        skip_video_generation = kwargs.get("skip_video_generation")
        return_hdmap_frames = kwargs.get("return_hdmap_frames")
        effective_seed = kwargs.get("effective_seed")
        # -------------------------------------------------------------------
        assert len(text_prompts) == 1, (  # ty:ignore[invalid-argument-type]
            f"[rank {self.rank}] Expected 1 text prompt, got {len(text_prompts)}"  # ty:ignore[invalid-argument-type]
        )
        assert self.n_cameras == len(camera_names), (  # ty:ignore[invalid-argument-type]
            f"[rank {self.rank}] Expected {self.n_cameras} camera names, got {len(camera_names)}"  # ty:ignore[invalid-argument-type]
        )
        assert self.n_cameras == len(camera_models_from_client), (  # ty:ignore[invalid-argument-type]
            f"[rank {self.rank}] Expected {self.n_cameras} camera models, got {len(camera_models_from_client)}"  # ty:ignore[invalid-argument-type]
        )
        assert self.n_cameras == len(rig_to_camera_transforms), (  # ty:ignore[invalid-argument-type]
            f"[rank {self.rank}] Expected {self.n_cameras} rig_to_camera transforms, got {len(rig_to_camera_transforms)}"  # ty:ignore[invalid-argument-type]
        )
        assert len(hdmap_parquets) > 0, (  # ty:ignore[invalid-argument-type]
            f"[rank {self.rank}] Expected non-empty HD map parquets, got {len(hdmap_parquets)}"  # ty:ignore[invalid-argument-type]
        )

        def short_text(text: str) -> str:
            return text[:20] + "..." if len(text) > 20 else text

        if VERBOSE:
            print(
                f"[Rank {self.rank}] Open session {session_id}\n"
                f"      ├────> # of initial frames: {len(initial_frames_list)}\n"
                f"      ├────> # of cameras: {len(camera_models_from_client)}\n"
                f"      ├────> Prompts: POSITIVE='{short_text(text_prompts[0].positive)}'\n"
                f"      │               NEGATIVE='{short_text(text_prompts[0].negative)}'\n"
                f"      ├────> HD map parquets: {len(hdmap_parquets)} bytes\n"
                f"      ├────> Flag: skip_video_generation: {skip_video_generation}\n"
                f"      ├────> Flag: return_hdmap_frames: {return_hdmap_frames}\n"
                f"      ╰────> Effective random seed: {effective_seed}\n"
            )

        # Clean up all existing sessions (we only keep one session at a time)
        # This handles cases where a previous session wasn't properly closed
        if self.sessions:
            logger.info(
                f"[Rank {self.rank}] Cleaning up {len(self.sessions)} existing session(s) before starting new one"
            )
            # Create a list of session IDs to avoid modifying dict during iteration
            existing_session_ids = list(self.sessions.keys())
            for old_session_id in existing_session_ids:
                self._cleanup_session(old_session_id)

        # Retrieve profiler and start time
        profiler = get_profiler()
        chunk_idx = profiler.get_chunk_idx(session_id)  # ty:ignore[invalid-argument-type]

        # 1. Convert to RGB tensor
        res_W, res_H = self.api.video_resolution_wh
        decoded_frames: list[torch.Tensor] = []
        with profiler.measure(
            "decode_initial_frames",
            session_id=session_id,  # ty: ignore[invalid-argument-type]
            chunk_idx=chunk_idx,
        ):
            for i, img_msg in enumerate(initial_frames_list):  # ty:ignore[invalid-argument-type]
                frame_tensor = decode_image(
                    img_msg.data,  # ty:ignore[unresolved-attribute]
                    video_model_pb2.ImageFormat.Name(img_msg.format),  # ty:ignore[unresolved-attribute]
                    target_resolution_hw=(res_H, res_W),
                )  # [3, H, W]
                decoded_frames.append(frame_tensor)
        # Stack → [V, 3, H, W], then add batch dim → [1, V, 3, H, W]
        initial_rgb_frames = torch.stack(decoded_frames, dim=0).unsqueeze(
            0
        )  # [B, V, 3, H, W]

        # 2. Load static world map from zip bytes
        with profiler.measure(
            "load_static_world_map",
            session_id=session_id,  # ty: ignore[invalid-argument-type]
            chunk_idx=chunk_idx,
        ):
            scene_data = load_static_world_from_zip_bytes(
                hdmap_parquets,  # ty:ignore[invalid-argument-type]
                camera_names=camera_names,  # ty:ignore[invalid-argument-type]
                target_resolution_hw=(res_H, res_W),
            )
        logger.info(
            f"[Rank {self.rank}] Loaded scene: {scene_data.scene_id}, num_frames: {scene_data.num_frames}"
        )

        # Store client-provided intrinsics on scene_data so create_renderer can find them
        for cam_name, cam_model in camera_models_from_client.items():  # ty:ignore[unresolved-attribute]
            scene_data.camera_models[cam_name] = cam_model

        # 3. Create multi-camera renderer
        renderer = self.api.create_renderer(scene_data, camera_names)  # ty:ignore[invalid-argument-type]

        # 4. Store session state (generation deferred to first render_video_chunk)
        for cam_name, rig_to_cam in rig_to_camera_transforms.items():  # ty:ignore[unresolved-attribute]
            rig_to_camera_transforms[cam_name] = torch.tensor(  # ty:ignore[invalid-assignment]
                rig_to_cam, device=self.device
            )
        session_state = SessionState(
            session_id=session_id,  # ty:ignore[invalid-argument-type]
            camera_names=camera_names,  # ty:ignore[invalid-argument-type]
            rig_to_camera_transforms=rig_to_camera_transforms,  # ty:ignore[invalid-argument-type]
            scene_data=scene_data,
            renderer=renderer,
            initial_rgb_frames=initial_rgb_frames,
            text_prompts=text_prompts,  # ty:ignore[invalid-argument-type]
            skip_video_generation=skip_video_generation,  # ty:ignore[invalid-argument-type]
            return_hdmap_frames=return_hdmap_frames,  # ty:ignore[invalid-argument-type]
            effective_seed=effective_seed,
        )
        self.sessions[session_id] = session_state  # ty:ignore[invalid-assignment]
        logger.info(
            f"[Rank {self.rank}] Session initialized, generation deferred to first render_video_chunk call"
        )

    def render_video_chunk_all_ranks(
        self, **kwargs
    ) -> video_model_pb2.VideoChunkReturn:
        """
        Render a video chunk.
        """
        # --- Start the rendering -----------------------------------------
        if self.is_master:
            send_signal(ControlSignal.VIDEO_CHUNK, self.device)
        else:
            assert not kwargs, (
                f"[rank {self.rank}] kwargs is not empty on non-master rank"
            )
        # Distribute data to let all ranks
        kwargs = sync_kwargs(kwargs)
        assert kwargs, f"[rank {self.rank}] kwargs is empty"
        session_id = kwargs.get("session_id")
        frame_timestamps_us = kwargs.get("frame_timestamps_us")
        object_info_per_frame = kwargs.get("object_info_per_frame")
        rig_poses_flu = torch.tensor(kwargs.get("rig_poses_flu"), device=self.device)
        # -------------------------------------------------------------------
        if VERBOSE:
            print(
                f"[Rank {self.rank}] Render video chunk for session {session_id}\n"
                f"      ├────> # of frames: {len(frame_timestamps_us)}\n"
                f"      ├────> # of dynamic objects: {len(object_info_per_frame)}\n"
                f"      ╰────> # of rig poses: {len(rig_poses_flu)}, type: {type(rig_poses_flu)}"
            )

        # Retrieve profiler and start time
        profiler = get_profiler()
        chunk_idx = profiler.get_chunk_idx(session_id)  # ty:ignore[invalid-argument-type]

        # Get session state
        session_state = self.sessions[session_id]  # ty:ignore[invalid-argument-type]
        skip_video_generation = session_state.skip_video_generation

        # multiGPU: split views to all ranks.
        V_group = self.api.V_group
        if V_group is not None:
            camera_names = split_inputs_cp_object_list(
                session_state.camera_names, cp_group=V_group
            )
        else:
            camera_names = session_state.camera_names

        # 3. Parse rig trajectory (FLU) and derive per-camera poses
        with profiler.measure(
            "parse_trajectory_continuation",
            session_id=session_id,  # ty: ignore[invalid-argument-type]
            chunk_idx=chunk_idx,
        ):
            # Compute per-camera trajectories:  camera_to_world[t] = rig_to_world[t] @ rig_to_camera
            camera_poses_per_view: dict[str, torch.Tensor] = {}
            for cam_name in camera_names:
                rig_to_cam = session_state.rig_to_camera_transforms[cam_name]
                camera_poses_per_view[cam_name] = compute_camera_poses_from_rig(
                    rig_poses_flu, rig_to_cam
                )  # ty:ignore[invalid-assignment]

        # Do a sanity check on the camera poses
        for cam_name, camera_pose in camera_poses_per_view.items():
            assert isinstance(camera_pose, torch.Tensor)
            assert camera_pose.device == self.device, (
                f"Camera pose for {cam_name} is on device {camera_pose.device}, expected {self.device}"
            )
            assert camera_pose.shape == (len(frame_timestamps_us), 4, 4), (  # ty:ignore[invalid-argument-type]
                f"Camera pose for {cam_name} has shape {camera_pose.shape}, expected ({len(frame_timestamps_us), 4, 4})"  # ty:ignore[invalid-argument-type]
            )
        assert len(camera_poses_per_view) == len(camera_names), (
            f"[Rank {self.rank}] Expected {len(camera_names)} camera poses, got {len(camera_poses_per_view)}"
        )

        # Generate frames
        is_first_chunk = not session_state.generation_started
        if is_first_chunk:
            logger.info(
                f"[Rank {self.rank}] Starting generation with {len(frame_timestamps_us)} frames (skip_video={skip_video_generation})..."  # ty:ignore[invalid-argument-type]
            )
            self._set_rollout_seed_for_next_generation(session_state.effective_seed)
            logger.info(
                f"[Rank {self.rank}] Using session random seed: {session_state.effective_seed} "
                "(None means no explicit per-rollout seed)"
            )
            with profiling_context(session_id, chunk_idx):  # ty:ignore[invalid-argument-type]
                with profiler.measure(
                    "start_generation_total",
                    session_id=session_id,  # ty:ignore[invalid-argument-type]
                    chunk_idx=chunk_idx,
                    num_frames=len(frame_timestamps_us),  # ty:ignore[invalid-argument-type]
                ):
                    output = self.api.start_generation(
                        text_prompts=session_state.text_prompts,
                        initial_rgb_frames=session_state.initial_rgb_frames,
                        renderer=session_state.renderer,
                        camera_names=camera_names,
                        camera_poses_per_view=camera_poses_per_view,
                        frame_timestamps_us=frame_timestamps_us,  # ty:ignore[invalid-argument-type]
                        skip_video_generation=skip_video_generation,
                    )
            session_state.bbox_state = output.state
            session_state.generation_started = True
        else:
            assert session_state.bbox_state is not None, (
                "bbox_state should be set after first chunk"
            )

            logger.info(
                f"[Rank {self.rank}] Continuing generation with {len(frame_timestamps_us)} frames (skip_video={skip_video_generation})..."  # ty:ignore[invalid-argument-type]
            )
            with profiling_context(session_id, chunk_idx):  # ty:ignore[invalid-argument-type]
                with profiler.measure(
                    "continue_generation_total",
                    session_id=session_id,  # ty:ignore[invalid-argument-type]
                    chunk_idx=chunk_idx,
                    num_frames=len(frame_timestamps_us),  # ty:ignore[invalid-argument-type]
                ):
                    output = self.api.continue_generation(
                        state=session_state.bbox_state,
                        camera_names=camera_names,
                        camera_poses_per_view=camera_poses_per_view,
                        frame_timestamps_us=frame_timestamps_us,  # ty:ignore[invalid-argument-type]
                        object_info_per_frame=object_info_per_frame,
                        skip_video_generation=skip_video_generation,
                    )
            session_state.bbox_state = output.state

        # # TODO: Somehow this is necessary to avoid creating black frames in the output video. Need to investigate why...
        # with profiler.measure("synchronize_cuda", session_id=session_id, chunk_idx=chunk_idx):
        #     torch.cuda.synchronize()

        # Build response data
        skip_video_generation = session_state.skip_video_generation

        # Encode per-camera outputs into CameraOutput messages
        # output.rgb_frames: [B, V, T, 3, H, W], output.condition_frames: [B, V, T, 3, H, W]
        camera_outputs = []
        with profiler.measure(
            "encode_output_frames",
            session_id=session_id,  # ty: ignore[invalid-argument-type]
            chunk_idx=chunk_idx,
        ):
            for v_idx, cam_name in enumerate(camera_names):
                cam_output = {
                    "rgb_frames": [],
                    "hdmap_condition_frames": [],
                }
                # Encode RGB frames for this camera
                if output.rgb_frames is not None:
                    # output.rgb_frames[0, v_idx] → [T, 3, H, W]
                    cam_rgb = output.rgb_frames[0, v_idx]  # [T, 3, H, W]
                    cam_output["rgb_frames"].extend(self._encode_images(cam_rgb))

                # Encode HDMap condition frames for this camera (if requested)
                if session_state.return_hdmap_frames or skip_video_generation:
                    cam_hdmap = output.condition_frames[0, v_idx]  # [T, 3, H, W]
                    logger.debug(
                        f"HDMap condition_frames[{v_idx}] '{cam_name}': "
                        f"shape={cam_hdmap.shape}, dtype={cam_hdmap.dtype}, "
                        f"min={cam_hdmap.min().item()}, max={cam_hdmap.max().item()}, "
                        f"mean={cam_hdmap.float().mean().item():.1f}"
                    )
                    cam_output["hdmap_condition_frames"].extend(
                        self._encode_images(cam_hdmap)
                    )

                camera_outputs.append(cam_output)

        # multiGPU: gather outputs from all ranks.
        logger.info(f"[Rank {self.rank}] Gathering camera outputs from all ranks")
        if V_group is not None:
            camera_outputs = cat_outputs_cp_object_list(camera_outputs, V_group)
        assert len(camera_outputs) == len(session_state.camera_names), (
            f"Expected {len(session_state.camera_names)} outputs, got {len(camera_outputs)}"
        )
        logger.info(
            f"[Rank {self.rank}] Outputs gathered from all ranks: {len(camera_outputs)}"
        )

        # Build response
        response = video_model_pb2.VideoChunkReturn()
        for cam_name, cam_output in zip(session_state.camera_names, camera_outputs):
            cam_output_pb = video_model_pb2.CameraOutput(camera_logical_id=cam_name)
            cam_output_pb.rgb_frames.extend(cam_output["rgb_frames"])
            cam_output_pb.hdmap_condition_frames.extend(
                cam_output["hdmap_condition_frames"]
            )
            response.camera_outputs.append(cam_output_pb)

        # Store finalization state for deferred execution (after gRPC response is sent)
        # This allows the response to be returned immediately while KV cache update
        # happens in parallel using the network idle time.
        session_state.pending_finalization_state = output.finalization_state

        return response

    def finalize_kv_cache_all_ranks(self, **kwargs) -> None:
        """
        Finalize KV cache update on all ranks.

        This is called after the gRPC response is sent to overlap KV cache update
        with network transfer time. All ranks must participate in this call.
        """
        # --- Start the finalization -----------------------------------------
        if self.is_master:
            send_signal(ControlSignal.FINALIZE_KV, self.device)
        else:
            assert not kwargs, (
                f"[rank {self.rank}] kwargs is not empty on non-master rank"
            )
        # Distribute data to let all ranks
        kwargs = sync_kwargs({"session_id": kwargs.get("session_id")})
        session_id = kwargs.get("session_id")
        # -------------------------------------------------------------------

        if session_id not in self.sessions:
            logger.warning(
                f"[Rank {self.rank}] Session {session_id} not found for finalization"
            )
            return

        session_state = self.sessions[session_id]
        finalization_state = session_state.pending_finalization_state

        if finalization_state is None:
            logger.debug(
                f"[Rank {self.rank}] No pending finalization for session {session_id}"
            )
            return

        # Clear pending state before executing
        session_state.pending_finalization_state = None

        # Execute KV cache update
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])
        tic = time.time_ns()
        if (
            session_state.bbox_state is None
            or session_state.bbox_state.pipeline_cache is None
        ):
            logger.warning(
                f"[Rank {self.rank}] Missing pipeline cache for finalization in session {session_id}"
            )
        else:
            self.api.finalize_block_generation(
                session_state.bbox_state.pipeline_cache,
                finalization_state,
            )
        duration_ns = time.time_ns() - tic
        logger.info(
            f"[Rank {self.rank}] Finalize block generation duration_ms={duration_ns / 1000000:.2f}"
        )

    def close_session_all_ranks(self, **kwargs) -> None:
        """
        Stop a session.
        """
        # --- Stop the session --------------------------------------------
        if self.is_master:
            send_signal(ControlSignal.CLOSE, self.device)
        else:
            assert not kwargs, (
                f"[rank {self.rank}] kwargs is not empty on non-master rank"
            )
        # Distribute data to let all ranks
        kwargs = sync_kwargs(kwargs)
        assert kwargs, f"[rank {self.rank}] kwargs is empty"
        # -------------------------------------------------------------------
        session_id = kwargs.get("session_id")

        # Clean up session resources (just session state)
        logger.info(f"[Rank {self.rank}] Closing session {session_id} on all ranks")
        self._cleanup_session(session_id)  # ty:ignore[invalid-argument-type]
        logger.info(
            f"[Rank {self.rank}] Session {session_id} closed and removed from storage"
        )

    def _encode_images(self, images: torch.Tensor) -> list[video_model_pb2.Image]:
        """
        Encode a batch of images using nvjpeg.

        Args:
            images: Tensor of shape [B, 3, H, W] uint8.

        Returns:
            List of bytes objects, one per image in the batch.
        """
        assert self.device.type == "cuda", "Images must be on GPU"
        if self.output_format == "jpeg":
            jpeg_list = nvjpeg.encode(
                images, quality=self.jpeg_quality, device_index=self.device.index
            )  # Encode to JPEG with quality using ludus_renderer
            return [
                video_model_pb2.Image(
                    data=jpeg, format=video_model_pb2.ImageFormat.JPEG
                )
                for jpeg in jpeg_list
            ]
        else:
            frame_msgs = []
            for frame in images:  # B, 3, H, W
                frame_np = frame.permute(1, 2, 0).cpu().numpy()
                frame_bytes, image_format = self._encode_single_image_cpu(frame_np)
                image_msg = video_model_pb2.Image(data=frame_bytes, format=image_format)
                frame_msgs.append(image_msg)
            return frame_msgs

    def _encode_single_image_cpu(
        self, image_np: np.ndarray
    ) -> tuple[bytes, video_model_pb2.ImageFormat]:
        """
        Encode a numpy image using the configured output format.

        Args:
            image_np: Numpy array [H, W, 3] uint8.

        Returns:
            Tuple of (encoded bytes, ImageFormat enum value).
        """
        logger.critical("Using CPU-based image encoding (this is slow)")
        if self.output_format == "jpeg":
            return (
                encode_image(image_np, format="JPEG", quality=self.jpeg_quality),
                video_model_pb2.ImageFormat.JPEG,
            )
        else:
            return (
                encode_image(image_np, format="PNG"),
                video_model_pb2.ImageFormat.PNG,
            )


class WorldModelService(
    WorldModelEngine, video_model_pb2_grpc.WorldModelServiceServicer
):
    """gRPC service for bbox-conditioned video generation."""

    def __init__(
        self,
        device: torch.device | str = torch.device("cuda:0"),
        output_format: str = "png",
        jpeg_quality: int = 90,
        recording_dir: Path | str | None = None,
        n_cameras: int = 1,
        local_attn_size: int | None = None,
        sink_size: int | None = None,
        context_parallel_size: int = 1,
        seed_for_every_rollout: int | None = None,
        resolution: str = "704p",
        encode_with_pixel_shuffle: bool = False,
        denoising_step_list: list[int] | None = None,
        num_frames_per_block: int | None = None,
        compile_net: bool = True,
        use_cuda_graphs: bool = True,
        s3_credential_path: str = "credentials/s3_checkpoint.secret",
        upsampler: str = "none",
        kv_cache_on_side_stream: bool = False,
        no_tae: bool = False,
    ):
        """
        Initialize the World Model gRPC service.

        Args:
            device: CUDA device used for inference.
            output_format: Output image format ("png" or "jpeg").
            jpeg_quality: JPEG quality (1-100) if using JPEG format.
            recording_dir: Directory to save session recordings (None to disable).
                           Each session will create a file named {session_id}.binlog in this directory.
            n_cameras: Number of camera views (1 = single-view, >1 = multi-view).
            local_attn_size: Local attention window size in latent frames.
            sink_size: Sink size in latent frames.
        """
        super().__init__(
            device=device,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            n_cameras=n_cameras,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            context_parallel_size=context_parallel_size,
            seed_for_every_rollout=seed_for_every_rollout,
            resolution=resolution,
            encode_with_pixel_shuffle=encode_with_pixel_shuffle,
            denoising_step_list=denoising_step_list,
            num_frames_per_block=num_frames_per_block,
            compile_net=compile_net,
            use_cuda_graphs=use_cuda_graphs,
            s3_credential_path=s3_credential_path,
            upsampler=upsampler,
            kv_cache_on_side_stream=kv_cache_on_side_stream,
            no_tae=no_tae,
        )

        # Session recording - per-session recorders will be created in start_session
        self.recording_dir: Path | None = None
        if recording_dir is not None:
            self.recording_dir = Path(recording_dir)
            self.recording_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Session recording enabled: recordings will be saved to {self.recording_dir}"
            )
        else:
            logger.info("Session recording disabled")
        self.recorders: dict[str, SessionRecorder] = {}  # session_id -> SessionRecorder

        # Finalization synchronization: ensures KV cache update completes before next request
        # The event is set when no finalization is pending, cleared when finalization starts
        self._finalization_done = threading.Event()
        self._finalization_done.set()  # Initially done (no pending finalization)

        logger.info("WorldModelService initialized successfully")

    def _cleanup_session(self, session_id: str) -> None:
        """
        Clean up a session and its associated resources.

        Closes the recorder if one exists and removes the session from storage.
        This is idempotent - safe to call multiple times.

        Args:
            session_id: The session ID to clean up.
        """
        # Close and remove recorder if it exists
        if session_id in self.recorders:
            try:
                self.recorders[session_id].close()
                logger.info(f"Closed recorder for session {session_id}")
            except Exception as e:
                logger.warning(f"Error closing recorder for session {session_id}: {e}")
            del self.recorders[session_id]
            logger.info(f"Session recorder session {session_id} closed")

        # Remove session from storage if it exists
        super()._cleanup_session(session_id)

    @capture_exceptions
    def start_session(
        self,
        request: video_model_pb2.SessionRequest,
        context: grpc.ServicerContext,
    ) -> video_model_pb2.SessionId:
        """
        Start a new generation session.

        Initializes the renderer with static world map and stores per-camera
        information for subsequent ``render_video_chunk`` calls.

        Args:
            request: SessionRequest with ``camera_specs``, ``initial_frames``,
                     ``static_world_map``, etc.
            context: gRPC context.

        Returns:
            SessionId for subsequent requests.
        """
        start_time_ns = time.time_ns()
        logger.info("Server received start_session request")

        # Clean up all existing sessions (we only keep one session at a time)
        # This handles cases where a previous session wasn't properly closed
        if self.sessions:
            logger.info(
                f"Cleaning up {len(self.sessions)} existing session(s) before starting new one"
            )
            for old_session_id in self.recorders.keys():
                if old_session_id not in self.sessions:
                    self._cleanup_session(old_session_id)

        # Generate unique session ID
        session_id = str(uuid.uuid4())
        logger.info(f"Created session: {session_id}")

        # Get profiler and chunk index
        profiler = get_profiler()
        _ = profiler.get_chunk_idx(session_id)

        # 1. Parse camera specs list — extract names, intrinsics, rig_to_camera
        camera_specs_raw = list(request.camera_specs)
        if not camera_specs_raw:
            raise ValueError(
                "SessionRequest.camera_specs must contain at least one CameraSpec"
            )

        camera_names: list[str] = []
        camera_models_from_client: dict[str, FThetaCamera] = {}
        rig_to_camera_transforms: dict[str, np.ndarray] = {}

        for i, spec in enumerate(camera_specs_raw):
            spec_dict = proto_to_dict(spec)
            cam_name = spec_dict.get("logical_id", f"camera_{i}")
            camera_names.append(cam_name)

            # Parse intrinsics (if provided by client)
            has_camera_model = (
                spec.HasField("ftheta_param")
                or spec.HasField("opencv_pinhole_param")
                or spec.HasField("opencv_fisheye_param")
            )
            if has_camera_model:
                camera_models_from_client[cam_name] = camera_spec_to_ftheta(spec_dict)

            # Parse rig_to_camera (FLU convention from client → convert to RDF)
            rig_to_cam_flu = parse_rig_to_camera(spec_dict)
            rig_to_camera_transforms[cam_name] = rig_to_cam_flu

        logger.info(f"Parsed {len(camera_names)} camera specs: {camera_names}")
        assert len(camera_names) == len(camera_models_from_client), (
            "Expected one camera model per camera name"
        )

        # 2a. Decode initial frames — one per camera
        initial_frames_list = list(request.initial_frames)
        if len(initial_frames_list) != len(camera_names):
            raise ValueError(
                f"Expected {len(camera_names)} initial_frames (one per camera_spec), got {len(initial_frames_list)}"
            )

        # 2b. Create text prompts — client-provided or AV defaults
        if request.HasField("text_prompt") and request.text_prompt.positive != "":
            text_prompts = [
                TextPrompt(
                    positive=request.text_prompt.positive,
                    negative=request.text_prompt.negative or "",
                )
            ]
            logger.info(f"Using client prompt: {request.text_prompt.positive[:50]}...")
        else:
            text_prompts = [TextPrompt(positive=AV_POSITIVE_PROMPT)]
            logger.info("Using default AV prompt")

        # 3. Parse debug options
        skip_video_generation = False
        return_hdmap_frames = False
        if request.HasField("debug_options"):
            skip_video_generation = request.debug_options.skip_video_generation
            return_hdmap_frames = request.debug_options.return_hdmap_frames
            if skip_video_generation:
                logger.info("HDMap-only mode enabled (skip_video_generation=True)")
            if return_hdmap_frames:
                logger.info("HDMap frames will be returned with video")
            if request.debug_options.return_bev_map:
                logger.warning(
                    "debug_options.return_bev_map is ignored; BEV rendering path was removed."
                )

        request_seed = int(request.random_seed)
        if request_seed != 0:
            effective_seed = request_seed
            seed_source = "request.random_seed"
        elif self.seed_for_every_rollout_default is None:
            effective_seed = None
            seed_source = "none"
        else:
            effective_seed = self.seed_for_every_rollout_default
            seed_source = "seed_for_every_rollout"
        logger.info(
            f"Using effective session seed={effective_seed} (source={seed_source})"
        )

        # 4. Start engine session
        self.open_session_on_all_ranks(
            session_id=session_id,
            camera_names=camera_names,
            camera_models_from_client=camera_models_from_client,
            rig_to_camera_transforms=rig_to_camera_transforms,
            initial_frames_list=initial_frames_list,
            text_prompts=text_prompts,
            hdmap_parquets=request.static_world_map.hdmap_parquets,
            skip_video_generation=skip_video_generation,
            return_hdmap_frames=return_hdmap_frames,
            effective_seed=effective_seed,
        )
        logger.info(f"Session {session_id} started successfully")

        # 5. Create response
        response = video_model_pb2.SessionId(session_id=session_id)

        # 6. Create per-session recorder if recording is enabled
        if self.recording_dir is not None:
            recording_path = self.recording_dir / f"{session_id}.binlog"
            recorder = SessionRecorder(recording_path)
            self.recorders[session_id] = recorder
            logger.info(f"Created recorder for session {session_id}: {recording_path}")

        # 7. Record request/response if recording is enabled
        if session_id in self.recorders:
            duration_ns = time.time_ns() - start_time_ns
            self.recorders[session_id].record_start_session(
                request, response, start_time_ns, duration_ns
            )

        return response

    @capture_exceptions
    def render_video_chunk(
        self,
        request: video_model_pb2.VideoChunkRequest,
        context: grpc.ServicerContext,
    ) -> video_model_pb2.VideoChunkReturn:
        """
        Generate a video chunk for an existing session.

        For the first call after start_session, this generates the initial chunk.
        For subsequent calls, this generates continuation chunks.

        The request provides a single ``rig_trajectory`` (ego-rig poses).
        Per-camera trajectories are computed server-side using the
        ``rig_to_camera`` transforms stored during ``start_session``.

        Args:
            request: VideoChunkRequest with ``session_id``, ``rig_trajectory``,
                     ``dynamic_state``.
            context: gRPC context.

        Returns:
            VideoChunkReturn with per-camera output frames and rig trajectory.
        """
        start_time_ns = time.time_ns()

        # Wait for previous finalization to complete before processing new request
        # This ensures KV cache is ready for the new generation
        self._finalization_done.wait()

        # 1. Get session ID
        session_id = request.session_id.session_id
        logger.info(
            f"Server received render_video_chunk request for session: {session_id}"
        )
        if session_id not in self.sessions:
            raise KeyError(f"Session not found: {session_id}")

        # Get profiler and chunk index
        profiler = get_profiler()
        chunk_idx = profiler.get_chunk_idx(session_id)

        # 2. Get session state
        session_state = self.sessions[session_id]

        # Determine chunk size
        is_first_chunk = not session_state.generation_started
        if is_first_chunk:
            chunk_size = self.api.initial_frame_chunk_size
            logger.info(
                f"First chunk: expecting {chunk_size} poses (initial generation)"
            )
        else:
            chunk_size = self.api.frame_chunk_size
            logger.info(f"Continuation chunk: expecting {chunk_size} poses")

        # 3. Parse rig trajectory (FLU → RDF) and derive per-camera poses
        # FIXME: do this on GPU ... should not do this actually ...
        with profiler.measure(
            "parse_trajectory", session_id=session_id, chunk_idx=chunk_idx
        ):
            trajectory_dict = proto_to_dict(request.rig_trajectory)
            client_poses = trajectory_dict.get("poses", [])

            if len(client_poses) < chunk_size:
                raise ValueError(
                    f"Client must provide rig poses. Expected {chunk_size} poses "
                    f"({'initial' if is_first_chunk else 'continuation'} chunk), "
                    f"got {len(client_poses)}."
                )

            logger.info(
                f"Using client-provided rig trajectory ({len(client_poses)} poses)"
            )
            rig_poses_flu, trajectory_timestamps_us = trajectory_to_camera_poses(
                client_poses
            )

            if len(rig_poses_flu) != chunk_size:
                raise ValueError(
                    f"Expected exactly {chunk_size} poses "
                    f"({'initial' if is_first_chunk else 'continuation'} chunk), "
                    f"got {len(rig_poses_flu)}."
                )

        # 4. Extract frame timestamps from the trajectory
        frame_timestamps_us = trajectory_timestamps_us[:chunk_size]
        num_frames = len(frame_timestamps_us)
        logger.info(
            f"Frame timestamps (us): [{frame_timestamps_us[0]}..{frame_timestamps_us[-1]}] "
            f"({len(frame_timestamps_us)} frames)"
        )

        # 5. Parse dynamic state
        with profiler.measure(
            "parse_dynamic_state", session_id=session_id, chunk_idx=chunk_idx
        ):
            dynamic_state_dict = proto_to_dict(request.dynamic_state)
            logger.debug(f"dynamic_state_dict: {dynamic_state_dict}")
            object_info_per_frame = [
                dynamic_state_to_object_info(dynamic_state_dict, ts)
                for ts in frame_timestamps_us
            ]
        logger.info(f"Parsed {len(object_info_per_frame)} frames with dynamic objects")

        # 6. Generate frames
        response = self.render_video_chunk_all_ranks(
            session_id=session_id,
            frame_timestamps_us=frame_timestamps_us,
            object_info_per_frame=object_info_per_frame,
            rig_poses_flu=rig_poses_flu,
        )

        # Echo back rig trajectory
        response.poses_and_timestamps_of_frames.CopyFrom(request.rig_trajectory)

        # Schedule KV cache finalization in background thread
        # This allows the gRPC response to be sent immediately while KV cache update
        # happens in parallel, utilizing the network transfer time.
        if session_state.pending_finalization_state is not None:
            self._finalization_done.clear()  # Mark finalization as pending

            def do_finalization():
                try:
                    self.finalize_kv_cache_all_ranks(session_id=session_id)
                finally:
                    self._finalization_done.set()  # Mark finalization as complete

            finalization_thread = threading.Thread(target=do_finalization, daemon=True)
            finalization_thread.start()

        # Increment chunk counter
        profiler.increment_chunk_idx(session_id)

        duration_ns = time.time_ns() - start_time_ns
        logger.info(
            f"Successfully generated {num_frames} frames for session {session_id}, duration_ms={duration_ns / 1000000:.2f}"
        )

        # Record request/response if recording is enabled
        if session_id in self.recorders:
            self.recorders[session_id].record_render_video_chunk(
                request, response, start_time_ns, duration_ns
            )

        return response

    @capture_exceptions
    def close_session(
        self,
        request: video_model_pb2.SessionCloseRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        """
        Close an existing generation session.

        Removes the session from internal storage and cleans up associated resources.

        Args:
            request: SessionCloseRequest with session_id to close.
            context: gRPC context.

        Returns:
            Empty response indicating successful closure.
        """
        start_time_ns = time.time_ns()

        # Wait for any pending finalization to complete before closing
        self._finalization_done.wait()

        # Get session ID
        session_id = request.session_id
        logger.info(f"Server received close_session request for session: {session_id}")

        # Check if session exists
        if session_id not in self.sessions:
            logger.warning(f"Session not found for close: {session_id}")
            # Return empty response even if session doesn't exist (idempotent)
            return common_pb2.Empty()

        # Create response
        response = common_pb2.Empty()

        # Record request/response if recording is enabled
        if session_id in self.recorders:
            duration_ns = time.time_ns() - start_time_ns
            self.recorders[session_id].record_close_session(
                request, response, start_time_ns, duration_ns
            )

        # Clean up session resources (recorder and session state)
        self.close_session_all_ranks(session_id=session_id)

        return response


class SessionState:
    """State maintained for each active session.

    Stores per-session information including multi-camera configuration
    (names, rig-to-camera extrinsics) and generation state.
    """

    def __init__(
        self,
        session_id: str,
        camera_names: list[str],
        rig_to_camera_transforms: dict[str, torch.Tensor],
        scene_data: SceneData,
        renderer: LudusRenderer,
        initial_rgb_frames: torch.Tensor,
        text_prompts: list,
        skip_video_generation: bool = False,
        return_hdmap_frames: bool = False,
        effective_seed: int | None = None,
    ):
        self.session_id = session_id
        self.camera_names = camera_names  # Ordered list of camera logical IDs
        self.rig_to_camera_transforms = (
            rig_to_camera_transforms  # cam_name → 4×4 rig_to_camera (RDF or FLU)
        )
        self.scene_data = scene_data
        self.renderer = renderer  # Created in start_session
        self.initial_rgb_frames = (
            initial_rgb_frames  # [1, V, 3, H, W] for start_generation
        )
        self.text_prompts = text_prompts

        # Debug options (from session request)
        self.skip_video_generation = skip_video_generation
        self.return_hdmap_frames = return_hdmap_frames
        self.effective_seed = effective_seed

        # Generation state (populated after first render_video_chunk)
        self.bbox_state: AlpadreamsConditioningState | None = None
        self.generation_started: bool = False

        # Pending KV cache finalization state (for async finalization after response)
        self.pending_finalization_state: dict | None = None

        for cam_name, rig_to_cam in rig_to_camera_transforms.items():
            assert isinstance(rig_to_cam, torch.Tensor), (
                f"Rig-to-camera transform for {cam_name} must be a torch.Tensor, got {type(rig_to_cam)}"
            )
            assert rig_to_cam.shape == (4, 4), (
                f"Rig-to-camera transform for {cam_name} must be a 4x4 matrix, got {rig_to_cam.shape}"
            )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "gRPC server for bbox-conditioned video generation. "
            "CUDA is required and context parallel size is derived from world size."
        )
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to bind the server to (default: 50051)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of worker threads (default: 10)",
    )
    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        help="Enable profiling and timing measurements",
    )
    parser.add_argument(
        "--profile_output",
        type=str,
        default="server_profile.json",
        help="Output file for profiling data (JSON)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["png", "jpeg"],
        default="png",
        help="Output image format for generated frames (default: png). JPEG is faster but lossy.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=90,
        help="JPEG quality (1-100) when using --output_format=jpeg (default: 90)",
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save session recordings. Each session will create a {session_id}.binlog file in this directory.",
    )
    parser.add_argument(
        "--n_cameras",
        type=int,
        default=1,
        help="Number of camera views. 1 = single-view, >1 = multi-view. Default: 1.",
    )
    parser.add_argument(
        "--local_attn_size",
        type=int,
        default=None,
        help=(
            "Local attention size in latent frames. Default: 21 for multi-view, 7 for single-view."
        ),
    )
    parser.add_argument(
        "--sink_size",
        type=int,
        default=None,
        help=(
            "Sink size in latent frames. Default: 3 for multi-view, 0 for single-view."
        ),
    )
    parser.add_argument(
        "--seed_for_every_rollout",
        type=int,
        default=None,
        help="Seed for every rollout. If None, only seed at the beginning of the server.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=["480p", "720p", "704p"],
        default="704p",
        help="Resolution of the video (default: 704p).",
    )
    parser.add_argument(
        "--encode_with_pixel_shuffle",
        action="store_true",
        help="Encode HDMap with pixel shuffle instead of VAE encoding.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=str,
        default="1000,500",
        help="Comma-separated list of denoising timesteps (default: '1000,500').",
    )
    parser.add_argument(
        "--num_frames_per_block",
        type=int,
        default=None,
        help=(
            "Number of pixel frames per block. Defaults to 16 for multi-view or "
            "pixel-shuffle mode, and 12 for single-view VAE-encoding."
        ),
    )
    parser.add_argument(
        "--no_compile_net",
        action="store_true",
        help="Disable torch.compile for the network.",
    )
    parser.add_argument(
        "--no_cuda_graphs",
        action="store_true",
        help="Disable CUDA graphs for DiT blocks.",
    )
    parser.add_argument(
        "--s3_credential_path",
        type=str,
        default="credentials/s3_checkpoint.secret",
        help="Path to S3 credential file for checkpoint download.",
    )
    parser.add_argument(
        "--upsampler",
        type=str,
        choices=["none", "realesrgan", "flashvsr"],
        default="none",
        help="Upsampler to use (default: none).",
    )
    parser.add_argument(
        "--kv_cache_on_side_stream",
        action="store_true",
        help="Use side stream for KV cache update.",
    )
    parser.add_argument(
        "--no_tae",
        action="store_true",
        help="Disable TAE decoder (use WanVAE for decoding instead of WanVAE_tiny). Required for parallel VAE decoding.",
    )
    return parser.parse_args()


def initialize_distributed(n_cameras: int) -> tuple[torch.device, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for inference in the Alpadreams gRPC server."
        )

    has_rank = "RANK" in os.environ
    has_world_size = "WORLD_SIZE" in os.environ
    if has_rank != has_world_size:
        raise RuntimeError(
            "Distributed launch expects both RANK and WORLD_SIZE to be set."
        )

    distributed_launch = has_rank and has_world_size
    if distributed_launch:
        distributed_init()
        world_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        world_rank = 0
        world_size = 1

    context_parallel_size = world_size
    if (
        context_parallel_size % n_cameras != 0
        and n_cameras % context_parallel_size != 0
    ):
        raise ValueError(
            f"CP size {context_parallel_size} must divide n_cameras {n_cameras} or vice versa"
        )

    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("CUDA device count must be >= 1 for inference.")
    local_rank = world_rank % device_count
    torch_device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(torch_device)

    logger.info(
        f"Rank {world_rank} initialized distributed with context_parallel_size {context_parallel_size}"
    )
    return torch_device, world_rank, context_parallel_size


def main() -> None:
    """Main entry point for the gRPC server."""
    # Suppress warnings from FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecate
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*sdp_kernel.*")

    args = parse_args()

    # Parse denoising steps from comma-separated string
    denoising_step_list = [int(x.strip()) for x in args.denoising_steps.split(",")]

    device, world_rank, context_parallel_size = initialize_distributed(args.n_cameras)
    logger.info(
        "Using flashdreams pipeline backend; checkpoints are loaded lazily via flashdreams checkpoint loader."
    )

    # Initialize profiler if requested
    if args.enable_profiling:
        logger.info(f"Profiling enabled, output: {args.profile_output}")
        logger.info("  Data will be saved when server stops (Ctrl+C)")
        profiler = init_profiler(enabled=True)

        # Register cleanup to save profiling data on shutdown
        def save_profiling_data():
            logger.info("Saving profiling data...")
            profiler.print_summary()
            profiler.save(args.profile_output)

        atexit.register(save_profiling_data)

    # Common model kwargs shared between WorldModelService (rank 0) and WorldModelEngine (other ranks)
    model_kwargs: dict[str, Any] = dict(
        device=device,
        output_format=args.output_format,
        jpeg_quality=args.jpeg_quality,
        n_cameras=args.n_cameras,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        context_parallel_size=context_parallel_size,
        resolution=args.resolution,
        encode_with_pixel_shuffle=args.encode_with_pixel_shuffle,
        denoising_step_list=denoising_step_list,
        num_frames_per_block=args.num_frames_per_block,
        compile_net=not args.no_compile_net,
        use_cuda_graphs=not args.no_cuda_graphs,
        s3_credential_path=args.s3_credential_path,
        upsampler=args.upsampler,
        kv_cache_on_side_stream=args.kv_cache_on_side_stream,
        no_tae=args.no_tae,
    )

    server: grpc.Server | None = None
    service: WorldModelService | None = None
    if world_rank == 0:  # Only rank 0 runs the HTTP server
        logger.info("=" * 80)
        logger.info("Starting gRPC World Model Service")
        logger.info(f"Host: {args.host}")
        logger.info(f"Port: {args.port}")
        logger.info(f"Device: {device}")
        logger.info("=" * 80)

        # Create service instance
        service = WorldModelService(
            **model_kwargs,
            recording_dir=args.record_dir,
            seed_for_every_rollout=args.seed_for_every_rollout,
        )

        # Create gRPC server with increased message size limits
        # TODO: revisit once we use JPEG + streaming for large responses
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=args.max_workers),
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
            ],
        )

        # Add service to server
        video_model_pb2_grpc.add_WorldModelServiceServicer_to_server(service, server)

        # Bind to address
        server_address = f"{args.host}:{args.port}"
        server.add_insecure_port(server_address)
        logger.info(f"Server listening on {server_address}")

        # Show copy-pasteable address when binding to all interfaces
        if args.host == "0.0.0.0":
            external_ip = get_external_ip()
            logger.info(f"Connect using: {external_ip}:{args.port}")

        # Start server
        server.start()
        logger.info("Server started successfully. Press Ctrl+C to stop.")

        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            logger.critical("Shutting down server...")
            server.stop(grace=5)
            logger.critical("Server stopped.")
        except Exception as e:
            logger.error(f"Error in server.wait_for_termination(): {e}")
            raise e

        send_signal(ControlSignal.EXIT, device)

    else:  # non-rank 0 processes
        engine = WorldModelEngine(
            seed_for_every_rollout=args.seed_for_every_rollout, **model_kwargs
        )
        try:
            engine.wait_for_termination()
        except KeyboardInterrupt:
            logger.critical(f"Shutting down engine on rank {world_rank}...")

    # Release CUDA graphs before destroying process group (otherwise may pin NCCL communicator memory)
    if world_rank == 0 and server is not None and service is not None:
        del server
        del service
    else:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Wait for rank 0 to finish saving before destroying process group
    if dist.is_initialized():
        dist.barrier()
        logger.info(
            f"[Rank {world_rank}] All ranks synchronized, destroying process group..."
        )
        dist.destroy_process_group()
    # Hierarchical CP completed successfully
    logger.critical("Done!")


if __name__ == "__main__":
    main()
