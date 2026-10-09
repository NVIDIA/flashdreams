# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import hashlib
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image

from flashdreams.infra.acceleration.frame_prefetch import LazyCudaFrame
from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.infra.postprocess import (
    VideoPostprocessChainConfig,
    VideoPostprocessStream,
)
from flashdreams.infra.video_output import lazy_rgb_frames_from_video_tensor
from interactive_drive.backends.base import RenderBackend
from interactive_drive.config import (
    BevConfig,
    ChunkConfig,
    RasterConfig,
    VehicleConfig,
)
from interactive_drive.rasterizer import LudusConditionRasterizer
from interactive_drive.types import (
    FrameChunk,
    PresentedFrame,
    SceneBundle,
    TrajectoryChunk,
    VideoModelTimings,
)

_FIRST_STEADY_STATE_WARMUP_MESSAGE = "Optimizing world model..."


class WorldModelRenderBackend(RenderBackend):
    def __init__(
        self,
        pipeline: StreamInferencePipeline[Any, Any, Any],
        chunk: ChunkConfig,
        raster: RasterConfig,
        bev: BevConfig | None = None,
        vehicle: VehicleConfig | None = None,
        postprocess: VideoPostprocessChainConfig | None = None,
        debug_condition_frame_dir: Path | None = None,
    ) -> None:
        super().__init__(chunk=chunk, raster=raster)
        self._debug_condition_frame_dir = debug_condition_frame_dir
        vehicle = vehicle or VehicleConfig()
        self._rasterizer = LudusConditionRasterizer(
            raster,
            bev=bev,
            ego_dimensions_lwh=(
                vehicle.aabb_length_m,
                vehicle.aabb_width_m,
                vehicle.aabb_height_m,
            ),
            max_chunk_frames=max(chunk.initial_chunk_frames, chunk.chunk_frames),
        )
        self._pipeline = pipeline
        self._postprocess = postprocess or VideoPostprocessChainConfig()
        self._postprocess_enabled = self._postprocess.is_enabled()
        self._postprocess_stream = self._new_postprocess_stream()
        self._cache: Any | None = None
        self._step_index = 0
        self._pending_finalization_index: int | None = None
        self._pending_raster_frames: deque[PresentedFrame] = deque()
        self._first_transition_frame: PresentedFrame | None = None
        self._scene: SceneBundle | None = None
        self._view_names: tuple[str, ...] = ()
        self._next_chunk_count = 0
        self._debug_first_chunk_condition_frames: tuple[np.ndarray, ...] | None = None

    @property
    def can_prewarm(self) -> bool:
        return True

    @property
    def optimizes_on_first_chunk(self) -> bool:
        # First chunk triggers compile / CUDA-graph capture / Triton autotune,
        # which can take minutes on the first launch.
        return True

    def warmup_model(self) -> None:
        if self._chunk.initial_chunk_frames != 5:
            raise ValueError(
                "The flashdreams world-model path is locked to a 5-frame first chunk."
            )
        get_num_output_frames = getattr(self._pipeline, "get_num_output_frames", None)

        if not callable(get_num_output_frames):
            return
        first_frames = int(get_num_output_frames(0))
        steady_frames = int(get_num_output_frames(1))
        if first_frames != self._chunk.initial_chunk_frames:
            raise ValueError(
                "Pipeline initial output size does not match the demo chunk: "
                f"{first_frames} vs {self._chunk.initial_chunk_frames}."
            )
        if steady_frames != self._chunk.chunk_frames:
            raise ValueError(
                "Pipeline steady output size does not match the demo chunk: "
                f"{steady_frames} vs {self._chunk.chunk_frames}."
            )

    def load_scene(self, scene: SceneBundle) -> None:
        self._scene = scene
        self._view_names = tuple(camera.logical_name for camera in scene.cameras)
        self._next_chunk_count = 0
        self._debug_first_chunk_condition_frames = self._load_debug_condition_frames(
            self._debug_condition_frame_dir
        )
        load_start = time.perf_counter()
        self._rasterizer.load_scene(scene)
        rasterizer_end = time.perf_counter()
        logger.info(
            "[world-model] load_scene "
            f"rasterizer_ms={(rasterizer_end - load_start) * 1000.0:.1f} "
            f"total_ms={(rasterizer_end - load_start) * 1000.0:.1f}",
        )

    def render_first_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        scene = self._require_scene()
        chunk_start = time.perf_counter()
        if self._debug_first_chunk_condition_frames is None:
            raster_chunk = self._rasterizer.render_chunk(
                rig_poses_world=trajectory.rig_poses_world,
                timestamps_us=trajectory.timestamps_us,
                physics_debug_frames=trajectory.physics_debug_frames,
            )
            raster_end = time.perf_counter()
            condition_views = _condition_views(raster_chunk.frames)
            display_frames = raster_chunk.frames
        else:
            if len(self._view_names) != 1:
                raise RuntimeError(
                    "The HD-map condition-frame override holds one camera's frames, "
                    f"so it cannot drive a rig of {len(self._view_names)}."
                )
            condition_views = [
                [frame.copy() for frame in self._debug_first_chunk_condition_frames]
            ]
            physx_frames = (
                self._rasterizer.render_physx_debug_lazy_frames(
                    trajectory.rig_poses_world,
                    trajectory.timestamps_us,
                    trajectory.physics_debug_frames,
                )
                if trajectory.physics_debug_frames
                else ()
            )
            raster_end = time.perf_counter()
            display_frames = tuple(
                PresentedFrame(
                    timestamp_us=int(timestamp_us),
                    view_rgb_host_uint8=(frame.copy(),),
                    depth_host_f32=None,
                    rgb_native=None,
                    depth_native=None,
                    physx_debug=(
                        trajectory.physics_debug_frames[index]
                        if trajectory.physics_debug_frames
                        else None
                    ),
                    physx_rgb_host_uint8=(
                        physx_frames[index] if physx_frames else None
                    ),
                )
                for index, (timestamp_us, frame) in enumerate(
                    zip(
                        trajectory.timestamps_us,
                        self._debug_first_chunk_condition_frames,
                        strict=True,
                    )
                )
            )
            logger.info(
                "[world-model] first_chunk using official hdmap override "
                f"dir={self._debug_condition_frame_dir}",
            )
        _log_prompt_handoff("first_chunk.start", scene)
        model_views = self._start_pipeline(
            scene.initial_rgbs, condition_views, scene.prompt
        )
        model_end = time.perf_counter()
        merged_frames = self._merge_frames(
            display_frames,
            model_views,
            annotate_first_transition=True,
        )
        merge_end = time.perf_counter()
        logger.info(
            "[world-model] first_chunk "
            f"frames={len(trajectory.timestamps_us)} "
            f"raster_ms={(raster_end - chunk_start) * 1000.0:.1f} "
            f"model_ms={(model_end - raster_end) * 1000.0:.1f} "
            f"merge_ms={(merge_end - model_end) * 1000.0:.1f} "
            f"total_ms={(merge_end - chunk_start) * 1000.0:.1f}",
        )
        return FrameChunk(
            frames=merged_frames,
            boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
            source_name="world_model",
            video_model_timings=VideoModelTimings(
                condition_start_time=chunk_start,
                condition_ready_time=raster_end,
                model_start_time=raster_end,
                model_ready_time=model_end,
                merge_start_time=model_end,
                merge_ready_time=merge_end,
            ),
        )

    def render_next_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        self._require_scene()
        chunk_start = time.perf_counter()
        raster_chunk = self._rasterizer.render_chunk(
            rig_poses_world=trajectory.rig_poses_world,
            timestamps_us=trajectory.timestamps_us,
            dynamic_actors=trajectory.dynamic_actors,
            physics_debug_frames=trajectory.physics_debug_frames,
        )
        raster_end = time.perf_counter()
        condition_views = _condition_views(raster_chunk.frames)
        model_views = self._continue_pipeline(condition_views)
        model_end = time.perf_counter()
        merged_frames = self._merge_frames(raster_chunk.frames, model_views)
        merge_end = time.perf_counter()
        self._next_chunk_count += 1
        total_ms = (merge_end - chunk_start) * 1000.0
        if trajectory.physx_timings is not None:
            timing = trajectory.physx_timings
            physx_timing = (
                f"physx_ms={timing.total_ms:.1f} "
                # f"physx_sync_ms={timing.synchronize_ms:.1f} "
                # f"physx_actor_update_ms={timing.actor_update_ms:.1f} "
                # f"physx_solver_ms={timing.solver_ms:.1f} "
                # f"physx_readback_ms={timing.readback_ms:.1f} "
                # f"physx_bridge_ms={timing.bridge_ms:.1f} "
                # f"physx_visible_actors={timing.max_visible_actors} "
                # f"physx_detached_actors={timing.max_detached_actors} "
            )
        else:
            physx_timing = (
                f"physx_ms={trajectory.physx_elapsed_s * 1000.0:.1f} "
                if trajectory.physx_elapsed_s is not None
                else ""
            )
        if (
            self._next_chunk_count <= 3
            or self._next_chunk_count % 10 == 0
            or total_ms > 500.0
        ):
            logger.info(
                "[world-model] next_chunk "
                f"index={self._next_chunk_count} "
                f"frames={len(trajectory.timestamps_us)} "
                f"{physx_timing}"
                f"raster_ms={(raster_end - chunk_start) * 1000.0:.1f} "
                f"model_ms={(model_end - raster_end) * 1000.0:.1f} "
                f"merge_ms={(merge_end - model_end) * 1000.0:.1f} "
                f"total_ms={total_ms:.1f}",
            )
        return FrameChunk(
            frames=merged_frames,
            boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
            source_name="world_model",
            video_model_timings=VideoModelTimings(
                condition_start_time=chunk_start,
                condition_ready_time=raster_end,
                model_start_time=raster_end,
                model_ready_time=model_end,
                merge_start_time=model_end,
                merge_ready_time=merge_end,
            ),
        )

    def reset(self) -> None:
        self._clear_pipeline(finalize_pending=False)
        self._next_chunk_count = 0

    def reset_scene_conditioning(self) -> None:
        self._clear_pipeline(finalize_pending=False)
        self._next_chunk_count = 0

    def set_postprocess_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and not self._postprocess.is_enabled():
            raise RuntimeError(
                "Cannot enable post-processing without --postprocess-preset."
            )
        if enabled == self._postprocess_enabled:
            return
        self._postprocess_enabled = enabled
        if self._postprocess_stream is not None:
            self._postprocess_stream.finish()
        self._pending_raster_frames.clear()
        self._first_transition_frame = None
        self._postprocess_stream = self._new_postprocess_stream()

    def close(self) -> None:
        self._clear_pipeline(finalize_pending=True, recreate_postprocess_stream=False)
        self._rasterizer.cleanup()

    def finalize(self) -> dict[str, float] | None:
        return self._finalize_pending()

    def finish(self) -> tuple[PresentedFrame, ...]:
        """Flush delayed post-processing output for a finite rollout."""
        self._finalize_pending()
        output = (
            None
            if self._postprocess_stream is None
            else self._postprocess_stream.finish()
        )
        model_views = [] if output is None else _model_view_frames(output.detach())
        _synchronize_cuda_frame_event([frame for view in model_views for frame in view])
        merged_frames = self._merge_frames((), model_views)
        if self._pending_raster_frames:
            raise RuntimeError(
                "Post-processing finished without emitting "
                f"{len(self._pending_raster_frames)} buffered frames."
            )
        return merged_frames

    def _start_pipeline(
        self,
        initial_rgbs: Sequence[object],
        condition_views: Sequence[Sequence[object]],
        prompt: str,
    ) -> list[list[LazyCudaFrame]]:
        self._clear_pipeline(finalize_pending=False)
        self._cache = self._initialize_cache(initial_rgbs, prompt)
        return self._step_pipeline(condition_views)

    def _continue_pipeline(
        self, condition_views: Sequence[Sequence[object]]
    ) -> list[list[LazyCudaFrame]]:
        if self._cache is None:
            raise RuntimeError(
                "render_first_chunk() must run before render_next_chunk()."
            )
        return self._step_pipeline(condition_views)

    def _step_pipeline(
        self, condition_views: Sequence[Sequence[object]]
    ) -> list[list[LazyCudaFrame]]:
        if self._cache is None:
            raise RuntimeError("The stream pipeline cache has not been initialized.")
        self._finalize_pending()
        expected_frames = (
            self._chunk.initial_chunk_frames
            if self._step_index == 0
            else self._chunk.chunk_frames
        )
        if len(condition_views) != len(self._view_names):
            raise ValueError(
                "Conditioning does not cover the scene's rig: "
                f"{len(condition_views)} views vs {len(self._view_names)} cameras."
            )
        for view_name, frames in zip(self._view_names, condition_views, strict=True):
            if len(frames) != expected_frames:
                raise ValueError(
                    "Condition chunk length does not match the demo chunk size: "
                    f"{len(frames)} vs {expected_frames} for view {view_name!r}."
                )
        step_index = self._step_index
        with torch.cuda.device(self._pipeline.device):
            video_chunk = self._pipeline.generate(
                autoregressive_index=step_index,
                cache=self._cache,
                input=self._condition_tensor(condition_views),
            )
        self._pending_finalization_index = step_index
        output = (
            video_chunk
            if self._postprocess_stream is None
            else self._postprocess_stream.process(
                video_chunk,
                autoregressive_index=step_index,
            )
        )
        self._step_index += 1
        return _model_view_frames(output.detach())

    def _initialize_cache(self, initial_rgbs: Sequence[object], prompt: str) -> Any:
        with torch.cuda.device(self._pipeline.device):
            return self._pipeline.initialize_cache(
                # ``text`` is ``[B, V]``, so the scene's one sentence is aimed
                # at each camera in turn. A scene describes a place rather than
                # a camera, and the pipeline sizes its caches from the view
                # count it reads here, which left a rig with caches for one.
                text=[[prompt for _ in self._view_names]],
                image=self._initial_rgb_tensor(initial_rgbs),
                view_names=list(self._view_names),
            )

    def _finalize_pending(self) -> dict[str, float] | None:
        if self._cache is None or self._pending_finalization_index is None:
            return {}
        with torch.cuda.device(self._pipeline.device):
            metrics = self._pipeline.finalize(
                autoregressive_index=self._pending_finalization_index,
                cache=self._cache,
            )
        self._pending_finalization_index = None
        return metrics

    def _clear_pipeline(
        self,
        *,
        finalize_pending: bool,
        recreate_postprocess_stream: bool = True,
    ) -> None:
        if finalize_pending:
            self._finalize_pending()
        else:
            self._pending_finalization_index = None
        self._cache = None
        self._step_index = 0
        if self._postprocess_stream is not None:
            self._postprocess_stream.finish()
        self._pending_raster_frames.clear()
        self._first_transition_frame = None
        if recreate_postprocess_stream:
            self._postprocess_stream = self._new_postprocess_stream()

    def _new_postprocess_stream(self) -> VideoPostprocessStream | None:
        if not self._postprocess_enabled:
            return None
        return VideoPostprocessStream(
            postprocess=self._postprocess,
            output_layout="bvtchw",
            fps=self._chunk.fps,
            per_view=False,
            world_size=1,
        )

    def _initial_rgb_tensor(self, frames: Sequence[object]) -> torch.Tensor:
        views = [
            torch.from_numpy(_rgb_hwc_uint8(frame)).permute(2, 0, 1) for frame in frames
        ]
        tensor = torch.stack(views, dim=0).unsqueeze(0).unsqueeze(2)
        return self._to_model_range(tensor)

    def _condition_tensor(
        self, condition_views: Sequence[Sequence[object]]
    ) -> torch.Tensor:
        cuda_views = _condition_cuda_views(condition_views)
        if cuda_views is not None:
            video = torch.stack(
                [view.permute(0, 3, 1, 2) for view in cuda_views], dim=0
            )
            return self._to_model_range(video.unsqueeze(0))
        video = np.stack(
            [
                np.stack([_rgb_hwc_uint8(frame) for frame in frames], axis=0)
                for frames in condition_views
            ],
            axis=0,
        )
        tensor = torch.from_numpy(np.ascontiguousarray(video))
        tensor = tensor.permute(0, 1, 4, 2, 3).unsqueeze(0)
        return self._to_model_range(tensor)

    def _to_model_range(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(device=self._pipeline.device, dtype=torch.bfloat16)
        return tensor / 127.5 - 1.0

    def _require_scene(self) -> SceneBundle:
        if self._scene is None:
            raise RuntimeError(
                "warmup() must be called before rendering world-model chunks"
            )
        return self._scene

    def _load_debug_condition_frames(
        self, condition_dir: Path | None
    ) -> tuple[np.ndarray, ...] | None:
        if condition_dir is None:
            return None
        frames: list[np.ndarray] = []
        for i in range(self._chunk.initial_chunk_frames):
            path = condition_dir / f"hdmap_{i:02d}.png"
            if not path.exists():
                raise FileNotFoundError(
                    f"debug_condition_frame_dir is missing required file {path}"
                )
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                if rgb.size != self._raster.resolution_wh:
                    rgb = rgb.resize(
                        self._raster.resolution_wh,
                        resample=Image.Resampling.BILINEAR,
                    )
                frames.append(np.array(rgb, dtype=np.uint8))
        return tuple(frames)

    def _merge_frames(
        self,
        raster_frames: Sequence[PresentedFrame],
        model_views: Sequence[Sequence[object]],
        *,
        annotate_first_transition: bool = False,
    ) -> tuple[PresentedFrame, ...]:
        self._pending_raster_frames.extend(raster_frames)
        if annotate_first_transition and raster_frames:
            self._first_transition_frame = raster_frames[-1]
        if model_views and len(model_views) != len(self._view_names):
            raise ValueError(
                "World-model output does not cover the scene's rig: "
                f"{len(model_views)} views vs {len(self._view_names)} cameras."
            )
        lengths = {len(frames) for frames in model_views}
        if len(lengths) > 1:
            raise ValueError(
                f"World-model views returned unequal frame counts: {sorted(lengths)}"
            )
        model_count = lengths.pop() if lengths else 0
        if model_count > len(self._pending_raster_frames):
            raise ValueError(
                "World-model output exceeds the buffered conditioning frames: "
                f"{model_count} vs {len(self._pending_raster_frames)}"
            )

        merged: list[PresentedFrame] = []
        for index in range(model_count):
            raster_frame = self._pending_raster_frames.popleft()
            merged.append(
                PresentedFrame(
                    timestamp_us=raster_frame.timestamp_us,
                    view_rgb_host_uint8=raster_frame.view_rgb_host_uint8,
                    depth_host_f32=raster_frame.depth_host_f32,
                    rgb_native=raster_frame.rgb_native,
                    depth_native=raster_frame.depth_native,
                    view_model_rgb_host_uint8=tuple(
                        view[index] for view in model_views
                    ),
                    bev_host_uint8=raster_frame.bev_host_uint8,
                    physx_debug=raster_frame.physx_debug,
                    physx_rgb_host_uint8=raster_frame.physx_rgb_host_uint8,
                    status_message=(
                        _FIRST_STEADY_STATE_WARMUP_MESSAGE
                        if raster_frame is self._first_transition_frame
                        else None
                    ),
                )
            )
            if raster_frame is self._first_transition_frame:
                self._first_transition_frame = None
        return tuple(merged)


def _rgb_hwc_uint8(frame: object) -> np.ndarray:
    return np.ascontiguousarray(
        np.array(np.asarray(frame, dtype=np.uint8)[..., :3], copy=True)
    )


def _condition_views(frames: Sequence[PresentedFrame]) -> list[list[object]]:
    """A chunk's conditioning as one column per camera, in rig order.

    Frames arrive a moment at a time and the pipeline wants a view at a time,
    so this is the transpose.
    """
    if not frames:
        return []
    views = frames[0].views
    for frame in frames:
        if frame.views != views:
            raise ValueError(
                "A raster chunk must hold the same cameras in every frame; "
                f"found {frame.views} where the chunk opened with {views}."
            )
    return [
        [frame.view_rgb_host_uint8[view] for frame in frames] for view in range(views)
    ]


def _model_view_frames(output: torch.Tensor) -> list[list[LazyCudaFrame]]:
    """One lazy frame list per view of a ``bvtchw`` chunk, in rig order."""
    return [
        lazy_rgb_frames_from_video_tensor(output, layout="bvtchw", view_index=view)
        for view in range(int(output.shape[1]))
    ]


def _condition_cuda_views(
    condition_views: Sequence[Sequence[object]],
) -> list[torch.Tensor] | None:
    """Every view as one CUDA video, or ``None`` if any view is not resident."""
    videos: list[torch.Tensor] = []
    for frames in condition_views:
        video = _condition_cuda_video(frames)
        if video is None:
            return None
        videos.append(video)
    return videos


def _condition_cuda_video(
    condition_frames: Sequence[object],
) -> torch.Tensor | None:
    tensors: list[torch.Tensor] = []
    device: torch.device | None = None
    for frame in condition_frames:
        if not isinstance(frame, LazyCudaFrame):
            return None
        try:
            tensor = frame.to_cuda_tensor()
        except RuntimeError:
            return None
        if (
            not torch.is_tensor(tensor)
            or not tensor.is_cuda
            or tensor.dtype != torch.uint8
            or tensor.ndim != 3
            or tensor.shape[-1] < 3
        ):
            return None
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            return None
        rgb = tensor[..., :3]
        tensors.append(rgb if rgb.is_contiguous() else rgb.contiguous())

    if not tensors:
        return None
    return torch.stack(tensors, dim=0)


def _synchronize_cuda_frame_event(frames: Sequence[object]) -> None:
    for frame in frames:
        to_cuda_event = getattr(frame, "to_cuda_event", None)
        event = to_cuda_event() if callable(to_cuda_event) else None
        if event is None:
            continue
        synchronize = getattr(event, "synchronize", None)
        if callable(synchronize):
            synchronize()


def _log_prompt_handoff(stage: str, scene: SceneBundle) -> None:
    prompt = scene.prompt
    prompt_text = " ".join(prompt.split())
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    logger.info(
        "[world-model] prompt_handoff "
        f"stage={stage!r} "
        f"scene={scene.scene_path.name!r} "
        f"prompt_sha256={prompt_hash!r} "
        f"length={len(prompt)} "
        f"text={prompt_text!r}",
    )
