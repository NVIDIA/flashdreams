# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from loguru import logger
from omnidreams.interactive_drive.backends.base import RenderBackend
from omnidreams.interactive_drive.config import (
    BevConfig,
    ChunkConfig,
    RasterConfig,
    WorldModelProfileConfig,
)
from omnidreams.interactive_drive.rasterizer import LudusConditionRasterizer
from omnidreams.interactive_drive.types import (
    FrameChunk,
    PresentedFrame,
    SceneBundle,
    TrajectoryChunk,
)
from omnidreams.interactive_drive.world_model.flashdreams_adapter import (
    FlashdreamsWorldModelSession,
)
from omnidreams.interactive_drive.world_model.manifest import WorldModelManifest
from PIL import Image

_FIRST_STEADY_STATE_WARMUP_MESSAGE = "Optimizing world model..."


class WorldModelRenderBackend(RenderBackend):
    def __init__(
        self,
        manifest: WorldModelManifest,
        chunk: ChunkConfig,
        raster: RasterConfig,
        profile: WorldModelProfileConfig | None = None,
        bev: BevConfig | None = None,
        offload_text_encoder: bool = False,
    ) -> None:
        super().__init__(chunk=chunk, raster=raster)
        self._manifest = manifest
        self._rasterizer = LudusConditionRasterizer(raster, bev=bev)
        self._session = FlashdreamsWorldModelSession(
            manifest,
            profile=profile,
            offload_text_encoder=offload_text_encoder,
        )
        self._scene: SceneBundle | None = None
        self._next_chunk_count = 0
        self._debug_first_chunk_condition_frames: tuple[np.ndarray, ...] | None = None

    @property
    def can_prewarm(self) -> bool:
        return self._session.can_prewarm

    @property
    def optimizes_on_first_chunk(self) -> bool:
        # First chunk triggers compile / CUDA-graph capture / Triton autotune,
        # which can take minutes on the first launch.
        return True

    def warmup_model(self) -> None:
        if self._manifest.resolution_wh != self._raster.resolution_wh:
            raise ValueError(
                "World-model manifest resolution does not match the renderer resolution: "
                f"{self._manifest.resolution_wh} vs {self._raster.resolution_wh}"
            )
        if self._manifest.fps != self._chunk.fps:
            raise ValueError(
                f"World-model manifest fps {self._manifest.fps} does not match chunk fps {self._chunk.fps}"
            )
        if self._manifest.num_frames_per_block != self._chunk.chunk_frames:
            raise ValueError(
                "World-model manifest num_frames_per_block does not match steady-state chunk size: "
                f"{self._manifest.num_frames_per_block} vs {self._chunk.chunk_frames}"
            )
        if self._chunk.initial_chunk_frames != 5:
            raise ValueError(
                "The flashdreams world-model path is locked to a 5-frame first chunk."
            )
        start = time.perf_counter()
        self._session.warmup_model()
        logger.info(
            f"[world-model] model warmup session_ms={(time.perf_counter() - start) * 1000.0:.1f}",
        )

    def load_scene(self, scene: SceneBundle) -> None:
        self._scene = scene
        self._next_chunk_count = 0
        self._debug_first_chunk_condition_frames = self._load_debug_condition_frames(
            self._manifest.debug_condition_frame_dir
        )
        load_start = time.perf_counter()
        self._rasterizer.load_scene(scene)
        rasterizer_end = time.perf_counter()
        # Per-scene conditioning prep. On the default path this is a no-op
        # (the prompt is re-embedded per rollout in the session); under
        # --offload-text-encoder it (re)builds the per-scene embeddings.
        _log_prompt_handoff("load_scene.prepare_for_scene", scene)
        self._session.prepare_for_scene(
            initial_rgb=scene.initial_rgb, prompt=scene.prompt
        )
        prepare_end = time.perf_counter()
        logger.info(
            "[world-model] load_scene "
            f"rasterizer_ms={(rasterizer_end - load_start) * 1000.0:.1f} "
            f"prepare_ms={(prepare_end - rasterizer_end) * 1000.0:.1f} "
            f"total_ms={(prepare_end - load_start) * 1000.0:.1f}",
        )

    def render_first_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        scene = self._require_scene()
        chunk_start = time.perf_counter()
        if self._debug_first_chunk_condition_frames is None:
            raster_chunk = self._rasterizer.render_chunk(
                rig_poses_world=trajectory.rig_poses_world,
                timestamps_us=trajectory.timestamps_us,
            )
            raster_end = time.perf_counter()
            condition_frames = [frame.rgb_host_uint8 for frame in raster_chunk.frames]
            display_frames = raster_chunk.frames
        else:
            raster_end = time.perf_counter()
            condition_frames = [
                frame.copy() for frame in self._debug_first_chunk_condition_frames
            ]
            display_frames = tuple(
                PresentedFrame(
                    timestamp_us=int(timestamp_us),
                    rgb_host_uint8=frame.copy(),
                    depth_host_f32=None,
                    rgb_native=None,
                    depth_native=None,
                )
                for timestamp_us, frame in zip(
                    trajectory.timestamps_us,
                    self._debug_first_chunk_condition_frames,
                    strict=True,
                )
            )
            logger.info(
                "[world-model] first_chunk using official hdmap override "
                f"dir={self._manifest.debug_condition_frame_dir}",
            )
        _log_prompt_handoff("first_chunk.start", scene)
        model_frames = self._session.start(
            scene.initial_rgb, condition_frames, scene.prompt
        )
        model_end = time.perf_counter()
        merged_frames = self._merge_frames(
            display_frames,
            model_frames,
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
            source_name="omnidreams",
        )

    def render_next_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        self._require_scene()
        chunk_start = time.perf_counter()
        raster_chunk = self._rasterizer.render_chunk(
            rig_poses_world=trajectory.rig_poses_world,
            timestamps_us=trajectory.timestamps_us,
        )
        raster_end = time.perf_counter()
        condition_frames = [frame.rgb_host_uint8 for frame in raster_chunk.frames]
        model_frames = self._session.continue_generation(condition_frames)
        model_end = time.perf_counter()
        merged_frames = self._merge_frames(raster_chunk.frames, model_frames)
        merge_end = time.perf_counter()
        self._next_chunk_count += 1
        total_ms = (merge_end - chunk_start) * 1000.0
        if (
            self._next_chunk_count <= 3
            or self._next_chunk_count % 10 == 0
            or total_ms > 500.0
        ):
            logger.info(
                "[world-model] next_chunk "
                f"index={self._next_chunk_count} "
                f"frames={len(trajectory.timestamps_us)} "
                f"raster_ms={(raster_end - chunk_start) * 1000.0:.1f} "
                f"model_ms={(model_end - raster_end) * 1000.0:.1f} "
                f"merge_ms={(merge_end - model_end) * 1000.0:.1f} "
                f"total_ms={total_ms:.1f}",
            )
        return FrameChunk(
            frames=merged_frames,
            boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
            source_name="omnidreams",
        )

    def reset(self) -> None:
        self._session.reset()
        self._next_chunk_count = 0

    def reset_scene_conditioning(self) -> None:
        self._session.reset(clear_precomputed_embeddings=True)
        self._next_chunk_count = 0

    def close(self) -> None:
        self._session.close()
        self._rasterizer.cleanup()

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
                if rgb.size != self._manifest.resolution_wh:
                    rgb = rgb.resize(
                        self._manifest.resolution_wh, resample=Image.Resampling.BILINEAR
                    )
                frames.append(np.array(rgb, dtype=np.uint8))
        return tuple(frames)

    def _merge_frames(
        self,
        raster_frames: Sequence[PresentedFrame],
        model_frames: Sequence[object],
        *,
        annotate_first_transition: bool = False,
    ) -> tuple[PresentedFrame, ...]:
        if len(raster_frames) != len(model_frames):
            raise ValueError(
                "World-model output frame count does not match the conditioning chunk size: "
                f"{len(model_frames)} vs {len(raster_frames)}"
            )

        merged: list[PresentedFrame] = []
        last_index = len(raster_frames) - 1
        for index, (raster_frame, model_rgb) in enumerate(
            zip(raster_frames, model_frames, strict=True)
        ):
            merged.append(
                PresentedFrame(
                    timestamp_us=raster_frame.timestamp_us,
                    rgb_host_uint8=raster_frame.rgb_host_uint8,
                    depth_host_f32=raster_frame.depth_host_f32,
                    rgb_native=raster_frame.rgb_native,
                    depth_native=raster_frame.depth_native,
                    model_rgb_host_uint8=model_rgb,
                    bev_host_uint8=raster_frame.bev_host_uint8,
                    status_message=(
                        _FIRST_STEADY_STATE_WARMUP_MESSAGE
                        if annotate_first_transition and index == last_index
                        else None
                    ),
                )
            )
        return tuple(merged)


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
