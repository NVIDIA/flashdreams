# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot replay runtime for the shared demo runner."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner_io import load_first_frame_tensor
from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import InferenceInput
from flashdreams.runtime.interfaces import InferenceSession
from flashdreams.runtime.types import StepRequest, StepResult, TimeWindow
from lingbot.encoder.camctrl import CamCtrlInput
from lingbot.encoder.utils import get_Ks_transformed, preprocess_example_poses
from lingbot.runner import (
    _INTRINSICS_REFERENCE_HEIGHT,
    _INTRINSICS_REFERENCE_WIDTH,
)

from .spec import LingbotReplayScenario

PipelineFactory = Callable[[Any, str], Any]


@dataclass(frozen=True, kw_only=True, slots=True)
class LingbotReplayRuntimeOptions:
    """Construction knobs for the replay runtime."""

    pipeline_config: Any
    pipeline_factory: PipelineFactory | None = None
    output_layout: VideoTensorLayout = "tchw"


class LingbotReplayRuntime:
    """Heavyweight Lingbot runtime consumed by ``run_inference_session``."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        options: LingbotReplayRuntimeOptions,
    ) -> None:
        self.config = config
        self.options = options
        if _is_torchrun_env() and not dist.is_initialized():
            init_distributed()

        if dist.is_initialized():
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
            device = f"cuda:{self.local_rank}"
        else:
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0
            device = config.device or "cuda"

        self.is_rank_zero = self.global_rank == 0
        factory = options.pipeline_factory or _default_pipeline_factory
        self.pipeline = factory(options.pipeline_config, device)

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        scenario = _scenario_from_inputs(inputs)
        return LingbotReplaySession(
            pipeline=self.pipeline,
            scenario=scenario,
            device=torch.device(f"cuda:{self.local_rank}")
            if dist.is_initialized()
            else torch.device(self.config.device or "cuda"),
            is_rank_zero=self.is_rank_zero,
            output_layout=self.options.output_layout,
        )

    def close(self) -> None:
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()
            del self.pipeline
        device = torch.device(self.config.device or "cuda")
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class LingbotReplaySession:
    """One MP4 replay rollout over a prepared Lingbot scenario."""

    def __init__(
        self,
        *,
        pipeline: Any,
        scenario: LingbotReplayScenario,
        device: torch.device,
        is_rank_zero: bool,
        output_layout: VideoTensorLayout,
    ) -> None:
        self.pipeline = pipeline
        self.scenario = scenario
        self.device = device
        self.is_rank_zero = is_rank_zero
        self.output_layout = output_layout
        self.dtype = torch.bfloat16
        self._closed = False
        self._step_index = 0
        self._frame_start = 0
        self._cache = self._initialize_cache()
        (
            self._camera_intrinsics,
            self._camera_poses,
            self._world_scale,
        ) = self._load_camera_controls()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
        if dist.is_initialized():
            dist.barrier()

    def next_step_request(self) -> StepRequest | None:
        if self._closed:
            return None
        if self._step_index >= self.scenario.total_blocks:
            return None
        num_frames = int(self.pipeline.get_num_output_frames(self._step_index))
        if self._frame_start + num_frames > self._camera_poses.shape[0]:
            return None
        return StepRequest(step_index=self._step_index)

    def step(self, inputs: InferenceInput) -> StepResult:
        del inputs
        if self._closed:
            raise RuntimeError("Lingbot replay session is closed.")

        step_index = self._step_index
        num_frames = int(self.pipeline.get_num_output_frames(step_index))
        frame_start = self._frame_start
        frame_end = frame_start + num_frames
        if frame_end > self._camera_poses.shape[0]:
            raise RuntimeError("Lingbot replay scenario ran out of camera frames.")

        if self.is_rank_zero:
            logger.info(
                "Lingbot demo replay step {} frames=[{}, {})",
                step_index,
                frame_start,
                frame_end,
            )
        camctrl_input = CamCtrlInput(
            intrinsics=self._camera_intrinsics[frame_start:frame_end],
            poses=self._camera_poses[frame_start:frame_end],
            world_scale=self._world_scale,
        )
        start_t = time.perf_counter()
        video_chunk = self.pipeline.generate(
            autoregressive_index=step_index,
            cache=self._cache,
            input=camctrl_input,
        )
        stats = self.pipeline.finalize(
            autoregressive_index=step_index,
            cache=self._cache,
        )
        elapsed_s = time.perf_counter() - start_t
        self._step_index += 1
        self._frame_start = frame_end

        metrics = _numeric_stats(stats)
        metrics.setdefault("model_step_s", elapsed_s)
        return StepResult(
            step_index=step_index,
            output=VideoStepResult.from_video_chunk(
                chunk_index=step_index,
                video_chunk=video_chunk,
                layout=self.output_layout,
                stats=metrics,
            ),
            frame_count=num_frames,
            output_window=TimeWindow(
                start_s=frame_start / self.scenario.fps,
                end_s=frame_end / self.scenario.fps,
            ),
            metrics=metrics,
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if inputs is not None:
            scenario = _scenario_from_inputs(inputs)
            if scenario != self.scenario:
                raise ValueError("Lingbot replay reset cannot swap scenarios.")
        cache = getattr(self, "_cache", None)
        if cache is not None:
            del self._cache
        self._cache = self._initialize_cache()
        self._step_index = 0
        self._frame_start = 0

    def close(self) -> None:
        self._closed = True
        cache = getattr(self, "_cache", None)
        if cache is not None:
            del self._cache

    def _initialize_cache(self) -> Any:
        scenario = self.scenario
        first_frames_t = load_first_frame_tensor(
            scenario.image_path,
            pixel_height=scenario.pixel_height,
            pixel_width=scenario.pixel_width,
            device=self.device,
            dtype=self.dtype,
            interpolation="cubic",
            install_hint="Install the lingbot plugin: pip install flashdreams-lingbot.",
        )
        return self.pipeline.initialize_cache(
            text=[scenario.prompt],
            image=first_frames_t,
        )

    def _load_camera_controls(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        scenario = self.scenario
        intrinsics = np.load(scenario.intrinsic_path)
        intrinsics_t = torch.from_numpy(np.asarray(intrinsics)).to(
            device=self.device,
            dtype=torch.float32,
        )
        camera_intrinsics_t = get_Ks_transformed(
            intrinsics_t,
            height_org=_INTRINSICS_REFERENCE_HEIGHT,
            width_org=_INTRINSICS_REFERENCE_WIDTH,
            height_resize=scenario.pixel_height,
            width_resize=scenario.pixel_width,
            height_final=scenario.pixel_height,
            width_final=scenario.pixel_width,
        )

        poses = np.load(scenario.pose_path)
        poses, world_scale = preprocess_example_poses(np.asarray(poses))
        camera_poses_t = torch.from_numpy(poses).to(
            device=self.device,
            dtype=torch.float32,
        )
        if self.is_rank_zero:
            logger.info(
                "Loaded Lingbot demo camera controls intrinsics={} poses={}",
                tuple(camera_intrinsics_t.shape),
                tuple(camera_poses_t.shape),
            )
        return camera_intrinsics_t, camera_poses_t, float(world_scale)


def _default_pipeline_factory(pipeline_config: Any, device: str) -> Any:
    return pipeline_config.setup().to(device=device).eval()


def _scenario_from_inputs(inputs: InferenceInput) -> LingbotReplayScenario:
    scenario = inputs.global_conditioning.get("scenario")
    if not isinstance(scenario, LingbotReplayScenario):
        raise TypeError(
            "Lingbot replay runtime requires global_conditioning['scenario'] "
            "to be a LingbotReplayScenario."
        )
    return scenario


def _numeric_stats(stats: Any) -> dict[str, float | int]:
    if not isinstance(stats, Mapping):
        return {}
    return {
        str(key): value
        for key, value in stats.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    }


def _is_torchrun_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


__all__ = [
    "LingbotReplayRuntime",
    "LingbotReplayRuntimeOptions",
    "LingbotReplaySession",
    "PipelineFactory",
]
