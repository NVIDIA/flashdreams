# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams replay runtime for the shared demo runner."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger
from omnidreams.runner import _load_video

from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner_io import (
    DEFAULT_RUNNER_INSTALL_HINT,
    load_first_frame_tensor,
)
from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import InferenceInput
from flashdreams.runtime.interfaces import InferenceSession
from flashdreams.runtime.types import StepRequest, StepResult

from .spec import OmnidreamsReplayScenario

PipelineFactory = Callable[[Any, str], Any]


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsReplayRuntimeOptions:
    """Construction knobs for the replay runtime."""

    pipeline_config: Any
    pipeline_factory: PipelineFactory | None = None
    output_layout: VideoTensorLayout = "bvtchw"


class OmnidreamsReplayRuntime:
    """Heavyweight OmniDreams runtime consumed by ``run_inference_session``."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        options: OmnidreamsReplayRuntimeOptions,
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
        return OmnidreamsReplaySession(
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


class OmnidreamsReplaySession:
    """One MP4 replay rollout over a prepared scenario."""

    def __init__(
        self,
        *,
        pipeline: Any,
        scenario: OmnidreamsReplayScenario,
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
        self._hdmap_videos = self._load_hdmaps()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
        if dist.is_initialized():
            dist.barrier()

    def next_step_request(self) -> StepRequest | None:
        if self._closed:
            return None
        if self._step_index >= self.scenario.total_blocks:
            return None
        num_frames = int(self.pipeline.get_num_frames(self._step_index))
        if self._frame_start + num_frames > self._hdmap_videos.shape[2]:
            return None
        return StepRequest(step_index=self._step_index)

    def step(self, inputs: InferenceInput) -> StepResult:
        del inputs
        if self._closed:
            raise RuntimeError("OmniDreams replay session is closed.")

        step_index = self._step_index
        num_frames = int(self.pipeline.get_num_frames(step_index))
        frame_end = self._frame_start + num_frames
        logger.info(
            "OmniDreams demo replay step {} frames=[{}, {})",
            step_index,
            self._frame_start,
            frame_end,
        )
        start_t = time.perf_counter()
        video_chunk = self.pipeline.generate(
            autoregressive_index=step_index,
            cache=self._cache,
            hdmap=self._hdmap_videos[:, :, self._frame_start : frame_end],
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
            metrics=metrics,
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if inputs is not None:
            scenario = _scenario_from_inputs(inputs)
            if scenario != self.scenario:
                raise ValueError("OmniDreams replay reset cannot swap scenarios.")
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
        first_frames = [
            load_first_frame_tensor(
                path,
                pixel_height=scenario.pixel_height,
                pixel_width=scenario.pixel_width,
                device=self.device,
                dtype=self.dtype,
                allow_video=True,
                install_hint=DEFAULT_RUNNER_INSTALL_HINT,
            )
            for path in scenario.first_frame_paths
        ]
        first_frames_t = torch.stack(first_frames, dim=0).unsqueeze(0)
        cache = self.pipeline.initialize_cache(
            text=[list(scenario.prompts)],
            image=first_frames_t,
            view_names=list(scenario.camera_names),
        )
        release = getattr(self.pipeline, "release_oneshot_encoders", None)
        if callable(release):
            release()
        return cache

    def _load_hdmaps(self) -> torch.Tensor:
        scenario = self.scenario
        videos = [
            _load_video(
                path,
                pixel_height=scenario.pixel_height,
                pixel_width=scenario.pixel_width,
                device=self.device,
                dtype=self.dtype,
            )
            for path in scenario.hdmap_video_paths
        ]
        # [B=1, V, T, C, H, W]
        hdmap_videos = torch.stack(videos, dim=0).unsqueeze(0)
        if self.is_rank_zero:
            logger.info(
                "Loaded OmniDreams demo HDMaps shape={} views={}",
                tuple(hdmap_videos.shape),
                len(scenario.camera_names),
            )
        return hdmap_videos


def _default_pipeline_factory(pipeline_config: Any, device: str) -> Any:
    return pipeline_config.setup().to(device=device).eval()


def _scenario_from_inputs(inputs: InferenceInput) -> OmnidreamsReplayScenario:
    scenario = inputs.global_conditioning.get("scenario")
    if not isinstance(scenario, OmnidreamsReplayScenario):
        raise TypeError(
            "OmniDreams replay runtime requires global_conditioning['scenario'] "
            "to be an OmnidreamsReplayScenario."
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
    "OmnidreamsReplayRuntime",
    "OmnidreamsReplayRuntimeOptions",
    "OmnidreamsReplaySession",
    "PipelineFactory",
]
