# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams runtime/session contracts for shared demo run modes."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger
from omnidreams.model_session import OmnidreamsModelSessionCore

from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner_io import (
    DEFAULT_RUNNER_INSTALL_HINT,
    load_first_frame_tensor,
)
from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import InferenceInput
from flashdreams.runtime.interfaces import InferenceSession
from flashdreams.runtime.types import StepRequest, StepRequirements, StepResult

from .spec import OmnidreamsLudusReplayScenario, OmnidreamsReplayScenario

OmnidreamsSessionScenario = OmnidreamsReplayScenario | OmnidreamsLudusReplayScenario

PipelineFactory = Callable[[Any, str], Any]


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsRuntimeOptions:
    """Construction knobs for the OmniDreams runtime."""

    pipeline_config: Any
    pipeline_factory: PipelineFactory | None = None
    output_layout: VideoTensorLayout = "bvtchw"
    release_oneshot_encoders_after_cache_init: bool = True


class OmnidreamsRuntime:
    """Heavyweight OmniDreams runtime consumed by shared demo run modes."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        options: OmnidreamsRuntimeOptions,
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
        return OmnidreamsSession(
            pipeline=self.pipeline,
            scenario=scenario,
            initial_inputs=inputs,
            device=torch.device(f"cuda:{self.local_rank}")
            if dist.is_initialized()
            else torch.device(self.config.device or "cuda"),
            is_rank_zero=self.is_rank_zero,
            output_layout=self.options.output_layout,
            rollout_seed=self.config.seed,
            release_oneshot_encoders_after_cache_init=(
                self.options.release_oneshot_encoders_after_cache_init
            ),
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


class OmnidreamsSession:
    """One OmniDreams rollout over a prepared scenario."""

    def __init__(
        self,
        *,
        pipeline: Any,
        scenario: OmnidreamsSessionScenario,
        initial_inputs: InferenceInput,
        device: torch.device,
        is_rank_zero: bool,
        output_layout: VideoTensorLayout,
        rollout_seed: int | None,
        release_oneshot_encoders_after_cache_init: bool,
    ) -> None:
        self.pipeline = pipeline
        self.scenario = scenario
        self._initial_inputs = initial_inputs
        self.device = device
        self.is_rank_zero = is_rank_zero
        self.output_layout = output_layout
        self.rollout_seed = rollout_seed
        self.release_oneshot_encoders_after_cache_init = (
            release_oneshot_encoders_after_cache_init
        )
        self.dtype = torch.bfloat16
        self._closed = False
        self._model_session = OmnidreamsModelSessionCore(
            pipeline=pipeline,
            output_stream_factory=lambda: VideoOutputStream(
                postprocess_stream=None,
                output_layout=self.output_layout,
            ),
        )
        self._model_session.reset(self._initialize_cache)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
        if dist.is_initialized():
            dist.barrier()

    def next_step_requirements(self) -> StepRequirements | None:
        if self._closed:
            return None
        step_index = self._model_session.step_index
        if step_index >= self.scenario.total_blocks:
            return None
        num_frames = self._model_session.next_num_frames()
        return StepRequirements(
            step_index=step_index,
            input_frame_count=num_frames,
        )

    def next_step_request(self) -> StepRequest | None:
        requirements = self.next_step_requirements()
        if requirements is None:
            return None
        metadata = dict(requirements.metadata)
        metadata["input_frame_count"] = requirements.input_frame_count
        if requirements.steady_output_frame_count is not None:
            metadata["steady_output_frame_count"] = (
                requirements.steady_output_frame_count
            )
        return StepRequest(
            step_index=requirements.step_index,
            inference_input_schema=requirements.inference_input_schema,
            metadata=metadata,
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        if self._closed:
            raise RuntimeError("OmniDreams replay session is closed.")

        step_index = self._model_session.step_index
        num_frames = self._model_session.next_num_frames()
        hdmap = _hdmap_from_inputs(inputs)
        if hdmap.shape[2] != num_frames:
            raise ValueError(
                "OmniDreams step HDMap frame count mismatch: "
                f"expected {num_frames}, got {hdmap.shape[2]}."
            )
        logger.info(
            "OmniDreams demo replay step {} frames={}",
            step_index,
            num_frames,
        )
        return self._model_session.step(hdmap)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if inputs is not None:
            scenario = _scenario_from_inputs(inputs)
            if scenario != self.scenario:
                raise ValueError("OmniDreams replay reset cannot swap scenarios.")
            self._initial_inputs = inputs
        self._model_session.reset(self._initialize_cache)

    def close(self) -> None:
        self._closed = True
        self._model_session.close()

    def _initialize_cache(self) -> Any:
        scenario = self.scenario
        _seed_pipeline_for_rollout(self.pipeline, self.rollout_seed)
        cache = self.pipeline.initialize_cache(
            text=_prompt_from_inputs(self._initial_inputs, scenario),
            image=_first_frame_from_inputs(
                self._initial_inputs,
                scenario=scenario,
                device=self.device,
                dtype=self.dtype,
            ),
            view_names=_view_names_from_inputs(self._initial_inputs, scenario),
        )
        if self.release_oneshot_encoders_after_cache_init:
            release = getattr(self.pipeline, "release_oneshot_encoders", None)
            if callable(release):
                release()
        return cache


def _default_pipeline_factory(pipeline_config: Any, device: str) -> Any:
    return pipeline_config.setup().to(device=device).eval()


def _scenario_from_inputs(inputs: InferenceInput) -> OmnidreamsSessionScenario:
    scenario = inputs.global_conditioning.get("scenario")
    if not isinstance(
        scenario,
        (OmnidreamsReplayScenario, OmnidreamsLudusReplayScenario),
    ):
        raise TypeError(
            "OmniDreams replay runtime requires global_conditioning['scenario'] "
            "to be an OmnidreamsReplayScenario or OmnidreamsLudusReplayScenario."
        )
    return scenario


def _prompt_from_inputs(
    inputs: InferenceInput,
    scenario: OmnidreamsSessionScenario,
) -> list[list[str]]:
    prompt = inputs.global_conditioning.get("prompt")
    if prompt is None:
        if not scenario.prompts:
            raise ValueError(
                "OmniDreams initial prompt is required when the scenario does "
                "not carry fallback prompts."
            )
        return [list(scenario.prompts)]
    if isinstance(prompt, str):
        return [[prompt]]
    if isinstance(prompt, Sequence):
        values = list(prompt)
        if all(isinstance(value, str) for value in values):
            return [[str(value) for value in values]]
        batches: list[list[str]] = []
        for batch in values:
            if not isinstance(batch, Sequence) or isinstance(batch, str):
                raise TypeError(
                    "OmniDreams initial prompt batches must be string sequences."
                )
            batches.append([str(item) for item in batch])
        return batches
    raise TypeError(
        "OmniDreams initial prompt must be a string or sequence of strings."
    )


def _first_frame_from_inputs(
    inputs: InferenceInput,
    *,
    scenario: OmnidreamsSessionScenario,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    first_frame = inputs.global_conditioning.get("first_frame")
    if isinstance(first_frame, torch.Tensor):
        return first_frame
    if first_frame is not None:
        raise TypeError("OmniDreams initial first_frame must be a torch.Tensor.")
    first_frame_paths = getattr(scenario, "first_frame_paths", ())
    if not first_frame_paths:
        raise ValueError(
            "OmniDreams initial first_frame tensor is required when the "
            "scenario does not carry fallback first_frame_paths."
        )
    first_frames = [
        load_first_frame_tensor(
            path,
            pixel_height=scenario.pixel_height,
            pixel_width=scenario.pixel_width,
            device=device,
            dtype=dtype,
            allow_video=True,
            install_hint=DEFAULT_RUNNER_INSTALL_HINT,
        )
        for path in first_frame_paths
    ]
    return torch.stack(first_frames, dim=0).unsqueeze(0)


def _seed_pipeline_for_rollout(pipeline: Any, seed: int | None) -> None:
    if seed is None:
        return
    diffusion_model = getattr(pipeline, "diffusion_model", None)
    rng = getattr(diffusion_model, "rng", None)
    if rng is None:
        return
    rng.manual_seed(int(seed))


def _view_names_from_inputs(
    inputs: InferenceInput,
    scenario: OmnidreamsSessionScenario,
) -> list[str]:
    value = inputs.metadata.get("view_names") or inputs.global_conditioning.get(
        "view_names"
    )
    if value is None:
        return list(scenario.camera_names)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise TypeError("OmniDreams view_names metadata must be a string sequence.")


def _hdmap_from_inputs(inputs: InferenceInput) -> torch.Tensor:
    hdmap = inputs.step.get("hdmap")
    if not isinstance(hdmap, torch.Tensor):
        raise TypeError("OmniDreams session step requires step['hdmap'] tensor.")
    if hdmap.ndim != 6:
        raise ValueError(
            "OmniDreams step['hdmap'] must have shape [B, V, T, C, H, W], "
            f"got {tuple(hdmap.shape)}."
        )
    return hdmap


def _is_torchrun_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


__all__ = [
    "OmnidreamsRuntime",
    "OmnidreamsRuntimeOptions",
    "OmnidreamsSession",
    "OmnidreamsSessionScenario",
    "PipelineFactory",
]
