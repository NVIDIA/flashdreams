# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-owned runtime and session for streaming pipeline applications."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from flashdreams.infra.config import derive_config
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.runtime import (
    InferenceInput,
    InferenceSession,
    StepRequest,
    StepRequirements,
    StepResult,
)

from .contracts import PipelineAppSpec, RuntimeMetadata


class PipelineAppRuntime:
    """Instantiate and own the reusable pipeline behind an application spec."""

    def __init__(
        self,
        *,
        spec: PipelineAppSpec,
        device: str,
        compile: bool | None = None,
        cuda_graph: bool | None = None,
    ) -> None:
        pipeline_config = _with_execution_options(
            spec.pipeline_config,
            compile=compile,
            cuda_graph=cuda_graph,
        )
        self.pipeline = pipeline_config.setup().to(device).eval()
        self.metadata = spec.metadata
        self.initial_input = spec.initial_input
        self._spec = spec
        self._closed = False

    def prepare_step_input(
        self, request: StepRequest | StepRequirements
    ) -> InferenceInput:
        """Return an empty step payload for prompt-conditioned pipelines."""
        del request
        return InferenceInput()

    def start_session(self, inputs: InferenceInput) -> "PipelineAppSession":
        """Create a finite session with cache state isolated from other sessions."""
        if self._closed:
            raise RuntimeError("Pipeline application runtime is closed.")
        return PipelineAppSession(
            pipeline=self.pipeline,
            inputs=inputs,
            spec=self._spec,
        )

    def peek_input_fps(self) -> float:
        """Return the host clock rate used for realtime presentation."""
        return float(self.metadata.fps)

    def peek_steady_output_num_frames(self) -> int:
        """Return the steady-state output chunk size for presentation queues."""
        return int(self.pipeline.get_num_output_frames(1))

    def close(self) -> None:
        """Release the shared pipeline and accelerator allocator state."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self.pipeline, "close", None)
        if callable(close):
            close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class PipelineAppSession(InferenceSession):
    """Host-owned finite autoregressive session for a pipeline app spec."""

    def __init__(
        self,
        *,
        pipeline: Any,
        inputs: InferenceInput,
        spec: PipelineAppSpec,
    ) -> None:
        self._pipeline = pipeline
        self._cache: object | None = spec.contract.initialize_cache(pipeline, inputs)
        self._metadata = spec.metadata
        self._result_metadata = spec.result_metadata
        self._total_steps = spec.total_steps
        self._step_index = 0
        self._closed = False

    def next_step_request(self) -> StepRequest | None:
        """Return the next finite rollout request, or ``None`` when complete."""
        if self._closed or self._step_index >= self._total_steps:
            return None
        return StepRequest(step_index=self._step_index)

    def step(self, inputs: InferenceInput) -> StepResult:
        """Generate and finalize one autoregressive video chunk."""
        del inputs
        if self._closed:
            raise RuntimeError("Pipeline application session is closed.")
        if self._step_index >= self._total_steps:
            raise RuntimeError("Pipeline application session is complete.")
        if self._cache is None:
            raise RuntimeError("Pipeline application session has no active cache.")

        index = self._step_index
        video = self._pipeline.generate(
            autoregressive_index=index,
            cache=self._cache,
        )
        metrics = _metrics(
            self._pipeline.finalize(
                autoregressive_index=index,
                cache=self._cache,
            )
        )
        self._step_index += 1
        if not isinstance(video, torch.Tensor):
            raise TypeError(
                "Pipeline generate() must return a torch.Tensor, got "
                f"{type(video).__name__}."
            )
        return StepResult.from_video_chunk(
            step_index=index,
            video_chunk=video.detach(),
            layout=self._metadata.output_layout,
            metrics=metrics,
            metadata=self._result_metadata,
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Reject reset because finite sessions use isolated cache state."""
        del inputs
        raise RuntimeError("Create a new session instead of resetting this one.")

    def close(self) -> None:
        """Release session-local cache state."""
        if self._closed:
            return
        self._closed = True
        cache = self._cache
        self._cache = None
        close = getattr(cache, "close", None)
        if callable(close):
            close()


def _with_execution_options(
    pipeline: StreamInferencePipelineConfig,
    *,
    compile: bool | None,
    cuda_graph: bool | None,
) -> StreamInferencePipelineConfig:
    transformer: dict[str, object] = {}
    if compile is not None:
        transformer["compile_network"] = compile
    if cuda_graph is not None:
        transformer["use_cuda_graph"] = cuda_graph
    if not transformer:
        return pipeline
    return derive_config(
        pipeline,
        diffusion_model={"transformer": transformer},
    )


def _metrics(value: object) -> Mapping[str, float | int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "Pipeline finalize() must return a metrics mapping or None, got "
            f"{type(value).__name__}."
        )
    metrics = dict(value)
    invalid = tuple(
        key for key, metric in metrics.items() if not isinstance(metric, (int, float))
    )
    if invalid:
        raise TypeError(f"Pipeline metrics must be numeric; invalid keys: {invalid}.")
    return metrics


__all__ = ["PipelineAppRuntime", "PipelineAppSession"]
