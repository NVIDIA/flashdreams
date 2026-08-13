# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-owned runtime and session for streaming pipeline applications."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from flashdreams.runtime import (
    InferenceInput,
    InferenceSession,
    StepRequest,
    StepResult,
)

from .contracts import AppConfig, PipelineAppSpec


class PipelineAppRuntime:
    """Instantiate and own the reusable pipeline behind an application spec."""

    def __init__(
        self,
        *,
        spec: PipelineAppSpec,
        config: AppConfig,
        device: str,
    ) -> None:
        self.pipeline = spec.pipeline_config.setup().to(device).eval()
        self._config = config
        self._spec = spec
        self._closed = False

    def start_session(self, inputs: InferenceInput) -> "PipelineAppSession":
        """Create an open-ended session with isolated pipeline cache state."""
        if self._closed:
            raise RuntimeError("Pipeline application runtime is closed.")
        return PipelineAppSession(
            pipeline=self.pipeline,
            inputs=inputs,
            spec=self._spec,
            config=self._config,
        )

    def peek_input_fps(self) -> float:
        """Return the host clock rate used for realtime presentation."""
        return float(self._config.fps)

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
    """Host-owned autoregressive session for a pipeline application."""

    def __init__(
        self,
        *,
        pipeline: Any,
        inputs: InferenceInput,
        spec: PipelineAppSpec,
        config: AppConfig,
    ) -> None:
        self._pipeline = pipeline
        self._cache: object | None = spec.initialize_cache(pipeline, inputs)
        self._config = config
        self._step_index = 0
        self._closed = False

    def next_step_request(self) -> StepRequest | None:
        """Return the next rollout request, or ``None`` after session closure."""
        if self._closed:
            return None
        return StepRequest(step_index=self._step_index)

    def step(self, inputs: InferenceInput) -> StepResult:
        """Generate and finalize one autoregressive video chunk."""
        del inputs
        if self._closed:
            raise RuntimeError("Pipeline application session is closed.")
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
            layout=self._config.output_layout,
            metrics=metrics,
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Reject reset because sessions use isolated cache state."""
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
