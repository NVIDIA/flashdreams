# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared synchronous Lingbot model-session state and execution."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import torch

from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime import StepResult, TimeWindow

OutputStreamFactory = Callable[[], VideoOutputStream]


class LingbotModelSessionCore:
    """Own one Lingbot cache, AR index, and generated-output stream."""

    def __init__(
        self,
        *,
        pipeline: Any,
        output_stream_factory: OutputStreamFactory,
    ) -> None:
        self.pipeline = pipeline
        self._output_stream_factory = output_stream_factory
        self._output_stream = output_stream_factory()
        self._cache: Any | None = None
        self._step_index = 0
        self._closed = False

    @property
    def cache(self) -> Any:
        if self._cache is None:
            raise RuntimeError("Lingbot model session is not initialized.")
        return self._cache

    @property
    def step_index(self) -> int:
        return self._step_index

    def next_num_frames(self) -> int:
        self._require_open()
        return int(self.pipeline.get_num_output_frames(self._step_index))

    def reset(self, *, prompt: str, first_frames: torch.Tensor) -> None:
        self._require_open()
        self._cache = None
        self._output_stream.finish()
        self._output_stream = self._output_stream_factory()
        self._cache = self.pipeline.initialize_cache(
            text=[prompt],
            image=first_frames,
        )
        self._step_index = 0

    def step(
        self,
        camctrl_input: Any,
        *,
        output_window: TimeWindow | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StepResult:
        self._require_open()
        step_index = self._step_index
        expected_frames = self.next_num_frames()
        start_t = time.perf_counter()
        video_chunk = self.pipeline.generate(
            autoregressive_index=step_index,
            cache=self.cache,
            input=camctrl_input,
        )
        stats = self.pipeline.finalize(
            autoregressive_index=step_index,
            cache=self.cache,
        )
        metrics = _numeric_metrics(stats)
        metrics.setdefault("model_step_s", time.perf_counter() - start_t)
        result = self._output_stream.process(
            video_chunk,
            autoregressive_index=step_index,
            metrics=metrics,
            metadata=metadata,
            output_window=output_window,
        )
        if result.frame_count != expected_frames:
            raise RuntimeError(
                f"Expected generated chunk to contain {expected_frames} frames, "
                f"got {result.frame_count}."
            )
        self._step_index += 1
        return result

    def replace_text_embeddings(self, text_embeddings: torch.Tensor) -> None:
        self._require_open()
        transformer = self.pipeline.diffusion_model.transformer
        replace = getattr(transformer, "replace_text_embeddings", None)
        if not callable(replace):
            raise RuntimeError(
                "Current Lingbot pipeline does not support text-context swapping."
            )
        replace(self.cache.transformer_cache, text_embeddings)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cache = None
        self._output_stream.finish()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Lingbot model session is closed.")


def _numeric_metrics(stats: object) -> dict[str, float | int]:
    if not isinstance(stats, Mapping):
        return {}
    return {
        str(name): value
        for name, value in stats.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


__all__ = ["LingbotModelSessionCore"]
