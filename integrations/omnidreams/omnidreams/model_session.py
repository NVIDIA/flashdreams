# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared synchronous OmniDreams pipeline-session execution."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import torch

from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime import StepResult

CacheFactory = Callable[[], Any]
OutputStreamFactory = Callable[[], VideoOutputStream]


class OmnidreamsModelSessionCore:
    """Own one raw-pipeline cache, AR index, finalization, and output stream."""

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
        self._pending_finalization_index: int | None = None
        self._closed = False

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def initialized(self) -> bool:
        return self._cache is not None and not self._closed

    def next_num_frames(self) -> int:
        self._require_open()
        return int(self.pipeline.get_num_frames(self._step_index))

    def reset(self, cache_factory: CacheFactory) -> None:
        self._require_open()
        if self._cache is not None or self._step_index != 0:
            self._clear(finalize_pending=False, recreate_output_stream=True)
        self._cache = cache_factory()

    def step(
        self,
        hdmap: torch.Tensor,
        *,
        delay_finalization: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> StepResult:
        self._require_initialized()
        self._finalize_pending()
        step_index = self._step_index
        expected_frames = self.next_num_frames()
        start_t = time.perf_counter()
        video_chunk = self.pipeline.generate(
            autoregressive_index=step_index,
            cache=self._cache,
            hdmap=hdmap,
        )
        metrics: dict[str, float | int] = {}
        if delay_finalization:
            self._pending_finalization_index = step_index
        else:
            metrics = _numeric_metrics(
                self.pipeline.finalize(
                    autoregressive_index=step_index,
                    cache=self._cache,
                )
            )
        metrics.setdefault("model_step_s", time.perf_counter() - start_t)
        result = self._output_stream.process(
            video_chunk,
            autoregressive_index=step_index,
            metrics=metrics,
            metadata=metadata,
        )
        if result.frame_count != expected_frames:
            raise RuntimeError(
                f"Expected generated chunk to contain {expected_frames} frames, "
                f"got {result.frame_count}."
            )
        self._step_index += 1
        return result

    def replace_output_stream(self, output_stream_factory: OutputStreamFactory) -> None:
        self._require_open()
        self._output_stream.finish()
        self._output_stream_factory = output_stream_factory
        self._output_stream = output_stream_factory()

    def finish_output(self) -> StepResult | None:
        """Flush and return the output postprocessor tail, when present."""
        self._require_open()
        return self._output_stream.finish()

    def clear(self, *, finalize_pending: bool = False) -> None:
        self._require_open()
        self._clear(
            finalize_pending=finalize_pending,
            recreate_output_stream=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._clear(finalize_pending=True, recreate_output_stream=False)
        self._closed = True

    def _clear(
        self,
        *,
        finalize_pending: bool,
        recreate_output_stream: bool,
    ) -> None:
        if finalize_pending:
            self._finalize_pending()
        else:
            self._pending_finalization_index = None
        self._cache = None
        self._step_index = 0
        self._output_stream.finish()
        if recreate_output_stream:
            self._output_stream = self._output_stream_factory()

    def _finalize_pending(self) -> None:
        if self._cache is None or self._pending_finalization_index is None:
            return
        self.pipeline.finalize(
            autoregressive_index=self._pending_finalization_index,
            cache=self._cache,
        )
        self._pending_finalization_index = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OmniDreams model session is closed.")

    def _require_initialized(self) -> None:
        self._require_open()
        if self._cache is None:
            raise RuntimeError("OmniDreams model session is not initialized.")


def _numeric_metrics(stats: object) -> dict[str, float | int]:
    if not isinstance(stats, Mapping):
        return {}
    return {
        str(name): value
        for name, value in stats.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


__all__ = ["OmnidreamsModelSessionCore"]
