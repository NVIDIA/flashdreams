# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stateful streaming post-processing for generated video outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

import torch
from loguru import logger
from torch import Tensor

from flashdreams.infra.postprocess.base import (
    VideoChunk,
    VideoPostprocessChainConfig,
    VideoPostProcessorSession,
    VideoSpec,
    VideoTensorLayout,
    concatenate_video_chunks,
    from_bvtchw,
    infer_video_spec_from_tensor_shape,
    to_bvtchw,
)
from flashdreams.infra.profiler import EventProfiler

RunnerConfigT = TypeVar("RunnerConfigT")


@dataclass(kw_only=True)
class _VideoPostprocessStreamState:
    """Mutable state owned by one :class:`VideoPostprocessStream`."""

    sessions: dict[int, VideoPostProcessorSession] = field(default_factory=dict)
    """Post-processing sessions keyed by view index, or ``-1`` for the
    whole output stream."""

    input_spec: VideoSpec | None = None
    """Specification inferred from the first input chunk."""

    num_views: int | None = None
    """Stable view count for per-view streams."""


@dataclass(frozen=True, kw_only=True)
class VideoPostprocessStepStats:
    """Profiling data for one AR chunk passed through post-processing."""

    elapsed_ms: float
    input_frames: int
    output_frames: int
    buffering: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        """Return a JSON-serializable representation."""
        return {
            "elapsed_ms": self.elapsed_ms,
            "input_frames": self.input_frames,
            "output_frames": self.output_frames,
            "buffering": self.buffering,
        }


class VideoPostprocessStream:
    """Process and collect decoded video chunks through one configured chain.

    This object belongs to the runner or serving output layer. It deliberately
    sits outside :class:`StreamInferencePipeline`, whose contract remains
    encode -> diffuse -> decode.
    """

    def __init__(
        self,
        *,
        postprocess: VideoPostprocessChainConfig,
        output_layout: VideoTensorLayout,
        fps: float | None = None,
        per_view: bool = False,
        world_size: int = 1,
        profile: bool = False,
        collect_output: bool = True,
        move_to_cpu: bool = True,
        empty_message: str = "post-processing emitted no video frames",
    ) -> None:
        postprocess.validate_execution(world_size=world_size)
        self.postprocess = postprocess
        self.output_layout = output_layout
        self.fps = fps
        self.per_view = per_view
        self.world_size = world_size
        self.profile = profile
        self.collect_output = collect_output
        self.move_to_cpu = move_to_cpu
        self.empty_message = empty_message
        self.state = _VideoPostprocessStreamState()
        self._time_dim = _video_layout_time_dim(output_layout)
        self._chunks: list[Tensor] = []
        self._closed = False
        self._prepared = False
        self.last_process_stats: VideoPostprocessStepStats | None = None

    def process(self, output: Tensor, *, autoregressive_index: int) -> Tensor:
        """Process and collect one decoded chunk."""
        if self._closed:
            raise RuntimeError("cannot process video after finish()")
        self.last_process_stats = None
        self._prepare(output)
        if not self.profile:
            result = apply_video_postprocess(
                postprocess=self.postprocess,
                output_layout=self.output_layout,
                fps=self.fps,
                per_view=self.per_view,
                state=self.state,
                autoregressive_index=autoregressive_index,
                output=output,
            )
            self._append_if_nonempty(result)
            return result

        profiler = self._create_event_profiler()
        result = apply_video_postprocess(
            postprocess=self.postprocess,
            output_layout=self.output_layout,
            fps=self.fps,
            per_view=self.per_view,
            state=self.state,
            autoregressive_index=autoregressive_index,
            output=output,
        )
        profiler.record("postprocess")
        elapsed_ms = profiler.sync_and_summarize()["postprocess"]
        input_frames = int(output.shape[self._time_dim])
        output_frames = int(result.shape[self._time_dim])
        self.last_process_stats = VideoPostprocessStepStats(
            elapsed_ms=elapsed_ms,
            input_frames=input_frames,
            output_frames=output_frames,
            buffering=output_frames == 0,
        )
        logger.info(
            f"postprocess AR {autoregressive_index} {elapsed_ms:.3f} ms | "
            f"input {tuple(output.shape)} output {tuple(result.shape)} | "
            f"buffering {self.last_process_stats.buffering}"
        )
        self._append_if_nonempty(result)
        return result

    def add_process_stats(self, stats: dict[str, float]) -> dict[str, object]:
        """Add the latest postprocess measurement to pipeline AR stats."""
        combined: dict[str, object] = dict(stats)
        if self.last_process_stats is not None:
            combined["postprocess"] = self.last_process_stats.as_dict()
        return combined

    def finish(self) -> Tensor | None:
        """Flush buffered output and return the collected rank-zero video."""
        if self._closed:
            return None
        self._closed = True
        if not self.profile:
            flushed = flush_video_postprocess(
                postprocess=self.postprocess,
                output_layout=self.output_layout,
                per_view=self.per_view,
                state=self.state,
            )
            if flushed is not None:
                self._append_if_nonempty(flushed)
            return self._collected_output()

        profiler = self._create_event_profiler()
        result = flush_video_postprocess(
            postprocess=self.postprocess,
            output_layout=self.output_layout,
            per_view=self.per_view,
            state=self.state,
        )
        profiler.record("postprocess_flush")
        elapsed_ms = profiler.sync_and_summarize()["postprocess_flush"]
        output_shape = None if result is None else tuple(result.shape)
        logger.info(f"postprocess flush {elapsed_ms:.3f} ms | output {output_shape}")
        if result is not None:
            self._append_if_nonempty(result)
        return self._collected_output()

    def _create_event_profiler(self) -> EventProfiler:
        # The stream owns its one explicit readiness barrier in ``_prepare``.
        # Profiling must not inject additional collectives between model calls.
        return EventProfiler(synchronize_distributed=False)

    def _prepare(self, output: Tensor) -> None:
        """Prepare sessions once, then synchronize distributed readiness."""
        if self._prepared or not self.postprocess.is_enabled():
            return
        prepare_video_postprocess(
            postprocess=self.postprocess,
            output_layout=self.output_layout,
            fps=self.fps,
            per_view=self.per_view,
            state=self.state,
            output=output,
        )
        if (
            self.postprocess.requires_all_ranks(world_size=self.world_size)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            # This is deliberately the only postprocess readiness collective.
            # All compilation/model warmup collectives have completed before a
            # rank can enter it, and the process-group timeout bounds failures.
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        self._prepared = True

    def _append_if_nonempty(self, output: Tensor) -> None:
        if not self.collect_output or output.shape[self._time_dim] == 0:
            return
        self._chunks.append(output.cpu() if self.move_to_cpu else output)

    def _collected_output(self) -> Tensor | None:
        if not self.collect_output:
            return None
        if not self._chunks:
            raise ValueError(self.empty_message)
        return torch.cat(self._chunks, dim=self._time_dim)


def create_runner_postprocess_stream(
    config: RunnerConfigT,
    *,
    world_size: int,
    is_rank_zero: bool = True,
    fps: float | None = None,
) -> VideoPostprocessStream | None:
    """Create a runner post-processing stream, or ``None`` when skipped."""
    postprocess = getattr(config, "postprocess")
    if not postprocess.is_enabled():
        return None
    postprocess.validate_execution(world_size=world_size)
    if (
        world_size > 1
        and not is_rank_zero
        and not postprocess.requires_all_ranks(world_size=world_size)
    ):
        return None

    output_layout = getattr(config, "postprocess_output_layout")
    if output_layout is None:
        raise ValueError(
            "RunnerConfig.postprocess is enabled but postprocess_output_layout "
            "is not set."
        )

    configured_fps = fps
    if configured_fps is None:
        configured_fps = getattr(config, "fps", getattr(config, "output_fps", None))

    return VideoPostprocessStream(
        postprocess=postprocess,
        output_layout=output_layout,
        fps=configured_fps,
        per_view=getattr(config, "postprocess_per_view"),
        world_size=world_size,
        collect_output=is_rank_zero,
        profile=bool(
            getattr(getattr(config, "pipeline", None), "enable_sync_and_profile", False)
        ),
    )


def _video_layout_time_dim(layout: VideoTensorLayout) -> int:
    if layout == "tchw":
        return 0
    if layout == "btchw":
        return 1
    if layout in ("bcthw", "bvtchw"):
        return 2
    raise ValueError(f"unsupported video layout: {layout}")


def apply_video_postprocess(
    *,
    postprocess: VideoPostprocessChainConfig,
    output_layout: VideoTensorLayout,
    fps: float | None,
    per_view: bool,
    state: _VideoPostprocessStreamState,
    autoregressive_index: int,
    output: Tensor,
) -> Tensor:
    """Process one decoded AR chunk and update post-processing state."""
    if not postprocess.is_enabled():
        return output

    layout = output_layout
    _validate_input_spec(state=state, output=output, layout=layout, fps=fps)
    if per_view:
        result = _postprocess_output_per_view(
            postprocess=postprocess,
            fps=fps,
            state=state,
            autoregressive_index=autoregressive_index,
            output=output,
            layout=layout,
        )
    else:
        result = _process_postprocess_chunk(
            postprocess=postprocess,
            fps=fps,
            state=state,
            autoregressive_index=autoregressive_index,
            session_key=-1,
            output=output,
            layout=layout,
        )

    return result


def prepare_video_postprocess(
    *,
    postprocess: VideoPostprocessChainConfig,
    output_layout: VideoTensorLayout,
    fps: float | None,
    per_view: bool,
    state: _VideoPostprocessStreamState,
    output: Tensor,
) -> None:
    """Create and prepare sessions before the first measured process call."""
    _validate_input_spec(state=state, output=output, layout=output_layout, fps=fps)
    if per_view:
        if output_layout != "bvtchw":
            raise ValueError(
                "postprocess_per_view requires a layout with an explicit view "
                f"axis; got {output_layout!r}."
            )
        canonical = to_bvtchw(output, layout=output_layout)
        views = canonical.shape[1]
        state.num_views = views
        for view_idx in range(views):
            view = canonical[:, view_idx : view_idx + 1]
            spec = infer_video_spec_from_tensor_shape(view, layout="bvtchw", fps=fps)
            session = postprocess.setup(spec)
            state.sessions[view_idx] = session
            session.prepare()
        return

    spec = infer_video_spec_from_tensor_shape(output, layout=output_layout, fps=fps)
    session = postprocess.setup(spec)
    state.sessions[-1] = session
    session.prepare()


def flush_video_postprocess(
    *,
    postprocess: VideoPostprocessChainConfig,
    output_layout: VideoTensorLayout,
    per_view: bool,
    state: _VideoPostprocessStreamState,
) -> Tensor | None:
    """Flush buffered post-processing output for the current rollout."""
    if not postprocess.is_enabled() or not state.sessions:
        return None

    layout = output_layout
    if per_view:
        flushed = _flush_postprocess_per_view(
            state=state,
            layout=layout,
        )
    else:
        session = state.sessions.get(-1)
        if session is None:
            return None
        flushed = _postprocess_chunks_to_tensor_or_none(
            session.flush(),
            layout=layout,
        )

    return flushed


def _postprocess_output_per_view(
    *,
    postprocess: VideoPostprocessChainConfig,
    fps: float | None,
    state: _VideoPostprocessStreamState,
    autoregressive_index: int,
    output: Tensor,
    layout: VideoTensorLayout,
) -> Tensor:
    if layout != "bvtchw":
        raise ValueError(
            "postprocess_per_view requires a layout with an explicit view "
            f"axis; got {layout!r}."
        )

    canonical = to_bvtchw(output, layout=layout)
    views = canonical.shape[1]
    if state.num_views is None:
        state.num_views = views
    elif state.num_views != views:
        raise ValueError(
            f"postprocess stream view count changed from {state.num_views} to {views}."
        )
    view_outputs: list[Tensor] = []
    for view_idx in range(canonical.shape[1]):
        view = canonical[:, view_idx : view_idx + 1]
        view_output = _process_postprocess_chunk(
            postprocess=postprocess,
            fps=fps,
            state=state,
            autoregressive_index=autoregressive_index,
            session_key=view_idx,
            output=view,
            layout="bvtchw",
        )
        view_outputs.append(to_bvtchw(view_output, layout="bvtchw"))
    output_shapes = {
        (item.shape[0], item.shape[2], item.shape[3], item.shape[4], item.shape[5])
        for item in view_outputs
    }
    if len(output_shapes) != 1:
        raise ValueError(
            "per-view post-processing must emit compatible chunks for every "
            f"view; got shapes {[tuple(item.shape) for item in view_outputs]}."
        )
    return from_bvtchw(torch.cat(view_outputs, dim=1), layout=layout)


def _process_postprocess_chunk(
    *,
    postprocess: VideoPostprocessChainConfig,
    fps: float | None,
    state: _VideoPostprocessStreamState,
    autoregressive_index: int,
    session_key: int,
    output: Tensor,
    layout: VideoTensorLayout,
) -> Tensor:
    session = state.sessions.get(session_key)
    if session is None:
        spec = infer_video_spec_from_tensor_shape(output, layout=layout, fps=fps)
        session = postprocess.setup(spec)
        state.sessions[session_key] = session

    chunks = session.process(
        VideoChunk(
            tensor=output,
            layout=layout,
            metadata={"autoregressive_index": autoregressive_index},
        )
    )
    return _postprocess_chunks_to_tensor(
        chunks,
        reference=output,
        layout=layout,
    )


def _flush_postprocess_per_view(
    *,
    state: _VideoPostprocessStreamState,
    layout: VideoTensorLayout,
) -> Tensor | None:
    view_outputs: list[Tensor | None] = []
    for view_idx in sorted(k for k in state.sessions if k >= 0):
        output = _postprocess_chunks_to_tensor_or_none(
            state.sessions[view_idx].flush(),
            layout="bvtchw",
        )
        view_outputs.append(
            None if output is None else to_bvtchw(output, layout="bvtchw")
        )

    if not view_outputs or all(output is None for output in view_outputs):
        return None
    if any(output is None for output in view_outputs):
        missing = [index for index, output in enumerate(view_outputs) if output is None]
        raise ValueError(
            "per-view post-processing must flush all views or none; "
            f"views without output: {missing}."
        )
    complete_outputs = [output for output in view_outputs if output is not None]
    temporal_sizes = {output.shape[2] for output in complete_outputs}
    if len(temporal_sizes) != 1:
        raise ValueError(
            "per-view post-processing produced different tail lengths: "
            f"{sorted(temporal_sizes)}."
        )
    return from_bvtchw(torch.cat(complete_outputs, dim=1), layout=layout)


def _postprocess_chunks_to_tensor(
    chunks: list[VideoChunk],
    *,
    reference: Tensor,
    layout: VideoTensorLayout,
) -> Tensor:
    if chunks:
        return concatenate_video_chunks(
            chunks,
            layout=layout,
        )
    # A streaming post-processor may consume this AR chunk without emitting
    # frames yet. Keep ``process()`` tensor-only by returning an empty tensor
    # in the caller's layout instead of ``None``; ``_append_if_nonempty`` skips
    # collection later by checking that the time dimension is zero.
    canonical = to_bvtchw(reference, layout=layout)[:, :, :0]
    return from_bvtchw(canonical, layout=layout)


def _postprocess_chunks_to_tensor_or_none(
    chunks: list[VideoChunk],
    *,
    layout: VideoTensorLayout,
) -> Tensor | None:
    if not chunks:
        return None
    return concatenate_video_chunks(
        chunks,
        layout=layout,
    )


def _validate_input_spec(
    *,
    state: _VideoPostprocessStreamState,
    output: Tensor,
    layout: VideoTensorLayout,
    fps: float | None,
) -> None:
    spec = infer_video_spec_from_tensor_shape(output, layout=layout, fps=fps)
    if state.input_spec is None:
        state.input_spec = spec
        return
    if spec != state.input_spec:
        raise ValueError(
            "postprocess input stream specification changed from "
            f"{state.input_spec!r} to {spec!r}."
        )
