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

import torch
from loguru import logger
from torch import Tensor

from flashdreams.infra.profiler import EventProfiler
from flashdreams.infra.postprocess.base import (
    VideoChunk,
    VideoPostprocessChainConfig,
    VideoPostprocessChainSession,
    VideoSpec,
    VideoTensorLayout,
    VideoValueRange,
    concatenate_video_chunks,
    from_bvtchw,
    infer_video_spec,
    to_bvtchw,
)


@dataclass(kw_only=True)
class VideoPostprocessStreamState:
    """Mutable state owned by one :class:`VideoPostprocessStream`."""

    sessions: dict[int, VideoPostprocessChainSession] = field(default_factory=dict)
    """Post-processing sessions keyed by view index, or ``-1`` for the
    whole output stream."""

    last_raw_output: Tensor | None = None
    """Decoded output from the most recent ``generate`` before post-processing."""

    last_output: Tensor | None = None
    """Tensor returned by the most recent ``generate`` after post-processing."""

    input_spec: VideoSpec | None = None
    """Specification inferred from the first input chunk."""

    num_views: int | None = None
    """Stable view count for per-view streams."""


class VideoPostprocessStream:
    """Process decoded video chunks through one configured stateful chain.

    This object belongs to the runner or serving output layer. It deliberately
    sits outside :class:`StreamInferencePipeline`, whose contract remains
    encode -> diffuse -> decode.
    """

    def __init__(
        self,
        *,
        postprocess: VideoPostprocessChainConfig,
        output_layout: VideoTensorLayout,
        output_value_range: VideoValueRange = "minus_one_one",
        fps: float | None = None,
        per_view: bool = False,
        world_size: int = 1,
        profile: bool = False,
    ) -> None:
        postprocess.validate_execution(world_size=world_size)
        self.postprocess = postprocess
        self.output_layout = output_layout
        self.output_value_range = output_value_range
        self.fps = fps
        self.per_view = per_view
        self.world_size = world_size
        self.profile = profile
        self.state = VideoPostprocessStreamState()
        self._closed = False

    def process(self, output: Tensor, *, autoregressive_index: int) -> Tensor:
        """Process one decoded chunk, possibly returning an empty time axis."""
        if self._closed:
            raise RuntimeError("cannot process video after finish()")
        if not self.profile:
            return apply_video_postprocess(
                postprocess=self.postprocess,
                output_layout=self.output_layout,
                output_value_range=self.output_value_range,
                fps=self.fps,
                per_view=self.per_view,
                state=self.state,
                autoregressive_index=autoregressive_index,
                output=output,
            )

        profiler = self._create_event_profiler()
        result = apply_video_postprocess(
            postprocess=self.postprocess,
            output_layout=self.output_layout,
            output_value_range=self.output_value_range,
            fps=self.fps,
            per_view=self.per_view,
            state=self.state,
            autoregressive_index=autoregressive_index,
            output=output,
        )
        profiler.record("postprocess")
        elapsed_ms = profiler.sync_and_summarize()["postprocess"]
        logger.info(
            f"postprocess AR {autoregressive_index} {elapsed_ms:.3f} ms | "
            f"input {tuple(output.shape)} output {tuple(result.shape)}"
        )
        return result

    def finish(self) -> Tensor | None:
        """Flush buffered output once; repeated calls return ``None``."""
        if self._closed:
            return None
        self._closed = True
        if not self.profile:
            return flush_video_postprocess(
                postprocess=self.postprocess,
                output_layout=self.output_layout,
                output_value_range=self.output_value_range,
                per_view=self.per_view,
                state=self.state,
            )

        profiler = self._create_event_profiler()
        result = flush_video_postprocess(
            postprocess=self.postprocess,
            output_layout=self.output_layout,
            output_value_range=self.output_value_range,
            per_view=self.per_view,
            state=self.state,
        )
        profiler.record("postprocess_flush")
        elapsed_ms = profiler.sync_and_summarize()["postprocess_flush"]
        output_shape = None if result is None else tuple(result.shape)
        logger.info(f"postprocess flush {elapsed_ms:.3f} ms | output {output_shape}")
        return result

    def _create_event_profiler(self) -> EventProfiler:
        return EventProfiler(
            synchronize_distributed=self.postprocess.requires_all_ranks(
                world_size=self.world_size
            )
        )


def apply_video_postprocess(
    *,
    postprocess: VideoPostprocessChainConfig,
    output_layout: VideoTensorLayout,
    output_value_range: VideoValueRange,
    fps: float | None,
    per_view: bool,
    state: VideoPostprocessStreamState,
    autoregressive_index: int,
    output: Tensor,
) -> Tensor:
    """Process one decoded AR chunk and update post-processing state."""
    state.last_raw_output = output
    if not postprocess.is_enabled():
        state.last_output = output
        return output

    layout = output_layout
    _validate_input_spec(state=state, output=output, layout=layout, fps=fps)
    if per_view:
        result = _postprocess_output_per_view(
            postprocess=postprocess,
            output_value_range=output_value_range,
            fps=fps,
            state=state,
            autoregressive_index=autoregressive_index,
            output=output,
            layout=layout,
        )
    else:
        result = _process_postprocess_chunk(
            postprocess=postprocess,
            output_value_range=output_value_range,
            fps=fps,
            state=state,
            autoregressive_index=autoregressive_index,
            session_key=-1,
            output=output,
            layout=layout,
        )

    state.last_output = result
    return result


def flush_video_postprocess(
    *,
    postprocess: VideoPostprocessChainConfig,
    output_layout: VideoTensorLayout,
    output_value_range: VideoValueRange,
    per_view: bool,
    state: VideoPostprocessStreamState,
) -> Tensor | None:
    """Flush buffered post-processing output for the current rollout."""
    if not postprocess.is_enabled() or not state.sessions:
        return None

    layout = output_layout
    if per_view:
        flushed = _flush_postprocess_per_view(
            output_value_range=output_value_range,
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
            output_value_range=output_value_range,
        )

    if flushed is not None:
        state.last_output = flushed
    return flushed


def _postprocess_output_per_view(
    *,
    postprocess: VideoPostprocessChainConfig,
    output_value_range: VideoValueRange,
    fps: float | None,
    state: VideoPostprocessStreamState,
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
            "postprocess stream view count changed from "
            f"{state.num_views} to {views}."
        )
    view_outputs: list[Tensor] = []
    for view_idx in range(canonical.shape[1]):
        view = canonical[:, view_idx : view_idx + 1]
        view_output = _process_postprocess_chunk(
            postprocess=postprocess,
            output_value_range=output_value_range,
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
    output_value_range: VideoValueRange,
    fps: float | None,
    state: VideoPostprocessStreamState,
    autoregressive_index: int,
    session_key: int,
    output: Tensor,
    layout: VideoTensorLayout,
) -> Tensor:
    session = state.sessions.get(session_key)
    if session is None:
        spec = infer_video_spec(output, layout=layout, fps=fps)
        session = postprocess.setup(spec)
        state.sessions[session_key] = session

    chunks = session.process(
        VideoChunk(
            tensor=output,
            layout=layout,
            value_range=output_value_range,
            metadata={"autoregressive_index": autoregressive_index},
        )
    )
    return _postprocess_chunks_to_tensor(
        chunks,
        reference=output,
        layout=layout,
        output_value_range=output_value_range,
    )


def _flush_postprocess_per_view(
    *,
    output_value_range: VideoValueRange,
    state: VideoPostprocessStreamState,
    layout: VideoTensorLayout,
) -> Tensor | None:
    view_outputs: list[Tensor | None] = []
    for view_idx in sorted(k for k in state.sessions if k >= 0):
        output = _postprocess_chunks_to_tensor_or_none(
            state.sessions[view_idx].flush(),
            layout="bvtchw",
            output_value_range=output_value_range,
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
    output_value_range: VideoValueRange,
) -> Tensor:
    if chunks:
        return concatenate_video_chunks(
            chunks,
            layout=layout,
            value_range=output_value_range,
        )
    canonical = to_bvtchw(reference, layout=layout)[:, :, :0]
    return from_bvtchw(canonical, layout=layout)


def _postprocess_chunks_to_tensor_or_none(
    chunks: list[VideoChunk],
    *,
    layout: VideoTensorLayout,
    output_value_range: VideoValueRange,
) -> Tensor | None:
    if not chunks:
        return None
    return concatenate_video_chunks(
        chunks,
        layout=layout,
        value_range=output_value_range,
    )


def _validate_input_spec(
    *,
    state: VideoPostprocessStreamState,
    output: Tensor,
    layout: VideoTensorLayout,
    fps: float | None,
) -> None:
    spec = infer_video_spec(output, layout=layout, fps=fps)
    if state.input_spec is None:
        state.input_spec = spec
        return
    if spec != state.input_spec:
        raise ValueError(
            "postprocess input stream specification changed from "
            f"{state.input_spec!r} to {spec!r}."
        )
