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

"""Streaming post-processing helpers for autoregressive pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.infra.postprocess.base import (
    VideoChunk,
    VideoPostprocessChainConfig,
    VideoPostprocessChainSession,
    VideoTensorLayout,
    VideoValueRange,
    concatenate_video_chunks,
    from_bvtchw,
    infer_video_spec,
    to_bvtchw,
)


@dataclass(kw_only=True)
class PipelinePostprocessState:
    """Per-rollout state for streaming pipeline post-processing."""

    sessions: dict[int, VideoPostprocessChainSession] = field(default_factory=dict)
    """Post-processing sessions keyed by view index, or ``-1`` for the
    whole output stream."""

    last_raw_output: Tensor | None = None
    """Decoded output from the most recent ``generate`` before post-processing."""

    last_output: Tensor | None = None
    """Tensor returned by the most recent ``generate`` after post-processing."""


def apply_pipeline_postprocess(
    *,
    postprocess: VideoPostprocessChainConfig,
    output_layout: VideoTensorLayout | None,
    output_value_range: VideoValueRange,
    fps: float | None,
    per_view: bool,
    state: PipelinePostprocessState,
    autoregressive_index: int,
    output: Tensor,
) -> Tensor:
    """Process one decoded AR chunk and update post-processing state."""
    state.last_raw_output = output
    if not postprocess.is_enabled():
        state.last_output = output
        return output

    layout = _validate_pipeline_postprocess(
        postprocess=postprocess,
        output_layout=output_layout,
    )
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


def flush_pipeline_postprocess(
    *,
    postprocess: VideoPostprocessChainConfig,
    output_layout: VideoTensorLayout | None,
    output_value_range: VideoValueRange,
    per_view: bool,
    state: PipelinePostprocessState,
) -> Tensor | None:
    """Flush buffered post-processing output for the current rollout."""
    if not postprocess.is_enabled() or not state.sessions:
        return None

    layout = _validate_pipeline_postprocess(
        postprocess=postprocess,
        output_layout=output_layout,
    )
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
    state: PipelinePostprocessState,
    autoregressive_index: int,
    output: Tensor,
    layout: VideoTensorLayout,
) -> Tensor:
    if layout not in ("bvtchw", "bvcthw"):
        raise ValueError(
            "postprocess_per_view requires a layout with an explicit view "
            f"axis; got {layout!r}."
        )

    canonical = to_bvtchw(output, layout=layout)
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
    return from_bvtchw(torch.cat(view_outputs, dim=1), layout=layout)


def _process_postprocess_chunk(
    *,
    postprocess: VideoPostprocessChainConfig,
    output_value_range: VideoValueRange,
    fps: float | None,
    state: PipelinePostprocessState,
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
    state: PipelinePostprocessState,
    layout: VideoTensorLayout,
) -> Tensor | None:
    view_outputs: list[Tensor] = []
    for view_idx in sorted(k for k in state.sessions if k >= 0):
        output = _postprocess_chunks_to_tensor_or_none(
            state.sessions[view_idx].flush(),
            layout="bvtchw",
            output_value_range=output_value_range,
        )
        if output is not None:
            view_outputs.append(to_bvtchw(output, layout="bvtchw"))

    if not view_outputs:
        return None
    return from_bvtchw(torch.cat(view_outputs, dim=1), layout=layout)


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


def _validate_pipeline_postprocess(
    *,
    postprocess: VideoPostprocessChainConfig,
    output_layout: VideoTensorLayout | None,
) -> VideoTensorLayout:
    if output_layout is None:
        raise ValueError(
            "postprocess_output_layout must be set when postprocess is enabled."
        )
    if (
        torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and not postprocess.uses_context_parallelism()
    ):
        for processor in postprocess.resolved_processors():
            if getattr(processor, "attention_mode", None) == "sparse":
                raise ValueError(
                    "FlashVSR sparse post-processing does not support "
                    "multi-GPU execution. Use the flashvsr-v1.1-full-attn "
                    "preset for context parallelism, or run without torchrun."
                )
    return output_layout
