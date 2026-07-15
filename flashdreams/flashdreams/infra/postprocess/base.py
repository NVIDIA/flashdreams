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

"""Video post-processing contracts for runner and serving outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import torch
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig, PrintableConfig

VideoTensorLayout = Literal[
    "tchw",
    "btchw",
    "bcthw",
    "bvtchw",
]
"""Supported RGB video tensor layouts at the post-processing boundary."""


@dataclass(frozen=True, kw_only=True)
class VideoSpec:
    """Static stream metadata passed to post-processors before first use."""

    height: int
    """Input stream frame height in pixels."""

    width: int
    """Input stream frame width in pixels."""

    fps: float | None = None
    """Frame rate for outputs that need to preserve timing metadata."""

    channels: int = 3
    """Number of color channels. FlashDreams video post-processing expects RGB."""


@dataclass(kw_only=True)
class VideoChunk:
    """One video segment exchanged between generators, post-processors, and sinks."""

    tensor: Tensor
    """RGB frames in ``[-1, 1]`` using the layout described by :attr:`layout`."""

    layout: VideoTensorLayout = "tchw"
    """Layout of :attr:`tensor`."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional per-chunk metadata carried through the post-processing chain."""


@dataclass(kw_only=True)
class VideoPostProcessorConfig(InstantiateConfig):
    """Base config for a video post-processor implementation."""

    _target: type["VideoPostProcessor"] = field(
        default_factory=lambda: VideoPostProcessor
    )

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        """Return the stream specification produced from ``input_spec``.

        Processors that resize video, change channels, or retime output must
        override this method so downstream sessions receive accurate metadata.
        """
        return input_spec

    def requires_all_ranks(self, *, world_size: int) -> bool:
        """Return whether this processor must execute on every rank."""
        del world_size
        return False

    def validate_execution(self, *, world_size: int) -> None:
        """Validate this processor for the requested distributed world."""
        del world_size


VideoPostProcessorConfigT = TypeVar(
    "VideoPostProcessorConfigT", bound=VideoPostProcessorConfig
)


class VideoPostProcessorSession(ABC):
    """Stateful per-stream post-processing session."""

    def prepare(self) -> None:
        """Prepare expensive runtime state before the first timed chunk.

        Most processors need no explicit preparation. Compiled distributed
        processors can override this hook to compile representative shapes on
        every rank before the stream enters its measured execution phase.
        Implementations must leave the session equivalent to a newly started
        stream when this method returns.
        """

    @abstractmethod
    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Process one input chunk and return zero or more output chunks.

        Returning an empty list means the session synchronously consumed the
        chunk but is buffering frames until a later chunk or ``flush()`` can
        complete its own output window.
        """

    @abstractmethod
    def flush(self) -> list[VideoChunk]:
        """Return any buffered output at end-of-stream."""


class VideoPostProcessor(ABC, Generic[VideoPostProcessorConfigT]):
    """Factory for stateful video post-processing sessions."""

    config: VideoPostProcessorConfigT

    def __init__(self, config: VideoPostProcessorConfigT) -> None:
        self.config = config

    @abstractmethod
    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        """Start processing one video stream."""


@dataclass(kw_only=True)
class VideoPostprocessChainConfig(PrintableConfig):
    """Ordered post-processing chain for generated video outputs."""

    processors: tuple[VideoPostProcessorConfig, ...] = ()
    """Post-processors to apply in order. Empty means no post-processing."""

    preset: str = ""
    """Optional registered post-processor preset appended after
    :attr:`processors`. Presets are discovered from the
    ``flashdreams.postprocess_presets`` entry-point group (for example
    ``flashvsr-v1.1-sparse-2.0`` when the FlashVSR integration is
    installed). Empty means no preset is appended."""

    def resolved_processors(self) -> tuple[VideoPostProcessorConfig, ...]:
        """Return :attr:`processors` plus any preset selected by name."""
        if not self.preset:
            return self.processors
        from flashdreams.plugins.registry import resolve_postprocess_preset

        return (*self.processors, resolve_postprocess_preset(self.preset))

    def is_enabled(self) -> bool:
        """Return whether any post-processing step is configured."""
        return bool(self.processors or self.preset)

    def requires_all_ranks(self, *, world_size: int) -> bool:
        """Return whether any processor must execute on every rank."""
        return any(
            processor.requires_all_ranks(world_size=world_size)
            for processor in self.resolved_processors()
        )

    def validate_execution(self, *, world_size: int) -> None:
        """Validate every processor for the requested distributed world."""
        for processor in self.resolved_processors():
            processor.validate_execution(world_size=world_size)

    def setup(self, spec: VideoSpec) -> VideoPostProcessorSession:
        """Instantiate post-processors and start per-stream sessions."""
        sessions: list[VideoPostProcessorSession] = []
        current_spec = spec
        for processor_config in self.resolved_processors():
            sessions.append(processor_config.setup().start(current_spec))
            current_spec = processor_config.output_spec(current_spec)
        return _VideoPostprocessChainSession(sessions=sessions)


class _VideoPostprocessChainSession(VideoPostProcessorSession):
    """Sequential execution of a configured post-processing chain."""

    def __init__(self, sessions: list[VideoPostProcessorSession]) -> None:
        self._sessions = sessions
        self._closed = False

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Process one chunk through every configured session."""
        if self._closed:
            raise RuntimeError("cannot process a post-processing chain after flush()")
        return self._process_chunks_through_sessions(
            first_session_index=0,
            chunks=[chunk],
        )

    def prepare(self) -> None:
        """Prepare every processor session in chain order."""
        for session in self._sessions:
            session.prepare()

    def flush(self) -> list[VideoChunk]:
        """Flush each session once and feed its tail output downstream.

        Repeated calls are idempotent and return no additional output.
        """
        if self._closed:
            return []
        # A failed flush is terminal: retrying could emit duplicate tails from
        # sessions that already completed before a later session raised.
        self._closed = True
        outputs: list[VideoChunk] = []
        for index, session in enumerate(self._sessions):
            tail_chunks = session.flush()
            if tail_chunks:
                # Tail output from this session has already passed earlier
                # sessions, so continue only through downstream sessions.
                outputs.extend(
                    self._process_chunks_through_sessions(
                        first_session_index=index + 1,
                        chunks=tail_chunks,
                    )
                )
        return outputs

    def _process_chunks_through_sessions(
        self, *, first_session_index: int, chunks: Iterable[VideoChunk]
    ) -> list[VideoChunk]:
        """Run chunks through the chain starting at ``first_session_index``."""
        pending_chunks = list(chunks)
        for session in self._sessions[first_session_index:]:
            emitted_chunks: list[VideoChunk] = []
            for chunk in pending_chunks:
                emitted_chunks.extend(session.process(chunk))
            pending_chunks = emitted_chunks
        return pending_chunks


def infer_video_spec_from_tensor_shape(
    tensor: Tensor,
    *,
    layout: VideoTensorLayout,
    fps: float | None = None,
) -> VideoSpec:
    """Infer :class:`VideoSpec` from a tensor shape and declared layout."""
    # Shape inference is metadata-only. Keep it explicit here so this does not
    # look like a tensor conversion or video-frame copy.
    if layout == "tchw":
        _assert_ndim(tensor, 4, layout)
        channels, height, width = tensor.shape[1], tensor.shape[2], tensor.shape[3]
    elif layout == "btchw":
        _assert_ndim(tensor, 5, layout)
        channels, height, width = tensor.shape[2], tensor.shape[3], tensor.shape[4]
    elif layout == "bcthw":
        _assert_ndim(tensor, 5, layout)
        channels, height, width = tensor.shape[1], tensor.shape[3], tensor.shape[4]
    elif layout == "bvtchw":
        _assert_ndim(tensor, 6, layout)
        channels, height, width = tensor.shape[3], tensor.shape[4], tensor.shape[5]
    else:
        raise ValueError(f"unsupported video layout: {layout}")
    return VideoSpec(height=height, width=width, fps=fps, channels=channels)


def concatenate_video_chunks(
    chunks: Iterable[VideoChunk],
    *,
    layout: VideoTensorLayout,
) -> Tensor:
    """Concatenate ``[-1, 1]`` chunks along time in the requested layout."""
    canonical_chunks = [
        to_bvtchw(chunk.tensor, layout=chunk.layout) for chunk in chunks
    ]
    if not canonical_chunks:
        raise ValueError("cannot concatenate an empty video chunk sequence")
    canonical = torch.cat(canonical_chunks, dim=2)
    return from_bvtchw(canonical, layout=layout)


def to_bvtchw(tensor: Tensor, *, layout: VideoTensorLayout) -> Tensor:
    """Convert a supported RGB video layout to canonical ``[B, V, T, C, H, W]``."""
    if layout == "tchw":
        _assert_ndim(tensor, 4, layout)
        return tensor.unsqueeze(0).unsqueeze(0)
    if layout == "btchw":
        _assert_ndim(tensor, 5, layout)
        return tensor.unsqueeze(1)
    if layout == "bcthw":
        _assert_ndim(tensor, 5, layout)
        return tensor.permute(0, 2, 1, 3, 4).unsqueeze(1)
    if layout == "bvtchw":
        _assert_ndim(tensor, 6, layout)
        return tensor
    raise ValueError(f"unsupported video layout: {layout}")


def from_bvtchw(tensor: Tensor, *, layout: VideoTensorLayout) -> Tensor:
    """Convert canonical ``[B, V, T, C, H, W]`` to a supported RGB layout."""
    _assert_ndim(tensor, 6, "bvtchw")
    if layout == "tchw":
        _assert_singleton_batch_and_view(tensor, layout)
        return tensor[0, 0]
    if layout == "btchw":
        _assert_singleton_view(tensor, layout)
        return tensor[:, 0]
    if layout == "bcthw":
        _assert_singleton_view(tensor, layout)
        return tensor[:, 0].permute(0, 2, 1, 3, 4)
    if layout == "bvtchw":
        return tensor
    raise ValueError(f"unsupported video layout: {layout}")


def _assert_ndim(tensor: Tensor, expected: int, layout: str) -> None:
    if tensor.ndim != expected:
        raise ValueError(
            f"layout {layout!r} expects {expected} dimensions; got {tensor.ndim}"
        )


def _assert_singleton_batch_and_view(tensor: Tensor, layout: str) -> None:
    if tensor.shape[0] != 1 or tensor.shape[1] != 1:
        raise ValueError(
            f"layout {layout!r} requires batch=1 and views=1; got "
            f"batch={tensor.shape[0]}, views={tensor.shape[1]}"
        )


def _assert_singleton_view(tensor: Tensor, layout: str) -> None:
    if tensor.shape[1] != 1:
        raise ValueError(
            f"layout {layout!r} requires views=1; got views={tensor.shape[1]}"
        )
