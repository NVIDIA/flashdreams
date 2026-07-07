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

VideoValueRange = Literal["minus_one_one", "zero_one", "uint8"]
"""Supported numeric encodings for RGB video tensors."""


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
    """Frames in the layout and value range described by sibling fields."""

    layout: VideoTensorLayout = "tchw"
    """Layout of :attr:`tensor`."""

    value_range: VideoValueRange = "minus_one_one"
    """Numeric encoding of :attr:`tensor`."""

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

    @abstractmethod
    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Process one input chunk and return zero or more output chunks."""

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
        return self._process_from(0, [chunk])

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
            flushed = session.flush()
            if flushed:
                outputs.extend(self._process_from(index + 1, flushed))
        return outputs

    def _process_from(
        self, start_index: int, chunks: Iterable[VideoChunk]
    ) -> list[VideoChunk]:
        current = list(chunks)
        for session in self._sessions[start_index:]:
            next_chunks: list[VideoChunk] = []
            for chunk in current:
                next_chunks.extend(session.process(chunk))
            current = next_chunks
        return current


def infer_video_spec(
    tensor: Tensor,
    *,
    layout: VideoTensorLayout,
    fps: float | None = None,
) -> VideoSpec:
    """Infer :class:`VideoSpec` from a video tensor and layout."""
    canonical = to_bvtchw(tensor, layout=layout)
    _, _, _, channels, height, width = canonical.shape
    return VideoSpec(height=height, width=width, fps=fps, channels=channels)


def concatenate_video_chunks(
    chunks: Iterable[VideoChunk],
    *,
    layout: VideoTensorLayout,
    value_range: VideoValueRange,
) -> Tensor:
    """Concatenate chunks along time and convert to the requested representation."""
    canonical_chunks = [
        to_bvtchw(
            to_minus_one_one(chunk.tensor, value_range=chunk.value_range),
            layout=chunk.layout,
        )
        for chunk in chunks
    ]
    if not canonical_chunks:
        raise ValueError("cannot concatenate an empty video chunk sequence")
    canonical = torch.cat(canonical_chunks, dim=2)
    return from_minus_one_one(
        from_bvtchw(canonical, layout=layout),
        value_range=value_range,
    )


def to_minus_one_one(tensor: Tensor, *, value_range: VideoValueRange) -> Tensor:
    """Convert RGB video values to floating point ``[-1, 1]``."""
    if value_range == "minus_one_one":
        return tensor.float() if not tensor.is_floating_point() else tensor
    if value_range == "zero_one":
        return tensor.float() * 2.0 - 1.0
    if value_range == "uint8":
        return tensor.float() / 127.5 - 1.0
    raise ValueError(f"unsupported video value range: {value_range}")


def from_minus_one_one(tensor: Tensor, *, value_range: VideoValueRange) -> Tensor:
    """Convert floating point ``[-1, 1]`` RGB values to another encoding."""
    if value_range == "minus_one_one":
        return tensor
    if value_range == "zero_one":
        return ((tensor.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    if value_range == "uint8":
        return (((tensor.float() + 1.0) / 2.0) * 255.0).clamp(0, 255).to(torch.uint8)
    raise ValueError(f"unsupported video value range: {value_range}")


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
