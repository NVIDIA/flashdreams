# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-first provider boundary used by :mod:`flashdreams_app`."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.runtime import InferenceInput


@dataclass(frozen=True, slots=True)
class AppRequest:
    """Parsed invocation supplied to an application provider."""

    mode: str
    """Selected presentation mode."""

    options: Mapping[str, Any]
    """Parsed presentation and provider options keyed by argument destination."""

    def __post_init__(self) -> None:
        if self.mode not in ("mp4", "webrtc"):
            raise ValueError(f"Unsupported application mode: {self.mode!r}.")
        if not isinstance(self.options, Mapping):
            raise TypeError("AppRequest.options must be a mapping.")


@dataclass(frozen=True, kw_only=True, slots=True)
class AppConfig:
    """Presentation configuration supplied by an application provider."""

    model_id: str
    """Stable model identity used by presentation and serving layers."""

    fps: int | float
    """Output video frame rate."""

    output_layout: VideoTensorLayout
    """Layout of video tensors returned by the pipeline."""

    video_width: int
    """Output video width in pixels."""

    video_height: int
    """Output video height in pixels."""

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("AppConfig.model_id must be non-empty.")
        if float(self.fps) <= 0:
            raise ValueError("AppConfig.fps must be > 0.")
        if self.video_width <= 0 or self.video_height <= 0:
            raise ValueError("AppConfig video dimensions must be > 0.")


@dataclass(frozen=True, kw_only=True, slots=True)
class PipelineAppSpec:
    """Mode-independent pipeline definition consumed by the application host."""

    pipeline_config: StreamInferencePipelineConfig
    """Pipeline config that the host constructs through ``setup()``."""

    initialize_cache: Callable[[Any, InferenceInput], object]
    """Create one cache from the constructed pipeline and session input."""

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_config, StreamInferencePipelineConfig):
            raise TypeError(
                "PipelineAppSpec.pipeline_config must be a "
                "StreamInferencePipelineConfig."
            )
        if not callable(self.initialize_cache):
            raise TypeError("PipelineAppSpec.initialize_cache must be callable.")


@dataclass(frozen=True, kw_only=True, slots=True)
class Mp4RunSpec:
    """Finite session inputs required by the MP4 presentation path."""

    initial_input: InferenceInput
    """Global conditioning used to start the finite session."""

    total_steps: int
    """Number of autoregressive steps to write to the output file."""

    def __post_init__(self) -> None:
        if not isinstance(self.initial_input, InferenceInput):
            raise TypeError("Mp4RunSpec.initial_input must be InferenceInput.")
        if isinstance(self.total_steps, bool) or not isinstance(self.total_steps, int):
            raise TypeError("Mp4RunSpec.total_steps must be an integer.")
        if self.total_steps <= 0:
            raise ValueError("Mp4RunSpec.total_steps must be > 0.")


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCRunSpec:
    """Session inputs required by the live WebRTC presentation path."""

    initial_input: InferenceInput
    """Global conditioning used to start each live session."""

    def __post_init__(self) -> None:
        if not isinstance(self.initial_input, InferenceInput):
            raise TypeError("WebRTCRunSpec.initial_input must be InferenceInput.")


@dataclass(frozen=True, kw_only=True, slots=True)
class AppSpec:
    """Pipeline definition paired with the selected mode's run data."""

    config: AppConfig
    """Model identity and video presentation configuration."""

    pipeline: PipelineAppSpec
    """Mode-independent pipeline definition."""

    run: Mp4RunSpec | WebRTCRunSpec
    """Inputs and limits for the selected presentation mode."""

    def __post_init__(self) -> None:
        if not isinstance(self.config, AppConfig):
            raise TypeError("AppSpec.config must be AppConfig.")
        if not isinstance(self.pipeline, PipelineAppSpec):
            raise TypeError("AppSpec.pipeline must be PipelineAppSpec.")
        if not isinstance(self.run, (Mp4RunSpec, WebRTCRunSpec)):
            raise TypeError("AppSpec.run must be Mp4RunSpec or WebRTCRunSpec.")


@runtime_checkable
class AppProvider(Protocol):
    """Required interface for an installed application-provider module."""

    def parse_options(
        self,
        parser: argparse.ArgumentParser,
        argv: Sequence[str],
    ) -> Mapping[str, Any]:
        """Parse provider arguments with the selected mode's host parser.

        Args:
            parser: Parser preconfigured with host-owned presentation options.
            argv: Arguments remaining after the provider and mode.

        Returns:
            Parsed presentation and provider options keyed by destination.
        """
        ...

    def create_app_spec(self, request: AppRequest) -> AppSpec:
        """Describe the pipeline application without constructing its runtime.

        Args:
            request: Parsed invocation supplied by the application host.

        Returns:
            Pipeline definition and run data for the selected presentation mode.
        """
        ...


__all__ = [
    "AppConfig",
    "AppProvider",
    "AppRequest",
    "AppSpec",
    "Mp4RunSpec",
    "PipelineAppSpec",
    "WebRTCRunSpec",
]
