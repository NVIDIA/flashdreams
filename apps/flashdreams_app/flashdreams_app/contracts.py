# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-first provider boundary used by :mod:`flashdreams_app`."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.runtime import InferenceInput, InferenceRuntime
from flashdreams.runtime.types import StepRequest, StepRequirements


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Provider-specific CLI values normalized by the application host."""

    options: Mapping[str, Any]


@dataclass(frozen=True, kw_only=True, slots=True)
class RuntimeMetadata:
    """Presentation facts published by an application runtime."""

    model_id: str
    fps: int | float
    output_layout: VideoTensorLayout
    video_width: int
    video_height: int

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("RuntimeMetadata.model_id must be non-empty.")
        if float(self.fps) <= 0:
            raise ValueError("RuntimeMetadata.fps must be > 0.")
        if self.video_width <= 0 or self.video_height <= 0:
            raise ValueError("RuntimeMetadata video dimensions must be > 0.")


PipelineCacheInitializer = Callable[[Any, InferenceInput], object]


@dataclass(frozen=True, kw_only=True, slots=True)
class PipelineContract:
    """Describe the model-specific operation needed to start a pipeline session.

    The host already understands the standard streaming pipeline operations:
    ``generate``, ``finalize``, and ``get_num_output_frames``. A provider only
    supplies the cache initialization that binds its global conditioning to a
    concrete pipeline implementation.
    """

    initialize_cache: PipelineCacheInitializer

    def __post_init__(self) -> None:
        if not callable(self.initialize_cache):
            raise TypeError("PipelineContract.initialize_cache must be callable.")


@dataclass(frozen=True, kw_only=True, slots=True)
class PipelineAppSpec:
    """Declarative application definition consumed by the generic host runtime."""

    pipeline_config: StreamInferencePipelineConfig
    contract: PipelineContract
    metadata: RuntimeMetadata
    initial_input: InferenceInput
    total_steps: int
    result_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_config, StreamInferencePipelineConfig):
            raise TypeError(
                "PipelineAppSpec.pipeline_config must be a "
                "StreamInferencePipelineConfig."
            )
        if not isinstance(self.contract, PipelineContract):
            raise TypeError("PipelineAppSpec.contract must be a PipelineContract.")
        if not isinstance(self.metadata, RuntimeMetadata):
            raise TypeError("PipelineAppSpec.metadata must be RuntimeMetadata.")
        if not isinstance(self.initial_input, InferenceInput):
            raise TypeError("PipelineAppSpec.initial_input must be InferenceInput.")
        if isinstance(self.total_steps, bool) or not isinstance(self.total_steps, int):
            raise TypeError("PipelineAppSpec.total_steps must be an integer.")
        if self.total_steps <= 0:
            raise ValueError("PipelineAppSpec.total_steps must be > 0.")
        object.__setattr__(
            self,
            "result_metadata",
            MappingProxyType(dict(self.result_metadata)),
        )


@runtime_checkable
class AppProvider(Protocol):
    """Provider module contract consumed by the application host."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register provider-specific arguments on the host parser."""
        ...

    def create_app_spec(self, config: AppConfig) -> PipelineAppSpec:
        """Create a declarative application specification."""
        ...


def require_pipeline_config(
    config: object, *, expected_name: str | None = None
) -> StreamInferencePipelineConfig:
    """Validate a pipeline provider result at the application boundary."""
    if not isinstance(config, StreamInferencePipelineConfig):
        raise TypeError(
            f"Pipeline provider returned {type(config).__name__}, expected "
            "StreamInferencePipelineConfig."
        )
    if expected_name is not None and config.name != expected_name:
        raise ValueError(
            f"Preset {expected_name!r} constructed pipeline {config.name!r}; "
            "the preset key and pipeline name must match."
        )
    return config


@runtime_checkable
class AppRuntime(InferenceRuntime, Protocol):
    """FlashDreams runtime plus the small surface required by the app host.

    Sessions use the standard :class:`~flashdreams.runtime.InferenceSession`
    protocol. Providers add only immutable initial inputs, per-step input
    preparation, and presentation metadata.
    """

    metadata: RuntimeMetadata
    initial_input: InferenceInput

    def prepare_step_input(
        self, request: StepRequest | StepRequirements
    ) -> InferenceInput:
        """Build model-facing input for one host-owned session step."""
        ...
