# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocols for model adapters, runtimes, and sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import (
    ModelInputs,
    ModelInputSchema,
    TimeWindow,
    UserInputSchema,
)

if TYPE_CHECKING:
    from flashdreams.runtime.mapping import InputMapping


@dataclass(frozen=True, kw_only=True, slots=True)
class StepRequest:
    """Model-session request for the next step's inputs."""

    step_index: int
    model_input_schema: ModelInputSchema | None = None
    user_input_window: TimeWindow | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("StepRequest.step_index must be >= 0.")


@dataclass(frozen=True, kw_only=True, slots=True)
class StepResult:
    """Generated output and metadata for one inference step."""

    step_index: int
    output: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("StepResult.step_index must be >= 0.")


@runtime_checkable
class InferenceSession(Protocol):
    """One rollout or stream with isolated cache/state."""

    def next_step_request(self) -> StepRequest:
        """Describe the model inputs needed for the next call to :meth:`step`."""
        ...

    def step(self, inputs: ModelInputs) -> StepResult:
        """Run one sequential inference step."""
        ...

    def reset(self, inputs: ModelInputs | None = None) -> None:
        """Reset this session's rollout state when the backend supports it."""
        ...

    def close(self) -> None:
        """Release per-session resources."""
        ...


@runtime_checkable
class InferenceRuntime(Protocol):
    """Heavyweight reusable runtime created from :class:`InferenceConfig`."""

    def start_session(self, inputs: ModelInputs) -> InferenceSession:
        """Create an isolated session from initial model inputs."""
        ...

    def close(self) -> None:
        """Release model/backend resources."""
        ...


@runtime_checkable
class ModelAdapter(Protocol):
    """Model-specific boundary that connects FlashDreams to a model runtime."""

    @property
    def model_id(self) -> str:
        """Stable model or adapter identity."""
        ...

    @property
    def model_input_schema(self) -> ModelInputSchema:
        """Initial and per-step model input requirements."""
        ...

    @property
    def user_input_schema(self) -> UserInputSchema | None:
        """User input capabilities this adapter can map directly, if any."""
        ...

    def default_input_mapping(self) -> "InputMapping | None":
        """Return the adapter's default user-to-model input mapping, if any."""
        ...

    def validate_config(self, config: InferenceConfig) -> None:
        """Fail early for unsupported runtime settings."""
        ...

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        """Initialize and return the heavyweight runtime."""
        ...
