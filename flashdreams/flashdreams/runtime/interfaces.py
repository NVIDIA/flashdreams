# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocols for model adapters, reusable runtimes, and sessions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import (
    ModelInputs,
    ModelInputSchema,
    UserInputSchema,
)
from flashdreams.runtime.mapping import InputMapping
from flashdreams.runtime.types import StepRequest, StepResult


@runtime_checkable
class InferenceSession(Protocol):
    """One rollout or stream with isolated model/cache state."""

    def next_step_request(self) -> StepRequest | None:
        """Describe the next step's inputs, or return ``None`` when complete."""
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


# Do not mark ModelAdapter runtime-checkable: properties make issubclass()
# unreliable, and isinstance() would only verify attribute presence.
class ModelAdapter(Protocol):
    """Model-specific boundary that declares defaults and creates runtimes.

    Adapters declare model-facing input requirements, optional user-input
    capabilities, and an optional default mapping between the two. Runtime,
    application, or benchmark code may override that mapping while preserving the
    same ``UserInputs`` to ``ModelInputs`` boundary.
    """

    @property
    def model_id(self) -> str:
        """Stable identity for the model adapter or runtime integration."""
        ...

    @property
    def model_input_schema(self) -> ModelInputSchema:
        """Model-facing initial and per-step input requirements."""
        ...

    @property
    def user_input_schema(self) -> UserInputSchema | None:
        """User inputs supported by the adapter's default mapping, if any."""
        ...

    def default_input_mapping(self) -> InputMapping | None:
        """Return the model-provided default user-to-model mapping, if any."""
        ...

    def validate_config(self, config: InferenceConfig) -> None:
        """Fail early for unsupported runtime settings."""
        ...

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        """Initialize and return the heavyweight runtime."""
        ...
