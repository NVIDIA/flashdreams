# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocols for model adapters, runtimes, and sessions."""

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
    """One rollout or stream with isolated cache/state."""

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

    def default_input_mapping(self) -> InputMapping | None:
        """Return the adapter's default user-to-model input mapping, if any."""
        ...

    def validate_config(self, config: InferenceConfig) -> None:
        """Fail early for unsupported runtime settings."""
        ...

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        """Initialize and return the heavyweight runtime."""
        ...
