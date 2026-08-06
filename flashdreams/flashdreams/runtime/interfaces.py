# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocols for model adapters, reusable runtimes, and sessions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import (
    CanonicalInputSchema,
    InferenceInput,
    InferenceInputSchema,
)
from flashdreams.runtime.mapping import InputMapping
from flashdreams.runtime.types import StepRequest, StepResult


@runtime_checkable
class InferenceSession(Protocol):
    """One rollout or stream with isolated model/cache state."""

    def next_step_request(self) -> StepRequest | None:
        """Return the next step's runtime request, or ``None`` when complete."""
        ...

    def step(self, inputs: InferenceInput) -> StepResult:
        """Run one sequential inference step."""
        ...

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Reset this session's rollout state when the backend supports it."""
        ...

    def close(self) -> None:
        """Release per-session resources."""
        ...


@runtime_checkable
class InferenceRuntime(Protocol):
    """Heavyweight reusable runtime created from :class:`InferenceConfig`."""

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        """Create an isolated session from global conditioning inputs."""
        ...

    def close(self) -> None:
        """Release model/backend resources."""
        ...


# Do not mark ModelAdapter runtime-checkable: properties make issubclass()
# unreliable, and isinstance() would only verify attribute presence.
class ModelAdapter(Protocol):
    """Model-specific boundary that declares defaults and creates runtimes.

    Adapters declare model-facing input requirements, the canonical modalities
    their default mapping consumes, and an optional default mapping between the
    two. Runtime, application, or benchmark code may override that mapping while
    preserving the same ``CanonicalInputs`` to ``InferenceInput`` boundary.
    """

    @property
    def model_id(self) -> str:
        """Stable identity for the model adapter or runtime integration."""
        ...

    @property
    def inference_input_schema(self) -> InferenceInputSchema:
        """Model-facing global conditioning and per-step input requirements."""
        ...

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema | None:
        """Canonical modalities the adapter's default mapping consumes."""
        ...

    def default_input_mapping(self) -> InputMapping | None:
        """Return the model-provided default canonical-to-model mapping."""
        ...

    def validate_config(self, config: InferenceConfig) -> None:
        """Fail early for unsupported runtime settings."""
        ...

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        """Initialize and return the heavyweight runtime."""
        ...
