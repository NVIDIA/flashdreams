# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input mapping boundary from user input windows to model inputs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flashdreams.runtime.inputs import (
    ModelInputs,
    ModelInputSchema,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.types import StepRequest


@runtime_checkable
class InputMapping(Protocol):
    """Convert user-facing inputs into model-facing inputs.

    A mapping may be supplied by the model adapter as a default or by an
    application/runtime override. Step mappings usually receive a timestamped
    event window selected by the runner for the current model step or chunk.
    """

    def validate(
        self,
        *,
        user_schema: UserInputSchema | None = None,
        model_schema: ModelInputSchema | None = None,
    ) -> None:
        """Fail early for obvious app, event-source, and model mismatches."""
        ...

    def map_initial_inputs(
        self,
        *,
        user_inputs: UserInputs,
        model_inputs: ModelInputs,
    ) -> ModelInputs:
        """Build initial model inputs before a session starts."""
        ...

    def map_step_inputs(
        self,
        *,
        user_inputs: UserInputs,
        model_inputs: ModelInputs,
        request: StepRequest,
    ) -> ModelInputs:
        """Build model inputs for one session step from the current input window."""
        ...


class IdentityInputMapping:
    """No-op mapper for fixed model-input or simple generation flows."""

    def validate(
        self,
        *,
        user_schema: UserInputSchema | None = None,
        model_schema: ModelInputSchema | None = None,
    ) -> None:
        del user_schema, model_schema

    def map_initial_inputs(
        self,
        *,
        user_inputs: UserInputs,
        model_inputs: ModelInputs,
    ) -> ModelInputs:
        del user_inputs
        return model_inputs

    def map_step_inputs(
        self,
        *,
        user_inputs: UserInputs,
        model_inputs: ModelInputs,
        request: StepRequest,
    ) -> ModelInputs:
        del user_inputs, request
        return model_inputs
