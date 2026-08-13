# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public application contract for FlashDreams packages."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import UserInputs
from flashdreams.runtime.types import StepRequirements, StepResult


class ApplicationSession(Protocol):
    """One application session with isolated generation state."""

    def next_event(self) -> StepRequirements | None: ...

    def generate(
        self,
        event: StepRequirements,
        user_input: UserInputs,
    ) -> StepResult: ...

    def close(self) -> None: ...


@runtime_checkable
class FlashDreamsApplication(Protocol):
    application_name: str
    description: str
    model_id: str
    config: InferenceConfig
    default_io_handler: str

    @property
    def scenario(self) -> object: ...

    def initialize(self, config: InferenceConfig) -> None: ...

    def create_session(self, launch_args: object) -> ApplicationSession: ...

    def close(self) -> None: ...


__all__ = ["ApplicationSession", "FlashDreamsApplication"]
