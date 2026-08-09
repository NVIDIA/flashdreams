# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-source and model-input-provider contracts for demo sessions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.inputs import InferenceInput, UserInputs
from flashdreams.runtime.types import StepRequest


@dataclass(frozen=True, kw_only=True, slots=True)
class ControlDecision:
    """Provider-authored control request for the current session."""

    reset: bool = False
    close_session: bool = False
    reset_input: InferenceInput | None = None
    provider_already_reset: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.reason is not None and not self.reason.strip():
            raise ValueError("ControlDecision.reason must be non-empty when set.")


@dataclass(frozen=True, kw_only=True, slots=True)
class UserInputWindow:
    """User/app inputs selected by a driver for one model step."""

    __hash__ = None

    start_s: float
    end_s: float
    frame_times: Sequence[float] = ()
    inputs: UserInputs = field(default_factory=UserInputs)
    control: ControlDecision | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError("UserInputWindow.start_s must be finite and >= 0.")
        if not math.isfinite(self.end_s) or self.end_s < self.start_s:
            raise ValueError(
                "UserInputWindow.end_s must be finite and >= start_s."
            )
        previous = -math.inf
        for frame_time in self.frame_times:
            if not math.isfinite(float(frame_time)):
                raise ValueError("UserInputWindow.frame_times must be finite.")
            if float(frame_time) < previous:
                raise ValueError(
                    "UserInputWindow.frame_times must be sorted in ascending order."
                )
            previous = float(frame_time)
        object.__setattr__(self, "frame_times", tuple(float(t) for t in self.frame_times))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class PreparedStep:
    """Model-facing input plus optional provider-authored control decision."""

    __hash__ = None

    inference_input: InferenceInput | None = None
    control: ControlDecision = field(default_factory=ControlDecision)


@runtime_checkable
class InputSource(Protocol):
    """Facts common to every demo session input source."""

    is_finite: bool
    is_deterministic: bool

    def is_finished(self) -> bool:
        """Return whether the driver should stop requesting windows."""
        ...


@runtime_checkable
class BatchInputSource(InputSource, Protocol):
    """Finite input source consumed by the batch driver."""

    def next_window(self, request: StepRequest) -> UserInputWindow:
        """Return the next batch input window for ``request``."""
        ...


@runtime_checkable
class RealtimeInputSource(InputSource, Protocol):
    """Realtime input source consumed by a future realtime driver."""

    async def next_realtime_window(
        self,
        *,
        request: StepRequest,
        clock: object,
    ) -> object:
        """Return the next realtime window result.

        The concrete realtime result shape lands with the realtime clock phase.
        Keeping this protocol separate now prevents batch sources from stubbing
        async behavior they never serve.
        """
        ...


@runtime_checkable
class ModelInputProvider(Protocol):
    """Model-owned conversion from user windows into model-facing inputs."""

    def prepare_initial_input(self) -> InferenceInput:
        """Prepare session-global model inputs."""
        ...

    def prepare_step(
        self,
        *,
        request: StepRequest,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        """Prepare one model step from a driver-owned user input window."""
        ...

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Reset provider-owned session state."""
        ...

    def close(self) -> None:
        """Release provider-owned resources."""
        ...


__all__ = [
    "BatchInputSource",
    "ControlDecision",
    "InputSource",
    "ModelInputProvider",
    "PreparedStep",
    "RealtimeInputSource",
    "UserInputWindow",
]
