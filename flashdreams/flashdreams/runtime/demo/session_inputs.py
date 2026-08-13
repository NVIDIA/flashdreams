# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-source and model-input-provider contracts for demo sessions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.inputs import (
    InferenceInput,
    InferenceInputSchema,
    TimeWindow,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.types import (
    BATCH_INPUT_FRAME_START_METADATA_KEY,
    StepRequest,
    StepRequirements,
)

if TYPE_CHECKING:
    from .spec import PreparedScenario
    from .timing import RealtimeClock, RealtimeWindowResult

BATCH_INPUT_FPS_METADATA_KEY = "batch_input_fps"
_DEFAULT_SESSION_HORIZON_S = 3600.0


@dataclass(frozen=True, kw_only=True, slots=True)
class ProviderCapabilities:
    """Model-provider capabilities used to validate run-mode compatibility."""

    supports_realtime_clock: bool = False
    supports_recorded_input: bool = False
    supports_reset: bool = False
    deterministic_given_inputs: bool = False
    user_input_schema: UserInputSchema = field(default_factory=UserInputSchema)
    inference_input_schema: InferenceInputSchema = field(
        default_factory=InferenceInputSchema
    )


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
            raise ValueError("UserInputWindow.end_s must be finite and >= start_s.")
        previous = -math.inf
        for frame_time in self.frame_times:
            if not math.isfinite(float(frame_time)):
                raise ValueError("UserInputWindow.frame_times must be finite.")
            if float(frame_time) < previous:
                raise ValueError(
                    "UserInputWindow.frame_times must be sorted in ascending order."
                )
            previous = float(frame_time)
        object.__setattr__(
            self, "frame_times", tuple(float(t) for t in self.frame_times)
        )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


class StepRequestWindowState:
    """Temporary bridge from legacy ``StepRequest`` windows to shared sources."""

    def __init__(self) -> None:
        self._request: StepRequest | None = None

    def store(self, request: StepRequest) -> None:
        self._request = request

    def request_for_window(self, step_index: int) -> StepRequest | None:
        request = self._request
        if request is None:
            return None
        if request.step_index != step_index:
            raise RuntimeError(
                "Batch input source request mismatch: "
                f"expected step {request.step_index}, got {step_index}."
            )
        return request

    def consume_for_step(self, request: StepRequirements) -> StepRequest:
        legacy_request = self.request_for_window(request.step_index)
        if legacy_request is not None:
            self.clear()
            return legacy_request
        return StepRequest(
            step_index=request.step_index,
            inference_input_schema=request.inference_input_schema,
            metadata=request.metadata,
        )

    def clear(self) -> None:
        self._request = None


class PreparedScenarioBatchInputSource:
    """Deterministic finite source for replay and benchmark scenarios."""

    is_finite = True
    is_deterministic = True

    def __init__(
        self,
        *,
        scenario: "PreparedScenario",
        request_state: StepRequestWindowState | None = None,
    ) -> None:
        self.user_input_schema = scenario.source_schema
        self._scenario = scenario
        self._user_inputs = scenario.user_inputs
        self._request_state = request_state or StepRequestWindowState()

    def is_finished(self) -> bool:
        # Finiteness is session-owned: sessions report completion by returning
        # ``None`` from their next-step method.
        return False

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        window = self._window_for_request(request)
        return UserInputWindow(
            start_s=window.start_s,
            end_s=window.end_s,
            inputs=self._user_inputs.window(window),
        )

    def _window_for_request(self, request: StepRequirements) -> TimeWindow:
        legacy_request = self._request_state.request_for_window(request.step_index)
        if legacy_request is not None and legacy_request.user_input_window is not None:
            return legacy_request.user_input_window
        timed_window = _timed_window_from_requirements(
            request=request,
            scenario_metadata=self._scenario.metadata,
        )
        if timed_window is not None:
            return timed_window
        return all_user_inputs_window(self._user_inputs)


def all_user_inputs_window(user_inputs: UserInputs) -> TimeWindow:
    """Return a stable whole-session fallback window for finite demo inputs."""
    if not user_inputs.events:
        return TimeWindow(start_s=0.0, end_s=_DEFAULT_SESSION_HORIZON_S)
    return TimeWindow(
        start_s=0.0,
        end_s=max(
            _DEFAULT_SESSION_HORIZON_S,
            math.nextafter(user_inputs.events[-1].timestamp_s, math.inf),
        ),
    )


def _timed_window_from_requirements(
    *,
    request: StepRequirements,
    scenario_metadata: Mapping[str, object],
) -> TimeWindow | None:
    frame_start = _optional_nonnegative_number(
        request.metadata,
        BATCH_INPUT_FRAME_START_METADATA_KEY,
        label="StepRequirements.metadata",
    )
    if frame_start is None:
        return None
    fps = _optional_positive_number(
        scenario_metadata,
        BATCH_INPUT_FPS_METADATA_KEY,
        label="PreparedScenario.metadata",
    )
    if fps is None:
        return None
    return TimeWindow(
        start_s=frame_start / fps,
        end_s=(frame_start + request.input_frame_count) / fps,
    )


def _optional_nonnegative_number(
    metadata: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> float | None:
    if key not in metadata:
        return None
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label}[{key!r}] must be numeric when set.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{label}[{key!r}] must be finite and >= 0.")
    return normalized


def _optional_positive_number(
    metadata: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> float | None:
    if key not in metadata:
        return None
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label}[{key!r}] must be numeric when set.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{label}[{key!r}] must be finite and > 0.")
    return normalized


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
    user_input_schema: UserInputSchema

    def is_finished(self) -> bool:
        """Return whether the driver should stop requesting windows."""
        ...


@runtime_checkable
class BatchInputSource(InputSource, Protocol):
    """Finite input source consumed by the batch driver."""

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        """Return the next batch input window for ``request``."""
        ...


@runtime_checkable
class RealtimeInputSource(InputSource, Protocol):
    """Realtime input source consumed by a future realtime driver."""

    async def next_realtime_window(
        self,
        *,
        request: StepRequirements,
        clock: "RealtimeClock",
    ) -> "RealtimeWindowResult":
        """Return the next realtime window result.

        The concrete realtime result shape lands with the realtime clock phase.
        Keeping this protocol separate now prevents batch sources from stubbing
        async behavior they never serve.
        """
        ...


@runtime_checkable
class ModelInputProvider(Protocol):
    """Model-owned conversion from user windows into model-facing inputs."""

    capabilities: ProviderCapabilities

    def prepare_initial_input(self) -> InferenceInput:
        """Prepare session-global model inputs."""
        ...

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        """Prepare one model step from a driver-owned user input window."""
        ...

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Reset provider-owned session state.

        Implementations must be idempotent so driver cleanup and reset control
        paths can safely converge after failures.
        """
        ...

    def close(self) -> None:
        """Release provider-owned resources.

        Implementations must be idempotent and tolerate cleanup after partial
        setup or earlier reset failures.
        """
        ...


__all__ = [
    "BATCH_INPUT_FPS_METADATA_KEY",
    "BATCH_INPUT_FRAME_START_METADATA_KEY",
    "BatchInputSource",
    "ControlDecision",
    "InputSource",
    "ModelInputProvider",
    "PreparedScenarioBatchInputSource",
    "PreparedStep",
    "ProviderCapabilities",
    "RealtimeInputSource",
    "StepRequestWindowState",
    "UserInputWindow",
    "all_user_inputs_window",
]
