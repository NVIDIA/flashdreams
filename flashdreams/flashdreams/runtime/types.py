# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain data carriers shared by runtime protocols and adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from flashdreams.infra.results import StepResult
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.inputs import InferenceInputSchema, TimeWindow

_STEP_REQUIREMENTS_INPUT_COUNT_METADATA_KEY = "input_frame_count"
_STEP_REQUIREMENTS_STEADY_OUTPUT_COUNT_METADATA_KEY = "steady_output_frame_count"
_STEP_REQUIREMENTS_USER_INPUT_METADATA_KEYS = frozenset(
    {
        "input_window",
        "user_input",
        "user_input_window",
        "user_inputs",
    }
)


@dataclass(frozen=True, kw_only=True, slots=True)
class StepRequirements:
    """Model-authored per-step requirements consumed by shared demo drivers."""

    __hash__ = None

    step_index: int
    input_frame_count: int = 1
    steady_output_frame_count: int | None = None
    inference_input_schema: InferenceInputSchema | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise TypeError("StepRequirements.step_index must be an integer.")
        if self.step_index < 0:
            raise ValueError("StepRequirements.step_index must be >= 0.")
        if isinstance(self.input_frame_count, bool) or not isinstance(
            self.input_frame_count, int
        ):
            raise TypeError("StepRequirements.input_frame_count must be an integer.")
        if self.input_frame_count <= 0:
            raise ValueError("StepRequirements.input_frame_count must be > 0.")
        if self.steady_output_frame_count is not None:
            if isinstance(self.steady_output_frame_count, bool) or not isinstance(
                self.steady_output_frame_count, int
            ):
                raise TypeError(
                    "StepRequirements.steady_output_frame_count must be an integer."
                )
            if self.steady_output_frame_count < 0:
                raise ValueError(
                    "StepRequirements.steady_output_frame_count must be >= 0."
                )
        user_input_keys = sorted(
            key
            for key in self.metadata
            if key in _STEP_REQUIREMENTS_USER_INPUT_METADATA_KEYS
        )
        if user_input_keys:
            joined = ", ".join(user_input_keys)
            raise ValueError(
                "StepRequirements.metadata must not include driver-owned user "
                f"input keys: {joined}."
            )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class StepRequest:
    """Per-step runtime request emitted by an inference session.

    This is not a schema declaration. ``user_input_window`` lets a runner drain
    or slice timestamped user events for the current step before invoking the
    selected ``InputMapping``.
    """

    __hash__ = None

    step_index: int
    inference_input_schema: InferenceInputSchema | None = None
    user_input_window: TimeWindow | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("StepRequest.step_index must be >= 0.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


def step_requirements_from_request(
    request: StepRequest,
    *,
    allow_user_input_window: bool = False,
) -> StepRequirements:
    """Adapt a legacy ``StepRequest`` that did not carry driver-owned inputs."""

    if request.user_input_window is not None and not allow_user_input_window:
        raise ValueError(
            "StepRequest.user_input_window cannot be adapted to StepRequirements; "
            "user input windows are driver-owned."
        )
    metadata = dict(request.metadata)
    input_frame_count = metadata.pop(_STEP_REQUIREMENTS_INPUT_COUNT_METADATA_KEY, 1)
    steady_output_frame_count = metadata.pop(
        _STEP_REQUIREMENTS_STEADY_OUTPUT_COUNT_METADATA_KEY,
        None,
    )
    return StepRequirements(
        step_index=request.step_index,
        input_frame_count=input_frame_count,
        steady_output_frame_count=steady_output_frame_count,
        inference_input_schema=request.inference_input_schema,
        metadata=metadata,
    )


__all__ = [
    "StepRequest",
    "StepRequirements",
    "StepResult",
    "step_requirements_from_request",
]
