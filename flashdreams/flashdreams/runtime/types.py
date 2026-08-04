# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain data carriers shared by runtime protocols and adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.inputs import ModelInputSchema, TimeWindow


@dataclass(frozen=True, kw_only=True, slots=True)
class StepRequest:
    """Model-session request for the next step's inputs.

    ``user_input_window`` lets a runner drain or slice timestamped user events for
    the current step before invoking the selected ``InputMapping``.
    """

    __hash__ = None

    step_index: int
    model_input_schema: ModelInputSchema | None = None
    user_input_window: TimeWindow | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("StepRequest.step_index must be >= 0.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class StepResult:
    """Generated output and metadata for one inference step."""

    __hash__ = None

    step_index: int
    output: Any = None
    frame_count: int | None = None
    output_window: TimeWindow | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("StepResult.step_index must be >= 0.")
        if self.frame_count is not None and self.frame_count < 0:
            raise ValueError("StepResult.frame_count must be >= 0.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        object.__setattr__(self, "metrics", freeze_mapping(self.metrics))
