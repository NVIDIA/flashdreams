# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User- and model-input envelopes for the experimental runtime API."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, kw_only=True, slots=True)
class TimeWindow:
    """Half-open time window in canonical seconds."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ValueError("TimeWindow bounds must be finite seconds.")
        if self.start_s < 0 or self.end_s < 0:
            raise ValueError("TimeWindow bounds must be non-negative.")
        if self.end_s < self.start_s:
            raise ValueError("TimeWindow.end_s must be >= start_s.")

    def contains(self, timestamp_s: float) -> bool:
        """Return whether ``timestamp_s`` falls within this half-open window."""
        return self.start_s <= timestamp_s < self.end_s


@dataclass(frozen=True, kw_only=True, slots=True)
class InputField:
    """Lightweight schema field for user snapshots or model inputs."""

    name: str
    required: bool = True
    semantic_type: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("InputField.name must be non-empty.")


@dataclass(frozen=True, kw_only=True, slots=True)
class UserInputSchema:
    """Minimal metadata for controls an app, transport, or trace can provide."""

    event_kinds: frozenset[str] = field(default_factory=frozenset)
    snapshot_fields: tuple[InputField, ...] = ()
    description: str = ""

    def supports_event_kinds(self, kinds: Iterable[str]) -> bool:
        """Return whether every requested event kind is declared supported."""
        requested = frozenset(kinds)
        if not requested:
            return True
        return requested.issubset(self.event_kinds)

    def missing_snapshot(self, inputs: "UserInputs") -> tuple[str, ...]:
        """Return required snapshot fields absent from ``inputs``."""
        return _missing_required(self.snapshot_fields, inputs.snapshot)

    def require_snapshot(self, inputs: "UserInputs") -> None:
        """Raise if required snapshot fields are absent."""
        missing = self.missing_snapshot(inputs)
        if missing:
            raise ValueError(f"Missing required user snapshot field(s): {missing}")


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelInputSchema:
    """Minimal metadata for model-facing initial and per-step inputs."""

    initial_fields: tuple[InputField, ...] = ()
    step_fields: tuple[InputField, ...] = ()
    description: str = ""

    def missing_initial(self, inputs: "ModelInputs") -> tuple[str, ...]:
        """Return required initial fields absent from ``inputs``."""
        return _missing_required(self.initial_fields, inputs.initial)

    def missing_step(self, inputs: "ModelInputs") -> tuple[str, ...]:
        """Return required per-step fields absent from ``inputs``."""
        return _missing_required(self.step_fields, inputs.step)

    def require_initial(self, inputs: "ModelInputs") -> None:
        """Raise if required initial fields are absent."""
        missing = self.missing_initial(inputs)
        if missing:
            raise ValueError(f"Missing required initial model input(s): {missing}")

    def require_step(self, inputs: "ModelInputs") -> None:
        """Raise if required per-step fields are absent."""
        missing = self.missing_step(inputs)
        if missing:
            raise ValueError(f"Missing required step model input(s): {missing}")


@dataclass(frozen=True, kw_only=True, slots=True)
class UserInputEvent:
    """Timestamped user-facing input event."""

    timestamp_s: float
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    event_id: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("UserInputEvent.timestamp_s must be finite and >= 0.")
        if not self.kind.strip():
            raise ValueError("UserInputEvent.kind must be non-empty.")


@dataclass(frozen=True, kw_only=True, slots=True)
class UserInputs:
    """Transport-neutral user inputs for live, replayed, synthetic, or no-op runs."""

    events: tuple[UserInputEvent, ...] = ()
    snapshot: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def window(self, time_window: TimeWindow) -> "UserInputs":
        """Return inputs with events filtered to ``time_window``."""
        return UserInputs(
            events=tuple(
                event
                for event in self.events
                if time_window.contains(event.timestamp_s)
            ),
            snapshot=self.snapshot,
            metadata=self.metadata,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelInputs:
    """Model-facing input payloads split by initial and per-step use."""

    initial: Mapping[str, Any] = field(default_factory=dict)
    step: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_step(self, step: Mapping[str, Any]) -> "ModelInputs":
        """Return a copy with replaced per-step payload."""
        return ModelInputs(initial=self.initial, step=step, metadata=self.metadata)


def _missing_required(
    fields: tuple[InputField, ...], payload: Mapping[str, Any]
) -> tuple[str, ...]:
    return tuple(
        field.name for field in fields if field.required and field.name not in payload
    )
