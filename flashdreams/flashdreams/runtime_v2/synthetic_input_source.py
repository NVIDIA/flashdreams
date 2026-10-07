# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic user input via descriptor file."""

from __future__ import annotations

import json
import math
import pydoc
import time
import types
from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path
from typing import Literal, Union, get_args, get_origin, get_type_hints

from numpy import uint64

from flashdreams.api_v2.input_source import SessionInputSource
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.user_input_event import KeyboardInputState
from flashdreams.runtime_v2.user_input_events import UserInputEvents


@dataclass(frozen=True, slots=True)
class _ScheduledEvent:
    """One event released at or after a zero-based UI poll."""

    poll: int
    event: UserInputEvent


class SyntheticInputSource(SessionInputSource):
    """Release descriptor events exactly once on their requested UI polls."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        """Load a version 1 synthetic-input descriptor.

        Args:
            path: JSON descriptor with ``version`` and ``events`` fields.
            clock_ns: Monotonic clock used for emitted event timestamps.

        Raises:
            TypeError: The descriptor has an invalid JSON value type.
            ValueError: The descriptor has an invalid or unsupported value.
        """
        self._events = _load_descriptor(Path(path))
        self._clock_ns = clock_ns
        self._session_started_ns: int | None = None
        self._next_event_index = 0
        self._poll = 0

    def open(self) -> None:
        """Start a new session-relative clock and replay cursor."""
        self._session_started_ns = self._clock_ns()
        self._next_event_index = 0
        self._poll = 0

    def close(self) -> None:
        """Clear the active session clock."""
        self._session_started_ns = None

    def get_user_input_events(self) -> UserInputEvents:
        """Return events due on this UI poll, stamped at emission time."""
        started_ns = self._session_started_ns
        if started_ns is None:
            raise RuntimeError("SyntheticInputSource.open() must run before polling.")
        elapsed_us = uint64(max(0, self._clock_ns() - started_ns) // 1_000)
        released: list[UserInputEvent] = []
        while (
            self._next_event_index < len(self._events)
            and self._events[self._next_event_index].poll <= self._poll
        ):
            scheduled = self._events[self._next_event_index]
            released.append(replace(scheduled.event, timestamp=elapsed_us))
            self._next_event_index += 1
        self._poll += 1
        return UserInputEvents(released)


def _load_descriptor(path: Path) -> list[_ScheduledEvent]:
    """Read and strictly decode a version 1 descriptor."""
    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read synthetic input descriptor {path}: {error}") from error
    if not isinstance(contents, dict) or set(contents) != {"version", "events"}:
        raise ValueError("Descriptor must contain exactly 'version' and 'events'.")
    if (
        isinstance(contents["version"], bool)
        or not isinstance(contents["version"], int)
        or contents["version"] != 1
    ):
        raise ValueError("Descriptor version must be the integer 1.")
    raw_events = contents["events"]
    if not isinstance(raw_events, list):
        raise TypeError("Descriptor events must be a JSON array.")
    events = [_decode_scheduled_event(raw_event) for raw_event in raw_events]
    return sorted(events, key=lambda scheduled: scheduled.poll)


def _decode_scheduled_event(raw_event: object) -> _ScheduledEvent:
    """Decode one flat schedule record."""
    if not isinstance(raw_event, dict) or set(raw_event) != {"on", "at", "event"}:
        raise ValueError("Each descriptor event must contain exactly 'on', 'at', and 'event'.")
    if raw_event["on"] != "ui_loop":
        raise ValueError("Descriptor event 'on' must be 'ui_loop'.")
    poll = raw_event["at"]
    if isinstance(poll, bool) or not isinstance(poll, int) or poll < 0:
        raise ValueError("Descriptor event 'at' must be a non-negative integer.")
    return _ScheduledEvent(poll=poll, event=_decode_event(raw_event["event"]))


def _decode_event(raw_event: object) -> UserInputEvent:
    """Decode an event using its concrete ``UserInputEvent`` field names."""
    if not isinstance(raw_event, dict):
        raise TypeError("Descriptor event payload must be a JSON object.")
    event_type_name = raw_event.get("type")
    if not isinstance(event_type_name, str):
        raise TypeError("Descriptor event type must be a string.")
    owner = UserInputEvent._type_name_owners.get(event_type_name)
    event_type = None if owner is None else pydoc.locate(owner)
    if (
        not isinstance(event_type, type)
        or not issubclass(event_type, UserInputEvent)
    ):
        raise ValueError(f"Unsupported synthetic input event type: {event_type_name!r}.")
    event_fields = {field.name: field for field in fields(event_type)}
    allowed_fields = set(event_fields) - {"timestamp"}
    if set(raw_event) - {"type"} != allowed_fields.intersection(raw_event):
        raise ValueError(f"Unexpected fields for {event_type_name!r} event.")
    hints = get_type_hints(event_type)
    values: dict[str, object] = {"timestamp": uint64(0)}
    for field_name, field in event_fields.items():
        if field_name == "timestamp":
            continue
        if field_name not in raw_event:
            if field.default is MISSING and field.default_factory is MISSING:
                raise ValueError(f"{event_type_name!r} event requires {field_name!r}.")
            continue
        values[field_name] = _decode_value(
            raw_event[field_name], hints[field_name], label=field_name
        )
    return event_type(**values)


def _decode_value(value: object, annotation: object, *, label: str) -> object:
    """Validate a JSON value against the concrete event field annotation."""
    origin = get_origin(annotation)
    if origin is Literal:
        if value not in get_args(annotation):
            raise ValueError(f"{label} has an unsupported value.")
        return value
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError(f"{label} must be a JSON array.")
        item_types = get_args(annotation)
        if len(item_types) == 2 and item_types[1] is Ellipsis:
            return tuple(_decode_value(item, item_types[0], label=label) for item in value)
        if len(value) != len(item_types):
            raise ValueError(f"{label} must contain {len(item_types)} values.")
        return tuple(
            _decode_value(item, item_type, label=label)
            for item, item_type in zip(value, item_types, strict=True)
        )
    if origin in {Union, types.UnionType}:
        for option in get_args(annotation):
            if option is type(None) and value is None:
                return None
            try:
                return _decode_value(value, option, label=label)
            except ValueError:
                pass
        raise ValueError(f"{label} has an invalid value.")
    if annotation is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be a boolean.")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} must be an integer.")
        return value
    if annotation is float:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label} must be a number.")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ValueError(f"{label} must be a string.")
        return value
    if annotation is KeyboardInputState:
        try:
            return KeyboardInputState(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be a keyboard input state.") from error
    raise TypeError(f"Unsupported descriptor field type for {label}: {annotation!r}.")


__all__ = ["SyntheticInputSource"]