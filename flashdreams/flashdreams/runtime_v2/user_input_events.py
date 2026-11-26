# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One batch of input events that an input source passes to a loop."""

from dataclasses import dataclass

from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    InputOwnerChangedUserInputEvent,
    UserInputEvent,
)


@dataclass(frozen=True)
class UserInputEvents:
    """One batch of input, ordered within its current ownership boundary.

    What :meth:`~flashdreams.api_v2.input_source.InputSource.get_user_input_events`
    returns and what a loop's ``step`` receives. The latest ownership boundary
    discards earlier input except terminal close requests. It precedes all new
    input regardless of timestamps; ordinary input is sorted by timestamp.
    """

    _data: list[UserInputEvent]
    """Immutable event collection data."""

    def __init__(self, events: list[UserInputEvent]) -> None:
        """Discard retired input and sort the current owner's events.

        Args:
            events: Events in arrival order across ownership boundaries;
                ordinary input may have unordered timestamps.
        """
        boundary = next(
            (
                index
                for index in range(len(events) - 1, -1, -1)
                if isinstance(events[index], InputOwnerChangedUserInputEvent)
            ),
            None,
        )
        prefix: list[UserInputEvent] = []
        if boundary is not None:
            prefix = [
                event
                for event in events[:boundary]
                if isinstance(event, CloseUserInputEvent)
            ] + [events[boundary]]
            events = events[boundary + 1 :]
        object.__setattr__(
            self,
            "_data",
            prefix + sorted(events, key=lambda event: event.get_timestamp()),
        )

    def get_events(self) -> list[UserInputEvent]:
        """Return a copy with the boundary before timestamp-ordered input."""
        return list(self._data)
