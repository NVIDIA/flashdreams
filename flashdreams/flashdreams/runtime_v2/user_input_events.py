# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One batch of input events, as an input source hands them over."""

from dataclasses import dataclass

from flashdreams.runtime_v2.user_input_event import UserInputEvent


@dataclass(frozen=True)
class UserInputEventsData:
    """Sorted events held by one :class:`UserInputEvents` batch."""

    events: list[UserInputEvent]
    """Input events ordered by timestamp."""


class UserInputEvents:
    """One batch of user input events, sorted by timestamp and not modifiable.

    What :meth:`~flashdreams.api_v2.input_source.InputSource.get_user_input_events`
    returns and what a loop's ``step`` receives. Sorting happens once, here, so
    neither end has to.
    """

    _data: UserInputEventsData
    """Immutable event collection data."""

    def __init__(self, events: list[UserInputEvent]) -> None:
        """
        Args:
            events: Events to hold, in any order.
        """
        self._data = UserInputEventsData(
            events=sorted(events, key=lambda event: event.get_timestamp()),
        )

    def get_events(self) -> list[UserInputEvent]:
        """Return a copy of the events, oldest first."""
        return list(self._data.events)
