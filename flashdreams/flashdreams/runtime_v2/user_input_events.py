# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One batch of input events that an input source passes to a loop."""

from dataclasses import dataclass

from flashdreams.runtime_v2.user_input_event import UserInputEvent


@dataclass(frozen=True)
class UserInputEvents:
    """One batch of user input events, sorted by timestamp and not modifiable.

    What :meth:`~flashdreams.api_v2.input_source.InputSource.get_user_input_events`
    returns and what a loop's ``step`` receives. Sorting happens once, here, so
    neither end has to.
    """

    _data: list[UserInputEvent]
    """Immutable event collection data."""

    def __init__(self, events: list[UserInputEvent]) -> None:
        """Sort the events by timestamp and hold them.

        Args:
            events: Events to hold, in any order.
        """
        self._data = sorted(events, key=lambda event: event.get_timestamp())

    def get_events(self) -> list[UserInputEvent]:
        """Return a copy of the events, oldest first."""
        return list(self._data)
