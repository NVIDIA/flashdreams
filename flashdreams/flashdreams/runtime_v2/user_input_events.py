# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One batch of input events that an input source passes to a loop."""

from collections.abc import Mapping
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

    _received_at_ns: Mapping[int, int]
    """Optional monotonic receipt times keyed by event identity."""

    def __init__(
        self,
        events: list[UserInputEvent],
        *,
        received_at_ns: Mapping[int, int] | None = None,
    ) -> None:
        """Sort the events by timestamp and hold them.

        Args:
            events: Events to hold, in any order.
            received_at_ns: Optional monotonic receipt times keyed by ``id(event)``.
        """
        object.__setattr__(
            self,
            "_data",
            sorted(events, key=lambda event: event.get_timestamp()),
        )
        object.__setattr__(self, "_received_at_ns", dict(received_at_ns or {}))

    def get_events(self) -> list[UserInputEvent]:
        """Return a copy of the events, oldest first."""
        return list(self._data)

    def received_at_ns(self, event: UserInputEvent) -> int | None:
        """Return the optional monotonic receipt time for ``event``."""
        return self._received_at_ns.get(id(event))
