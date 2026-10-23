# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application input handling protocol."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from flashdreams.runtime_v2.user_input_events import UserInputEvents


@runtime_checkable
class InputSource(Protocol):
    """Provide user input.

    The caller calls :meth:`get_user_input_events` when it needs the events that
    have arrived since it last asked.

    Created by the runtime, never by an application.
    """

    @abstractmethod
    def get_user_input_events(self) -> UserInputEvents:
        """Return the events that arrived since the previous call.

        Each event is returned exactly once, so an implementation buffers what
        arrives and clears that buffer here. Returning an event twice makes the
        caller apply it twice.

        Returns:
            Events in timestamp order, empty when nothing arrived.
        """
        ...


@runtime_checkable
class TimestampedInputSource(InputSource, Protocol):
    """Input source whose event timestamps share the runtime monotonic clock."""

    @property
    @abstractmethod
    def input_timestamp_origin_ns(self) -> int | None:
        """Return the runtime-clock origin for session-relative timestamps."""
        ...


__all__ = ["InputSource", "TimestampedInputSource"]
