# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application output delivery protocol."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult


@runtime_checkable
class OutputSink(Protocol):
    """Write to the sink while the sink is open.

    Created by the runtime, never by an application. A session returns results
    and the runtime writes them here, from one thread.
    """

    @abstractmethod
    def open(self, session_desc: SessionDesc) -> None:
        """Enable writing for a session that emits ``session_desc``-shaped results.

        Args:
            session_desc: Output description declared by the session. The sink
                configures itself from it and may reject a description it
                cannot present.

        Raises:
            ValueError: The description declares output this sink cannot
                consume, including audio for a video-only sink.
        """
        ...

    @abstractmethod
    def write(self, result: StepResult) -> None:
        """Consume one result.

        Called after :meth:`open` and before :meth:`close`.

        Args:
            result: Generated output for the completed step.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Finish pending writes and release resources."""
        ...


@runtime_checkable
class AbortableOutputSink(OutputSink, Protocol):
    """An output sink that can discard an incomplete session atomically.

    Both terminal operations must be idempotent. The runtime calls ``close``
    only for a successful run. It calls ``abort`` after cancellation or failure,
    including when ``open``, ``write``, or ``close`` raised partway through.

    Atomicity is per sink. A run with multiple abortable sinks does not provide
    a cross-sink prepare/commit transaction, so one sink may publish before a
    later sink's commit fails.
    """

    @abstractmethod
    def abort(self) -> None:
        """Release resources and discard all unpublished output."""
        ...


__all__ = ["AbortableOutputSink", "OutputSink"]
