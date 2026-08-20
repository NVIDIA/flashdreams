# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output sink handing everything it is given to several others."""

from collections.abc import Sequence

from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult


class CompositeOutputSink(OutputSink):
    """Write every result to each of several sinks, in the order given.

    A run has one sink, and a benchmark run wants two things written: the video
    and what generating it cost. Rather than teach either sink about the other,
    this is the one the runner drives and they are what it drives.

    Results are passed on rather than copied, so a sink that keeps what it is
    given shares it with the others. Nothing here reads a result, so what
    arrives is what each sink sees.
    """

    def __init__(self, *sinks: OutputSink) -> None:
        """
        Args:
            sinks: Sinks to write to, in the order they should be written to.
        """
        self._sinks: Sequence[OutputSink] = sinks

    def open(self, session_desc: SessionDesc) -> None:
        """Open each sink for the same session.

        Args:
            session_desc: Output description declared by the session.

        Raises:
            Whatever a sink raises. The ones already opened are left to
            :meth:`close`, which the runner calls for a sink that failed to
            open.
        """
        for sink in self._sinks:
            sink.open(session_desc)

    def write(self, result: StepResult) -> None:
        """Hand one result to each sink.

        Args:
            result: Generated output for the completed step.

        Raises:
            Whatever a sink raises, which ends the run, since a run writing
            half of what it was asked to write has failed.
        """
        for sink in self._sinks:
            sink.write(result)

    def close(self) -> None:
        """Close every sink, whatever any of them does.

        A sink that fails to close does not stop the others being closed: for a
        file that is the difference between one unusable output and two. The
        first failure is what this raises, being the one that explains the run.
        """
        failure: Exception | None = None
        for sink in self._sinks:
            try:
                sink.close()
            except Exception as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure
