# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window for a run whose output is a file nobody is watching."""

from pathlib import Path

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.metrics_output_sink import MetricsOutputSink
from flashdreams.runtime_v2.mp4_output_sink import Mp4OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class Mp4ClientWindow(IClientWindow):
    """Write a session's results to an MP4 file, reporting no input.

    Every run goes through ``run_session``, and ``run_session`` drives a session
    against a window. A run writing a file has no client to press a key or to
    close the window, so this is the window it is given: input is always empty,
    and every result is encoded as it arrives.

    Two things a caller has to get right, because nothing here can:

    - Give it a session that finishes. Nothing here ever reports a close, so a
      run only ends when the session says it has generated everything, or when
      ``run_session`` is called directly with ``steps``.
    - Leave ``when_full`` alone. The default holds generation back until encoding
      has caught up, where ``WhenFull.DROP_OLDEST`` would quietly leave frames
      out of the file.

    What a run against this window generates is what a run against any other
    window would generate, since a session is given the same empty input every
    step. The one thing that differs is how often ``ISession.step_ui`` is called,
    which follows a wall clock, so a session generating differently because of
    what it did there generates differently here run to run. That is why it must
    not.

    Everything about the file itself — what an ``open`` accepts, what a failed
    encode raises — belongs to
    :class:`~flashdreams.runtime_v2.mp4_output_sink.Mp4OutputSink`, which this
    owns and delegates to.
    """

    def __init__(
        self, path: str | Path, *, stats_path: str | Path | None = None
    ) -> None:
        """
        Args:
            path: MP4 file to write. Parent directories are created.
            stats_path: JSON file to record what each step measured in, or
                ``None`` to measure nothing. A benchmark asks for one and gets
                both files from the one run.
        """
        self._path = Path(path)
        self._sinks: tuple[OutputSink, ...] = (Mp4OutputSink(path),)
        if stats_path is not None:
            self._sinks += (MetricsOutputSink(stats_path),)

    @property
    def path(self) -> Path:
        """File this writes, for a caller that has to say where the run went."""
        return self._path

    def get_user_input_events(self) -> UserInputEvents:
        """Report nothing, since there is no client to take input from.

        Returns:
            An empty batch, on every call.
        """
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        """Prepare to write a session's output."""
        for sink in self._sinks:
            sink.open(session_desc)

    def write(self, result: StepResult) -> None:
        """Encode one step's frames, and record what the step measured."""
        for sink in self._sinks:
            sink.write(result)

    def close(self) -> None:
        """Finish every file, which is what makes the MP4 playable.

        One file failing to finish does not stop the other being finished, and
        the first failure is the one raised.
        """
        failure: Exception | None = None
        for sink in self._sinks:
            try:
                sink.close()
            except Exception as error:
                failure = failure or error
        if failure is not None:
            raise failure
