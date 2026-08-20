# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window for a run whose output is a file nobody is watching."""

from pathlib import Path

from flashdreams.api_v2.client_window import IClientWindow
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

    - Pass ``steps``. Nothing here ever reports a close, so a run left to end on
      its own never ends.
    - Leave ``when_full`` alone. The default holds generation back until encoding
      has caught up, where ``WhenFull.DROP_OLDEST`` would quietly leave frames
      out of the file.

    What a run against this window generates is what a run against any other
    window would generate, since a session is given the same empty input every
    step. The one thing that differs is how often ``ISession.step_ui`` is called,
    which follows a wall clock, so a session generating differently because of
    what it did there generates differently here run to run. That is why it must
    not.
    """

    def __init__(self, path: str | Path) -> None:
        """
        Args:
            path: MP4 file to write. Parent directories are created. Encoding
                happens on the run's I/O thread, so nothing here needs locking.
        """
        self._sink = Mp4OutputSink(path)

    def get_user_input_events(self) -> UserInputEvents:
        """Report nothing, since there is no client to take input from.

        Returns:
            An empty batch, on every call.
        """
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        """Prepare to encode a session's output.

        Args:
            session_desc: Output description declared by the session. Its frame
                size becomes the file's, and its ``frames_per_second_for_step``
                becomes the rate the file plays back at.

        Raises:
            ValueError: The frames are an odd number of pixels wide or high,
                which this cannot encode. The run ends before generating
                anything.
        """
        self._sink.open(session_desc)

    def write(self, result: StepResult) -> None:
        """Encode one step's frames.

        Args:
            result: Generated output for the completed step.

        Raises:
            RuntimeError: The encoder stopped, which ends the run: half of what
                was asked for is not a run that succeeded.
            ValueError: ``result`` does not match the description the session
                declared.
        """
        self._sink.write(result)

    def close(self) -> None:
        """Finish the file, which is what makes it playable.

        Raises:
            RuntimeError: The encoder failed, so the file is unusable, and the
                run reports it rather than leaving it to be discovered.
        """
        self._sink.close()
