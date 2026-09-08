# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output sink writing what a session generates to an MP4 file."""

from pathlib import Path

from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_encoder import Mp4Encoder, result_to_rgb24_frames


class Mp4OutputSink(OutputSink):
    """Encode results into an MP4 file.

    Each result is encoded as it arrives, and the run is bounded by whatever
    drives it: a file has no client to ask for it to end.

    Encoding belongs to :class:`Mp4Encoder`, which needs an ``ffmpeg``
    executable on ``PATH``. This class is the part that implements
    :class:`~flashdreams.api_v2.output_sink.OutputSink`.
    """

    def __init__(self, path: str | Path) -> None:
        """
        Args:
            path: File to write. Parent directories are created.
        """
        self._path = Path(path)
        self._session_desc: SessionDesc | None = None
        self._encoder: Mp4Encoder | None = None

    def open(self, session_desc: SessionDesc) -> None:
        """Prepare to encode a session's output.

        Encoding starts with the first result, so a run that generates nothing
        leaves no file behind.

        Args:
            session_desc: Output description declared by the session. Its frame
                size becomes the file's, and its ``frames_per_second_for_step``
                becomes the rate the file plays back at.

        Raises:
            ValueError: The frames are an odd number of pixels wide or high,
                which this cannot encode.
        """
        self._session_desc = session_desc
        self._encoder = Mp4Encoder(
            self._path,
            width=session_desc.video_width,
            height=session_desc.video_height,
            frames_per_second=session_desc.frames_per_second_for_step,
        )

    def write(self, result: StepResult) -> None:
        """Encode the frames in ``result``.

        Args:
            result: Generated output for the completed step.

        Raises:
            RuntimeError: Called before :meth:`open`, or the encoder stopped.
            ValueError: ``result`` does not match the description this sink was
                opened with.
        """
        if self._session_desc is None or self._encoder is None:
            raise RuntimeError("Mp4OutputSink.open() must run before write().")
        self._encoder.write(result_to_rgb24_frames(result, self._session_desc))

    def close(self) -> None:
        """Finish the file.

        Can be called on a sink that was never opened, or twice.

        Raises:
            RuntimeError: The encoder failed, so the file is unusable.
        """
        self._session_desc = None
        encoder = self._encoder
        if encoder is None:
            return
        self._encoder = None
        encoder.close()
