# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output sink writing what a session generates to an MP4 file."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_encoder import Mp4Encoder, result_to_rgb24_tensor


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
        self._encoder_path: Path | None = None
        self._canvas_size: tuple[int, int] | None = None
        self._presentation_size: tuple[int, int] | None = None

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
        self._encoder_path = self._path
        self._canvas_size = (session_desc.video_width, session_desc.video_height)
        self._presentation_size = (
            session_desc.video_width,
            session_desc.video_height,
        )

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Change the presentation size while retaining a growing MP4 canvas.

        Session-sized input is resampled on its current device to the requested
        size. Smaller presentations are placed at the canvas origin and padded
        on the right and bottom. When either dimension exceeds the canvas,
        frames encoded so far are migrated into a new canvas at the same origin
        before encoding continues.

        Args:
            new_window_size: Requested ``(width, height)`` in pixels.

        Raises:
            RuntimeError: Called before :meth:`open`, or video migration fails.
            ValueError: An expanded MP4 canvas dimension is odd and cannot be
                encoded as ``yuv420p``.
        """
        session_desc = self._session_desc
        encoder = self._encoder
        canvas_size = self._canvas_size
        presentation_size = self._presentation_size
        if session_desc is None or encoder is None or canvas_size is None:
            raise RuntimeError("Mp4OutputSink.open() must run before resizing.")
        width, height = new_window_size
        canvas_width = max(canvas_size[0], width)
        canvas_height = max(canvas_size[1], height)
        if (canvas_width, canvas_height) != canvas_size:
            if encoder.has_started:
                self._expand_canvas(canvas_width, canvas_height)
            else:
                encoder.request_frame_size(width=canvas_width, height=canvas_height)
            self._canvas_size = (canvas_width, canvas_height)
        self._presentation_size = (width, height)

    def write(self, result: StepResult) -> None:
        """Encode the frames in ``result``.

        Args:
            result: Generated output for the completed step.

        Raises:
            RuntimeError: Called before :meth:`open`, or the encoder stopped.
            ValueError: ``result`` does not match the description this sink was
                opened with.
        """
        session_desc = self._session_desc
        encoder = self._encoder
        canvas_size = self._canvas_size
        presentation_size = self._presentation_size
        if (
            session_desc is None
            or encoder is None
            or canvas_size is None
            or presentation_size is None
        ):
            raise RuntimeError("Mp4OutputSink.open() must run before write().")
        frames = result_to_rgb24_tensor(
            result,
            session_desc,
            presentation_size,
        )
        canvas_width, canvas_height = canvas_size
        if frames.shape[1:3] != (canvas_height, canvas_width):
            composited = torch.zeros(
                (len(frames), canvas_height, canvas_width, frames.shape[3]),
                dtype=frames.dtype,
                device=frames.device,
            )
            composited[:, : frames.shape[1], : frames.shape[2]] = frames
            frames = composited
        encoder.write(frames.cpu().numpy())

    def close(self) -> None:
        """Finish the file.

        Can be called on a sink that was never opened, or twice.

        Raises:
            RuntimeError: The encoder failed, so the file is unusable.
        """
        self._session_desc = None
        encoder = self._encoder
        encoder_path = self._encoder_path
        if encoder is None:
            return
        self._encoder = None
        self._encoder_path = None
        self._canvas_size = None
        self._presentation_size = None
        encoder.close()
        if encoder_path is not None and encoder_path != self._path:
            encoder_path.replace(self._path)

    def _expand_canvas(self, width: int, height: int) -> None:
        """Migrate encoded frames into a larger top-left-aligned canvas."""
        encoder = self._encoder
        encoder_path = self._encoder_path
        session_desc = self._session_desc
        if encoder is None or encoder_path is None or session_desc is None:
            raise RuntimeError("Mp4OutputSink.open() must run before resizing.")

        temporary_path = self._temporary_path()
        try:
            expanded_encoder = Mp4Encoder(
                temporary_path,
                width=width,
                height=height,
                frames_per_second=session_desc.frames_per_second_for_step,
            )
        except BaseException as error:
            try:
                temporary_path.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

        # Do not leave a closed encoder active: writing to it again would
        # overwrite the existing MP4.
        self._encoder = None
        try:
            encoder.close()
            if encoder_path != self._path:
                encoder_path.replace(self._path)
                self._encoder_path = self._path
            # When MPEG-DASH is supported, this rewrite should become a new
            # segment boundary carrying the expanded video descriptor.
            expanded_encoder.copy_from_mp4(self._path)
        except BaseException as error:
            try:
                expanded_encoder.close()
                temporary_path.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        self._encoder = expanded_encoder
        self._encoder_path = temporary_path

    def _temporary_path(self) -> Path:
        """Create a sibling MP4 path used until the expanded video is complete."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=self._path.parent,
            prefix=f".{self._path.stem}-",
            suffix=self._path.suffix,
            delete=False,
        ) as temporary_file:
            return Path(temporary_file.name)
