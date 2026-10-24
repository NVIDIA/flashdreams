# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AV1 WebM recording through native encoder and muxer bindings."""

import math
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import numpy.typing as npt

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_tensor, rgb_to_yuv420


class WebmClientWindow(IClientWindow):
    """Record AV1 WebM with bounded encoder buffering and atomic publication.

    Call from the UI thread. Smaller presentations are padded at the canvas
    origin; expanding the canvas decodes and re-encodes earlier frames.
    The native dependencies load when the window is constructed.

    Value of Webm over avif is that we can trivially encode one frame at a time instead of
    saving all frames of a video in host memory for encoding in one go (streamable format).
    """

    def __init__(self, path: str | Path, *, native_rgb: bool = True) -> None:
        """Locate the native bindings and configure the destination.

        Args:
            path: Destination published when recording closes successfully.
            native_rgb: Convert and submit RGB in one native call. Disable to
                use the NumPy reference conversion.
        """
        try:
            import flashdreams_webm
        except ImportError as error:
            raise RuntimeError(
                "WebM native dependency flashdreams-webm is missing. Run "
                "uv sync --reinstall-package flashdreams-webm with a C/C++ "
                "toolchain (Visual Studio C++ Build Tools on Windows; "
                "YASM on x86 Linux)."
            ) from error
        self._native = flashdreams_webm
        self._native_rgb = native_rgb
        self._writer: flashdreams_webm.Writer | None = None
        self._path = Path(path)
        self._staged: Path | None = None
        self._session_desc: SessionDesc | None = None
        self._canvas_size = (0, 0)
        self._presentation_size = (0, 0)
        self._input_events: list[UserInputEvent] = []

    @property
    def path(self) -> Path:
        """Return the output path."""
        return self._path

    def get_user_input_events(self) -> UserInputEvents:
        """Return queued events once."""
        events, self._input_events = self._input_events, []
        return UserInputEvents(events)

    def put_user_input_event(self, event: UserInputEvent) -> None:
        self._input_events.append(event)

    def open(self, session_desc: SessionDesc) -> None:
        """Configure recording; an empty run creates no file."""
        if self._session_desc is not None:
            raise RuntimeError("WebmClientWindow is already open.")
        fps = session_desc.frames_per_second_for_step
        if not math.isfinite(fps) or not 0 < fps <= 240:
            raise ValueError("WebM frame rate must be finite and in (0, 240].")
        self._validate_size((session_desc.video_width, session_desc.video_height))
        self._session_desc = session_desc
        self._canvas_size = (session_desc.video_width, session_desc.video_height)
        self._presentation_size = self._canvas_size
        self._input_events = []

    @staticmethod
    def _validate_size(size: tuple[int, int]) -> None:
        if not (0 < size[0] <= 16384 and 0 < size[1] <= 8704):
            raise ValueError("WebM dimensions must be positive and at most 16384x8704.")

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Resize the presentation, expanding and migrating the canvas if needed."""
        if self._session_desc is None:
            raise RuntimeError("WebmClientWindow.open() must run before resizing.")
        self._validate_size(new_window_size)
        canvas = tuple(max(a, b) for a, b in zip(self._canvas_size, new_window_size))
        size = (canvas[0], canvas[1])
        if size != self._canvas_size:
            if self._writer is not None:
                self._expand_canvas(size)
            else:
                self._canvas_size = size
        self._presentation_size = new_window_size

    def write(self, result: StepResult) -> None:
        """Encode incoming frames, blocking when the encoder falls behind."""
        if self._session_desc is None:
            raise RuntimeError("WebmClientWindow.open() must run before write().")
        frames = result_to_rgb24_tensor(
            result, self._session_desc, self._presentation_size
        )
        if not len(frames):
            return
        try:
            if self._writer is None:
                self._start()
            for frame in frames.cpu():
                if self._native_rgb:
                    assert self._writer is not None
                    self._writer.write_rgb(frame.numpy())
                else:
                    self._write_planes(rgb_to_yuv420(frame.numpy()))
        except BaseException:
            self._discard()
            raise

    def _start(self) -> None:
        assert self._session_desc is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=self._path.parent, suffix=".webm", delete=False
        ) as file:
            self._staged = Path(file.name)
        fps = Fraction(
            str(self._session_desc.frames_per_second_for_step)
        ).limit_denominator(1_000_000)
        self._writer = self._native.Writer(
            str(self._staged),
            *self._canvas_size,
            fps.numerator,
            fps.denominator,
        )

    def _write_planes(self, planes: tuple[npt.NDArray[np.uint8], ...]) -> None:
        assert self._writer is not None
        width, height = self._writer.width, self._writer.height
        padded = []
        for channel, plane in enumerate(planes):
            h, w = (height, width) if channel == 0 else (height // 2, width // 2)
            padded.append(
                np.pad(
                    plane,
                    ((0, h - plane.shape[0]), (0, w - plane.shape[1])),
                    constant_values=16 if channel == 0 else 128,
                ).tobytes()
            )
        self._writer.write(b"".join(padded))

    def _expand_canvas(self, size: tuple[int, int]) -> None:
        assert self._writer is not None and self._staged is not None
        source = self._staged
        reader = None
        try:
            # ponytail: growth re-encodes history with quality loss; use segments for frequent resizing.
            self._writer.close()
            self._writer = None
            self._staged = None
            reader = self._native.Reader(str(source))
            self._canvas_size = size
            self._start()
            while (planes := reader.read()) is not None:
                self._write_planes(
                    tuple(
                        np.frombuffer(plane, dtype=np.uint8).reshape(
                            (reader.height, reader.width)
                            if index == 0
                            else ((reader.height + 1) // 2, (reader.width + 1) // 2)
                        )
                        for index, plane in enumerate(planes)
                    )
                )
        except BaseException:
            self._discard()
            raise
        finally:
            if reader is not None:
                reader.close()
            source.unlink(missing_ok=True)

    def close(self) -> None:
        """Flush delayed frames and atomically replace the destination on success."""
        try:
            if self._writer is not None:
                self._writer.close()
                assert self._staged is not None
                self._staged.replace(self._path)
        finally:
            self._discard()

    def _discard(self) -> None:
        self._session_desc = None
        writer, self._writer = self._writer, None
        try:
            if writer is not None:
                writer.abort()
        finally:
            if self._staged is not None:
                self._staged.unlink(missing_ok=True)
                self._staged = None
