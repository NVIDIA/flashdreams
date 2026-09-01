# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output sink writing generated frames as PNG files."""

from pathlib import Path

from PIL import Image

from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_frames


class PngOutputSink(OutputSink):
    """Write a single frame or numbered frame sequence as PNG."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._session_desc: SessionDesc | None = None
        self._frame_index = 0

    def open(self, session_desc: SessionDesc) -> None:
        """Remember the expected output shape and reset numbering."""
        self._session_desc = session_desc
        self._frame_index = 0

    def _frame_path(self) -> Path:
        if self._frame_index == 0:
            return self.path
        return self.path.with_name(
            f"{self.path.stem}-{self._frame_index:05d}{self.path.suffix}"
        )

    def write(self, result: StepResult) -> None:
        """Write every frame in ``result`` as lossless RGB PNG."""
        if self._session_desc is None:
            raise RuntimeError("PngOutputSink.open() must run before write().")
        frames = result_to_rgb24_frames(result, self._session_desc)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            Image.fromarray(frame, mode="RGB").save(self._frame_path())
            self._frame_index += 1

    def close(self) -> None:
        """Close the sink; PNG writes are synchronous."""
        self._session_desc = None


__all__ = ["PngOutputSink"]
