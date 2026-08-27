# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transactional native WebM output for v2 sessions."""

import importlib
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import numpy.typing as npt
from numpy import uint8

from flashdreams.api_v2.output_sink import AbortableOutputSink
from flashdreams.core.exceptions import add_exception_note
from flashdreams.runtime_v2.mp4_audio import F32leAudioStager
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_frames

_LOGGER = logging.getLogger(__name__)


class _NativeWriter(Protocol):
    """Native writer surface consumed by the transactional sink."""

    def write_video(self, frames: npt.NDArray[uint8]) -> None:
        """Encode contiguous RGB24 frames into the private packet spool."""

    def close(self, audio_path: str | Path | None = None) -> None:
        """Mux and finalize the staged WebM file."""

    def abort(self) -> None:
        """Discard native staging state."""


class _WebmBackend(Protocol):
    """Optional companion module surface loaded for native WebM mode."""

    WebmWriter: Callable[
        [str | Path, int, int, int, str, int, int],
        _NativeWriter,
    ]
    """Native incremental WebM writer type."""

    def select_video_codec(self) -> str:
        """Return the cached or benchmark-selected VPx codec."""


class WebmOutputSink(AbortableOutputSink):
    """Encode results into a native VP8/VP9 and Opus WebM file.

    Native video packets and normalized PCM remain below a unique sibling
    staging directory until the companion extension has finalized the WebM
    container. The requested target is replaced only after that succeeds.
    """

    def __init__(self, path: str | Path) -> None:
        """
        Args:
            path: WebM file to write. Parent directories are created.

        Raises:
            RuntimeError: The optional ``flashdreams-webm`` wheel is unavailable
                or its machine-local codec benchmark fails.
        """
        self._path = Path(path)
        self._backend = _load_webm_backend()
        self._codec = self._backend.select_video_codec()
        self._session_desc: SessionDesc | None = None
        self._writer: _NativeWriter | None = None
        self._audio_stager: F32leAudioStager | None = None
        self._staging_dir: Path | None = None
        self._staged_output_path: Path | None = None
        self._frames_written = 0

    @property
    def path(self) -> Path:
        """Return the requested output path."""
        return self._path

    @property
    def codec(self) -> str:
        """Return the benchmark-selected native video codec."""
        return self._codec

    def open(self, session_desc: SessionDesc) -> None:
        """Create private native video and PCM staging for one session.

        Args:
            session_desc: Declared frame size, rate, and optional audio format.

        Raises:
            ValueError: The frame or audio format is unsupported by WebM.
            RuntimeError: The native writer cannot initialize.
        """
        if self._session_desc is not None or self._staging_dir is not None:
            raise RuntimeError("WebmOutputSink is already open.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self._path.name}.",
                dir=self._path.parent,
            )
        )
        staged_output_path = staging_dir / "output.webm"
        writer: _NativeWriter | None = None
        audio_stager: F32leAudioStager | None = None
        try:
            writer = self._backend.WebmWriter(
                staged_output_path,
                session_desc.video_width,
                session_desc.video_height,
                session_desc.frames_per_second_for_step,
                self._codec,
                session_desc.audio_sample_rate or 0,
                session_desc.audio_channels or 0,
            )
            if session_desc.audio_sample_rate is not None:
                assert session_desc.audio_channels is not None
                audio_stager = F32leAudioStager(
                    staging_dir / "audio.f32le",
                    sample_rate=session_desc.audio_sample_rate,
                    channels=session_desc.audio_channels,
                )
        except BaseException as error:
            if audio_stager is not None:
                try:
                    audio_stager.abort()
                except BaseException as cleanup_error:  # noqa: BLE001
                    add_exception_note(
                        error, f"Audio staging cleanup also failed: {cleanup_error!r}"
                    )
            if writer is not None:
                try:
                    writer.abort()
                except BaseException as cleanup_error:  # noqa: BLE001
                    add_exception_note(
                        error, f"Native writer cleanup also failed: {cleanup_error!r}"
                    )
            try:
                shutil.rmtree(staging_dir)
            except BaseException as cleanup_error:  # noqa: BLE001
                add_exception_note(
                    error, f"Staging cleanup also failed: {cleanup_error!r}"
                )
            raise
        self._session_desc = session_desc
        self._writer = writer
        self._audio_stager = audio_stager
        self._staging_dir = staging_dir
        self._staged_output_path = staged_output_path
        self._frames_written = 0

    def write(self, result: StepResult) -> None:
        """Encode one generated result and stage any synchronized PCM.

        Args:
            result: Generated output for the completed step.

        Raises:
            RuntimeError: Called before :meth:`open`, or native encoding fails.
            ValueError: ``result`` disagrees with the declared session format.
        """
        if self._session_desc is None or self._writer is None:
            raise RuntimeError("WebmOutputSink.open() must run before write().")
        frames = result_to_rgb24_frames(result, self._session_desc)
        if result.audio is not None:
            audio_stager = self._audio_stager
            if audio_stager is None:
                raise ValueError(
                    "A result carried audio for a session that declared none."
                )
            audio_stager.write(result.audio)
        self._writer.write_video(frames)
        self._frames_written += len(frames)

    def close(self) -> None:
        """Finalize and atomically publish a complete staged WebM file.

        Can be called before :meth:`open` or after a completed close. An empty
        session preserves any existing target and publishes nothing.

        Raises:
            RuntimeError: Native encoding or muxing fails to produce output.
            OSError: The staged file cannot be published atomically.
        """
        if self._session_desc is None:
            return
        writer = self._writer
        if writer is None:
            raise RuntimeError("WebM transaction has no native writer.")
        if self._frames_written == 0:
            writer.abort()
            self._writer = None
            if self._audio_stager is not None:
                self._audio_stager.abort()
                self._audio_stager = None
            self._discard_staging()
            self._clear_transaction()
            return

        audio_path: Path | None = None
        audio_stager = self._audio_stager
        if audio_stager is not None:
            session_desc = self._session_desc
            assert session_desc.audio_sample_rate is not None
            expected_samples = round(
                self._frames_written
                * session_desc.audio_sample_rate
                / session_desc.frames_per_second_for_step
            )
            audio_stager.finish(expected_samples)
            audio_path = audio_stager.path
            self._audio_stager = None
        writer.close(audio_path)
        self._writer = None

        staged_output_path = self._staged_output_path
        if staged_output_path is None or not staged_output_path.is_file():
            raise RuntimeError("Native WebM encoding produced no staged output file.")
        os.replace(staged_output_path, self._path)
        self._complete_transaction()

    def abort(self) -> None:
        """Discard staged output without changing the requested target.

        This operation is idempotent. Ownership is retained when a component
        cannot clean up so a later abort can retry before removing its staging.
        """
        failure: BaseException | None = None
        writer = self._writer
        if writer is not None:
            try:
                writer.abort()
            except BaseException as error:  # noqa: BLE001
                failure = error
            else:
                self._writer = None
        audio_stager = self._audio_stager
        if audio_stager is not None:
            try:
                audio_stager.abort()
            except BaseException as error:  # noqa: BLE001
                if failure is None:
                    failure = error
                else:
                    add_exception_note(
                        failure, f"Audio staging cleanup also failed: {error!r}"
                    )
            else:
                self._audio_stager = None
        if failure is None:
            try:
                self._discard_staging()
            except BaseException as error:  # noqa: BLE001
                failure = error
        if failure is None:
            self._clear_transaction()
        if failure is not None:
            raise failure

    def _discard_staging(self) -> None:
        staging_dir = self._staging_dir
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)

    def _complete_transaction(self) -> None:
        staging_dir = self._staging_dir
        if staging_dir is not None:
            try:
                shutil.rmtree(staging_dir)
            except OSError as error:
                _LOGGER.warning(
                    "Committed %s but could not remove staging directory %s: %s",
                    self._path,
                    staging_dir,
                    error,
                )
        self._clear_transaction()

    def _clear_transaction(self) -> None:
        self._session_desc = None
        self._writer = None
        self._audio_stager = None
        self._staging_dir = None
        self._staged_output_path = None
        self._frames_written = 0


def _load_webm_backend() -> _WebmBackend:
    """Load the optional native companion with an installation-focused error."""
    try:
        module: ModuleType = importlib.import_module("flashdreams_webm")
    except ImportError as error:
        raise RuntimeError(
            "Native WebM output needs the optional flashdreams-webm companion; "
            "install it with `pip install 'flashdreams[webm]'`."
        ) from error
    return cast(_WebmBackend, module)


__all__ = ["WebmOutputSink"]
