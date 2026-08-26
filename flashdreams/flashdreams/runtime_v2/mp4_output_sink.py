# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output sink writing what a session generates to an MP4 file."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

from flashdreams.api_v2.output_sink import AbortableOutputSink
from flashdreams.runtime_v2.mp4_audio import F32leAudioStager, Mp4AudioMuxer
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_encoder import Mp4Encoder, result_to_rgb24_frames


_LOGGER = logging.getLogger(__name__)


class Mp4OutputSink(AbortableOutputSink):
    """Encode results into an MP4 file.

    Each result is encoded as it arrives, and the run is bounded by whatever
    drives it: a file has no client to ask for it to end.

    Encoding belongs to :class:`Mp4Encoder`, which needs an ``ffmpeg``
    executable on ``PATH``. This class is the part that implements
    :class:`~flashdreams.api_v2.output_sink.OutputSink`.
    """

    def __init__(self, path: str | Path, *, audio_codec: str | None = None) -> None:
        """
        Args:
            path: File to write. Parent directories are created.
            audio_codec: Host FFmpeg audio encoder selected for synchronized
                output. An audio session is rejected unless this product-level
                choice is explicit.
        """
        self._path = Path(path)
        self._audio_codec = audio_codec
        self._session_desc: SessionDesc | None = None
        self._encoder: Mp4Encoder | None = None
        self._audio_stager: F32leAudioStager | None = None
        self._muxer: Mp4AudioMuxer | None = None
        self._staging_dir: Path | None = None
        self._staged_video_path: Path | None = None
        self._staged_output_path: Path | None = None
        self._frames_written = 0

    def open(self, session_desc: SessionDesc) -> None:
        """Prepare to encode a session's output.

        Encoding starts with the first result, so a run that generates nothing
        leaves no file behind.

        Args:
            session_desc: Output description declared by the session. Its frame
                size becomes the file's, and its ``frames_per_second_for_step``
                becomes the rate the file plays back at.

        Raises:
            ValueError: The frames are an odd number of pixels wide or high, or
                the session declares audio without an explicitly selected
                public codec.
        """
        if session_desc.audio_sample_rate is not None and self._audio_codec is None:
            raise ValueError(
                "Mp4OutputSink needs an explicitly selected audio codec for "
                "an audio session."
            )
        if self._session_desc is not None or self._staging_dir is not None:
            raise RuntimeError("Mp4OutputSink is already open.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self._path.name}.",
                dir=self._path.parent,
            )
        )
        staged_video_path = staging_dir / "video.mp4"
        staged_output_path = staging_dir / "output.mp4"
        encoder: Mp4Encoder | None = None
        audio_stager: F32leAudioStager | None = None
        try:
            encoder = Mp4Encoder(
                staged_video_path,
                width=session_desc.video_width,
                height=session_desc.video_height,
                frames_per_second=session_desc.frames_per_second_for_step,
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
                except BaseException as cleanup_error:
                    cast(Any, error).add_note(
                        f"Audio staging cleanup also failed: {cleanup_error!r}"
                    )
            if encoder is not None:
                try:
                    encoder.abort()
                except BaseException as cleanup_error:
                    cast(Any, error).add_note(
                        f"Video encoder cleanup also failed: {cleanup_error!r}"
                    )
            try:
                shutil.rmtree(staging_dir)
            except BaseException as cleanup_error:
                cast(Any, error).add_note(
                    f"Staging cleanup also failed: {cleanup_error!r}"
                )
            raise
        self._session_desc = session_desc
        self._encoder = encoder
        self._audio_stager = audio_stager
        self._staging_dir = staging_dir
        self._staged_video_path = staged_video_path
        self._staged_output_path = staged_output_path
        self._frames_written = 0

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
        frames = result_to_rgb24_frames(result, self._session_desc)
        if result.audio is not None:
            audio_stager = self._audio_stager
            if audio_stager is None:
                raise ValueError(
                    "A result carried audio for a session that declared none."
                )
            audio_stager.write(result.audio)
        self._encoder.write(frames)
        self._frames_written += len(frames)

    def close(self) -> None:
        """Commit a complete staged MP4 atomically.

        Can be called on a sink that was never opened, or twice. An empty
        session preserves any existing target and publishes nothing.

        Raises:
            RuntimeError: Encoding failed or did not produce its staged file.
            OSError: The staged file could not be published atomically.
        """
        if self._session_desc is None:
            return
        encoder = self._encoder
        if encoder is not None:
            encoder.close()
            self._encoder = None
        if self._frames_written == 0:
            if self._audio_stager is not None:
                self._audio_stager.abort()
                self._audio_stager = None
            self._discard_staging()
            self._clear_transaction()
            return
        staged_video_path = self._staged_video_path
        if staged_video_path is None or not staged_video_path.is_file():
            raise RuntimeError("MP4 encoding produced no staged video file.")
        staged_output_path = staged_video_path
        audio_stager = self._audio_stager
        if audio_stager is not None:
            session_desc = self._session_desc
            assert session_desc.audio_sample_rate is not None
            assert session_desc.audio_channels is not None
            assert self._audio_codec is not None
            expected_samples = round(
                self._frames_written
                * session_desc.audio_sample_rate
                / session_desc.frames_per_second_for_step
            )
            audio_stager.finish(expected_samples)
            self._audio_stager = None
            staged_output_path = self._staged_output_path
            if staged_output_path is None:
                raise RuntimeError("MP4 audio mux has no staged output path.")
            muxer = Mp4AudioMuxer(
                video_path=staged_video_path,
                audio_path=audio_stager.path,
                output_path=staged_output_path,
                sample_rate=session_desc.audio_sample_rate,
                channels=session_desc.audio_channels,
                audio_codec=self._audio_codec,
            )
            self._muxer = muxer
            muxer.close()
            self._muxer = None
            if not staged_output_path.is_file():
                raise RuntimeError("MP4 audio mux produced no staged output file.")
        os.replace(staged_output_path, self._path)
        self._complete_transaction()

    def abort(self) -> None:
        """Discard staged output without changing the target.

        This is idempotent and is valid after a partial ``open``, ``write``, or
        ``close`` failure.
        """
        failure: BaseException | None = None
        muxer = self._muxer
        self._muxer = None
        if muxer is not None:
            try:
                muxer.abort()
            except BaseException as error:
                failure = error
        encoder = self._encoder
        self._encoder = None
        if encoder is not None:
            try:
                encoder.abort()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    cast(Any, failure).add_note(
                        f"Video encoder cleanup also failed: {error!r}"
                    )
        audio_stager = self._audio_stager
        self._audio_stager = None
        if audio_stager is not None:
            try:
                audio_stager.abort()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    cast(Any, failure).add_note(
                        f"Audio staging cleanup also failed: {error!r}"
                    )
        cleanup_failed = False
        try:
            self._discard_staging()
        except BaseException as error:
            cleanup_failed = True
            if failure is None:
                failure = error
            else:
                cast(Any, failure).add_note(f"Staging cleanup also failed: {error!r}")
        self._clear_transaction(clear_staging=not cleanup_failed)
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

    def _clear_transaction(self, *, clear_staging: bool = True) -> None:
        self._session_desc = None
        self._encoder = None
        self._audio_stager = None
        self._muxer = None
        if clear_staging:
            self._staging_dir = None
            self._staged_video_path = None
            self._staged_output_path = None
        self._frames_written = 0
