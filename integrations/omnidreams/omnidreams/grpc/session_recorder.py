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

"""Session recording for gRPC requests and responses.

This module provides the SessionRecorder class which records all gRPC
requests and responses to a binary log file for later replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from loguru import logger
from omnidreams.grpc.protos import common_pb2, video_model_pb2
from omnidreams.grpc.recording_io import write_log_entry


class SessionRecorder:
    """Records gRPC requests/responses as length-prefixed LogEntry protos.

    Call ``record_*`` after each RPC and ``close()`` when finished.
    """

    def __init__(self, output_path: Path | str):
        """Create a recorder for ``output_path`` (file opened lazily on first write)."""
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: BinaryIO | None = None
        self._seq = 0
        logger.info(f"Recording session to: {self.output_path}")

    def _ensure_open(self) -> BinaryIO:
        """Lazily open the output file on first write."""
        if self._file is None:
            self._file = open(self.output_path, "wb")
        return self._file

    def record_start_session(
        self,
        request: video_model_pb2.SessionRequest,
        response: video_model_pb2.SessionId,
        timestamp_ns: int,
        duration_ns: int,
    ) -> None:
        """Record a start_session RPC call to the binary log.

        Args:
            request: The SessionRequest message.
            response: The SessionId response message.
            timestamp_ns: Wall clock time when request was received (nanoseconds).
            duration_ns: Time taken to process the request (nanoseconds).
        """
        entry = video_model_pb2.LogEntry(
            seq=self._seq,
            timestamp_ns=timestamp_ns,
            duration_ns=duration_ns,
        )
        entry.start_session.request.CopyFrom(request)
        entry.start_session.response.CopyFrom(response)

        write_log_entry(self._ensure_open(), entry)
        self._seq += 1
        logger.debug(f"Recorded start_session (seq={self._seq - 1})")

    def record_render_video_chunk(
        self,
        request: video_model_pb2.VideoChunkRequest,
        response: video_model_pb2.VideoChunkReturn,
        timestamp_ns: int,
        duration_ns: int,
    ) -> None:
        """Record a render_video_chunk RPC call to the binary log.

        Args:
            request: The VideoChunkRequest message.
            response: The VideoChunkReturn response message.
            timestamp_ns: Wall clock time when request was received (nanoseconds).
            duration_ns: Time taken to process the request (nanoseconds).
        """
        entry = video_model_pb2.LogEntry(
            seq=self._seq,
            timestamp_ns=timestamp_ns,
            duration_ns=duration_ns,
        )
        entry.render_video_chunk.request.CopyFrom(request)
        entry.render_video_chunk.response.CopyFrom(response)

        write_log_entry(self._ensure_open(), entry)
        self._seq += 1
        logger.debug(f"Recorded render_video_chunk (seq={self._seq - 1})")

    def record_close_session(
        self,
        request: video_model_pb2.SessionCloseRequest,
        response: common_pb2.Empty,
        timestamp_ns: int,
        duration_ns: int,
    ) -> None:
        """Record a close_session RPC call to the binary log.

        Args:
            request: The SessionCloseRequest message.
            response: The Empty response message.
            timestamp_ns: Wall clock time when request was received (nanoseconds).
            duration_ns: Time taken to process the request (nanoseconds).
        """
        entry = video_model_pb2.LogEntry(
            seq=self._seq,
            timestamp_ns=timestamp_ns,
            duration_ns=duration_ns,
        )
        entry.close_session.request.CopyFrom(request)
        entry.close_session.response.CopyFrom(response)

        write_log_entry(self._ensure_open(), entry)
        self._seq += 1
        logger.debug(f"Recorded close_session (seq={self._seq - 1})")

    def close(self) -> None:
        """Close the recording file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info(f"Recording saved: {self.output_path} ({self._seq} entries)")
