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

"""Length-prefixed protobuf stream I/O for session recording.

This module provides functions to read and write LogEntry messages
to binary files using a simple length-prefixed format:

    [4-byte big-endian size][message bytes][4-byte size][message bytes]...

This format allows streaming reads without loading the entire file.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from omnidreams.grpc.protos import video_model_pb2


def write_log_entry(file: BinaryIO, entry: video_model_pb2.LogEntry) -> None:
    """Write a length-prefixed LogEntry to a binary file.

    Args:
        file: Binary file handle open for writing.
        entry: LogEntry message to write.
    """
    message_bytes = entry.SerializeToString()
    size_prefix = struct.pack(">L", len(message_bytes))
    file.write(size_prefix + message_bytes)


def read_log_entries(
    path: Path | str,
    raise_on_malformed: bool = False,
) -> Iterator[video_model_pb2.LogEntry]:
    """Read LogEntry messages from a recording file.

    Args:
        path: Path to the recording file.
        raise_on_malformed: If True, raise IOError on malformed data.
            If False (default), log warning and stop iteration.

    Yields:
        LogEntry messages from the file.

    Raises:
        IOError: If raise_on_malformed=True and file is corrupted.
    """
    with open(path, "rb") as file:
        yield from read_log_entries_from_stream(file, raise_on_malformed)


def read_log_entries_from_stream(
    file: BinaryIO,
    raise_on_malformed: bool = False,
) -> Iterator[video_model_pb2.LogEntry]:
    """Read LogEntry messages from an open binary stream.

    Args:
        file: Binary file handle open for reading.
        raise_on_malformed: If True, raise IOError on malformed data.

    Yields:
        LogEntry messages from the stream.
    """
    from loguru import logger

    while True:
        size_prefix = file.read(4)
        if size_prefix == b"":  # EOF
            break

        if len(size_prefix) < 4:
            error = f"Malformed recording (incomplete size prefix: {len(size_prefix)} bytes)"
            if raise_on_malformed:
                raise IOError(error)
            logger.warning(error)
            break

        (message_size,) = struct.unpack(">L", size_prefix)
        message_bytes = file.read(message_size)

        if len(message_bytes) != message_size:
            error = f"Malformed recording (expected {message_size} bytes, got {len(message_bytes)})"
            if raise_on_malformed:
                raise IOError(error)
            logger.warning(error)
            break

        entry = video_model_pb2.LogEntry.FromString(message_bytes)
        yield entry


def count_log_entries(path: Path | str) -> int:
    """Count the number of LogEntry messages in a recording file.

    Args:
        path: Path to the recording file.

    Returns:
        Number of entries in the file.
    """
    count = 0
    for _ in read_log_entries(path):
        count += 1
    return count
