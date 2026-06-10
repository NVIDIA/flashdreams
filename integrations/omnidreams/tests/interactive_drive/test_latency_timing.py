# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import time

from omnidreams.interactive_drive.runtime.timing import ChunkTimes


def _make_chunk(chunk_index: int = 0, chunk_size: int = 4) -> ChunkTimes:
    now = time.perf_counter()
    return ChunkTimes.create(
        chunk_index=chunk_index,
        input_sample_time=now,
        request_time=now,
        request_poses_ready_time=now + 0.001,
        intended_present_times=[
            now + 0.5 + frame * (1.0 / 30.0) for frame in range(chunk_size)
        ],
    )


def test_chunk_times_create_allocates_frame_times() -> None:
    chunk = _make_chunk(chunk_size=3)
    assert len(chunk.frames) == 3
    assert [frame.frame_index for frame in chunk.frames] == [0, 1, 2]
