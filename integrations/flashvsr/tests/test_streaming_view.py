# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from flashdreams.serving.uplift.streaming_view import (
    StreamingViewer,
    _ViewerPlaybackChunk,
)

pytestmark = pytest.mark.ci_cpu


def _viewer(
    *,
    playback_fps: float = 30.0,
    max_fps: float = 60.0,
    chunk_queue_depth: int = 8,
) -> StreamingViewer:
    return StreamingViewer(
        host="127.0.0.1",
        port=0,
        jpeg_quality=90,
        jpeg_backend="pillow",
        chunk_queue_depth=chunk_queue_depth,
        max_fps=max_fps,
        playback_fps=playback_fps,
        frame_stride=1,
    )


def test_first_viewer_chunk_uses_playback_fps_not_infer_time() -> None:
    viewer = _viewer(playback_fps=30.0)
    item = _ViewerPlaybackChunk(elapsed_ms=140.0, ready_at=100.0, jpegs=[b"x"] * 8)

    interval_s = viewer._frame_interval_s(item, 8, queued_chunks=0)

    assert interval_s == pytest.approx(1.0 / 30.0)


def test_viewer_ignores_short_term_ready_cadence_without_backlog() -> None:
    viewer = _viewer(playback_fps=30.0)
    item = _ViewerPlaybackChunk(
        elapsed_ms=140.0,
        ready_at=100.0 + 8.0 / 45.0,
        jpegs=[b"x"] * 8,
    )

    interval_s = viewer._frame_interval_s(item, 8, queued_chunks=1)

    assert interval_s == pytest.approx(1.0 / 30.0)


def test_viewer_speeds_up_when_ready_queue_has_backlog() -> None:
    viewer = _viewer(playback_fps=30.0, max_fps=60.0, chunk_queue_depth=8)
    item = _ViewerPlaybackChunk(
        elapsed_ms=20.0,
        ready_at=100.0,
        jpegs=[b"x"] * 8,
    )

    interval_s = viewer._frame_interval_s(item, 8, queued_chunks=4)

    assert interval_s == pytest.approx(1.0 / 45.0)


def test_viewer_catchup_is_capped_by_max_fps() -> None:
    viewer = _viewer(playback_fps=30.0, max_fps=60.0, chunk_queue_depth=8)
    item = _ViewerPlaybackChunk(elapsed_ms=20.0, ready_at=100.0, jpegs=[b"x"] * 8)

    interval_s = viewer._frame_interval_s(item, 8, queued_chunks=8)

    assert interval_s == pytest.approx(1.0 / 60.0)
