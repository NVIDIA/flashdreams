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

"""CPU tests for the builtin video output handler."""

from pathlib import Path

import pytest
import torch
from flashdreams.runtime.builtin.inference_output.frame_chunk import (
    FrameChunkOutput,
)
from flashdreams.runtime.builtin.inference_output.handler import (
    video_output_handler,
)
from flashdreams.runtime.builtin.inference_output.handler.video_output_handler import (
    VideoOutputHandler,
)
from torch import Tensor

pytestmark = pytest.mark.ci_cpu


def _frame_chunk(
    value: Tensor,
    *,
    start_timestamp: float = 0.0,
    fps: float = 24.0,
) -> FrameChunkOutput:
    """Build a frame chunk with presentation metadata."""
    return FrameChunkOutput(
        value=value,
        start_timestamp=start_timestamp,
        fps=fps,
    )


## Artifact writing


def test_video_output_handler_collects_tiles_and_writes_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify finish concatenates time and tiles views like OmniDreams."""
    written_videos: list[Tensor] = []
    written_paths: list[Path] = []
    written_options: list[tuple[float, str, str]] = []

    def write_video_tensor(
        video: Tensor,
        path: str | Path,
        *,
        fps: float,
        layout: str,
        install_hint: str,
    ) -> Path:
        written_videos.append(video)
        written_paths.append(Path(path))
        written_options.append((float(fps), layout, install_hint))
        return Path(path)

    monkeypatch.setattr(
        video_output_handler,
        "write_video_tensor",
        write_video_tensor,
    )
    first = torch.empty(1, 2, 2, 3, 2, 3)
    first[:, 0, 0].fill_(-1.0)
    first[:, 1, 0].fill_(-0.5)
    first[:, 0, 1].fill_(0.0)
    first[:, 1, 1].fill_(0.5)
    second = torch.empty(1, 2, 1, 3, 2, 3)
    second[:, 0, 0].fill_(0.75)
    second[:, 1, 0].fill_(1.0)
    artifact_path = tmp_path / "nested" / "artifact.mp4"
    handler = VideoOutputHandler(artifact_path)

    assert handler(_frame_chunk(first)) is None
    assert handler(_frame_chunk(second, start_timestamp=2.0 / 24.0)) is None
    assert handler.finish() == artifact_path
    assert handler.finish() == artifact_path

    # One write contains all temporal chunks. Each frame places view zero to the
    # left of view one, matching the OmniDreams runner's THWC canvas.
    assert written_paths == [artifact_path]
    assert len(written_videos) == 1
    canvas = written_videos[0]
    assert canvas.shape == (3, 2, 6, 3)
    torch.testing.assert_close(canvas[0, :, :3], torch.full((2, 3, 3), -1.0))
    torch.testing.assert_close(canvas[0, :, 3:], torch.full((2, 3, 3), -0.5))
    torch.testing.assert_close(canvas[1, :, :3], torch.zeros(2, 3, 3))
    torch.testing.assert_close(canvas[1, :, 3:], torch.full((2, 3, 3), 0.5))
    torch.testing.assert_close(canvas[2, :, :3], torch.full((2, 3, 3), 0.75))
    torch.testing.assert_close(canvas[2, :, 3:], torch.ones(2, 3, 3))
    assert written_options[0][:2] == (24.0, "thwc")
    assert written_options[0][2]


## Stream validation and lifecycle


def test_video_output_handler_rejects_empty_finish(tmp_path: Path) -> None:
    """Verify finish requires at least one received frame chunk."""
    handler = VideoOutputHandler(tmp_path / "empty.mp4")

    with pytest.raises(ValueError, match="without frame chunks"):
        handler.finish()


def test_video_output_handler_rejects_invalid_or_inconsistent_chunks(
    tmp_path: Path,
) -> None:
    """Verify stream shape, frame rate, and timestamp invariants."""
    handler = VideoOutputHandler(tmp_path / "invalid.mp4")

    with pytest.raises(ValueError, match="rank-6"):
        handler(_frame_chunk(torch.zeros(1, 1, 3, 2, 2)))

    handler(_frame_chunk(torch.zeros(1, 1, 2, 3, 2, 2)))
    with pytest.raises(ValueError, match="share.*dimensions"):
        handler(
            _frame_chunk(
                torch.zeros(1, 1, 1, 3, 3, 2),
                start_timestamp=2.0 / 24.0,
            )
        )
    with pytest.raises(ValueError, match="fps 24.0"):
        handler(
            _frame_chunk(
                torch.zeros(1, 1, 1, 3, 2, 2),
                start_timestamp=2.0 / 24.0,
                fps=30.0,
            )
        )
    with pytest.raises(ValueError, match="contiguous.*timestamp"):
        handler(
            _frame_chunk(
                torch.zeros(1, 1, 1, 3, 2, 2),
                start_timestamp=1.0,
            )
        )


def test_video_output_handler_rejects_chunks_after_finish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify a successfully finished handler cannot receive more chunks."""
    monkeypatch.setattr(
        video_output_handler,
        "write_video_tensor",
        lambda _video, path, **_kwargs: Path(path),
    )
    handler = VideoOutputHandler(tmp_path / "finished.mp4")
    chunk = _frame_chunk(torch.zeros(1, 1, 1, 3, 2, 2))
    handler(chunk)
    handler.finish()

    with pytest.raises(RuntimeError, match="after finish"):
        handler(chunk)
