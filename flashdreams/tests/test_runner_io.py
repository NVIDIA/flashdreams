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

"""CPU tests for shared runner I/O helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import flashdreams.infra.runner_io as runner_io
from flashdreams.infra.runner_io import (
    ensure_output_dir,
    load_first_frame_tensor,
    read_first_frame_rgb,
    read_video_fps,
    resolve_input_path,
    resolve_prompt_value,
    runner_artifact_path,
    runner_stats_path,
    video_tensor_to_uint8,
    write_runner_stats,
    write_video_tensor,
)

pytestmark = pytest.mark.ci_cpu


def test_resolve_prompt_value_reads_first_non_empty_line(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("\n  first prompt  \nsecond prompt\n")

    assert resolve_prompt_value("inline") == "inline"
    assert resolve_prompt_value(prompt_path) == "first prompt"


def test_resolve_prompt_value_rejects_empty_values(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("\n  \n")

    with pytest.raises(ValueError, match="has no non-empty lines"):
        resolve_prompt_value(prompt_path)
    with pytest.raises(ValueError, match="must be a non-empty string"):
        resolve_prompt_value("")


def test_resolve_input_path_passes_local_paths_through(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    assert resolve_input_path(tmp_path / "frame.png", cache_dir=cache_dir) == (
        tmp_path / "frame.png"
    )
    assert resolve_input_path("relative.mp4", cache_dir=cache_dir) == Path(
        "relative.mp4"
    )


def test_resolve_input_path_downloads_urls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Path, str | None, runner_io.InputAssetValidator | None]] = []

    def validator(path: Path) -> object:
        return path

    def fake_download_to_cache(
        url: str,
        *,
        cache_dir: Path,
        filename: str | None = None,
        validator: runner_io.InputAssetValidator | None = None,
    ) -> Path:
        calls.append((url, cache_dir, filename, validator))
        return cache_dir / (filename or "asset.bin")

    monkeypatch.setattr(runner_io, "_download_to_cache", fake_download_to_cache)

    resolved = resolve_input_path(
        "https://example.test/asset.png",
        cache_dir=tmp_path / "cache",
        filename="input.png",
        validator=validator,
    )

    assert resolved == tmp_path / "cache" / "input.png"
    assert calls == [
        ("https://example.test/asset.png", tmp_path / "cache", "input.png", validator)
    ]


def test_runner_artifact_and_stats_paths(tmp_path: Path) -> None:
    output_dir = ensure_output_dir(tmp_path / "nested")

    assert output_dir.is_dir()
    assert runner_artifact_path(output_dir, "demo-runner", "mp4") == (
        output_dir / "demo-runner.mp4"
    )
    assert runner_artifact_path(output_dir, "demo-runner", ".mp4") == (
        output_dir / "demo-runner.mp4"
    )
    assert runner_stats_path(output_dir, "demo-runner") == (
        output_dir / "stats_demo-runner.json"
    )


def test_write_runner_stats_matches_existing_json_format(tmp_path: Path) -> None:
    stats_path = write_runner_stats(
        tmp_path, "demo", [{"autoregressive_index": 0, "total_ms": 12.5}]
    )

    assert stats_path == tmp_path / "stats_demo.json"
    assert stats_path.read_text() == (
        '[\n  {\n    "autoregressive_index": 0,\n    "total_ms": 12.5\n  }\n]'
    )


def test_video_tensor_to_uint8_converts_tchw_layout() -> None:
    video = torch.tensor(
        [
            [
                [[-1.0, 0.0], [1.0, 2.0]],
                [[-2.0, 0.5], [0.0, 1.0]],
                [[1.0, -1.0], [0.0, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )

    frames = video_tensor_to_uint8(video, layout="tchw")

    assert frames.dtype == np.uint8
    assert frames.shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(
        frames,
        np.array(
            [[[[0, 0, 255], [127, 191, 0]], [[255, 127, 127], [255, 255, 127]]]],
            dtype=np.uint8,
        ),
    )


def test_video_tensor_to_uint8_converts_thwc_layout() -> None:
    video = torch.tensor(
        [
            [
                [[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0]],
                [[1.0, 1.0, -1.0], [2.0, -2.0, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )

    frames = video_tensor_to_uint8(video, layout="thwc")

    assert frames.dtype == np.uint8
    assert frames.shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(
        frames,
        np.array(
            [[[[0, 127, 255], [191, 63, 127]], [[255, 255, 0], [255, 0, 127]]]],
            dtype=np.uint8,
        ),
    )


def test_video_tensor_to_uint8_converts_bcthw_layout() -> None:
    video = torch.full((1, 3, 2, 4, 5), -1.0, dtype=torch.float32)

    frames = video_tensor_to_uint8(video, layout="bcthw")

    assert frames.shape == (2, 4, 5, 3)
    assert frames.dtype == np.uint8
    assert frames.max() == 0


def test_video_tensor_to_uint8_rejects_multi_batch_bcthw() -> None:
    video = torch.zeros((2, 3, 1, 4, 5), dtype=torch.float32)

    with pytest.raises(ValueError, match="expects a single batch element"):
        video_tensor_to_uint8(video, layout="bcthw")


def test_read_first_frame_rgb_rejects_empty_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_media = types.ModuleType("mediapy")

    def read_video(path: str) -> np.ndarray:
        return np.empty((0, 2, 2, 3), dtype=np.uint8)

    setattr(fake_media, "read_video", read_video)
    monkeypatch.setitem(sys.modules, "mediapy", fake_media)

    with pytest.raises(ValueError, match="video has no frames"):
        read_first_frame_rgb(Path("empty.mp4"))


def test_read_video_fps_uses_lazy_mediapy_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_media = types.ModuleType("mediapy")
    calls: list[str] = []

    class VideoMetadata:
        fps = 23.976

        @classmethod
        def from_path(cls, path: str) -> "VideoMetadata":
            calls.append(path)
            return cls()

    setattr(fake_media, "VideoMetadata", VideoMetadata)
    monkeypatch.setitem(sys.modules, "mediapy", fake_media)

    assert read_video_fps(Path("clip.mp4")) == pytest.approx(23.976)
    assert calls == ["clip.mp4"]


def test_write_video_tensor_lazy_imports_mediapy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, np.ndarray, int]] = []
    fake_media = types.ModuleType("mediapy")

    def write_video(path: str, frames: np.ndarray, *, fps: int) -> None:
        calls.append((path, frames, fps))

    setattr(fake_media, "write_video", write_video)
    monkeypatch.setitem(sys.modules, "mediapy", fake_media)

    out_path = tmp_path / "out.mp4"
    returned = write_video_tensor(
        torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        out_path,
        fps=16,
        layout="tchw",
    )

    assert returned == out_path
    assert calls[0][0] == str(out_path)
    assert calls[0][1].shape == (1, 2, 2, 3)
    assert calls[0][1].dtype == np.uint8
    assert calls[0][2] == 16


def test_load_first_frame_tensor_uses_requested_resize_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_media = types.ModuleType("mediapy")
    fake_cv2 = types.ModuleType("cv2")
    calls: dict[str, Any] = {}

    setattr(fake_cv2, "INTER_CUBIC", 2)
    setattr(fake_cv2, "INTER_NEAREST", 0)
    setattr(fake_cv2, "INTER_LINEAR", 1)
    setattr(fake_cv2, "INTER_AREA", 3)
    setattr(fake_cv2, "INTER_LANCZOS4", 4)

    def read_image(path: str) -> np.ndarray:
        calls["path"] = path
        return np.full((2, 3, 4), 127, dtype=np.uint8)

    def resize(image: np.ndarray, dsize: tuple[int, int], **kwargs: int) -> np.ndarray:
        calls["dsize"] = dsize
        calls["kwargs"] = kwargs
        width, height = dsize
        return np.full((height, width, 3), image[0, 0, 0], dtype=image.dtype)

    setattr(fake_media, "read_image", read_image)
    setattr(fake_cv2, "resize", resize)
    monkeypatch.setitem(sys.modules, "mediapy", fake_media)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    tensor = load_first_frame_tensor(
        Path("frame.png"),
        pixel_height=4,
        pixel_width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        interpolation="cubic",
    )

    assert calls == {
        "path": "frame.png",
        "dsize": (5, 4),
        "kwargs": {"interpolation": 2},
    }
    assert tensor.shape == (1, 3, 4, 5)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.full_like(tensor, 127.0 / 127.5 - 1.0))
