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

"""CPU checks for packaged tutorials, playback timing, and decoder cleanup."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
from handdreams import tutorials
from handdreams.tutorials import TutorialVideo
from PIL import Image

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "name,count,digest",
    [
        (
            "press_triggers.mp4",
            166,
            "8c3068cb992c5e38248db553cb93cb26e0f93587d1d93126311b690418f33d3f",
        ),
        (
            "align_arms.mp4",
            122,
            "2b0c7380c4ebb6aa798aa052decfaa52f394313b88ddf453c5519d7a4a3acf1a",
        ),
    ],
)
def test_original_assets_animate_loop_and_package(name, count, digest) -> None:
    asset = tutorials._ASSET_DIR / name
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == digest
    project = Path(tutorials.__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(project.read_text())
    assert (
        "assets/tutorials/*.mp4"
        in metadata["tool"]["setuptools"]["package-data"]["handdreams"]
    )

    player = TutorialVideo()
    try:
        assert player._container is None
        first = player.frame(name, 120, 160, now=10.0)
        assert first is not None and first.mode == "RGB"
        assert first.size == (120, 144)
        assert player.frame(name, 120, 160, now=10.001) is first
        for index in range(1, count):
            frame = player.frame(name, 120, 160, now=10 + index / 30)
            assert frame is not None
            if index == 30:
                assert frame.tobytes() != first.tobytes()
        looped = player.frame(name, 120, 160, now=10 + count / 30 + 1e-6)
        assert looped is not None
        assert looped.tobytes() == first.tobytes()
    finally:
        player.close()
    assert player._container is None and player._current is None


def test_pts_timing_bounded_catchup_switch_and_close(monkeypatch) -> None:
    opened = []

    class Container:
        def __init__(self) -> None:
            self.streams = SimpleNamespace(video=[SimpleNamespace(average_rate=30)])
            self.decoded = 0
            self.closed = 0

        def decode(self, **kwargs):
            for index in range(100):
                self.decoded += 1
                image = Image.new("RGB", (10, 12), (index, 0, 0))
                yield SimpleNamespace(
                    time=50 + index / 10,
                    width=10,
                    height=12,
                    reformat=lambda image=image, **kwargs: SimpleNamespace(
                        to_image=lambda: image.resize(
                            (kwargs["width"], kwargs["height"])
                        )
                    ),
                )

        def close(self):
            self.closed += 1

    def open_video(*args):
        container = Container()
        opened.append(container)
        return container

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=open_video))
    player = TutorialVideo()
    first = player.frame("press_triggers.mp4", 10, 12, now=7.0)
    assert len(opened) == 1 and opened[0].decoded == 2
    assert player.frame("press_triggers.mp4", 10, 12, now=7.05) is first
    second = player.frame("press_triggers.mp4", 10, 12, now=7.1)
    assert second is not None
    assert second.getpixel((0, 0)) == (1, 0, 0)
    decoded = opened[0].decoded
    player.frame("press_triggers.mp4", 10, 12, now=500.0)
    assert opened[0].decoded - decoded <= 3
    resized = player.frame("press_triggers.mp4", 5, 6, now=500.0)
    assert resized is not None
    assert resized.size == (5, 6)

    player.frame("align_arms.mp4", 10, 12, now=500.0)
    assert opened[0].closed == 1 and len(opened) == 2
    player.close()
    player.close()
    assert opened[1].closed == 1
    assert player._frames is None and player._pending is None
    assert player.frame("align_arms.mp4", 10, 12, now=600.0) is not None
    assert len(opened) == 3
    player.close()


@pytest.mark.parametrize("failure", ["missing", "corrupt", "decoder"])
def test_unavailable_video_warns_once_and_stays_disabled(
    failure, tmp_path, monkeypatch, caplog
) -> None:
    if failure == "corrupt":
        (tmp_path / "press_triggers.mp4").write_bytes(b"not an MP4")
    elif failure == "decoder":
        monkeypatch.setitem(sys.modules, "av", None)
    player = TutorialVideo(asset_dir=tmp_path)
    assert player.frame("press_triggers.mp4", 120, 144, now=0.0) is None
    player.close()
    assert player.frame("press_triggers.mp4", 120, 144, now=1.0) is None
    assert len(caplog.records) == 1
    assert "press_triggers.mp4 unavailable" in caplog.text
    assert player._container is None


@pytest.mark.parametrize("failure", ["empty", "decode"])
def test_empty_or_broken_decoder_is_closed(failure, monkeypatch, caplog) -> None:
    closed = []

    def frames(**kwargs):
        if failure == "decode":
            raise RuntimeError("decode failed")
        return iter(())

    container = SimpleNamespace(
        streams=SimpleNamespace(video=[SimpleNamespace(average_rate=30)]),
        decode=frames,
        close=lambda: closed.append(True),
    )
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: container))
    player = TutorialVideo()
    assert player.frame("align_arms.mp4", 120, 144, now=0.0) is None
    assert player.frame("align_arms.mp4", 120, 144, now=1.0) is None
    player.close()
    assert closed == [True] and len(caplog.records) == 1


def test_tutorial_names_are_local_and_bounds_are_positive() -> None:
    player = TutorialVideo()
    with pytest.raises(ValueError, match="unknown tutorial"):
        player.frame("../press_triggers.mp4", 120, 144, now=0.0)
    with pytest.raises(ValueError, match="positive"):
        player.frame("press_triggers.mp4", 0, 144, now=0.0)
