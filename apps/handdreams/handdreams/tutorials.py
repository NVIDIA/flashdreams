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

"""Bounded CPU playback of the bundled controller tutorials."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from av import VideoFrame
    from av.container import InputContainer

_ASSET_DIR = Path(__file__).parent / "assets" / "tutorials"
_NAMES = frozenset({"press_triggers.mp4", "align_arms.mp4"})
_MAX_DECODE_PER_TICK = 3
"""Limit catch-up work so a delayed UI tick cannot decode the entire clip."""


class TutorialVideo:
    """UI-owned local decoder retaining the current frame and one lookahead."""

    def __init__(self, asset_dir: Path = _ASSET_DIR) -> None:
        self._asset_dir = asset_dir
        self._failed: set[str] = set()
        self._container: InputContainer | None = None
        self._frames: Iterator[VideoFrame] | None = None
        self._name: str | None = None
        self._current: VideoFrame | None = None
        self._pending: VideoFrame | None = None
        self._image: Image.Image | None = None
        self._size = (0, 0)
        self._started_at = 0.0
        self._origin: float | None = None
        self._current_time = 0.0
        self._pending_time = 0.0
        self._index = 0
        self._interval = 1 / 30

    def frame(
        self, name: str, width: int, height: int, now: float
    ) -> Image.Image | None:
        """Return the current RGB image fitted inside the requested bounds.

        ``now`` is a monotonic timestamp. Missing or corrupt clips warn once
        per player and return ``None`` so the setup instructions remain usable.
        Call ``close`` when leaving guided setup or replacing the UI session.
        """
        if name not in _NAMES:
            raise ValueError(f"unknown tutorial: {name!r}")
        if min(width, height) < 1:
            raise ValueError("tutorial bounds must be positive")
        if name != self._name:
            self.close()
        if name in self._failed:
            return None
        try:
            if self._container is None:
                import av

                self._container = av.open(str(self._asset_dir / name))
                stream = self._container.streams.video[0]
                stream.thread_count = 1
                self._interval = 1 / float(stream.average_rate or 30)
                self._frames = iter(self._container.decode(video=0))
                self._name = name
                self._started_at = now
            assert self._frames is not None
            for _ in range(_MAX_DECODE_PER_TICK):
                if self._pending is None:
                    self._pending = next(self._frames, None)
                    if self._pending is None:
                        if self._index == 0:
                            raise ValueError("tutorial contains no video frames")
                        if now < self._started_at + self._current_time + self._interval:
                            break
                        self._container.seek(0)
                        self._frames = iter(self._container.decode(video=0))
                        self._origin = None
                        self._index = 0
                        self._started_at = now
                        continue
                    timestamp = self._pending.time
                    if self._origin is None:
                        self._origin = float(timestamp or 0.0)
                    self._pending_time = (
                        float(timestamp) - self._origin
                        if timestamp is not None
                        else self._index * self._interval
                    )
                    self._index += 1
                if self._pending_time > now - self._started_at + 1e-9:
                    break
                self._current = self._pending
                self._current_time = self._pending_time
                self._pending = None
                self._image = None
            else:
                # ponytail: after three frames, resume from here instead of
                # catching up indefinitely; longer clips need async playback.
                self._started_at = max(self._started_at, now - self._current_time)
            if self._current is None:
                return None
            if self._image is None or self._size != (width, height):
                scale = min(width / self._current.width, height / self._current.height)
                self._image = self._current.reformat(
                    width=max(1, round(self._current.width * scale)),
                    height=max(1, round(self._current.height * scale)),
                    format="rgb24",
                ).to_image()
                self._size = (width, height)
            return self._image
        except Exception as error:  # noqa: BLE001 - setup remains usable without video
            self._failed.add(name)
            self.close()
            logging.getLogger(__name__).warning(
                "Tutorial %s unavailable: %s", name, error
            )
            return None

    def close(self) -> None:
        """Release playback resources, retaining failures to avoid repeated retries."""
        container, self._container = self._container, None
        self._frames = None
        self._name = None
        self._current = self._pending = None
        self._image = None
        self._origin = None
        self._index = 0
        self._current_time = self._pending_time = 0.0
        if container is not None:
            container.close()
