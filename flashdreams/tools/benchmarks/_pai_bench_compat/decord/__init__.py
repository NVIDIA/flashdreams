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

"""Small OpenCV-backed subset of decord used by local PAI-Bench metrics."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class _Bridge:
    def __init__(self) -> None:
        self.mode = "native"

    def set_bridge(self, mode: str) -> None:
        if mode not in ("native", "torch"):
            raise ValueError(f"unsupported decord compatibility bridge: {mode}")
        self.mode = mode


bridge = _Bridge()


@dataclass(frozen=True)
class _CpuContext:
    device_id: int = 0


def cpu(device_id: int = 0) -> _CpuContext:
    return _CpuContext(device_id=device_id)


class _NativeBatch:
    def __init__(self, frames: np.ndarray) -> None:
        self._frames = frames

    def asnumpy(self) -> np.ndarray:
        return self._frames

    def __array__(self, dtype: Any = None) -> np.ndarray:
        if dtype is None:
            return self._frames
        return self._frames.astype(dtype)

    def __getitem__(self, index: int) -> "_NativeBatch":
        return _NativeBatch(self._frames[index])


class VideoReader:
    """Compatibility subset of ``decord.VideoReader`` for MP4 frame sampling."""

    def __init__(
        self,
        uri: str | Path,
        ctx: _CpuContext | None = None,
        width: int | None = None,
        height: int | None = None,
        num_threads: int = 0,
    ) -> None:
        del ctx, num_threads
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The local PAI-Bench decord compatibility shim requires "
                "opencv-python-headless."
            ) from exc

        self._cv2 = cv2
        self._path = str(uri)
        self._width = _positive_int_or_none(width)
        self._height = _positive_int_or_none(height)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"failed to open video: {self._path}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self._fps <= 0:
            self._fps = 30.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self._frame_count < 0:
            self._frame_count = 0

    def __len__(self) -> int:
        return self._frame_count

    def __getitem__(self, index: int) -> _NativeBatch | Any:
        return self._format_batch(np.expand_dims(self._read_frame(index), axis=0))[0]

    def __iter__(self) -> Iterator[_NativeBatch | Any]:
        for index in range(len(self)):
            yield self[index]

    def get_avg_fps(self) -> float:
        return self._fps

    def get_batch(self, indices: Iterable[int] | Sequence[int]) -> _NativeBatch | Any:
        frames = [self._read_frame(index) for index in _normalize_indices(indices)]
        if not frames:
            frames = [self._read_frame(0)]
        return self._format_batch(np.stack(frames, axis=0))

    def close(self) -> None:
        self._cap.release()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - destructors must not raise.
            pass

    def _read_frame(self, index: int) -> np.ndarray:
        if self._frame_count > 0:
            index = max(0, min(int(index), self._frame_count - 1))
        else:
            index = max(0, int(index))
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {index} from {self._path}")
        if self._width is not None and self._height is not None:
            frame = self._cv2.resize(
                frame,
                (self._width, self._height),
                interpolation=self._cv2.INTER_LINEAR,
            )
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def _format_batch(self, frames: np.ndarray) -> _NativeBatch | Any:
        frames = frames.astype(np.uint8, copy=False)
        if bridge.mode == "torch":
            import torch  # noqa: PLC0415

            return torch.from_numpy(frames)
        return _NativeBatch(frames)


def _normalize_indices(indices: Iterable[int] | Sequence[int]) -> list[int]:
    tolist = getattr(indices, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, Iterable):
            return [int(index) for index in converted]
    return [int(index) for index in indices]


def _positive_int_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None
