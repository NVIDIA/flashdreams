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


"""HDMap video input handler for Omnidreams inference."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from flashdreams.infra.runner_io import (
    DEFAULT_RUNNER_INSTALL_HINT,
    read_video_rgb,
    resize_rgb_video,
    rgb_video_to_normalized_tensor,
)
from flashdreams.runtime.input_system import UserInputHandler
from omnidreams.runtime.inference_session import InferenceUserCondition


class HDMapInputHandler(UserInputHandler):
    """Iterate over an HDMap video as model-ready inference conditions.

    Args:
        hdmap_video_path: Path to an RGB HDMap video.
        get_num_frames: Optional function mapping an autoregressive step index to
            its required number of pixel frames. Pass the pipeline's
            ``get_num_frames`` method when feeding an Omnidreams inference session.
            When omitted, each call returns one frame.
        num_frames: Exact number of frames to return across all calls. The count
            must end on a chunk boundary. Mutually exclusive with ``num_chunks``.
        num_chunks: Exact number of complete conditions to return. Mutually
            exclusive with ``num_frames``.
        pixel_height: Optional resize target height. Must be supplied together
            with ``pixel_width``.
        pixel_width: Optional resize target width. Must be supplied together
            with ``pixel_height``.
        device: Device on which returned HDMap tensors are stored.
        dtype: Floating-point dtype used for normalized HDMap pixels.

    Raises:
        TypeError: ``dtype`` is not floating point or the frame-count provider
            returns a non-integer value while resolving a requested limit.
        ValueError: The decoded video is malformed, a limit is invalid, the exact
            frame count does not end on a chunk boundary, or the video is too short.
    """

    def __init__(
        self,
        hdmap_video_path: str | Path,
        *,
        get_num_frames: Callable[[int], int] | None = None,
        num_frames: int | None = None,
        num_chunks: int | None = None,
        pixel_height: int | None = None,
        pixel_width: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Load and normalize the HDMap video for iterative consumption."""
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point; got {dtype}")
        _validate_rollout_limits(num_frames=num_frames, num_chunks=num_chunks)
        _validate_resize_dimensions(
            pixel_height=pixel_height,
            pixel_width=pixel_width,
        )

        self.hdmap_video_path = Path(hdmap_video_path)
        video = read_video_rgb(
            self.hdmap_video_path,
            install_hint=DEFAULT_RUNNER_INSTALL_HINT,
        )
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(
                "expected an RGB HDMap video with shape [T, H, W, 3]; "
                f"got {tuple(video.shape)}"
            )
        if any(size <= 0 for size in video.shape):
            raise ValueError(
                f"HDMap video must have non-empty dimensions: {self.hdmap_video_path}"
            )
        if pixel_height is not None and pixel_width is not None:
            video = resize_rgb_video(
                video,
                pixel_height=pixel_height,
                pixel_width=pixel_width,
                install_hint=DEFAULT_RUNNER_INSTALL_HINT,
            )

        hdmap = rgb_video_to_normalized_tensor(
            video,
            device=torch.device(device),
            dtype=dtype,
        )
        # A path represents one rollout and one camera view. Each call slices the
        # temporal axis while retaining the condition's [B, V, T, C, H, W] layout.
        self._hdmap = hdmap.unsqueeze(0).unsqueeze(0)
        self._get_num_frames = get_num_frames or _one_frame_per_condition
        self._autoregressive_index = 0
        self._next_frame_index = 0
        self._num_chunks = self._resolve_num_chunks(
            num_frames=num_frames,
            num_chunks=num_chunks,
        )

    def __call__(self) -> InferenceUserCondition:
        """Return the next complete HDMap condition.

        Returns:
            Normalized HDMap pixels for the next inference step.

        Raises:
            TypeError: The frame-count provider does not return an integer.
            ValueError: The frame-count provider returns a non-positive count.
            StopIteration: The configured limit or available complete video chunks
                have been exhausted.
        """
        if (
            self._num_chunks is not None
            and self._autoregressive_index >= self._num_chunks
        ):
            raise StopIteration

        num_frames = self._validated_num_frames(self._autoregressive_index)
        end_frame_index = self._next_frame_index + num_frames
        if end_frame_index > self._hdmap.shape[2]:
            raise StopIteration

        condition = InferenceUserCondition(
            hdmap=self._hdmap[:, :, self._next_frame_index : end_frame_index]
        )
        self._next_frame_index = end_frame_index
        self._autoregressive_index += 1
        return condition

    def _validated_num_frames(self, autoregressive_index: int) -> int:
        """Return the validated frame count for one autoregressive step."""
        num_frames = self._get_num_frames(autoregressive_index)
        if isinstance(num_frames, bool) or not isinstance(num_frames, int):
            raise TypeError(
                "get_num_frames must return an integer; "
                f"got {num_frames!r} at autoregressive index "
                f"{autoregressive_index}"
            )
        if num_frames <= 0:
            raise ValueError(
                "get_num_frames must return a positive value; "
                f"got {num_frames} at autoregressive index "
                f"{autoregressive_index}"
            )
        return num_frames

    def _resolve_num_chunks(
        self,
        *,
        num_frames: int | None,
        num_chunks: int | None,
    ) -> int | None:
        """Resolve an optional exact frame or chunk limit to a chunk count."""
        available_num_frames = int(self._hdmap.shape[2])
        if num_frames is not None and num_frames > available_num_frames:
            raise ValueError(
                f"requested rollout requires {num_frames} HDMap frames; "
                f"video contains {available_num_frames}"
            )

        resolved_num_chunks = num_chunks
        required_num_frames = 0
        if num_frames is not None:
            resolved_num_chunks = 0
            while required_num_frames < num_frames:
                required_num_frames += self._validated_num_frames(resolved_num_chunks)
                resolved_num_chunks += 1
            if required_num_frames != num_frames:
                raise ValueError(
                    f"num_frames={num_frames} does not end on an autoregressive "
                    f"chunk boundary; the next boundary is {required_num_frames}"
                )
        elif num_chunks is not None:
            required_num_frames = sum(
                self._validated_num_frames(index) for index in range(num_chunks)
            )

        if required_num_frames > available_num_frames:
            raise ValueError(
                f"requested rollout requires {required_num_frames} HDMap frames; "
                f"video contains {available_num_frames}"
            )
        return resolved_num_chunks


def _validate_rollout_limits(*, num_frames: int | None, num_chunks: int | None) -> None:
    """Validate optional mutually exclusive rollout limits."""
    if num_frames is not None and num_chunks is not None:
        raise ValueError("num_frames and num_chunks are mutually exclusive")
    for name, value in (("num_frames", num_frames), ("num_chunks", num_chunks)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _validate_resize_dimensions(
    *,
    pixel_height: int | None,
    pixel_width: int | None,
) -> None:
    """Validate optional paired resize dimensions."""
    if (pixel_height is None) != (pixel_width is None):
        raise ValueError("pixel_height and pixel_width must be supplied together")
    for name, value in (
        ("pixel_height", pixel_height),
        ("pixel_width", pixel_width),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _one_frame_per_condition(_autoregressive_index: int) -> int:
    """Return the path-only handler's single-frame chunk size."""
    return 1


__all__ = ["HDMapInputHandler"]
