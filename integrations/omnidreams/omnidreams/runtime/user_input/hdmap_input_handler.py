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

"""HDMap video input handler for OmniDreams inference."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from flashdreams.infra.runner_io import (
    DEFAULT_RUNNER_INSTALL_HINT,
    read_video_rgb,
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
            get_num_frames method when feeding an OmniDreams inference session.
            When omitted, each call returns one frame.
        device: Device on which returned HDMap tensors are stored.
        dtype: Floating-point dtype used for normalized HDMap pixels.

    Raises:
        TypeError: If dtype is not a floating-point dtype.
        ValueError: If the decoded video is empty or malformed.
    """

    def __init__(
        self,
        hdmap_video_path: str | Path,
        *,
        get_num_frames: Callable[[int], int] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Load and normalize the HDMap video for iterative consumption."""
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point; got {dtype}")

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

    def __call__(self) -> InferenceUserCondition:
        """Return the next complete HDMap condition.

        Returns:
            Normalized HDMap pixels for the next inference step.

        Raises:
            TypeError: If the frame-count provider does not return an integer.
            ValueError: If the frame-count provider returns a non-positive count.
            StopIteration: If no complete condition remains in the video.
        """
        num_frames = self._get_num_frames(self._autoregressive_index)
        if isinstance(num_frames, bool) or not isinstance(num_frames, int):
            raise TypeError(
                "get_num_frames must return an integer; "
                f"got {num_frames!r} at autoregressive index "
                f"{self._autoregressive_index}"
            )
        if num_frames <= 0:
            raise ValueError(
                "get_num_frames must return a positive value; "
                f"got {num_frames} at autoregressive index "
                f"{self._autoregressive_index}"
            )

        end_frame_index = self._next_frame_index + num_frames
        if end_frame_index > self._hdmap.shape[2]:
            raise StopIteration

        condition = InferenceUserCondition(
            hdmap=self._hdmap[:, :, self._next_frame_index : end_frame_index]
        )
        self._next_frame_index = end_frame_index
        self._autoregressive_index += 1
        return condition


def _one_frame_per_condition(_autoregressive_index: int) -> int:
    """Return the path-only handler's single-frame chunk size."""
    return 1


__all__ = ["HDMapInputHandler"]
