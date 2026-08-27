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

"""Video frame normalization shared by file output backends."""

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_RGB_CHANNELS = 3
"""Colour channels an encoded frame carries."""


def result_to_rgb24_frames(
    result: StepResult, session_desc: SessionDesc
) -> npt.NDArray[np.uint8]:
    """Convert one result to the ``[T, H, W, C]`` uint8 frames an encoder reads.

    A pixel's value is read by dtype: a floating point tensor holds ``[-1, 1]``,
    which is what FlashDreams models emit, and an integer tensor holds raw
    ``0``-``255`` values. A result carrying one colour channel has it repeated
    across all three.

    Args:
        result: Generated output for one step.
        session_desc: Description the output is expected to match.

    Returns:
        Frames as uint8 RGB, oldest first.

    Raises:
        ValueError: ``result`` does not match ``session_desc``, carries more than
            one sequence of frames, or disagrees with itself over how many frames
            it carries.
    """
    if result.output_layout is not session_desc.output_layout:
        raise ValueError(
            f"Output was described as {session_desc.output_layout.value} but "
            f"arrived as {result.output_layout.value}."
        )
    frames = _to_tchw(result.output.detach(), result.output_layout)
    if frames.shape[0] != result.frame_count:
        raise ValueError(
            f"Result claims {result.frame_count} frames but carries {frames.shape[0]}."
        )
    if frames.shape[1] not in (1, _RGB_CHANNELS):
        raise ValueError(
            f"Expected one or {_RGB_CHANNELS} colour channels, got {frames.shape[1]}."
        )
    if frames.shape[2:] != (session_desc.video_height, session_desc.video_width):
        height, width = frames.shape[2:]
        described = f"{session_desc.video_width}x{session_desc.video_height}"
        raise ValueError(
            f"Output was described as {described} but arrived as {width}x{height}."
        )

    if frames.shape[1] == 1:
        frames = frames.repeat(1, _RGB_CHANNELS, 1, 1)
    if frames.is_floating_point():
        frames = ((frames.to(torch.float32).clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    frames = frames.clamp(0, 255).to(torch.uint8)
    return frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _to_tchw(output: Tensor, layout: VideoTensorLayout) -> Tensor:
    """Return ``output`` as ``[T, C, H, W]``, whatever ``layout`` it arrived in.

    Raises:
        ValueError: The tensor carries more than one sequence of frames, or does
            not have the shape its layout claims.
    """
    if layout is VideoTensorLayout.tchw:
        _require_dimensions(output, layout, 4)
        return output
    if layout is VideoTensorLayout.btchw:
        _require_dimensions(output, layout, 5)
        _require_one(output.shape[0], "batch", layout)
        return output[0]
    if layout is VideoTensorLayout.bcthw:
        _require_dimensions(output, layout, 5)
        _require_one(output.shape[0], "batch", layout)
        return output[0].permute(1, 0, 2, 3)
    if layout is VideoTensorLayout.bvtchw:
        _require_dimensions(output, layout, 6)
        _require_one(output.shape[0], "batch", layout)
        _require_one(output.shape[1], "view", layout)
        return output[0, 0]
    raise ValueError(f"Unsupported output layout: {layout.value}.")


def _require_dimensions(
    output: Tensor, layout: VideoTensorLayout, expected: int
) -> None:
    """Check that a tensor has as many dimensions as its layout names."""
    if output.ndim != expected:
        raise ValueError(
            f"Layout {layout.value} expects {expected} dimensions, got "
            f"{tuple(output.shape)}."
        )


def _require_one(size: int, name: str, layout: VideoTensorLayout) -> None:
    """Confirm a file carries exactly one sequence along ``name``."""
    if size != 1:
        raise ValueError(
            f"A video file holds one sequence of frames, so {layout.value} output "
            f"must have a {name} of 1, got {size}."
        )


__all__ = ["result_to_rgb24_frames"]
