# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output of one generation step."""

from dataclasses import dataclass, field

from torch import Tensor

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


@dataclass(frozen=True, slots=True)
class StepResult:
    """Generated output returned by one inference step."""

    step_index: int
    """Zero-based index of the step that produced this result."""
    output: Tensor
    """Generated frames, laid out as ``output_layout`` says."""
    frame_count: int
    """Number of frames in ``output``."""
    output_layout: VideoTensorLayout
    """Layout of ``output``."""
    metrics: dict[str, float | int] = field(default_factory=dict)
    """Measurements for this step, such as timings, keyed by name."""
