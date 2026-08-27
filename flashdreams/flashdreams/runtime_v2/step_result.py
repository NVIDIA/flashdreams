# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output of one generation step."""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.runtime_v2.cuda_utils import resolve_cuda_device
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


@dataclass(frozen=True, slots=True)
class StepResult:
    """Generated output returned by one inference step.

    A model loop returns a list of these, one per channel, and a UI loop returns
    one. Channels in the same list must agree about ``frame_count``.
    """

    step_index: int
    """Zero-based index of the step that produced this result."""

    output: Tensor
    """Generated frames, laid out as ``output_layout`` says. Floating-point
    values are read as ``[-1, 1]`` and integer values as ``[0, 255]``."""

    frame_count: int
    """Number of frames in ``output``."""

    output_layout: VideoTensorLayout
    """Layout of ``output``."""

    metrics: dict[str, float | int] = field(default_factory=dict)
    """Measurements for this step, such as timings, keyed by name. Recorded only
    when a run asked for a metrics sink, and only from a model loop."""

    output_ready_event: torch.cuda.Event | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    """CUDA event recorded after ``output`` was fully produced.

    A consumer reading a CUDA output on another stream must wait for this event.
    The event deliberately does not expose or borrow the producer's stream.
    """

    def __post_init__(self) -> None:
        """Reject invalid CUDA readiness metadata."""
        output_ready_event = self.output_ready_event
        if output_ready_event is None:
            return
        if not self.output.is_cuda:
            raise ValueError("CPU StepResult output cannot use a CUDA-ready event.")
        output_device = resolve_cuda_device(self.output.device)
        event_device = output_ready_event.device
        if event_device is None:
            raise ValueError("StepResult output-ready event must already be recorded.")
        event_device = resolve_cuda_device(event_device)
        if event_device != output_device:
            raise ValueError("StepResult output and output-ready event must match.")
