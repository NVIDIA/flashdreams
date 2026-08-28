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

    _output_ready_event: torch.cuda.Event | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    """CUDA event recorded after ``output`` was fully produced.

    A consumer reading a CUDA output on another stream must wait for this event.
    The event deliberately does not expose or borrow the producer's stream.
    """

    def __post_init__(self) -> None:
        """Capture CUDA readiness on the output's current producer stream."""
        if not self.output.is_cuda:
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.output.device))
        object.__setattr__(self, "_output_ready_event", event)

    def wait_for_output(self, stream: torch.cuda.Stream | None = None) -> None:
        """Order this result before a CUDA consumer stream without blocking.

        The method enqueues an event wait when readiness metadata is present and
        retains the output allocation for the consumer stream. CPU output is a
        no-op.

        Args:
            stream: Stream that will consume ``output``. ``None`` uses the
                current stream on the output device.

        Raises:
            ValueError: The consumer stream and output use different devices.
        """
        if not self.output.is_cuda:
            return
        consumer = stream
        if consumer is None:
            consumer = torch.cuda.current_stream(self.output.device)
        if resolve_cuda_device(consumer.device) != resolve_cuda_device(
            self.output.device
        ):
            raise ValueError("StepResult output and consumer stream must match.")
        if self._output_ready_event is not None:
            consumer.wait_event(self._output_ready_event)
        self.output.record_stream(consumer)
