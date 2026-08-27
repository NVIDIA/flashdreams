# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output of one generation step."""

from dataclasses import InitVar, dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Any

import torch
from torch import Tensor

from flashdreams.runtime_v2.cuda_utils import resolve_cuda_device
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


@dataclass(frozen=True, slots=True, init=False)
class StepResult:
    """Generated output returned by one inference step.

    A model loop returns a list of these, one per channel, and a UI loop returns
    one. Channels in the same list must agree about ``frame_count``.
    Use :meth:`replace` to copy a result without losing its CUDA readiness.
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

    output_ready_event: InitVar[torch.cuda.Event | None]
    """Init-only CUDA readiness override."""

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

    def __init__(
        self,
        step_index: int,
        output: Tensor,
        frame_count: int,
        output_layout: VideoTensorLayout,
        metrics: dict[str, float | int] | None = None,
        output_ready_event: torch.cuda.Event | None = None,
    ) -> None:
        """Create a result and capture CUDA producer-stream readiness.

        A missing or ``None`` ``output_ready_event`` records a new event on the
        current stream for CUDA output. A custom event describes output
        produced on another stream.

        Args:
            step_index: Zero-based index of the producing step.
            output: Generated output tensor.
            frame_count: Number of frames in ``output``.
            output_layout: Layout of ``output``.
            metrics: Optional measurements for this step.
            output_ready_event: Recorded producer event, or ``None`` to record
                the current stream automatically for CUDA output.
        """
        if output_ready_event is None and output.is_cuda:
            output_ready_event = torch.cuda.Event()
            output_ready_event.record(torch.cuda.current_stream(output.device))
        if metrics is None:
            metrics = {}

        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "output_layout", output_layout)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "_output_ready_event", output_ready_event)
        self._validate_output_ready_event()

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

    def replace(self, **changes: Any) -> "StepResult":
        """Return a changed result without losing CUDA readiness metadata.

        Readiness is preserved when ``output`` is unchanged. Replacing the
        output uses the constructor's automatic current-stream behavior unless
        ``output_ready_event`` is also supplied.
        """
        if "output_ready_event" not in changes:
            changes["output_ready_event"] = (
                None if "output" in changes else self._output_ready_event
            )
        return dataclass_replace(self, **changes)

    def _validate_output_ready_event(self) -> None:
        """Reject invalid CUDA readiness metadata."""
        output_ready_event = self._output_ready_event
        if output_ready_event is None:
            return
        if not isinstance(output_ready_event, torch.cuda.Event):
            raise TypeError("output_ready_event must be a CUDA event or None.")
        if not self.output.is_cuda:
            raise ValueError("CPU StepResult output cannot use a CUDA-ready event.")
        output_device = resolve_cuda_device(self.output.device)
        event_device = output_ready_event.device
        if event_device is None:
            raise ValueError("StepResult output-ready event must already be recorded.")
        event_device = resolve_cuda_device(event_device)
        if event_device != output_device:
            raise ValueError("StepResult output and output-ready event must match.")
