# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output of one generation step."""

from dataclasses import InitVar, dataclass, field

import torch
from torch import Tensor

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


@dataclass(frozen=True, slots=True, match_args=False)
class StepResult:
    """Generated output returned by one inference step.

    A model loop returns a list of these, one per channel. A UI loop returns
    zero or one in a list. Channels in the same list must agree about
    ``frame_count``.
    """

    step_index: int
    """Zero-based index of the step that produced this result."""

    output: InitVar[Tensor]
    """Generated frames accepted by the constructor."""

    frame_count: int
    """Number of frames in ``output``."""

    output_layout: VideoTensorLayout
    """Layout of ``output``."""

    metrics: dict[str, float | int] | None = field(default_factory=dict)
    """Measurements for this step, or ``None`` when profiling is disabled."""

    is_latent: bool = False
    """Whether ``output`` is a pre-decode latent chunk rather than RGB frames.

    A latent result is streamed to a token-streaming client that decodes it
    itself; it is never presented or encoded as pixels. ``read_output`` still
    returns it with the same CUDA-readiness guarantees, but RGB consumers --
    ``_frame_at`` and the video encoder -- reject it loudly.
    """

    _output: Tensor = field(init=False, repr=False)
    """Generated frames, laid out as ``output_layout`` says. Floating-point
    values are read as ``[-1, 1]`` and integer values as ``[0, 255]``."""

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

    def __post_init__(self, output: Tensor) -> None:
        """Capture CUDA readiness on the output's current producer stream."""
        object.__setattr__(self, "_output", output)
        if not output.is_cuda:
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(output.device))
        object.__setattr__(self, "_output_ready_event", event)

    def read_output(self, *, sync_with_event: bool = True) -> Tensor:
        """Return the output, optionally ordered before the current CUDA stream.

        Every CUDA read retains the output allocation for the current stream.
        When ``sync_with_event`` is true, the current stream also waits for the
        producer event without blocking the host. Disabling it skips only that
        event wait.

        Args:
            sync_with_event: Whether the current stream waits for the producer
                event before consuming the output.

        Returns:
            Generated frames laid out as :attr:`output_layout` specifies.
        """
        output = self._output
        if not output.is_cuda:
            return output
        consumer = torch.cuda.current_stream(output.device)
        if sync_with_event and self._output_ready_event is not None:
            consumer.wait_event(self._output_ready_event)
        output.record_stream(consumer)
        return output

    @classmethod
    def latent(
        cls,
        *,
        step_index: int,
        output: Tensor,
        frame_count: int,
        output_layout: VideoTensorLayout = VideoTensorLayout.bvtchw,
        metrics: dict[str, float | int] | None = None,
    ) -> "StepResult":
        """Build a latent (pre-decode) result for the token-streaming path.

        ``output`` is the clean latent chunk and ``output_layout`` names its
        axis order (``bvtchw`` = ``[B, V, T, Cl, Hl, Wl]``), not an RGB layout.
        The result is marked :attr:`is_latent`, so it streams to the token
        client and is never composited or encoded as pixels.
        """
        return cls(
            step_index=step_index,
            output=output,
            frame_count=frame_count,
            output_layout=output_layout,
            metrics=metrics,
            is_latent=True,
        )
