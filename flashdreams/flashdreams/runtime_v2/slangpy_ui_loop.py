# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UI loop drawing SlangPy widgets over the model output."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, final

import torch
from torch import Tensor

from flashdreams.api_v2.loop import IUILoop
from flashdreams.runtime_v2.cuda_utils import resolve_cuda_device
from flashdreams.runtime_v2.slangpy_ui_renderer import (
    _SlangPyUIRenderer,
    _UIRenderer,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_StateT = TypeVar("_StateT")

_PRESENTATION_STREAM_PRIORITY = -1
"""Prefer short interactive presentation work over queued model kernels."""


class SlangPyUILoop(IUILoop[_StateT], ABC, Generic[_StateT]):
    """Render a SlangPy UI over an optional model frame.

    Subclass this and implement :meth:`step_ui` instead of ``step``: the widget
    tree is drawn once per UI tick, and whatever :meth:`step_ui` returns is
    composited beneath it. Needs CUDA, Vulkan/CUDA interop and SlangPy, so the
    renderer is created on the first render rather than at construction.
    """

    def __init__(
        self,
        *,
        renderer: _UIRenderer | None = None,
        width: int | None = None,
        height: int | None = None,
        device: torch.device | None = None,
        presentation_stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Configure a SlangPy UI loop without creating GPU resources.

        Args:
            renderer: Rendering backend; ``None`` creates the SlangPy backend.
            width: Render-target width, required for the default renderer.
            height: Render-target height, required for the default renderer.
            device: Fixed CUDA device for prioritized presentation. Omitting it
                preserves the caller's current-stream behavior.
            presentation_stream: CUDA stream to use instead of lazily creating
                a high-priority stream on ``device``.

        Raises:
            ValueError: The default renderer has no output dimensions.
        """
        if renderer is None:
            if width is None or height is None:
                raise ValueError(
                    "width and height are required when renderer is not supplied."
                )
            renderer = _SlangPyUIRenderer(width=width, height=height)
        self.renderer = renderer
        self._presentation_device = None if device is None else torch.device(device)
        self._presentation_stream = presentation_stream
        if presentation_stream is not None:
            stream_device = resolve_cuda_device(presentation_stream.device)
            if self._presentation_device is None:
                self._presentation_device = stream_device
            elif self._presentation_device.type != "cuda" or (
                resolve_cuda_device(self._presentation_device) != stream_device
            ):
                raise ValueError(
                    "The SlangPy presentation stream and device must match."
                )

    @abstractmethod
    def step_ui(
        self, ui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Draw widgets and optionally return the frame beneath them.

        Args:
            ui: SlangPy UI surface. ``ui.screen`` takes top-level widgets, and
                every public ``slangpy.ui`` type is reachable from it.
            step_index: Zero-based index since the latest reset.
            events: Input events not seen by this loop before.

        Returns:
            A ``[C, H, W]`` frame to composite beneath the widgets, usually from
            :meth:`presented_model_frame`, or ``None`` for widgets on black.
        """
        ...

    @final
    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Render the UI over the optional back-buffer returned by :meth:`step_ui`.

        Returns:
            One composited frame, as ``[1, C, H, W]``. Sessions using this loop
            therefore declare a ``tchw`` layout.
        """
        back_buffer: Tensor | None = None

        def draw(ui: Any, index: int, current_events: UserInputEvents) -> None:
            nonlocal back_buffer
            back_buffer = self.step_ui(ui, index, current_events)

        def render_frame() -> Tensor:
            overlay = self.renderer.render(step_index, events, draw)
            if (
                back_buffer is not None
                and back_buffer.is_floating_point()
                and overlay.is_floating_point()
                and overlay.dtype != back_buffer.dtype
            ):
                overlay = overlay.to(dtype=back_buffer.dtype)
            return self._presentation_manager.composite(back_buffer, overlay)

        output_ready_event = None
        presentation_stream = self._get_presentation_stream()
        if presentation_stream is None:
            frame = render_frame()
            if frame.is_cuda:
                output_ready_event = torch.cuda.Event()
                output_ready_event.record(torch.cuda.current_stream(frame.device))
        else:
            device = resolve_cuda_device(presentation_stream.device)
            with torch.cuda.device(device), torch.cuda.stream(presentation_stream):
                frame = render_frame()
                if not frame.is_cuda or resolve_cuda_device(frame.device) != device:
                    raise ValueError(
                        "The SlangPy UI output and presentation stream must share "
                        "a CUDA device."
                    )
                output_ready_event = torch.cuda.Event()
                output_ready_event.record(presentation_stream)
        return StepResult(
            step_index=step_index,
            output=frame.unsqueeze(0),
            frame_count=1,
            output_layout=self.output_layout,
            output_ready_event=output_ready_event,
        )

    def reset(self) -> None:
        """Reset renderer state after a session reset event."""
        self.renderer.reset()

    def close(self) -> None:
        """Release the renderer."""
        if self._presentation_stream is not None:
            self._presentation_stream.synchronize()
        self.renderer.close()

    def _get_presentation_stream(self) -> torch.cuda.Stream | None:
        """Return the configured or lazily created presentation stream."""
        device = self._presentation_device
        if device is None or device.type != "cuda":
            return None
        device = resolve_cuda_device(device)
        if self._presentation_stream is None:
            with torch.cuda.device(device):
                self._presentation_stream = torch.cuda.Stream(
                    device=device,
                    priority=_PRESENTATION_STREAM_PRIORITY,
                )
        return self._presentation_stream


__all__ = ["SlangPyUILoop"]
