# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy UI loop for FlashDreams applications."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, final

from torch import Tensor

from flashdreams.api_v2.thread import IUILoop
from flashdreams.runtime_v2._slangpy_ui_renderer import (
    _SlangPyUIRenderer,
    _UIRenderer,
)
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_StateT = TypeVar("_StateT")


class SlangPyUILoop(IUILoop[_StateT], ABC, Generic[_StateT]):
    """Render a SlangPy UI over an optional model frame."""

    def __init__(
        self,
        *,
        state: _StateT,
        frequency: int,
        output_layout: VideoTensorLayout,
        presentation_manager: PresentationManager,
        renderer: _UIRenderer | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Configure a SlangPy UI loop without creating GPU resources.

        Args:
            state: State used by the UI loop.
            frequency: Maximum UI iterations per second.
            output_layout: Layout used for the compositing result.
            presentation_manager: Buffer containing model frames.
            renderer: Rendering backend; ``None`` creates the SlangPy backend.
            width: Render-target width, required for the default renderer.
            height: Render-target height, required for the default renderer.

        Raises:
            ValueError: ``output_layout`` is not ``tchw``, or the default
                renderer has no output dimensions.
        """
        if output_layout is not VideoTensorLayout.tchw:
            raise ValueError("SlangPy UI rendering requires tchw output.")
        super().__init__(
            state=state,
            frequency=frequency,
            output_layout=output_layout,
            presentation_manager=presentation_manager,
        )
        if renderer is None:
            if width is None or height is None:
                raise ValueError(
                    "width and height are required when renderer is not supplied."
                )
            renderer = _SlangPyUIRenderer(width=width, height=height)
        self.renderer = renderer

    @abstractmethod
    def step_ui(
        self, ui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Draw widgets and optionally return the frame beneath them."""
        ...

    @final
    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Render the UI over the optional back-buffer returned by :meth:`step_ui`."""
        back_buffer: Tensor | None = None

        def draw(ui: Any, index: int, current_events: UserInputEvents) -> None:
            nonlocal back_buffer
            back_buffer = self.step_ui(ui, index, current_events)

        overlay = self.renderer.render(step_index, events, draw)
        frame = self._presentation_manager.composite(back_buffer, overlay)
        return StepResult(
            step_index=step_index,
            output=frame.unsqueeze(0),
            frame_count=1,
            output_layout=self.output_layout,
        )

    def reset(self) -> None:
        """Reset renderer state after a session reset event."""
        self.renderer.reset()

    def close(self) -> None:
        """Release the renderer."""
        self.renderer.close()


__all__ = ["SlangPyUILoop"]
