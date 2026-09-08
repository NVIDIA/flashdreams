# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UI loop drawing SlangPy widgets over the model output."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, final

from torch import Tensor

from flashdreams.api_v2.loop import IUILoop
from flashdreams.runtime_v2.slangpy_ui_renderer import (
    _SlangPyUIRenderer,
    _UIRenderer,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.ui_compositing import prepare_ui_back_buffer
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_StateT = TypeVar("_StateT")


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
    ) -> None:
        """Configure a SlangPy UI loop without creating GPU resources.

        Args:
            renderer: Rendering backend; ``None`` creates the SlangPy backend.
            width: Render-target width, required for the default renderer.
            height: Render-target height, required for the default renderer.

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

        overlay = self.renderer.render(step_index, events, draw)
        back_buffer = prepare_ui_back_buffer(back_buffer, overlay)
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
