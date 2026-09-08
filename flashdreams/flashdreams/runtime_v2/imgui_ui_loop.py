# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immediate Dear ImGui UI loop rendered through SlangPy."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, final

from torch import Tensor

from flashdreams.api_v2.loop import IUILoop
from flashdreams.runtime_v2.imgui_ui_renderer import _ImGuiUIRenderer
from flashdreams.runtime_v2.slangpy_ui_renderer import _UIRenderer
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.ui_compositing import prepare_ui_back_buffer
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_StateT = TypeVar("_StateT")


class ImGuiUILoop(IUILoop[_StateT], ABC, Generic[_StateT]):
    """Render immediate Dear ImGui controls over an optional model frame.

    Subclass this and implement :meth:`step_ui`. The ``imgui`` argument exposes
    ``imgui_bundle.imgui`` and adds an image-like pixel convenience form:
    ``imgui.image(key, pixels, size=(width, height))``.
    """

    def __init__(
        self,
        *,
        renderer: _UIRenderer | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Configure an ImGui loop without eagerly creating GPU resources."""
        if renderer is None:
            if width is None or height is None:
                raise ValueError(
                    "width and height are required when renderer is not supplied."
                )
            renderer = _ImGuiUIRenderer(width=width, height=height)
        self.renderer = renderer

    @abstractmethod
    def step_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Draw one immediate-mode UI frame and return its optional back buffer."""
        ...

    @final
    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Render and composite one ImGui frame."""
        back_buffer: Tensor | None = None

        def draw(imgui: Any, index: int, current_events: UserInputEvents) -> None:
            nonlocal back_buffer
            back_buffer = self.step_ui(imgui, index, current_events)

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


__all__ = ["ImGuiUILoop"]
