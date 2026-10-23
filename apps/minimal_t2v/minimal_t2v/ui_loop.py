# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ImGui prompt controls sending messages to the tutorial model loop."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch import Tensor

from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop
from flashdreams.runtime_v2.user_input_events import UserInputEvents

if TYPE_CHECKING:
    from .model_loop import TextToVideoState


@dataclass
class MinimalT2VUIState:
    """Editable prompt owned by the UI thread."""

    model_loop: IModelLoop["TextToVideoState"]
    """Destination of prompt update messages."""
    text: str = ""
    """Text currently displayed in the prompt field."""


class MinimalT2VUILoop(ImGuiUILoop[MinimalT2VUIState]):
    """Send prompt updates without directly mutating model state."""

    def step_ui(
        self, imgui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Draw prompt controls over the latest generated frame."""
        imgui.begin("Minimal T2V")
        try:
            _, self.state.text = imgui.input_text("Prompt", self.state.text)
            if imgui.button("Send"):
                invoke_async(
                    self.state.model_loop,
                    lambda state, text=self.state.text: state.set_prompt(text),
                )
        finally:
            imgui.end()
        return self.presented_model_frame()
