# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immediate-mode prompt controls for interactive text-to-video sessions."""

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop
from flashdreams.runtime_v2.user_input_events import UserInputEvents


@dataclass(slots=True)
class T2VUIState:
    """Editable prompt state owned by the text-to-video UI loop."""

    prompt: str = ""
    """Prompt currently displayed in the input field."""

    message: str = ""
    """Validation or progress message displayed below the controls."""


class T2VImGuiUILoop(ImGuiUILoop[T2VUIState]):
    """Draw a prompt field that starts a replacement text-to-video session."""

    def step_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Draw prompt controls over the latest generated model frame."""
        del step_index, events
        imgui.set_next_window_pos(imgui.ImVec2(16.0, 16.0), imgui.Cond_.once)
        imgui.set_next_window_size(imgui.ImVec2(520.0, 130.0), imgui.Cond_.once)
        imgui.begin("Text to video")
        try:
            imgui.text("Prompt")
            _, self.state.prompt = imgui.input_text("##t2v-prompt", self.state.prompt)
            if imgui.button("New session"):
                prompt = self.state.prompt.strip()
                if prompt:
                    self.state.prompt = prompt
                    self.state.message = "Starting new session…"
                    self.request_new_session({"prompt": prompt})
                else:
                    self.state.message = "Enter a prompt before starting a session."
            if self.state.message:
                imgui.text(self.state.message)
        finally:
            imgui.end()
        return self.presented_model_frame()


__all__ = ["T2VImGuiUILoop", "T2VUIState"]
