# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session wiring for the Minimal T2V model and UI loops."""

from typing import Any

from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc

from .model_loop import TextToVideoLoop, TextToVideoState
from .ui_loop import MinimalT2VUILoop, MinimalT2VUIState


class MinimalT2VSession(ISession):
    """One interactive tutorial session sharing the application's pipeline."""

    def __init__(
        self, pipeline: Any, desc: SessionDesc, prompt: str, total_blocks: int
    ) -> None:
        """Keep session inputs until the runtime initializes the loops."""
        self.pipeline = pipeline
        self.desc = desc
        self.prompt = prompt
        self.total_blocks = total_blocks

    @property
    def session_desc(self) -> SessionDesc:
        """Return the description used to configure the window."""
        return self.desc

    def init(self) -> None:
        """Register independent model and UI state."""
        model_loop = self.register_model_loop(
            TextToVideoLoop,
            state=TextToVideoState(
                self.pipeline, self.desc, self.total_blocks, self.prompt
            ),
        )
        self.register_ui_loop(
            MinimalT2VUILoop,
            state=MinimalT2VUIState(model_loop, self.prompt),
            width=self.desc.video_width,
            height=self.desc.video_height,
        )
