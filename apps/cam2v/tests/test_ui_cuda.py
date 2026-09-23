# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA coverage for prioritized Cam2V SlangPy presentation."""

import queue
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from cam2v import Cam2VSlangPyUILoop, Cam2VUIState

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_gpu


class _CudaOverlayRenderer:
    """Minimal injected SlangPy renderer for stream-ordering coverage."""

    def __init__(self, *, device: torch.device, width: int, height: int) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.closed = False
        self.render_stream: torch.cuda.Stream | None = None
        self.ui = SimpleNamespace(
            screen=object(),
            Window=lambda *args, **kwargs: object(),
            Text=lambda parent, text: SimpleNamespace(text=text),
        )

    def render(
        self,
        step_index: int,
        events: UserInputEvents,
        step_ui: Any,
    ) -> torch.Tensor:
        """Run the Cam2V widget callback and return a transparent overlay."""
        self.render_stream = torch.cuda.current_stream(self.device)
        step_ui(self.ui, step_index, events)
        return torch.zeros(
            (4, self.height, self.width),
            device=self.device,
            dtype=torch.float32,
        )

    def reset(self) -> None:
        """Match the injected renderer protocol."""

    def close(self) -> None:
        """Record renderer shutdown."""
        self.closed = True
