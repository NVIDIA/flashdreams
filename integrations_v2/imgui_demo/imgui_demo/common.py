# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared session defaults and background model thread for ImGui demos."""

from dataclasses import dataclass

import torch

from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

DEFAULT_SESSION_DESC = SessionDesc(
    output_layout=VideoTensorLayout.tchw,
    frames_per_second_for_ui=60,
    frames_per_second_for_step=30,
    video_width=640,
    video_height=480,
)
"""Output dimensions and rates shared by both ImGui demos."""


@dataclass(frozen=True, slots=True)
class BackgroundModelState:
    """Configuration owned by a demo's background model thread."""

    session_desc: SessionDesc
    """Output layout and dimensions for generated background frames."""

    device: torch.device | str
    """Device used for generated frames."""


class BackgroundModelThread(IThread[BackgroundModelState]):
    """Generate a dark background beneath an ImGui UI layer."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Return one dark background frame."""
        del events
        desc = self.state.session_desc
        return [
            StepResult(
                step_index=step_index,
                output=torch.full(
                    (1, 3, desc.video_height, desc.video_width),
                    -0.85,
                    dtype=torch.float32,
                    device=self.state.device,
                ),
                frame_count=1,
                output_layout=desc.output_layout,
            )
        ]

    def reset(self) -> None:
        return


def require_tchw(session_desc: SessionDesc, demo_name: str) -> None:
    """Reject a layout the ImGui demos cannot render."""
    if session_desc.output_layout is not VideoTensorLayout.tchw:
        raise ValueError(
            f"The {demo_name} demo requires tchw output, got "
            f"{session_desc.output_layout.value}."
        )
