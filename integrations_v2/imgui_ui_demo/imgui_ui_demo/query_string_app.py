# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Query-string-controlled ImGui background application."""

from collections.abc import Sequence
from dataclasses import dataclass
from re import fullmatch
from typing import Any
from urllib.parse import unquote

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import QueryStringUserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


class BackgroundModelLoop(IModelLoop[tuple[SessionDesc, torch.device | str]]):
    """Generate a dark background beneath the text-input UI layer."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Return one dark background frame."""
        del events
        desc, device = self.state
        return [
            StepResult(
                step_index=step_index,
                output=torch.full(
                    (1, 3, 1, 1),
                    -0.85,
                    dtype=torch.float32,
                    device=device,
                ),
                frame_count=1,
                output_layout=desc.output_layout,
            )
        ]

    def reset(self) -> None:
        return


@dataclass(slots=True)
class QueryStringState:
    """Background state owned by the ImGui UI loop."""

    rgb: tuple[int, int, int] = (0, 0, 0)
    """Current red, green, and blue channel values in ``[0, 255]``."""


def _parse_rgb_query_string(query_string: str) -> tuple[int, int, int]:
    """Parse a ``?(r,g,b)`` URL query into RGB channel values.

    Args:
        query_string: Raw URL query without its leading question mark.

    Returns:
        Red, green, and blue channel values in ``[0, 255]``.

    Raises:
        ValueError: The query is not a tuple of three integer channel values.
    """
    match = fullmatch(
        r"\(([0-9]+),([0-9]+),([0-9]+)\)",
        unquote(query_string),
    )
    if match is None:
        raise ValueError(
            "Query string must have the form '(r,g,b)' with integer channels."
        )
    red, green, blue = (int(channel) for channel in match.groups())
    if any(channel > 255 for channel in (red, green, blue)):
        raise ValueError(
            "Query string must have the form '(r,g,b)' with channels in [0, 255]."
        )
    return red, green, blue


class QueryStringImGuiUILoop(ImGuiUILoop[QueryStringState]):
    """Render the RGB color selected by the connecting browser's URL."""

    def step_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor:
        """Apply query-string events and return the selected background."""
        del imgui, step_index
        for event in events.get_events():
            if isinstance(event, QueryStringUserInputEvent):
                self.state.rgb = _parse_rgb_query_string(event.query_string)
        return torch.tensor(self.state.rgb, dtype=torch.uint8).view(3, 1, 1)


class QueryStringSession(ISession):
    """Run the query-string-controlled ImGui background."""

    def __init__(
        self,
        session_desc: SessionDesc,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        """Configure one query-string session.

        Args:
            session_desc: Output dimensions and loop frequencies.
            device: Device used for the fallback model frame.

        Raises:
            ValueError: The session output layout is not ``tchw``.
        """
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                "The query-string demo requires tchw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._session_desc = session_desc
        self._device = device

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved session description."""
        return self._session_desc

    def init(self) -> None:
        """Register the query-string UI and fallback model loops."""
        self.register_ui_loop(
            QueryStringImGuiUILoop,
            state=QueryStringState(),
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
        )
        self.register_model_loop(
            BackgroundModelLoop,
            state=(self._session_desc, self._device),
        )


class QueryStringApplication(IApplication):
    """Create query-string-controlled ImGui sessions."""

    def init(self, commandline_args: Sequence[str]) -> None:
        """Reject application-specific arguments."""
        if commandline_args:
            raise ValueError("The query-string demo takes no application arguments.")

    def session_desc(self) -> SessionDesc:
        """Return the demo's established dimensions and rates."""
        return SessionDesc(video_width=640, video_height=480)

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized query-string session."""
        return QueryStringSession(session_desc)


def create_app() -> IApplication:
    """Return a new query-string application."""
    return QueryStringApplication()
