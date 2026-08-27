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

"""Client window that writes a native WebM file."""

from pathlib import Path

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.webm_output_sink import WebmOutputSink


class WebmClientWindow(IClientWindow):
    """Write UI frames to a native WebM file and report no input."""

    def __init__(self, path: str | Path) -> None:
        """
        Args:
            path: WebM file to write. Parent directories are created.
        """
        self._video_sink = WebmOutputSink(path)

    @property
    def path(self) -> Path:
        """Return the output path."""
        return self._video_sink.path

    @property
    def codec(self) -> str:
        """Return the benchmark-selected VPx codec."""
        return self._video_sink.codec

    def get_user_input_events(self) -> UserInputEvents:
        """Return an empty input batch."""
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        """Prepare native WebM staging for the session."""
        self._video_sink.open(session_desc)

    def write(self, result: StepResult) -> None:
        """Encode one result produced by the UI thread."""
        self._video_sink.write(result)

    def close(self) -> None:
        """Finish and atomically publish the WebM file."""
        self._video_sink.close()

    def abort(self) -> None:
        """Discard incomplete WebM staging without changing the target."""
        self._video_sink.abort()


__all__ = ["WebmClientWindow"]
