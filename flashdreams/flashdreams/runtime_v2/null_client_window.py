# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window that discards output and reports no input."""

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class NullClientWindow(IClientWindow):
    """Discard output and report no input."""

    def get_user_input_events(self) -> UserInputEvents:
        """Return an empty input batch."""
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        """Accept a session without opening an output."""
        del session_desc

    def write(self, result: StepResult) -> None:
        """Discard one result."""
        del result

    def close(self) -> None:
        """Close without releasing any resources."""
        return
