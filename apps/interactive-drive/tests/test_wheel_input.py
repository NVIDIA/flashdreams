# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from flashdreams.runtime import TimeWindow, UserInputEvent, UserInputs
from interactive_drive_app.input.wheel import WheelToDriverCommand

pytestmark = pytest.mark.ci_cpu


def test_wheel_state_produces_direct_manual_driver_command() -> None:
    converter = WheelToDriverCommand()

    command = converter.convert(
        UserInputs(
            events=(
                UserInputEvent(
                    timestamp_s=0.1,
                    event_type="wheel_state",
                    payload={
                        "steer": -0.5,
                        "throttle": 0.75,
                        "brake": 0.1,
                        "reverse": True,
                    },
                ),
            )
        ),
        TimeWindow(start_s=0.0, end_s=1.0),
    )

    assert command is not None
    assert command["steer"] == -0.5
    assert command["steer_is_direct"]
    assert command["manual_control"]
    assert command["reverse"]
