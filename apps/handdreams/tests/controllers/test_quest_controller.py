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

"""HandDreams controller-boundary validation regressions."""

import pytest
from handdreams.controllers import quest_controller

pytestmark = pytest.mark.ci_cpu


def _frame() -> dict:
    return {
        "client_capture_ns": 123,
        **{
            side: {
                "position": [0.0, 1.0, -0.5],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            }
            for side in ("left", "right")
        },
    }


def test_controller_validation_keeps_capture_clocks_without_http_metadata() -> None:
    raw = _frame()
    raw.update(
        source_client_capture_ns=456,
        source_frame_index=7,
        source_capture_ns=123,
        source_time_origin_ns=100,
        source_frame_index_is_exact=True,
        client_upload_queued_ns=124,
        client_upload_started_ns=125,
    )
    frame = quest_controller.validate_controller_frame(raw)
    assert set(frame) == {
        "left",
        "right",
        "client_capture_ns",
        "source_client_capture_ns",
    }
    assert frame["client_capture_ns"] == 123
    assert frame["source_client_capture_ns"] == 456


@pytest.mark.parametrize("side", ["left", "right"])
def test_face_buttons_do_not_map_to_demo_commands(side: str) -> None:
    raw = _frame()
    raw[side].update(
        primary_button=True, secondary_button=True, recenter=True, gripper_open=0.25
    )
    frame = quest_controller.validate_controller_frame(raw)
    hand = frame[side]
    assert "next_scene" not in hand
    assert "previous_scene" not in hand
    assert "toggle_live" not in hand
    assert "recenter" not in hand
    assert hand["gripper_open"] == 0.25
    assert "primary_button" not in hand
    assert "secondary_button" not in hand


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan"), "0.5"])
def test_demo_gripper_validation_remains_strict(value: object) -> None:
    raw = _frame()
    raw["left"]["gripper_open"] = value
    with pytest.raises(ValueError, match="gripper_open"):
        quest_controller.validate_controller_frame(raw)
