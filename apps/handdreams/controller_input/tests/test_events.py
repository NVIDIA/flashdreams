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

"""Atomic controller event validation and isolation across runtime readers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Literal

import numpy as np
import pytest
from flashdreams_controller_input.events import (
    ControllerConnectionStatus,
    ControllerFrameEvent,
    ControllerSnapshot,
    ControllerStatusEvent,
)
from flashdreams_controller_input.pose import (
    OPENXR_LOCAL_GRIP_POSE_CONVENTION,
    OPENXR_STAGE_GRIP_POSE_CONVENTION,
    WRIST_PROXY_POSE_CONVENTION,
)

from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_connection_status_is_immutable_and_not_a_controller_frame() -> None:
    status = ControllerConnectionStatus(headset="Press the power button")
    event = ControllerStatusEvent(timestamp=np.uint64(1), status=status)
    assert isinstance(event, UserInputEvent)
    assert not isinstance(event, ControllerFrameEvent)
    assert event.get_type_name() == "controller_input_status"
    assert event.get_timestamp() == np.uint64(1)
    assert event != replace(event)
    assert status.web_client == status.active_input == ""
    with pytest.raises(FrozenInstanceError):
        status.headset = "Ready"  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        event.status = ControllerConnectionStatus()  # ty: ignore[invalid-assignment]


def _snapshot(
    *,
    position: tuple[float, float, float] = (1.0, 2.0, 3.0),
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    axes: tuple[float, ...] = (0.0, 0.0),
    trigger: float = 0.0,
    grip: float = 0.0,
    tracked: bool = True,
    battery: float | None = None,
    primary_button: bool = False,
    secondary_button: bool = False,
) -> ControllerSnapshot:
    return ControllerSnapshot(
        position=position,
        orientation=orientation,
        axes=axes,
        trigger=trigger,
        grip=grip,
        tracked=tracked,
        battery=battery,
        primary_button=primary_button,
        secondary_button=secondary_button,
    )


def _event(
    *,
    action: Literal["state", "disconnected"] = "state",
    frame_id: int | None = 0,
    client_capture_ns: int | None = 0,
    pose_convention: str | None = OPENXR_LOCAL_GRIP_POSE_CONVENTION,
    left: ControllerSnapshot | None = _snapshot(),
    right: ControllerSnapshot | None = _snapshot(),
) -> ControllerFrameEvent:
    return ControllerFrameEvent(
        timestamp=np.uint64(5),
        action=action,
        frame_id=frame_id,
        client_capture_ns=client_capture_ns,
        pose_convention=pose_convention,
        left=left,
        right=right,
    )


@pytest.mark.parametrize(
    "convention",
    [
        OPENXR_LOCAL_GRIP_POSE_CONVENTION,
        OPENXR_STAGE_GRIP_POSE_CONVENTION,
        WRIST_PROXY_POSE_CONVENTION,
    ],
)
def test_state_materializes_the_controller_frame_schema(convention: str) -> None:
    """Preserve both hands, source identity, and supported pose conventions."""
    event = _event(
        pose_convention=convention,
        left=_snapshot(
            trigger=0.25,
            grip=0.75,
            tracked=False,
            battery=0.5,
            primary_button=True,
            secondary_button=True,
        ),
        right=_snapshot(secondary_button=True),
    )

    assert isinstance(event, UserInputEvent)
    assert event.get_type_name() == "controller_input_frame"
    assert event.get_timestamp() == np.uint64(5)
    assert event != replace(event)
    assert event.to_controller_frame() == {
        "source_frame_id": 0,
        "client_capture_ns": 0,
        "pose_convention": convention,
        "left": {
            "position": [1.0, 2.0, 3.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "axes": [0.0, 0.0],
            "trigger": 0.25,
            "grip": 0.75,
            "tracked": False,
            "battery": 0.5,
            "primary_button": True,
            "secondary_button": True,
        },
        "right": {
            "position": [1.0, 2.0, 3.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "axes": [0.0, 0.0],
            "trigger": 0.0,
            "grip": 0.0,
            "tracked": True,
            "primary_button": False,
            "secondary_button": True,
        },
    }


def test_snapshots_detach_source_lists_and_materialized_frames() -> None:
    """Keep event data immutable when producers and consumers mutate lists."""
    source = {
        "position": [1, 2, 3],
        "orientation": [0, 0, 0, 1],
        "axes": [0, 0, 9],
    }
    # Exercise normalization of mutable vectors outside the annotated contract.
    snapshot = ControllerSnapshot(**source)  # ty: ignore[invalid-argument-type]
    assert snapshot.axes == (0.0, 0.0)
    event = _event(left=snapshot, right=snapshot)
    expected = event.to_controller_frame()

    for field, values in source.items():
        values[0] = 99
        actual = getattr(snapshot, field)
        assert isinstance(actual, tuple)
        assert all(isinstance(value, float) for value in actual)
        assert list(actual) == expected["left"][field]
    with pytest.raises(FrozenInstanceError):
        snapshot.tracked = False  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        event.frame_id = 99  # ty: ignore[invalid-assignment]

    materialized = event.to_controller_frame()
    for field in source:
        materialized["left"][field][0] = 99
        assert materialized["right"][field] == expected["right"][field]
    materialized["right"]["trigger"] = 1.0
    materialized["source_frame_id"] = 99
    assert event.to_controller_frame() == expected


def test_emulation_is_explicit_and_does_not_change_pipeline_values() -> None:
    physical = _event()
    emulated = replace(physical, is_emulated=True)
    assert physical.is_emulated is False
    assert emulated.is_emulated is True
    assert emulated.to_controller_frame() == physical.to_controller_frame()
    with pytest.raises(FrozenInstanceError):
        emulated.is_emulated = False  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
def test_emulation_flag_requires_a_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="is_emulated must be a boolean"):
        replace(_event(), is_emulated=value)


def test_disconnect_cannot_retain_an_emulation_flag() -> None:
    with pytest.raises(ValueError, match="cannot contain controller state"):
        ControllerFrameEvent(
            timestamp=np.uint64(5), action="disconnected", is_emulated=True
        )


@pytest.mark.parametrize(
    "field", ["left", "right", "frame_id", "client_capture_ns", "pose_convention"]
)
def test_state_requires_a_complete_frame(field: str) -> None:
    """Reject state events missing either hand or shared frame metadata."""
    with pytest.raises(ValueError):
        _event(**{field: None})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [("position", (0.0, 0.0)), ("orientation", (0.0, 0.0, 1.0))],
)
def test_snapshot_rejects_wrong_vector_shapes(field: str, value: tuple) -> None:
    """Reject controller vectors that cannot represent a complete pose."""
    with pytest.raises(ValueError):
        _snapshot(**{field: value})  # ty: ignore[invalid-argument-type]


def test_state_preserves_absent_axes_as_an_omitted_field() -> None:
    """Keep unavailable thumbstick inputs distinct from malformed axis arrays."""
    event = _event(left=_snapshot(axes=()))
    assert event.left is not None
    assert event.left.axes == ()
    assert "axes" not in event.to_controller_frame()["left"]


def test_state_does_not_repair_incomplete_axes() -> None:
    """Leave nonempty malformed data for the consumer's strict validation."""
    event = _event(left=_snapshot(axes=(0.25,)))
    assert event.to_controller_frame()["left"]["axes"] == [0.25]


@pytest.mark.parametrize("field", ["frame_id", "client_capture_ns"])
@pytest.mark.parametrize("value", [-1, True, False, 1.5, "1"])
def test_state_requires_nonnegative_integer_identity(field: str, value: object) -> None:
    """Reject ambiguous or negative source frame identifiers and capture times."""
    with pytest.raises(ValueError):
        _event(**{field: value})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("field", "value"), [("pose_convention", "unknown"), ("action", "unknown")]
)
def test_state_rejects_unknown_action_or_pose_convention(
    field: str, value: str
) -> None:
    """Reject unknown actions and controller coordinate conventions."""
    with pytest.raises(ValueError):
        _event(**{field: value})  # ty: ignore[invalid-argument-type]


def test_disconnected_event_has_no_state_payload() -> None:
    """Keep a disconnect distinct from a usable controller frame."""
    event = ControllerFrameEvent(timestamp=np.uint64(0), action="disconnected")

    for field in ("left", "right", "frame_id", "client_capture_ns", "pose_convention"):
        assert getattr(event, field) is None
        with pytest.raises(ValueError):
            replace(event, **{field: getattr(_event(), field)})
    with pytest.raises(ValueError):
        event.to_controller_frame()


def test_event_buffer_readers_receive_the_same_immutable_frame() -> None:
    """Let UI and model readers materialize independent controller payloads."""
    event = _event()
    buffer = EventBuffer()
    buffer.register(1)
    buffer.register(2)
    buffer.append(UserInputEvents([event]))

    ui_events, _ = buffer.read(1)
    ui_event = ui_events.get_events()[0]
    assert isinstance(ui_event, ControllerFrameEvent)
    assert ui_event is event
    ui_event.to_controller_frame()["left"]["position"][0] = 99
    assert buffer.collect_garbage() == 0

    model_events, _ = buffer.read(2)
    model_event = model_events.get_events()[0]
    assert isinstance(model_event, ControllerFrameEvent)
    assert model_event is event
    assert model_event.to_controller_frame()["left"]["position"] == [1.0, 2.0, 3.0]
    assert buffer.collect_garbage() == 1
    assert buffer.read(1)[0].get_events() == []
    assert buffer.read(2)[0].get_events() == []
