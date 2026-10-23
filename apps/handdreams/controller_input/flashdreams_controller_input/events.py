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

"""Immutable controller frames and connection status for FlashDreams V2 apps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from flashdreams.api_v2.user_input_event import UserInputEvent

from .pose import (
    OPENXR_LOCAL_GRIP_POSE_CONVENTION,
    OPENXR_STAGE_GRIP_POSE_CONVENTION,
    WRIST_PROXY_POSE_CONVENTION,
)

_SUPPORTED_POSE_CONVENTIONS = frozenset(
    {
        OPENXR_LOCAL_GRIP_POSE_CONVENTION,
        OPENXR_STAGE_GRIP_POSE_CONVENTION,
        WRIST_PROXY_POSE_CONVENTION,
    }
)


@dataclass(frozen=True, slots=True)
class ControllerConnectionStatus:
    """Source-owned connection guidance, independent of controller pose tracking."""

    headset: str = ""
    """Local headset discovery or setup stage; empty when not reported."""
    web_client: str = ""
    """Actual browser-input service state, not inferred from tracked poses."""
    web_client_url: str = ""
    """Public connection URL; empty until browser input can be offered."""
    certificate_url: str = ""
    """Browser certificate-trust URL, when required before connecting."""
    active_input: str = ""
    """Source currently owning controller input, or a waiting explanation."""


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class ControllerStatusEvent(UserInputEvent):
    """Immutable setup snapshot delivered even before controllers are tracked."""

    status: ControllerConnectionStatus
    """Complete source status replacing the previous snapshot."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the plugin's registered V2 event type."""
        return "controller_input_status"


@dataclass(frozen=True, slots=True, kw_only=True)
class ControllerSnapshot:
    """Immutable device values without application button bindings."""

    position: tuple[float, float, float]
    """Position in the frame's pose convention, in meters."""
    orientation: tuple[float, float, float, float]
    """Quaternion in XYZW order, without additional normalization."""
    axes: tuple[float, ...] = ()
    """The first two source thumbstick axes, or empty when unavailable."""
    trigger: float = 0.0
    """Analog trigger value reported by the source."""
    grip: float = 0.0
    """Analog grip value reported by the source."""
    tracked: bool = True
    """Whether the source reports a valid tracked pose."""
    battery: float | None = None
    """Reported battery fraction, or ``None`` when unavailable."""
    primary_button: bool = False
    """Primary face button (X on the left, A on the right)."""
    secondary_button: bool = False
    """Secondary face button (Y on the left, B on the right)."""

    def __post_init__(self) -> None:
        for name in ("position", "orientation", "axes"):
            values = tuple(float(value) for value in getattr(self, name))
            object.__setattr__(self, name, values[:2] if name == "axes" else values)
        if len(self.position) != 3 or len(self.orientation) != 4:
            raise ValueError("controller pose has invalid shape")
        for name in ("trigger", "grip"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("tracked", "primary_button", "secondary_button"):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.battery is not None:
            object.__setattr__(self, "battery", float(self.battery))


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class ControllerFrameEvent(UserInputEvent):
    """One atomic controller pair or a source-wide disconnect."""

    action: Literal["state", "disconnected"]
    """Whether this event contains poses or breaks source continuity."""
    frame_id: int | None = None
    """Monotonic source identity; absent on disconnect."""
    client_capture_ns: int | None = None
    """Original capture time, separate from the host-relative session timestamp."""
    pose_convention: str | None = None
    """Shared pose convention; absent on disconnect."""
    left: ControllerSnapshot | None = None
    """Left controller; required for state and absent on disconnect."""
    right: ControllerSnapshot | None = None
    """Right controller; required for state and absent on disconnect."""
    is_emulated: bool = False
    """Explicit active-client emulation signal; unknown clients retain guided setup."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the plugin's registered V2 event type."""
        return "controller_input_frame"

    def __post_init__(self) -> None:
        if type(self.is_emulated) is not bool:
            raise ValueError("controller is_emulated must be a boolean")
        if self.action == "disconnected":
            if self.is_emulated or any(
                value is not None
                for value in (
                    self.frame_id,
                    self.client_capture_ns,
                    self.pose_convention,
                    self.left,
                    self.right,
                )
            ):
                raise ValueError("disconnected events cannot contain controller state")
            return
        if self.action != "state":
            raise ValueError(f"unsupported controller action: {self.action!r}")
        if not all(
            isinstance(hand, ControllerSnapshot) for hand in (self.left, self.right)
        ):
            raise ValueError("controller state requires both left and right snapshots")
        for name in ("frame_id", "client_capture_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"controller state requires non-negative integer {name}"
                )
        if self.pose_convention not in _SUPPORTED_POSE_CONVENTIONS:
            raise ValueError(
                f"unsupported controller pose convention: {self.pose_convention!r}"
            )

    def to_controller_frame(self) -> dict[str, Any]:
        """Materialize a fresh mutable payload without changing device values."""
        if self.action != "state":
            raise ValueError("disconnected events do not contain controller poses")
        frame: dict[str, Any] = {
            "client_capture_ns": self.client_capture_ns,
            "source_frame_id": self.frame_id,
            "pose_convention": self.pose_convention,
        }
        for side, hand in (("left", self.left), ("right", self.right)):
            assert hand is not None
            frame[side] = {
                "position": list(hand.position),
                "orientation": list(hand.orientation),
                "trigger": hand.trigger,
                "grip": hand.grip,
                "tracked": hand.tracked,
                "primary_button": hand.primary_button,
                "secondary_button": hand.secondary_button,
            }
            if hand.axes:
                frame[side]["axes"] = list(hand.axes)
            if hand.battery is not None:
                frame[side]["battery"] = hand.battery
        return frame
