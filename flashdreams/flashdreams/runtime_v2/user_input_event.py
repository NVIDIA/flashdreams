# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input events, each a timestamp plus the data for one input modality."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from numpy import uint64

from flashdreams.api_v2.user_input_event_data import UserInputEventData


class KeyboardInputState(Enum):
    """State transition reported by a keyboard input event."""

    RELEASED = "Released"
    """The key changed to the released state."""

    PRESSED = "Pressed"
    """The key changed to the pressed state."""


@dataclass(frozen=True, slots=True, eq=False)
class NumeralKeypadUserInputEventData(UserInputEventData):
    """User input event data for numeral keypad."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "numeral_keypad"

    value: int = 0
    """The number pressed."""


@dataclass(frozen=True, slots=True, eq=False)
class KeyboardUserInputEventData(UserInputEventData):
    """User input event data for keyboard."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "keyboard"

    key: str
    """Identifier of the key this event refers to, e.g. ``"r"``."""
    state: KeyboardInputState
    """State transition reported for ``key``."""


@dataclass(frozen=True, slots=True, eq=False)
class CloseUserInputEventData(UserInputEventData):
    """The client asked to end the run, or went away.

    A window reports this for its X button, a quit shortcut, or a client that
    disconnected. ``run_session`` stops the run when it sees one.
    """

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "close"


@dataclass(frozen=True, slots=True, eq=False)
class ResetUserInputEventData(UserInputEventData):
    """The client asked to start the run over.

    Each registered thread resets before its next ``step``, and its step index
    starts again from zero. The window stays open.
    """

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "reset"


@dataclass(frozen=True, slots=True, eq=False)
class MouseUserInputEventData(UserInputEventData):
    """User input event data for mouse."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "mouse"

    action: Literal["move", "button", "wheel"] = "move"
    """Mouse action represented by this event."""
    x: float = 0.0
    """Horizontal pointer coordinate normalized to the video viewport."""
    y: float = 0.0
    """Vertical pointer coordinate normalized to the video viewport."""
    button: int = 0
    """SlangPy-compatible mouse button index for a button action."""
    pressed: bool = False
    """Whether ``button`` is down for a button action."""
    wheel_x: float = 0.0
    """Horizontal wheel delta for a wheel action."""
    wheel_y: float = 0.0
    """Vertical wheel delta for a wheel action."""


@dataclass(frozen=True, slots=True, eq=False)
class FocusUserInputEventData(UserInputEventData):
    """Client viewport focus change."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "focus"

    focused: bool = False
    """Whether the video viewport owns keyboard focus."""


@dataclass(frozen=True, slots=True, eq=False)
class TouchUserInputEventData(UserInputEventData):
    """One browser or native touch-point update."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "touch"

    action: Literal["start", "move", "end", "cancel"] = "move"
    """Touch lifecycle action."""
    touch_id: int = 0
    """Client-local identifier that remains stable for the touch."""
    x: float = 0.0
    """Horizontal coordinate normalized to the video viewport."""
    y: float = 0.0
    """Vertical coordinate normalized to the video viewport."""
    pressure: float = 0.0
    """Normalized contact pressure."""
    primary: bool = False
    """Whether this is the primary touch point."""


@dataclass(frozen=True, slots=True, eq=False)
class GamepadUserInputEventData(UserInputEventData):
    """A complete gamepad state or connection transition.

    State snapshots keep axes and analog button values together so a model
    loop can consume one internally consistent controller sample.
    """

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "gamepad"

    action: Literal["connected", "disconnected", "state"] = "state"
    """Kind of gamepad update."""
    index: int = 0
    """Client-local gamepad index."""
    controller_id: str = ""
    """Controller identifier reported by the client."""
    mapping: str = ""
    """Controller mapping name, such as ``"standard"``."""
    axes: tuple[float, ...] = ()
    """Normalized analog axes, generally in ``[-1, 1]``."""
    buttons: tuple[float, ...] = ()
    """Analog button values, generally in ``[0, 1]``."""
    pressed: tuple[bool, ...] = ()
    """Digital pressed state corresponding to :attr:`buttons`."""


@dataclass(frozen=True, slots=True, eq=False)
class GameWheelUserInputEventData(UserInputEventData):
    """Normalized steering-wheel controls.

    A client that knows its wheel mapping can publish this higher-level event
    instead of making applications guess which raw gamepad axes are pedals.
    """

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "game_wheel"

    action: Literal["connected", "disconnected", "state"] = "state"
    """Kind of wheel update."""
    index: int = 0
    """Client-local controller index."""
    controller_id: str = ""
    """Controller identifier reported by the client."""
    steering: float = 0.0
    """Steering position in ``[-1, 1]``."""
    throttle: float = 0.0
    """Throttle position in ``[0, 1]``."""
    brake: float = 0.0
    """Brake position in ``[0, 1]``."""
    clutch: float = 0.0
    """Clutch position in ``[0, 1]``."""
    buttons: tuple[bool, ...] = ()
    """Digital wheel-button states."""


@dataclass(frozen=True, slots=True, eq=False)
class XRControllerUserInputEventData(UserInputEventData):
    """One normalized XR-controller state."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "xr_controller"

    action: Literal["connected", "disconnected", "state"] = "state"
    """Kind of XR-controller update."""
    handedness: Literal["left", "right", "none"] = "none"
    """Hand associated with the controller."""
    controller_id: str = ""
    """Controller identifier reported by the client."""
    axes: tuple[float, ...] = ()
    """Normalized touchpad or thumb-stick axes."""
    buttons: tuple[float, ...] = ()
    """Analog button values."""
    pressed: tuple[bool, ...] = ()
    """Digital button states."""
    position: tuple[float, float, float] | None = None
    """Optional controller position in the current XR reference space."""
    orientation: tuple[float, float, float, float] | None = None
    """Optional controller quaternion in ``(x, y, z, w)`` order."""


@dataclass(frozen=True, slots=True, eq=False)
class UnknownUserInputEventData(UserInputEventData):
    """User input event data for unknown."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "unknown"


@dataclass(frozen=True, slots=True)
class UserInputEvent:
    """User input event."""

    timestamp: uint64
    """Timestamp in microseconds since the start of the session."""

    event_data: UserInputEventData
    """Event data."""

    def get_timestamp(self) -> uint64:
        """Return the timestamp."""
        return self.timestamp

    def get_event_data(self) -> UserInputEventData:
        """Return the event data structure with type & data."""
        return self.event_data
