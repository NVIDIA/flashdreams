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
    """ImGui-compatible mouse button index for a button action."""
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


# Below are stubbed input event data implementations for future clients.


@dataclass(frozen=True, slots=True, eq=False)
class TouchUserInputEventData(UserInputEventData):
    """User input event data for touch."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "touch"


@dataclass(frozen=True, slots=True, eq=False)
class GamepadUserInputEventData(UserInputEventData):
    """User input event data for gamepad."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "gamepad"


@dataclass(frozen=True, slots=True, eq=False)
class GameWheelUserInputEventData(UserInputEventData):
    """User input event data for game wheel."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "game_wheel"


@dataclass(frozen=True, slots=True, eq=False)
class XRControllerUserInputEventData(UserInputEventData):
    """User input event data for XR controllers."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "xr_controller"


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
