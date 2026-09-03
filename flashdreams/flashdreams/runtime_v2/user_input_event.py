# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete user input events for supported input modalities."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from flashdreams.api_v2.user_input_event import UserInputEvent


class KeyboardInputState(Enum):
    """State transition reported by a keyboard input event."""

    RELEASED = "Released"
    """The key changed to the released state."""

    PRESSED = "Pressed"
    """The key changed to the pressed state."""


@dataclass(frozen=True, slots=True, eq=False)
class NumeralKeypadUserInputEvent(UserInputEvent):
    """User input event for a numeral keypad."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "numeral_keypad"

    value: int = 0
    """The number pressed."""


@dataclass(frozen=True, slots=True, eq=False)
class KeyboardUserInputEvent(UserInputEvent):
    """User input event for a keyboard."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "keyboard"

    key: str
    """Identifier of the key this event refers to, e.g. ``"r"``."""
    state: KeyboardInputState
    """State transition reported for ``key``."""


@dataclass(frozen=True, slots=True, eq=False)
class CloseUserInputEvent(UserInputEvent):
    """The client asked to end the run, or went away.

    A window reports this for its X button, a quit shortcut, or a client that
    disconnected. ``run_session`` stops the run when it sees one.
    """

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "close"


@dataclass(frozen=True, slots=True, eq=False)
class ResetUserInputEvent(UserInputEvent):
    """The client asked to start the run over.

    Every registered loop resets before its next ``step``, its step index starts
    again from zero, and frames generated before the reset are discarded rather
    than presented. The window stays open.
    """

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "reset"


@dataclass(frozen=True, slots=True, eq=False)
class MouseUserInputEvent(UserInputEvent):
    """User input event for a mouse."""

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
class FocusUserInputEvent(UserInputEvent):
    """Client viewport focus change."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "focus"

    focused: bool = False
    """Whether the video viewport owns keyboard focus."""


@dataclass(frozen=True, slots=True, eq=False)
class TouchUserInputEvent(UserInputEvent):
    """User input event for touch."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "touch"

    action: Literal["start", "move", "end", "cancel"] = "move"
    """Touch action represented by this event."""
    touch_id: int = 0
    """Browser touch-point identifier."""
    x: float = 0.0
    """Horizontal touch coordinate normalized to the video viewport."""
    y: float = 0.0
    """Vertical touch coordinate normalized to the video viewport."""
    pressure: float = 0.0
    """Normalized touch pressure."""
    primary: bool = False
    """Whether this is the primary touch point."""


@dataclass(frozen=True, slots=True, eq=False)
class GamepadUserInputEvent(UserInputEvent):
    """User input event for a gamepad."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "gamepad"

    action: Literal["connected", "disconnected", "state"] = "state"
    """Gamepad lifecycle or state action."""
    index: int = 0
    """Browser gamepad index."""
    controller_id: str = ""
    """Controller identifier supplied by the client."""
    mapping: str = ""
    """Controller mapping name, such as ``"standard"``."""
    axes: tuple[float, ...] = ()
    """Normalized gamepad axis values."""
    buttons: tuple[float, ...] = ()
    """Normalized analog button values."""
    pressed: tuple[bool, ...] = ()
    """Digital pressed state corresponding to ``buttons``."""


@dataclass(frozen=True, slots=True, eq=False)
class GameWheelUserInputEvent(UserInputEvent):
    """User input event for a game wheel."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "game_wheel"

    action: Literal["connected", "disconnected", "state"] = "state"
    """Wheel lifecycle or state action."""
    index: int = 0
    """Client controller index."""
    controller_id: str = ""
    """Controller identifier supplied by the client."""
    steering: float = 0.0
    """Normalized steering value in ``[-1, 1]``."""
    throttle: float = 0.0
    """Normalized throttle value in ``[0, 1]``."""
    brake: float = 0.0
    """Normalized brake value in ``[0, 1]``."""
    clutch: float = 0.0
    """Normalized clutch value in ``[0, 1]``."""
    buttons: tuple[bool, ...] = ()
    """Digital wheel button states."""


@dataclass(frozen=True, slots=True, eq=False)
class XRControllerUserInputEvent(UserInputEvent):
    """User input event for XR controllers."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "xr_controller"

    action: Literal["connected", "disconnected", "state"] = "state"
    """XR controller lifecycle or state action."""
    handedness: Literal["left", "right", "none"] = "none"
    """Hand associated with the controller."""
    controller_id: str = ""
    """Controller identifier supplied by the client."""
    axes: tuple[float, ...] = ()
    """Normalized XR controller axis values."""
    buttons: tuple[float, ...] = ()
    """Normalized analog button values."""
    pressed: tuple[bool, ...] = ()
    """Digital pressed state corresponding to ``buttons``."""
    position: tuple[float, float, float] | None = None
    """Optional controller position in client XR space."""
    orientation: tuple[float, float, float, float] | None = None
    """Optional controller quaternion in client XR space."""


@dataclass(frozen=True, slots=True, eq=False)
class UnknownUserInputEvent(UserInputEvent):
    """User input event for an unknown input modality."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "unknown"
