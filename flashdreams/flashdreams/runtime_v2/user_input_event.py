# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input events, each a timestamp plus the data for one input modality."""

from dataclasses import dataclass

from numpy import uint64

from flashdreams.api_v2.user_input_event_data import UserInputEventData


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

    key: str = ""
    """Identifier of the key this event refers to, e.g. ``"r"``."""
    pressed: bool = False
    """Whether the key is down; ``False`` marks a key-up edge."""


@dataclass(frozen=True, slots=True, eq=False)
class ActivationUserInputEventData(UserInputEventData):
    """User input event data for an explicitly activated controller."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "activation"

    active: bool = False
    """Whether the browser currently owns active controls."""


@dataclass(frozen=True, slots=True, eq=False)
class MouseUserInputEventData(UserInputEventData):
    """User input event data for relative mouse motion."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "mouse"

    movement_x: float = 0.0
    """Horizontal relative motion since the previous browser event."""
    movement_y: float = 0.0
    """Vertical relative motion since the previous browser event."""


@dataclass(frozen=True, slots=True, eq=False)
class CloseUserInputEventData(UserInputEventData):
    """The client explicitly asked to end the run.

    A window reports this for its Stop button or a quit shortcut.
    ``run_session`` stops the run when it sees one.
    """

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "close"


@dataclass(frozen=True, slots=True, eq=False)
class ResetUserInputEventData(UserInputEventData):
    """The client asked to start the run over.

    ``run_session`` calls :meth:`ISession.reset` when it sees one, and the step
    index starts again from zero. The window stays open.
    """

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "reset"


# Below are stubbed input event data implementations for future implementation.
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
