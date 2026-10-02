# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete user input events for supported input modalities."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from flashdreams.api_v2.user_input_event import UserInputEvent

MAX_SELECTED_FILE_BYTES = 32 * 1024 * 1024
"""Hard ceiling for one selected file. Windows clamp requested budgets to this."""


def clamp_selected_file_max_bytes(max_bytes: int | None) -> int:
    """Return a file-size budget that cannot exceed ``MAX_SELECTED_FILE_BYTES``.

    Args:
        max_bytes: Requested budget in bytes, or ``None`` for the ceiling.

    Returns:
        The requested budget when it is below the ceiling, otherwise the ceiling.

    Raises:
        TypeError: ``max_bytes`` is not ``None`` or an integer.
        ValueError: ``max_bytes`` is not positive.
    """
    if max_bytes is None:
        return MAX_SELECTED_FILE_BYTES
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer.")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0.")
    return min(max_bytes, MAX_SELECTED_FILE_BYTES)


def normalize_selected_file_accept(accept: Sequence[str] = ()) -> tuple[str, ...]:
    """Return accepted filename suffixes such as ``.png``.

    Args:
        accept: Suffixes the client may choose. Empty means any type.

    Returns:
        The suffixes as a tuple, unchanged except for the sequence type.

    Raises:
        TypeError: ``accept`` is not a sequence of strings.
        ValueError: A suffix is missing the leading ``.`` or contains a path.
    """
    if isinstance(accept, str) or not isinstance(accept, Sequence):
        raise TypeError("accept must be a sequence of suffixes.")
    suffixes: list[str] = []
    for suffix in accept:
        if not isinstance(suffix, str):
            raise TypeError("accept suffixes must be strings.")
        if (
            not suffix.startswith(".")
            or suffix == "."
            or "/" in suffix
            or "\\" in suffix
        ):
            raise ValueError(
                "accept suffixes must look like ``.png`` and cannot contain a path."
            )
        suffixes.append(suffix)
    return tuple(suffixes)


def selected_file_suffix_allowed(name: str, accept: tuple[str, ...]) -> bool:
    """Return whether ``name`` matches ``accept``.

    Empty ``accept`` allows any name. Comparison is case-insensitive.
    """
    if not accept:
        return True
    lower = name.lower()
    return any(lower.endswith(suffix.lower()) for suffix in accept)


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
    """Horizontal pointer coordinate in video-width units."""
    y: float = 0.0
    """Vertical pointer coordinate in video-height units."""
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
class QueryStringUserInputEvent(UserInputEvent):
    """Query string supplied by a browser when it connects."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "query_string"

    query_string: str = ""
    """Raw query string without its leading question mark."""


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


class SelectedFilesStatus(Enum):
    """Outcome of one client file-selector request."""

    OK = "ok"
    """The client chose a file that passed type and size checks."""

    CANCELLED = "cancelled"
    """The user dismissed the selector."""

    TOO_LARGE = "too_large"
    """The chosen file exceeded ``max_bytes``."""

    DISALLOWED_TYPE = "disallowed_type"
    """The chosen file did not match ``accept``. Checked before size."""

    UNAVAILABLE = "unavailable"
    """No picker could run: headless output, a dropped peer, or an unreadable file."""


def selected_file_policy_status(
    name: str,
    size: int | None,
    *,
    accept: tuple[str, ...] = (),
    max_bytes: int,
) -> SelectedFilesStatus | None:
    """Return why a chosen file is rejected, or ``None`` if it is allowed.

    Windows call this so every client uses the same order: type before size.
    A too-large disallowed file is ``DISALLOWED_TYPE``. Applications only read
    :class:`SelectedFilesUserInputEvent` ``status``. Pass ``size=None`` to
    check only the name.

    Args:
        name: File name to match against ``accept``.
        size: File size in bytes, or ``None`` to skip the size check.
        accept: Filename suffixes such as ``.png``. Empty allows any type.
        max_bytes: Maximum allowed size in bytes.

    Returns:
        ``DISALLOWED_TYPE``, ``TOO_LARGE``, or ``None`` when the file passes.
    """
    if not selected_file_suffix_allowed(name, accept):
        return SelectedFilesStatus.DISALLOWED_TYPE
    if size is not None and size > max_bytes:
        return SelectedFilesStatus.TOO_LARGE
    return None


@dataclass(frozen=True, slots=True)
class SelectedFile:
    """One file chosen by a client file selector."""

    name: str
    """File name shown to the application."""

    data: bytes
    """File contents."""


@dataclass(frozen=True, slots=True, eq=False)
class SelectedFilesUserInputEvent(UserInputEvent):
    """Files chosen for one :meth:`IClientWindow.request_selected_files` call.

    ``status`` is why the request completed. ``OK`` is the only value with
    files; every other value uses an empty ``files`` tuple.
    """

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "selected_files"

    request_id: str
    """Identifier supplied with the matching file-selection request."""

    status: SelectedFilesStatus
    """Why this request completed. ``OK`` is the only value with files."""

    files: tuple[SelectedFile, ...] = ()
    """Chosen files. Empty unless ``status`` is ``OK``."""

    def __post_init__(self) -> None:
        """Reject an ``ok`` event without files, or files on a failed pick."""
        has_files = bool(self.files)
        if self.status is SelectedFilesStatus.OK:
            if not has_files:
                raise ValueError("ok selected-files events must include files.")
            return
        if has_files:
            raise ValueError("failed selected-files events must not include files.")


@dataclass(frozen=True, slots=True, eq=False)
class UnknownUserInputEvent(UserInputEvent):
    """User input event for an unknown input modality."""

    @classmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        return "unknown"
