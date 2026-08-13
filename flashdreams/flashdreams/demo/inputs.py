# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named pull-based input state for public demo IO handlers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from flashdreams.runtime.demo.session_inputs import UserInputWindow
from flashdreams.runtime.inputs import UserInputEvent, UserInputs
from flashdreams.runtime.keyboard import (
    DEFAULT_SUPPORTED_KEYS,
    DRIVING_SUPPORTED_KEYS,
    KeyboardState,
    WSAD_SUPPORTED_KEYS,
    normalize_key,
)


class InputName(str, Enum):
    """Closed public names for pull-based input state queries."""

    KEYBOARD = "keyboard"
    MOUSE_POSITION = "mouse_position"
    MOUSE_BUTTON = "mouse_button"
    HEAD_POSITION = "head_position"
    HAND_POSITION = "hand_position"


@dataclass(frozen=True, kw_only=True, slots=True)
class KeyboardInputState:
    """Keyboard state derived from one deterministic user-input window."""

    pressed_keys: frozenset[str] = field(default_factory=frozenset)
    effective_keys: frozenset[str] = field(default_factory=frozenset)

    def is_pressed(self, key: str) -> bool:
        """Return whether ``key`` is pressed in this state view."""
        return normalize_key(key) in self.pressed_keys


class InputStateDecoder(Protocol):
    """Decode one named input state from the current user-input window."""

    name: InputName

    def state_from_window(
        self,
        *,
        modality: str,
        window: UserInputWindow,
    ) -> Any:
        """Return the named input state for ``window``."""
        ...


class InputStateDecoderRegistry:
    """Registry of named input decoders used by public IO handlers."""

    def __init__(self, decoders: Sequence[InputStateDecoder] = ()) -> None:
        self._decoders: dict[InputName, InputStateDecoder] = {}
        for decoder in decoders:
            self.register(decoder)

    def register(self, decoder: InputStateDecoder) -> None:
        """Register one named input decoder."""
        decoder_name = getattr(decoder, "name", None)
        if not isinstance(decoder_name, InputName) or not callable(
            getattr(decoder, "state_from_window", None)
        ):
            raise TypeError("decoder must implement the InputStateDecoder protocol.")
        if decoder_name in self._decoders:
            raise ValueError(f"Input decoder {decoder_name.value!r} is already set.")
        self._decoders[decoder_name] = decoder

    @property
    def decoders(self) -> Mapping[InputName, InputStateDecoder]:
        """Return registered decoders by name."""
        return dict(self._decoders)

    def state_from_window(
        self,
        *,
        modality: str,
        name: InputName | str,
        window: UserInputWindow,
    ) -> Any:
        """Return one named state view over ``window``."""
        key_name = _legacy_key_name(name)
        if key_name is not None:
            keyboard = self.state_from_window(
                modality=modality,
                name=InputName.KEYBOARD,
                window=window,
            )
            if isinstance(keyboard, KeyboardInputState):
                return keyboard.is_pressed(key_name)
            return False

        input_name = _coerce_input_name(name)
        if input_name is None:
            return None
        decoder = self._decoders.get(input_name)
        if decoder is None:
            return None
        return decoder.state_from_window(modality=modality, window=window)


@dataclass(frozen=True, slots=True)
class KeyboardInputStateDecoder:
    """Decode keyboard edge events into a pullable level-triggered state."""

    name: InputName = InputName.KEYBOARD

    def state_from_window(
        self,
        *,
        modality: str,
        window: UserInputWindow,
    ) -> KeyboardInputState | None:
        if modality.strip().lower() not in {"", "keyboard"}:
            return None

        inputs = window.inputs
        initial_keys = _snapshot_pressed_keys(inputs.snapshot)
        state = KeyboardState(
            pressed_keys=set(initial_keys),
            supported_keys=_supported_keys(inputs, initial_keys),
        )
        for event in inputs.events:
            keyboard_event = _keyboard_event_name(event)
            if keyboard_event is None:
                continue
            key = event.payload.get("key")
            if isinstance(key, str):
                state.apply_event(event=keyboard_event, key=key)
        return KeyboardInputState(
            pressed_keys=state.snapshot(),
            effective_keys=state.resolved_effective_keys(),
        )


@dataclass(frozen=True, slots=True)
class SnapshotInputStateDecoder:
    """Return named non-keyboard values from the current window snapshot."""

    name: InputName

    def state_from_window(
        self,
        *,
        modality: str,
        window: UserInputWindow,
    ) -> Any:
        del modality
        snapshot = window.inputs.snapshot
        if self.name.value in snapshot:
            return snapshot[self.name.value]
        nested = snapshot.get(self.name.value.split("_", maxsplit=1)[0])
        if isinstance(nested, Mapping):
            return nested.get(self.name.value)
        return None


def create_default_input_state_decoder_registry() -> InputStateDecoderRegistry:
    """Create the standard named input decoder registry."""
    return InputStateDecoderRegistry(
        (
            KeyboardInputStateDecoder(),
            SnapshotInputStateDecoder(InputName.MOUSE_POSITION),
            SnapshotInputStateDecoder(InputName.MOUSE_BUTTON),
            SnapshotInputStateDecoder(InputName.HEAD_POSITION),
            SnapshotInputStateDecoder(InputName.HAND_POSITION),
        )
    )


def input_state_from_window(
    window: UserInputWindow,
    *,
    modality: str,
    name: InputName | str,
) -> Any:
    """Decode one named input state from ``window`` using default decoders."""
    return create_default_input_state_decoder_registry().state_from_window(
        modality=modality,
        name=name,
        window=window,
    )


def _coerce_input_name(name: InputName | str) -> InputName | None:
    if isinstance(name, InputName):
        return name
    try:
        return InputName(name)
    except ValueError:
        return None


def _legacy_key_name(name: InputName | str) -> str | None:
    if not isinstance(name, str):
        return None
    if not name.startswith("key_"):
        return None
    return name.removeprefix("key_")


def _snapshot_pressed_keys(snapshot: Mapping[str, Any]) -> frozenset[str]:
    value = snapshot.get("pressed_keys")
    if value is None:
        keyboard = snapshot.get("keyboard")
        if isinstance(keyboard, Mapping):
            value = keyboard.get("pressed_keys")
    if isinstance(value, str):
        return frozenset({normalize_key(value)})
    if isinstance(value, Sequence):
        return frozenset(
            normalize_key(key)
            for key in value
            if isinstance(key, str) and key.strip()
        )
    return frozenset()


def _supported_keys(
    inputs: UserInputs,
    initial_keys: frozenset[str],
) -> frozenset[str]:
    event_keys = {
        normalize_key(key)
        for event in inputs.events
        for key in (event.payload.get("key"),)
        if isinstance(key, str) and key.strip()
    }
    return frozenset(
        set(DEFAULT_SUPPORTED_KEYS)
        | set(DRIVING_SUPPORTED_KEYS)
        | set(WSAD_SUPPORTED_KEYS)
        | set(initial_keys)
        | event_keys
    )


def _keyboard_event_name(event: UserInputEvent) -> str | None:
    normalized = event.event_type.strip().lower().replace(".", "_")
    if normalized in {"key_down", "keyboard_keydown", "keydown"}:
        return "keydown"
    if normalized in {"key_up", "keyboard_keyup", "keyup"}:
        return "keyup"
    return None


__all__ = [
    "InputName",
    "InputStateDecoder",
    "InputStateDecoderRegistry",
    "KeyboardInputState",
    "KeyboardInputStateDecoder",
    "SnapshotInputStateDecoder",
    "create_default_input_state_decoder_registry",
    "input_state_from_window",
]
