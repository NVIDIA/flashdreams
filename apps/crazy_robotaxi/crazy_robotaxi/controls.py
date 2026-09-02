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

"""Versioned user control bindings and input evaluation."""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import Field, dataclass, field, fields, replace
from pathlib import Path
from typing import Literal, cast

from omnidreams_game_engine.types import DriverCommand
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    GamepadUserInputEvent,
    GameWheelUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

ControlDevice = Literal["keyboard", "gamepad", "wheel"]
ControlDirection = Literal["negative", "positive", "bidirectional"]
ControlAction = Literal[
    "restart", "toggle_hints", "style", "weather", "coins", "obstacle"
]

_RESERVED_KEYS = frozenset({"escape", "backspace", "delete"})
"""Keys reserved for cancelling or clearing a binding capture."""

_WHEEL_AXES = frozenset({"steering", "throttle", "brake", "clutch"})
"""Semantic axes carried by :class:`GameWheelUserInputEvent`."""

_DISCRETE_ACTIONS: dict[str, ControlAction] = {
    "restart": "restart",
    "toggle_hints": "toggle_hints",
    "cycle_style": "style",
    "cycle_weather": "weather",
    "toggle_coins": "coins",
    "spawn_obstacle": "obstacle",
}


class ControlsError(ValueError):
    """Invalid user-authored control bindings."""


@dataclass(frozen=True, slots=True)
class InputBinding:
    """One keyboard key, device button, or directed analog control."""

    kind: Literal["key", "button", "axis"]
    """Input value kind."""

    code: str | int
    """Canonical key, semantic wheel axis, or zero-based input index."""

    direction: ControlDirection = "positive"
    """Axis direction used as activation, or ``bidirectional`` for steering."""

    invert: bool = False
    """Whether a bidirectional steering value changes sign."""


BindingSlots = tuple[InputBinding | None, InputBinding | None]


def _key(name: str) -> InputBinding:
    return InputBinding("key", name)


def _button(index: int) -> InputBinding:
    return InputBinding("button", index)


def _axis(
    code: str | int,
    *,
    direction: ControlDirection = "positive",
    invert: bool = False,
) -> InputBinding:
    return InputBinding("axis", code, direction=direction, invert=invert)


def _bindings_field(
    default: BindingSlots,
    label: str,
    *,
    kind: Literal["scalar", "steering"] = "scalar",
    feature: str | None = None,
) -> BindingSlots:
    return field(
        default=default,
        metadata={"label": label, "kind": kind, "feature": feature},
    )


@dataclass(frozen=True, slots=True)
class KeyboardControls:
    """Keyboard gameplay bindings."""

    drive_forward: BindingSlots = _bindings_field(
        (_key("w"), _key("up")), "DRIVE FORWARD"
    )
    """Keys that drive forward."""

    reverse: BindingSlots = _bindings_field((_key("s"), _key("down")), "REVERSE")
    """Keys that brake or drive in reverse."""

    steer_left: BindingSlots = _bindings_field((_key("a"), _key("left")), "STEER LEFT")
    """Keys that steer left."""

    steer_right: BindingSlots = _bindings_field(
        (_key("d"), _key("right")), "STEER RIGHT"
    )
    """Keys that steer right."""

    handbrake: BindingSlots = _bindings_field((_key("space"), None), "HANDBRAKE")
    """Keys that apply the handbrake."""

    restart: BindingSlots = _bindings_field((_key("r"), None), "RESTART GAME")
    """Keys that restart the current game."""

    toggle_hints: BindingSlots = _bindings_field(
        (_key("h"), None), "SHOW OR HIDE CONTROL HINTS"
    )
    """Keys that toggle gameplay control hints."""

    cycle_style: BindingSlots = _bindings_field(
        (_key("k"), None), "CYCLE STYLE", feature="style"
    )
    """Keys that request the next live-edit style."""

    cycle_weather: BindingSlots = _bindings_field(
        (_key("v"), None), "CYCLE WEATHER", feature="weather"
    )
    """Keys that request the next live-edit weather state."""

    toggle_coins: BindingSlots = _bindings_field(
        (_key("c"), None), "TOGGLE COINS", feature="coins"
    )
    """Keys that toggle live-edit coins."""

    spawn_obstacle: BindingSlots = _bindings_field(
        (_key("o"), None), "SPAWN OBSTACLE", feature="obstacle"
    )
    """Keys that spawn a live-edit obstacle."""


@dataclass(frozen=True, slots=True)
class GamepadControls:
    """Standard-gamepad gameplay bindings."""

    steer: BindingSlots = _bindings_field(
        (_axis(0, direction="bidirectional", invert=True), None),
        "STEER",
        kind="steering",
    )
    """Axes that steer, oriented so positive means left."""

    throttle: BindingSlots = _bindings_field((_button(7), None), "THROTTLE")
    """Controls that apply forward throttle."""

    brake: BindingSlots = _bindings_field((_button(6), None), "BRAKE")
    """Controls that apply the service brake."""

    reverse: BindingSlots = _bindings_field((_button(5), None), "REVERSE (HOLD)")
    """Controls that select reverse while held."""

    handbrake: BindingSlots = _bindings_field((None, None), "HANDBRAKE")
    """Controls that apply the handbrake."""

    restart: BindingSlots = _bindings_field((_button(9), None), "RESTART GAME")
    """Controls that restart the current game."""

    toggle_hints: BindingSlots = _bindings_field(
        (None, None), "SHOW OR HIDE CONTROL HINTS"
    )
    """Controls that toggle gameplay control hints."""

    cycle_style: BindingSlots = _bindings_field(
        (None, None), "CYCLE STYLE", feature="style"
    )
    """Controls that request the next live-edit style."""

    cycle_weather: BindingSlots = _bindings_field(
        (None, None), "CYCLE WEATHER", feature="weather"
    )
    """Controls that request the next live-edit weather state."""

    toggle_coins: BindingSlots = _bindings_field(
        (None, None), "TOGGLE COINS", feature="coins"
    )
    """Controls that toggle live-edit coins."""

    spawn_obstacle: BindingSlots = _bindings_field(
        (None, None), "SPAWN OBSTACLE", feature="obstacle"
    )
    """Controls that spawn a live-edit obstacle."""


@dataclass(frozen=True, slots=True)
class WheelControls:
    """Semantic steering-wheel gameplay bindings."""

    steer: BindingSlots = _bindings_field(
        (_axis("steering", direction="bidirectional", invert=True), None),
        "STEER",
        kind="steering",
    )
    """Semantic axes that steer, oriented so positive means left."""

    throttle: BindingSlots = _bindings_field((_axis("throttle"), None), "THROTTLE")
    """Controls that apply forward throttle."""

    brake: BindingSlots = _bindings_field((_axis("brake"), None), "BRAKE")
    """Controls that apply the service brake."""

    reverse: BindingSlots = _bindings_field((None, None), "REVERSE (HOLD)")
    """Controls that select reverse while held."""

    handbrake: BindingSlots = _bindings_field((None, None), "HANDBRAKE")
    """Controls that apply the handbrake."""

    restart: BindingSlots = _bindings_field((None, None), "RESTART GAME")
    """Controls that restart the current game."""

    toggle_hints: BindingSlots = _bindings_field(
        (None, None), "SHOW OR HIDE CONTROL HINTS"
    )
    """Controls that toggle gameplay control hints."""

    cycle_style: BindingSlots = _bindings_field(
        (None, None), "CYCLE STYLE", feature="style"
    )
    """Controls that request the next live-edit style."""

    cycle_weather: BindingSlots = _bindings_field(
        (None, None), "CYCLE WEATHER", feature="weather"
    )
    """Controls that request the next live-edit weather state."""

    toggle_coins: BindingSlots = _bindings_field(
        (None, None), "TOGGLE COINS", feature="coins"
    )
    """Controls that toggle live-edit coins."""

    spawn_obstacle: BindingSlots = _bindings_field(
        (None, None), "SPAWN OBSTACLE", feature="obstacle"
    )
    """Controls that spawn a live-edit obstacle."""


DeviceControls = KeyboardControls | GamepadControls | WheelControls


@dataclass(frozen=True, slots=True)
class ControlsConfig:
    """Active bindings for every supported input device."""

    keyboard: KeyboardControls = KeyboardControls()
    """Keyboard bindings."""

    gamepad: GamepadControls = GamepadControls()
    """Standard-gamepad bindings."""

    wheel: WheelControls = WheelControls()
    """Semantic steering-wheel bindings."""

    def for_device(self, device: ControlDevice) -> DeviceControls:
        """Return bindings for ``device``."""
        return getattr(self, device)


def default_controls_dir() -> Path:
    """Return the platform-style per-user controls directory."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "crazy-robotaxi" / "controls"


def controls_fields(settings: DeviceControls) -> tuple[Field[BindingSlots], ...]:
    """Return action fields in menu and schema order."""
    return cast(tuple[Field[BindingSlots], ...], fields(settings))


def control_label(item: Field[BindingSlots]) -> str:
    """Return the player-facing action label for ``item``."""
    return str(item.metadata["label"])


@dataclass(slots=True)
class ControlsDocument:
    """Round-trip YAML document for one input device."""

    device: ControlDevice
    """Input device described by this document."""

    path: Path
    """Resolved document path."""

    defaults: DeviceControls
    """Built-in device defaults."""

    settings: DeviceControls
    """Saved device bindings."""

    _yaml: YAML
    """Round-trip YAML parser retaining authored comments."""

    _document: CommentedMap
    """Mutable round-trip mapping synchronized during saves."""

    @classmethod
    def load(cls, path: Path, device: ControlDevice) -> "ControlsDocument":
        """Load sparse bindings for one input device."""
        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        resolved = path.expanduser().resolve()
        if resolved.exists():
            try:
                raw = yaml.load(resolved.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ControlsError(f"Could not parse {resolved}: {exc}") from exc
            if raw is None:
                raw = CommentedMap()
            if not isinstance(raw, CommentedMap):
                raise ControlsError(f"{resolved} must contain a YAML mapping")
            document = raw
        else:
            document = CommentedMap()
            document.yaml_set_start_comment(
                f"Crazy Robotaxi {device} controls. Omitted actions use defaults."
            )
        version = document.get("schema_version", 1)
        if type(version) is not int or version != 1:
            raise ControlsError(f"{resolved}: schema_version must be 1")
        defaults = ControlsConfig().for_device(device)
        values = {
            key: value for key, value in document.items() if key != "schema_version"
        }
        settings = _overlay_controls(defaults, values, device)
        return cls(device, resolved, defaults, settings, yaml, document)

    def save(self, settings: DeviceControls) -> None:
        """Validate and atomically save sparse device bindings."""
        _validate_controls(settings, self.device)
        desired = CommentedMap()
        desired["schema_version"] = 1
        for item in controls_fields(settings):
            value = getattr(settings, item.name)
            if value != getattr(self.defaults, item.name):
                desired[item.name] = [
                    None if binding is None else _binding_to_yaml(binding, self.device)
                    for binding in value
                ]
        _sync_mapping(self._document, desired)
        buffer = io.StringIO()
        self._yaml.dump(self._document, buffer)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(buffer.getvalue())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        self.settings = settings


def load_controls_documents(
    directory: Path,
) -> dict[ControlDevice, ControlsDocument]:
    """Load all device documents from ``directory``."""
    resolved = directory.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ControlsError(f"Controls path is not a directory: {resolved}")
    return {
        device: ControlsDocument.load(resolved / f"{device}.yaml", device)
        for device in ("keyboard", "gamepad", "wheel")
    }


def controls_config(
    documents: Mapping[ControlDevice, ControlsDocument],
) -> ControlsConfig:
    """Build active controls from loaded device documents."""
    return ControlsConfig(
        keyboard=cast(KeyboardControls, documents["keyboard"].settings),
        gamepad=cast(GamepadControls, documents["gamepad"].settings),
        wheel=cast(WheelControls, documents["wheel"].settings),
    )


def canonical_key(value: str) -> str:
    """Return a stable key name without collapsing arrow keys into WASD."""
    lowered = value.lower()
    if lowered == " " or lowered.strip() == "spacebar":
        return "space"
    normalized = lowered.strip()
    return {
        "arrowup": "up",
        "arrowdown": "down",
        "arrowleft": "left",
        "arrowright": "right",
        "esc": "escape",
        "del": "delete",
    }.get(normalized, normalized)


def keyboard_driver_command(
    settings: KeyboardControls, pressed_keys: set[str]
) -> DriverCommand:
    """Map retained keyboard state through configured driving bindings."""
    pressed = {canonical_key(key) for key in pressed_keys}
    forward = _keyboard_action_value(settings.drive_forward, pressed) > 0.5
    reverse = _keyboard_action_value(settings.reverse, pressed) > 0.5
    handbrake = _keyboard_action_value(settings.handbrake, pressed) > 0.5
    steer = _keyboard_action_value(settings.steer_left, pressed)
    steer -= _keyboard_action_value(settings.steer_right, pressed)
    return DriverCommand(
        throttle=1.0 if forward != reverse and not handbrake else 0.0,
        brake=1.0 if handbrake else 0.0,
        steer=steer,
        reverse=reverse and not forward,
        manual_control=handbrake,
    )


def keyboard_drive_key(settings: KeyboardControls, key: str) -> str | None:
    """Canonicalize ``key`` only when it is bound to a driving action."""
    normalized = canonical_key(key)
    driving_actions = (
        settings.drive_forward,
        settings.reverse,
        settings.steer_left,
        settings.steer_right,
        settings.handbrake,
    )
    return (
        normalized
        if any(
            binding is not None and binding.code == normalized
            for slots in driving_actions
            for binding in slots
        )
        else None
    )


def gamepad_driver_command(
    settings: GamepadControls, event: GamepadUserInputEvent
) -> DriverCommand | None:
    """Map one standard-gamepad snapshot through configured bindings."""
    if event.action == "disconnected":
        return None
    if event.action != "state":
        return None
    return DriverCommand(
        throttle=_event_action_value(settings.throttle, event),
        brake=_event_action_value(settings.brake, event),
        steer=_steering_value(settings.steer, event),
        reverse=_event_action_value(settings.reverse, event) > 0.5,
        handbrake=_event_action_value(settings.handbrake, event) > 0.5,
        steer_is_direct=True,
        manual_control=True,
    )


def wheel_driver_command(
    settings: WheelControls, event: GameWheelUserInputEvent
) -> DriverCommand | None:
    """Map one semantic wheel snapshot through configured bindings."""
    if event.action == "disconnected":
        return None
    if event.action != "state":
        return None
    return DriverCommand(
        throttle=_event_action_value(settings.throttle, event),
        brake=_event_action_value(settings.brake, event),
        steer=_steering_value(settings.steer, event),
        reverse=_event_action_value(settings.reverse, event) > 0.5,
        handbrake=_event_action_value(settings.handbrake, event) > 0.5,
        steer_is_direct=True,
        manual_control=True,
    )


@dataclass(slots=True)
class BoundActionState:
    """Detect configured rising-edge actions across input snapshots."""

    controls: ControlsConfig
    """Active process-start bindings."""

    _pressed_keys: set[str] = field(default_factory=set)
    """Canonical keyboard keys currently held."""

    _active_by_controller: dict[tuple[ControlDevice, int, str], set[ControlAction]] = (
        field(default_factory=dict)
    )
    """Actions active in each controller's previous snapshot."""

    def apply(self, events: UserInputEvents) -> frozenset[ControlAction]:
        """Consume input events and return newly activated actions."""
        activated: set[ControlAction] = set()
        for event in events.get_events():
            if isinstance(event, FocusUserInputEvent):
                if not event.focused:
                    self._pressed_keys.clear()
                continue
            if isinstance(event, KeyboardUserInputEvent):
                key = canonical_key(str(event.key))
                was_pressed = key in self._pressed_keys
                if event.state is KeyboardInputState.RELEASED:
                    self._pressed_keys.discard(key)
                    continue
                self._pressed_keys.add(key)
                if not was_pressed:
                    activated.update(
                        _keyboard_discrete_actions(self.controls.keyboard, key)
                    )
                continue
            if isinstance(event, GamepadUserInputEvent):
                activated.update(
                    self._controller_actions("gamepad", self.controls.gamepad, event)
                )
            elif isinstance(event, GameWheelUserInputEvent):
                activated.update(
                    self._controller_actions("wheel", self.controls.wheel, event)
                )
        return frozenset(activated)

    def _controller_actions(
        self,
        device: Literal["gamepad", "wheel"],
        settings: GamepadControls | WheelControls,
        event: GamepadUserInputEvent | GameWheelUserInputEvent,
    ) -> set[ControlAction]:
        key = (device, event.index, event.controller_id)
        if event.action in {"connected", "disconnected"}:
            self._active_by_controller.pop(key, None)
            return set()
        if event.action != "state":
            return set()
        current = {
            action
            for item in controls_fields(settings)
            if (action := _DISCRETE_ACTIONS.get(item.name)) is not None
            and _event_action_value(getattr(settings, item.name), event) > 0.5
        }
        previous = self._active_by_controller.get(key, set())
        self._active_by_controller[key] = current
        return current - previous


def update_binding(
    settings: DeviceControls,
    action: str,
    slot: int,
    binding: InputBinding | None,
) -> DeviceControls:
    """Replace one slot and swap an existing duplicate binding."""
    if slot not in (0, 1):
        raise IndexError("binding slot must be 0 or 1")
    if action not in {item.name for item in controls_fields(settings)}:
        raise KeyError(action)
    slots = list(cast(BindingSlots, getattr(settings, action)))
    previous = slots[slot]
    duplicate: tuple[str, int] | None = None
    if binding is not None:
        for item in controls_fields(settings):
            for other_slot, current in enumerate(getattr(settings, item.name)):
                if current == binding and (item.name, other_slot) != (action, slot):
                    duplicate = (item.name, other_slot)
                    break
            if duplicate is not None:
                break
    slots[slot] = binding
    updated = replace(settings, **{action: cast(BindingSlots, tuple(slots))})
    if duplicate is None:
        return updated
    duplicate_action, duplicate_slot = duplicate
    duplicate_slots = list(cast(BindingSlots, getattr(updated, duplicate_action)))
    duplicate_slots[duplicate_slot] = previous
    return replace(
        updated,
        **{duplicate_action: cast(BindingSlots, tuple(duplicate_slots))},
    )


def binding_display(device: ControlDevice, binding: InputBinding | None) -> str:
    """Return a compact player-facing binding label."""
    if binding is None:
        return "UNBOUND"
    if binding.kind == "key":
        key = str(binding.code)
        return {
            "space": "SPACE",
            "up": "UP ARROW",
            "down": "DOWN ARROW",
            "left": "LEFT ARROW",
            "right": "RIGHT ARROW",
        }.get(key, key.upper())
    if binding.kind == "button":
        index = int(binding.code)
        if device == "gamepad":
            return _gamepad_button_name(index)
        return f"BUTTON {index + 1}"
    suffix = ""
    if binding.direction != "bidirectional":
        suffix = " +" if binding.direction == "positive" else " -"
    elif binding.invert:
        suffix = " (INVERTED)"
    if device == "gamepad":
        name = {
            0: "LEFT STICK X",
            1: "LEFT STICK Y",
            2: "RIGHT STICK X",
            3: "RIGHT STICK Y",
        }.get(int(binding.code), f"AXIS {int(binding.code) + 1}")
    else:
        name = str(binding.code).replace("_", " ").upper()
    return name + suffix


def capture_binding(
    device: ControlDevice,
    action_kind: Literal["scalar", "steering"],
    event: object,
    baseline: GamepadUserInputEvent | GameWheelUserInputEvent | None = None,
) -> InputBinding | None:
    """Return a binding when ``event`` crosses the capture threshold."""
    if device == "keyboard":
        if not isinstance(event, KeyboardUserInputEvent):
            return None
        if event.state is not KeyboardInputState.PRESSED:
            return None
        key = canonical_key(str(event.key))
        return None if not key or key in _RESERVED_KEYS else _key(key)
    if device == "gamepad" and isinstance(event, GamepadUserInputEvent):
        if event.action != "state":
            return None
        base = baseline if isinstance(baseline, GamepadUserInputEvent) else None
        if action_kind == "steering":
            candidate = _moved_numeric(event.axes, () if base is None else base.axes)
            if candidate is None:
                return None
            index, value = candidate
            return _axis(index, direction="bidirectional", invert=value < 0.0)
        candidate_button = _pressed_numeric(
            _gamepad_button_values(event),
            () if base is None else _gamepad_button_values(base),
        )
        if candidate_button is not None:
            return _button(candidate_button)
        candidate_axis = _moved_numeric(event.axes, () if base is None else base.axes)
        if candidate_axis is None:
            return None
        index, value = candidate_axis
        return _axis(index, direction="positive" if value > 0.0 else "negative")
    if device == "wheel" and isinstance(event, GameWheelUserInputEvent):
        if event.action != "state":
            return None
        base = baseline if isinstance(baseline, GameWheelUserInputEvent) else None
        if action_kind == "steering":
            candidate = _moved_numeric(
                (event.steering,), () if base is None else (base.steering,)
            )
            if candidate is None:
                return None
            _index, value = candidate
            return _axis(
                "steering",
                direction="bidirectional",
                invert=value < 0.0,
            )
        old_buttons = () if base is None else base.buttons
        for index, pressed in enumerate(event.buttons):
            if pressed and (index >= len(old_buttons) or not old_buttons[index]):
                return _button(index)
        axes = (event.steering, event.throttle, event.brake, event.clutch)
        old_axes = (
            ()
            if base is None
            else (base.steering, base.throttle, base.brake, base.clutch)
        )
        candidate = _moved_numeric(axes, old_axes)
        if candidate is None:
            return None
        index, value = candidate
        return _axis(
            ("steering", "throttle", "brake", "clutch")[index],
            direction="positive" if value > 0.0 else "negative",
        )
    return None


def _overlay_controls(
    defaults: DeviceControls,
    values: Mapping[str, object],
    device: ControlDevice,
) -> DeviceControls:
    known = {item.name: item for item in controls_fields(defaults)}
    unknown = sorted(set(values) - set(known), key=str)
    if unknown:
        raise ControlsError(
            f"{device} controls have unknown keys: "
            + ", ".join(str(key) for key in unknown)
        )
    updates: dict[str, BindingSlots] = {}
    for name, raw in values.items():
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ControlsError(f"{device}.{name} must be a sequence")
        if len(raw) > 2:
            raise ControlsError(f"{device}.{name} supports at most two bindings")
        item = known[name]
        parsed = [
            None
            if binding is None
            else _binding_from_yaml(
                binding,
                device,
                cast(Literal["scalar", "steering"], item.metadata["kind"]),
                f"{device}.{name}.{index}",
            )
            for index, binding in enumerate(raw)
        ]
        parsed.extend([None] * (2 - len(parsed)))
        updates[name] = cast(BindingSlots, tuple(parsed))
    settings = replace(defaults, **updates)
    _validate_controls(settings, device)
    return settings


def _binding_from_yaml(
    raw: object,
    device: ControlDevice,
    action_kind: Literal["scalar", "steering"],
    context: str,
) -> InputBinding:
    if device == "keyboard":
        if not isinstance(raw, str):
            raise ControlsError(f"{context} must be a key name")
        key = canonical_key(raw)
        if not key or key in _RESERVED_KEYS:
            raise ControlsError(f"{context} uses a reserved key")
        return _key(key)
    if not isinstance(raw, Mapping):
        raise ControlsError(f"{context} must be an axis or button mapping")
    mapping = cast(Mapping[str, object], raw)
    if "button" in mapping:
        if action_kind == "steering" or set(mapping) != {"button"}:
            raise ControlsError(f"{context} is not a valid button binding")
        button = mapping["button"]
        if type(button) is not int or button < 0:
            raise ControlsError(f"{context}.button must be a non-negative integer")
        return _button(button)
    allowed = {"axis", "invert"} if action_kind == "steering" else {"axis", "direction"}
    if "axis" not in mapping or set(mapping) - allowed:
        raise ControlsError(f"{context} is not a valid axis binding")
    axis = mapping["axis"]
    if device == "gamepad":
        if type(axis) is not int or axis < 0:
            raise ControlsError(f"{context}.axis must be a non-negative integer")
    elif not isinstance(axis, str) or axis not in _WHEEL_AXES:
        raise ControlsError(f"{context}.axis must name a semantic wheel axis")
    if action_kind == "steering":
        inverted = mapping.get("invert", False)
        if type(inverted) is not bool:
            raise ControlsError(f"{context}.invert must be a boolean")
        return _axis(axis, direction="bidirectional", invert=inverted)
    direction = mapping.get("direction", "positive")
    if direction not in ("negative", "positive"):
        raise ControlsError(f"{context}.direction must be negative or positive")
    return _axis(axis, direction=cast(ControlDirection, direction))


def _binding_to_yaml(
    binding: InputBinding, device: ControlDevice
) -> str | dict[str, object]:
    if device == "keyboard":
        return str(binding.code)
    if binding.kind == "button":
        return {"button": int(binding.code)}
    if binding.direction == "bidirectional":
        return {"axis": binding.code, "invert": binding.invert}
    return {"axis": binding.code, "direction": binding.direction}


def _validate_controls(settings: DeviceControls, device: ControlDevice) -> None:
    expected_type = {
        "keyboard": KeyboardControls,
        "gamepad": GamepadControls,
        "wheel": WheelControls,
    }[device]
    if not isinstance(settings, expected_type):
        raise ControlsError(f"{device} controls use the wrong device schema")
    seen: dict[InputBinding, str] = {}
    for item in controls_fields(settings):
        slots = getattr(settings, item.name)
        if len(slots) != 2:
            raise ControlsError(f"{device}.{item.name} must contain two slots")
        for binding in slots:
            if binding is None:
                continue
            try:
                reparsed = _binding_from_yaml(
                    _binding_to_yaml(binding, device),
                    device,
                    cast(Literal["scalar", "steering"], item.metadata["kind"]),
                    f"{device}.{item.name}",
                )
            except (ControlsError, TypeError, ValueError) as exc:
                raise ControlsError(
                    f"{device}.{item.name} has an invalid binding"
                ) from exc
            if reparsed != binding:
                raise ControlsError(f"{device}.{item.name} has an invalid binding")
            previous = seen.get(binding)
            if previous is not None:
                raise ControlsError(
                    f"{device}.{item.name} duplicates binding from {device}.{previous}"
                )
            seen[binding] = item.name


def _keyboard_action_value(slots: BindingSlots, pressed: set[str]) -> float:
    return max(
        (
            1.0
            for binding in slots
            if binding is not None and str(binding.code) in pressed
        ),
        default=0.0,
    )


def _keyboard_discrete_actions(
    settings: KeyboardControls, key: str
) -> set[ControlAction]:
    return {
        action
        for item in controls_fields(settings)
        if (action := _DISCRETE_ACTIONS.get(item.name)) is not None
        and any(
            binding is not None and binding.code == key
            for binding in getattr(settings, item.name)
        )
    }


def _event_action_value(
    slots: BindingSlots,
    event: GamepadUserInputEvent | GameWheelUserInputEvent,
) -> float:
    return max(
        (_binding_value(binding, event) for binding in slots if binding is not None),
        default=0.0,
    )


def _steering_value(
    slots: BindingSlots,
    event: GamepadUserInputEvent | GameWheelUserInputEvent,
) -> float:
    values = [
        _binding_value(binding, event) for binding in slots if binding is not None
    ]
    return max(values, key=abs, default=0.0)


def _binding_value(
    binding: InputBinding,
    event: GamepadUserInputEvent | GameWheelUserInputEvent,
) -> float:
    if binding.kind == "button":
        index = int(binding.code)
        if isinstance(event, GamepadUserInputEvent):
            analog = event.buttons[index] if index < len(event.buttons) else 0.0
            digital = index < len(event.pressed) and event.pressed[index]
            return max(analog, float(digital))
        return float(event.buttons[index]) if index < len(event.buttons) else 0.0
    if isinstance(event, GamepadUserInputEvent):
        index = int(binding.code)
        value = event.axes[index] if index < len(event.axes) else 0.0
    else:
        value = float(getattr(event, str(binding.code), 0.0))
    if binding.direction == "bidirectional":
        return -value if binding.invert else value
    return max(0.0, value if binding.direction == "positive" else -value)


def _moved_numeric(
    values: Sequence[float], baseline: Sequence[float]
) -> tuple[int, float] | None:
    moved = [
        (abs(value), index, value)
        for index, value in enumerate(values)
        if abs(value) >= 0.5
        and abs(baseline[index] if index < len(baseline) else 0.0) < 0.5
    ]
    if not moved:
        return None
    magnitude, index, value = max(moved)
    return index, value


def _pressed_numeric(values: Sequence[float], baseline: Sequence[float]) -> int | None:
    return next(
        (
            index
            for index, value in enumerate(values)
            if value >= 0.5 and (index >= len(baseline) or baseline[index] < 0.5)
        ),
        None,
    )


def _gamepad_button_values(event: GamepadUserInputEvent) -> tuple[float, ...]:
    """Return button values including clients that only report digital state."""
    size = max(len(event.buttons), len(event.pressed))
    return tuple(
        max(
            event.buttons[index] if index < len(event.buttons) else 0.0,
            float(event.pressed[index]) if index < len(event.pressed) else 0.0,
        )
        for index in range(size)
    )


def _gamepad_button_name(index: int) -> str:
    return {
        0: "A / CROSS / B",
        1: "B / CIRCLE / A",
        2: "X / SQUARE / Y",
        3: "Y / TRIANGLE / X",
        4: "L / LB / L1",
        5: "R / RB / R1",
        6: "LT / L2 / ZL",
        7: "RT / R2 / ZR",
        8: "BACK / SHARE / MINUS",
        9: "START / MENU / PLUS",
        10: "LEFT STICK BUTTON",
        11: "RIGHT STICK BUTTON",
        12: "D-PAD UP",
        13: "D-PAD DOWN",
        14: "D-PAD LEFT",
        15: "D-PAD RIGHT",
        16: "GUIDE",
    }.get(index, f"BUTTON {index + 1}")


def _sync_mapping(target: CommentedMap, desired: Mapping[str, object]) -> None:
    for key in list(target):
        if key not in desired:
            del target[key]
    for key, value in desired.items():
        if key not in target or target[key] != value:
            target[key] = value


__all__ = [
    "BindingSlots",
    "BoundActionState",
    "ControlAction",
    "ControlDevice",
    "ControlsConfig",
    "ControlsDocument",
    "ControlsError",
    "DeviceControls",
    "GamepadControls",
    "InputBinding",
    "KeyboardControls",
    "WheelControls",
    "binding_display",
    "canonical_key",
    "capture_binding",
    "control_label",
    "controls_config",
    "controls_fields",
    "default_controls_dir",
    "gamepad_driver_command",
    "keyboard_driver_command",
    "keyboard_drive_key",
    "load_controls_documents",
    "update_binding",
    "wheel_driver_command",
]
