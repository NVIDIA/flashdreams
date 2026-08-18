# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw device input to canonical modality conversion.

This is the ``raw input -> canonicalized input`` leg. Applications consume
:class:`~flashdreams.runtime.inputs.CanonicalInputs`; they never read raw device
events. Adding a keyboard, gamepad, or force-feedback wheel is therefore a
:meth:`InputCanonicalizer.register` call that touches no application, mapping,
or model code.

Converters are stateful, because HID input is edge-triggered while per-step
conditioning is level-triggered: a key held across a step emits no events yet
still means full throttle. Feed windows in session order and call
:meth:`InputCanonicalizer.reset` at a rollout boundary; replaying the same
window sequence then reproduces the same canonical inputs.

This layer covers live user control only. Global conditioning such as a prompt
or conditioning frame is application-owned and reaches ``InferenceInput``
directly, without passing through canonicalization.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.inputs import (
    CanonicalInputs,
    CanonicalInputSchema,
    CanonicalModality,
    TimeWindow,
    UserInputCapability,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.keyboard import (
    DEFAULT_SUPPORTED_KEYS,
    DRIVING_SUPPORTED_KEYS,
    KeyboardState,
    normalize_key,
)

DriverBindings = Mapping[str, frozenset[str]]
CameraBindings = Mapping[str, tuple[str, str]]

DEFAULT_DRIVING_BINDINGS: DriverBindings = MappingProxyType(
    {
        "throttle": frozenset({"w", "up"}),
        "brake": frozenset({"s", "down"}),
        "steer_left": frozenset({"a", "left"}),
        "steer_right": frozenset({"d", "right"}),
        "stop": frozenset({"space"}),
        "reverse": frozenset(),
    }
)
"""Default key bindings for :class:`KeyboardToDriverCommand`.

Bindings are data so a layout can be rebound without editing the converter, and
so the set of tracked keys is derived from them rather than declared twice.
"""

_DRIVER_ACTIONS = frozenset(DEFAULT_DRIVING_BINDINGS)

DEFAULT_CAMERA_BINDINGS: CameraBindings = MappingProxyType(
    {
        "move_forward": ("w", "s"),
        "move_right": ("e", "q"),
        "yaw": ("a", "d"),
        "pitch": ("i", "k"),
    }
)
"""Default axis bindings for :class:`KeyboardToCameraCommand`.

Each pair is ``(positive, negative)``. Alternate yaw keys ``j`` and ``l`` are
normalized onto ``a`` and ``d`` so the canonical axes stay layout-independent.
"""

_CAMERA_AXES = frozenset(DEFAULT_CAMERA_BINDINGS)
_CAMERA_KEY_ALIASES: Mapping[str, str] = MappingProxyType({"j": "a", "l": "d"})


@dataclass(frozen=True, kw_only=True, slots=True)
class DeviceConverterSchema:
    """Metadata for one device-to-canonical-modality converter."""

    name: str
    """Unique converter name."""

    produces: CanonicalModality
    """Canonical modality produced by the converter."""

    consumes: tuple[UserInputCapability, ...] = ()
    """Raw input capabilities required from a source."""

    device_kind: str | None = None
    """Device family recorded in canonical input metadata."""

    priority: int = 0
    """Selection priority among converters producing the same modality."""

    accepted_keys: frozenset[str] | None = None
    """Keys accepted by transport filtering; ``None`` keeps its fallback policy."""

    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )
    """Converter-specific metadata not interpreted by the runtime."""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("DeviceConverterSchema.name must be non-empty.")
        if not isinstance(self.produces, CanonicalModality):
            raise TypeError("produces must be a CanonicalModality object.")
        if self.accepted_keys is not None:
            if isinstance(self.accepted_keys, str):
                raise TypeError("accepted_keys must be a collection of key strings.")
            if not all(isinstance(key, str) for key in self.accepted_keys):
                raise TypeError("accepted_keys must contain strings.")
            if any(not key.strip() for key in self.accepted_keys):
                raise ValueError("accepted_keys must not contain empty keys.")
            object.__setattr__(self, "accepted_keys", frozenset(self.accepted_keys))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@runtime_checkable
class DeviceConverter(Protocol):
    """Contract for turning one device's raw events into a canonical modality."""

    @property
    def schema(self) -> DeviceConverterSchema:
        """Return converter metadata used for source selection."""
        ...

    def reset(self) -> None:
        """Drop accumulated device state at a session or rollout boundary."""
        ...

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, Any] | None:
        """Return the modality value for ``window``, or ``None`` if inactive.

        ``user_inputs`` is already filtered to ``window``. Returning ``None``
        lets a present-but-idle device yield to a lower-priority one.
        """
        ...


DRIVER_COMMAND = CanonicalModality(
    name="driver_command",
    payload_fields=frozenset({"throttle", "brake", "steer", "stop", "reverse"}),
    description=(
        "Normalized driving intent. throttle/brake are in [0, 1], steer is in "
        "[-1, 1] with positive meaning left."
    ),
)

CAMERA_COMMAND = CanonicalModality(
    name="camera_command",
    payload_fields=frozenset({*_CAMERA_AXES, "segments"}),
    description=(
        "Free-camera intent. move_forward, move_right, yaw, and pitch are in "
        "[-1, 1] and hold the level state at the end of the window. segments "
        "carries the piecewise-constant timeline inside the window."
    ),
)


class KeyboardToDriverCommand:
    """Convert keyboard edges into :data:`DRIVER_COMMAND` level state.

    Mirrors the mapping the Omnidreams interactive-drive keyboard backend
    already uses, so a keyboard reaches a model through the shared layer with
    the same semantics it has today.
    """

    def __init__(
        self,
        *,
        name: str = "keyboard-to-driver-command",
        bindings: DriverBindings = DEFAULT_DRIVING_BINDINGS,
        priority: int = 0,
    ) -> None:
        unknown = sorted(set(bindings) - _DRIVER_ACTIONS)
        if unknown:
            raise ValueError(
                f"Unknown driver actions in bindings: {unknown}. "
                f"Supported actions: {sorted(_DRIVER_ACTIONS)}."
            )
        self._bindings = {
            action: frozenset(normalize_key(key) for key in bindings.get(action, ()))
            for action in _DRIVER_ACTIONS
        }
        # Tracked keys are derived, so they cannot drift from the bindings and
        # silently make an action unreachable.
        self._supported_keys = frozenset(
            key for keys in self._bindings.values() for key in keys
        )
        uses_default_bindings = self._bindings == DEFAULT_DRIVING_BINDINGS
        if uses_default_bindings:
            self._supported_keys = DRIVING_SUPPORTED_KEYS
        self._state = KeyboardState(supported_keys=self._supported_keys)
        self._schema = DeviceConverterSchema(
            name=name,
            produces=DRIVER_COMMAND,
            device_kind="keyboard",
            priority=priority,
            accepted_keys=self._supported_keys,
            consumes=(
                UserInputCapability(
                    event_type="key_down",
                    payload_fields=frozenset({"key"}),
                ),
                UserInputCapability(
                    event_type="key_up",
                    payload_fields=frozenset({"key"}),
                ),
            ),
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        return self._schema

    def reset(self) -> None:
        self._state = KeyboardState(supported_keys=self._supported_keys)

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, Any] | None:
        del window
        for event in user_inputs.events:
            if event.event_type not in {"key_down", "key_up"}:
                continue
            key = event.payload.get("key")
            if not isinstance(key, str):
                continue
            self._state.apply_event(
                event="keydown" if event.event_type == "key_down" else "keyup",
                key=key,
            )

        pressed = {normalize_key(key) for key in self._state.snapshot()}

        def held(action: str) -> bool:
            return bool(self._bindings[action] & pressed)

        steer = 0.0
        if held("steer_left"):
            steer += 1.0
        if held("steer_right"):
            steer -= 1.0
        return DRIVER_COMMAND.value(
            {
                "throttle": 1.0 if held("throttle") else 0.0,
                "brake": 1.0 if held("brake") else 0.0,
                "steer": steer,
                "stop": held("stop"),
                "reverse": held("reverse"),
            }
        )


class KeyboardToCameraCommand:
    """Convert keyboard edges into :data:`CAMERA_COMMAND` level state."""

    def __init__(
        self,
        *,
        name: str = "keyboard-to-camera-command",
        supported_keys: frozenset[str] = DEFAULT_SUPPORTED_KEYS,
        priority: int = 0,
    ) -> None:
        self._supported_keys = frozenset(normalize_key(key) for key in supported_keys)
        self._state = KeyboardState(supported_keys=self._supported_keys)
        self._schema = DeviceConverterSchema(
            name=name,
            produces=CAMERA_COMMAND,
            device_kind="keyboard",
            priority=priority,
            accepted_keys=self._supported_keys,
            consumes=(
                UserInputCapability(
                    event_type="key_down",
                    payload_fields=frozenset({"key"}),
                ),
                UserInputCapability(
                    event_type="key_up",
                    payload_fields=frozenset({"key"}),
                ),
            ),
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        return self._schema

    def reset(self) -> None:
        self._state = KeyboardState(supported_keys=self._supported_keys)

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, Any] | None:
        segments: list[tuple[float, float, dict[str, float]]] = []
        segment_start = window.start_s
        axes = _camera_axes_from_keys(self._state.resolved_effective_keys())

        for event in user_inputs.events:
            if event.event_type not in {"key_down", "key_up"}:
                continue
            key = event.payload.get("key")
            if not isinstance(key, str):
                continue
            edge_t = min(max(float(event.timestamp_s), window.start_s), window.end_s)
            if edge_t > segment_start:
                segments.append((segment_start, edge_t, axes))
                segment_start = edge_t
            self._state.apply_event(
                event="keydown" if event.event_type == "key_down" else "keyup",
                key=key,
            )
            axes = _camera_axes_from_keys(self._state.resolved_effective_keys())

        if window.end_s > segment_start or not segments:
            segments.append((segment_start, window.end_s, axes))

        return CAMERA_COMMAND.value({**axes, "segments": tuple(segments)})


def _camera_axes_from_keys(pressed: Iterable[str]) -> dict[str, float]:
    keys = {_CAMERA_KEY_ALIASES.get(key, key) for key in pressed}
    axes: dict[str, float] = {}
    for axis, (positive, negative) in DEFAULT_CAMERA_BINDINGS.items():
        value = 0.0
        if positive in keys:
            value += 1.0
        if negative in keys:
            value -= 1.0
        axes[axis] = value
    return axes


class ScriptedModality:
    """Emit pre-authored canonical values, for benchmarks, replay, and tests.

    Mocking input should not require knowing the raw device vocabulary. This
    converter consumes no raw capabilities, so it is feedable by any source
    -- including an empty :class:`UserInputSchema` -- and application code is
    identical between a real run and a scripted one.

    ``timeline`` is ``(start_s, value)`` pairs. Values are level-triggered and
    held until the next entry begins, matching how live converters behave. An
    entry applies to a window once it has begun by the window's end, and
    ``None`` is returned for windows before the first entry.
    """

    def __init__(
        self,
        *,
        modality: CanonicalModality,
        timeline: Sequence[tuple[float, Mapping[str, Any]]],
        name: str | None = None,
        device_kind: str | None = "scripted",
        priority: int = 0,
    ) -> None:
        entries = tuple(sorted(timeline, key=lambda entry: entry[0]))
        for start_s, value in entries:
            if start_s < 0:
                raise ValueError("timeline start_s must be >= 0.")
            modality.value(value)
        self._entries = tuple(
            (start_s, modality.value(value)) for start_s, value in entries
        )
        self._modality = modality
        self._schema = DeviceConverterSchema(
            name=name or f"scripted-{modality.name}",
            produces=modality,
            device_kind=device_kind,
            priority=priority,
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        return self._schema

    def reset(self) -> None:
        # The timeline is a pure function of the window, so replay is
        # deterministic without any state to clear.
        return None

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, Any] | None:
        del user_inputs
        current: Mapping[str, Any] | None = None
        for start_s, value in self._entries:
            if start_s < window.end_s:
                current = value
            else:
                break
        return current


class InputCanonicalizer:
    """Registry of device converters plus the raw-to-canonical rewrite.

    Registration is the whole extension point: a new device is a converter
    registered against an existing modality, and a new modality is a converter
    registered with a new :class:`CanonicalModality`.
    """

    def __init__(self, converters: Iterable[DeviceConverter] = ()) -> None:
        self._converters: list[DeviceConverter] = []
        for converter in converters:
            self.register(converter)

    def register(self, converter: DeviceConverter) -> None:
        """Register one device converter."""
        if not isinstance(converter, DeviceConverter):
            raise TypeError("converter must implement the DeviceConverter protocol.")
        name = converter.schema.name
        if any(existing.schema.name == name for existing in self._converters):
            raise ValueError(
                f"A device converter named {name!r} is already registered."
            )
        self._converters.append(converter)

    @property
    def converters(self) -> tuple[DeviceConverter, ...]:
        """Return every registered converter."""
        return tuple(self._converters)

    def reset(self) -> None:
        """Reset every registered converter's device state."""
        for converter in self._converters:
            converter.reset()

    def converters_for(
        self,
        source_schema: UserInputSchema,
    ) -> tuple[DeviceConverter, ...]:
        """Return converters this source can feed, highest priority first."""
        feedable = [
            converter
            for converter in self._converters
            if all(
                source_schema.supports(capability)
                for capability in converter.schema.consumes
            )
        ]
        # Sort is stable, so equal-priority converters keep registration order.
        return tuple(sorted(feedable, key=lambda each: -each.schema.priority))

    def unavailable_converters(
        self,
        source_schema: UserInputSchema,
    ) -> tuple[DeviceConverter, ...]:
        """Return converters this source cannot feed, for diagnostics."""
        feedable = {id(converter) for converter in self.converters_for(source_schema)}
        return tuple(
            converter for converter in self._converters if id(converter) not in feedable
        )

    def canonical_schema(
        self,
        source_schema: UserInputSchema,
    ) -> CanonicalInputSchema:
        """Return the canonical modalities this raw source can supply.

        This is the boundary an application declares against. A mapping that
        consumes ``driver_command`` then matches a keyboard source, a wheel
        source, or any device registered later.
        """
        modalities: list[CanonicalModality] = []
        for converter in self.converters_for(source_schema):
            modality = converter.schema.produces
            if modality not in modalities:
                modalities.append(modality)
        return CanonicalInputSchema(
            modalities=tuple(modalities),
            description=source_schema.description,
        )

    def canonicalize(
        self,
        user_inputs: UserInputs,
        *,
        window: TimeWindow,
        source_schema: UserInputSchema,
    ) -> CanonicalInputs:
        """Convert one raw window into canonical inputs.

        Every feedable converter sees the window so its device state stays
        current even while another device has precedence; that way unplugging
        the higher-priority device does not resume from stale state. Among
        converters producing the same modality, the highest-priority one that
        returned a value wins.
        """
        windowed = user_inputs.window(window)
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for converter in self.converters_for(source_schema):
            value = converter.convert(windowed, window)
            modality = converter.schema.produces
            if value is not None and modality.name not in values:
                values[modality.name] = value
                if converter.schema.device_kind is not None:
                    sources[modality.name] = converter.schema.device_kind

        metadata: dict[str, Any] = {}
        if sources:
            metadata["canonical_sources"] = freeze_mapping(sources)
        return CanonicalInputs(values=values, metadata=metadata)
