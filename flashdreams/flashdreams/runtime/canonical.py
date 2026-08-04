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

Global-conditioning converters behave differently on purpose. They emit only
when a value actually changed in the window, so a downstream non-empty global
slot means "update this", not "re-apply it every step".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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
from flashdreams.serving.realtime.input import (
    DRIVING_SUPPORTED_KEYS,
    KeyboardState,
    normalize_key,
)


@dataclass(frozen=True, kw_only=True, slots=True)
class DeviceConverterSchema:
    """Metadata for one device-to-canonical-modality converter."""

    name: str
    produces: CanonicalModality
    consumes: tuple[UserInputCapability, ...] = ()
    device_kind: str | None = None
    priority: int = 0
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("DeviceConverterSchema.name must be non-empty.")
        if not isinstance(self.produces, CanonicalModality):
            raise TypeError("produces must be a CanonicalModality object.")
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
        lets a present-but-idle device yield to a lower-priority one, and lets a
        global-conditioning converter stay silent when nothing changed.
        """
        ...


DRIVER_COMMAND = CanonicalModality(
    name="driver_command",
    phase="step",
    payload_fields=frozenset({"throttle", "brake", "steer", "stop", "reverse"}),
    description=(
        "Normalized driving intent. throttle/brake are in [0, 1], steer is in "
        "[-1, 1] with positive meaning left."
    ),
)

CONDITIONING_PROMPT = CanonicalModality(
    name="conditioning_prompt",
    phase="global",
    payload_fields=frozenset({"prompt"}),
    description="Global conditioning prompt; may change mid-rollout.",
)

CONDITIONING_FRAME = CanonicalModality(
    name="conditioning_frame",
    phase="global",
    payload_fields=frozenset({"image"}),
    description="Global conditioning frame; may change mid-rollout.",
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
        supported_keys: frozenset[str] = DRIVING_SUPPORTED_KEYS,
        priority: int = 0,
    ) -> None:
        self._supported_keys = supported_keys
        self._state = KeyboardState(supported_keys=supported_keys)
        self._schema = DeviceConverterSchema(
            name=name,
            produces=DRIVER_COMMAND,
            device_kind="keyboard",
            priority=priority,
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
        steer = 0.0
        if {"a", "left"} & pressed:
            steer += 1.0
        if {"d", "right"} & pressed:
            steer -= 1.0
        return DRIVER_COMMAND.value(
            {
                "throttle": 1.0 if {"w", "up"} & pressed else 0.0,
                "brake": 1.0 if {"s", "down"} & pressed else 0.0,
                "steer": steer,
                "stop": "space" in pressed,
                "reverse": False,
            }
        )


class LatestEventToModality:
    """Emit a global-conditioning modality when its source event fires.

    Global conditioning is transient by design: this returns ``None`` in windows
    where nothing changed, so downstream code only sees an update request when
    the user actually changed something.
    """

    def __init__(
        self,
        *,
        modality: CanonicalModality,
        event_type: str,
        payload_fields: frozenset[str] | None = None,
        name: str | None = None,
        device_kind: str | None = None,
        priority: int = 0,
    ) -> None:
        if modality.phase != "global":
            raise ValueError(
                "LatestEventToModality is for global conditioning; "
                f"{modality.name!r} declares phase {modality.phase!r}."
            )
        self._modality = modality
        self._event_type = event_type
        fields = modality.payload_fields if payload_fields is None else payload_fields
        self._schema = DeviceConverterSchema(
            name=name or f"{event_type}-to-{modality.name}",
            produces=modality,
            device_kind=device_kind,
            priority=priority,
            consumes=(
                UserInputCapability(event_type=event_type, payload_fields=fields),
            ),
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        return self._schema

    def reset(self) -> None:
        return None

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, Any] | None:
        del window
        latest = None
        for event in user_inputs.events:
            if event.event_type == self._event_type:
                latest = event
        if latest is None:
            return None
        return self._modality.value(
            {
                name: latest.payload[name]
                for name in self._modality.payload_fields
                if name in latest.payload
            }
        )


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
        by_phase: dict[str, dict[str, Any]] = {"global": {}, "step": {}}
        sources: dict[str, str] = {}
        for converter in self.converters_for(source_schema):
            value = converter.convert(windowed, window)
            modality = converter.schema.produces
            slot = by_phase[modality.phase]
            if value is not None and modality.name not in slot:
                slot[modality.name] = value
                if converter.schema.device_kind is not None:
                    sources[modality.name] = converter.schema.device_kind

        metadata: dict[str, Any] = {}
        if sources:
            metadata["canonical_sources"] = freeze_mapping(sources)
        return CanonicalInputs(
            global_conditioning=by_phase["global"],
            per_step=by_phase["step"],
            metadata=metadata,
        )
