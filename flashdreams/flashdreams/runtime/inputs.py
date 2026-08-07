# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User- and model-input envelopes for the experimental runtime API."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from flashdreams.infra.time import TimeWindow
from flashdreams.runtime._utils import freeze_mapping

InputPhase = Literal["global_conditioning", "step"]

INPUT_PHASES: tuple[InputPhase, ...] = ("global_conditioning", "step")


def validate_phase(value: str) -> InputPhase:
    """Return ``value`` as a validated :data:`InputPhase`."""
    if value not in INPUT_PHASES:
        raise ValueError(
            f"phase must be 'global_conditioning' or 'step', got {value!r}."
        )
    return cast(InputPhase, value)


@dataclass(frozen=True, kw_only=True, slots=True)
class InputField:
    """Lightweight schema field for user snapshots or model inputs.

    ``name`` is the model-facing input role and payload key, such as ``prompt``
    or ``negative_prompt``. ``input_modality``, ``frequency_consumed``, and
    ``metadata`` are query hints only. Adapter-owned validation still decides
    concrete shape, dtype, units, and tensor layout.
    """

    name: str
    required: bool = True
    input_modality: str | None = None
    frequency_consumed: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("InputField.name must be non-empty.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class UserInputCapability:
    """One user event a source or mapping can provide, at payload granularity.

    ``UserInputSchema.event_types`` declares only that an event type exists. A
    capability additionally pins the payload fields carried by that event, so a
    mapping can state that it needs ``key_down`` events that actually carry a
    ``key``.
    """

    event_type: str
    input_modality: str | None = None
    payload_fields: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )
    description: str = ""

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("UserInputCapability.event_type must be non-empty.")
        for payload_field in self.payload_fields:
            if not payload_field.strip():
                raise ValueError("payload field names must be non-empty.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def is_satisfied_by(self, provider: "UserInputCapability") -> bool:
        """Return whether ``provider`` can satisfy this consumed capability."""
        if self.event_type != provider.event_type:
            return False
        input_modality_ok = (
            self.input_modality is None
            or provider.input_modality is None
            or self.input_modality == provider.input_modality
        )
        return input_modality_ok and self.payload_fields.issubset(
            provider.payload_fields
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class UserInputSchema:
    """Minimal metadata for user events a source or mapping can provide."""

    event_types: frozenset[str] = field(default_factory=frozenset)
    snapshot_fields: tuple[InputField, ...] = ()
    capabilities: tuple[UserInputCapability, ...] = ()
    description: str = ""

    def supports_event_types(self, event_types: Iterable[str]) -> bool:
        """Return whether every requested event type is declared supported."""
        requested = frozenset(event_types)
        if not requested:
            return True
        return requested.issubset(self.declared_event_types())

    def declared_event_types(self) -> frozenset[str]:
        """Return event types from ``event_types`` and from ``capabilities``."""
        return self.event_types | frozenset(
            capability.event_type for capability in self.capabilities
        )

    def declared_capabilities(self) -> tuple[UserInputCapability, ...]:
        """Return capabilities, widened with bare ``event_types`` entries.

        A plain ``event_types`` entry carries no payload promise, so it is
        modeled as a capability with no payload fields. Coarse schemas written
        before capabilities existed therefore still satisfy any consumer that
        does not require specific payload fields.
        """
        declared = list(self.capabilities)
        covered = {capability.event_type for capability in declared}
        declared.extend(
            UserInputCapability(event_type=event_type)
            for event_type in sorted(self.event_types - covered)
        )
        return tuple(declared)

    def supports(self, capability: UserInputCapability) -> bool:
        """Return whether this source can satisfy ``capability``."""
        return any(
            capability.is_satisfied_by(provider)
            for provider in self.declared_capabilities()
        )

    def validate_event(self, event: "UserInputEvent") -> None:
        """Validate one event against the event types this source declares."""
        matching = [
            capability
            for capability in self.declared_capabilities()
            if capability.event_type == event.event_type
        ]
        if not matching:
            raise ValueError(
                f"User input source does not provide event type {event.event_type!r}."
            )
        payload_keys = set(event.payload)
        if not any(
            capability.payload_fields.issubset(payload_keys) for capability in matching
        ):
            expected = sorted(
                {
                    payload_field
                    for capability in matching
                    for payload_field in capability.payload_fields
                }
            )
            raise ValueError(
                f"Event {event.event_type!r} payload is missing required "
                f"fields: {expected}."
            )

    def missing_snapshot(self, inputs: "UserInputs") -> tuple[str, ...]:
        """Return required snapshot fields absent from ``inputs``."""
        return _missing_required(self.snapshot_fields, inputs.snapshot)

    def require_snapshot(self, inputs: "UserInputs") -> None:
        """Raise if required snapshot fields are absent."""
        missing = self.missing_snapshot(inputs)
        if missing:
            raise ValueError(f"Missing required user snapshot field(s): {missing}")


@dataclass(frozen=True, kw_only=True, slots=True)
class InferenceInputSchema:
    """Minimal metadata for global conditioning and per-step inputs."""

    global_conditioning_fields: tuple[InputField, ...] = ()
    """Model inputs carried in the global conditioning slot."""

    step_fields: tuple[InputField, ...] = ()
    """Model inputs required for one session step."""

    description: str = ""

    def fields_for(self, phase: InputPhase) -> tuple[InputField, ...]:
        """Return every declared field for ``phase``."""
        return (
            self.global_conditioning_fields
            if validate_phase(phase) == "global_conditioning"
            else self.step_fields
        )

    def required_fields(
        self,
        phase: InputPhase | None = None,
    ) -> tuple[tuple[InputPhase, InputField], ...]:
        """Return required fields as ``(phase, field)``, optionally filtered."""
        return self._select(phase, required=True)

    def optional_fields(
        self,
        phase: InputPhase | None = None,
    ) -> tuple[tuple[InputPhase, InputField], ...]:
        """Return optional fields as ``(phase, field)``, optionally filtered."""
        return self._select(phase, required=False)

    def field_for(self, *, name: str, phase: InputPhase) -> InputField | None:
        """Return one declared field, if present."""
        for input_field in self.fields_for(phase):
            if input_field.name == name:
                return input_field
        return None

    def _select(
        self,
        phase: InputPhase | None,
        *,
        required: bool,
    ) -> tuple[tuple[InputPhase, InputField], ...]:
        phases = INPUT_PHASES if phase is None else (validate_phase(phase),)
        return tuple(
            (each_phase, input_field)
            for each_phase in phases
            for input_field in self.fields_for(each_phase)
            if input_field.required is required
        )

    def missing_global_conditioning(self, inputs: "InferenceInput") -> tuple[str, ...]:
        """Return required global conditioning fields absent from ``inputs``."""
        return _missing_required(
            self.global_conditioning_fields,
            inputs.global_conditioning,
        )

    def missing_step(self, inputs: "InferenceInput") -> tuple[str, ...]:
        """Return required per-step fields absent from ``inputs``."""
        return _missing_required(self.step_fields, inputs.step)

    def require_global_conditioning(self, inputs: "InferenceInput") -> None:
        """Raise if required global conditioning fields are absent."""
        missing = self.missing_global_conditioning(inputs)
        if missing:
            raise ValueError(
                f"Missing required global conditioning input(s): {missing}"
            )

    def require_step(self, inputs: "InferenceInput") -> None:
        """Raise if required per-step fields are absent."""
        missing = self.missing_step(inputs)
        if missing:
            raise ValueError(f"Missing required step model input(s): {missing}")


@dataclass(frozen=True, kw_only=True, slots=True)
class UserInputEvent:
    """User-facing input event timestamped in seconds since session start.

    Live runtimes, transports, replay loaders, or benchmark drivers stamp events
    before queuing them for input mapping. Payload schema is intentionally minimal
    in T1; concrete event catalogs belong to follow-up input-mapping work.
    """

    __hash__ = None

    timestamp_s: float
    event_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    source_event_id: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("UserInputEvent.timestamp_s must be finite and >= 0.")
        if not self.event_type.strip():
            raise ValueError("UserInputEvent.event_type must be non-empty.")
        object.__setattr__(self, "payload", freeze_mapping(self.payload))


@dataclass(frozen=True, kw_only=True, slots=True)
class UserInputs:
    """Transport-neutral user input batch or window.

    Events must be in non-decreasing timestamp order. Runtimes can pass the full
    input history, a drained queue batch, or a session-requested time window to an
    ``InputMapping``.
    """

    __hash__ = None

    events: tuple[UserInputEvent, ...] = ()
    snapshot: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        previous_timestamp_s = -math.inf
        for event in self.events:
            if event.timestamp_s < previous_timestamp_s:
                raise ValueError(
                    "UserInputs.events must be sorted by non-decreasing timestamp_s."
                )
            previous_timestamp_s = event.timestamp_s
        object.__setattr__(self, "snapshot", freeze_mapping(self.snapshot))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def window(self, time_window: TimeWindow) -> "UserInputs":
        """Return inputs with events filtered to ``time_window``."""
        return UserInputs(
            events=tuple(
                event
                for event in self.events
                if time_window.contains(event.timestamp_s)
            ),
            snapshot=self.snapshot,
            metadata=self.metadata,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class CanonicalModality:
    """A device-independent user input an application consumes.

    This is the middle layer of ``raw input -> canonicalized input -> encoded
    inference input``. Applications and benchmarks declare and consume
    modalities; they never read raw device events, so adding a new device is a
    converter registration rather than an application change.

    Modalities describe live user control only. Global conditioning such as a
    prompt or conditioning frame is application-owned and reaches
    :class:`InferenceInput` directly, without passing through this layer.
    """

    name: str
    payload_fields: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("CanonicalModality.name must be non-empty.")
        for payload_field in self.payload_fields:
            if not payload_field.strip():
                raise ValueError("payload field names must be non-empty.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def is_satisfied_by(self, provider: "CanonicalModality") -> bool:
        """Return whether ``provider`` can satisfy this consumed modality."""
        return self.name == provider.name and self.payload_fields.issubset(
            provider.payload_fields
        )

    def value(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return ``payload`` frozen, checking it covers this modality."""
        missing = sorted(self.payload_fields - set(payload))
        if missing:
            raise ValueError(
                f"Canonical modality {self.name!r} requires payload fields "
                f"{missing}, which the converter did not produce."
            )
        return freeze_mapping(payload)


@dataclass(frozen=True, kw_only=True, slots=True)
class CanonicalInputSchema:
    """Canonical modalities an application can be fed by a given source."""

    modalities: tuple[CanonicalModality, ...] = ()
    description: str = ""

    def supports(self, modality: CanonicalModality) -> bool:
        """Return whether this source can supply ``modality``."""
        return any(modality.is_satisfied_by(provided) for provided in self.modalities)


@dataclass(frozen=True, kw_only=True, slots=True)
class CanonicalInputs:
    """Canonicalized user input for one step, keyed by modality name.

    Values are level-triggered and normally present every step: a key held down
    emits no events but still means full throttle. Global conditioning does not
    appear here; it is application-owned and reaches :class:`InferenceInput`
    directly.
    """

    __hash__ = None

    values: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_mapping(self.values))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class InferenceInput:
    """Encoded inputs for one :class:`InferenceSession` call.

    Two conditioning slots:

    - ``global_conditioning``: values that condition the whole rollout, such as
      the conditioning frame or prompt. Session start/reset establishes this
      state; a step call may carry a non-empty payload to request an update when
      the model supports it.
    - ``step``: values needed to generate the next chunk or frame.
    """

    __hash__ = None

    global_conditioning: Mapping[str, Any] = field(default_factory=dict)
    step: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "global_conditioning", freeze_mapping(self.global_conditioning)
        )
        object.__setattr__(self, "step", freeze_mapping(self.step))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def for_phase(self, phase: InputPhase) -> Mapping[str, Any]:
        """Return the payload mapping for ``phase``."""
        return (
            self.global_conditioning
            if validate_phase(phase) == "global_conditioning"
            else self.step
        )


def _missing_required(
    fields: tuple[InputField, ...], payload: Mapping[str, Any]
) -> tuple[str, ...]:
    return tuple(
        input_field.name
        for input_field in fields
        if input_field.required and input_field.name not in payload
    )
