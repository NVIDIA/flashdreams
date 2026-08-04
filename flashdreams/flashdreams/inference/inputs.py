# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-input, model-input, and mapping contracts for inference runtimes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast, runtime_checkable

InputPhase = Literal["initial", "step"]


def _normalized_token(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalized_optional_token(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _normalized_token(value, field_name=field_name)


def _normalized_payload_fields(values: Iterable[str]) -> frozenset[str]:
    return frozenset(
        _normalized_token(value, field_name="payload field") for value in values
    )


def _immutable_metadata_map(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy metadata into a read-only map without rewriting its keys.

    Metadata is an open-ended pass-through for adapter hints, so keys are
    validated but never normalized: a key must round-trip unchanged.
    """
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError(f"metadata keys must be strings, got {type(key).__name__}.")
        if not key:
            raise ValueError("metadata keys must be non-empty strings.")
        normalized[key] = value
    return MappingProxyType(normalized)


def _merged_metadata(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Union two metadata maps, keeping ``first`` on conflicting keys."""
    merged = dict(second)
    merged.update(first)
    return merged


def _validate_phase(value: str) -> InputPhase:
    if value not in {"initial", "step"}:
        raise ValueError(f"phase must be 'initial' or 'step', got {value!r}.")
    return cast(InputPhase, value)


def _field_key(field: "ModelInputField") -> tuple[InputPhase, str]:
    return field.phase, field.name


def _model_field_matches(
    produced: "ModelInputField",
    required: "ModelInputField",
) -> bool:
    if _field_key(produced) != _field_key(required):
        return False
    payload_kind_ok = (
        produced.payload_kind is None
        or required.payload_kind is None
        or produced.payload_kind == required.payload_kind
    )
    lifecycle_ok = (
        produced.lifecycle is None
        or required.lifecycle is None
        or produced.lifecycle == required.lifecycle
    )
    return payload_kind_ok and lifecycle_ok


@dataclass(frozen=True, slots=True)
class UserInputEvent:
    """One user-facing input event in session time.

    Static session setup values, such as prompts or initial frames, are still
    represented as events. Snapshots are derived views over events rather than
    primary inputs.
    """

    timestamp_s: float
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        timestamp_s = float(self.timestamp_s)
        if not isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and >= 0.")
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(
            self,
            "kind",
            _normalized_token(self.kind, field_name="event kind"),
        )
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        object.__setattr__(
            self,
            "session_id",
            _normalized_optional_token(self.session_id, field_name="session_id"),
        )
        object.__setattr__(
            self,
            "source",
            _normalized_optional_token(self.source, field_name="source"),
        )

    def __hash__(self) -> int:
        # ``payload`` is an arbitrary mapping and therefore unhashable, so the
        # generated hash would raise. Hash the identity fields instead; equality
        # still compares payloads, and equal events still hash equal.
        return hash((self.timestamp_s, self.kind, self.session_id, self.source))


@dataclass(frozen=True, slots=True)
class UserInputWindow:
    """A deterministic time window over user input events."""

    start_s: float
    end_s: float
    events: tuple[UserInputEvent, ...] = ()

    def __post_init__(self) -> None:
        start_s = float(self.start_s)
        end_s = float(self.end_s)
        if not isfinite(start_s) or not isfinite(end_s):
            raise ValueError("window bounds must be finite.")
        if end_s < start_s:
            raise ValueError("end_s must be >= start_s.")
        object.__setattr__(self, "start_s", start_s)
        object.__setattr__(self, "end_s", end_s)
        events = _sorted_events(self.events)
        for event in events:
            if not start_s <= event.timestamp_s <= end_s:
                raise ValueError(
                    f"Event {event.kind!r} at t={event.timestamp_s} is outside "
                    f"window [{start_s}, {end_s}]."
                )
        object.__setattr__(self, "events", events)

    def events_of_kind(self, kind: str) -> tuple[UserInputEvent, ...]:
        """Return all events of ``kind`` inside this window."""
        kind = _normalized_token(kind, field_name="event kind")
        return tuple(event for event in self.events if event.kind == kind)

    def latest(self, kind: str) -> UserInputEvent | None:
        """Return the latest event of ``kind`` inside this window, if any."""
        events = self.events_of_kind(kind)
        return events[-1] if events else None


@dataclass(frozen=True, slots=True)
class UserInputTrace:
    """Ordered replayable user-input event trace."""

    events: tuple[UserInputEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", _sorted_events(self.events))

    @classmethod
    def from_events(cls, events: Sequence[UserInputEvent]) -> "UserInputTrace":
        """Build a trace from any event sequence."""
        return cls(events=tuple(events))

    def window(
        self,
        *,
        start_s: float,
        end_s: float,
        include_start: bool = True,
        include_end: bool = False,
    ) -> UserInputWindow:
        """Slice the trace into a deterministic event window."""
        start_s = float(start_s)
        end_s = float(end_s)
        if end_s < start_s:
            raise ValueError("end_s must be >= start_s.")

        def _starts_in(event: UserInputEvent) -> bool:
            if include_start:
                return event.timestamp_s >= start_s
            return event.timestamp_s > start_s

        def _ends_in(event: UserInputEvent) -> bool:
            if include_end:
                return event.timestamp_s <= end_s
            return event.timestamp_s < end_s

        return UserInputWindow(
            start_s=start_s,
            end_s=end_s,
            events=tuple(
                event for event in self.events if _starts_in(event) and _ends_in(event)
            ),
        )

    def events_of_kind(self, kind: str) -> tuple[UserInputEvent, ...]:
        """Return all events of ``kind`` in this trace."""
        kind = _normalized_token(kind, field_name="event kind")
        return tuple(event for event in self.events if event.kind == kind)

    def latest(self, kind: str) -> UserInputEvent | None:
        """Return the latest event of ``kind`` in this trace, if any."""
        events = self.events_of_kind(kind)
        return events[-1] if events else None


def _sorted_events(events: Sequence[UserInputEvent]) -> tuple[UserInputEvent, ...]:
    indexed_events = tuple(enumerate(events))
    for _, event in indexed_events:
        if not isinstance(event, UserInputEvent):
            raise TypeError(f"Expected UserInputEvent, got {type(event).__name__}.")
    return tuple(
        event
        for _, event in sorted(
            indexed_events,
            key=lambda item: (item[1].timestamp_s, item[0]),
        )
    )


@dataclass(frozen=True, slots=True)
class UserInputCapability:
    """Lightweight metadata for a user event a source or mapper can provide."""

    event_kind: str
    payload_kind: str | None = None
    payload_fields: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_kind",
            _normalized_token(self.event_kind, field_name="event kind"),
        )
        object.__setattr__(
            self,
            "payload_kind",
            _normalized_optional_token(
                self.payload_kind,
                field_name="payload_kind",
            ),
        )
        object.__setattr__(
            self,
            "payload_fields",
            _normalized_payload_fields(self.payload_fields),
        )
        object.__setattr__(self, "metadata", _immutable_metadata_map(self.metadata))

    def is_satisfied_by(self, provider: "UserInputCapability") -> bool:
        """Return whether ``provider`` can satisfy this consumed capability."""
        if self.event_kind != provider.event_kind:
            return False
        payload_kind_ok = (
            self.payload_kind is None
            or provider.payload_kind is None
            or self.payload_kind == provider.payload_kind
        )
        return payload_kind_ok and self.payload_fields.issubset(provider.payload_fields)


@dataclass(frozen=True, slots=True)
class UserInputSchema:
    """Metadata describing what a source can provide."""

    capabilities: tuple[UserInputCapability, ...] = ()
    name: str = "user-input-source"
    source_kind: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalized_token(self.name, field_name="schema name"),
        )
        object.__setattr__(
            self,
            "source_kind",
            _normalized_optional_token(self.source_kind, field_name="source_kind"),
        )
        object.__setattr__(self, "metadata", _immutable_metadata_map(self.metadata))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        for capability in self.capabilities:
            if not isinstance(capability, UserInputCapability):
                raise TypeError(
                    "capabilities must contain UserInputCapability objects."
                )

    def supports(self, capability: UserInputCapability) -> bool:
        """Return whether this source can satisfy ``capability``."""
        return any(
            capability.is_satisfied_by(provider) for provider in self.capabilities
        )

    def validate_event(self, event: UserInputEvent) -> None:
        """Validate an event against this source schema."""
        matching = [
            capability
            for capability in self.capabilities
            if capability.event_kind == event.kind
        ]
        if not matching:
            raise ValueError(
                f"User input source {self.name!r} does not provide event "
                f"kind {event.kind!r}."
            )
        payload_keys = set(event.payload)
        if not any(
            capability.payload_fields.issubset(payload_keys) for capability in matching
        ):
            expected = sorted(
                {
                    field_name
                    for capability in matching
                    for field_name in capability.payload_fields
                }
            )
            raise ValueError(
                f"Event {event.kind!r} payload is missing required fields "
                f"for source {self.name!r}: {expected}."
            )


@dataclass(frozen=True, slots=True)
class ModelInputField:
    """Lightweight metadata for one semantic model-facing input field."""

    name: str
    phase: InputPhase
    required: bool = True
    payload_kind: str | None = None
    update_policy: str | None = None
    lifecycle: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalized_token(self.name, field_name="model input name"),
        )
        object.__setattr__(self, "phase", _validate_phase(self.phase))
        object.__setattr__(
            self,
            "payload_kind",
            _normalized_optional_token(
                self.payload_kind,
                field_name="payload_kind",
            ),
        )
        object.__setattr__(
            self,
            "update_policy",
            _normalized_optional_token(
                self.update_policy,
                field_name="update_policy",
            ),
        )
        object.__setattr__(
            self,
            "lifecycle",
            _normalized_optional_token(
                self.lifecycle,
                field_name="lifecycle",
            ),
        )
        object.__setattr__(self, "metadata", _immutable_metadata_map(self.metadata))


@dataclass(frozen=True, slots=True)
class ModelInputSchema:
    """Metadata describing what a model or session expects."""

    fields: tuple[ModelInputField, ...] = ()
    name: str = "model-input-consumer"
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalized_token(self.name, field_name="schema name"),
        )
        object.__setattr__(self, "metadata", _immutable_metadata_map(self.metadata))
        object.__setattr__(self, "fields", tuple(self.fields))
        seen: set[tuple[InputPhase, str]] = set()
        for field_def in self.fields:
            if not isinstance(field_def, ModelInputField):
                raise TypeError("fields must contain ModelInputField objects.")
            key = _field_key(field_def)
            if key in seen:
                phase, name = key
                raise ValueError(
                    f"Duplicate model input field {name!r} for phase {phase!r}."
                )
            seen.add(key)

    def required_fields(
        self,
        *,
        phase: InputPhase | None = None,
    ) -> tuple[ModelInputField, ...]:
        """Return required fields, optionally filtered by phase."""
        return tuple(
            field_def
            for field_def in self.fields
            if field_def.required and (phase is None or field_def.phase == phase)
        )

    def optional_fields(
        self,
        *,
        phase: InputPhase | None = None,
    ) -> tuple[ModelInputField, ...]:
        """Return optional fields, optionally filtered by phase."""
        return tuple(
            field_def
            for field_def in self.fields
            if not field_def.required and (phase is None or field_def.phase == phase)
        )

    def field_for(self, *, name: str, phase: InputPhase) -> ModelInputField | None:
        """Return one field definition, if present."""
        name = _normalized_token(name, field_name="model input name")
        phase = _validate_phase(phase)
        for field_def in self.fields:
            if field_def.name == name and field_def.phase == phase:
                return field_def
        return None


@dataclass(frozen=True, slots=True)
class ModelInputs:
    """Semantic model-facing input payloads split by runtime phase."""

    initial: Mapping[str, Any] = field(default_factory=dict)
    step: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial", _immutable_payload_map(self.initial))
        object.__setattr__(self, "step", _immutable_payload_map(self.step))

    @classmethod
    def initial_only(cls, values: Mapping[str, Any]) -> "ModelInputs":
        """Create model inputs containing only initial values."""
        return cls(initial=values)

    @classmethod
    def step_only(cls, values: Mapping[str, Any]) -> "ModelInputs":
        """Create model inputs containing only per-step values."""
        return cls(step=values)

    def for_phase(self, phase: InputPhase) -> Mapping[str, Any]:
        """Return the payload mapping for ``phase``."""
        phase = _validate_phase(phase)
        return self.initial if phase == "initial" else self.step


def _immutable_payload_map(values: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        normalized[_normalized_token(str(key), field_name="model input key")] = value
    return MappingProxyType(normalized)


def missing_required_inputs(
    inputs: ModelInputs,
    schema: ModelInputSchema,
    *,
    phase: InputPhase | None = None,
) -> tuple[ModelInputField, ...]:
    """Return required model fields absent from ``inputs``."""
    return tuple(
        field_def
        for field_def in schema.required_fields(phase=phase)
        if field_def.name not in inputs.for_phase(field_def.phase)
    )


def undeclared_model_inputs(
    inputs: ModelInputs,
    mapper_schema: "InputMapperSchema",
) -> tuple[tuple[InputPhase, str], ...]:
    """Return payload keys a mapper produced but did not declare in its schema.

    Mapper schemas are hand-written, so they can drift from what
    ``build_initial_inputs`` / ``build_step_inputs`` actually return. Mapper
    tests can use this to keep the declared compatibility surface honest.
    """
    declared = {
        (field_def.phase, field_def.name) for field_def in mapper_schema.produces
    }
    phases: tuple[InputPhase, ...] = ("initial", "step")
    return tuple(
        (phase, key)
        for phase in phases
        for key in inputs.for_phase(phase)
        if (phase, key) not in declared
    )


@dataclass(frozen=True, slots=True)
class InputMapperSchema:
    """Metadata for a mapper that converts user events into model inputs."""

    consumes: tuple[UserInputCapability, ...] = ()
    produces: tuple[ModelInputField, ...] = ()
    name: str = "input-mapper"
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalized_token(self.name, field_name="mapper name"),
        )
        object.__setattr__(self, "metadata", _immutable_metadata_map(self.metadata))
        object.__setattr__(self, "consumes", tuple(self.consumes))
        object.__setattr__(self, "produces", tuple(self.produces))
        for capability in self.consumes:
            if not isinstance(capability, UserInputCapability):
                raise TypeError("consumes must contain UserInputCapability objects.")
        for field_def in self.produces:
            if not isinstance(field_def, ModelInputField):
                raise TypeError("produces must contain ModelInputField objects.")

    def can_produce(self, model_field: ModelInputField) -> bool:
        """Return whether this mapper can produce ``model_field``."""
        return any(
            _model_field_matches(produced, model_field) for produced in self.produces
        )


@runtime_checkable
class InputMapper(Protocol):
    """Contract for user-event to model-input conversion."""

    @property
    def schema(self) -> InputMapperSchema:
        """Return mapper metadata used for compatibility checks."""
        ...

    def build_initial_inputs(self, trace: UserInputTrace) -> ModelInputs:
        """Build session-start model inputs from a user-input trace."""
        ...

    def build_step_inputs(self, window: UserInputWindow) -> ModelInputs:
        """Build per-step model inputs from one user-input window."""
        ...


@dataclass(frozen=True, slots=True)
class StaticInputMapper:
    """Mapper for fixed or already prepared model inputs.

    This is the no-op mapping path for runs that do not need live user
    controls, such as prompt-only CLI runs or model-input replay. It consumes
    no user events and returns configured model-facing payloads.
    """

    schema: InputMapperSchema
    inputs: ModelInputs = field(default_factory=ModelInputs)

    @classmethod
    def from_inputs(
        cls,
        *,
        inputs: ModelInputs,
        name: str = "static-input-mapper",
    ) -> "StaticInputMapper":
        """Create a static mapper whose produced fields come from ``inputs``."""
        produces = tuple(
            ModelInputField(name=field_name, phase=phase, required=False)
            for phase, values in (
                ("initial", inputs.initial),
                ("step", inputs.step),
            )
            for field_name in values
        )
        return cls(
            schema=InputMapperSchema(name=name, produces=produces),
            inputs=inputs,
        )

    def build_initial_inputs(self, trace: UserInputTrace) -> ModelInputs:
        """Return configured initial model inputs."""
        del trace
        return ModelInputs(initial=self.inputs.initial)

    def build_step_inputs(self, window: UserInputWindow) -> ModelInputs:
        """Return configured per-step model inputs."""
        del window
        return ModelInputs(step=self.inputs.step)


@dataclass(frozen=True, slots=True)
class MappingCompatibility:
    """Compatibility report for one source, model schema, and mapping.

    ``mapper_schema`` is the full requested mapping surface. Mappers whose
    consumed capabilities the source cannot provide are reported separately in
    ``unavailable_mapper_schemas`` and are excluded from the satisfied/available
    model-field reports, so those lists only name fields that can really be
    produced by this source.
    """

    source_schema: UserInputSchema
    model_schema: ModelInputSchema
    mapper_schema: InputMapperSchema
    missing_source_capabilities: tuple[UserInputCapability, ...]
    missing_required_model_fields: tuple[ModelInputField, ...]
    satisfied_required_model_fields: tuple[ModelInputField, ...]
    available_optional_model_fields: tuple[ModelInputField, ...]
    unavailable_mapper_schemas: tuple[InputMapperSchema, ...] = ()

    @property
    def can_drive(self) -> bool:
        """Return whether this source can drive this model through the mapping.

        Mappers that the source cannot feed do not block the run unless they
        were the only way to produce a required model input.
        """
        return (
            not self.missing_source_capabilities
            and not self.missing_required_model_fields
        )

    @property
    def unavailable_mapper_names(self) -> tuple[str, ...]:
        """Return names of mappers dropped because the source cannot feed them."""
        return tuple(schema.name for schema in self.unavailable_mapper_schemas)

    def raise_if_incompatible(self) -> None:
        """Raise a compact error when this mapping cannot drive the model."""
        if self.can_drive:
            return
        problems: list[str] = []
        if self.missing_source_capabilities:
            missing = ", ".join(
                capability.event_kind for capability in self.missing_source_capabilities
            )
            problems.append(f"missing source capabilities: {missing}")
        if self.missing_required_model_fields:
            missing = ", ".join(
                f"{field_def.phase}:{field_def.name}"
                for field_def in self.missing_required_model_fields
            )
            problems.append(f"missing required model inputs: {missing}")
        if self.unavailable_mapper_schemas:
            problems.append(
                "unavailable mappers: " + ", ".join(self.unavailable_mapper_names)
            )
        raise ValueError(
            f"Input mapper {self.mapper_schema.name!r} cannot drive model "
            f"{self.model_schema.name!r} from source {self.source_schema.name!r}: "
            + "; ".join(problems)
        )


def _mapper_is_feedable(
    source_schema: UserInputSchema,
    mapper_schema: InputMapperSchema,
) -> bool:
    return all(
        source_schema.supports(capability) for capability in mapper_schema.consumes
    )


def _build_compatibility(
    *,
    source_schema: UserInputSchema,
    model_schema: ModelInputSchema,
    mapper_schemas: Sequence[InputMapperSchema],
    reported_schema: InputMapperSchema,
) -> MappingCompatibility:
    feedable: list[InputMapperSchema] = []
    unavailable: list[InputMapperSchema] = []
    for mapper_schema in mapper_schemas:
        if _mapper_is_feedable(source_schema, mapper_schema):
            feedable.append(mapper_schema)
        else:
            unavailable.append(mapper_schema)

    usable = combine_mapper_schemas(feedable, name=reported_schema.name)
    required_fields = model_schema.required_fields()
    missing_required_model_fields = tuple(
        field_def for field_def in required_fields if not usable.can_produce(field_def)
    )
    satisfied_required_model_fields = tuple(
        field_def for field_def in required_fields if usable.can_produce(field_def)
    )
    available_optional_model_fields = tuple(
        field_def
        for field_def in model_schema.optional_fields()
        if usable.can_produce(field_def)
    )

    # Only capabilities that block a required model input make the mapping
    # unusable. A dropped mapper that fed nothing but optional fields degrades
    # instead of vetoing the run.
    missing_source_capabilities: list[UserInputCapability] = []
    seen_missing: set[UserInputCapability] = set()
    for mapper_schema in unavailable:
        if not any(
            mapper_schema.can_produce(field_def)
            for field_def in missing_required_model_fields
        ):
            continue
        for capability in mapper_schema.consumes:
            if source_schema.supports(capability) or capability in seen_missing:
                continue
            seen_missing.add(capability)
            missing_source_capabilities.append(capability)

    return MappingCompatibility(
        source_schema=source_schema,
        model_schema=model_schema,
        mapper_schema=reported_schema,
        missing_source_capabilities=tuple(missing_source_capabilities),
        missing_required_model_fields=missing_required_model_fields,
        satisfied_required_model_fields=satisfied_required_model_fields,
        available_optional_model_fields=available_optional_model_fields,
        unavailable_mapper_schemas=tuple(unavailable),
    )


def check_mapping_compatibility(
    *,
    source_schema: UserInputSchema,
    model_schema: ModelInputSchema,
    mapper_schema: InputMapperSchema,
) -> MappingCompatibility:
    """Check whether a user-input source can drive a model through a mapper."""
    if not isinstance(mapper_schema, InputMapperSchema):
        raise TypeError("mapper_schema must be an InputMapperSchema object.")
    return _build_compatibility(
        source_schema=source_schema,
        model_schema=model_schema,
        mapper_schemas=(mapper_schema,),
        reported_schema=mapper_schema,
    )


def combine_mapper_schemas(
    mapper_schemas: Sequence[InputMapperSchema],
    *,
    name: str = "input-mapper-set",
) -> InputMapperSchema:
    """Combine independently declared mappers into one compatibility surface.

    Duplicate entries are collapsed. Because ``metadata`` is excluded from
    equality, the metadata of collapsed duplicates is merged rather than
    dropped, with the first declaration winning on conflicting keys.
    """
    consumes: list[UserInputCapability] = []
    produces: list[ModelInputField] = []
    consumes_index: dict[UserInputCapability, int] = {}
    produces_index: dict[ModelInputField, int] = {}

    for mapper_schema in mapper_schemas:
        if not isinstance(mapper_schema, InputMapperSchema):
            raise TypeError("mapper_schemas must contain InputMapperSchema objects.")
        for capability in mapper_schema.consumes:
            index = consumes_index.get(capability)
            if index is None:
                consumes_index[capability] = len(consumes)
                consumes.append(capability)
            elif capability.metadata:
                consumes[index] = replace(
                    consumes[index],
                    metadata=_merged_metadata(
                        consumes[index].metadata, capability.metadata
                    ),
                )
        for field_def in mapper_schema.produces:
            index = produces_index.get(field_def)
            if index is None:
                produces_index[field_def] = len(produces)
                produces.append(field_def)
            elif field_def.metadata:
                produces[index] = replace(
                    produces[index],
                    metadata=_merged_metadata(
                        produces[index].metadata, field_def.metadata
                    ),
                )

    return InputMapperSchema(
        name=name,
        consumes=tuple(consumes),
        produces=tuple(produces),
    )


def check_mapping_set_compatibility(
    *,
    source_schema: UserInputSchema,
    model_schema: ModelInputSchema,
    mapper_schemas: Sequence[InputMapperSchema],
    name: str = "input-mapper-set",
) -> MappingCompatibility:
    """Check compatibility for a composed set of input mappers.

    Each mapper keeps its own ``consumes``/``produces`` link, so a mapper the
    source cannot feed only costs the model inputs that mapper produced.
    """
    mapper_schemas = tuple(mapper_schemas)
    return _build_compatibility(
        source_schema=source_schema,
        model_schema=model_schema,
        mapper_schemas=mapper_schemas,
        reported_schema=combine_mapper_schemas(mapper_schemas, name=name),
    )
