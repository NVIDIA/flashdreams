# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input mapping boundary from canonical inputs to encoded inference inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.inputs import (
    INPUT_PHASES,
    CanonicalInputs,
    CanonicalInputSchema,
    CanonicalModality,
    InferenceInput,
    InferenceInputSchema,
    InputField,
    InputPhase,
)
from flashdreams.runtime.types import StepRequest


@runtime_checkable
class InputMapping(Protocol):
    """Convert user-facing inputs into model-facing inputs.

    A mapping may be supplied by the model adapter as a default or by an
    application/runtime override. Step mappings usually receive a timestamped
    event window selected by the runner for the current model step or chunk.
    """

    def validate(
        self,
        *,
        canonical_schema: CanonicalInputSchema | None = None,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        """Fail early for obvious app, event-source, and model mismatches."""
        ...

    def map_global_conditioning_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
    ) -> InferenceInput:
        """Build global conditioning inputs for session start or reset."""
        ...

    def map_step_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
        request: StepRequest,
    ) -> InferenceInput:
        """Build model inputs for one session step from the current input window."""
        ...


class IdentityInputMapping:
    """No-op mapper for fixed model-input or simple generation flows."""

    def validate(
        self,
        *,
        canonical_schema: CanonicalInputSchema | None = None,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        del canonical_schema, inference_input_schema

    def map_global_conditioning_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
    ) -> InferenceInput:
        del canonical_inputs
        return inference_input

    def map_step_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
        request: StepRequest,
    ) -> InferenceInput:
        del canonical_inputs, request
        return inference_input


@dataclass(frozen=True, kw_only=True, slots=True)
class InputMappingSchema:
    """Declarative compatibility surface for one mapping.

    ``InputMapping.validate`` fails a run late and opaquely: it raises, but it
    cannot answer which optional model inputs a source would enable, or which
    missing user capability is responsible for an unreachable model input. This
    schema makes those questions answerable before runtime initialization.
    """

    name: str = "input-mapping"
    consumes: tuple[CanonicalModality, ...] = ()
    produces_global_conditioning: tuple[InputField, ...] = ()
    produces_step: tuple[InputField, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("InputMappingSchema.name must be non-empty.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def produces_for(self, phase: InputPhase) -> tuple[InputField, ...]:
        """Return the fields this mapping produces for ``phase``."""
        return (
            self.produces_global_conditioning
            if phase == "global_conditioning"
            else self.produces_step
        )

    def can_produce(self, phase: InputPhase, required: InputField) -> bool:
        """Return whether this mapping can produce ``required`` in ``phase``."""
        return any(
            _field_matches(produced, required) for produced in self.produces_for(phase)
        )


def _field_matches(produced: InputField, required: InputField) -> bool:
    if produced.name != required.name:
        return False
    input_modality_ok = (
        produced.input_modality is None
        or required.input_modality is None
        or produced.input_modality == required.input_modality
    )
    return input_modality_ok


@dataclass(frozen=True, kw_only=True, slots=True)
class MappingCompatibility:
    """Compatibility report for one source, model schema, and mapping set.

    Mappings whose consumed capabilities the source cannot provide are reported
    in ``unavailable_mapping_schemas`` and excluded from the satisfied/available
    reports, so those lists only name model inputs that can really be produced.
    """

    __hash__ = None

    canonical_schema: CanonicalInputSchema
    inference_input_schema: InferenceInputSchema
    mapping_schema: InputMappingSchema
    missing_modalities: tuple[CanonicalModality, ...] = ()
    missing_required_model_fields: tuple[tuple[InputPhase, InputField], ...] = ()
    satisfied_required_model_fields: tuple[tuple[InputPhase, InputField], ...] = ()
    available_optional_model_fields: tuple[tuple[InputPhase, InputField], ...] = ()
    unavailable_mapping_schemas: tuple[InputMappingSchema, ...] = ()

    @property
    def can_drive(self) -> bool:
        """Return whether this source can drive this model through the mapping.

        A mapping the source cannot feed does not block the run unless it was
        the only way to produce a required model input.
        """
        return not (self.missing_required_model_fields or self.missing_modalities)

    @property
    def unavailable_mapping_names(self) -> tuple[str, ...]:
        """Return names of mappings dropped because the source cannot feed them."""
        return tuple(schema.name for schema in self.unavailable_mapping_schemas)

    def raise_if_incompatible(self) -> None:
        """Raise a compact error when this mapping cannot drive the model."""
        if self.can_drive:
            return
        problems: list[str] = []
        if self.missing_modalities:
            missing = ", ".join(modality.name for modality in self.missing_modalities)
            problems.append(f"missing canonical modalities: {missing}")
        if self.missing_required_model_fields:
            missing = ", ".join(
                f"{phase}:{input_field.name}"
                for phase, input_field in self.missing_required_model_fields
            )
            problems.append(f"missing required model inputs: {missing}")
        if self.unavailable_mapping_schemas:
            problems.append(
                "unavailable mappings: " + ", ".join(self.unavailable_mapping_names)
            )
        raise ValueError(
            f"Input mapping {self.mapping_schema.name!r} cannot drive this model "
            f"from the selected source: " + "; ".join(problems)
        )


def _source_can_feed(
    canonical_schema: CanonicalInputSchema,
    mapping_schema: InputMappingSchema,
) -> bool:
    return all(
        canonical_schema.supports(modality) for modality in mapping_schema.consumes
    )


def combine_mapping_schemas(
    mapping_schemas: Sequence[InputMappingSchema],
    *,
    name: str = "input-mapping-set",
) -> InputMappingSchema:
    """Combine independently declared mappings into one compatibility surface.

    Duplicates are collapsed. Because ``metadata`` is excluded from equality,
    the metadata of collapsed duplicates is merged rather than dropped, with the
    first declaration winning on conflicting keys.
    """
    consumes: list[CanonicalModality] = []
    produces: dict[InputPhase, list[InputField]] = {
        "global_conditioning": [],
        "step": [],
    }

    def _merge(target: list[Any], value: Any) -> None:
        for index, existing in enumerate(target):
            if existing == value:
                if value.metadata:
                    target[index] = replace(
                        existing,
                        metadata={**dict(value.metadata), **dict(existing.metadata)},
                    )
                return
        target.append(value)

    for mapping_schema in mapping_schemas:
        if not isinstance(mapping_schema, InputMappingSchema):
            raise TypeError("mapping_schemas must contain InputMappingSchema objects.")
        for modality in mapping_schema.consumes:
            _merge(consumes, modality)
        for phase in INPUT_PHASES:
            for input_field in mapping_schema.produces_for(phase):
                _merge(produces[phase], input_field)

    return InputMappingSchema(
        name=name,
        consumes=tuple(consumes),
        produces_global_conditioning=tuple(produces["global_conditioning"]),
        produces_step=tuple(produces["step"]),
    )


def _build_compatibility(
    *,
    canonical_schema: CanonicalInputSchema,
    inference_input_schema: InferenceInputSchema,
    mapping_schemas: Sequence[InputMappingSchema],
    reported_schema: InputMappingSchema,
) -> MappingCompatibility:
    feedable: list[InputMappingSchema] = []
    unavailable: list[InputMappingSchema] = []
    for mapping_schema in mapping_schemas:
        if _source_can_feed(canonical_schema, mapping_schema):
            feedable.append(mapping_schema)
        else:
            unavailable.append(mapping_schema)

    usable = combine_mapping_schemas(feedable, name=reported_schema.name)
    required = inference_input_schema.required_fields()
    missing_required = tuple(
        (phase, input_field)
        for phase, input_field in required
        if not usable.can_produce(phase, input_field)
    )
    satisfied_required = tuple(
        (phase, input_field)
        for phase, input_field in required
        if usable.can_produce(phase, input_field)
    )
    available_optional = tuple(
        (phase, input_field)
        for phase, input_field in inference_input_schema.optional_fields()
        if usable.can_produce(phase, input_field)
    )

    # Only capabilities that block a required model input make the mapping
    # unusable. A dropped mapping that fed nothing but optional fields degrades
    # the run instead of vetoing it.
    missing_modalities: list[CanonicalModality] = []
    for mapping_schema in unavailable:
        if not any(
            mapping_schema.can_produce(phase, input_field)
            for phase, input_field in missing_required
        ):
            continue
        for modality in mapping_schema.consumes:
            if canonical_schema.supports(modality) or modality in missing_modalities:
                continue
            missing_modalities.append(modality)

    return MappingCompatibility(
        canonical_schema=canonical_schema,
        inference_input_schema=inference_input_schema,
        mapping_schema=reported_schema,
        missing_modalities=tuple(missing_modalities),
        missing_required_model_fields=missing_required,
        satisfied_required_model_fields=satisfied_required,
        available_optional_model_fields=available_optional,
        unavailable_mapping_schemas=tuple(unavailable),
    )


def check_mapping_compatibility(
    *,
    canonical_schema: CanonicalInputSchema,
    inference_input_schema: InferenceInputSchema,
    mapping_schema: InputMappingSchema,
) -> MappingCompatibility:
    """Check whether a user-input source can drive a model through a mapping."""
    if not isinstance(mapping_schema, InputMappingSchema):
        raise TypeError("mapping_schema must be an InputMappingSchema object.")
    return _build_compatibility(
        canonical_schema=canonical_schema,
        inference_input_schema=inference_input_schema,
        mapping_schemas=(mapping_schema,),
        reported_schema=mapping_schema,
    )


def check_mapping_set_compatibility(
    *,
    canonical_schema: CanonicalInputSchema,
    inference_input_schema: InferenceInputSchema,
    mapping_schemas: Sequence[InputMappingSchema],
    name: str = "input-mapping-set",
) -> MappingCompatibility:
    """Check compatibility for a composed set of mappings.

    Each mapping keeps its own consumes/produces link, so a mapping the source
    cannot feed only costs the model inputs that mapping produced.
    """
    mapping_schemas = tuple(mapping_schemas)
    return _build_compatibility(
        canonical_schema=canonical_schema,
        inference_input_schema=inference_input_schema,
        mapping_schemas=mapping_schemas,
        reported_schema=combine_mapping_schemas(mapping_schemas, name=name),
    )


def undeclared_inference_inputs(
    inputs: InferenceInput,
    mapping_schema: InputMappingSchema,
) -> tuple[tuple[InputPhase, str], ...]:
    """Return payload keys a mapping produced but did not declare.

    Mapping schemas are hand-written, so they drift from what
    ``map_global_conditioning_inputs``/``map_step_inputs`` actually return.
    Mapping tests can use this to keep the declared compatibility surface
    honest.
    """
    return tuple(
        (phase, key)
        for phase in INPUT_PHASES
        for key in inputs.for_phase(phase)
        if not any(
            declared.name == key for declared in mapping_schema.produces_for(phase)
        )
    )


@runtime_checkable
class DeclaresMappingSchema(Protocol):
    """Optional refinement of :class:`InputMapping` that declares its surface."""

    @property
    def mapping_schema(self) -> InputMappingSchema:
        """Return the declarative compatibility surface for this mapping."""
        ...
