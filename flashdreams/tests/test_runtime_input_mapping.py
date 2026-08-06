# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for declarative input-mapping compatibility in the runtime API.

These cover the T2/T3 contract: sources declare what user events they can
provide at payload granularity, models declare required and optional global
conditioning/per-step inputs, and a mapping declares what it consumes and
produces so compatibility can be answered before expensive runtime
initialization.
"""

from __future__ import annotations

from typing import Any

import pytest

from flashdreams.runtime import (
    DRIVER_COMMAND,
    CanonicalInputs,
    CanonicalInputSchema,
    CanonicalModality,
    IdentityInputMapping,
    InferenceInput,
    InferenceInputSchema,
    InputField,
    InputMappingSchema,
    StepRequest,
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
    check_mapping_compatibility,
    check_mapping_set_compatibility,
    combine_mapping_schemas,
    undeclared_inference_inputs,
)

pytestmark = pytest.mark.ci_cpu

KEY_DOWN = UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"}))
KEY_UP = UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"}))
PROMPT_SET = UserInputCapability(
    event_type="prompt_set",
    input_modality="text",
    payload_fields=frozenset({"prompt"}),
)
FRAME_SET = UserInputCapability(
    event_type="initial_frame_set", payload_fields=frozenset({"image"})
)

BROWSER_SOURCE = UserInputSchema(
    capabilities=(KEY_DOWN, KEY_UP, PROMPT_SET, FRAME_SET),
    description="browser webrtc client",
)

CAMERA_LOOK = CanonicalModality(
    name="camera_look", payload_fields=frozenset({"yaw", "pitch"})
)

CANONICAL_ALL = CanonicalInputSchema(modalities=(DRIVER_COMMAND, CAMERA_LOOK))

# Global conditioning is application-owned and does not come from a canonical
# modality, so this mapping consumes nothing and only declares what it produces.
PROMPT_MAPPING = InputMappingSchema(
    name="prompt",
    produces_global_conditioning=(InputField(name="prompt", input_modality="text"),),
)
FRAME_MAPPING = InputMappingSchema(
    name="conditioning-frame",
    produces_global_conditioning=(
        InputField(name="global_conditioning_frame", required=False),
    ),
)
STEERING_MAPPING = InputMappingSchema(
    name="driver-command-to-steering",
    consumes=(DRIVER_COMMAND,),
    produces_step=(InputField(name="steering"),),
)
LOOK_MAPPING = InputMappingSchema(
    name="camera-look",
    consumes=(CAMERA_LOOK,),
    produces_step=(InputField(name="camera_delta", required=False),),
)

DRIVING_MODEL = InferenceInputSchema(
    global_conditioning_fields=(InputField(name="prompt", input_modality="text"),),
    step_fields=(
        InputField(name="steering"),
        InputField(name="camera_delta", required=False),
    ),
)


# --- user input events and windowing ------------------------------------


def test_session_start_values_are_represented_as_events() -> None:
    inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.0, event_type="prompt_set", payload={"prompt": "drive"}
            ),
            UserInputEvent(
                timestamp_s=0.5, event_type="key_down", payload={"key": "w"}
            ),
        )
    )

    assert inputs.events[0].event_type == "prompt_set"
    assert inputs.events[0].payload["prompt"] == "drive"


def test_windowing_is_half_open_and_deterministic() -> None:
    inputs = UserInputs(
        events=tuple(
            UserInputEvent(timestamp_s=t, event_type="key_down", payload={"key": "w"})
            for t in (0.0, 0.5, 1.0, 1.5)
        )
    )

    windowed = inputs.window(TimeWindow(start_s=0.5, end_s=1.5))

    assert [event.timestamp_s for event in windowed.events] == [0.5, 1.0]


def test_out_of_order_events_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        UserInputs(
            events=(
                UserInputEvent(timestamp_s=1.0, event_type="key_down"),
                UserInputEvent(timestamp_s=0.5, event_type="key_up"),
            )
        )


# --- user input schemas -------------------------------------------------


def test_source_declares_capabilities_at_payload_granularity() -> None:
    assert BROWSER_SOURCE.supports(KEY_DOWN)
    assert not BROWSER_SOURCE.supports(
        UserInputCapability(
            event_type="key_down", payload_fields=frozenset({"key", "modifiers"})
        )
    )


def test_bare_event_types_still_satisfy_payload_free_consumers() -> None:
    """Coarse pre-capability schemas keep working against the finer query."""
    coarse = UserInputSchema(event_types=frozenset({"reset"}))

    assert coarse.supports(UserInputCapability(event_type="reset"))
    assert not coarse.supports(
        UserInputCapability(event_type="reset", payload_fields=frozenset({"reason"}))
    )
    assert coarse.supports_event_types({"reset"})


def test_capabilities_widen_declared_event_types() -> None:
    assert "key_down" in BROWSER_SOURCE.declared_event_types()
    assert BROWSER_SOURCE.supports_event_types({"key_down", "prompt_set"})


def test_input_modality_mismatch_blocks_capability_match() -> None:
    source = UserInputSchema(
        capabilities=(
            UserInputCapability(event_type="prompt_set", input_modality="embedding"),
        )
    )

    assert not source.supports(
        UserInputCapability(event_type="prompt_set", input_modality="text")
    )


def test_event_validation_reports_missing_payload_fields() -> None:
    event = UserInputEvent(timestamp_s=0.0, event_type="key_down", payload={})

    with pytest.raises(ValueError, match="missing required"):
        BROWSER_SOURCE.validate_event(event)


def test_event_validation_rejects_undeclared_event_type() -> None:
    event = UserInputEvent(timestamp_s=0.0, event_type="wheel_axis")

    with pytest.raises(ValueError, match="does not provide event type"):
        BROWSER_SOURCE.validate_event(event)


# --- model input schemas ------------------------------------------------


def test_model_declares_required_and_optional_fields_per_phase() -> None:
    required = DRIVING_MODEL.required_fields()
    optional = DRIVING_MODEL.optional_fields()

    assert {(phase, f.name) for phase, f in required} == {
        ("global_conditioning", "prompt"),
        ("step", "steering"),
    }
    assert {(phase, f.name) for phase, f in optional} == {("step", "camera_delta")}


def test_required_fields_can_be_filtered_by_phase() -> None:
    step_only = DRIVING_MODEL.required_fields("step")

    assert [f.name for _, f in step_only] == ["steering"]


def test_field_lookup_is_phase_scoped() -> None:
    assert (
        DRIVING_MODEL.field_for(name="prompt", phase="global_conditioning") is not None
    )
    assert DRIVING_MODEL.field_for(name="prompt", phase="step") is None


def test_invalid_phase_is_rejected() -> None:
    bad_phase: Any = "final"

    with pytest.raises(ValueError, match="phase must be"):
        DRIVING_MODEL.fields_for(bad_phase)


def test_inference_input_expose_payload_per_phase() -> None:
    inputs = InferenceInput(
        global_conditioning={"prompt": "drive"}, step={"steering": 0.25}
    )

    assert inputs.for_phase("global_conditioning")["prompt"] == "drive"
    assert inputs.for_phase("step")["steering"] == 0.25


def test_step_context_can_carry_global_conditioning_update_payload() -> None:
    inputs = InferenceInput(
        global_conditioning={"prompt": "heavy rain"},
        step={"steering": 0.25},
    )

    assert inputs.global_conditioning["prompt"] == "heavy rain"
    assert inputs.step["steering"] == 0.25


def test_field_metadata_is_queryable() -> None:
    field = InputField(
        name="prompt",
        frequency_consumed="once",
        metadata={"coordinates": "opencv_c2w"},
    )

    assert field.frequency_consumed == "once"
    assert field.metadata["coordinates"] == "opencv_c2w"


def test_metadata_is_excluded_from_field_equality() -> None:
    plain = InputField(name="prompt")
    annotated = InputField(name="prompt", metadata={"note": "hint"})

    assert plain == annotated


# --- mapping compatibility ----------------------------------------------


def test_compatible_source_model_and_mapping_can_drive() -> None:
    compatibility = check_mapping_set_compatibility(
        canonical_schema=CANONICAL_ALL,
        inference_input_schema=DRIVING_MODEL,
        mapping_schemas=(PROMPT_MAPPING, STEERING_MAPPING, LOOK_MAPPING),
    )

    assert compatibility.can_drive
    assert {(p, f.name) for p, f in compatibility.satisfied_required_model_fields} == {
        ("global_conditioning", "prompt"),
        ("step", "steering"),
    }
    assert {(p, f.name) for p, f in compatibility.available_optional_model_fields} == {
        ("step", "camera_delta")
    }


def test_missing_required_model_field_blocks_the_run() -> None:
    compatibility = check_mapping_set_compatibility(
        canonical_schema=CANONICAL_ALL,
        inference_input_schema=DRIVING_MODEL,
        mapping_schemas=(PROMPT_MAPPING,),
    )

    assert not compatibility.can_drive
    assert [f.name for _, f in compatibility.missing_required_model_fields] == [
        "steering"
    ]


def test_missing_source_capability_is_reported_when_it_blocks() -> None:
    no_wheel = CanonicalInputSchema(modalities=(CAMERA_LOOK,))

    compatibility = check_mapping_set_compatibility(
        canonical_schema=no_wheel,
        inference_input_schema=DRIVING_MODEL,
        mapping_schemas=(PROMPT_MAPPING, STEERING_MAPPING),
    )

    assert not compatibility.can_drive
    assert compatibility.unavailable_mapping_names == ("driver-command-to-steering",)
    assert {m.name for m in compatibility.missing_modalities} == {"driver_command"}


def test_unfeedable_optional_mapping_degrades_instead_of_vetoing() -> None:
    """Losing a mapping that fed only optional fields must not block the run."""
    no_look = CanonicalInputSchema(modalities=(DRIVER_COMMAND,))

    compatibility = check_mapping_set_compatibility(
        canonical_schema=no_look,
        inference_input_schema=DRIVING_MODEL,
        mapping_schemas=(PROMPT_MAPPING, STEERING_MAPPING, LOOK_MAPPING),
    )

    assert compatibility.can_drive
    assert compatibility.unavailable_mapping_names == ("camera-look",)
    # The dropped mapping's field must not be advertised as available.
    assert compatibility.available_optional_model_fields == ()


def test_optional_field_needs_mapping_support_to_be_available() -> None:
    compatibility = check_mapping_set_compatibility(
        canonical_schema=CANONICAL_ALL,
        inference_input_schema=DRIVING_MODEL,
        mapping_schemas=(PROMPT_MAPPING, STEERING_MAPPING),
    )

    assert compatibility.can_drive
    assert compatibility.available_optional_model_fields == ()


def test_global_conditioning_mapping_matches_global_conditioning_field() -> None:
    model = InferenceInputSchema(
        global_conditioning_fields=(
            InputField(name="camera_trajectory", frequency_consumed="per_step"),
        )
    )
    mapping = InputMappingSchema(
        name="trajectory",
        produces_global_conditioning=(
            InputField(name="camera_trajectory", frequency_consumed="once"),
        ),
    )

    compatibility = check_mapping_compatibility(
        canonical_schema=CANONICAL_ALL,
        inference_input_schema=model,
        mapping_schema=mapping,
    )

    assert compatibility.can_drive


def test_unspecified_input_modality_stays_permissive() -> None:
    model = InferenceInputSchema(
        global_conditioning_fields=(InputField(name="prompt"),)
    )

    compatibility = check_mapping_compatibility(
        canonical_schema=CANONICAL_ALL,
        inference_input_schema=model,
        mapping_schema=PROMPT_MAPPING,
    )

    assert compatibility.can_drive


def test_raise_if_incompatible_names_both_failure_kinds() -> None:
    compatibility = check_mapping_set_compatibility(
        canonical_schema=CanonicalInputSchema(modalities=(CAMERA_LOOK,)),
        inference_input_schema=DRIVING_MODEL,
        mapping_schemas=(PROMPT_MAPPING, STEERING_MAPPING),
    )

    with pytest.raises(ValueError) as excinfo:
        compatibility.raise_if_incompatible()

    message = str(excinfo.value)
    assert "missing canonical modalities" in message
    assert "missing required model inputs" in message


def test_raise_if_incompatible_is_a_no_op_when_compatible() -> None:
    compatibility = check_mapping_set_compatibility(
        canonical_schema=CANONICAL_ALL,
        inference_input_schema=DRIVING_MODEL,
        mapping_schemas=(PROMPT_MAPPING, FRAME_MAPPING, STEERING_MAPPING),
    )

    compatibility.raise_if_incompatible()


def test_check_mapping_compatibility_rejects_a_non_schema() -> None:
    not_a_schema: Any = object()

    with pytest.raises(TypeError, match="InputMappingSchema"):
        check_mapping_compatibility(
            canonical_schema=CANONICAL_ALL,
            inference_input_schema=DRIVING_MODEL,
            mapping_schema=not_a_schema,
        )


# --- mapping schema composition -----------------------------------------


def test_combining_mappings_unions_their_surfaces() -> None:
    combined = combine_mapping_schemas((PROMPT_MAPPING, STEERING_MAPPING))

    assert {m.name for m in combined.consumes} == {"driver_command"}
    assert [f.name for f in combined.produces_global_conditioning] == ["prompt"]
    assert [f.name for f in combined.produces_step] == ["steering"]


def test_duplicate_declarations_collapse_and_merge_metadata() -> None:
    first = InputMappingSchema(
        name="a",
        produces_global_conditioning=(
            InputField(name="prompt", metadata={"source": "a"}),
        ),
    )
    second = InputMappingSchema(
        name="b",
        produces_global_conditioning=(
            InputField(name="prompt", metadata={"source": "b", "extra": "kept"}),
        ),
    )

    combined = combine_mapping_schemas((first, second))

    assert len(combined.produces_global_conditioning) == 1
    metadata = combined.produces_global_conditioning[0].metadata
    assert metadata["source"] == "a"
    assert metadata["extra"] == "kept"


def test_combine_rejects_non_schema_entries() -> None:
    not_a_schema: Any = object()

    with pytest.raises(TypeError, match="InputMappingSchema"):
        combine_mapping_schemas((PROMPT_MAPPING, not_a_schema))


# --- declaration drift --------------------------------------------------


def test_undeclared_inference_input_catches_schema_drift() -> None:
    produced = InferenceInput(
        global_conditioning={"prompt": "drive"}, step={"steering": 0.0}
    )

    undeclared = undeclared_inference_inputs(produced, PROMPT_MAPPING)

    assert undeclared == (("step", "steering"),)


def test_declared_outputs_report_no_drift() -> None:
    combined = combine_mapping_schemas((PROMPT_MAPPING, STEERING_MAPPING))
    produced = InferenceInput(
        global_conditioning={"prompt": "drive"}, step={"steering": 0.0}
    )

    assert undeclared_inference_inputs(produced, combined) == ()


# --- interoperability with the T1 envelope ------------------------------


def test_identity_mapping_needs_no_declared_surface() -> None:
    """Fixed-input runs stay possible without any schema declaration."""
    mapping = IdentityInputMapping()
    fixed = InferenceInput(
        global_conditioning={"prompt": "fixed"}, step={"steering": 0.0}
    )

    mapped = mapping.map_step_inputs(
        canonical_inputs=CanonicalInputs(),
        inference_input=fixed,
        request=StepRequest(step_index=0),
    )

    assert mapped.step["steering"] == 0.0


def test_empty_mapping_set_cannot_satisfy_a_required_field() -> None:
    compatibility = check_mapping_set_compatibility(
        canonical_schema=CANONICAL_ALL,
        inference_input_schema=DRIVING_MODEL,
        mapping_schemas=(),
    )

    assert not compatibility.can_drive
    assert len(compatibility.missing_required_model_fields) == 2


def test_model_with_no_requirements_is_always_drivable() -> None:
    compatibility = check_mapping_set_compatibility(
        canonical_schema=CanonicalInputSchema(),
        inference_input_schema=InferenceInputSchema(),
        mapping_schemas=(),
    )

    assert compatibility.can_drive
