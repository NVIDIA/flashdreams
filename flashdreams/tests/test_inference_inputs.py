# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest

from flashdreams.inference import (
    InputMapperSchema,
    ModelInputField,
    ModelInputSchema,
    ModelInputs,
    StaticInputMapper,
    UserInputCapability,
    UserInputEvent,
    UserInputSchema,
    UserInputTrace,
    UserInputWindow,
    check_mapping_compatibility,
    check_mapping_set_compatibility,
    combine_mapper_schemas,
    missing_required_inputs,
)

pytestmark = pytest.mark.ci_cpu


def test_user_input_trace_orders_events_and_slices_windows() -> None:
    key_down = UserInputEvent(
        timestamp_s=1.0,
        kind="key_down",
        payload={"key": "w"},
    )
    key_up = UserInputEvent(
        timestamp_s=1.0,
        kind="key_up",
        payload={"key": "w"},
    )
    prompt = UserInputEvent(
        timestamp_s=0.0,
        kind="prompt_set",
        payload={"prompt": "drive forward"},
    )

    trace = UserInputTrace.from_events([key_down, key_up, prompt])

    assert [event.kind for event in trace.events] == [
        "prompt_set",
        "key_down",
        "key_up",
    ]
    window = trace.window(start_s=0.5, end_s=1.0, include_end=True)
    assert window.events == (key_down, key_up)
    assert window.latest("key_down") is key_down


def test_static_startup_values_are_user_input_events() -> None:
    trace = UserInputTrace.from_events(
        [
            UserInputEvent(
                timestamp_s=0.0,
                kind="initial_frame_set",
                payload={"image": b"encoded-image"},
            ),
            UserInputEvent(
                timestamp_s=0.0,
                kind="scene_selected",
                payload={"scene_id": "scene-a"},
            ),
        ]
    )

    initial_frame = trace.latest("initial_frame_set")
    scene = trace.latest("scene_selected")
    assert initial_frame is not None
    assert scene is not None
    assert initial_frame.payload["image"] == b"encoded-image"
    assert scene.payload["scene_id"] == "scene-a"


def test_user_input_schema_declares_and_validates_source_capabilities() -> None:
    schema = UserInputSchema(
        name="browser",
        source_kind="live",
        metadata={"transport": "webrtc"},
        capabilities=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_kind="text",
                payload_fields=frozenset({"prompt"}),
                metadata={"source_widget": "prompt-box"},
            ),
            UserInputCapability(
                event_kind="key_down",
                payload_fields=frozenset({"key"}),
            ),
        ),
    )

    assert schema.supports(
        UserInputCapability(
            event_kind="prompt_set",
            payload_kind="text",
            payload_fields=frozenset({"prompt"}),
        )
    )
    assert schema.metadata["transport"] == "webrtc"
    assert schema.capabilities[0].metadata["source_widget"] == "prompt-box"
    schema.validate_event(
        UserInputEvent(
            timestamp_s=0.0,
            kind="prompt_set",
            payload={"prompt": "a prompt"},
        )
    )
    with pytest.raises(ValueError, match="missing required fields"):
        schema.validate_event(
            UserInputEvent(
                timestamp_s=0.0,
                kind="prompt_set",
                payload={"text": "wrong key"},
            )
        )


def test_model_input_schema_declares_required_optional_and_update_metadata() -> None:
    schema = ModelInputSchema(
        name="steering-model",
        metadata={"model_family": "example"},
        fields=(
            ModelInputField(
                name="prompt",
                phase="initial",
                required=True,
                payload_kind="text",
                update_policy="step_boundary",
                lifecycle="cache_init",
                metadata={"max_tokens": 256},
            ),
            ModelInputField(
                name="steering",
                phase="step",
                required=True,
                lifecycle="step_input",
            ),
            ModelInputField(name="first_frame", phase="initial", required=False),
        ),
    )

    assert [field.name for field in schema.required_fields()] == [
        "prompt",
        "steering",
    ]
    assert [field.name for field in schema.optional_fields()] == ["first_frame"]
    prompt_field = schema.field(name="prompt", phase="initial")
    assert prompt_field is not None
    assert prompt_field.update_policy == "step_boundary"
    assert prompt_field.lifecycle == "cache_init"
    assert prompt_field.metadata["max_tokens"] == 256
    assert schema.metadata["model_family"] == "example"


def test_model_inputs_report_missing_required_fields() -> None:
    schema = ModelInputSchema(
        fields=(
            ModelInputField(name="prompt", phase="initial"),
            ModelInputField(name="steering", phase="step"),
        )
    )
    inputs = ModelInputs(initial={"prompt": "drive"}, step={})

    missing = missing_required_inputs(inputs, schema)

    assert [(field.phase, field.name) for field in missing] == [
        ("step", "steering")
    ]


def test_mapping_compatibility_reports_satisfied_missing_and_optional_fields() -> None:
    source = UserInputSchema(
        name="keyboard-app",
        capabilities=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
            UserInputCapability(
                event_kind="key_down",
                payload_fields=frozenset({"key"}),
            ),
            UserInputCapability(
                event_kind="key_up",
                payload_fields=frozenset({"key"}),
            ),
        ),
    )
    model = ModelInputSchema(
        name="drive-model",
        fields=(
            ModelInputField(name="prompt", phase="initial"),
            ModelInputField(name="steering", phase="step"),
            ModelInputField(name="first_frame", phase="initial", required=False),
        ),
    )
    mapper = InputMapperSchema(
        name="keyboard-drive",
        consumes=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
            UserInputCapability(
                event_kind="key_down",
                payload_fields=frozenset({"key"}),
            ),
            UserInputCapability(
                event_kind="key_up",
                payload_fields=frozenset({"key"}),
            ),
        ),
        produces=(
            ModelInputField(name="prompt", phase="initial"),
            ModelInputField(name="steering", phase="step"),
        ),
    )

    compatibility = check_mapping_compatibility(
        source_schema=source,
        model_schema=model,
        mapper_schema=mapper,
    )

    assert compatibility.can_drive
    assert [field.name for field in compatibility.satisfied_required_model_fields] == [
        "prompt",
        "steering",
    ]
    assert compatibility.available_optional_model_fields == ()


def test_mapping_compatibility_reports_available_optional_model_input() -> None:
    source = UserInputSchema(
        capabilities=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
            UserInputCapability(
                event_kind="initial_frame_set",
                payload_fields=frozenset({"image"}),
            ),
        )
    )
    model = ModelInputSchema(
        fields=(
            ModelInputField(name="prompt", phase="initial"),
            ModelInputField(name="first_frame", phase="initial", required=False),
        )
    )
    mapper = InputMapperSchema(
        consumes=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
            UserInputCapability(
                event_kind="initial_frame_set",
                payload_fields=frozenset({"image"}),
            ),
        ),
        produces=(
            ModelInputField(name="prompt", phase="initial"),
            ModelInputField(name="first_frame", phase="initial", required=False),
        ),
    )

    compatibility = check_mapping_compatibility(
        source_schema=source,
        model_schema=model,
        mapper_schema=mapper,
    )

    assert compatibility.can_drive
    assert [
        (field.phase, field.name)
        for field in compatibility.available_optional_model_fields
    ] == [("initial", "first_frame")]


def test_mapping_compatibility_reports_missing_required_model_input() -> None:
    source = UserInputSchema(
        capabilities=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
        )
    )
    model = ModelInputSchema(
        fields=(
            ModelInputField(name="prompt", phase="initial"),
            ModelInputField(name="camera_trajectory", phase="step"),
        )
    )
    mapper = InputMapperSchema(
        consumes=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
        ),
        produces=(ModelInputField(name="prompt", phase="initial"),),
    )

    compatibility = check_mapping_compatibility(
        source_schema=source,
        model_schema=model,
        mapper_schema=mapper,
    )

    assert not compatibility.can_drive
    assert [
        (field.phase, field.name)
        for field in compatibility.missing_required_model_fields
    ] == [("step", "camera_trajectory")]
    with pytest.raises(ValueError, match="missing required model inputs"):
        compatibility.raise_if_incompatible()


def test_mapping_compatibility_reports_missing_source_capability() -> None:
    source = UserInputSchema(
        capabilities=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
        )
    )
    model = ModelInputSchema(
        fields=(ModelInputField(name="steering", phase="step"),)
    )
    mapper = InputMapperSchema(
        consumes=(
            UserInputCapability(
                event_kind="controller_axis",
                payload_fields=frozenset({"axis", "value"}),
            ),
        ),
        produces=(ModelInputField(name="steering", phase="step"),),
    )

    compatibility = check_mapping_compatibility(
        source_schema=source,
        model_schema=model,
        mapper_schema=mapper,
    )

    assert not compatibility.can_drive
    assert [
        capability.event_kind
        for capability in compatibility.missing_source_capabilities
    ] == ["controller_axis"]


def test_mapping_set_compatibility_supports_composed_model_inputs() -> None:
    source = UserInputSchema(
        name="browser-with-controls",
        capabilities=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
            UserInputCapability(
                event_kind="initial_frame_set",
                payload_fields=frozenset({"image"}),
            ),
            UserInputCapability(
                event_kind="key_down",
                payload_fields=frozenset({"key"}),
            ),
            UserInputCapability(
                event_kind="key_up",
                payload_fields=frozenset({"key"}),
            ),
        ),
    )
    model = ModelInputSchema(
        name="lingbot-like",
        fields=(
            ModelInputField(
                name="prompt",
                phase="initial",
                lifecycle="cache_init",
            ),
            ModelInputField(
                name="first_frame",
                phase="initial",
                lifecycle="cache_init",
            ),
            ModelInputField(
                name="camera_trajectory",
                phase="step",
                lifecycle="step_input",
            ),
        ),
    )
    prompt_mapper = InputMapperSchema(
        name="prompt",
        consumes=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
        ),
        produces=(
            ModelInputField(
                name="prompt",
                phase="initial",
                lifecycle="cache_init",
            ),
        ),
    )
    frame_mapper = InputMapperSchema(
        name="first-frame",
        consumes=(
            UserInputCapability(
                event_kind="initial_frame_set",
                payload_fields=frozenset({"image"}),
            ),
        ),
        produces=(
            ModelInputField(
                name="first_frame",
                phase="initial",
                lifecycle="cache_init",
            ),
        ),
    )
    camera_mapper = InputMapperSchema(
        name="keyboard-to-camera",
        consumes=(
            UserInputCapability(
                event_kind="key_down",
                payload_fields=frozenset({"key"}),
            ),
            UserInputCapability(
                event_kind="key_up",
                payload_fields=frozenset({"key"}),
            ),
        ),
        produces=(
            ModelInputField(
                name="camera_trajectory",
                phase="step",
                lifecycle="step_input",
            ),
        ),
    )

    compatibility = check_mapping_set_compatibility(
        source_schema=source,
        model_schema=model,
        mapper_schemas=(prompt_mapper, frame_mapper, camera_mapper),
        name="browser-lingbot",
    )

    assert compatibility.can_drive
    assert compatibility.mapper_schema.name == "browser-lingbot"
    assert [field.name for field in compatibility.satisfied_required_model_fields] == [
        "prompt",
        "first_frame",
        "camera_trajectory",
    ]
    assert [capability.event_kind for capability in compatibility.mapper_schema.consumes] == [
        "prompt_set",
        "initial_frame_set",
        "key_down",
        "key_up",
    ]


def test_mapper_schema_combination_deduplicates_shared_capabilities() -> None:
    shared_prompt = UserInputCapability(
        event_kind="prompt_set",
        payload_fields=frozenset({"prompt"}),
    )
    prompt_field = ModelInputField(name="prompt", phase="initial")

    combined = combine_mapper_schemas(
        (
            InputMapperSchema(
                name="prompt-a",
                consumes=(shared_prompt,),
                produces=(prompt_field,),
            ),
            InputMapperSchema(
                name="prompt-b",
                consumes=(shared_prompt,),
                produces=(prompt_field,),
            ),
        )
    )

    assert combined.consumes == (shared_prompt,)
    assert combined.produces == (prompt_field,)


def test_lifecycle_mismatch_does_not_satisfy_model_field() -> None:
    source = UserInputSchema(name="fixed")
    model = ModelInputSchema(
        fields=(
            ModelInputField(
                name="video_dimensions",
                phase="initial",
                lifecycle="runtime_config",
            ),
        )
    )
    mapper = InputMapperSchema(
        produces=(
            ModelInputField(
                name="video_dimensions",
                phase="initial",
                lifecycle="cache_init",
            ),
        )
    )

    compatibility = check_mapping_compatibility(
        source_schema=source,
        model_schema=model,
        mapper_schema=mapper,
    )

    assert not compatibility.can_drive
    assert [
        (field.phase, field.name, field.lifecycle)
        for field in compatibility.missing_required_model_fields
    ] == [("initial", "video_dimensions", "runtime_config")]


def test_metadata_is_queryable_but_not_part_of_compatibility_matching() -> None:
    source = UserInputSchema(
        capabilities=(
            UserInputCapability(
                event_kind="pose_path_set",
                payload_kind="path",
                payload_fields=frozenset({"path"}),
                metadata={"file_format": "npy"},
            ),
        ),
        metadata={"owner": "future-integration"},
    )
    model = ModelInputSchema(
        fields=(
            ModelInputField(
                name="camera_trajectory",
                phase="initial",
                payload_kind="c2w_sequence",
                lifecycle="rollout_binding",
                metadata={"coordinates": "opencv_c2w"},
            ),
        ),
        metadata={"model_family": "future-world-model"},
    )
    mapper = InputMapperSchema(
        consumes=(
            UserInputCapability(
                event_kind="pose_path_set",
                payload_kind="path",
                payload_fields=frozenset({"path"}),
                metadata={"accepted_suffixes": (".npy",)},
            ),
        ),
        produces=(
            ModelInputField(
                name="camera_trajectory",
                phase="initial",
                payload_kind="c2w_sequence",
                lifecycle="rollout_binding",
                metadata={"shape": "[F,4,4]"},
            ),
        ),
        metadata={"mapper_family": "camera-path-loader"},
    )

    compatibility = check_mapping_compatibility(
        source_schema=source,
        model_schema=model,
        mapper_schema=mapper,
    )

    assert compatibility.can_drive
    assert source.metadata["owner"] == "future-integration"
    assert model.fields[0].metadata["coordinates"] == "opencv_c2w"
    assert mapper.produces[0].metadata["shape"] == "[F,4,4]"


def test_sana_wm_like_schema_uses_open_ended_model_inputs() -> None:
    source = UserInputSchema(
        name="sana-wm-cli-like",
        capabilities=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
            UserInputCapability(
                event_kind="initial_frame_set",
                payload_fields=frozenset({"image"}),
            ),
            UserInputCapability(
                event_kind="camera_action_set",
                payload_fields=frozenset({"action"}),
                metadata={"dsl": "<keys>-<frames>"},
            ),
            UserInputCapability(
                event_kind="intrinsics_set",
                payload_fields=frozenset({"intrinsics"}),
                metadata={"optional": True},
            ),
            UserInputCapability(
                event_kind="rollout_parameter_set",
                payload_fields=frozenset({"name", "value"}),
            ),
        ),
    )
    model = ModelInputSchema(
        name="sana-wm-like",
        fields=(
            ModelInputField(name="prompt", phase="initial", lifecycle="cache_init"),
            ModelInputField(
                name="negative_prompt",
                phase="initial",
                required=False,
                lifecycle="cache_init",
            ),
            ModelInputField(
                name="first_frame",
                phase="initial",
                lifecycle="cache_init",
            ),
            ModelInputField(
                name="camera_trajectory_c2w",
                phase="initial",
                payload_kind="c2w_sequence",
                lifecycle="rollout_binding",
                metadata={"shape": "[F,4,4]"},
            ),
            ModelInputField(
                name="camera_intrinsics_vec4",
                phase="initial",
                required=False,
                payload_kind="intrinsics_vec4_sequence",
                lifecycle="rollout_binding",
                metadata={"shape": "[F,4]"},
            ),
            ModelInputField(
                name="stage1_sampling",
                phase="initial",
                lifecycle="rollout_binding",
                metadata={"fields": ("steps", "cfg_scale", "flow_shift", "seed")},
            ),
            ModelInputField(
                name="streaming_chunking",
                phase="initial",
                required=False,
                lifecycle="rollout_binding",
                metadata={"fields": ("num_frame_per_block", "cached_blocks")},
            ),
        ),
        metadata={"model_family": "sana-wm"},
    )
    prompt_mapper = InputMapperSchema(
        produces=(ModelInputField(name="prompt", phase="initial"),),
        consumes=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
        ),
    )
    frame_mapper = InputMapperSchema(
        produces=(ModelInputField(name="first_frame", phase="initial"),),
        consumes=(
            UserInputCapability(
                event_kind="initial_frame_set",
                payload_fields=frozenset({"image"}),
            ),
        ),
    )
    camera_mapper = InputMapperSchema(
        produces=(
            ModelInputField(
                name="camera_trajectory_c2w",
                phase="initial",
                payload_kind="c2w_sequence",
                lifecycle="rollout_binding",
            ),
            ModelInputField(
                name="camera_intrinsics_vec4",
                phase="initial",
                payload_kind="intrinsics_vec4_sequence",
                lifecycle="rollout_binding",
            ),
        ),
        consumes=(
            UserInputCapability(
                event_kind="camera_action_set",
                payload_fields=frozenset({"action"}),
            ),
            UserInputCapability(
                event_kind="intrinsics_set",
                payload_fields=frozenset({"intrinsics"}),
            ),
        ),
    )
    sampling_mapper = InputMapperSchema(
        produces=(
            ModelInputField(
                name="stage1_sampling",
                phase="initial",
                lifecycle="rollout_binding",
            ),
        ),
        consumes=(
            UserInputCapability(
                event_kind="rollout_parameter_set",
                payload_fields=frozenset({"name", "value"}),
            ),
        ),
    )

    compatibility = check_mapping_set_compatibility(
        source_schema=source,
        model_schema=model,
        mapper_schemas=(prompt_mapper, frame_mapper, camera_mapper, sampling_mapper),
    )

    assert compatibility.can_drive
    assert [field.name for field in compatibility.satisfied_required_model_fields] == [
        "prompt",
        "first_frame",
        "camera_trajectory_c2w",
        "stage1_sampling",
    ]
    assert [field.name for field in compatibility.available_optional_model_fields] == [
        "camera_intrinsics_vec4",
    ]


@dataclass(frozen=True)
class PromptMapper:
    schema: InputMapperSchema = InputMapperSchema(
        name="prompt",
        consumes=(
            UserInputCapability(
                event_kind="prompt_set",
                payload_fields=frozenset({"prompt"}),
            ),
        ),
        produces=(ModelInputField(name="prompt", phase="initial"),),
    )

    def build_initial_inputs(self, trace: UserInputTrace) -> ModelInputs:
        prompt = trace.latest("prompt_set")
        return ModelInputs.initial_only(
            {"prompt": prompt.payload["prompt"]} if prompt is not None else {}
        )

    def build_step_inputs(self, window: UserInputWindow) -> ModelInputs:
        del window
        return ModelInputs()


@dataclass(frozen=True)
class InitialFrameMapper:
    schema: InputMapperSchema = InputMapperSchema(
        name="initial-frame",
        consumes=(
            UserInputCapability(
                event_kind="initial_frame_set",
                payload_fields=frozenset({"image"}),
            ),
        ),
        produces=(
            ModelInputField(name="first_frame", phase="initial", required=False),
        ),
    )

    def build_initial_inputs(self, trace: UserInputTrace) -> ModelInputs:
        event = trace.latest("initial_frame_set")
        return ModelInputs.initial_only(
            {"first_frame": event.payload["image"]} if event is not None else {}
        )

    def build_step_inputs(self, window: UserInputWindow) -> ModelInputs:
        del window
        return ModelInputs()


@dataclass(frozen=True)
class KeyboardSteeringMapper:
    schema: InputMapperSchema = InputMapperSchema(
        name="keyboard-steering",
        consumes=(
            UserInputCapability(
                event_kind="key_down",
                payload_fields=frozenset({"key"}),
            ),
            UserInputCapability(
                event_kind="key_up",
                payload_fields=frozenset({"key"}),
            ),
        ),
        produces=(ModelInputField(name="steering", phase="step"),),
    )

    def build_initial_inputs(self, trace: UserInputTrace) -> ModelInputs:
        del trace
        return ModelInputs()

    def build_step_inputs(self, window: UserInputWindow) -> ModelInputs:
        steering = 0.0
        for event in window.events:
            key = event.payload.get("key")
            if event.kind == "key_down" and key == "a":
                steering = 1.0
            elif event.kind == "key_down" and key == "d":
                steering = -1.0
            elif event.kind == "key_up" and key in {"a", "d"}:
                steering = 0.0
        return ModelInputs.step_only({"steering": steering})


@dataclass(frozen=True)
class CameraTrajectoryMapper:
    schema: InputMapperSchema = InputMapperSchema(
        name="camera-trajectory",
        consumes=(
            UserInputCapability(
                event_kind="camera_pose",
                payload_fields=frozenset({"pose"}),
            ),
        ),
        produces=(ModelInputField(name="camera_trajectory", phase="step"),),
    )

    def build_initial_inputs(self, trace: UserInputTrace) -> ModelInputs:
        del trace
        return ModelInputs()

    def build_step_inputs(self, window: UserInputWindow) -> ModelInputs:
        return ModelInputs.step_only(
            {
                "camera_trajectory": tuple(
                    event.payload["pose"]
                    for event in window.events_of_kind("camera_pose")
                )
            }
        )


def test_fake_prompt_and_initial_frame_mappers_build_initial_model_inputs() -> None:
    trace = UserInputTrace.from_events(
        [
            UserInputEvent(
                timestamp_s=0.0,
                kind="prompt_set",
                payload={"prompt": "a road"},
            ),
            UserInputEvent(
                timestamp_s=0.0,
                kind="initial_frame_set",
                payload={"image": b"frame"},
            ),
        ]
    )

    assert PromptMapper().build_initial_inputs(trace).initial == {
        "prompt": "a road"
    }
    assert InitialFrameMapper().build_initial_inputs(trace).initial == {
        "first_frame": b"frame"
    }


def test_static_input_mapper_supports_fixed_model_inputs() -> None:
    inputs = ModelInputs(
        initial={"prompt": "fixed prompt"},
        step={"camera_trajectory": ("pose-a", "pose-b")},
    )
    mapper = StaticInputMapper.from_inputs(inputs=inputs, name="fixed")
    source = UserInputSchema(name="no-live-controls")
    model = ModelInputSchema(
        fields=(
            ModelInputField(name="prompt", phase="initial"),
            ModelInputField(name="camera_trajectory", phase="step"),
        )
    )

    compatibility = check_mapping_compatibility(
        source_schema=source,
        model_schema=model,
        mapper_schema=mapper.schema,
    )

    assert compatibility.can_drive
    assert mapper.build_initial_inputs(UserInputTrace()).initial == {
        "prompt": "fixed prompt"
    }
    assert mapper.build_step_inputs(UserInputWindow(start_s=0.0, end_s=1.0)).step == {
        "camera_trajectory": ("pose-a", "pose-b")
    }


def test_fake_keyboard_mapper_builds_step_model_inputs() -> None:
    trace = UserInputTrace.from_events(
        [
            UserInputEvent(timestamp_s=0.1, kind="key_down", payload={"key": "a"}),
            UserInputEvent(timestamp_s=0.2, kind="key_up", payload={"key": "a"}),
            UserInputEvent(timestamp_s=0.3, kind="key_down", payload={"key": "d"}),
        ]
    )

    inputs = KeyboardSteeringMapper().build_step_inputs(
        trace.window(start_s=0.0, end_s=0.4)
    )

    assert inputs.step == {"steering": -1.0}


def test_fake_camera_trajectory_mapper_builds_step_model_inputs() -> None:
    trace = UserInputTrace.from_events(
        [
            UserInputEvent(
                timestamp_s=0.1,
                kind="camera_pose",
                payload={"pose": "pose-a"},
            ),
            UserInputEvent(
                timestamp_s=0.2,
                kind="camera_pose",
                payload={"pose": "pose-b"},
            ),
        ]
    )

    inputs = CameraTrajectoryMapper().build_step_inputs(
        trace.window(start_s=0.0, end_s=0.3)
    )

    assert inputs.step == {"camera_trajectory": ("pose-a", "pose-b")}
