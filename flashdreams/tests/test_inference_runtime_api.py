# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from flashdreams.runtime import (
    IdentityInputMapping,
    InMemoryMetricsRecorder,
    InferenceConfig,
    InputField,
    ModelInputs,
    ModelInputSchema,
    NullOutputTarget,
    StepRequest,
    StepResult,
    TimeWindow,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)

pytestmark = pytest.mark.ci_cpu


def test_inference_config_keeps_runtime_settings_separate() -> None:
    config = InferenceConfig(
        model_id="lingbot-world",
        preset_id="fast-taehv",
        backend="local",
        precision="bf16",
        compile=False,
        runtime_options={"chunk_size": 3},
    )

    assert config.model_id == "lingbot-world"
    assert config.preset_id == "fast-taehv"
    assert config.runtime_options["chunk_size"] == 3
    assert not hasattr(config, "prompt")
    assert not hasattr(config, "output_dir")


def test_model_input_schema_validates_initial_and_step_payloads() -> None:
    schema = ModelInputSchema(
        initial_fields=(
            InputField(name="prompt"),
            InputField(name="first_frame"),
        ),
        step_fields=(InputField(name="camera_poses"),),
    )
    inputs = ModelInputs(initial={"prompt": "drive", "first_frame": object()})

    schema.require_initial(inputs)
    assert schema.missing_step(inputs) == ("camera_poses",)

    with pytest.raises(ValueError, match="camera_poses"):
        schema.require_step(inputs)


def test_user_inputs_filter_timestamped_event_windows() -> None:
    inputs = UserInputs(
        events=(
            UserInputEvent(timestamp_s=0.1, kind="keyboard.keydown", payload={"key": "w"}),
            UserInputEvent(timestamp_s=0.4, kind="keyboard.keyup", payload={"key": "w"}),
            UserInputEvent(timestamp_s=0.8, kind="reset"),
        )
    )

    windowed = inputs.window(TimeWindow(start_s=0.25, end_s=0.75))

    assert [event.kind for event in windowed.events] == ["keyboard.keyup"]


def test_user_input_schema_declares_event_capabilities() -> None:
    schema = UserInputSchema(
        event_kinds=frozenset({"keyboard.keydown", "keyboard.keyup", "reset"})
    )

    assert schema.supports_event_kinds(["keyboard.keydown", "reset"])
    assert not schema.supports_event_kinds(["prompt.update"])


def test_identity_input_mapping_leaves_model_inputs_unchanged() -> None:
    mapping = IdentityInputMapping()
    model_inputs = ModelInputs(initial={"prompt": "fixed"}, step={"hdmap": object()})
    request = StepRequest(step_index=0)

    assert (
        mapping.map_initial_inputs(
            user_inputs=UserInputs(),
            model_inputs=model_inputs,
        )
        is model_inputs
    )
    assert (
        mapping.map_step_inputs(
            user_inputs=UserInputs(),
            model_inputs=model_inputs,
            request=request,
        )
        is model_inputs
    )


def test_null_output_target_counts_and_optionally_stores_results() -> None:
    target = NullOutputTarget(store_results=True)
    result = StepResult(step_index=0, output=b"frame")

    target.open()
    target.write(result)
    artifacts = target.close()

    assert artifacts == ()
    assert target.output_count == 1
    assert target.results == [result]
    with pytest.raises(RuntimeError, match="closed output target"):
        target.write(StepResult(step_index=1))


def test_in_memory_metrics_recorder_uses_seconds_for_timing() -> None:
    recorder = InMemoryMetricsRecorder()

    recorder.record_timing("model_step", 0.125, step_index=2)

    assert len(recorder.samples) == 1
    sample = recorder.samples[0]
    assert sample.name == "model_step"
    assert sample.value == pytest.approx(0.125)
    assert sample.unit == "s"
    assert sample.category == "timing"
    assert sample.step_index == 2
