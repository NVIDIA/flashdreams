# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import fields
from typing import Any, cast

import pytest

from flashdreams.runtime import (
    CanonicalInputs,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InMemoryMetricsRecorder,
    InputField,
    NullOutputTarget,
    OutputArtifact,
    RuntimeMetricSample,
    StepRequest,
    StepResult,
    TimeWindow,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)

pytestmark = pytest.mark.ci_cpu


def test_inference_config_keeps_runtime_settings_separate() -> None:
    denied_app_fields = {"prompt", "output_dir", "browser_settings"}
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
    assert denied_app_fields.isdisjoint(field.name for field in fields(InferenceConfig))
    with pytest.raises(TypeError):
        cast(Any, config.runtime_options)["chunk_size"] = 4


def test_inference_config_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        InferenceConfig(model_id=" ")


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: InputField(name=" "), "InputField.name"),
        (lambda: TimeWindow(start_s=1.0, end_s=0.0), "end_s"),
        (lambda: TimeWindow(start_s=-1.0, end_s=0.0), "non-negative"),
        (lambda: TimeWindow(start_s=0.0, end_s=float("nan")), "finite"),
        (
            lambda: UserInputEvent(timestamp_s=-1.0, event_type="keydown"),
            "timestamp_s",
        ),
        (lambda: UserInputEvent(timestamp_s=0.0, event_type=" "), "event_type"),
        (lambda: StepRequest(step_index=-1), "step_index"),
        (lambda: StepResult(step_index=-1), "step_index"),
        (lambda: StepResult(step_index=0, frame_count=-1), "frame_count"),
        (lambda: RuntimeMetricSample(name=" ", value=1.0), "name"),
        (lambda: RuntimeMetricSample(name="sample", value=float("nan")), "finite"),
        (lambda: OutputArtifact(kind=" ", uri="artifact://demo"), "kind"),
        (lambda: OutputArtifact(kind="mp4", uri=" "), "uri"),
    ],
)
def test_runtime_envelopes_reject_invalid_values(factory: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cast(Any, factory)()


def test_runtime_metric_sample_rejects_bool_values() -> None:
    with pytest.raises(TypeError, match="numeric"):
        RuntimeMetricSample(name="sample", value=True)


def test_schema_validates_global_conditioning_and_step_payloads() -> None:
    schema = InferenceInputSchema(
        global_conditioning_fields=(
            InputField(name="prompt"),
            InputField(name="global_conditioning_frame"),
        ),
        step_fields=(InputField(name="camera_poses"),),
    )
    inputs = InferenceInput(
        global_conditioning={"prompt": "drive", "global_conditioning_frame": object()}
    )

    schema.require_global_conditioning(inputs)
    assert schema.missing_step(inputs) == ("camera_poses",)

    with pytest.raises(ValueError, match="camera_poses"):
        schema.require_step(inputs)


def test_user_inputs_filter_timestamped_event_windows() -> None:
    inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.1,
                event_type="keyboard.keydown",
                payload={"key": "w"},
            ),
            UserInputEvent(
                timestamp_s=0.4,
                event_type="keyboard.keyup",
                payload={"key": "w"},
            ),
            UserInputEvent(timestamp_s=0.8, event_type="reset"),
        )
    )

    windowed = inputs.window(TimeWindow(start_s=0.25, end_s=0.75))

    assert [event.event_type for event in windowed.events] == ["keyboard.keyup"]


def test_user_inputs_require_sorted_events() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        UserInputs(
            events=(
                UserInputEvent(timestamp_s=1.0, event_type="late"),
                UserInputEvent(timestamp_s=0.5, event_type="early"),
            )
        )


def test_user_input_schema_declares_event_capabilities() -> None:
    schema = UserInputSchema(
        event_types=frozenset({"keyboard.keydown", "keyboard.keyup", "reset"})
    )

    assert schema.supports_event_types(["keyboard.keydown", "reset"])
    assert not schema.supports_event_types(["prompt.update"])


def test_user_input_schema_validates_required_snapshot_fields() -> None:
    schema = UserInputSchema(
        snapshot_fields=(
            InputField(name="pressed_keys"),
            InputField(name="prompt", required=False),
        )
    )
    inputs = UserInputs(snapshot={"pressed_keys": frozenset({"w"})})

    schema.require_snapshot(inputs)
    assert schema.missing_snapshot(UserInputs()) == ("pressed_keys",)

    with pytest.raises(ValueError, match="pressed_keys"):
        schema.require_snapshot(UserInputs())


def test_identity_input_mapping_leaves_inference_input_unchanged() -> None:
    mapping = IdentityInputMapping()
    inference_input = InferenceInput(
        global_conditioning={"prompt": "fixed"}, step={"hdmap": object()}
    )
    request = StepRequest(step_index=0)

    assert (
        mapping.map_global_conditioning_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=inference_input,
        )
        is inference_input
    )
    assert (
        mapping.map_step_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=inference_input,
            request=request,
        )
        is inference_input
    )


def test_null_output_target_counts_and_optionally_stores_results() -> None:
    target = NullOutputTarget(store_results=True)
    result = StepResult(step_index=0, output=b"frame")

    assert target.closed
    with pytest.raises(RuntimeError, match="closed output target"):
        target.write(result)

    target.open()
    assert not target.closed
    target.write(result)
    artifacts = target.close()

    assert target.closed
    assert artifacts == ()
    assert target.output_count == 1
    assert target.results == [result]
    with pytest.raises(RuntimeError, match="closed output target"):
        target.write(StepResult(step_index=1))


def test_null_output_target_open_resets_per_run_state() -> None:
    target = NullOutputTarget(store_results=True)

    target.open()
    target.write(StepResult(step_index=0, output=b"first"))
    target.close()
    target.open()

    assert target.output_count == 0
    assert target.results == []
    target.write(StepResult(step_index=0, output=b"second"))
    assert target.output_count == 1
    assert target.results == [StepResult(step_index=0, output=b"second")]


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


def test_timing_metric_samples_must_use_seconds() -> None:
    with pytest.raises(ValueError, match="unit='s'"):
        RuntimeMetricSample(
            name="model_step",
            value=12.5,
            unit="ms",
            category="timing",
        )
