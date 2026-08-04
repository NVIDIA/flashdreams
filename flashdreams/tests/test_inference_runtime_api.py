# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import fields
from typing import Any, cast

import pytest

from flashdreams.runtime import (
    CanonicalInputs,
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceRuntime,
    InferenceSession,
    InMemoryMetricsRecorder,
    InputCanonicalizer,
    InputField,
    InputMapping,
    MetricsRecorder,
    ModelAdapter,
    NullOutputTarget,
    OutputArtifact,
    OutputTarget,
    RuntimeMetricSample,
    StepRequest,
    StepResult,
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)

pytestmark = pytest.mark.ci_cpu


_SESSION_HORIZON_S = 3600.0

_KEYBOARD_SOURCE = UserInputSchema(
    capabilities=(
        UserInputCapability(
            event_type="keyboard.keydown", payload_fields=frozenset({"key"})
        ),
    )
)
_KEYBOARD_CANONICALIZER = InputCanonicalizer()


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


def test_inference_input_schema_validates_initial_and_step_payloads() -> None:
    schema = InferenceInputSchema(
        global_fields=(
            InputField(name="prompt"),
            InputField(name="global_conditioning_frame"),
        ),
        step_fields=(InputField(name="camera_poses"),),
    )
    inputs = InferenceInput(
        global_conditioning={"prompt": "drive", "global_conditioning_frame": object()}
    )

    schema.require_global(inputs)
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
        mapping.map_global_inputs(
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


def test_runtime_api_components_compose_for_sequential_session() -> None:
    adapter = _FakeAdapter()
    config = InferenceConfig(model_id="fake-model")
    user_inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.25,
                event_type="keyboard.keydown",
                payload={"key": "w"},
            ),
        )
    )
    inference_input = InferenceInput(global_conditioning={"prompt": "drive forward"})
    output = NullOutputTarget(store_results=True)
    metrics = InMemoryMetricsRecorder()

    adapter.validate_config(config)
    mapping = adapter.default_input_mapping()
    assert mapping is not None
    _drive_two_step_session(
        adapter=adapter,
        config=config,
        mapping=mapping,
        canonicalizer=_KEYBOARD_CANONICALIZER,
        source_schema=_KEYBOARD_SOURCE,
        user_inputs=user_inputs,
        inference_input=inference_input,
        output=output,
        metrics=metrics,
    )

    assert output.output_count == 2
    assert [result.output for result in output.results] == ["chunk-0", "chunk-1"]
    assert [result.frame_count for result in output.results] == [3, 3]
    assert output.results[0].output_window == TimeWindow(start_s=0.0, end_s=0.5)
    assert [sample.step_index for sample in metrics.samples] == [0, 1]
    assert metrics.closed


def test_reference_loop_validates_mapping_before_runtime_creation() -> None:
    mapping = _OrderCheckingMapping()
    adapter = _OrderCheckingAdapter(mapping=mapping)

    _drive_two_step_session(
        adapter=adapter,
        config=InferenceConfig(model_id="fake-model"),
        mapping=mapping,
        canonicalizer=_KEYBOARD_CANONICALIZER,
        source_schema=_KEYBOARD_SOURCE,
        user_inputs=UserInputs(),
        inference_input=InferenceInput(global_conditioning={"prompt": "drive forward"}),
        output=NullOutputTarget(),
        metrics=InMemoryMetricsRecorder(),
    )

    assert mapping.validated
    assert adapter.created_runtime_after_validate


def test_reference_loop_closes_runtime_when_session_start_fails() -> None:
    adapter = _FailingStartAdapter()
    output = NullOutputTarget()
    metrics = InMemoryMetricsRecorder()

    with pytest.raises(RuntimeError, match="start failed"):
        _drive_two_step_session(
            adapter=adapter,
            config=InferenceConfig(model_id="fake-model"),
            mapping=IdentityInputMapping(),
            canonicalizer=_KEYBOARD_CANONICALIZER,
            source_schema=_KEYBOARD_SOURCE,
            user_inputs=UserInputs(),
            inference_input=InferenceInput(
                global_conditioning={"prompt": "drive forward"}
            ),
            output=output,
            metrics=metrics,
        )

    assert adapter.runtime is not None
    assert adapter.runtime.closed
    assert output.closed
    assert metrics.closed


def _drive_two_step_session(
    *,
    adapter: ModelAdapter,
    config: InferenceConfig,
    mapping: InputMapping,
    canonicalizer: InputCanonicalizer,
    source_schema: UserInputSchema,
    user_inputs: UserInputs,
    inference_input: InferenceInput,
    output: OutputTarget,
    metrics: MetricsRecorder,
) -> None:
    mapping.validate(
        canonical_schema=adapter.canonical_input_schema,
        inference_input_schema=adapter.inference_input_schema,
    )
    initial_inputs = mapping.map_global_inputs(
        canonical_inputs=canonicalizer.canonicalize(
            user_inputs,
            window=TimeWindow(start_s=0.0, end_s=_SESSION_HORIZON_S),
            source_schema=source_schema,
        ),
        inference_input=inference_input,
    )
    runtime = adapter.create_runtime(config)
    session: InferenceSession | None = None
    output_opened = False
    try:
        session = runtime.start_session(initial_inputs)
        output.open()
        output_opened = True
        while (request := session.next_step_request()) is not None:
            step_inputs = mapping.map_step_inputs(
                canonical_inputs=canonicalizer.canonicalize(
                    user_inputs,
                    window=request.user_input_window
                    or TimeWindow(start_s=0.0, end_s=_SESSION_HORIZON_S),
                    source_schema=source_schema,
                ),
                # The global slot stays empty in steady state. A mapping that
                # sees ``canonical_inputs.has_global_change`` fills it via
                # ``with_global_update`` to request a mid-rollout swap.
                inference_input=InferenceInput(
                    step={"chunk_index": request.step_index},
                ),
                request=request,
            )
            result = session.step(step_inputs)
            output.write(result)
            metrics.record_timing(
                "model_step",
                float(result.metrics["model_step_s"]),
                step_index=result.step_index,
            )
    finally:
        if output_opened:
            output.close()
        if session is not None:
            session.close()
        runtime.close()
        metrics.close()


class _FakeAdapter:
    model_id = "fake-model"
    inference_input_schema = InferenceInputSchema(
        global_fields=(InputField(name="prompt"),),
        step_fields=(InputField(name="chunk_index"),),
    )
    canonical_input_schema = CanonicalInputSchema()

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return _FakeRuntime(inference_input_schema=self.inference_input_schema)


class _FakeRuntime:
    def __init__(self, *, inference_input_schema: InferenceInputSchema) -> None:
        self._inference_input_schema = inference_input_schema
        self.closed = False

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        self._inference_input_schema.require_global(inputs)
        return _FakeSession(inference_input_schema=self._inference_input_schema)

    def close(self) -> None:
        self.closed = True


class _FailingRuntime(_FakeRuntime):
    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        del inputs
        raise RuntimeError("start failed")


class _FakeSession:
    def __init__(self, *, inference_input_schema: InferenceInputSchema) -> None:
        self._inference_input_schema = inference_input_schema
        self.step_index = 0
        self.closed = False

    def next_step_request(self) -> StepRequest | None:
        if self.step_index >= 2:
            return None
        return StepRequest(
            step_index=self.step_index,
            inference_input_schema=self._inference_input_schema,
            user_input_window=TimeWindow(
                start_s=0.5 * self.step_index,
                end_s=0.5 * (self.step_index + 1),
            ),
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        self._inference_input_schema.require_step(inputs)
        result = StepResult(
            step_index=self.step_index,
            output=f"chunk-{self.step_index}",
            frame_count=3,
            output_window=TimeWindow(
                start_s=0.5 * self.step_index,
                end_s=0.5 * (self.step_index + 1),
            ),
            metrics={"model_step_s": 0.01},
        )
        self.step_index += 1
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self.step_index = 0

    def close(self) -> None:
        self.closed = True


class _OrderCheckingMapping(IdentityInputMapping):
    def __init__(self) -> None:
        self.validated = False

    def validate(
        self,
        *,
        canonical_schema: CanonicalInputSchema | None = None,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        super().validate(
            canonical_schema=canonical_schema,
            inference_input_schema=inference_input_schema,
        )
        self.validated = True


class _OrderCheckingAdapter(_FakeAdapter):
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self, *, mapping: _OrderCheckingMapping) -> None:
        self._mapping = mapping
        self.created_runtime_after_validate = False

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        self.created_runtime_after_validate = self._mapping.validated
        return _FakeRuntime(inference_input_schema=self.inference_input_schema)


class _FailingStartAdapter(_FakeAdapter):
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self) -> None:
        self.runtime: _FailingRuntime | None = None

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        self.runtime = _FailingRuntime(
            inference_input_schema=self.inference_input_schema
        )
        return self.runtime
