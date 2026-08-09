# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

import flashdreams.runtime.runner as runner_module
from flashdreams.runtime import (
    DRIVER_COMMAND,
    CanonicalInputs,
    CanonicalInputSchema,
    CanonicalModality,
    DeviceConverterSchema,
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
    InputMappingSchema,
    MetricsSnapshot,
    NullOutputTarget,
    OutputArtifact,
    RuntimeMetricSample,
    StepRequest,
    StepResult,
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
    run_inference_session,
)

pytestmark = pytest.mark.ci_cpu


def test_run_inference_session_completes_two_step_run() -> None:
    adapter = _FakeAdapter()
    output = NullOutputTarget(store_results=True)
    metrics = InMemoryMetricsRecorder()

    artifacts = run_inference_session(
        adapter=adapter,
        config=InferenceConfig(model_id="fake-model"),
        mapping=_ChunkIndexMapping(),
        canonicalizer=InputCanonicalizer(),
        source_schema=UserInputSchema(),
        user_inputs=UserInputs(),
        initial_inputs=InferenceInput(global_conditioning={"prompt": "drive forward"}),
        output=output,
        metrics=metrics,
    )

    assert artifacts == ()
    assert output.closed
    assert output.output_count == 2
    assert [result.output for result in output.results] == ["chunk-0", "chunk-1"]
    assert [result.frame_count for result in output.results] == [3, 3]
    assert output.results[0].output_window == TimeWindow(start_s=0.0, end_s=0.5)
    assert adapter.runtime is not None
    assert adapter.runtime.closed
    assert adapter.runtime.session is not None
    assert adapter.runtime.session.closed
    assert [sample.name for sample in metrics.samples] == ["model_step", "model_step"]
    assert [sample.step_index for sample in metrics.samples] == [0, 1]
    assert metrics.closed


def test_run_inference_session_delegates_to_shared_batch_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Mapping[str, object]] = []
    artifact = OutputArtifact(kind="test/artifact", uri="memory://artifact")

    def _fake_helper(**kwargs: object) -> tuple[OutputArtifact, ...]:
        calls.append(kwargs)
        return (artifact,)

    monkeypatch.setattr(
        runner_module,
        "_run_inference_session_with_shared_batch",
        _fake_helper,
    )
    adapter = _FakeAdapter()
    config = InferenceConfig(model_id="fake-model")
    mapping = _ChunkIndexMapping()
    canonicalizer = InputCanonicalizer()
    source_schema = UserInputSchema()
    user_inputs = UserInputs()
    initial_inputs = InferenceInput(global_conditioning={"prompt": "drive forward"})
    output = NullOutputTarget()
    metrics = InMemoryMetricsRecorder()

    artifacts = runner_module.run_inference_session(
        adapter=adapter,
        config=config,
        mapping=mapping,
        canonicalizer=canonicalizer,
        source_schema=source_schema,
        user_inputs=user_inputs,
        initial_inputs=initial_inputs,
        output=output,
        metrics=metrics,
    )

    assert artifacts == (artifact,)
    assert len(calls) == 1
    call = calls[0]
    assert call["adapter"] is adapter
    assert call["config"] is config
    assert call["mapping"] is mapping
    assert call["canonicalizer"] is canonicalizer
    assert call["source_schema"] is source_schema
    assert call["user_inputs"] is user_inputs
    assert call["initial_inputs"] is initial_inputs
    assert call["output"] is output
    assert call["metrics"] is metrics


def test_runner_preserves_initial_step_inputs_for_identity_mapping() -> None:
    adapter = _FakeAdapter()

    run_inference_session(
        adapter=adapter,
        config=InferenceConfig(model_id="fake-model"),
        mapping=IdentityInputMapping(),
        canonicalizer=InputCanonicalizer(),
        source_schema=UserInputSchema(),
        user_inputs=UserInputs(),
        initial_inputs=InferenceInput(
            global_conditioning={"prompt": "drive forward"},
            step={"chunk_index": 42},
        ),
        output=NullOutputTarget(),
        metrics=InMemoryMetricsRecorder(),
    )

    assert adapter.runtime is not None
    assert adapter.runtime.session is not None
    assert [dict(inputs.step) for inputs in adapter.runtime.session.step_inputs] == [
        {"chunk_index": 42},
        {"chunk_index": 42},
    ]
    assert [
        dict(inputs.global_conditioning)
        for inputs in adapter.runtime.session.step_inputs
    ] == [{}, {}]


def test_runner_validates_mapping_before_runtime_creation() -> None:
    mapping = _OrderCheckingMapping()
    adapter = _OrderCheckingAdapter(mapping=mapping)

    run_inference_session(
        adapter=adapter,
        config=InferenceConfig(model_id="fake-model"),
        mapping=mapping,
        canonicalizer=InputCanonicalizer(),
        source_schema=UserInputSchema(),
        user_inputs=UserInputs(),
        initial_inputs=InferenceInput(global_conditioning={"prompt": "drive forward"}),
        output=NullOutputTarget(),
        metrics=InMemoryMetricsRecorder(),
    )

    assert mapping.validated
    assert adapter.created_runtime_after_validate


def test_runner_closes_runtime_when_session_start_fails() -> None:
    adapter = _FailingStartAdapter()
    output = _RecordingOutputTarget()
    metrics = InMemoryMetricsRecorder()

    with pytest.raises(RuntimeError, match="start failed"):
        run_inference_session(
            adapter=adapter,
            config=InferenceConfig(model_id="fake-model"),
            mapping=_ChunkIndexMapping(),
            canonicalizer=InputCanonicalizer(),
            source_schema=UserInputSchema(),
            user_inputs=UserInputs(),
            initial_inputs=InferenceInput(
                global_conditioning={"prompt": "drive forward"}
            ),
            output=output,
            metrics=metrics,
        )

    assert adapter.runtime is not None
    assert adapter.runtime.closed
    assert output.events == ()
    assert metrics.closed


def test_runner_does_not_canonicalize_global_conditioning() -> None:
    mapping = _CanonicalRecordingMapping()

    run_inference_session(
        adapter=_FakeAdapter(),
        config=InferenceConfig(model_id="fake-model"),
        mapping=mapping,
        canonicalizer=InputCanonicalizer([_CountingDeviceConverter()]),
        source_schema=UserInputSchema(
            capabilities=(
                UserInputCapability(
                    event_type="stateful_event",
                    payload_fields=frozenset(),
                ),
            )
        ),
        user_inputs=UserInputs(
            events=(UserInputEvent(timestamp_s=0.75, event_type="stateful_event"),)
        ),
        initial_inputs=InferenceInput(global_conditioning={"prompt": "drive forward"}),
        output=NullOutputTarget(),
        metrics=InMemoryMetricsRecorder(),
    )

    assert mapping.global_canonical_values == {}
    assert mapping.step_canonical_values == (
        {"stateful_counter": {"count": 0}},
        {"stateful_counter": {"count": 1}},
    )


def test_runner_closes_opened_resources_after_output_failure() -> None:
    events: list[str] = []
    adapter = _RecordingAdapter(events=events)
    output = _FailingWriteOutputTarget(events=events)
    metrics = _RecordingMetricsRecorder(events=events)

    with pytest.raises(RuntimeError, match="write failed"):
        run_inference_session(
            adapter=adapter,
            config=InferenceConfig(model_id="fake-model"),
            mapping=_ChunkIndexMapping(),
            canonicalizer=InputCanonicalizer(),
            source_schema=UserInputSchema(),
            user_inputs=UserInputs(),
            initial_inputs=InferenceInput(
                global_conditioning={"prompt": "drive forward"}
            ),
            output=output,
            metrics=metrics,
        )

    assert events == [
        "runtime.start_session",
        "output.open",
        "session.step:0",
        "output.write:0",
        "output.close",
        "session.close",
        "runtime.close",
        "metrics.close",
    ]


def test_runner_attempts_later_cleanup_when_output_close_fails() -> None:
    events: list[str] = []
    adapter = _RecordingAdapter(events=events)
    output = _FailingCloseOutputTarget(events=events)
    metrics = _RecordingMetricsRecorder(events=events)

    with pytest.raises(RuntimeError, match="close failed"):
        run_inference_session(
            adapter=adapter,
            config=InferenceConfig(model_id="fake-model"),
            mapping=_ChunkIndexMapping(),
            canonicalizer=InputCanonicalizer(),
            source_schema=UserInputSchema(),
            user_inputs=UserInputs(),
            initial_inputs=InferenceInput(
                global_conditioning={"prompt": "drive forward"}
            ),
            output=output,
            metrics=metrics,
        )

    assert events == [
        "runtime.start_session",
        "output.open",
        "session.step:0",
        "output.write:0",
        "session.step:1",
        "output.write:1",
        "output.close",
        "session.close",
        "runtime.close",
        "metrics.close",
    ]


def test_runner_checks_declared_mapping_compatibility_before_runtime_creation() -> None:
    adapter = _DrivingAdapter()
    mapping = _UnfeedableDriverCommandMapping()
    metrics = InMemoryMetricsRecorder()

    with pytest.raises(ValueError, match="cannot drive this model"):
        run_inference_session(
            adapter=adapter,
            config=InferenceConfig(model_id="fake-model"),
            mapping=mapping,
            canonicalizer=InputCanonicalizer(),
            source_schema=UserInputSchema(),
            user_inputs=UserInputs(),
            initial_inputs=InferenceInput(
                global_conditioning={"prompt": "drive forward"}
            ),
            output=NullOutputTarget(),
            metrics=metrics,
        )

    assert not adapter.create_runtime_called
    assert not mapping.validated
    assert metrics.closed


class _ChunkIndexMapping:
    mapping_schema = InputMappingSchema(
        name="chunk-index",
        produces_global_conditioning=(InputField(name="prompt"),),
        produces_step=(InputField(name="chunk_index"),),
    )

    def __init__(self) -> None:
        self.validated = False

    def validate(
        self,
        *,
        canonical_schema: CanonicalInputSchema | None = None,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        del canonical_schema, inference_input_schema
        self.validated = True

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
        del canonical_inputs
        return InferenceInput(
            global_conditioning=inference_input.global_conditioning,
            step={"chunk_index": request.step_index},
            metadata=inference_input.metadata,
        )


class _UnfeedableDriverCommandMapping(_ChunkIndexMapping):
    mapping_schema = InputMappingSchema(
        name="driver-command",
        consumes=(DRIVER_COMMAND,),
        produces_global_conditioning=(InputField(name="prompt"),),
        produces_step=(InputField(name="steering"),),
    )

    def map_step_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
        request: StepRequest,
    ) -> InferenceInput:
        del request
        return InferenceInput(
            global_conditioning=inference_input.global_conditioning,
            step={
                "steering": canonical_inputs.values[DRIVER_COMMAND.name]["steer"],
            },
            metadata=inference_input.metadata,
        )


class _CanonicalRecordingMapping(_ChunkIndexMapping):
    def __init__(self) -> None:
        super().__init__()
        self.global_canonical_values: Mapping[str, Any] | None = None
        self._step_canonical_values: list[Mapping[str, Any]] = []

    @property
    def step_canonical_values(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._step_canonical_values)

    def map_global_conditioning_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
    ) -> InferenceInput:
        self.global_canonical_values = canonical_inputs.values
        return super().map_global_conditioning_inputs(
            canonical_inputs=canonical_inputs,
            inference_input=inference_input,
        )

    def map_step_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
        request: StepRequest,
    ) -> InferenceInput:
        self._step_canonical_values.append(canonical_inputs.values)
        return super().map_step_inputs(
            canonical_inputs=canonical_inputs,
            inference_input=inference_input,
            request=request,
        )


_STATEFUL_COUNTER = CanonicalModality(
    name="stateful_counter",
    payload_fields=frozenset({"count"}),
)


class _CountingDeviceConverter:
    schema = DeviceConverterSchema(
        name="stateful-counter",
        produces=_STATEFUL_COUNTER,
        consumes=(UserInputCapability(event_type="stateful_event"),),
    )

    def __init__(self) -> None:
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, Any] | None:
        del window
        self.count += len(user_inputs.events)
        return _STATEFUL_COUNTER.value({"count": self.count})


class _FakeAdapter:
    model_id = "fake-model"
    inference_input_schema = InferenceInputSchema(
        global_conditioning_fields=(InputField(name="prompt"),),
        step_fields=(InputField(name="chunk_index"),),
    )
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self) -> None:
        self.runtime: _FakeRuntime | None = None
        self.create_runtime_called = False

    def default_input_mapping(self) -> InputMapping:
        return _ChunkIndexMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        self.create_runtime_called = True
        self.runtime = _FakeRuntime(inference_input_schema=self.inference_input_schema)
        return self.runtime


class _DrivingAdapter(_FakeAdapter):
    inference_input_schema = InferenceInputSchema(
        global_conditioning_fields=(InputField(name="prompt"),),
        step_fields=(InputField(name="steering"),),
    )


class _OrderCheckingMapping(_ChunkIndexMapping):
    pass


class _OrderCheckingAdapter(_FakeAdapter):
    def __init__(self, *, mapping: _OrderCheckingMapping) -> None:
        super().__init__()
        self._mapping = mapping
        self.created_runtime_after_validate = False

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        self.created_runtime_after_validate = self._mapping.validated
        self.create_runtime_called = True
        self.runtime = _FakeRuntime(inference_input_schema=self.inference_input_schema)
        return self.runtime


class _FailingStartAdapter(_FakeAdapter):
    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        self.create_runtime_called = True
        self.runtime = _FailingRuntime(
            inference_input_schema=self.inference_input_schema
        )
        return self.runtime


class _RecordingAdapter(_FakeAdapter):
    def __init__(self, *, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        self.create_runtime_called = True
        self.runtime = _RecordingRuntime(
            inference_input_schema=self.inference_input_schema,
            events=self._events,
        )
        return self.runtime


class _FakeRuntime:
    def __init__(self, *, inference_input_schema: InferenceInputSchema) -> None:
        self._inference_input_schema = inference_input_schema
        self.session: _FakeSession | None = None
        self.closed = False

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        self._inference_input_schema.require_global_conditioning(inputs)
        self.session = _FakeSession(inference_input_schema=self._inference_input_schema)
        return self.session

    def close(self) -> None:
        self.closed = True


class _FailingRuntime(_FakeRuntime):
    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        del inputs
        raise RuntimeError("start failed")


class _RecordingRuntime(_FakeRuntime):
    def __init__(
        self,
        *,
        inference_input_schema: InferenceInputSchema,
        events: list[str],
    ) -> None:
        super().__init__(inference_input_schema=inference_input_schema)
        self._events = events

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        self._events.append("runtime.start_session")
        self._inference_input_schema.require_global_conditioning(inputs)
        self.session = _RecordingSession(
            inference_input_schema=self._inference_input_schema,
            events=self._events,
        )
        return self.session

    def close(self) -> None:
        self._events.append("runtime.close")
        super().close()


class _FakeSession:
    def __init__(self, *, inference_input_schema: InferenceInputSchema) -> None:
        self._inference_input_schema = inference_input_schema
        self.step_index = 0
        self.step_inputs: list[InferenceInput] = []
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
        self.step_inputs.append(inputs)
        result = StepResult(
            step_index=self.step_index,
            output=f"chunk-{self.step_index}",
            frame_count=3,
            output_window=TimeWindow(
                start_s=0.5 * self.step_index,
                end_s=0.5 * (self.step_index + 1),
            ),
            metrics={"model_step_s": 0.01, "frames": 3},
        )
        self.step_index += 1
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self.step_index = 0

    def close(self) -> None:
        self.closed = True


class _RecordingSession(_FakeSession):
    def __init__(
        self,
        *,
        inference_input_schema: InferenceInputSchema,
        events: list[str],
    ) -> None:
        super().__init__(inference_input_schema=inference_input_schema)
        self._events = events

    def step(self, inputs: InferenceInput) -> StepResult:
        self._events.append(f"session.step:{self.step_index}")
        return super().step(inputs)

    def close(self) -> None:
        self._events.append("session.close")
        super().close()


class _RecordingOutputTarget:
    def __init__(self, *, events: list[str] | None = None) -> None:
        self._events = events
        self._opened = False

    @property
    def events(self) -> tuple[str, ...]:
        return () if self._events is None else tuple(self._events)

    def open(self) -> None:
        self._opened = True
        if self._events is not None:
            self._events.append("output.open")

    def write(self, result: StepResult) -> None:
        if not self._opened:
            raise RuntimeError("Cannot write to a closed output target.")
        if self._events is not None:
            self._events.append(f"output.write:{result.step_index}")

    def close(self) -> Sequence[OutputArtifact]:
        self._opened = False
        if self._events is not None:
            self._events.append("output.close")
        return ()


class _FailingWriteOutputTarget(_RecordingOutputTarget):
    def write(self, result: StepResult) -> None:
        super().write(result)
        raise RuntimeError("write failed")


class _FailingCloseOutputTarget(_RecordingOutputTarget):
    def close(self) -> Sequence[OutputArtifact]:
        super().close()
        raise RuntimeError("close failed")


class _RecordingMetricsRecorder:
    def __init__(self, *, events: list[str]) -> None:
        self._events = events
        self.samples: list[RuntimeMetricSample] = []

    def record(self, sample: RuntimeMetricSample) -> None:
        self.samples.append(sample)

    def record_timing(
        self,
        name: str,
        duration_s: float,
        *,
        step_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.record(
            RuntimeMetricSample(
                name=name,
                value=duration_s,
                unit="s",
                step_index=step_index,
                category="timing",
                metadata={} if metadata is None else metadata,
            )
        )

    def record_step(
        self,
        *,
        request: object,
        user_window: object,
        inference_input: object,
        result: object,
        decision: object,
    ) -> None:
        del request, user_window, inference_input, result, decision

    def record_control(
        self,
        *,
        request: object,
        user_window: object,
        control: object,
    ) -> None:
        del request, user_window, control

    def record_error(self, exc: Exception, action: object) -> None:
        del exc, action

    def record_catch_up(self, decision: object) -> None:
        del decision

    def record_cleanup_error(self, exc: Exception) -> None:
        del exc

    def record_orphaned_cleanup(self, exc: Exception) -> None:
        del exc

    def record_session(self, result: object) -> None:
        del result

    def record_session_error(self, exc: Exception) -> None:
        del exc

    def close(self) -> MetricsSnapshot:
        self._events.append("metrics.close")
        return MetricsSnapshot()
