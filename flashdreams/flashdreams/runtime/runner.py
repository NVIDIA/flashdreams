# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal synchronous standard runner for the runtime API."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import (
    CanonicalInputs,
    CanonicalInputSchema,
    InferenceInput,
    InferenceInputSchema,
    TimeWindow,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.interfaces import (
    InferenceRuntime,
    InferenceSession,
    ModelAdapter,
)
from flashdreams.runtime.mapping import (
    DeclaresMappingSchema,
    InputMapping,
    check_mapping_compatibility,
)
from flashdreams.runtime.metrics import MetricsRecorder
from flashdreams.runtime.output import OutputArtifact, OutputTarget
from flashdreams.runtime.types import (
    StepRequest,
    StepRequirements,
    StepResult,
    step_requirements_from_request,
)

if TYPE_CHECKING:
    from flashdreams.runtime.demo.host import RuntimeHost
    from flashdreams.runtime.demo.outputs import OutputDecision, SessionInfo
    from flashdreams.runtime.demo.run_modes import RunResult
    from flashdreams.runtime.demo.session_inputs import PreparedStep, UserInputWindow

_DEFAULT_SESSION_HORIZON_S = 3600.0


def run_inference_session(
    *,
    adapter: ModelAdapter,
    config: InferenceConfig,
    mapping: InputMapping,
    canonicalizer: InputCanonicalizer,
    source_schema: UserInputSchema,
    user_inputs: UserInputs,
    initial_inputs: InferenceInput,
    output: OutputTarget,
    metrics: MetricsRecorder,
) -> tuple[OutputArtifact, ...]:
    """Run one sequential inference session through the shared batch driver.

    The signature and failure semantics remain compatible with the original
    replay runner while the implementation delegates the step loop to the shared
    demo runtime pipeline.
    """

    return _run_inference_session_with_shared_batch(
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


def _run_inference_session_with_shared_batch(
    *,
    adapter: ModelAdapter,
    config: InferenceConfig,
    mapping: InputMapping,
    canonicalizer: InputCanonicalizer,
    source_schema: UserInputSchema,
    user_inputs: UserInputs,
    initial_inputs: InferenceInput,
    output: OutputTarget,
    metrics: MetricsRecorder,
) -> tuple[OutputArtifact, ...]:
    lifecycle = _LegacyBatchLifecycle()
    request_state = _LegacyStepRequestState()
    runtime = _LegacyLazyRuntime(
        adapter=adapter,
        config=config,
        lifecycle=lifecycle,
        request_state=request_state,
    )
    from flashdreams.runtime.demo.host import RuntimeHost

    host = RuntimeHost(runtime)
    metrics_recorder = _LegacyRunnerMetricsRecorder(metrics)
    output_sink = _LegacyOutputTargetSink(
        output=output,
        lifecycle=lifecycle,
        host=host,
    )
    primary_error: BaseException | None = None

    try:
        _validate_legacy_runner_inputs(
            adapter=adapter,
            config=config,
            mapping=mapping,
            canonicalizer=canonicalizer,
            source_schema=source_schema,
        )
        provider = _LegacyMappedModelInputProvider(
            mapping=mapping,
            canonicalizer=canonicalizer,
            source_schema=source_schema,
            user_inputs=user_inputs,
            initial_inputs=initial_inputs,
            inference_input_schema=adapter.inference_input_schema,
            request_state=request_state,
        )
        input_source = _LegacyBatchInputSource(
            source_schema=source_schema,
            user_inputs=user_inputs,
            request_state=request_state,
        )
        result = _run_shared_batch_session(
            host=host,
            provider=provider,
            input_source=input_source,
            output_sink=output_sink,
            metrics=metrics_recorder,
        )
        _raise_legacy_runner_error(
            result=result,
            output_sink=output_sink,
            metrics=metrics_recorder,
        )
        return tuple(result.artifacts)
    except BaseException as exc:
        primary_error = exc
        if not metrics_recorder.closed:
            _close_metrics_suppressing_secondary(metrics_recorder)
        raise
    finally:
        try:
            host.close()
        except BaseException:
            if primary_error is None:
                raise


def _validate_legacy_runner_inputs(
    *,
    adapter: ModelAdapter,
    config: InferenceConfig,
    mapping: InputMapping,
    canonicalizer: InputCanonicalizer,
    source_schema: UserInputSchema,
) -> CanonicalInputSchema:
    adapter.validate_config(config)
    canonical_schema = canonicalizer.canonical_schema(source_schema)
    _check_declared_mapping_compatibility(
        mapping=mapping,
        canonical_schema=canonical_schema,
        adapter=adapter,
    )
    mapping.validate(
        canonical_schema=canonical_schema,
        inference_input_schema=adapter.inference_input_schema,
    )
    return canonical_schema


def _run_shared_batch_session(
    *,
    host: RuntimeHost,
    provider: "_LegacyMappedModelInputProvider",
    input_source: "_LegacyBatchInputSource",
    output_sink: "_LegacyOutputTargetSink",
    metrics: "_LegacyRunnerMetricsRecorder",
) -> RunResult:
    from flashdreams.runtime.demo.drivers import BatchSessionDriver
    from flashdreams.runtime.demo.pipeline import StepPipeline
    from flashdreams.runtime.demo.run_modes import SessionEdges

    return BatchSessionDriver().run_one_session(
        host=host,
        provider=provider,
        session_edges=SessionEdges(
            input_source=input_source,
            output_sink=output_sink,
            cleanup_tasks=set(),
            metrics=metrics,
        ),
        pipeline=StepPipeline(),
    )


class _LegacyLazyRuntime:
    """Create the legacy runtime only after global inputs are mapped."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        config: InferenceConfig,
        lifecycle: "_LegacyBatchLifecycle",
        request_state: "_LegacyStepRequestState",
    ) -> None:
        self._adapter = adapter
        self._config = config
        self._lifecycle = lifecycle
        self._request_state = request_state

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        runtime = self._lifecycle.runtime
        if runtime is None:
            runtime = self._adapter.create_runtime(self._config)
            self._lifecycle.set_runtime(runtime)
        session = runtime.start_session(inputs)
        self._lifecycle.set_session(session)
        return _LegacySessionAdapter(
            session=session,
            request_state=self._request_state,
        )

    def close(self) -> None:
        self._lifecycle.close_runtime_direct()


class _LegacyBatchLifecycle:
    """Own legacy model resources whose close order is caller-visible."""

    def __init__(self) -> None:
        self.runtime: InferenceRuntime | None = None
        self.session: InferenceSession | None = None
        self._session_close_attempted = False
        self._runtime_close_attempted = False

    def set_runtime(self, runtime: InferenceRuntime) -> None:
        self.runtime = runtime

    def set_session(self, session: InferenceSession) -> None:
        self.session = session
        self._session_close_attempted = False

    def close_session_via_host(self, host: RuntimeHost) -> None:
        host.call(self.close_session_direct)

    def close_runtime_via_host(self, host: RuntimeHost) -> None:
        host.call(self.close_runtime_direct)

    def close_session_direct(self) -> None:
        if self.session is None or self._session_close_attempted:
            return
        self._session_close_attempted = True
        self.session.close()

    def close_runtime_direct(self) -> None:
        if self.runtime is None or self._runtime_close_attempted:
            return
        self._runtime_close_attempted = True
        self.runtime.close()


class _LegacySessionAdapter:
    """Expose old sessions through the new StepRequirements boundary."""

    def __init__(
        self,
        *,
        session: InferenceSession,
        request_state: "_LegacyStepRequestState",
    ) -> None:
        self._session = session
        self._request_state = request_state

    def next_step_requirements(self) -> StepRequirements | None:
        request = self._session.next_step_request()
        if request is None:
            self._request_state.clear()
            return None
        self._request_state.store(request)
        return step_requirements_from_request(
            request,
            allow_user_input_window=True,
        )

    def next_step_request(self) -> StepRequest | None:
        return self._session.next_step_request()

    def session_info(self) -> SessionInfo:
        from flashdreams.runtime.demo.outputs import SessionInfo

        session_info = getattr(self._session, "session_info", None)
        if not callable(session_info):
            return SessionInfo()
        value = session_info()
        if not isinstance(value, SessionInfo):
            raise TypeError(
                "session.session_info() must return SessionInfo, "
                f"got {type(value).__name__}."
            )
        return value

    def step(self, inputs: InferenceInput) -> StepResult:
        return self._session.step(inputs)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        self._session.reset(inputs)

    def close(self) -> None:
        return None


class _LegacyStepRequestState:
    """Share the current legacy request between the session, source, and provider."""

    def __init__(self) -> None:
        self._request: StepRequest | None = None

    def store(self, request: StepRequest) -> None:
        self._request = request

    def require_for_window(self, step_index: int) -> StepRequest:
        request = self._request
        if request is None:
            raise RuntimeError("Legacy input source has no active step request.")
        if request.step_index != step_index:
            raise RuntimeError(
                "Legacy input source request mismatch: "
                f"expected step {request.step_index}, got {step_index}."
            )
        return request

    def consume_for_step(self, step_index: int) -> StepRequest:
        request = self.require_for_window(step_index)
        self._request = None
        return request

    def clear(self) -> None:
        self._request = None


class _LegacyBatchInputSource:
    is_finite = True
    is_deterministic = True

    def __init__(
        self,
        *,
        source_schema: UserInputSchema,
        user_inputs: UserInputs,
        request_state: _LegacyStepRequestState,
    ) -> None:
        self.user_input_schema = source_schema
        self._user_inputs = user_inputs
        self._request_state = request_state

    def is_finished(self) -> bool:
        return False

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        from flashdreams.runtime.demo.session_inputs import UserInputWindow

        legacy_request = self._request_state.require_for_window(request.step_index)
        window = legacy_request.user_input_window or _all_user_inputs_window(
            self._user_inputs
        )
        return UserInputWindow(
            start_s=window.start_s,
            end_s=window.end_s,
            inputs=self._user_inputs,
        )


class _LegacyMappedModelInputProvider:
    def __init__(
        self,
        *,
        mapping: InputMapping,
        canonicalizer: InputCanonicalizer,
        source_schema: UserInputSchema,
        user_inputs: UserInputs,
        initial_inputs: InferenceInput,
        inference_input_schema: InferenceInputSchema,
        request_state: _LegacyStepRequestState,
    ) -> None:
        from flashdreams.runtime.demo.session_inputs import ProviderCapabilities

        self.capabilities = ProviderCapabilities(
            supports_recorded_input=True,
            deterministic_given_inputs=True,
            user_input_schema=source_schema,
            inference_input_schema=inference_input_schema,
        )
        self._mapping = mapping
        self._canonicalizer = canonicalizer
        self._source_schema = source_schema
        self._user_inputs = user_inputs
        self._initial_inputs = initial_inputs
        self._request_state = request_state
        self._step_base_inputs = InferenceInput(
            step=initial_inputs.step,
            metadata=initial_inputs.metadata,
        )

    def prepare_initial_input(self) -> InferenceInput:
        self._canonicalizer.reset()
        return self._mapping.map_global_conditioning_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=self._initial_inputs,
        )

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        from flashdreams.runtime.demo.session_inputs import PreparedStep

        legacy_request = self._request_state.consume_for_step(request.step_index)
        canonical_inputs = self._canonicalizer.canonicalize(
            self._user_inputs,
            window=TimeWindow(start_s=user_window.start_s, end_s=user_window.end_s),
            source_schema=self._source_schema,
        )
        return PreparedStep(
            inference_input=self._mapping.map_step_inputs(
                canonical_inputs=canonical_inputs,
                inference_input=self._step_base_inputs,
                request=legacy_request,
            )
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._canonicalizer.reset()

    def close(self) -> None:
        return None


class _LegacyOutputTargetSink:
    produces_artifacts = True

    def __init__(
        self,
        *,
        output: OutputTarget,
        lifecycle: _LegacyBatchLifecycle,
        host: RuntimeHost,
    ) -> None:
        self._output = output
        self._lifecycle = lifecycle
        self._host = host
        self._opened = False
        self._closed = False
        self._artifacts: tuple[OutputArtifact, ...] = ()
        self.cleanup_error: BaseException | None = None

    def open(self, session_info: SessionInfo) -> None:
        del session_info
        self._output.open()
        self._opened = True
        self._closed = False
        self._artifacts = ()
        self.cleanup_error = None

    def begin_generation(self, generation: int) -> None:
        del generation

    def write(self, result: StepResult) -> OutputDecision:
        from flashdreams.runtime.demo.outputs import OutputDecision

        self._output.write(result)
        return OutputDecision()

    def close(self) -> Sequence[OutputArtifact]:
        if self._closed:
            if self.cleanup_error is not None:
                raise self.cleanup_error
            return self._artifacts

        self._closed = True
        cleanup_error: BaseException | None = None
        artifacts: tuple[OutputArtifact, ...] = ()

        def remember_error(exc: BaseException) -> None:
            nonlocal cleanup_error
            if cleanup_error is None:
                cleanup_error = exc

        if self._opened:
            try:
                artifacts = tuple(self._output.close())
            except BaseException as exc:
                remember_error(exc)
        try:
            self._lifecycle.close_session_via_host(self._host)
        except BaseException as exc:
            remember_error(exc)
        try:
            self._lifecycle.close_runtime_via_host(self._host)
        except BaseException as exc:
            remember_error(exc)

        self._artifacts = artifacts
        self.cleanup_error = cleanup_error
        if cleanup_error is not None:
            raise cleanup_error
        return self._artifacts


class _LegacyRunnerMetricsRecorder:
    def __init__(self, metrics: MetricsRecorder) -> None:
        self._metrics = metrics
        self.closed = False
        self.close_error: BaseException | None = None

    def record(self, sample: Any) -> None:
        self._metrics.record(sample)

    def record_timing(
        self,
        name: str,
        duration_s: float,
        *,
        step_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._metrics.record_timing(
            name,
            duration_s,
            step_index=step_index,
            metadata=metadata,
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
        del request, user_window, inference_input, decision
        if isinstance(result, StepResult):
            _record_timing_metrics(self._metrics, result)

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

    def close(self) -> Any:
        if self.closed:
            if self.close_error is not None:
                raise self.close_error
            return None
        self.closed = True
        try:
            return self._metrics.close()
        except BaseException as exc:
            self.close_error = exc
            raise


def _raise_legacy_runner_error(
    *,
    result: RunResult,
    output_sink: _LegacyOutputTargetSink,
    metrics: _LegacyRunnerMetricsRecorder,
) -> None:
    if result.error is not None:
        raise result.error
    if output_sink.cleanup_error is not None:
        raise output_sink.cleanup_error
    if metrics.close_error is not None:
        raise metrics.close_error


def _close_metrics_suppressing_secondary(
    metrics: _LegacyRunnerMetricsRecorder,
) -> None:
    try:
        metrics.close()
    except BaseException:
        return


def _check_declared_mapping_compatibility(
    *,
    mapping: InputMapping,
    canonical_schema: CanonicalInputSchema,
    adapter: ModelAdapter,
) -> None:
    if not isinstance(mapping, DeclaresMappingSchema):
        return
    compatibility = check_mapping_compatibility(
        canonical_schema=canonical_schema,
        inference_input_schema=adapter.inference_input_schema,
        mapping_schema=mapping.mapping_schema,
    )
    compatibility.raise_if_incompatible()


def _all_user_inputs_window(user_inputs: UserInputs) -> TimeWindow:
    if not user_inputs.events:
        return TimeWindow(start_s=0.0, end_s=_DEFAULT_SESSION_HORIZON_S)
    return TimeWindow(
        start_s=0.0,
        end_s=max(
            _DEFAULT_SESSION_HORIZON_S,
            math.nextafter(user_inputs.events[-1].timestamp_s, math.inf),
        ),
    )


def _record_timing_metrics(
    metrics: MetricsRecorder,
    result: StepResult,
) -> None:
    for name, value in result.metrics.items():
        if not name.endswith("_s") or isinstance(value, bool):
            continue
        sample_name = name[:-2] or name
        metrics.record_timing(
            sample_name,
            float(value),
            step_index=result.step_index,
        )


__all__ = ["run_inference_session"]
