# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared replay demo runner."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from flashdreams.demo.io import OutputDecision, OutputSink
from flashdreams.demo.outputs import build_output_sink
from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import (
    CanonicalInputs,
    CanonicalInputSchema,
    InferenceInput,
    InferenceInputSchema,
    TimeWindow,
    UserInputSchema,
)
from flashdreams.runtime.interfaces import InferenceRuntime, InferenceSession
from flashdreams.runtime.mapping import (
    DeclaresMappingSchema,
    InputMapping,
    check_mapping_compatibility,
)
from flashdreams.runtime.metrics import MetricsRecorder, NullMetricsRecorder
from flashdreams.runtime.output import OutputArtifact, OutputTarget
from flashdreams.runtime.types import (
    StepRequest,
    StepRequirements,
    StepResult,
    step_requirements_from_request,
)

from .drivers import BatchSessionDriver, run_demo_session
from .host import ModelWarmupPlan, RuntimeHost
from .pipeline import StepPipeline
from .run_modes import (
    Mp4ErrorPolicy,
    NullErrorPolicy,
    RunContext,
    RunModeCapabilities,
    RunResult,
    SessionEdges,
    SingleSessionAdmissionPolicy,
)
from .session_inputs import (
    PreparedScenarioBatchInputSource,
    PreparedStep,
    ProviderCapabilities,
    StepRequestWindowState,
    UserInputWindow,
)
from .spec import (
    DemoAdapter,
    DemoSpec,
    OutputSpec,
    PreparedScenario,
    WebRTCOutputSpec,
)

OutputTargetFactory = Callable[[OutputSpec], OutputTarget]
OutputSinkFactory = Callable[[OutputSpec], OutputSink]


def run_replay_demo(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    output_target_factory: OutputTargetFactory | None = None,
    output_sink_factory: OutputSinkFactory = build_output_sink,
    metrics: MetricsRecorder | None = None,
) -> RunResult:
    """Run one prepared replay scenario through the shared batch demo path."""
    _require_supported_mode(
        mode=spec.input_mode,
        supported=adapter.supported_input_modes(),
        label="input_mode",
    )
    if spec.input_mode != "replay":
        raise ValueError(
            "run_replay_demo requires input_mode='replay', "
            f"got input_mode={spec.input_mode!r}."
        )
    _require_supported_mode(
        mode=spec.output.mode,
        supported=adapter.supported_output_modes(),
        label="output.mode",
    )
    if isinstance(spec.output, WebRTCOutputSpec):
        raise ValueError("run_replay_demo does not support WebRTC output.")

    prepared = adapter.prepare_scenario(spec)
    mapping = _scenario_mapping(prepared=prepared, adapter=adapter)
    if spec.config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")

    if output_target_factory is not None:
        output_sink_factory = _output_target_sink_factory(output_target_factory)

    return _run_replay_demo_with_run_mode(
        spec=spec,
        adapter=adapter,
        prepared=prepared,
        mapping=mapping,
        output_sink_factory=output_sink_factory,
        metrics=metrics,
    )


def _output_target_sink_factory(
    output_target_factory: OutputTargetFactory,
) -> OutputSinkFactory:
    def create_output_sink(output_spec: OutputSpec) -> "_OutputTargetSink":
        return _OutputTargetSink(output_target_factory(output_spec))

    return create_output_sink


class _OutputTargetSink(OutputSink):
    produces_artifacts = True

    def __init__(self, output: OutputTarget) -> None:
        self._output = output
        self._closed = True
        self._artifacts: tuple[OutputArtifact, ...] | None = None

    def open(self, session_info: object) -> None:
        del session_info
        self._output.open()
        self._closed = False
        self._artifacts = None

    def begin_generation(self, generation: int) -> None:
        del generation

    def write(self, result: StepResult) -> OutputDecision:
        self._output.write(result)
        return OutputDecision()

    def close(self) -> Sequence[OutputArtifact]:
        if self._artifacts is not None:
            return self._artifacts
        if self._closed:
            self._artifacts = ()
            return self._artifacts
        self._closed = True
        self._artifacts = tuple(self._output.close())
        return self._artifacts


def _run_replay_demo_with_run_mode(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    prepared: "PreparedScenario",
    mapping: InputMapping | None,
    output_sink_factory: OutputSinkFactory,
    metrics: MetricsRecorder | None,
) -> RunResult:
    config = _require_config(spec)
    if mapping is None:
        if not callable(getattr(adapter, "create_model_input_provider", None)):
            raise ValueError(
                "Demo scenario did not provide an input mapping, and the adapter "
                "has no model input provider or default input mapping."
            )
        adapter.validate_config(config)
    else:
        _validate_replay_mapping(
            adapter=adapter,
            config=config,
            mapping=mapping,
            source_schema=prepared.source_schema,
            canonicalizer=prepared.canonicalizer,
        )
    request_state = StepRequestWindowState()
    runtime = _ReplayRuntimeAdapter(
        runtime=adapter.create_runtime(config),
        request_state=request_state,
    )
    host = RuntimeHost(runtime)
    mode = _ReplayRunMode(
        request_state=request_state,
        output_sink_factory=output_sink_factory,
        run_metrics=metrics or NullMetricsRecorder(),
    )
    replay_adapter = _ReplayProviderAdapter(
        adapter=adapter,
        mapping=mapping,
        request_state=request_state,
    )
    context = mode.create_run_context(
        spec=spec,
        adapter=replay_adapter,
        host=host,
        model_warmup_plan=ModelWarmupPlan(),
    )
    try:
        return run_demo_session(
            context=context,
            spec=spec,
            scenario=prepared,
            adapter=replay_adapter,
            run_mode=mode,
            pipeline=StepPipeline(),
        )
    finally:
        context.close()
        host.close()


class _ReplayRunMode:
    name = "replay"
    capabilities = RunModeCapabilities(
        requires_finite_input=True,
        supports_artifacts=True,
    )

    def __init__(
        self,
        *,
        request_state: StepRequestWindowState,
        output_sink_factory: OutputSinkFactory,
        run_metrics: MetricsRecorder,
    ) -> None:
        self._request_state = request_state
        self._output_sink_factory = output_sink_factory
        self._run_metrics = run_metrics

    def validate_run(self, *, spec: DemoSpec, adapter: DemoAdapter) -> None:
        del spec, adapter

    def validate_session(
        self,
        *,
        spec: DemoSpec,
        scenario: "PreparedScenario",
        adapter: DemoAdapter,
        provider: object,
    ) -> None:
        del spec, scenario, adapter, provider

    def create_run_context(
        self,
        *,
        spec: DemoSpec,
        adapter: DemoAdapter,
        host: RuntimeHost,
        model_warmup_plan: ModelWarmupPlan,
    ) -> RunContext:
        del spec, adapter
        return RunContext(
            host=host,
            run_metrics=self._run_metrics,
            admission=SingleSessionAdmissionPolicy(
                health_check=lambda: host.is_healthy
            ),
            model_warmup_plan=model_warmup_plan,
        )

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: "PreparedScenario",
        provider: object,
        adapter: DemoAdapter,
    ) -> SessionEdges:
        del provider, adapter
        return SessionEdges(
            input_source=_ReplayBatchInputSource(
                scenario=scenario,
                request_state=self._request_state,
            ),
            output_sink=self._output_sink_factory(spec.output),
            cleanup_tasks=context.cleanup_tasks,
            error_policy=(
                Mp4ErrorPolicy() if spec.output.mode == "mp4" else NullErrorPolicy()
            ),
        )

    def select_driver(self) -> BatchSessionDriver:
        return BatchSessionDriver()


class _ReplayProviderAdapter:
    def __init__(
        self,
        *,
        adapter: DemoAdapter,
        mapping: InputMapping | None,
        request_state: StepRequestWindowState,
    ) -> None:
        self._adapter = adapter
        self._mapping = mapping
        self._request_state = request_state

    @property
    def model_id(self) -> str:
        return self._adapter.model_id

    @property
    def inference_input_schema(self) -> InferenceInputSchema:
        return self._adapter.inference_input_schema

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema | None:
        return self._adapter.canonical_input_schema

    def default_input_mapping(self) -> InputMapping | None:
        return self._adapter.default_input_mapping()

    def supported_input_modes(self) -> tuple[str, ...]:
        return self._adapter.supported_input_modes()

    def supported_output_modes(self) -> tuple[str, ...]:
        return self._adapter.supported_output_modes()

    def validate_config(self, config: InferenceConfig) -> None:
        self._adapter.validate_config(config)

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        return self._adapter.create_runtime(config)

    def prepare_scenario(self, spec: DemoSpec) -> "PreparedScenario":
        return self._adapter.prepare_scenario(spec)

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: "PreparedScenario",
    ) -> object:
        create_provider = getattr(self._adapter, "create_model_input_provider", None)
        if callable(create_provider):
            return create_provider(spec, scenario)
        if self._mapping is None:
            raise ValueError(
                "Replay adapter requires an input mapping when no model input "
                "provider is available."
            )
        return _ReplayMappingModelInputProvider(
            adapter=self._adapter,
            scenario=scenario,
            mapping=self._mapping,
            request_state=self._request_state,
        )


class _ReplayRuntimeAdapter:
    def __init__(
        self,
        *,
        runtime: InferenceRuntime,
        request_state: StepRequestWindowState,
    ) -> None:
        self._runtime = runtime
        self._request_state = request_state

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        return _ReplaySessionAdapter(
            session=self._runtime.start_session(inputs),
            request_state=self._request_state,
        )

    def close(self) -> None:
        self._runtime.close()


class _ReplaySessionAdapter:
    def __init__(
        self,
        *,
        session: InferenceSession,
        request_state: StepRequestWindowState,
    ) -> None:
        self._session = session
        self._request_state = request_state

    def next_step_requirements(self) -> StepRequirements | None:
        next_requirements = getattr(self._session, "next_step_requirements", None)
        if callable(next_requirements):
            value = next_requirements()
            self._request_state.clear()
            return value

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

    def step(self, inputs: InferenceInput) -> StepResult:
        return self._session.step(inputs)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        self._session.reset(inputs)

    def close(self) -> None:
        self._session.close()


_ReplayBatchInputSource = PreparedScenarioBatchInputSource


class _ReplayMappingModelInputProvider:
    def __init__(
        self,
        *,
        adapter: DemoAdapter,
        scenario: "PreparedScenario",
        mapping: InputMapping,
        request_state: StepRequestWindowState,
    ) -> None:
        self.capabilities = ProviderCapabilities(
            supports_recorded_input=True,
            deterministic_given_inputs=True,
            user_input_schema=scenario.source_schema,
            inference_input_schema=adapter.inference_input_schema,
        )
        self._scenario = scenario
        self._mapping = mapping
        self._request_state = request_state
        self._step_base_inputs = InferenceInput(
            step=scenario.initial_inputs.step,
            metadata=scenario.initial_inputs.metadata,
        )

    def prepare_initial_input(self) -> InferenceInput:
        self._scenario.canonicalizer.reset()
        return self._mapping.map_global_conditioning_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=self._scenario.initial_inputs,
        )

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        legacy_request = self._request_state.consume_for_step(request)
        canonical_inputs = self._scenario.canonicalizer.canonicalize(
            self._scenario.user_inputs,
            window=TimeWindow(start_s=user_window.start_s, end_s=user_window.end_s),
            source_schema=self._scenario.source_schema,
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
        self._scenario.canonicalizer.reset()

    def close(self) -> None:
        return None


def _validate_replay_mapping(
    *,
    adapter: DemoAdapter,
    config: InferenceConfig,
    mapping: InputMapping,
    source_schema: UserInputSchema,
    canonicalizer: InputCanonicalizer,
) -> None:
    adapter.validate_config(config)
    canonical_schema = canonicalizer.canonical_schema(source_schema)
    if isinstance(mapping, DeclaresMappingSchema):
        compatibility = check_mapping_compatibility(
            canonical_schema=canonical_schema,
            inference_input_schema=adapter.inference_input_schema,
            mapping_schema=mapping.mapping_schema,
        )
        compatibility.raise_if_incompatible()
    mapping.validate(
        canonical_schema=canonical_schema,
        inference_input_schema=adapter.inference_input_schema,
    )


def _require_config(spec: DemoSpec) -> InferenceConfig:
    if spec.config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    return spec.config


def _scenario_mapping(
    *,
    prepared: PreparedScenario,
    adapter: DemoAdapter,
) -> InputMapping | None:
    if prepared.mapping is not None:
        return prepared.mapping
    default_input_mapping = getattr(adapter, "default_input_mapping", None)
    if not callable(default_input_mapping):
        return None
    return default_input_mapping()


def _require_supported_mode(
    *,
    mode: str,
    supported: tuple[str, ...],
    label: str,
) -> None:
    if mode in supported:
        return
    supported_text = ", ".join(repr(each) for each in supported) or "<none>"
    raise ValueError(
        f"Unsupported demo {label}={mode!r}; supported modes: {supported_text}."
    )


__all__ = [
    "OutputSinkFactory",
    "OutputTargetFactory",
    "run_replay_demo",
]
