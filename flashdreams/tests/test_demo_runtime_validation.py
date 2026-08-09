# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

import flashdreams.runtime.demo as demo_api
from flashdreams.runtime import (
    DRIVER_COMMAND,
    CanonicalInputs,
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceRuntime,
    InputCanonicalizer,
    InputField,
    InputMapping,
    InputMappingSchema,
    KeyboardToDriverCommand,
    OutputArtifact,
    StepRequest,
    StepRequirements,
    StepResult,
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    OutputDecision,
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    ResolvedRunCapabilities,
    RunContext,
    RunModeCapabilities,
    RunResult,
    RuntimeHost,
    SessionEdges,
    SessionInfo,
    UserInputWindow,
    resolve_run_capabilities,
    validate_resolved_run,
)
from flashdreams.runtime.demo.timing import RealtimeWindowResult

pytestmark = pytest.mark.ci_cpu

KEY_SCHEMA = UserInputSchema(
    capabilities=(
        UserInputCapability(
            event_type="key_down",
            payload_fields=frozenset({"key"}),
        ),
        UserInputCapability(
            event_type="key_up",
            payload_fields=frozenset({"key"}),
        ),
    )
)


def test_provider_capabilities_declare_raw_and_inference_schemas() -> None:
    capabilities = ProviderCapabilities(
        supports_recorded_input=True,
        deterministic_given_inputs=True,
        user_input_schema=KEY_SCHEMA,
        inference_input_schema=InferenceInputSchema(
            step_fields=(InputField(name="driver_command"),),
        ),
    )

    assert capabilities.user_input_schema.supports(
        UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"}))
    )
    assert capabilities.inference_input_schema.missing_step(InferenceInput()) == (
        "driver_command",
    )


def test_provider_that_wraps_mapping_canonicalizes_raw_inputs_first() -> None:
    mapping = _DriverCommandMapping()
    provider = _MappingBackedProvider(mapping=mapping, source_schema=KEY_SCHEMA)
    source = _BatchInputSource(user_input_schema=KEY_SCHEMA)
    edges = _edges(input_source=source)
    run_mode = _RunMode(
        RunModeCapabilities(requires_finite_input=True, supports_artifacts=True)
    )
    resolved = resolve_run_capabilities(
        spec=_spec(seed=7, output=Mp4OutputSpec(path="out.mp4", fps=12)),
        provider=provider,
        session_edges=edges,
    )

    validate_resolved_run(
        spec=_spec(seed=7),
        adapter=_Adapter(),
        provider=provider,
        run_mode=run_mode,
        session_edges=edges,
        resolved=resolved,
    )
    prepared = provider.prepare_step(
        request=StepRequirements(step_index=4),
        user_window=UserInputWindow(
            start_s=0.0,
            end_s=1.0,
            inputs=UserInputs(
                events=(
                    UserInputEvent(
                        timestamp_s=0.1,
                        event_type="key_down",
                        payload={"key": "w"},
                    ),
                )
            ),
        ),
    )

    assert mapping.validated_with is not None
    assert mapping.validated_with[0] == CanonicalInputSchema(
        modalities=(DRIVER_COMMAND,),
        description=KEY_SCHEMA.description,
    )
    assert prepared.inference_input is not None
    assert prepared.inference_input.step["request_step"] == 4
    assert prepared.inference_input.step["driver_command"]["throttle"] == 1.0


def test_raw_user_schema_validation_is_not_canonical_schema_validation() -> None:
    provider = _Provider(
        ProviderCapabilities(
            supports_recorded_input=True,
            user_input_schema=KEY_SCHEMA,
            inference_input_schema=InferenceInputSchema(
                step_fields=(InputField(name="driver_command"),),
            ),
        )
    )
    source = _BatchInputSource(
        user_input_schema=UserInputSchema(event_types=frozenset({"key_down"}))
    )
    run_mode = _RunMode(
        RunModeCapabilities(requires_finite_input=True, supports_artifacts=True)
    )
    edges = _edges(input_source=source)
    resolved = resolve_run_capabilities(
        spec=_spec(seed=1),
        provider=provider,
        session_edges=edges,
    )

    with pytest.raises(ValueError, match="raw user input schema"):
        validate_resolved_run(
            spec=_spec(seed=1),
            adapter=_Adapter(),
            provider=provider,
            run_mode=run_mode,
            session_edges=edges,
            resolved=resolved,
        )


def test_mp4_rejects_provider_without_recorded_input_support() -> None:
    provider = _Provider(
        ProviderCapabilities(
            supports_recorded_input=False,
            supports_realtime_clock=True,
        )
    )
    run_mode = _RunMode(
        RunModeCapabilities(requires_finite_input=True, supports_artifacts=True)
    )
    edges = _edges(output_sink=_OutputSink(produces_artifacts=True))
    resolved = resolve_run_capabilities(
        spec=_spec(seed=1, output=Mp4OutputSpec(path="out.mp4", fps=12)),
        provider=provider,
        session_edges=edges,
    )

    with pytest.raises(ValueError, match="recorded input"):
        validate_resolved_run(
            spec=_spec(seed=1),
            adapter=_Adapter(),
            provider=provider,
            run_mode=run_mode,
            session_edges=edges,
            resolved=resolved,
        )


def test_webrtc_rejects_provider_without_realtime_input_support() -> None:
    provider = _Provider(ProviderCapabilities(supports_recorded_input=True))
    run_mode = _RunMode(
        RunModeCapabilities(
            realtime=True,
            supports_backpressure=True,
            supports_interactive_events=True,
        )
    )
    edges = _edges(input_source=_RealtimeInputSource(), clock=_Clock(realtime=True))
    resolved = resolve_run_capabilities(
        spec=_spec(seed=1),
        provider=provider,
        session_edges=edges,
    )

    with pytest.raises(ValueError, match="realtime input"):
        validate_resolved_run(
            spec=_spec(seed=1),
            adapter=_Adapter(),
            provider=provider,
            run_mode=run_mode,
            session_edges=edges,
            resolved=resolved,
        )


def test_realtime_run_mode_rejects_batch_input_source() -> None:
    provider = _Provider(ProviderCapabilities(supports_realtime_clock=True))
    run_mode = _RunMode(RunModeCapabilities(realtime=True))
    edges = _edges(input_source=_BatchInputSource(), clock=_Clock(realtime=True))
    resolved = resolve_run_capabilities(
        spec=_spec(seed=1),
        provider=provider,
        session_edges=edges,
    )

    with pytest.raises(ValueError, match="RealtimeInputSource"):
        validate_resolved_run(
            spec=_spec(seed=1),
            adapter=_Adapter(),
            provider=provider,
            run_mode=run_mode,
            session_edges=edges,
            resolved=resolved,
        )


def test_determinism_resolves_from_provider_source_clock_and_seed() -> None:
    provider = _Provider(
        ProviderCapabilities(
            supports_recorded_input=True,
            supports_reset=True,
            deterministic_given_inputs=True,
        )
    )

    deterministic = resolve_run_capabilities(
        spec=_spec(seed=123),
        provider=provider,
        session_edges=_edges(clock=_Clock(deterministic=True)),
    )
    unseeded = resolve_run_capabilities(
        spec=_spec(seed=None),
        provider=provider,
        session_edges=_edges(clock=_Clock(deterministic=True)),
    )
    nondeterministic_source = resolve_run_capabilities(
        spec=_spec(seed=123),
        provider=provider,
        session_edges=_edges(
            input_source=_BatchInputSource(deterministic=False),
            clock=_Clock(deterministic=True),
        ),
    )

    assert deterministic == ResolvedRunCapabilities(
        finite=True,
        deterministic=True,
        realtime=False,
        resettable=True,
        produces_artifacts=True,
    )
    assert not unseeded.deterministic
    assert not nondeterministic_source.deterministic


def test_no_general_purpose_input_mapping_provider_is_exported() -> None:
    assert not hasattr(demo_api, "InputMappingProvider")


def test_reset_control_updates_provider_and_session_together() -> None:
    reset_input = InferenceInput(global_conditioning={"prompt": "reset"})
    provider = _ResettingProvider(reset_input=reset_input)
    session = _ResettableSession(num_steps=2)
    edges = _edges(input_source=_BatchInputSource(num_windows=2))

    result = demo_api.BatchSessionDriver().run_one_session(
        host=RuntimeHost(_Runtime(session=session)),
        provider=provider,
        session_edges=edges,
        pipeline=demo_api.StepPipeline(),
    )

    assert result.status == "completed"
    assert session.reset_inputs == [reset_input]
    assert provider.reset_inputs == [reset_input]
    assert len(session.step_inputs) == 1


def _spec(
    *,
    seed: int | None,
    output: Any | None = None,
) -> DemoSpec:
    return DemoSpec(
        model_id="fake-demo",
        input_mode="replay",
        output=output or NullOutputSpec(),
        config=InferenceConfig(model_id="fake-demo", seed=seed),
    )


def _edges(
    *,
    input_source: Any | None = None,
    output_sink: Any | None = None,
    clock: Any | None = None,
) -> SessionEdges:
    return SessionEdges(
        input_source=input_source or _BatchInputSource(),
        output_sink=output_sink or _OutputSink(produces_artifacts=True),
        cleanup_tasks=set(),
        clock=clock,
    )


class _Adapter:
    model_id = "fake-demo"
    inference_input_schema = InferenceInputSchema()
    canonical_input_schema = None

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("null", "mp4")

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        del config

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        del config
        raise NotImplementedError

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        del spec
        return PreparedScenario(initial_inputs=InferenceInput())


class _Provider:
    def __init__(self, capabilities: ProviderCapabilities) -> None:
        self.capabilities = capabilities

    def prepare_initial_input(self) -> InferenceInput:
        return InferenceInput()

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        del request, user_window
        return PreparedStep(inference_input=InferenceInput())

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        return


class _MappingBackedProvider(_Provider):
    def __init__(
        self, *, mapping: "_DriverCommandMapping", source_schema: UserInputSchema
    ) -> None:
        self.mapping = mapping
        self.canonicalizer = InputCanonicalizer((KeyboardToDriverCommand(),))
        self.source_schema = source_schema
        capabilities = ProviderCapabilities(
            supports_recorded_input=True,
            supports_reset=True,
            deterministic_given_inputs=True,
            user_input_schema=source_schema,
            inference_input_schema=InferenceInputSchema(
                step_fields=(InputField(name="driver_command"),),
            ),
        )
        super().__init__(capabilities)
        self.mapping.validate(
            canonical_schema=self.canonicalizer.canonical_schema(source_schema),
            inference_input_schema=capabilities.inference_input_schema,
        )

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        canonical_inputs = self.canonicalizer.canonicalize(
            user_window.inputs,
            window=TimeWindow(start_s=user_window.start_s, end_s=user_window.end_s),
            source_schema=self.source_schema,
        )
        inference_input = self.mapping.map_step_inputs(
            canonical_inputs=canonical_inputs,
            inference_input=InferenceInput(),
            request=StepRequest(step_index=request.step_index),
        )
        return PreparedStep(inference_input=inference_input)


class _DriverCommandMapping:
    mapping_schema = InputMappingSchema(
        name="driver-command",
        consumes=(DRIVER_COMMAND,),
        produces_step=(InputField(name="driver_command"),),
    )

    def __init__(self) -> None:
        self.validated_with: (
            tuple[
                CanonicalInputSchema | None,
                InferenceInputSchema | None,
            ]
            | None
        ) = None

    def validate(
        self,
        *,
        canonical_schema: CanonicalInputSchema | None = None,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        if canonical_schema is not None and not canonical_schema.supports(
            DRIVER_COMMAND
        ):
            raise ValueError("mapping cannot be fed")
        if inference_input_schema is not None:
            inference_input_schema.require_step(
                InferenceInput(step={"driver_command": object()})
            )
        self.validated_with = (canonical_schema, inference_input_schema)

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
        del inference_input
        return InferenceInput(
            step={
                "driver_command": canonical_inputs.values["driver_command"],
                "request_step": request.step_index,
            }
        )


class _RunMode:
    name = "fake"

    def __init__(self, capabilities: RunModeCapabilities) -> None:
        self.capabilities = capabilities

    def validate_run(self, *, spec: DemoSpec, adapter: Any) -> None:
        del spec, adapter

    def validate_session(
        self,
        *,
        spec: DemoSpec,
        scenario: PreparedScenario,
        adapter: Any,
        provider: Any,
    ) -> None:
        del spec, scenario, adapter, provider

    def create_run_context(
        self,
        *,
        spec: DemoSpec,
        adapter: Any,
        host: RuntimeHost,
        model_warmup_plan: Any,
    ) -> RunContext:
        del spec, adapter, model_warmup_plan
        return RunContext(
            host=host,
            run_metrics=demo_api.InMemorySessionMetricsRecorder(),
            admission=demo_api.SingleSessionAdmissionPolicy(),
        )

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: PreparedScenario,
        provider: Any,
        adapter: Any,
    ) -> SessionEdges:
        del context, spec, scenario, provider, adapter
        return _edges()

    def select_driver(self) -> Any:
        raise NotImplementedError


class _BatchInputSource:
    is_finite = True

    def __init__(
        self,
        *,
        user_input_schema: UserInputSchema | None = None,
        deterministic: bool = True,
        num_windows: int = 1,
    ) -> None:
        self.user_input_schema = user_input_schema or UserInputSchema()
        self.is_deterministic = deterministic
        self.num_windows = num_windows
        self.index = 0

    def is_finished(self) -> bool:
        return self.index >= self.num_windows

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        del request
        self.index += 1
        return UserInputWindow(start_s=0.0, end_s=1.0)


class _RealtimeInputSource:
    is_finite = False
    is_deterministic = False
    user_input_schema = UserInputSchema()

    def is_finished(self) -> bool:
        return False

    async def next_realtime_window(
        self,
        *,
        request: StepRequirements,
        clock: Any,
    ) -> RealtimeWindowResult:
        del request, clock
        return RealtimeWindowResult(window=UserInputWindow(start_s=0.0, end_s=1.0))


class _Clock:
    def __init__(self, *, realtime: bool = False, deterministic: bool = True) -> None:
        self.is_realtime = realtime
        self.is_deterministic = deterministic


class _OutputSink:
    def __init__(self, *, produces_artifacts: bool) -> None:
        self.produces_artifacts = produces_artifacts

    def open(self, session_info: SessionInfo) -> None:
        del session_info

    def begin_generation(self, generation: int) -> None:
        del generation

    def write(self, result: StepResult) -> OutputDecision:
        del result
        return OutputDecision()

    def close(self) -> Sequence[OutputArtifact]:
        return ()


class _ResettingProvider(_Provider):
    def __init__(self, *, reset_input: InferenceInput) -> None:
        self.reset_input = reset_input
        self.prepare_count = 0
        self.reset_inputs: list[InferenceInput | None] = []
        super().__init__(
            ProviderCapabilities(
                supports_recorded_input=True,
                supports_reset=True,
                deterministic_given_inputs=True,
            )
        )

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        del request, user_window
        self.prepare_count += 1
        if self.prepare_count == 1:
            return PreparedStep(
                control=demo_api.ControlDecision(
                    reset=True,
                    reset_input=self.reset_input,
                )
            )
        return PreparedStep(inference_input=InferenceInput(step={"after_reset": True}))

    def reset(self, inputs: InferenceInput | None = None) -> None:
        self.reset_inputs.append(inputs)


class _Runtime:
    def __init__(self, *, session: "_ResettableSession") -> None:
        self.session = session

    def start_session(self, inputs: InferenceInput) -> "_ResettableSession":
        del inputs
        return self.session

    def close(self) -> None:
        return


class _ResettableSession:
    def __init__(self, *, num_steps: int) -> None:
        self.num_steps = num_steps
        self.next_request_index = 0
        self.reset_inputs: list[InferenceInput | None] = []
        self.step_inputs: list[InferenceInput] = []

    def session_info(self) -> SessionInfo:
        return SessionInfo()

    def next_step_requirements(self) -> StepRequirements | None:
        if self.next_request_index >= self.num_steps:
            return None
        request = StepRequirements(step_index=self.next_request_index)
        self.next_request_index += 1
        return request

    def next_step_request(self) -> StepRequest | None:
        requirements = self.next_step_requirements()
        if requirements is None:
            return None
        return StepRequest(step_index=requirements.step_index)

    def step(self, inputs: InferenceInput) -> StepResult:
        self.step_inputs.append(inputs)
        return StepResult(step_index=len(self.step_inputs) - 1, output=None)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        self.reset_inputs.append(inputs)
        self.next_request_index = 0

    def close(self) -> None:
        return
