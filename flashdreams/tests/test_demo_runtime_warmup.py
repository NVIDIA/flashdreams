# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import Any

import pytest

from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceRuntime,
    InferenceSession,
    InputMapping,
    StepRequest,
    StepRequirements,
    StepResult,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    InMemorySessionMetricsRecorder,
    ModelWarmupPlan,
    NullOutputSpec,
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    RunContext,
    RuntimeHost,
    UserInputWindow,
    WarmupSessionInputs,
    build_model_warmup_plan,
    warmup_run_context,
)

pytestmark = pytest.mark.ci_cpu


def test_model_warmup_plan_uses_temporary_provider_on_worker_thread() -> None:
    setup_thread_id = threading.get_ident()
    runtime = _WarmupRuntime()
    host = RuntimeHost(runtime)
    adapter = _WarmupAdapter(warmup_steps=2)
    spec = _spec()
    scenario = adapter.prepare_scenario(spec)

    try:
        plan = build_model_warmup_plan(
            host=host,
            adapter=adapter,
            spec=spec,
            scenario=scenario,
        )
        real_provider = host.call(adapter.create_model_input_provider, spec, scenario)
    finally:
        host.close()

    warmup_provider = adapter.warmup_providers[0]
    assert adapter.warmup_thread_id == host.worker.worker_thread_id
    assert adapter.warmup_thread_id != setup_thread_id
    assert warmup_provider is not real_provider
    assert warmup_provider.close_count == 1
    assert real_provider.close_count == 0
    assert plan == ModelWarmupPlan(
        sessions=(
            WarmupSessionInputs(
                initial_input=InferenceInput(
                    global_conditioning={"provider": "warmup"}
                ),
                step_inputs=(
                    InferenceInput(step={"provider": "warmup", "step": 0}),
                    InferenceInput(step={"provider": "warmup", "step": 1}),
                ),
            ),
        ),
    )


def test_runtime_host_warmup_uses_runtime_session_api() -> None:
    runtime = _WarmupRuntime()
    host = RuntimeHost(runtime)
    adapter = _WarmupAdapter(warmup_steps=2)
    spec = _spec()
    scenario = adapter.prepare_scenario(spec)

    try:
        plan = build_model_warmup_plan(
            host=host,
            adapter=adapter,
            spec=spec,
            scenario=scenario,
        )
        host.warmup(plan)
    finally:
        host.close()

    assert runtime.events[:4] == [
        (
            "start_session",
            InferenceInput(global_conditioning={"provider": "warmup"}),
        ),
        ("step", InferenceInput(step={"provider": "warmup", "step": 0})),
        ("step", InferenceInput(step={"provider": "warmup", "step": 1})),
        "session.close",
    ]


def test_run_mode_warmup_context_warms_transport_without_model_session() -> None:
    runtime = _WarmupRuntime()
    host = RuntimeHost(runtime)
    adapter = _WarmupAdapter(warmup_steps=0)
    spec = _spec()
    scenario = adapter.prepare_scenario(spec)
    transport = _TransportWarmupService()
    mode = _TransportWarmupRunMode()
    context = RunContext(
        host=host,
        run_metrics=InMemorySessionMetricsRecorder(),
        admission=_Admission(),
        model_warmup_plan=ModelWarmupPlan(),
        services={"transport": transport},
    )

    try:
        warmup_run_context(
            context=context,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            run_mode=mode,
        )

        assert runtime.events == []
        assert mode.warmup_calls == 1
        assert transport.warmup_calls == 1
    finally:
        host.close()


def test_model_warmup_is_excluded_from_run_metrics() -> None:
    runtime = _WarmupRuntime()
    host = RuntimeHost(runtime)
    adapter = _WarmupAdapter(warmup_steps=1)
    spec = _spec()
    scenario = adapter.prepare_scenario(spec)
    metrics = InMemorySessionMetricsRecorder()

    try:
        plan = build_model_warmup_plan(
            host=host,
            adapter=adapter,
            spec=spec,
            scenario=scenario,
        )
        context = RunContext(
            host=host,
            run_metrics=metrics,
            admission=_Admission(),
            model_warmup_plan=plan,
        )

        warmup_run_context(
            context=context,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            run_mode=object(),
        )

        assert runtime.events[:3] == [
            (
                "start_session",
                InferenceInput(global_conditioning={"provider": "warmup"}),
            ),
            ("step", InferenceInput(step={"provider": "warmup", "step": 0})),
            "session.close",
        ]
        assert metrics.sessions == []
        assert metrics.step_count == 0
        assert metrics.control_count == 0
    finally:
        host.close()


def test_adapter_without_model_warmup_hook_gets_empty_plan() -> None:
    host = RuntimeHost(_WarmupRuntime())
    adapter = _NoWarmupAdapter()
    spec = _spec()
    scenario = adapter.prepare_scenario(spec)

    try:
        plan = build_model_warmup_plan(
            host=host,
            adapter=adapter,
            spec=spec,
            scenario=scenario,
        )
    finally:
        host.close()

    assert plan == ModelWarmupPlan()


def _spec() -> DemoSpec:
    return DemoSpec(
        model_id="fake-demo",
        input_mode="replay",
        output=NullOutputSpec(),
    )


def _scenario() -> PreparedScenario:
    return PreparedScenario(initial_inputs=InferenceInput())


class _WarmupAdapter:
    model_id = "fake-demo"
    inference_input_schema = InferenceInputSchema()
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self, *, warmup_steps: int) -> None:
        self.warmup_steps = warmup_steps
        self.warmup_thread_id: int | None = None
        self.warmup_providers: list[_WarmupProvider] = []
        self.real_providers: list[_WarmupProvider] = []

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("null",)

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return _WarmupRuntime()

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        del spec
        return _scenario()

    def create_model_warmup_sessions(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> tuple[WarmupSessionInputs, ...]:
        del spec, scenario
        self.warmup_thread_id = threading.get_ident()
        provider = _WarmupProvider(name="warmup")
        self.warmup_providers.append(provider)
        try:
            initial_input = provider.prepare_initial_input()
            step_inputs = []
            for step_index in range(self.warmup_steps):
                prepared = provider.prepare_step(
                    request=StepRequirements(step_index=step_index),
                    user_window=UserInputWindow(
                        start_s=float(step_index),
                        end_s=float(step_index + 1),
                    ),
                )
                if prepared.inference_input is None:
                    raise RuntimeError("Warmup provider returned no step input.")
                step_inputs.append(prepared.inference_input)
            return (
                WarmupSessionInputs(
                    initial_input=initial_input,
                    step_inputs=tuple(step_inputs),
                ),
            )
        finally:
            provider.close()

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> "_WarmupProvider":
        del spec, scenario
        provider = _WarmupProvider(name="real")
        self.real_providers.append(provider)
        return provider


class _NoWarmupAdapter:
    model_id = "fake-demo"
    inference_input_schema = InferenceInputSchema()
    canonical_input_schema = CanonicalInputSchema()

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("null",)

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return _WarmupRuntime()

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        del spec
        return _scenario()


class _WarmupProvider:
    capabilities = ProviderCapabilities(supports_recorded_input=True)

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.close_count = 0

    def prepare_initial_input(self) -> InferenceInput:
        return InferenceInput(global_conditioning={"provider": self.name})

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        del user_window
        return PreparedStep(
            inference_input=InferenceInput(
                step={"provider": self.name, "step": request.step_index}
            )
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        self.close_count += 1


class _WarmupRuntime:
    def __init__(self) -> None:
        self.events: list[object] = []

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        self.events.append(("start_session", inputs))
        return _WarmupSession(events=self.events)

    def close(self) -> None:
        self.events.append("runtime.close")


class _WarmupSession:
    def __init__(self, *, events: list[object]) -> None:
        self.events = events
        self.next_step = 0

    def next_step_request(self) -> StepRequest | None:
        request = StepRequest(step_index=self.next_step)
        self.next_step += 1
        return request

    def step(self, inputs: InferenceInput) -> StepResult:
        self.events.append(("step", inputs))
        return StepResult(step_index=self.next_step, output=None)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self.next_step = 0

    def close(self) -> None:
        self.events.append("session.close")


class _TransportWarmupService:
    def __init__(self) -> None:
        self.warmup_calls = 0

    def warmup(self) -> None:
        self.warmup_calls += 1


class _TransportWarmupRunMode:
    def __init__(self) -> None:
        self.warmup_calls = 0

    def warmup_context(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: PreparedScenario,
        adapter: Any,
    ) -> None:
        del spec, scenario, adapter
        transport = context.services["transport"]
        if not isinstance(transport, _TransportWarmupService):
            raise TypeError("Expected fake transport warmup service.")
        transport.warmup()
        self.warmup_calls += 1


class _Admission:
    def try_reserve(self) -> None:
        return None
