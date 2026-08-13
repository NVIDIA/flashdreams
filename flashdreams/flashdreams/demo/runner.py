# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public runner facade for demo applications."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from flashdreams.runtime.canonical import CanonicalInputSchema
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.demo.drivers import (
    BatchSessionDriver,
    run_demo_session,
    run_demo_session_async,
)
from flashdreams.runtime.demo.host import ModelWarmupPlan, RuntimeHost
from flashdreams.runtime.demo.outputs import OutputDecision, SessionInfo
from flashdreams.runtime.demo.pipeline import StepPipeline
from flashdreams.runtime.demo.run_modes import (
    AsyncSessionDriver,
    InMemorySessionMetricsRecorder,
    NoopTransportService,
    RunContext,
    RunMode,
    RunModeCapabilities,
    RunResult,
    SessionDriver,
    SessionEdges,
    SessionMetricsRecorder,
    SingleSessionAdmissionPolicy,
)
from flashdreams.runtime.demo.session_inputs import (
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from flashdreams.runtime.demo.spec import (
    DemoAdapter,
    DemoSpec,
    NullOutputSpec,
    PreparedScenario,
)
from flashdreams.runtime.inputs import (
    InferenceInput,
    InferenceInputSchema,
    UserInputSchema,
)
from flashdreams.runtime.interfaces import InferenceRuntime, InferenceSession
from flashdreams.runtime.mapping import InputMapping
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepRequirements, StepResult

from .application import Application, ApplicationSession, IOHandler


@dataclass(slots=True)
class Runner:
    """Run a public demo application through the shared demo runtime.

    This facade exists for application authors. It adapts ``Application`` and
    ``IOHandler`` to the existing runtime ``run_demo_session`` helpers, so the
    model worker boundary, ``StepPipeline``, metrics, output decisions, and
    cleanup behavior stay in the runtime implementation.
    """

    io_handler: IOHandler
    app: Application
    launch_args: Sequence[str] = ()
    host: RuntimeHost | None = None
    metrics: SessionMetricsRecorder | None = None
    pipeline: StepPipeline | None = None
    run_mode: RunMode | None = None
    model_id: str | None = None

    def run(self) -> RunResult:
        """Run one session and return its shared runtime result."""
        if _run_mode_is_async(self._selected_run_mode()):
            return asyncio.run(self.run_async())
        return self._run_sync()

    async def run_async(self) -> RunResult:
        """Run one session through the async helper when the run mode needs it."""
        run_mode = self._selected_run_mode()
        if not _run_mode_is_async(run_mode):
            return self._run_sync(run_mode=run_mode)

        host, owns_host = self._selected_host()
        context = self._create_context(host)
        spec = self._create_spec(run_mode)
        scenario = _runner_scenario()
        adapter = _RunnerDemoAdapter(app=self.app, spec=spec, scenario=scenario)
        try:
            self.app.init(tuple(self.launch_args))
            return await run_demo_session_async(
                context=context,
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                run_mode=run_mode,
                pipeline=self.pipeline or StepPipeline(),
            )
        finally:
            await context.close_async()
            if owns_host:
                host.close()

    def _run_sync(self, *, run_mode: RunMode | None = None) -> RunResult:
        selected_run_mode = run_mode or self._selected_run_mode()
        host, owns_host = self._selected_host()
        context = self._create_context(host)
        spec = self._create_spec(selected_run_mode)
        scenario = _runner_scenario()
        adapter = _RunnerDemoAdapter(app=self.app, spec=spec, scenario=scenario)
        try:
            self.app.init(tuple(self.launch_args))
            return run_demo_session(
                context=context,
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                run_mode=selected_run_mode,
                pipeline=self.pipeline or StepPipeline(),
            )
        finally:
            context.close()
            if owns_host:
                host.close()

    def _selected_run_mode(self) -> RunMode:
        if self.run_mode is not None:
            return self.run_mode
        return _IOHandlerRunMode(self.io_handler)

    def _selected_host(self) -> tuple[RuntimeHost, bool]:
        if self.host is not None:
            return self.host, False
        return RuntimeHost(_ApplicationRuntime(self.app)), True

    def _create_context(self, host: RuntimeHost) -> RunContext:
        return RunContext(
            host=host,
            run_metrics=self.metrics or InMemorySessionMetricsRecorder(),
            admission=SingleSessionAdmissionPolicy(
                health_check=lambda: host.is_healthy,
            ),
            model_warmup_plan=ModelWarmupPlan(),
        )

    def _create_spec(self, run_mode: RunMode) -> DemoSpec:
        model_id = self.model_id or _application_model_id(self.app)
        return DemoSpec(
            model_id=model_id,
            input_mode=run_mode.name,
            output=NullOutputSpec(),
            config=InferenceConfig(model_id=model_id),
        )


@dataclass(slots=True)
class _ApplicationRuntime:
    app: Application

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        del inputs
        session = self.app.create_session()
        if not isinstance(session, ApplicationSession):
            raise TypeError(
                "Application.create_session() must return ApplicationSession, "
                f"got {type(session).__name__}."
            )
        session.init()
        return cast(InferenceSession, session)

    def close(self) -> None:
        self.app.close()


@dataclass(slots=True)
class _RunnerDemoAdapter:
    app: Application
    spec: DemoSpec
    scenario: PreparedScenario

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def inference_input_schema(self) -> InferenceInputSchema:
        return InferenceInputSchema()

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema:
        return CanonicalInputSchema()

    def default_input_mapping(self) -> InputMapping | None:
        return None

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.spec.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return cast(InferenceRuntime, _ApplicationRuntime(self.app))

    def supported_input_modes(self) -> tuple[str, ...]:
        return (self.spec.input_mode,)

    def supported_output_modes(self) -> tuple[str, ...]:
        return (self.spec.output.mode,)

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if spec != self.spec:
            raise ValueError("Runner received an unexpected DemoSpec.")
        return self.scenario

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: Any,
    ) -> "_RunnerModelInputProvider":
        del spec, scenario
        return _RunnerModelInputProvider()


@dataclass(slots=True)
class _RunnerModelInputProvider:
    capabilities: ProviderCapabilities = field(
        default_factory=lambda: ProviderCapabilities(
            supports_recorded_input=True,
            inference_input_schema=InferenceInputSchema(),
        )
    )
    closed: bool = False

    def prepare_initial_input(self) -> InferenceInput:
        return InferenceInput()

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        return PreparedStep(
            inference_input=InferenceInput(
                step={
                    "step_index": request.step_index,
                    "user_window": user_window,
                },
                metadata=request.metadata,
            )
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _IOHandlerRunMode:
    io_handler: IOHandler
    driver: SessionDriver | AsyncSessionDriver = field(
        default_factory=BatchSessionDriver
    )
    name: str = "public-runner"
    capabilities: RunModeCapabilities = field(
        default_factory=lambda: RunModeCapabilities(supports_artifacts=True)
    )

    def validate_run(self, *, spec: DemoSpec, adapter: DemoAdapter) -> None:
        del spec, adapter

    def validate_session(
        self,
        *,
        spec: DemoSpec,
        scenario: Any,
        adapter: DemoAdapter,
        provider: Any,
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
            run_metrics=InMemorySessionMetricsRecorder(),
            admission=SingleSessionAdmissionPolicy(
                health_check=lambda: host.is_healthy,
            ),
            model_warmup_plan=model_warmup_plan,
        )

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: Any,
        provider: Any,
        adapter: DemoAdapter,
    ) -> SessionEdges:
        del spec, scenario, provider, adapter
        return SessionEdges(
            input_source=_IOHandlerBatchInputSource(self.io_handler),
            output_sink=_IOHandlerOutputSink(self.io_handler),
            cleanup_tasks=context.cleanup_tasks,
            transport=NoopTransportService(),
        )

    def select_driver(self) -> SessionDriver | AsyncSessionDriver:
        return self.driver


@dataclass(slots=True)
class _IOHandlerBatchInputSource:
    io_handler: IOHandler
    is_finite: bool = False
    is_deterministic: bool = False
    user_input_schema: UserInputSchema = field(default_factory=UserInputSchema)

    def is_finished(self) -> bool:
        return self.io_handler.should_exit()

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        return self.io_handler.next_window(request)


@dataclass(slots=True)
class _IOHandlerOutputSink:
    io_handler: IOHandler
    produces_artifacts: bool = True
    _generation_started: bool = field(default=False, init=False, repr=False)

    def open(self, session_info: SessionInfo) -> None:
        self.io_handler.open(session_info)

    def begin_generation(self, generation: int) -> None:
        self._generation_started = True
        self.io_handler.begin_generation(generation)

    def write(self, result: StepResult) -> OutputDecision:
        if not self._generation_started:
            self.begin_generation(0)
        return self.io_handler.emit_chunk(result)

    def close(self) -> Sequence[OutputArtifact]:
        return self.io_handler.close()


def _run_mode_is_async(run_mode: RunMode) -> bool:
    driver = run_mode.select_driver()
    return inspect.iscoroutinefunction(driver.run_one_session)


def _application_model_id(app: Application) -> str:
    model_id = getattr(app, "model_id", None)
    if isinstance(model_id, str) and model_id.strip():
        return model_id
    return app.__class__.__name__ or "demo-application"


def _runner_scenario() -> PreparedScenario:
    return PreparedScenario(initial_inputs=InferenceInput())


__all__ = ["Runner"]
