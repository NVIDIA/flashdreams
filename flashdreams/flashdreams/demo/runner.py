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
    run_demo_session,
    run_demo_session_async,
)
from flashdreams.runtime.demo.host import ModelWarmupPlan, RuntimeHost
from flashdreams.runtime.demo.pipeline import StepPipeline
from flashdreams.runtime.demo.run_modes import (
    InMemorySessionMetricsRecorder,
    RunContext,
    RunMode,
    RunResult,
    SessionMetricsRecorder,
    SingleSessionAdmissionPolicy,
)
from flashdreams.runtime.demo.session_inputs import (
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from flashdreams.runtime.demo.spec import DemoSpec, NullOutputSpec, PreparedScenario
from flashdreams.runtime.inputs import (
    InferenceInput,
    InferenceInputSchema,
)
from flashdreams.runtime.interfaces import InferenceRuntime, InferenceSession
from flashdreams.runtime.mapping import InputMapping
from flashdreams.runtime.types import StepRequirements

from .application import Application, ApplicationSession, IOHandler
from .io import IOHandlerRunMode


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
        run_mode = getattr(self.io_handler, "run_mode", None)
        if run_mode is not None:
            return cast(RunMode, run_mode)
        return IOHandlerRunMode(self.io_handler)

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
