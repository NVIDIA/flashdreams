# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public runner facade for demo applications."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from flashdreams.runtime.canonical import CanonicalInputSchema
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.demo.drivers import (
    run_demo_session,
    run_demo_session_async,
    uncancel_current_task,
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
from flashdreams.runtime.demo.spec import (
    DemoAdapter,
    DemoSpec,
    NullOutputSpec,
    PreparedScenario,
)
from flashdreams.runtime.inputs import (
    InferenceInput,
    InferenceInputSchema,
)
from flashdreams.runtime.interfaces import InferenceRuntime, InferenceSession
from flashdreams.runtime.mapping import InputMapping
from flashdreams.runtime.types import StepRequirements

from .application import (
    Application,
    ApplicationSession,
    DemoAdapterApplication,
    IOHandler,
)
from .io import IOHandlerRunMode, ReplayIOHandler


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
        spec = self._create_spec(run_mode)
        context = self._create_context(host)
        result: RunResult | None = None
        primary_error: BaseException | None = None
        app_initialized = False
        app_cleanup: _ApplicationCleanup | None = None
        remove_app_close_hook: Callable[[], None] | None = None
        try:
            if not host.is_healthy:
                result = RunResult.rejected(reason="busy")
                context.run_metrics.record_session(result)
                return result
            app_cleanup = _ApplicationCleanup(self.app)
            app_initialized = True
            self.app.init(tuple(self.launch_args))
            remove_app_close_hook = host.add_close_hook(app_cleanup.close)
            scenario = self._create_scenario()
            if isinstance(self.app, DemoAdapterApplication):
                _configure_replay_io_handler(self.io_handler, scenario)
            adapter = _RunnerDemoAdapter(app=self.app, spec=spec, scenario=scenario)
            result = await run_demo_session_async(
                context=context,
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                run_mode=run_mode,
                pipeline=self.pipeline or StepPipeline(),
            )
            return result
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            await _await_runner_cleanup(
                _close_runner_resources_async(
                    context=context,
                    host=host,
                    app_cleanup=app_cleanup,
                    app_initialized=app_initialized,
                    remove_app_close_hook=remove_app_close_hook,
                    owns_host=owns_host,
                    run_result=result,
                    primary_error=primary_error,
                ),
                preserve_primary=_has_primary_outcome(
                    run_result=result,
                    primary_error=primary_error,
                ),
                preserved_error=primary_error
                or (None if result is None else result.error),
            )

    def _run_sync(self, *, run_mode: RunMode | None = None) -> RunResult:
        selected_run_mode = run_mode or self._selected_run_mode()
        host, owns_host = self._selected_host()
        spec = self._create_spec(selected_run_mode)
        context = self._create_context(host)
        result: RunResult | None = None
        primary_error: BaseException | None = None
        app_initialized = False
        app_cleanup: _ApplicationCleanup | None = None
        remove_app_close_hook: Callable[[], None] | None = None
        try:
            if not host.is_healthy:
                result = RunResult.rejected(reason="busy")
                context.run_metrics.record_session(result)
                return result
            app_cleanup = _ApplicationCleanup(self.app)
            app_initialized = True
            self.app.init(tuple(self.launch_args))
            remove_app_close_hook = host.add_close_hook(app_cleanup.close)
            scenario = self._create_scenario()
            if isinstance(self.app, DemoAdapterApplication):
                _configure_replay_io_handler(self.io_handler, scenario)
            adapter = _RunnerDemoAdapter(app=self.app, spec=spec, scenario=scenario)
            result = run_demo_session(
                context=context,
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                run_mode=selected_run_mode,
                pipeline=self.pipeline or StepPipeline(),
            )
            return result
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            _close_runner_resources(
                context=context,
                host=host,
                app_cleanup=app_cleanup,
                app_initialized=app_initialized,
                remove_app_close_hook=remove_app_close_hook,
                owns_host=owns_host,
                run_result=result,
                primary_error=primary_error,
            )

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
        if isinstance(self.app, DemoAdapterApplication):
            return self.app.spec
        model_id = self.model_id or _application_model_id(self.app)
        return DemoSpec(
            model_id=model_id,
            input_mode=run_mode.name,
            output=NullOutputSpec(),
            config=InferenceConfig(model_id=model_id),
        )

    def _create_scenario(self) -> PreparedScenario:
        if isinstance(self.app, DemoAdapterApplication):
            scenario = self.app.prepared_scenario
            if scenario is None:
                raise RuntimeError(
                    "DemoAdapterApplication did not prepare a scenario during init."
                )
            return scenario
        return _runner_scenario()


@dataclass(slots=True)
class _ApplicationRuntime:
    app: Application

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        start_session = getattr(self.app, "start_session", None)
        if callable(start_session):
            session = start_session(inputs)
        else:
            session = self.app.create_session()
        if not isinstance(session, ApplicationSession):
            raise TypeError(
                "Application.create_session() must return ApplicationSession, "
                f"got {type(session).__name__}."
            )
        session.init()
        return cast(InferenceSession, session)

    def close(self) -> None:
        return None


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
        adapter = _application_adapter(self.app)
        if adapter is not None:
            return adapter.inference_input_schema
        return InferenceInputSchema()

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema:
        adapter = _application_adapter(self.app)
        if adapter is not None and adapter.canonical_input_schema is not None:
            return adapter.canonical_input_schema
        return CanonicalInputSchema()

    def default_input_mapping(self) -> InputMapping | None:
        adapter = _application_adapter(self.app)
        if adapter is not None:
            default_input_mapping = getattr(adapter, "default_input_mapping", None)
            if callable(default_input_mapping):
                return default_input_mapping()
        return None

    def validate_config(self, config: InferenceConfig) -> None:
        adapter = _application_adapter(self.app)
        if adapter is not None:
            adapter.validate_config(config)
            return
        if config.model_id != self.spec.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return cast(InferenceRuntime, _ApplicationRuntime(self.app))

    def supported_input_modes(self) -> tuple[str, ...]:
        adapter = _application_adapter(self.app)
        if adapter is not None:
            return adapter.supported_input_modes()
        return (self.spec.input_mode,)

    def supported_output_modes(self) -> tuple[str, ...]:
        adapter = _application_adapter(self.app)
        if adapter is not None:
            return adapter.supported_output_modes()
        return (self.spec.output.mode,)

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if spec != self.spec:
            raise ValueError("Runner received an unexpected DemoSpec.")
        return self.scenario

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: Any,
    ) -> object:
        adapter = _application_adapter(self.app)
        if adapter is not None:
            create_provider = getattr(adapter, "create_model_input_provider", None)
            if callable(create_provider):
                return create_provider(spec, scenario)
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
class _ApplicationCleanup:
    app: Application
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.app.close()
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


def _configure_replay_io_handler(
    io_handler: IOHandler,
    scenario: PreparedScenario,
) -> None:
    if isinstance(io_handler, ReplayIOHandler):
        io_handler.configure_replay_inputs(
            replay_log=scenario.user_inputs,
            user_input_schema=scenario.source_schema,
        )


def _close_runner_resources(
    *,
    context: RunContext,
    host: RuntimeHost,
    app_cleanup: _ApplicationCleanup | None,
    app_initialized: bool,
    remove_app_close_hook: Callable[[], None] | None,
    owns_host: bool,
    run_result: RunResult | None,
    primary_error: BaseException | None,
) -> None:
    errors: list[Exception] = []
    _record_cleanup_error(errors, context.close)
    if app_initialized and app_cleanup is not None:
        _close_application(errors=errors, host=host, cleanup=app_cleanup)
    if remove_app_close_hook is not None:
        _record_cleanup_error(errors, remove_app_close_hook)
    if owns_host:
        _record_cleanup_error(errors, host.close)
    if primary_error is not None:
        _record_cleanup_notes(primary_error, errors)
        return
    if run_result is not None and run_result.status != "completed":
        _record_cleanup_notes(run_result.error, errors)
        return
    _raise_first_cleanup_error(errors)


async def _close_runner_resources_async(
    *,
    context: RunContext,
    host: RuntimeHost,
    app_cleanup: _ApplicationCleanup | None,
    app_initialized: bool,
    remove_app_close_hook: Callable[[], None] | None,
    owns_host: bool,
    run_result: RunResult | None,
    primary_error: BaseException | None,
) -> None:
    errors: list[Exception] = []
    try:
        await context.close_async()
    except Exception as exc:
        errors.append(exc)
    if app_initialized and app_cleanup is not None:
        await _close_application_async(errors=errors, host=host, cleanup=app_cleanup)
    if remove_app_close_hook is not None:
        _record_cleanup_error(errors, remove_app_close_hook)
    if owns_host:
        _record_cleanup_error(errors, host.close)
    if primary_error is not None:
        _record_cleanup_notes(primary_error, errors)
        return
    if run_result is not None and run_result.status != "completed":
        _record_cleanup_notes(run_result.error, errors)
        return
    _raise_first_cleanup_error(errors)


async def _await_runner_cleanup(
    cleanup: Coroutine[Any, Any, None],
    *,
    preserve_primary: bool = False,
    preserved_error: BaseException | None = None,
) -> None:
    cleanup_task = asyncio.create_task(cleanup)
    was_cancelled = False
    cleanup_error: BaseException | None = None

    while True:
        try:
            await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as exc:
            if cleanup_task.done():
                cleanup_error = exc
                break
            was_cancelled = True
            uncancel_current_task()
        except Exception as exc:
            cleanup_error = exc
            break

    if was_cancelled:
        if preserve_primary:
            _record_cleanup_notes_for_preserved_outcome(preserved_error, cleanup_error)
            return
        cancellation = asyncio.CancelledError("cancelled during runner cleanup")
        _record_cancelled_cleanup_note(cancellation, cleanup_error)
        raise cancellation from None
    if cleanup_error is not None:
        if preserve_primary:
            _record_cleanup_notes_for_preserved_outcome(preserved_error, cleanup_error)
            return
        raise cleanup_error


def _record_cleanup_error(
    errors: list[Exception],
    cleanup: Any,
    /,
    *args: Any,
) -> None:
    try:
        cleanup(*args)
    except Exception as exc:
        errors.append(exc)


def _close_application(
    *,
    errors: list[Exception],
    host: RuntimeHost,
    cleanup: _ApplicationCleanup,
) -> None:
    if cleanup.closed:
        return
    invoked = False

    def close_app() -> None:
        nonlocal invoked
        invoked = True
        cleanup.close()

    try:
        host.call(close_app)
    except Exception as exc:
        if cleanup.closed:
            return
        if not invoked and host.is_closed:
            _close_application_after_closed_host(
                errors=errors,
                cleanup=cleanup,
                host_error=exc,
            )
            return
        errors.append(exc)


async def _close_application_async(
    *,
    errors: list[Exception],
    host: RuntimeHost,
    cleanup: _ApplicationCleanup,
) -> None:
    if cleanup.closed:
        return
    invoked = False

    def close_app() -> None:
        nonlocal invoked
        invoked = True
        cleanup.close()

    try:
        await host.call_async(close_app)
    except Exception as exc:
        if cleanup.closed:
            return
        if not invoked and host.is_closed:
            _close_application_after_closed_host(
                errors=errors,
                cleanup=cleanup,
                host_error=exc,
            )
            return
        errors.append(exc)


def _close_application_after_closed_host(
    *,
    errors: list[Exception],
    cleanup: _ApplicationCleanup,
    host_error: Exception,
) -> None:
    # An externally owned host may already be torn down by the time runner
    # cleanup runs. At that point worker dispatch is impossible, so the
    # idempotent app cleanup is the last leak-prevention fallback.
    fallback_errors: list[Exception] = []
    _record_cleanup_error(fallback_errors, cleanup.close)
    if cleanup.closed:
        return
    cleanup_error = _closed_host_cleanup_error(host_error)
    _record_cleanup_notes(cleanup_error, fallback_errors)
    errors.append(cleanup_error)


def _raise_first_cleanup_error(errors: Sequence[Exception]) -> None:
    if not errors:
        return
    first = errors[0]
    _record_cleanup_notes(first, errors[1:])
    raise first


def _record_cleanup_notes(
    primary: BaseException | None,
    errors: Sequence[BaseException],
) -> None:
    if primary is None:
        return
    add_note = getattr(primary, "add_note", None)
    for extra in errors:
        if callable(add_note):
            add_note(f"Additional cleanup error: {extra!r}")


def _record_cancelled_cleanup_note(
    cancellation: asyncio.CancelledError,
    cleanup_error: BaseException | None,
) -> None:
    if cleanup_error is None:
        return
    add_note = getattr(cancellation, "add_note", None)
    if callable(add_note):
        add_note(f"Cleanup failed: {cleanup_error!r}")


def _record_cleanup_notes_for_preserved_outcome(
    preserved_error: BaseException | None,
    cleanup_error: BaseException | None,
) -> None:
    if cleanup_error is not None:
        _record_cleanup_notes(preserved_error, (cleanup_error,))


def _has_primary_outcome(
    *,
    run_result: RunResult | None,
    primary_error: BaseException | None,
) -> bool:
    if primary_error is not None:
        return True
    return run_result is not None and run_result.status != "completed"


def _closed_host_cleanup_error(exc: Exception) -> RuntimeError:
    try:
        raise RuntimeError(
            "Application cleanup could not be dispatched because the RuntimeHost "
            "is closed."
        ) from exc
    except RuntimeError as cleanup_error:
        return cleanup_error


def _application_adapter(app: Application) -> DemoAdapter | None:
    if isinstance(app, DemoAdapterApplication):
        return app.adapter
    return None


__all__ = ["Runner"]
