# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for application adapters over the shared demo runtime."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, ClassVar

import pytest

from flashdreams.demo import (
    CallableIOFactory,
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    LocalWindowIOFactory,
    Mp4IOFactory,
    NullInputHandler,
    NullIOFactory,
    NullOutputSink,
    OutputDecision,
    SessionInfo,
)
from flashdreams.demo.bridge import (
    _CANONICAL_INPUT_WINDOW_KEY,
    ApplicationCanonicalInputProvider,
    ApplicationDemoAdapter,
    IOFactoryInputSource,
    IOFactoryRunMode,
    OrderedIOFactoryEdges,
    application_demo_spec,
    application_scenario,
    output_spec_for,
)
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import (
    CAMERA_COMMAND,
    DRIVER_COMMAND,
    CanonicalInputSchema,
    CanonicalInputWindow,
    CanonicalModality,
    InferenceInput,
    InputCanonicalizer,
    StepRequirements,
    StepResult,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.canonical import DeviceConverterSchema
from flashdreams.runtime.demo import (
    IOFactoryOutputSpec,
    LocalWindowOutputSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    RuntimeHost,
    StepPipeline,
    build_model_warmup_plan,
    run_demo_session,
)
from flashdreams.runtime.output import OutputArtifact

pytestmark = pytest.mark.ci_cpu

_CONTROL = CanonicalModality(
    name="control",
    payload_fields=frozenset({"amount"}),
)


class _RecordingApplicationSession(IFlashDreamsApplicationSession):
    def __init__(self, app: _RecordingApplication) -> None:
        self.app = app
        self.step_index = 0
        self.closed = False

    def init(self) -> None:
        self.app.events.append("session.init")

    def session_info(self) -> SessionInfo:
        return SessionInfo()

    def next_step_requirements(self) -> StepRequirements | None:
        if self.step_index >= self.app.step_count:
            return None
        return StepRequirements(step_index=self.step_index)

    def step(self, inputs: CanonicalInputWindow) -> StepResult:
        self.app.windows.append(inputs)
        current = self.step_index
        self.step_index += 1
        if self.app.fail_step == current:
            raise RuntimeError("step failed")
        return StepResult(step_index=current, output=object())

    def close(self) -> None:
        self.closed = True
        self.app.events.append("session.close")


class _RecordingApplication(IFlashDreamsApplication):
    def __init__(
        self,
        *,
        input_schema: CanonicalInputSchema | None = None,
        step_count: int = 1,
        fail_step: int | None = None,
    ) -> None:
        self._input_schema = input_schema or CanonicalInputSchema()
        self.step_count = step_count
        self.fail_step = fail_step
        self.events: list[str] = []
        self.windows: list[CanonicalInputWindow] = []
        self.session: _RecordingApplicationSession | None = None

    @property
    def input_schema(self) -> CanonicalInputSchema:
        return self._input_schema

    def init(self, commandline_args: tuple[str, ...] | list[str]) -> None:
        self.events.append(f"app.init:{tuple(commandline_args)!r}")

    def create_session(self) -> IFlashDreamsApplicationSession:
        self.session = _RecordingApplicationSession(self)
        return self.session


class _ResettableApplicationSession(_RecordingApplicationSession):
    def reset(self) -> None:
        self.step_index = 0
        self.app.events.append(f"session.reset:{threading.get_ident()}")


class _ResettableApplication(_RecordingApplication):
    @property
    def supports_session_reset(self) -> bool:
        return True

    def create_session(self) -> IFlashDreamsApplicationSession:
        self.session = _ResettableApplicationSession(self)
        return self.session


class _BlockingResettableApplicationSession(_ResettableApplicationSession):
    def step(self, inputs: CanonicalInputWindow) -> StepResult:
        app = self.app
        assert isinstance(app, _BlockingResettableApplication)
        app.step_in_flight = True
        app.step_started.set()
        try:
            assert app.release_step.wait(timeout=2.0)
            return super().step(inputs)
        finally:
            app.step_in_flight = False

    def reset(self) -> None:
        app = self.app
        assert isinstance(app, _BlockingResettableApplication)
        assert not app.step_in_flight
        app.reset_started.set()
        super().reset()


class _BlockingResettableApplication(_ResettableApplication):
    def __init__(self) -> None:
        super().__init__()
        self.step_started = threading.Event()
        self.release_step = threading.Event()
        self.reset_started = threading.Event()
        self.step_in_flight = False

    def create_session(self) -> IFlashDreamsApplicationSession:
        self.session = _BlockingResettableApplicationSession(self)
        return self.session


class _RecordingHandler:
    is_finite = True
    is_deterministic = True

    def __init__(self, windows: list[CanonicalInputWindow]) -> None:
        self.windows = windows
        self.index = 0
        self.events: list[str] = []

    def open(self, session_info: SessionInfo) -> None:
        del session_info
        self.events.append("input.open")

    def current_inputs(self) -> CanonicalInputWindow:
        value = self.windows[min(self.index, len(self.windows) - 1)]
        self.index += 1
        return value

    def close(self) -> None:
        self.events.append("input.close")


class _RecordingSink:
    produces_artifacts = True

    def __init__(
        self,
        *,
        fail_open: bool = False,
        fail_write: bool = False,
    ) -> None:
        self.fail_open = fail_open
        self.fail_write = fail_write
        self.events: list[str] = []
        self.artifact = OutputArtifact(kind="test", uri="memory://artifact")

    def open(self, session_info: SessionInfo) -> None:
        del session_info
        self.events.append("output.open")
        if self.fail_open:
            raise RuntimeError("output open failed")

    def begin_generation(self, generation: int) -> None:
        self.events.append(f"output.generation:{generation}")

    def write(self, result: StepResult) -> OutputDecision:
        self.events.append(f"output.write:{result.step_index}")
        if self.fail_write:
            raise RuntimeError("output write failed")
        return OutputDecision()

    def close(self) -> tuple[OutputArtifact, ...]:
        self.events.append("output.close")
        return (self.artifact,)


def _empty_window(start_s: float = 0.0, end_s: float = 1.0) -> CanonicalInputWindow:
    return CanonicalInputWindow(window=TimeWindow(start_s=start_s, end_s=end_s))


def _run_bridge(
    app: _RecordingApplication,
    handler: _RecordingHandler,
    sink: _RecordingSink | NullOutputSink,
):
    spec = application_demo_spec(
        app=app,
        application_slug="recording-app",
        output=IOFactoryOutputSpec(),
    )
    scenario = application_scenario(app, realtime=False)
    adapter = ApplicationDemoAdapter(app=app, spec=spec, scenario=scenario)
    mode = IOFactoryRunMode(input_handler=handler, output_sink=sink)
    config = spec.config
    assert config is not None
    host = RuntimeHost(adapter.create_runtime(config))
    plan = build_model_warmup_plan(
        host=host,
        adapter=adapter,
        spec=spec,
        scenario=scenario,
    )
    context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=host,
        model_warmup_plan=plan,
    )
    return spec, scenario, adapter, mode, host, context


def test_application_run_records_completed_session_metrics() -> None:
    app = _RecordingApplication(step_count=2)
    handler = _RecordingHandler([_empty_window(), _empty_window(1.0, 2.0)])
    sink = _RecordingSink()
    spec, scenario, adapter, mode, host, context = _run_bridge(app, handler, sink)
    try:
        result = run_demo_session(
            context=context,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            run_mode=mode,
            pipeline=StepPipeline(),
        )
        summary = context.close()
    finally:
        host.close()

    assert result.status == "completed"
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 2
    assert [session.status for session in summary.sessions] == ["completed"]
    assert result.artifacts == (sink.artifact,)
    assert handler.events == ["input.open", "input.close"]
    assert sink.events == [
        "output.open",
        "output.generation:0",
        "output.write:0",
        "output.write:1",
        "output.close",
    ]


def test_application_reset_delegates_on_the_model_worker() -> None:
    app = _ResettableApplication()
    spec = application_demo_spec(
        app=app,
        application_slug="resettable-app",
        output=IOFactoryOutputSpec(),
    )
    scenario = application_scenario(app, realtime=False)
    adapter = ApplicationDemoAdapter(app=app, spec=spec, scenario=scenario)
    provider = adapter.create_model_input_provider(spec, scenario)
    assert isinstance(provider, ApplicationCanonicalInputProvider)
    config = spec.config
    assert config is not None
    host = RuntimeHost(adapter.create_runtime(config))
    try:
        host.preload()
        session = host.call(host.start_session, InferenceInput())
        host.call(session.reset)
        host.call(session.close)

        assert provider.capabilities.supports_reset
        reset_event = next(
            event for event in app.events if event.startswith("session.reset:")
        )
        assert int(reset_event.rsplit(":", 1)[1]) == host.worker.worker_thread_id
        assert host.worker.worker_thread_id != threading.get_ident()
    finally:
        host.close()


@pytest.mark.asyncio
async def test_application_reset_waits_for_an_in_flight_step() -> None:
    app = _BlockingResettableApplication()
    spec = application_demo_spec(
        app=app,
        application_slug="blocking-resettable-app",
        output=IOFactoryOutputSpec(),
    )
    scenario = application_scenario(app, realtime=False)
    adapter = ApplicationDemoAdapter(app=app, spec=spec, scenario=scenario)
    config = spec.config
    assert config is not None
    host = RuntimeHost(adapter.create_runtime(config))
    try:
        host.preload()
        session = await host.call_async(host.start_session, InferenceInput())
        step_input = InferenceInput(step={_CANONICAL_INPUT_WINDOW_KEY: _empty_window()})
        step_task = asyncio.create_task(host.call_async(session.step, step_input))
        for _ in range(100):
            if app.step_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert app.step_started.is_set()

        reset_task = asyncio.create_task(host.call_async(session.reset))
        await asyncio.sleep(0.05)
        assert not app.reset_started.is_set()

        app.release_step.set()
        await step_task
        await reset_task
        await host.call_async(session.close)
    finally:
        app.release_step.set()
        host.close()

    assert app.reset_started.is_set()


def test_application_run_rejects_a_second_reservation_as_busy() -> None:
    app = _RecordingApplication()
    parts = _run_bridge(app, _RecordingHandler([_empty_window()]), NullOutputSink())
    spec, scenario, adapter, mode, host, context = parts
    reservation = context.admission.try_reserve()
    assert reservation is not None
    try:
        result = run_demo_session(
            context=context,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            run_mode=mode,
            pipeline=StepPipeline(),
        )
        assert result.status == "rejected"
        assert result.reason == "busy"
    finally:
        reservation.release()
        context.close()
        host.close()


def test_canonical_handler_value_reaches_application_with_driver_window() -> None:
    carried = CanonicalInputWindow(
        values={"control": {"amount": 0.75}},
        window=TimeWindow(start_s=50.0, end_s=60.0),
    )
    provider = ApplicationCanonicalInputProvider(
        input_schema=CanonicalInputSchema(modalities=(_CONTROL,)),
        canonicalizer=InputCanonicalizer(),
        source_schema=UserInputSchema(),
    )

    prepared = provider.prepare_step(
        request=StepRequirements(step_index=0),
        user_window=_user_window_with_carrier(carried, start_s=1.0, end_s=3.0),
    )

    assert prepared.inference_input is not None
    window = prepared.inference_input.step[_CANONICAL_INPUT_WINDOW_KEY]
    assert isinstance(window, CanonicalInputWindow)
    assert window.values == {"control": {"amount": 0.75}}
    assert window.window == TimeWindow(start_s=1.0, end_s=3.0)


def test_canonical_handler_value_reaches_application_step() -> None:
    app = _RecordingApplication(
        input_schema=CanonicalInputSchema(modalities=(_CONTROL,))
    )
    handler = _RecordingHandler(
        [
            CanonicalInputWindow(
                values={"control": {"amount": 0.75}},
                window=TimeWindow(start_s=1.0, end_s=3.0),
            )
        ]
    )
    spec, scenario, adapter, mode, host, context = _run_bridge(
        app,
        handler,
        NullOutputSink(),
    )
    try:
        result = run_demo_session(
            context=context,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            run_mode=mode,
            pipeline=StepPipeline(),
        )
    finally:
        context.close()
        host.close()

    assert result.status == "completed"
    assert len(app.windows) == 1
    assert app.windows[0].values == {"control": {"amount": 0.75}}
    assert app.windows[0].window == TimeWindow(start_s=1.0, end_s=3.0)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({}, "missing requested modalities"),
        (
            {
                "control": {"amount": 0.5},
                "undeclared": {"amount": 1.0},
            },
            "undeclared modalities",
        ),
    ],
)
def test_canonical_handler_values_are_validated_against_application_schema(
    values: dict[str, dict[str, float]],
    message: str,
) -> None:
    provider = ApplicationCanonicalInputProvider(
        input_schema=CanonicalInputSchema(modalities=(_CONTROL,)),
        canonicalizer=InputCanonicalizer(),
        source_schema=UserInputSchema(),
    )
    carried = CanonicalInputWindow(
        values=values,
        window=TimeWindow(start_s=0.0, end_s=1.0),
    )

    with pytest.raises(ValueError, match=message):
        provider.prepare_step(
            request=StepRequirements(step_index=0),
            user_window=_user_window_with_carrier(
                carried,
                start_s=0.0,
                end_s=1.0,
            ),
        )


class _WindowDurationConverter:
    schema: ClassVar[DeviceConverterSchema] = DeviceConverterSchema(
        name="window-duration",
        produces=_CONTROL,
        consumes=(
            UserInputCapability(
                event_type="key_down",
                payload_fields=frozenset({"key"}),
            ),
        ),
    )

    def __init__(self) -> None:
        self.held = False

    def reset(self) -> None:
        self.held = False

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> dict[str, float] | None:
        for event in user_inputs.events:
            if event.event_type == "key_down":
                self.held = True
            elif event.event_type == "key_up":
                self.held = False
        if not self.held:
            return None
        return {"amount": window.end_s - window.start_s}


def test_raw_input_canonicalization_uses_each_driver_window_duration() -> None:
    source_schema = UserInputSchema(
        capabilities=(
            UserInputCapability(
                event_type="key_down",
                payload_fields=frozenset({"key"}),
            ),
        )
    )
    provider = ApplicationCanonicalInputProvider(
        input_schema=CanonicalInputSchema(modalities=(_CONTROL,)),
        canonicalizer=InputCanonicalizer((_WindowDurationConverter(),)),
        source_schema=source_schema,
    )
    event = UserInputEvent(
        timestamp_s=0.5,
        event_type="key_down",
        payload={"key": "w"},
    )

    short = provider.prepare_step(
        request=StepRequirements(step_index=0),
        user_window=_raw_user_window((event,), start_s=0.0, end_s=1.0),
    )
    long = provider.prepare_step(
        request=StepRequirements(step_index=1),
        user_window=_raw_user_window((), start_s=1.0, end_s=3.0),
    )

    assert _prepared_amount(short) == pytest.approx(1.0)
    assert _prepared_amount(long) == pytest.approx(2.0)


def test_ordered_edges_close_input_when_output_open_fails() -> None:
    handler = _RecordingHandler([_empty_window()])
    sink = _RecordingSink(fail_open=True)
    edges = OrderedIOFactoryEdges(input_handler=handler, output_sink=sink)

    with pytest.raises(RuntimeError, match="output open failed"):
        edges.open(SessionInfo())

    assert handler.events == ["input.open", "input.close"]
    assert sink.events == ["output.open", "output.close"]


def test_io_factory_source_delegates_optional_handler_capabilities() -> None:
    source = IOFactoryInputSource(_RecordingHandler([_empty_window()]))

    assert source.is_finite
    assert source.is_deterministic


def test_failed_step_closes_resources_and_returns_artifacts() -> None:
    app = _RecordingApplication(fail_step=0)
    handler = _RecordingHandler([_empty_window()])
    sink = _RecordingSink()
    spec, scenario, adapter, mode, host, context = _run_bridge(app, handler, sink)
    try:
        result = run_demo_session(
            context=context,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            run_mode=mode,
            pipeline=StepPipeline(),
        )
    finally:
        context.close()
        host.close()

    assert result.status == "failed"
    assert result.artifacts == (sink.artifact,)
    assert app.session is not None and app.session.closed
    assert handler.events[-1] == "input.close"
    assert sink.events[-1] == "output.close"


def test_application_scenario_selects_converters_only_for_raw_realtime_input() -> None:
    interactive = _RecordingApplication(
        input_schema=CanonicalInputSchema(modalities=(DRIVER_COMMAND, CAMERA_COMMAND))
    )
    uncontrolled = _RecordingApplication()

    batch = application_scenario(interactive, realtime=False)
    realtime = application_scenario(interactive, realtime=True)
    empty_realtime = application_scenario(uncontrolled, realtime=True)

    assert batch.source_schema == UserInputSchema()
    assert batch.canonicalizer.converters == ()
    assert [
        converter.schema.produces for converter in realtime.canonicalizer.converters
    ] == [DRIVER_COMMAND, CAMERA_COMMAND]
    assert empty_realtime.canonicalizer.converters == ()


def test_application_factories_map_to_truthful_output_specs(tmp_path) -> None:
    local = output_spec_for(LocalWindowIOFactory(title="Local", fps=18.0))
    mp4 = output_spec_for(
        Mp4IOFactory(
            output_path=tmp_path / "demo.mp4",
            fps=None,
            output_layout=None,
        )
    )
    null = output_spec_for(NullIOFactory(store_results=True))
    opaque = output_spec_for(
        CallableIOFactory(
            input_factory=lambda _schema: NullInputHandler(),
            output_factory=NullOutputSink,
        )
    )

    assert local == LocalWindowOutputSpec(title="Local", fps=18.0)
    assert mp4 == Mp4OutputSpec(
        path=tmp_path / "demo.mp4",
        fps=None,
        output_layout=None,
    )
    assert null == NullOutputSpec(store_results=True)
    assert opaque == IOFactoryOutputSpec()


def _user_window_with_carrier(
    carried: CanonicalInputWindow,
    *,
    start_s: float,
    end_s: float,
):
    from flashdreams.runtime.demo import UserInputWindow

    return UserInputWindow(
        start_s=start_s,
        end_s=end_s,
        metadata={_CANONICAL_INPUT_WINDOW_KEY: carried},
    )


def _raw_user_window(
    events: tuple[UserInputEvent, ...],
    *,
    start_s: float,
    end_s: float,
):
    from flashdreams.runtime.demo import UserInputWindow

    return UserInputWindow(
        start_s=start_s,
        end_s=end_s,
        inputs=UserInputs(events=events),
    )


def _prepared_amount(prepared: Any) -> float:
    assert prepared.inference_input is not None
    window = prepared.inference_input.step[_CANONICAL_INPUT_WINDOW_KEY]
    assert isinstance(window, CanonicalInputWindow)
    return float(window.values["control"]["amount"])
