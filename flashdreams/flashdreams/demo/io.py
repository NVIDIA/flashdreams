# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public demo IO handler factories."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from flashdreams.runtime.demo.drivers import BatchSessionDriver
from flashdreams.runtime.demo.host import ModelWarmupPlan, RuntimeHost
from flashdreams.runtime.demo.outputs import (
    NullOutputSink,
    OutputDecision,
    OutputSink,
    SessionInfo,
)
from flashdreams.runtime.demo.run_modes import (
    AsyncSessionDriver,
    InMemorySessionMetricsRecorder,
    NoopTransportService,
    RunContext,
    RunModeCapabilities,
    RunResult,
    SessionDriver,
    SessionEdges,
    SingleSessionAdmissionPolicy,
)
from flashdreams.runtime.demo.session_inputs import UserInputWindow
from flashdreams.runtime.demo.spec import DemoAdapter, DemoSpec
from flashdreams.runtime.inputs import UserInputs, UserInputSchema
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepRequirements, StepResult

from .application import FrameOutputSink, IOHandler

RunSessionCallback = Callable[[IOHandler], RunResult]
ServeCallback = Callable[[], object]


@runtime_checkable
class IOHandlerServer(Protocol):
    """Server-shaped IO factory for transports that create handlers per peer."""

    def serve(self, run_session: RunSessionCallback) -> RunResult:
        """Serve one or more sessions by passing handlers to ``run_session``."""
        ...


@dataclass(slots=True)
class ReplayIOHandler:
    """Batch IO handler for replay/null-style public runner sessions."""

    replay_log: UserInputs | None = None
    output_sink: OutputSink | FrameOutputSink | None = None
    metric_output_sink: FrameOutputSink | None = None
    is_finite: bool = True
    is_deterministic: bool = True
    user_input_schema: UserInputSchema = field(default_factory=UserInputSchema)
    _input_source: "_ReplayIOInputSource" = field(init=False, repr=False)
    _output_sink: OutputSink | FrameOutputSink = field(init=False, repr=False)
    _opened_session_info: SessionInfo | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _generation: int | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._input_source = _ReplayIOInputSource(self.replay_log or UserInputs())
        self._output_sink = self.output_sink or NullOutputSink()

    @property
    def run_mode(self) -> "IOHandlerRunMode":
        """Return the runtime run mode backing this IO handler."""
        return IOHandlerRunMode(
            io_handler=self,
            name="replay",
            capabilities=RunModeCapabilities(
                requires_finite_input=True,
                supports_artifacts=True,
            ),
        )

    def open(self, session_info: SessionInfo) -> None:
        self._opened_session_info = session_info
        open_output = getattr(self._output_sink, "open", None)
        if callable(open_output):
            open_output(session_info)

    def next_window(self, requirements: StepRequirements) -> UserInputWindow:
        return self._input_source.next_window(requirements)

    def get_user_input_state(self, modality: str, name: str) -> Any:
        del modality, name
        return None

    def begin_generation(self, generation: int) -> None:
        self._generation = generation
        begin_generation = getattr(self._output_sink, "begin_generation", None)
        if callable(begin_generation):
            begin_generation(generation)

    def emit_chunk(self, result: StepResult) -> OutputDecision:
        write = getattr(self._output_sink, "write", None)
        if callable(write):
            decision = write(result)
            if not isinstance(decision, OutputDecision):
                raise TypeError(
                    "OutputSink.write must return OutputDecision, "
                    f"got {type(decision).__name__}."
                )
        else:
            timestamp_s = _result_timestamp_s(result)
            handle_output = getattr(self._output_sink, "handle_output")
            handle_output(timestamp_s, result)
            decision = OutputDecision()
        if self.metric_output_sink is not None:
            self.metric_output_sink.handle_output(_result_timestamp_s(result), result)
        return decision

    def should_exit(self) -> bool:
        return self._closed

    def close(self) -> Sequence[OutputArtifact]:
        self._closed = True
        close = getattr(self._output_sink, "close", None)
        if callable(close):
            artifacts = close()
            if artifacts is None:
                return ()
            return tuple(artifacts)
        return ()


@dataclass(slots=True)
class NativeWindowIOHandler(ReplayIOHandler):
    """Placeholder native-window factory result until native edges are adopted."""

    @property
    def run_mode(self) -> "IOHandlerRunMode":
        return IOHandlerRunMode(
            io_handler=self,
            name="native-window",
            capabilities=RunModeCapabilities(supports_artifacts=True),
        )


@dataclass(slots=True)
class WebRTCIOHandlerServer:
    """Server-shaped testable facade for WebRTC-style per-connection handlers."""

    host: str
    port: int
    viewport_size: tuple[int, int]
    handlers: Sequence[IOHandler] = ()

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("WebRTC host must be non-empty.")
        if not (0 < int(self.port) < 65536):
            raise ValueError("WebRTC port must be between 1 and 65535.")
        width, height = self.viewport_size
        if width <= 0 or height <= 0:
            raise ValueError("WebRTC viewport dimensions must be > 0.")
        self.handlers = tuple(self.handlers)

    def serve(self, run_session: RunSessionCallback) -> RunResult:
        result = RunResult(status="completed")
        for handler in self.handlers:
            result = run_session(handler)
            if result.status not in {"completed", "skipped"}:
                return result
        return result


@dataclass(slots=True)
class CallbackIOHandlerServer:
    """Server adapter for existing transports during IO-factory adoption."""

    callback: ServeCallback

    def serve(self, run_session: RunSessionCallback) -> RunResult:
        del run_session
        return _coerce_run_result(self.callback())


@dataclass(slots=True)
class IOHandlerRunMode:
    """Runtime run-mode adapter for public ``IOHandler`` instances."""

    io_handler: IOHandler
    name: str = "public-runner"
    capabilities: RunModeCapabilities = field(
        default_factory=lambda: RunModeCapabilities(supports_artifacts=True)
    )
    driver: SessionDriver | AsyncSessionDriver = field(
        default_factory=BatchSessionDriver
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
            input_source=IOHandlerBatchInputSource(self.io_handler),
            output_sink=IOHandlerOutputSink(self.io_handler),
            cleanup_tasks=context.cleanup_tasks,
            transport=NoopTransportService(),
        )

    def select_driver(self) -> SessionDriver | AsyncSessionDriver:
        return self.driver


@dataclass(slots=True)
class IOHandlerBatchInputSource:
    """Batch input source adapter for public ``IOHandler`` windows."""

    io_handler: IOHandler

    @property
    def is_finite(self) -> bool:
        return bool(getattr(self.io_handler, "is_finite", False))

    @property
    def is_deterministic(self) -> bool:
        return bool(getattr(self.io_handler, "is_deterministic", False))

    @property
    def user_input_schema(self) -> UserInputSchema:
        schema = getattr(self.io_handler, "user_input_schema", None)
        if isinstance(schema, UserInputSchema):
            return schema
        return UserInputSchema()

    def is_finished(self) -> bool:
        return self.io_handler.should_exit()

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        return self.io_handler.next_window(request)


@dataclass(slots=True)
class IOHandlerOutputSink:
    """Output sink adapter for public ``IOHandler`` chunks."""

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


def create_replay_io_handler(
    replay_log: UserInputs | None = None,
    output_sink: OutputSink | FrameOutputSink | None = None,
    metric_output_sink: FrameOutputSink | None = None,
) -> ReplayIOHandler:
    """Create a public replay IO handler."""
    return ReplayIOHandler(
        replay_log=replay_log,
        output_sink=output_sink,
        metric_output_sink=metric_output_sink,
    )


def create_native_window_io_handler(
    viewport_size: tuple[int, int],
) -> NativeWindowIOHandler:
    """Create a placeholder native-window-shaped IO handler.

    The real native-window runtime wiring is a later migration phase. Keeping the
    factory name public now lets callers select the mode without importing
    runtime internals.
    """
    width, height = viewport_size
    if width <= 0 or height <= 0:
        raise ValueError("Native window viewport dimensions must be > 0.")
    return NativeWindowIOHandler()


def create_webrtc_io_handler(
    host: str,
    port: int,
    viewport_size: tuple[int, int],
    *,
    handlers: Sequence[IOHandler] = (),
) -> IOHandlerServer:
    """Create a server-shaped WebRTC IO facade."""
    return WebRTCIOHandlerServer(
        host=host,
        port=port,
        viewport_size=viewport_size,
        handlers=handlers,
    )


class _ReplayIOInputSource:
    def __init__(self, replay_log: UserInputs) -> None:
        self._replay_log = replay_log

    def next_window(self, requirements: StepRequirements) -> UserInputWindow:
        start_s = float(requirements.step_index)
        end_s = float(requirements.step_index + requirements.input_frame_count)
        return UserInputWindow(
            start_s=start_s,
            end_s=end_s,
            inputs=self._replay_log,
        )


def _result_timestamp_s(result: StepResult) -> float:
    if result.output_window is None:
        return float(result.step_index)
    return result.output_window.start_s


def _coerce_run_result(value: object) -> RunResult:
    if isinstance(value, RunResult):
        return value
    return RunResult(status="completed")


__all__ = [
    "CallbackIOHandlerServer",
    "IOHandlerBatchInputSource",
    "IOHandlerOutputSink",
    "IOHandlerRunMode",
    "IOHandlerServer",
    "NativeWindowIOHandler",
    "ReplayIOHandler",
    "RunSessionCallback",
    "ServeCallback",
    "WebRTCIOHandlerServer",
    "create_native_window_io_handler",
    "create_replay_io_handler",
    "create_webrtc_io_handler",
]
