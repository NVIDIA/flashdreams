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

"""Application adapters for the shared demo runtime entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from flashdreams.demo.factories import (
    LocalWindowIOFactory,
    Mp4IOFactory,
    NullIOFactory,
)
from flashdreams.demo.io import (
    InputHandler,
    IOFactory,
    OutputDecision,
    OutputSink,
    SessionInfo,
)
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime.canonical import (
    CAMERA_COMMAND,
    DRIVER_COMMAND,
    DeviceConverter,
    InputCanonicalizer,
    KeyboardToCameraCommand,
    KeyboardToDriverCommand,
)
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.demo.drivers import BatchSessionDriver
from flashdreams.runtime.demo.host import (
    ModelWarmupPlan,
    RuntimeHost,
    WarmupSessionInputs,
)
from flashdreams.runtime.demo.run_modes import (
    DefaultErrorPolicy,
    InMemorySessionMetricsRecorder,
    NoopTransportService,
    RunContext,
    RunModeCapabilities,
    SessionEdges,
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
    IOFactoryOutputSpec,
    LocalWindowOutputSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    OutputSpec,
    PreparedScenario,
)
from flashdreams.runtime.inputs import (
    CanonicalInputSchema,
    CanonicalInputWindow,
    InferenceInput,
    InferenceInputSchema,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepRequest, StepRequirements, StepResult

if TYPE_CHECKING:
    from flashdreams.demo.application import (
        IFlashDreamsApplication,
        IFlashDreamsApplicationSession,
    )

_CANONICAL_INPUT_WINDOW_KEY = "flashdreams.application.canonical_input_window"
"""Private carrier key between the application provider and session adapter."""


def _application_inference_input(
    canonical_inputs: CanonicalInputWindow,
) -> InferenceInput:
    """Wrap canonical application inputs in the private runtime carrier."""
    return InferenceInput(
        step={_CANONICAL_INPUT_WINDOW_KEY: canonical_inputs},
    )


@dataclass(slots=True)
class ApplicationRuntime:
    """Create application sessions through the hosted runtime boundary."""

    app: IFlashDreamsApplication
    """Application that owns session construction."""

    _preloaded_session: ApplicationSession | None = field(
        default=None,
        init=False,
        repr=False,
    )
    """Initialized session retained for the first managed run."""

    _steady_output_frame_count: int | None = field(
        default=None,
        init=False,
        repr=False,
    )
    """Cached steady output size exposed to WebRTC offer negotiation."""

    _closed: bool = field(default=False, init=False, repr=False)
    """Whether application-lifetime resources have been released."""

    def preload(self) -> None:
        """Initialize and retain the first application session."""
        if self._preloaded_session is not None:
            return
        session = self._create_session()
        try:
            session_info = session.session_info()
        except BaseException:
            session.close()
            raise
        self._steady_output_frame_count = max(
            1,
            session_info.steady_output_frame_count or 1,
        )
        self._preloaded_session = session

    def start_session(self, inputs: InferenceInput) -> ApplicationSession:
        """Create and initialize one application session."""
        del inputs
        session = self._preloaded_session
        if session is not None:
            self._preloaded_session = None
            return session
        return self._create_session()

    def peek_steady_output_num_frames(self) -> int:
        """Return the output size cached during model-worker preload."""
        if self._steady_output_frame_count is None:
            raise RuntimeError("Application runtime must be preloaded before serving.")
        return self._steady_output_frame_count

    def _create_session(self) -> ApplicationSession:
        from flashdreams.demo.application import IFlashDreamsApplicationSession

        session = self.app.create_session()
        if not isinstance(session, IFlashDreamsApplicationSession):
            raise TypeError(
                "IFlashDreamsApplication.create_session() must return "
                "IFlashDreamsApplicationSession."
            )
        session.init()
        return ApplicationSession(session)

    def close(self) -> None:
        """Release runtime-owned resources."""
        if self._closed:
            return
        self._closed = True
        session = self._preloaded_session
        self._preloaded_session = None
        session_error: BaseException | None = None
        try:
            if session is not None:
                session.close()
        except BaseException as exc:
            session_error = exc
        try:
            self.app.close()
        except BaseException as exc:
            if session_error is None:
                raise
            _add_exception_note(
                session_error,
                f"Application cleanup failed: {exc}",
            )
        if session_error is not None:
            raise session_error


@dataclass(slots=True)
class ApplicationSession:
    """Expose an application session through the inference-session contract."""

    session: IFlashDreamsApplicationSession
    """Application-owned model session."""

    def session_info(self) -> SessionInfo:
        """Return validated metadata for input and output setup."""
        value = self.session.session_info()
        if not isinstance(value, SessionInfo):
            raise TypeError(
                "IFlashDreamsApplicationSession.session_info() must return SessionInfo."
            )
        return value

    def next_step_requirements(self) -> StepRequirements | None:
        """Return the requirements for the next application step."""
        return self.session.next_step_requirements()

    def next_step_request(self) -> StepRequest | None:
        """Project application requirements onto the runtime request contract."""
        requirements = self.next_step_requirements()
        if requirements is None:
            return None
        metadata = dict(requirements.metadata)
        metadata["input_frame_count"] = requirements.input_frame_count
        if requirements.steady_output_frame_count is not None:
            metadata["steady_output_frame_count"] = (
                requirements.steady_output_frame_count
            )
        return StepRequest(
            step_index=requirements.step_index,
            inference_input_schema=requirements.inference_input_schema,
            metadata=metadata,
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        """Run one application step with its canonical input window."""
        canonical_inputs = inputs.step.get(_CANONICAL_INPUT_WINDOW_KEY)
        if not isinstance(canonical_inputs, CanonicalInputWindow):
            raise TypeError(
                "Application input provider must supply a CanonicalInputWindow."
            )
        return self.session.step(canonical_inputs)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Reset the application session's per-generation state."""
        del inputs
        self.session.reset()

    def close(self) -> None:
        """Close the application-owned session."""
        self.session.close()


@dataclass(slots=True)
class ApplicationCanonicalInputProvider:
    """Convert driver-owned windows into application canonical inputs."""

    input_schema: CanonicalInputSchema
    """Canonical modalities consumed by the application."""

    canonicalizer: InputCanonicalizer
    """Raw-input converter registry prepared for the selected run path."""

    source_schema: UserInputSchema
    """Raw capabilities supplied by the selected input source."""

    initial_inputs: InferenceInput = field(default_factory=InferenceInput)
    """Session-global inputs retained by the prepared scenario."""

    inference_input_schema: InferenceInputSchema = field(
        default_factory=InferenceInputSchema
    )
    """Model-facing input schema declared by an optional adapter."""

    supports_reset: bool = False
    """Whether the paired application session supports generation resets."""

    capabilities: ProviderCapabilities = field(init=False)
    """Runtime capabilities derived from the prepared source schema."""

    def __post_init__(self) -> None:
        self.capabilities = ProviderCapabilities(
            supports_realtime_clock=True,
            supports_reset=self.supports_reset,
            deterministic_given_inputs=False,
            user_input_schema=self.source_schema,
            inference_input_schema=self.inference_input_schema,
        )

    def prepare_initial_input(self) -> InferenceInput:
        """Return the scenario's application-owned initial input."""
        return self.initial_inputs

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        """Prepare one canonical application input at the driver's time window."""
        del request
        step_window = TimeWindow(
            start_s=user_window.start_s,
            end_s=user_window.end_s,
        )
        carried = user_window.metadata.get(_CANONICAL_INPUT_WINDOW_KEY)
        if carried is not None:
            if not isinstance(carried, CanonicalInputWindow):
                raise TypeError(
                    "Application input source carrier must be a CanonicalInputWindow."
                )
            canonical_window = replace(carried, window=step_window)
        else:
            canonical = self.canonicalizer.canonicalize(
                user_window.inputs,
                window=step_window,
                source_schema=self.source_schema,
            )
            canonical_window = CanonicalInputWindow(
                values=canonical.values,
                metadata=canonical.metadata,
                window=step_window,
            )
        validate_declared_modalities(canonical_window, self.input_schema)
        return PreparedStep(
            inference_input=_application_inference_input(canonical_window)
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Reset canonical conversion and optional initial model inputs."""
        if not self.supports_reset:
            raise RuntimeError("FlashDreams application sessions do not support reset.")
        if inputs is not None:
            self.initial_inputs = inputs
        self.canonicalizer.reset()

    def close(self) -> None:
        """Leave input and output cleanup to the session edges."""


@dataclass(slots=True)
class ApplicationDemoAdapter:
    """Present an application through the shared demo-adapter contract."""

    app: IFlashDreamsApplication
    """Application exposed through the runtime seam."""

    spec: DemoSpec
    """Resolved application run specification."""

    scenario: PreparedScenario
    """Prepared raw-to-canonical input scenario."""

    @property
    def model_id(self) -> str:
        """Return the application model identity."""
        return self.spec.model_id

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema:
        """Return the application's canonical input contract."""
        return self.app.input_schema

    @property
    def inference_input_schema(self) -> InferenceInputSchema:
        """Return the optional model adapter's inference-input contract."""
        delegate = _application_model_adapter(self.app)
        schema = getattr(delegate, "inference_input_schema", None)
        return (
            schema
            if isinstance(schema, InferenceInputSchema)
            else InferenceInputSchema()
        )

    def default_input_mapping(self) -> None:
        """Return no mapping because ``app.step`` owns canonical conversion."""
        return

    def validate_config(self, config: InferenceConfig) -> None:
        """Validate application identity and optional adapter configuration."""
        if config.model_id != self.spec.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")
        delegate = _application_model_adapter(self.app)
        validate = getattr(delegate, "validate_config", None)
        if callable(validate):
            validate(config)

    def create_runtime(self, config: InferenceConfig) -> ApplicationRuntime:
        """Create the hosted application runtime."""
        self.validate_config(config)
        return ApplicationRuntime(self.app)

    def supported_input_modes(self) -> tuple[str, ...]:
        """Return the input mode selected for this application run."""
        return (self.spec.input_mode,)

    def supported_output_modes(self) -> tuple[str, ...]:
        """Return the output mode selected for this application run."""
        return (self.spec.output.mode,)

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        """Return the prepared application scenario."""
        if spec != self.spec:
            raise ValueError("Application adapter received an unexpected DemoSpec.")
        return self.scenario

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> object:
        """Create an optional model provider or the canonical application bridge."""
        delegate = _application_model_adapter(self.app)
        create_provider = getattr(delegate, "create_model_input_provider", None)
        if callable(create_provider):
            return create_provider(spec, scenario)
        return ApplicationCanonicalInputProvider(
            input_schema=self.app.input_schema,
            canonicalizer=scenario.canonicalizer,
            source_schema=scenario.source_schema,
            initial_inputs=scenario.initial_inputs,
            inference_input_schema=self.inference_input_schema,
            supports_reset=self.app.supports_session_reset,
        )

    def create_model_warmup_sessions(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> Sequence[WarmupSessionInputs]:
        """Translate application warmup windows into hosted runtime inputs."""
        from flashdreams.demo.application import ApplicationWarmupSessionInputs

        sessions = self.app.create_model_warmup_sessions(spec, scenario)
        runtime_sessions: list[WarmupSessionInputs] = []
        for session in sessions:
            if not isinstance(session, ApplicationWarmupSessionInputs):
                raise TypeError(
                    "IFlashDreamsApplication.create_model_warmup_sessions() must "
                    "return only ApplicationWarmupSessionInputs."
                )
            runtime_sessions.append(
                WarmupSessionInputs(
                    initial_input=InferenceInput(),
                    step_inputs=tuple(
                        _application_inference_input(value)
                        for value in session.step_inputs
                    ),
                )
            )
        return tuple(runtime_sessions)


@dataclass(slots=True)
class IOFactoryInputSource:
    """Adapt a canonical application input handler to a batch input source."""

    input_handler: InputHandler
    """Handler that has already canonicalized application input."""

    user_input_schema: UserInputSchema = field(default_factory=UserInputSchema)
    """Empty schema because canonical handlers expose no raw event stage."""

    is_finite: bool = field(init=False)
    """Whether the wrapped handler declares a finite input stream."""

    is_deterministic: bool = field(init=False)
    """Whether the wrapped handler declares deterministic input."""

    def __post_init__(self) -> None:
        self.is_finite = bool(getattr(self.input_handler, "is_finite", False))
        self.is_deterministic = bool(
            getattr(self.input_handler, "is_deterministic", False)
        )

    def is_finished(self) -> bool:
        """Let the application session declare completion."""
        return False

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        """Return the handler's canonical input with an explicit carrier."""
        del request
        canonical_inputs = self.input_handler.current_inputs()
        if not isinstance(canonical_inputs, CanonicalInputWindow):
            raise TypeError(
                "InputHandler.current_inputs() must return CanonicalInputWindow."
            )
        return UserInputWindow(
            start_s=canonical_inputs.window.start_s,
            end_s=canonical_inputs.window.end_s,
            inputs=UserInputs(
                snapshot=canonical_inputs.values,
                metadata=canonical_inputs.metadata,
            ),
            metadata={_CANONICAL_INPUT_WINDOW_KEY: canonical_inputs},
        )


@dataclass(slots=True)
class OrderedIOFactoryEdges:
    """Pair application input and output lifecycles in dependency order."""

    input_handler: InputHandler
    """Input resource opened before and closed after output."""

    output_sink: OutputSink
    """Output resource opened after and closed before input."""

    _opened: bool = field(default=False, init=False, repr=False)
    """Whether resource opening has started."""

    _closed: bool = field(default=False, init=False, repr=False)
    """Whether both resource close paths have run."""

    _artifacts: tuple[OutputArtifact, ...] = field(default=(), init=False, repr=False)
    """Artifacts cached after idempotent closure."""

    produces_artifacts: bool = field(init=False)
    """Whether the wrapped output can produce artifacts."""

    def __post_init__(self) -> None:
        self.produces_artifacts = self.output_sink.produces_artifacts

    def open(self, session_info: SessionInfo) -> None:
        """Open input before output and start the initial generation."""
        self._opened = True
        try:
            self.input_handler.open(session_info)
            self.output_sink.open(session_info)
            self.output_sink.begin_generation(0)
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_exc:  # noqa: BLE001 - preserve failure
                _add_exception_note(
                    exc,
                    f"Application I/O cleanup failed: {cleanup_exc}",
                )
            raise

    def begin_generation(self, generation: int) -> None:
        """Begin one output generation."""
        self.output_sink.begin_generation(generation)

    def write(self, result: StepResult) -> OutputDecision:
        """Write one result to the application output sink."""
        return self.output_sink.write(result)

    def close(self) -> Sequence[OutputArtifact]:
        """Close output before input and preserve both cleanup failures."""
        if self._closed:
            return self._artifacts
        self._closed = True
        output_error: BaseException | None = None
        if self._opened:
            try:
                self._artifacts = tuple(self.output_sink.close())
            except BaseException as exc:  # noqa: BLE001 - finish paired cleanup
                output_error = exc
            try:
                self.input_handler.close()
            except BaseException as exc:  # noqa: BLE001 - finish paired cleanup
                if output_error is None:
                    output_error = exc
                else:
                    _add_exception_note(
                        output_error,
                        f"Input handler close failed: {exc}",
                    )
        if output_error is not None:
            raise output_error
        return self._artifacts


@dataclass(slots=True)
class IOFactoryRunMode:
    """Run one application over caller-owned input and output resources."""

    input_handler: InputHandler
    """Canonical input handler for this run."""

    output_sink: OutputSink
    """Application output sink for this run."""

    name: str = "application"
    """Stable run-mode identity used by logs and validation."""

    capabilities: RunModeCapabilities = field(
        default_factory=lambda: RunModeCapabilities(supports_artifacts=True)
    )
    """Application batch-mode capabilities."""

    def validate_run(self, *, spec: DemoSpec, adapter: DemoAdapter) -> None:
        """Validate selected modes and application configuration."""
        _require_supported_mode(
            mode=spec.input_mode,
            supported=adapter.supported_input_modes(),
            label="input_mode",
        )
        _require_supported_mode(
            mode=spec.output.mode,
            supported=adapter.supported_output_modes(),
            label="output.mode",
        )
        if spec.config is None:
            raise RuntimeError("DemoSpec.config was not initialized.")
        adapter.validate_config(spec.config)

    def validate_session(
        self,
        *,
        spec: DemoSpec,
        scenario: PreparedScenario,
        adapter: DemoAdapter,
        provider: object,
    ) -> None:
        """Validate the prepared session boundary."""
        del spec, scenario, adapter, provider

    def create_run_context(
        self,
        *,
        spec: DemoSpec,
        adapter: DemoAdapter,
        host: RuntimeHost,
        model_warmup_plan: ModelWarmupPlan,
    ) -> RunContext:
        """Create run-scoped admission, metrics, and warmup services."""
        del spec, adapter
        return RunContext(
            host=host,
            run_metrics=InMemorySessionMetricsRecorder(),
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
        scenario: PreparedScenario,
        provider: object,
        adapter: DemoAdapter,
    ) -> SessionEdges:
        """Create one complete batch session-edge bundle."""
        del spec, scenario, provider, adapter
        return SessionEdges(
            input_source=IOFactoryInputSource(self.input_handler),
            output_sink=OrderedIOFactoryEdges(
                input_handler=self.input_handler,
                output_sink=self.output_sink,
            ),
            cleanup_tasks=context.cleanup_tasks,
            error_policy=DefaultErrorPolicy(),
            transport=NoopTransportService(),
        )

    def select_driver(self) -> BatchSessionDriver:
        """Return the synchronous application session driver."""
        return BatchSessionDriver()


def application_demo_spec(
    *,
    app: IFlashDreamsApplication,
    application_slug: str,
    output: OutputSpec,
    input_mode: str = "application",
    config: InferenceConfig | None = None,
) -> DemoSpec:
    """Build a demo specification for one initialized application."""
    del app
    return DemoSpec(
        model_id=application_slug,
        input_mode=input_mode,
        output=output,
        config=config or InferenceConfig(model_id=application_slug),
    )


def application_scenario(
    app: IFlashDreamsApplication,
    *,
    realtime: bool,
) -> PreparedScenario:
    """Build the raw-to-canonical scenario for an application run path."""
    schema = app.input_schema
    if not realtime:
        return PreparedScenario(
            initial_inputs=InferenceInput(),
            source_schema=UserInputSchema(),
            canonicalizer=InputCanonicalizer(),
            mapping=None,
        )

    from flashdreams.serving.webrtc.services import WEBRTC_USER_INPUT_SCHEMA

    available_modalities = (DRIVER_COMMAND, CAMERA_COMMAND)
    unsupported = [
        modality.name
        for modality in schema.modalities
        if not any(
            modality.is_satisfied_by(available) for available in available_modalities
        )
    ]
    if unsupported:
        raise ValueError(
            "WebRTC input cannot provide canonical modalities: "
            f"{sorted(set(unsupported))}."
        )
    requested = frozenset(modality.name for modality in schema.modalities)
    converters: list[DeviceConverter] = []
    if DRIVER_COMMAND.name in requested:
        converters.append(KeyboardToDriverCommand())
    if CAMERA_COMMAND.name in requested:
        converters.append(KeyboardToCameraCommand())
    return PreparedScenario(
        initial_inputs=InferenceInput(),
        source_schema=WEBRTC_USER_INPUT_SCHEMA,
        canonicalizer=InputCanonicalizer(converters),
        mapping=None,
    )


def output_spec_for(factory: IOFactory) -> OutputSpec:
    """Describe the output mode selected by an application I/O factory."""
    if isinstance(factory, LocalWindowIOFactory):
        return LocalWindowOutputSpec(title=factory.title, fps=factory.fps)
    if isinstance(factory, Mp4IOFactory):
        return Mp4OutputSpec(
            path=factory.output_path,
            fps=factory.fps,
            output_layout=factory.output_layout,
            move_to_cpu=factory.move_to_cpu,
        )
    if isinstance(factory, NullIOFactory):
        return NullOutputSpec(store_results=factory.store_results)
    return IOFactoryOutputSpec()


def validate_declared_modalities(
    inputs: CanonicalInputWindow,
    input_schema: CanonicalInputSchema,
) -> None:
    """Validate one canonical window against the application schema."""
    expected = {modality.name: modality for modality in input_schema.modalities}
    unknown = sorted(set(inputs.values) - set(expected))
    if unknown:
        raise ValueError(f"Canonical inputs contain undeclared modalities: {unknown}.")
    missing = sorted(set(expected) - set(inputs.values))
    if missing:
        raise ValueError(
            f"Canonical inputs are missing requested modalities: {missing}."
        )
    for name, value in inputs.values.items():
        if not isinstance(value, Mapping):
            raise TypeError(f"Canonical input {name!r} must be a named field mapping.")
        expected[name].value(value)


def current_application_inputs(
    handler: InputHandler,
    input_schema: CanonicalInputSchema,
) -> CanonicalInputWindow:
    """Read and validate one canonical window from an application handler."""
    inputs = handler.current_inputs()
    if not isinstance(inputs, CanonicalInputWindow):
        raise TypeError(
            "InputHandler.current_inputs() must return CanonicalInputWindow."
        )
    validate_declared_modalities(inputs, input_schema)
    return inputs


def _application_model_adapter(app: IFlashDreamsApplication) -> Any | None:
    return getattr(app, "model_adapter", None)


def _require_supported_mode(
    *,
    mode: str,
    supported: Sequence[str],
    label: str,
) -> None:
    if mode not in supported:
        raise ValueError(
            f"Unsupported {label}={mode!r}; expected one of {tuple(supported)!r}."
        )


def _add_exception_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


__all__ = [
    "ApplicationCanonicalInputProvider",
    "ApplicationDemoAdapter",
    "ApplicationRuntime",
    "ApplicationSession",
    "IOFactoryInputSource",
    "IOFactoryRunMode",
    "OrderedIOFactoryEdges",
    "application_demo_spec",
    "application_scenario",
    "current_application_inputs",
    "output_spec_for",
    "validate_declared_modalities",
]
