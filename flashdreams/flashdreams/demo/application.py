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

"""Transport-neutral contracts and host loop for FlashDreams applications."""

from __future__ import annotations

import asyncio
import importlib
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from flashdreams.demo.factories import (
    LocalWindowIOFactory,
    Mp4IOFactory,
    NullInputHandler,
    NullIOFactory,
    ProvidedIOFactory,
    WebRTCApplicationServing,
)
from flashdreams.demo.io import (
    InputHandler,
    IOFactory,
    OutputSink,
    SessionInfo,
)
from flashdreams.demo.outputs import LocalWindowOutputSink
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import CanonicalInputSchema, CanonicalInputWindow
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepRequirements, StepResult

if TYPE_CHECKING:
    from flashdreams.runtime.demo.spec import DemoSpec, PreparedScenario

APPLICATION_ENTRY_POINT_GROUP = "flashdreams.applications"
"""Entry-point group whose values expose a zero-argument ``create_app`` factory."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ApplicationWarmupSessionInputs:
    """Canonical step inputs used to warm one temporary application session."""

    step_inputs: Sequence[CanonicalInputWindow] = ()
    """Ordered canonical windows that exercise the application's model shapes."""

    def __post_init__(self) -> None:
        step_inputs = tuple(self.step_inputs)
        if not all(isinstance(value, CanonicalInputWindow) for value in step_inputs):
            raise TypeError(
                "Application warmup step inputs must be CanonicalInputWindow values."
            )
        object.__setattr__(self, "step_inputs", step_inputs)


class IFlashDreamsApplicationSession(ABC):
    """One isolated application session with sequential model state."""

    @abstractmethod
    def init(self) -> None:
        """Initialize model and per-session resources."""

    def session_info(self) -> SessionInfo:
        """Return sink-facing metadata after session initialization."""
        return SessionInfo()

    @abstractmethod
    def next_step_requirements(self) -> StepRequirements | None:
        """Return requirements for the next step, or ``None`` when complete."""

    @abstractmethod
    def step(self, inputs: CanonicalInputWindow) -> StepResult:
        """Produce one model result for previously declared requirements."""

    def reset(self) -> None:
        """Reset per-generation state when the application supports reuse."""
        raise NotImplementedError(
            "This FlashDreams application session does not support reset."
        )

    def close(self) -> None:
        """Release optional per-session resources."""


class IFlashDreamsApplication(ABC):
    """Application factory boundary independent of presentation backend."""

    @property
    @abstractmethod
    def input_schema(self) -> CanonicalInputSchema:
        """Declare the named canonical inputs consumed by this application."""

    @property
    def supports_session_reset(self) -> bool:
        """Return whether sessions support resetting per-generation state."""
        return False

    @abstractmethod
    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse application arguments and validate startup state."""

    @abstractmethod
    def create_session(self) -> IFlashDreamsApplicationSession:
        """Create one isolated application session."""

    def create_model_warmup_sessions(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> Sequence[ApplicationWarmupSessionInputs]:
        """Return canonical inputs for optional temporary model warmup sessions."""
        del spec, scenario
        return ()

    def close(self) -> None:
        """Release optional application-lifetime model resources."""

    def createSession(self) -> IFlashDreamsApplicationSession:
        """Create a session through the package-facing compatibility spelling."""
        return self.create_session()


ApplicationFactory = Callable[[], IFlashDreamsApplication]
"""Zero-argument factory exported from an application package as ``create_app``."""


def registered_application_slugs() -> tuple[str, ...]:
    """Return installed application slugs in stable display order."""
    return tuple(
        sorted(
            {item.name for item in entry_points(group=APPLICATION_ENTRY_POINT_GROUP)}
        )
    )


def create_application(
    application_slug: str,
) -> tuple[IFlashDreamsApplication, list[str]]:
    """Load the application package registered for an exact slug.

    Args:
        application_slug: User-facing application slug.

    Returns:
        The created application and package-derived arguments, currently empty.

    Raises:
        LookupError: No installed application package matches the slug.
        TypeError: The package factory does not return the application contract.
    """
    if not application_slug.strip():
        raise ValueError("application_slug must be non-empty.")

    registered = sorted(
        entry_points(group=APPLICATION_ENTRY_POINT_GROUP),
        key=lambda item: item.name,
    )
    for entry_point in registered:
        if entry_point.name == application_slug:
            return _create_from_entry_point(entry_point), []

    module = _import_application_module(application_slug)
    factory = getattr(module, "create_app", None)
    if not callable(factory):
        raise TypeError(
            f"Application module {module.__name__!r} does not expose create_app()."
        )
    return _validate_application(factory(), origin=module.__name__), []


@overload
def run_application(
    application_slug: str,
    commandline_args: Sequence[str] = (),
    *,
    io_factory: WebRTCApplicationServing,
    input_handler: None = None,
    output_sink: None = None,
) -> object: ...


@overload
def run_application(
    application_slug: str,
    commandline_args: Sequence[str] = (),
    *,
    io_factory: IOFactory | None = None,
    input_handler: InputHandler | None = None,
    output_sink: OutputSink | None = None,
) -> tuple[OutputArtifact, ...]: ...


def run_application(
    application_slug: str,
    commandline_args: Sequence[str] = (),
    *,
    io_factory: IOFactory | WebRTCApplicationServing | None = None,
    input_handler: InputHandler | None = None,
    output_sink: OutputSink | None = None,
) -> object:
    """Run an application with host-owned canonical input handling.

    Args:
        application_slug: Installed application or concrete demo slug.
        commandline_args: Arguments forwarded to the application.
        io_factory: Factory for input handling and output delivery.
        input_handler: Caller-owned input handler; ``None`` uses the factory.
        output_sink: Caller-owned output sink; ``None`` uses the factory.

    Returns:
        Persistent artifacts for bounded runs, or the WebRTC server result.

    Raises:
        TypeError: The application, handler, sink, or input values violate their
            declared contracts.
        ValueError: Direct I/O objects are combined with ``io_factory``, or the
            current canonical inputs do not match the application schema.
    """
    application, slug_args = create_application(application_slug)
    resolved_factory = _resolve_io_factory(
        io_factory=io_factory,
        input_handler=input_handler,
        output_sink=output_sink,
    )
    application.init([*slug_args, *commandline_args])

    input_schema = application.input_schema
    if not isinstance(input_schema, CanonicalInputSchema):
        raise TypeError(
            "IFlashDreamsApplication.input_schema must be a CanonicalInputSchema."
        )
    if isinstance(resolved_factory, WebRTCApplicationServing):
        return serve_application_webrtc(
            app=application,
            application_slug=application_slug,
            serving=resolved_factory,
        )

    resolved_input = resolved_factory.create_input_handler(input_schema)
    resolved_output = resolved_factory.create_output_sink()
    if not isinstance(resolved_input, InputHandler):
        raise TypeError("IOFactory.create_input_handler() must return an InputHandler.")
    if not isinstance(resolved_output, OutputSink):
        raise TypeError("IOFactory.create_output_sink() must return an OutputSink.")

    from flashdreams.demo.bridge import (
        ApplicationDemoAdapter,
        IOFactoryRunMode,
        application_demo_spec,
        application_scenario,
        output_spec_for,
    )
    from flashdreams.runtime.demo.drivers import run_demo_session
    from flashdreams.runtime.demo.host import RuntimeHost
    from flashdreams.runtime.demo.pipeline import StepPipeline
    from flashdreams.runtime.demo.run_modes import (
        build_model_warmup_plan,
        warmup_run_context,
    )

    spec = application_demo_spec(
        app=application,
        application_slug=application_slug,
        output=output_spec_for(resolved_factory),
    )
    scenario = application_scenario(application, realtime=False)
    adapter = ApplicationDemoAdapter(
        app=application,
        spec=spec,
        scenario=scenario,
    )
    run_mode = IOFactoryRunMode(
        input_handler=resolved_input,
        output_sink=resolved_output,
    )
    run_mode.validate_run(spec=spec, adapter=adapter)
    if spec.config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    host = RuntimeHost(adapter.create_runtime(spec.config))
    try:
        model_warmup_plan = build_model_warmup_plan(
            host=host,
            adapter=adapter,
            spec=spec,
            scenario=scenario,
        )
        context = run_mode.create_run_context(
            spec=spec,
            adapter=adapter,
            host=host,
            model_warmup_plan=model_warmup_plan,
        )
        try:
            warmup_run_context(
                context=context,
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                run_mode=run_mode,
            )
            result = run_demo_session(
                context=context,
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                run_mode=run_mode,
                pipeline=StepPipeline(),
            )
        finally:
            context.close()
    finally:
        host.close()
    if result.status != "completed":
        if result.error is not None:
            raise result.error
        raise RuntimeError(
            result.reason or f"Application run ended with {result.status}."
        )
    return tuple(result.artifacts)


def serve_application_webrtc(
    *,
    app: IFlashDreamsApplication,
    application_slug: str,
    serving: WebRTCApplicationServing,
) -> object:
    """Serve an initialized application through the shared WebRTC manager."""
    from flashdreams.demo.bridge import (
        ApplicationDemoAdapter,
        application_demo_spec,
        application_scenario,
    )
    from flashdreams.runtime.demo import bootstrap as demo_bootstrap
    from flashdreams.runtime.demo.host import RuntimeHost
    from flashdreams.runtime.demo.run_modes import build_model_warmup_plan
    from flashdreams.runtime.demo.spec import WebRTCAppResources, WebRTCOutputSpec
    from flashdreams.serving.webrtc import demo as webrtc_demo
    from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager

    demo_bootstrap.configure_logging()
    distributed = demo_bootstrap.initialize_cuda_distributed(default_device="cuda")
    host: RuntimeHost | None = None
    manager: BaseWebRTCSessionManager[Any, Any] | None = None
    try:
        output = WebRTCOutputSpec(
            host=serving.host,
            port=serving.port,
            fps=serving.fps,
            video_width=serving.video_width,
            video_height=serving.video_height,
            warmup_chunks=serving.warmup_chunks,
            warmup_timeout_s=serving.warmup_timeout_s,
            client_liveness_timeout_s=serving.client_liveness_timeout_s,
            preload_name=application_slug,
        )
        spec = application_demo_spec(
            app=app,
            application_slug=application_slug,
            output=output,
            input_mode="webrtc",
            config=InferenceConfig(
                model_id=application_slug,
                device=str(distributed.device),
            ),
        )
        scenario = application_scenario(app, realtime=True)
        adapter = ApplicationDemoAdapter(app=app, spec=spec, scenario=scenario)
        config = spec.config
        if config is None:
            raise RuntimeError("DemoSpec.config was not initialized.")
        runtime = adapter.create_runtime(config)
        host = RuntimeHost(runtime)
        model_warmup_plan = build_model_warmup_plan(
            host=host,
            adapter=adapter,
            spec=spec,
            scenario=scenario,
        )
        manager = BaseWebRTCSessionManager(
            runtime=runtime,
            runtime_config=output,
            fps=output.fps,
            identity=spec.model_id,
            busy_message="A WebRTC application session is already active.",
            warmup_label=f"{application_slug} WebRTC",
            client_liveness_timeout_s=output.client_liveness_timeout_s,
            activate_without_input=not app.input_schema.modalities,
            model_warmup_plan=model_warmup_plan,
            shared_host=host,
            shared_adapter=adapter,
            shared_spec=spec,
            shared_scenario=scenario,
            keep_connection_after_completed=True,
        )
        return webrtc_demo.serve_webrtc_demo(
            output=output,
            model_id=spec.model_id,
            session_manager=manager,
            app_resources=WebRTCAppResources(preload_name=application_slug),
            world_rank=distributed.world_rank,
        )
    except BaseException as exc:
        _cleanup_webrtc_startup_failure(
            manager=manager,
            host=host,
            primary_error=exc,
            world_rank=distributed.world_rank,
        )
        raise


def _resolve_io_factory(
    *,
    io_factory: IOFactory | WebRTCApplicationServing | None,
    input_handler: InputHandler | None,
    output_sink: OutputSink | None,
) -> IOFactory | WebRTCApplicationServing:
    if io_factory is not None:
        if input_handler is not None or output_sink is not None:
            raise ValueError(
                "Pass io_factory or direct input/output objects, not both."
            )
        return io_factory
    if input_handler is None and output_sink is None:
        return LocalWindowIOFactory()
    return ProvidedIOFactory(
        input_handler=(
            input_handler if input_handler is not None else NullInputHandler()
        ),
        output_sink=(
            output_sink if output_sink is not None else LocalWindowOutputSink()
        ),
    )


def _cleanup_webrtc_startup_failure(
    *,
    manager: Any | None,
    host: Any | None,
    primary_error: BaseException,
    world_rank: int,
) -> None:
    from flashdreams.runtime.demo import bootstrap as demo_bootstrap

    errors: list[BaseException] = []
    if manager is None:
        if host is not None:
            _record_cleanup_error(errors, host.close)
    else:
        _record_cleanup_error(errors, manager.send_exit_signal)
        _record_cleanup_error(errors, _shutdown_webrtc_manager, manager)
    _record_cleanup_error(
        errors,
        demo_bootstrap.cleanup_cuda_distributed,
        world_rank=world_rank,
        synchronize_distributed=False,
    )
    add_note = getattr(primary_error, "add_note", None)
    for cleanup_error in errors:
        if callable(add_note):
            add_note(f"Additional WebRTC startup cleanup error: {cleanup_error!r}")


def _record_cleanup_error(
    errors: list[BaseException],
    cleanup: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> None:
    try:
        cleanup(*args, **kwargs)
    except BaseException as cleanup_error:  # noqa: BLE001
        errors.append(cleanup_error)


def _shutdown_webrtc_manager(manager: Any) -> None:
    asyncio.run(manager.shutdown())


def _create_from_entry_point(entry_point: EntryPoint) -> IFlashDreamsApplication:
    value = entry_point.load()
    application = value() if callable(value) else value
    return _validate_application(application, origin=entry_point.value)


def _validate_application(value: Any, *, origin: str) -> IFlashDreamsApplication:
    if not isinstance(value, IFlashDreamsApplication):
        raise TypeError(
            f"Application factory {origin!r} returned {type(value).__name__}; "
            "expected IFlashDreamsApplication."
        )
    return value


def _import_application_module(slug: str) -> Any:
    module_name = slug.replace("-", "_")
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    available = ", ".join(
        sorted(ep.name for ep in entry_points(group=APPLICATION_ENTRY_POINT_GROUP))
    )
    raise LookupError(
        f"No FlashDreams application package matches {slug!r}. "
        f"Installed applications: {available or '(none)'}."
    )


def _parse_host_io(
    application_slug: str,
    args: Sequence[str],
) -> tuple[IOFactory | WebRTCApplicationServing, list[str]]:
    output_kind = _selected_output(args)
    output_path: Path | None = None
    output_fps: float | None = None
    host = "127.0.0.1"
    port = 8080
    application_args: list[str] = []
    host_options = (
        {"--output", "--host", "--port"}
        if output_kind == "webrtc"
        else {"--output", "--output-path", "--output-fps"}
    )
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in host_options:
            if index + 1 >= len(args):
                raise ValueError(f"{argument} requires a value.")
            value = args[index + 1]
            if argument == "--output-path":
                output_path = Path(value)
            elif argument == "--output-fps":
                output_fps = float(value)
            elif argument == "--host":
                host = value
            elif argument == "--port":
                port = int(value)
            index += 2
            continue
        application_args.append(argument)
        index += 1

    if output_kind == "local-window":
        return LocalWindowIOFactory(fps=output_fps), application_args
    if output_kind == "null":
        return NullIOFactory(), application_args
    if output_kind == "mp4":
        path = output_path or Path("outputs") / f"{application_slug}.mp4"
        return Mp4IOFactory(output_path=path, fps=output_fps), application_args
    if output_kind == "webrtc":
        if not 1 <= port <= 65535:
            raise ValueError("--port must be between 1 and 65535.")
        return (
            WebRTCApplicationServing(
                application_slug,
                host=host,
                port=port,
            ),
            application_args,
        )
    raise ValueError(
        f"Unsupported output {output_kind!r}; expected local-window, null, mp4, or webrtc."
    )


def entrypoint(argv: Sequence[str] | None = None) -> None:
    """Run an application through a direct ``flashdreams-run <slug>`` command."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: flashdreams-run APPLICATION "
            "[--output local-window|null|mp4|webrtc] [--host HOST] [--port PORT] "
            "[APPLICATION_ARGS ...]"
        )
        return
    application_slug = args.pop(0)
    io_factory, application_args = _parse_host_io(application_slug, args)
    if isinstance(io_factory, WebRTCApplicationServing):
        run_application(
            application_slug,
            application_args,
            io_factory=io_factory,
        )
        return
    artifacts = run_application(
        application_slug,
        application_args,
        io_factory=io_factory,
    )
    for artifact in artifacts:
        print(artifact.uri)


def _selected_output(args: Sequence[str]) -> str:
    try:
        index = args.index("--output")
    except ValueError:
        return "local-window"
    if index + 1 >= len(args):
        raise ValueError("--output requires a value.")
    return args[index + 1]


__all__ = [
    "APPLICATION_ENTRY_POINT_GROUP",
    "ApplicationFactory",
    "IFlashDreamsApplication",
    "IFlashDreamsApplicationSession",
    "create_application",
    "entrypoint",
    "registered_application_slugs",
    "run_application",
    "serve_application_webrtc",
]
