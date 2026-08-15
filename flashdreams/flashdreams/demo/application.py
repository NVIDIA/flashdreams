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
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

from flashdreams.demo.factories import (
    ApplicationWebRTCIOFactory,
    CallableIOFactory,
    LocalWindowIOFactory,
    Mp4IOFactory,
    NullInputHandler,
    ProvidedIOFactory,
)
from flashdreams.demo.io import (
    InputHandler,
    IOFactory,
    OutputSink,
    SessionInfo,
)
from flashdreams.demo.outputs import LocalWindowOutputSink, NullOutputSink
from flashdreams.runtime.inputs import CanonicalInputSchema, CanonicalInputWindow
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepRequirements, StepResult

APPLICATION_ENTRY_POINT_GROUP = "flashdreams.applications"
"""Entry-point group whose values expose a zero-argument ``create_app`` factory."""


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

    def close(self) -> None:
        """Release optional per-session resources."""


class IFlashDreamsApplication(ABC):
    """Application factory boundary independent of presentation backend."""

    @property
    @abstractmethod
    def input_schema(self) -> CanonicalInputSchema:
        """Declare the named canonical inputs consumed by this application."""

    @abstractmethod
    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse application arguments and validate startup state."""

    @abstractmethod
    def create_session(self) -> IFlashDreamsApplicationSession:
        """Create one isolated application session."""

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


def run_application(
    application_slug: str,
    commandline_args: Sequence[str] = (),
    *,
    io_factory: IOFactory | None = None,
    input_handler: InputHandler | None = None,
    output_sink: OutputSink | None = None,
) -> tuple[OutputArtifact, ...]:
    """Run an application with host-owned canonical input handling.

    Args:
        application_slug: Installed application or concrete demo slug.
        commandline_args: Arguments forwarded to the application.
        io_factory: Factory for input handling and output delivery.
        input_handler: Caller-owned input handler; ``None`` uses the factory.
        output_sink: Caller-owned output sink; ``None`` uses the factory.

    Returns:
        Persistent artifacts returned by the output sink.

    Raises:
        TypeError: The application, handler, sink, or input values violate their
            declared contracts.
        ValueError: Direct I/O objects are combined with ``io_factory``, or the
            current canonical inputs do not match the application schema.
    """
    application, slug_args = create_application(application_slug)
    input_schema = application.input_schema
    if not isinstance(input_schema, CanonicalInputSchema):
        raise TypeError(
            "IFlashDreamsApplication.input_schema must be a CanonicalInputSchema."
        )
    resolved_factory = _resolve_io_factory(
        io_factory=io_factory,
        input_handler=input_handler,
        output_sink=output_sink,
    )
    resolved_input = resolved_factory.create_input_handler(input_schema)
    resolved_output = resolved_factory.create_output_sink()
    if not isinstance(resolved_input, InputHandler):
        raise TypeError("IOFactory.create_input_handler() must return an InputHandler.")
    if not isinstance(resolved_output, OutputSink):
        raise TypeError("IOFactory.create_output_sink() must return an OutputSink.")

    application.init([*slug_args, *commandline_args])
    from flashdreams.runtime.demo.application_runtime import (
        run_batch_application_session,
        run_realtime_application_session,
    )

    if isinstance(resolved_factory, ApplicationWebRTCIOFactory):
        result = asyncio.run(
            run_realtime_application_session(
                application=application,
                input_handler=resolved_input,
                input_schema=input_schema,
                output_sink=resolved_output,
            )
        )
    else:
        result = run_batch_application_session(
            application=application,
            input_handler=resolved_input,
            input_schema=input_schema,
            output_sink=resolved_output,
        )
    if result.status != "completed":
        if result.error is not None:
            raise result.error
        raise RuntimeError(
            result.reason or f"Application run ended with {result.status}."
        )
    return tuple(result.artifacts)


def _resolve_io_factory(
    *,
    io_factory: IOFactory | None,
    input_handler: InputHandler | None,
    output_sink: OutputSink | None,
) -> IOFactory:
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
) -> tuple[IOFactory, list[str]]:
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
        return (
            CallableIOFactory(lambda _schema: NullInputHandler(), NullOutputSink),
            application_args,
        )
    if output_kind == "mp4":
        path = output_path or Path("outputs") / f"{application_slug}.mp4"
        return Mp4IOFactory(output_path=path, fps=output_fps), application_args
    if output_kind == "webrtc":
        if not 1 <= port <= 65535:
            raise ValueError("--port must be between 1 and 65535.")
        return (
            ApplicationWebRTCIOFactory(
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
]
