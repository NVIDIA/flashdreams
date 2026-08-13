# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public demo application contracts.

This module is the author-facing facade above ``flashdreams.runtime.demo``. The
runtime package owns execution, providers, sinks, drivers, and worker affinity;
these protocols name the smaller surface demo applications should eventually
implement directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.demo.outputs import (
    OutputDecision,
    OutputSink,
    SessionInfo,
)
from flashdreams.runtime.demo.session_inputs import UserInputWindow
from flashdreams.runtime.demo.spec import DemoAdapter, DemoSpec, PreparedScenario
from flashdreams.runtime.inputs import InferenceInput
from flashdreams.runtime.interfaces import InferenceRuntime, InferenceSession
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import (
    StepRequest,
    StepRequirements,
    StepResult,
    step_requirements_from_request,
)


@runtime_checkable
class ApplicationSession(Protocol):
    """One model session exposed through the public demo API."""

    def init(self) -> None:
        """Initialize per-session resources before the first step."""
        ...

    def session_info(self) -> SessionInfo:
        """Return output-facing metadata known after session setup."""
        ...

    def next_step_requirements(self) -> StepRequirements | None:
        """Return requirements for the next step, or ``None`` when complete."""
        ...

    def step(self, model_input: InferenceInput) -> StepResult:
        """Run one inference step from model-facing inputs."""
        ...

    def reset(self, model_input: InferenceInput | None = None) -> None:
        """Reset rollout state when the backend supports it."""
        ...

    def close(self) -> None:
        """Release per-session resources."""
        ...


@runtime_checkable
class Application(Protocol):
    """Public demo application facade."""

    def init(self, launch_args: Sequence[str]) -> None:
        """Initialize application-level launch state."""
        ...

    def create_session(self) -> ApplicationSession:
        """Create one model session.

        Public runners must call this on the same worker that will execute the
        session so model construction and stepping share worker affinity.
        """
        ...

    def close(self) -> None:
        """Release application-level resources after all sessions have closed."""
        ...


@runtime_checkable
class IOHandler(Protocol):
    """Public facade over runtime input, output, and transport edges."""

    def open(self, session_info: SessionInfo) -> None:
        """Prepare input/output resources after session setup."""
        ...

    def next_window(self, requirements: StepRequirements) -> UserInputWindow:
        """Return the user-input window selected for one model step.

        Model-specific conversion from this window to ``InferenceInput`` remains
        the responsibility of ``ModelInputProvider`` in the runtime layer.
        """
        ...

    def get_user_input_state(self, modality: str, name: str) -> Any:
        """Return the current named input state for interactive applications."""
        ...

    def begin_generation(self, generation: int) -> None:
        """Start a new output generation."""
        ...

    def emit_chunk(self, result: StepResult) -> OutputDecision:
        """Deliver one generated step result."""
        ...

    def should_exit(self) -> bool:
        """Return whether the surrounding run should stop."""
        ...

    def close(self) -> Sequence[OutputArtifact]:
        """Finalize resources and return produced artifacts."""
        ...


@runtime_checkable
class FrameOutputSink(Protocol):
    """Narrow file/comparison tail used by higher-level output handlers."""

    def handle_output(self, timestamp_s: float, chunk: StepResult) -> None:
        """Consume one timestamped generated chunk."""
        ...


@dataclass(slots=True)
class InferenceSessionApplicationAdapter:
    """Adapt an existing runtime session to ``ApplicationSession``."""

    session: InferenceSession

    def init(self) -> None:
        init = getattr(self.session, "init", None)
        if callable(init):
            init()

    def session_info(self) -> SessionInfo:
        session_info = getattr(self.session, "session_info", None)
        if not callable(session_info):
            return SessionInfo()
        value = session_info()
        if not isinstance(value, SessionInfo):
            raise TypeError(
                "session.session_info() must return SessionInfo, "
                f"got {type(value).__name__}."
            )
        return value

    def next_step_requirements(self) -> StepRequirements | None:
        next_requirements = getattr(self.session, "next_step_requirements", None)
        if callable(next_requirements):
            value = next_requirements()
            if value is None or isinstance(value, StepRequirements):
                return value
            raise TypeError(
                "session.next_step_requirements() must return StepRequirements "
                f"or None, got {type(value).__name__}."
            )

        request = self.session.next_step_request()
        if request is None:
            return None
        if not isinstance(request, StepRequest):
            raise TypeError(
                "session.next_step_request() must return StepRequest or None, "
                f"got {type(request).__name__}."
            )
        return step_requirements_from_request(
            request,
            allow_user_input_window=True,
        )

    def step(self, model_input: InferenceInput) -> StepResult:
        return self.session.step(model_input)

    def reset(self, model_input: InferenceInput | None = None) -> None:
        self.session.reset(model_input)

    def close(self) -> None:
        self.session.close()


@dataclass(slots=True)
class DemoAdapterApplication:
    """Adapt an existing ``DemoAdapter`` to the public ``Application`` shape."""

    adapter: DemoAdapter
    spec: DemoSpec
    _scenario: PreparedScenario | None = field(default=None, init=False, repr=False)
    _runtimes: list[InferenceRuntime] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def init(self, launch_args: Sequence[str]) -> None:
        if launch_args:
            raise ValueError(
                "DemoAdapterApplication does not support launch arguments; "
                "configure the DemoSpec before constructing the application."
            )
        config = _require_config(self.spec)
        self.adapter.validate_config(config)
        self._scenario = self.adapter.prepare_scenario(self.spec)

    def create_session(self) -> ApplicationSession:
        scenario = self._scenario
        if scenario is None:
            self.init(())
            scenario = self._scenario
        if scenario is None:
            raise RuntimeError("DemoAdapterApplication failed to prepare a scenario.")
        runtime = self.adapter.create_runtime(_require_config(self.spec))
        self._runtimes.append(runtime)
        return InferenceSessionApplicationAdapter(
            runtime.start_session(scenario.initial_inputs)
        )

    def start_session(self, inputs: InferenceInput) -> ApplicationSession:
        """Create a runtime session from runner-provided initial model inputs."""
        scenario = self._scenario
        if scenario is None:
            self.init(())
            scenario = self._scenario
        if scenario is None:
            raise RuntimeError("DemoAdapterApplication failed to prepare a scenario.")
        runtime = self.adapter.create_runtime(_require_config(self.spec))
        self._runtimes.append(runtime)
        return InferenceSessionApplicationAdapter(runtime.start_session(inputs))

    @property
    def prepared_scenario(self) -> PreparedScenario | None:
        """Return the scenario materialized by ``init(...)``, if any."""
        return self._scenario

    def close(self) -> None:
        errors: list[Exception] = []
        while self._runtimes:
            runtime = self._runtimes.pop()
            try:
                runtime.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"DemoAdapterApplication.close failed for {len(errors)} runtime(s)."
            ) from errors[0]


@dataclass(slots=True)
class RuntimeOutputSinkFrameAdapter:
    """Adapt a runtime ``OutputSink`` to the narrow frame-output tail."""

    output_sink: OutputSink

    def handle_output(self, timestamp_s: float, chunk: StepResult) -> None:
        del timestamp_s
        self.output_sink.write(chunk)


def _require_config(spec: DemoSpec) -> InferenceConfig:
    config = spec.config
    if config is None:
        raise RuntimeError("DemoSpec.config must be populated before use.")
    return config


IApplication = Application
IApplicationSession = ApplicationSession
IOutputSink = FrameOutputSink


__all__ = [
    "Application",
    "ApplicationSession",
    "DemoAdapterApplication",
    "FrameOutputSink",
    "IApplication",
    "IApplicationSession",
    "IOutputSink",
    "IOHandler",
    "InferenceSessionApplicationAdapter",
    "RuntimeOutputSinkFrameAdapter",
]
