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

"""Application adapters for the shared batch and realtime session drivers."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flashdreams.demo.io import InputHandler, OutputDecision, OutputSink, SessionInfo
from flashdreams.runtime.inputs import (
    CanonicalInputSchema,
    CanonicalInputWindow,
    InferenceInput,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepRequest, StepRequirements, StepResult

from .drivers import BatchSessionDriver, RealtimeSessionDriver
from .host import RuntimeHost
from .pipeline import StepPipeline
from .run_modes import (
    InMemorySessionMetricsRecorder,
    RunResult,
    SessionEdges,
)
from .session_inputs import (
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from .timing import (
    AlwaysActiveActivationPolicy,
    CatchUpDecision,
    CatchUpPolicy,
    RealtimeClock,
    RealtimeWindowResult,
)

if TYPE_CHECKING:
    from flashdreams.demo.application import (
        IFlashDreamsApplication,
        IFlashDreamsApplicationSession,
    )

_CANONICAL_INPUT_WINDOW_KEY = "flashdreams.application.canonical_input_window"
"""Private inference-input key used only across the application adapter boundary."""


# This is a hacky shim to get legacy & `IFlashdreams*` code working together with the `*SessionDriver` code.
# This will and should be removed ASAP.
class _ApplicationRuntime:
    """Create application sessions through the driver's hosted runtime boundary."""

    def __init__(self, application: "IFlashDreamsApplication") -> None:
        self._application = application

    def start_session(self, inputs: InferenceInput) -> "_ApplicationSession":
        """Create and initialize one application session."""
        del inputs
        from flashdreams.demo.application import IFlashDreamsApplicationSession

        session = self._application.create_session()
        if not isinstance(session, IFlashDreamsApplicationSession):
            raise TypeError(
                "IFlashDreamsApplication.create_session() must return "
                "IFlashDreamsApplicationSession."
            )
        session.init()
        return _ApplicationSession(session)

    def close(self) -> None:
        """Close runtime-owned resources after the driver closes its session."""


class _ApplicationSession:
    """Expose an application session through the inference-session contract."""

    def __init__(self, session: "IFlashDreamsApplicationSession") -> None:
        self._session = session

    def session_info(self) -> SessionInfo:
        """Return validated metadata for input and output setup."""
        value = self._session.session_info()
        if not isinstance(value, SessionInfo):
            raise TypeError(
                "IFlashDreamsApplicationSession.session_info() must return SessionInfo."
            )
        return value

    def next_step_requirements(self) -> StepRequirements | None:
        """Return the application requirements for the driver's next step."""
        return self._session.next_step_requirements()

    def next_step_request(self) -> StepRequest | None:
        """Project application requirements onto the legacy session request."""
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
        return self._session.step(canonical_inputs)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Reject resets because application sessions do not declare reset support."""
        del inputs
        raise RuntimeError("FlashDreams application sessions do not support reset.")

    def close(self) -> None:
        """Close the underlying application session."""
        self._session.close()


class _ApplicationInputSource:
    """Read and validate canonical inputs for batch and realtime drivers."""

    is_finite = False
    is_deterministic = False
    user_input_schema = UserInputSchema()

    def __init__(
        self,
        *,
        handler: InputHandler,
        schema: CanonicalInputSchema,
    ) -> None:
        self._handler = handler
        self._schema = schema

    def is_finished(self) -> bool:
        """Let the application session declare completion through next-step state."""
        return False

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        """Return the current validated canonical input window."""
        del request
        canonical_inputs = _current_application_inputs(
            self._handler,
            self._schema,
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

    async def next_realtime_window(
        self,
        *,
        request: StepRequirements,
        clock: RealtimeClock,
    ) -> RealtimeWindowResult:
        """Yield once, then return the current canonical realtime inputs."""
        del clock
        await asyncio.sleep(0)
        return RealtimeWindowResult(window=self.next_window(request))


class _ApplicationInputProvider:
    """Pass driver-owned canonical windows through to an application session."""

    capabilities = ProviderCapabilities(supports_realtime_clock=True)

    def prepare_initial_input(self) -> InferenceInput:
        """Return the empty initial input used to create an application session."""
        return InferenceInput()

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        """Wrap one canonical window in the inference-session envelope."""
        del request
        canonical_inputs = user_window.metadata.get(_CANONICAL_INPUT_WINDOW_KEY)
        if not isinstance(canonical_inputs, CanonicalInputWindow):
            raise TypeError(
                "Application input source must provide a CanonicalInputWindow."
            )
        return PreparedStep(
            inference_input=InferenceInput(
                step={_CANONICAL_INPUT_WINDOW_KEY: canonical_inputs}
            )
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Reject resets because application sessions do not declare reset support."""
        del inputs
        raise RuntimeError("FlashDreams application sessions do not support reset.")

    def close(self) -> None:
        """Leave I/O cleanup to the session-edge output wrapper."""


class _ApplicationOutputEdges:
    """Preserve application input/output open and close ordering for SessionEdges."""

    def __init__(self, *, input_handler: InputHandler, output_sink: OutputSink) -> None:
        self._input_handler = input_handler
        self._output_sink = output_sink
        self._generation: int | None = None

    @property
    def produces_artifacts(self) -> bool:
        """Return whether the wrapped output can produce persistent artifacts."""
        return self._output_sink.produces_artifacts

    def open(self, session_info: SessionInfo) -> None:
        """Open input before output and start the initial generation."""
        self._input_handler.open(session_info)
        self._output_sink.open(session_info)
        self.begin_generation(0)

    def begin_generation(self, generation: int) -> None:
        """Begin each output generation exactly once."""
        if generation == self._generation:
            return
        self._output_sink.begin_generation(generation)
        self._generation = generation

    def write(self, result: StepResult) -> OutputDecision:
        """Write one canonical result to the wrapped output sink."""
        return self._output_sink.write(result)

    def close(self) -> Sequence[OutputArtifact]:
        """Close output before input and return persistent artifacts."""
        try:
            return self._output_sink.close()
        finally:
            self._input_handler.close()


class _ApplicationRealtimeClock:
    """Apply application output backpressure without inventing a media timeline."""

    is_realtime = True
    is_deterministic = False

    def now(self) -> float:
        """Return the current monotonic time."""
        return time.monotonic()

    def anchor(self, wall_time_s: float) -> None:
        """Validate a clock anchor without creating an application timeline."""
        if not math.isfinite(wall_time_s):
            raise ValueError("wall_time_s must be finite.")

    async def wait_until_window_end(self, end_s: float) -> None:
        """Skip timeline pacing because the application output owns its cadence."""
        del end_s

    async def apply_backpressure(self, requested_s: float) -> None:
        """Delay the next step by the output-requested duration."""
        if not math.isfinite(requested_s) or requested_s < 0:
            raise ValueError("requested_s must be finite and >= 0.")
        await asyncio.sleep(requested_s)

    def catch_up(
        self,
        *,
        request: StepRequirements,
        max_lag_s: float,
        policy: CatchUpPolicy,
    ) -> CatchUpDecision:
        """Return an empty catch-up decision for output-paced applications."""
        del request, max_lag_s, policy
        return CatchUpDecision()


def run_batch_application_session(
    *,
    application: "IFlashDreamsApplication",
    input_handler: InputHandler,
    input_schema: CanonicalInputSchema,
    output_sink: OutputSink,
) -> RunResult:
    """Run one application through the shared batch driver."""
    host, provider, edges = _application_runtime_parts(
        application=application,
        input_handler=input_handler,
        input_schema=input_schema,
        output_sink=output_sink,
        realtime=False,
    )
    try:
        return BatchSessionDriver().run_one_session(
            host=host,
            provider=provider,
            session_edges=edges,
            pipeline=StepPipeline(),
        )
    finally:
        host.close()


async def run_realtime_application_session(
    *,
    application: "IFlashDreamsApplication",
    input_handler: InputHandler,
    input_schema: CanonicalInputSchema,
    output_sink: OutputSink,
) -> RunResult:
    """Run one application through the shared realtime driver."""
    host, provider, edges = _application_runtime_parts(
        application=application,
        input_handler=input_handler,
        input_schema=input_schema,
        output_sink=output_sink,
        realtime=True,
    )
    try:
        return await RealtimeSessionDriver().run_one_session(
            host=host,
            provider=provider,
            session_edges=edges,
            pipeline=StepPipeline(),
        )
    finally:
        host.close()


def _application_runtime_parts(
    *,
    application: "IFlashDreamsApplication",
    input_handler: InputHandler,
    input_schema: CanonicalInputSchema,
    output_sink: OutputSink,
    realtime: bool,
) -> tuple[RuntimeHost, _ApplicationInputProvider, SessionEdges]:
    source = _ApplicationInputSource(handler=input_handler, schema=input_schema)
    edges = SessionEdges(
        input_source=source,
        output_sink=_ApplicationOutputEdges(
            input_handler=input_handler,
            output_sink=output_sink,
        ),
        cleanup_tasks=set(),
        metrics=InMemorySessionMetricsRecorder(),
        clock=_ApplicationRealtimeClock() if realtime else None,
        activation=AlwaysActiveActivationPolicy() if realtime else None,
    )
    return (
        RuntimeHost(_ApplicationRuntime(application)),
        _ApplicationInputProvider(),
        edges,
    )


def _current_application_inputs(
    handler: InputHandler,
    input_schema: CanonicalInputSchema,
) -> CanonicalInputWindow:
    inputs = handler.current_inputs()
    if not isinstance(inputs, CanonicalInputWindow):
        raise TypeError(
            "InputHandler.current_inputs() must return CanonicalInputWindow."
        )
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
    return inputs


__all__ = [
    "run_batch_application_session",
    "run_realtime_application_session",
]
