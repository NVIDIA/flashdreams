# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental shared demo API above the inference runtime API."""

from flashdreams.runtime.demo.drivers import (
    BatchSessionDriver,
    DriverInvariantError,
    run_demo_session,
)
from flashdreams.runtime.demo.host import RuntimeHost
from flashdreams.runtime.demo.outputs import (
    NullOutputSink,
    OutputDecision,
    OutputSink,
    SessionInfo,
    build_output_target,
)
from flashdreams.runtime.demo.pipeline import StepOutcome, StepPipeline
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.runtime.demo.run_modes import (
    DefaultErrorPolicy,
    ErrorAction,
    InMemorySessionMetricsRecorder,
    MetricsSnapshot,
    NoopTransportService,
    RunContext,
    RunMode,
    RunResult,
    RunSummary,
    SessionEdges,
    SingleSessionAdmissionPolicy,
)
from flashdreams.runtime.demo.session_inputs import (
    BatchInputSource,
    ControlDecision,
    InputSource,
    ModelInputProvider,
    PreparedStep,
    RealtimeInputSource,
    UserInputWindow,
)
from flashdreams.runtime.demo.spec import (
    DemoAdapter,
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    OutputSpec,
    PreparedScenario,
    WebRTCAppResources,
    WebRTCOutputSpec,
)

__all__ = [
    "BatchInputSource",
    "BatchSessionDriver",
    "ControlDecision",
    "DefaultErrorPolicy",
    "DemoAdapter",
    "DemoSpec",
    "DriverInvariantError",
    "ErrorAction",
    "InMemorySessionMetricsRecorder",
    "InputSource",
    "MetricsSnapshot",
    "ModelInputProvider",
    "Mp4OutputSpec",
    "NoopTransportService",
    "NullOutputSpec",
    "NullOutputSink",
    "OutputDecision",
    "OutputSpec",
    "OutputSink",
    "PreparedScenario",
    "PreparedStep",
    "RealtimeInputSource",
    "RunContext",
    "RunMode",
    "RunResult",
    "RunSummary",
    "RuntimeHost",
    "SessionEdges",
    "SessionInfo",
    "SingleSessionAdmissionPolicy",
    "StepOutcome",
    "StepPipeline",
    "UserInputWindow",
    "WebRTCAppResources",
    "WebRTCOutputSpec",
    "build_output_target",
    "run_demo_session",
    "run_replay_demo",
]
