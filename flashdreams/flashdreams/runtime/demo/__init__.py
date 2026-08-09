# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental shared demo API above the inference runtime API."""

from flashdreams.runtime.demo.drivers import (
    BatchSessionDriver,
    DriverInvariantError,
    run_demo_session,
    run_demo_session_async,
)
from flashdreams.runtime.demo.host import (
    ModelWarmupPlan,
    RuntimeHost,
    WarmupSessionInputs,
)
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
    AsyncSessionDriver,
    DefaultErrorPolicy,
    ErrorAction,
    InMemorySessionMetricsRecorder,
    MetricsSnapshot,
    NoopTransportService,
    RunContext,
    RunMode,
    RunModeWarmup,
    RunResult,
    RunSummary,
    SessionDriver,
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
    "AsyncSessionDriver",
    "InMemorySessionMetricsRecorder",
    "InputSource",
    "MetricsSnapshot",
    "ModelWarmupPlan",
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
    "RunModeWarmup",
    "RunResult",
    "RunSummary",
    "RuntimeHost",
    "SessionEdges",
    "SessionDriver",
    "SessionInfo",
    "SingleSessionAdmissionPolicy",
    "StepOutcome",
    "StepPipeline",
    "UserInputWindow",
    "WarmupSessionInputs",
    "WebRTCAppResources",
    "WebRTCOutputSpec",
    "build_output_target",
    "run_demo_session",
    "run_demo_session_async",
    "run_replay_demo",
]
