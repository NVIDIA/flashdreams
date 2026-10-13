# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral FlashDreams application hosting and I/O API."""

from flashdreams.demo.application import (
    APPLICATION_ENTRY_POINT_GROUP,
    ApplicationWarmupSessionInputs,
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    create_application,
    run_application,
)
from flashdreams.demo.factories import (
    CallableIOFactory,
    LocalWindowIOFactory,
    Mp4IOFactory,
    NullInputHandler,
    NullIOFactory,
    ProvidedIOFactory,
    WebRTCApplicationServing,
    WebRTCIOFactory,
)
from flashdreams.demo.io import (
    InputHandler,
    IOFactory,
    OutputDecision,
    OutputSink,
    SessionInfo,
)
from flashdreams.demo.local_input import SlangPyLocalInputHandler
from flashdreams.demo.outputs import (
    BenchmarkStatsOutputSink,
    CompositeOutputSink,
    CompositeOutputSinkError,
    LocalWindowOutputSink,
    Mp4OutputSink,
    NullOutputSink,
    WebRTCOutputSink,
    build_benchmark_output_sink,
)
from flashdreams.runtime.inputs import (
    CanonicalInputs,
    CanonicalInputSchema,
    CanonicalInputWindow,
)

__all__ = [
    "APPLICATION_ENTRY_POINT_GROUP",
    "ApplicationWarmupSessionInputs",
    "BenchmarkStatsOutputSink",
    "CallableIOFactory",
    "CompositeOutputSink",
    "CompositeOutputSinkError",
    "IFlashDreamsApplication",
    "IFlashDreamsApplicationSession",
    "IOFactory",
    "CanonicalInputWindow",
    "CanonicalInputs",
    "CanonicalInputSchema",
    "InputHandler",
    "LocalWindowIOFactory",
    "Mp4IOFactory",
    "Mp4OutputSink",
    "NullInputHandler",
    "NullIOFactory",
    "NullOutputSink",
    "OutputDecision",
    "OutputSink",
    "ProvidedIOFactory",
    "SessionInfo",
    "SlangPyLocalInputHandler",
    "LocalWindowOutputSink",
    "WebRTCApplicationServing",
    "WebRTCIOFactory",
    "WebRTCOutputSink",
    "build_benchmark_output_sink",
    "create_application",
    "run_application",
]
