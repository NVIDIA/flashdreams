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

"""Transport-neutral FlashDreams application hosting and I/O API."""

from flashdreams.demo.application import (
    APPLICATION_ENTRY_POINT_GROUP,
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    create_application,
    run_application,
)
from flashdreams.demo.factories import (
    ApplicationWebRTCIOFactory,
    CallableIOFactory,
    LocalWindowIOFactory,
    Mp4IOFactory,
    NullInputHandler,
    ProvidedIOFactory,
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
    DeferredWebRTCOutputSink,
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
    "ApplicationWebRTCIOFactory",
    "BenchmarkStatsOutputSink",
    "CallableIOFactory",
    "CompositeOutputSink",
    "CompositeOutputSinkError",
    "DeferredWebRTCOutputSink",
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
    "NullOutputSink",
    "OutputDecision",
    "OutputSink",
    "ProvidedIOFactory",
    "SessionInfo",
    "SlangPyLocalInputHandler",
    "LocalWindowOutputSink",
    "WebRTCIOFactory",
    "WebRTCOutputSink",
    "build_benchmark_output_sink",
    "create_application",
    "run_application",
]
