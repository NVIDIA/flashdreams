# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public demo application authoring API."""

from flashdreams.demo.app import (
    DemoApplication,
    create_demo_application,
    run_replay_application,
)
from flashdreams.demo.application import (
    Application,
    ApplicationSession,
    DemoAdapterApplication,
    FrameOutputSink,
    IApplication,
    IApplicationSession,
    InferenceSessionApplicationAdapter,
    IOHandler,
    IOutputSink,
    RuntimeOutputSinkFrameAdapter,
)
from flashdreams.demo.inputs import (
    InputName,
    InputStateDecoder,
    InputStateDecoderRegistry,
    KeyboardInputState,
    KeyboardInputStateDecoder,
    SnapshotInputStateDecoder,
    create_default_input_state_decoder_registry,
    input_state_from_window,
)
from flashdreams.demo.io import (
    CallbackIOHandlerServer,
    IOHandlerServer,
    NativeWindowIOHandler,
    ReplayIOHandler,
    WebRTCIOHandlerServer,
    create_native_window_io_handler,
    create_replay_io_handler,
    create_webrtc_io_handler,
)
from flashdreams.demo.runner import Runner
from flashdreams.runtime.demo.outputs import (
    BenchmarkStatsOutputSink,
    ComparisonOutputMismatchError,
    ComparisonOutputSink,
    FileOutputSink,
)

__all__ = [
    "Application",
    "ApplicationSession",
    "BenchmarkStatsOutputSink",
    "CallbackIOHandlerServer",
    "ComparisonOutputMismatchError",
    "ComparisonOutputSink",
    "DemoAdapterApplication",
    "DemoApplication",
    "FileOutputSink",
    "FrameOutputSink",
    "IApplication",
    "IApplicationSession",
    "IOutputSink",
    "IOHandler",
    "IOHandlerServer",
    "InputName",
    "InputStateDecoder",
    "InputStateDecoderRegistry",
    "InferenceSessionApplicationAdapter",
    "KeyboardInputState",
    "KeyboardInputStateDecoder",
    "NativeWindowIOHandler",
    "ReplayIOHandler",
    "RuntimeOutputSinkFrameAdapter",
    "Runner",
    "SnapshotInputStateDecoder",
    "WebRTCIOHandlerServer",
    "create_native_window_io_handler",
    "create_default_input_state_decoder_registry",
    "create_demo_application",
    "create_replay_io_handler",
    "create_webrtc_io_handler",
    "input_state_from_window",
    "run_replay_application",
]
