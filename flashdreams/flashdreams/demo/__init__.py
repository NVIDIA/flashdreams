# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public demo application authoring API."""

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
from flashdreams.demo.io import (
    IOHandlerServer,
    NativeWindowIOHandler,
    ReplayIOHandler,
    WebRTCIOHandlerServer,
    create_native_window_io_handler,
    create_replay_io_handler,
    create_webrtc_io_handler,
)
from flashdreams.demo.runner import Runner

__all__ = [
    "Application",
    "ApplicationSession",
    "DemoAdapterApplication",
    "FrameOutputSink",
    "IApplication",
    "IApplicationSession",
    "IOutputSink",
    "IOHandler",
    "IOHandlerServer",
    "InferenceSessionApplicationAdapter",
    "NativeWindowIOHandler",
    "ReplayIOHandler",
    "RuntimeOutputSinkFrameAdapter",
    "Runner",
    "WebRTCIOHandlerServer",
    "create_native_window_io_handler",
    "create_replay_io_handler",
    "create_webrtc_io_handler",
]
