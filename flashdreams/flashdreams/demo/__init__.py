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
