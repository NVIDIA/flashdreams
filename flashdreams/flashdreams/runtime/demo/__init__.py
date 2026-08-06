# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental shared demo API above the inference runtime API."""

from flashdreams.runtime.demo.app import run_flashdreams_demo, serve_flashdreams_demo
from flashdreams.runtime.demo.outputs import build_output_target
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.runtime.demo.spec import (
    DemoAdapter,
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    OutputSpec,
    PreparedScenario,
    WebRTCOutputSpec,
)

__all__ = [
    "DemoAdapter",
    "DemoSpec",
    "Mp4OutputSpec",
    "NullOutputSpec",
    "OutputSpec",
    "PreparedScenario",
    "WebRTCOutputSpec",
    "build_output_target",
    "run_flashdreams_demo",
    "run_replay_demo",
    "serve_flashdreams_demo",
]
