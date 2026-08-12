# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Lingbot demo adapter built on ``flashdreams.runtime.demo``."""

from lingbot.demo.adapter import LingbotDemoAdapter
from lingbot.demo.providers import LingbotInputProvider
from lingbot.demo.spec import (
    DEFAULT_LINGBOT_PRESET,
    LINGBOT_MODEL_ID,
    LingbotReplayInputs,
    LingbotWebRTCScenario,
)

__all__ = [
    "DEFAULT_LINGBOT_PRESET",
    "LINGBOT_MODEL_ID",
    "LingbotDemoAdapter",
    "LingbotInputProvider",
    "LingbotReplayInputs",
    "LingbotWebRTCScenario",
]
