# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental OmniDreams demo adapter built on ``flashdreams.runtime.demo``."""

from omnidreams.demo.adapter import OmnidreamsDemoAdapter
from omnidreams.demo.providers import (
    LudusSceneConditioningProvider,
    PrecomputedHDMapProvider,
)
from omnidreams.demo.spec import (
    DEFAULT_OMNIDREAMS_PRESET,
    OMNIDREAMS_CONDITIONING_LUDUS,
    OMNIDREAMS_CONDITIONING_MODES,
    OMNIDREAMS_CONDITIONING_PRECOMPUTED,
    OMNIDREAMS_MODEL_ID,
    OmnidreamsKeyboardTraceEvent,
    OmnidreamsLudusReplayScenario,
    OmnidreamsReplayScenario,
    OmnidreamsWebRTCScenario,
)

__all__ = [
    "DEFAULT_OMNIDREAMS_PRESET",
    "OMNIDREAMS_CONDITIONING_LUDUS",
    "OMNIDREAMS_CONDITIONING_MODES",
    "OMNIDREAMS_CONDITIONING_PRECOMPUTED",
    "OMNIDREAMS_MODEL_ID",
    "LudusSceneConditioningProvider",
    "OmnidreamsDemoAdapter",
    "OmnidreamsKeyboardTraceEvent",
    "OmnidreamsLudusReplayScenario",
    "OmnidreamsReplayScenario",
    "OmnidreamsWebRTCScenario",
    "PrecomputedHDMapProvider",
]
