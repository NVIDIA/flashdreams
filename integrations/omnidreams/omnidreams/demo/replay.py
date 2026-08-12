# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility aliases for the OmniDreams runtime module."""

from __future__ import annotations

from .runtime import (
    OmnidreamsRuntime,
    OmnidreamsRuntimeOptions,
    OmnidreamsSession,
    OmnidreamsSessionScenario,
    PipelineFactory,
)

OmnidreamsReplayRuntimeOptions = OmnidreamsRuntimeOptions
OmnidreamsReplayRuntime = OmnidreamsRuntime
OmnidreamsReplaySession = OmnidreamsSession

__all__ = [
    "OmnidreamsReplayRuntime",
    "OmnidreamsReplayRuntimeOptions",
    "OmnidreamsReplaySession",
    "OmnidreamsRuntime",
    "OmnidreamsRuntimeOptions",
    "OmnidreamsSession",
    "OmnidreamsSessionScenario",
    "PipelineFactory",
]
