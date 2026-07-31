# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental inference runtime API envelope.

This package defines the small v0 boundary above ``flashdreams.infra``. It is
intentionally additive while integrations migrate onto it.
"""

from flashdreams.runtime.config import ExecutionBackend, InferenceConfig, Precision
from flashdreams.runtime.inputs import (
    InputField,
    ModelInputs,
    ModelInputSchema,
    TimeWindow,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.interfaces import (
    InferenceRuntime,
    InferenceSession,
    ModelAdapter,
)
from flashdreams.runtime.mapping import IdentityInputMapping, InputMapping
from flashdreams.runtime.metrics import (
    InMemoryMetricsRecorder,
    MetricsRecorder,
    NullMetricsRecorder,
    RuntimeMetricSample,
)
from flashdreams.runtime.output import NullOutputTarget, OutputArtifact, OutputTarget
from flashdreams.runtime.types import StepRequest, StepResult

__all__ = [
    "ExecutionBackend",
    "IdentityInputMapping",
    "InferenceConfig",
    "InferenceRuntime",
    "InferenceSession",
    "InMemoryMetricsRecorder",
    "InputField",
    "InputMapping",
    "MetricsRecorder",
    "ModelAdapter",
    "ModelInputs",
    "ModelInputSchema",
    "NullMetricsRecorder",
    "NullOutputTarget",
    "OutputArtifact",
    "OutputTarget",
    "Precision",
    "RuntimeMetricSample",
    "StepRequest",
    "StepResult",
    "TimeWindow",
    "UserInputEvent",
    "UserInputs",
    "UserInputSchema",
]
