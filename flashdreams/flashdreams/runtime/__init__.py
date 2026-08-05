# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental inference runtime API envelope.

This package defines the small v0 boundary above ``flashdreams.infra``. It is
intentionally additive while integrations migrate onto it.
"""

from flashdreams.runtime.canonical import (
    DEFAULT_DRIVING_BINDINGS,
    DRIVER_COMMAND,
    DeviceConverter,
    DeviceConverterSchema,
    InputCanonicalizer,
    KeyboardToDriverCommand,
    ScriptedModality,
)
from flashdreams.runtime.config import ExecutionBackend, InferenceConfig, Precision
from flashdreams.runtime.inputs import (
    INPUT_PHASES,
    SESSION_START_ONLY,
    CanonicalInputs,
    CanonicalInputSchema,
    CanonicalModality,
    InferenceInput,
    InferenceInputSchema,
    InputField,
    InputPhase,
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
    validate_phase,
)
from flashdreams.runtime.interfaces import (
    InferenceRuntime,
    InferenceSession,
    ModelAdapter,
)
from flashdreams.runtime.mapping import (
    DeclaresMappingSchema,
    IdentityInputMapping,
    InputMapping,
    InputMappingSchema,
    MappingCompatibility,
    check_mapping_compatibility,
    check_mapping_set_compatibility,
    combine_mapping_schemas,
    undeclared_inference_inputs,
)
from flashdreams.runtime.metrics import (
    InMemoryMetricsRecorder,
    MetricsRecorder,
    NullMetricsRecorder,
    RuntimeMetricSample,
)
from flashdreams.runtime.output import NullOutputTarget, OutputArtifact, OutputTarget
from flashdreams.runtime.types import StepRequest, StepResult

__all__ = [
    "CanonicalInputs",
    "CanonicalInputSchema",
    "CanonicalModality",
    "check_mapping_compatibility",
    "check_mapping_set_compatibility",
    "combine_mapping_schemas",
    "DeclaresMappingSchema",
    "DEFAULT_DRIVING_BINDINGS",
    "DeviceConverter",
    "DeviceConverterSchema",
    "DRIVER_COMMAND",
    "ExecutionBackend",
    "IdentityInputMapping",
    "InferenceConfig",
    "InferenceInput",
    "InferenceInputSchema",
    "InferenceRuntime",
    "InferenceSession",
    "InMemoryMetricsRecorder",
    "INPUT_PHASES",
    "InputCanonicalizer",
    "InputField",
    "InputMapping",
    "InputMappingSchema",
    "InputPhase",
    "KeyboardToDriverCommand",
    "MappingCompatibility",
    "MetricsRecorder",
    "ModelAdapter",
    "NullMetricsRecorder",
    "NullOutputTarget",
    "OutputArtifact",
    "OutputTarget",
    "Precision",
    "RuntimeMetricSample",
    "ScriptedModality",
    "SESSION_START_ONLY",
    "StepRequest",
    "StepResult",
    "TimeWindow",
    "undeclared_inference_inputs",
    "UserInputCapability",
    "UserInputEvent",
    "UserInputs",
    "UserInputSchema",
    "validate_phase",
]
