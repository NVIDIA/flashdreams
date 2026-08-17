# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental inference runtime API envelope.

This package defines the small v0 boundary above ``flashdreams.infra``. It is
intentionally additive while integrations migrate onto it.
"""

from flashdreams.runtime.canonical import (
    CAMERA_COMMAND,
    DEFAULT_CAMERA_BINDINGS,
    DEFAULT_DRIVING_BINDINGS,
    DRIVER_COMMAND,
    DeviceConverter,
    DeviceConverterSchema,
    InputCanonicalizer,
    KeyboardToCameraCommand,
    KeyboardToDriverCommand,
    ScriptedModality,
)
from flashdreams.runtime.config import ExecutionBackend, InferenceConfig, Precision
from flashdreams.runtime.inputs import (
    INPUT_PHASES,
    CanonicalInputs,
    CanonicalInputSchema,
    CanonicalInputWindow,
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
from flashdreams.runtime.keyboard import (
    DEFAULT_SUPPORTED_KEYS,
    DRIVING_SUPPORTED_KEYS,
    KEY_ALIASES,
    WSAD_SUPPORTED_KEYS,
    ImageRequest,
    KeyboardState,
    PromptRequest,
    ResetRequest,
    SparseInputSnapshot,
    normalize_key,
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
    MetricsSnapshot,
    NullMetricsRecorder,
    RuntimeMetricSample,
)
from flashdreams.runtime.output import NullOutputTarget, OutputArtifact, OutputTarget
from flashdreams.runtime.types import (
    StepRequest,
    StepRequirements,
    StepResult,
    step_requirements_from_request,
)
from flashdreams.runtime.video_output import Mp4VideoOutputTarget
from flashdreams.runtime.worker import ModelExecutionWorker, ThreadAffineRuntimeWorker

__all__ = [
    "CAMERA_COMMAND",
    "CanonicalInputWindow",
    "CanonicalInputs",
    "CanonicalInputSchema",
    "CanonicalModality",
    "check_mapping_compatibility",
    "check_mapping_set_compatibility",
    "combine_mapping_schemas",
    "DeclaresMappingSchema",
    "DEFAULT_CAMERA_BINDINGS",
    "DEFAULT_DRIVING_BINDINGS",
    "DEFAULT_SUPPORTED_KEYS",
    "DeviceConverter",
    "DeviceConverterSchema",
    "DRIVING_SUPPORTED_KEYS",
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
    "ImageRequest",
    "KEY_ALIASES",
    "KeyboardState",
    "KeyboardToCameraCommand",
    "KeyboardToDriverCommand",
    "MappingCompatibility",
    "MetricsRecorder",
    "MetricsSnapshot",
    "ModelAdapter",
    "ModelExecutionWorker",
    "Mp4VideoOutputTarget",
    "NullMetricsRecorder",
    "NullOutputTarget",
    "OutputArtifact",
    "OutputTarget",
    "Precision",
    "PromptRequest",
    "ResetRequest",
    "RuntimeMetricSample",
    "ScriptedModality",
    "SparseInputSnapshot",
    "StepRequest",
    "StepRequirements",
    "StepResult",
    "TimeWindow",
    "ThreadAffineRuntimeWorker",
    "step_requirements_from_request",
    "undeclared_inference_inputs",
    "UserInputCapability",
    "UserInputEvent",
    "UserInputs",
    "UserInputSchema",
    "WSAD_SUPPORTED_KEYS",
    "normalize_key",
    "validate_phase",
]
