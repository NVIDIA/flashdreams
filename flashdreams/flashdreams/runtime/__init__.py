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
from flashdreams.runtime.driving import (
    ControlSnapshot,
    DriverCommand,
    TrajectoryChunk,
    VehicleState,
)
from flashdreams.runtime.inputs import (
    INPUT_PHASES,
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
from flashdreams.runtime.interactive import (
    InteractiveEvent,
    InteractiveInferenceWorker,
    InteractiveSessionEnded,
    InteractiveSessionJob,
    InteractiveStep,
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
from flashdreams.runtime.output import (
    NullOutputTarget,
    OutputArtifact,
    OutputTarget,
    PollableOutputTarget,
)
from flashdreams.runtime.output_schema import (
    RGB_VIDEO,
    DeclaresInferenceOutput,
    DeclaresOutputRequirement,
    InferenceOutputSchema,
    OutputCompatibility,
    OutputTargetRequirement,
    check_output_compatibility,
    require_output_compatibility,
)
from flashdreams.runtime.runner import run_inference_session
from flashdreams.runtime.sources import (
    QueuedUserInputSource,
    SessionAwareUserInputSource,
    UserInputSource,
)
from flashdreams.runtime.types import StepRequest, StepResult
from flashdreams.runtime.video_output import Mp4VideoOutputTarget

__all__ = [
    "DEFAULT_DRIVING_BINDINGS",
    "DRIVER_COMMAND",
    "INPUT_PHASES",
    "RGB_VIDEO",
    "CanonicalInputSchema",
    "CanonicalInputs",
    "CanonicalModality",
    "ControlSnapshot",
    "DeclaresInferenceOutput",
    "DeclaresMappingSchema",
    "DeclaresOutputRequirement",
    "DeviceConverter",
    "DeviceConverterSchema",
    "DriverCommand",
    "ExecutionBackend",
    "IdentityInputMapping",
    "InMemoryMetricsRecorder",
    "InferenceConfig",
    "InferenceInput",
    "InferenceInputSchema",
    "InferenceOutputSchema",
    "InferenceRuntime",
    "InferenceSession",
    "InputCanonicalizer",
    "InputField",
    "InputMapping",
    "InputMappingSchema",
    "InputPhase",
    "InteractiveEvent",
    "InteractiveInferenceWorker",
    "InteractiveSessionEnded",
    "InteractiveSessionJob",
    "InteractiveStep",
    "KeyboardToDriverCommand",
    "MappingCompatibility",
    "MetricsRecorder",
    "ModelAdapter",
    "Mp4VideoOutputTarget",
    "NullMetricsRecorder",
    "NullOutputTarget",
    "OutputArtifact",
    "OutputCompatibility",
    "OutputTarget",
    "OutputTargetRequirement",
    "PollableOutputTarget",
    "Precision",
    "QueuedUserInputSource",
    "RuntimeMetricSample",
    "ScriptedModality",
    "SessionAwareUserInputSource",
    "StepRequest",
    "StepResult",
    "TimeWindow",
    "TrajectoryChunk",
    "UserInputCapability",
    "UserInputEvent",
    "UserInputSchema",
    "UserInputSource",
    "UserInputs",
    "VehicleState",
    "check_mapping_compatibility",
    "check_mapping_set_compatibility",
    "check_output_compatibility",
    "combine_mapping_schemas",
    "require_output_compatibility",
    "run_inference_session",
    "undeclared_inference_inputs",
    "validate_phase",
]
