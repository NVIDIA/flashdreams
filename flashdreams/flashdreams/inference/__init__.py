# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference runtime input contracts.

This package contains the lightweight ``UserInputs`` / ``ModelInputs``
contracts used by the experimental runtime API. The objects here describe
capabilities and mapping boundaries; model-specific tensor validation stays in
the model adapter or session implementation.
"""

from flashdreams.inference.inputs import (
    InputMapper,
    InputMapperSchema,
    InputPhase,
    MappingCompatibility,
    ModelInputField,
    ModelInputSchema,
    ModelInputs,
    StaticInputMapper,
    UserInputCapability,
    UserInputEvent,
    UserInputSchema,
    UserInputTrace,
    UserInputWindow,
    check_mapping_compatibility,
    check_mapping_set_compatibility,
    combine_mapper_schemas,
    missing_required_inputs,
    undeclared_model_inputs,
)

__all__ = [
    "InputMapper",
    "InputMapperSchema",
    "InputPhase",
    "MappingCompatibility",
    "ModelInputField",
    "ModelInputSchema",
    "ModelInputs",
    "StaticInputMapper",
    "UserInputCapability",
    "UserInputEvent",
    "UserInputSchema",
    "UserInputTrace",
    "UserInputWindow",
    "check_mapping_compatibility",
    "check_mapping_set_compatibility",
    "combine_mapper_schemas",
    "missing_required_inputs",
    "undeclared_model_inputs",
]
