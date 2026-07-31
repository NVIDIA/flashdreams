# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-facing configuration envelope."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

ExecutionBackend = Literal["local", "local-distributed", "external", "hosted"]
"""Execution backend families the v0 envelope leaves room for."""

Precision = Literal["auto", "fp32", "fp16", "bf16"]
"""Coarse runtime precision choices."""


@dataclass(frozen=True, kw_only=True, slots=True)
class InferenceConfig:
    """Model/runtime execution settings.

    Prompts, user controls, browser settings, output paths, and benchmark
    directories intentionally live outside this object.
    """

    model_id: str
    """Stable model or adapter identity."""

    preset_id: str | None = None
    """Optional preset identity under :attr:`model_id`."""

    checkpoint: str | Path | None = None
    """Optional checkpoint or model-asset selector understood by the adapter."""

    backend: ExecutionBackend = "local"
    """Runtime backend family."""

    device: str | None = None
    """Optional device selector such as ``cuda`` or ``cuda:0``."""

    precision: Precision = "auto"
    """Preferred compute precision."""

    compile: bool | None = None
    """Whether model compilation is requested, disabled, or left to the adapter."""

    cuda_graph: bool | None = None
    """Whether CUDA graph capture is requested, disabled, or left to the adapter."""

    attention_backend: str | None = None
    """Optional attention implementation selector."""

    cache_policy: str | None = None
    """Optional cache policy selector."""

    runtime_options: Mapping[str, Any] = field(default_factory=dict)
    """Adapter/backend-specific runtime options."""

    resource_hints: Mapping[str, Any] = field(default_factory=dict)
    """Cheap resource hints for launchers, schedulers, or hosted backends."""

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("InferenceConfig.model_id must be non-empty.")
