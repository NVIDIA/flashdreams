# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-facing configuration envelope."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from flashdreams.runtime._utils import freeze_mapping

ExecutionBackend = Literal["local", "local-distributed", "external", "hosted"]
"""Where and how inference compute is run."""

Precision = Literal["auto", "fp32", "fp16", "bf16"]
"""Coarse runtime precision choices."""


@dataclass(frozen=True, kw_only=True, slots=True)
class InferenceConfig:
    """Runtime settings that affect model execution.

    Prompts, user controls, browser settings, output paths, and benchmark
    directories intentionally live outside this object. The typed optimization
    fields cover common cross-backend knobs; open-ended adapter-specific choices
    can use :attr:`runtime_options`.
    """

    __hash__ = None

    model_id: str
    """Stable identity for the model adapter or runtime integration."""

    preset_id: str | None = None
    """Optional preset identity under :attr:`model_id`."""

    checkpoint: str | Path | None = None
    """Optional checkpoint or model-asset selector understood by the adapter."""

    backend: ExecutionBackend = "local"
    """Execution placement and backend family for inference compute."""

    device: str | None = None
    """Optional device selector such as ``cuda`` or ``cuda:0``; ``None`` leaves placement to the adapter/backend."""

    precision: Precision = "auto"
    """Preferred compute precision."""

    compile: bool | None = None
    """Optional - Whether model compilation is requested or disabled. `None` means left to the adapter to decide."""

    cuda_graph: bool | None = None
    """Optional - Whether CUDA graph capture is requested or disabled. `None` means left to the adapter to decide."""

    attention_backend: str | None = None
    """Optional attention implementation selector; ``None`` leaves the choice to the adapter."""

    cache_policy: str | None = None
    """Optional cache policy selector; ``None`` leaves the choice to the adapter."""

    seed: int | None = None
    """Optional seed used when resolving deterministic demo/runtime behavior."""

    runtime_options: Mapping[str, Any] = field(default_factory=dict)
    """Adapter/backend-specific runtime options."""

    resource_hints: Mapping[str, Any] = field(default_factory=dict)
    """Resource hints for launchers, schedulers, or hosted backends."""

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("InferenceConfig.model_id must be non-empty.")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("InferenceConfig.seed must be an integer.")
            if self.seed < 0:
                raise ValueError("InferenceConfig.seed must be >= 0.")
        object.__setattr__(
            self, "runtime_options", freeze_mapping(self.runtime_options)
        )
        object.__setattr__(self, "resource_hints", freeze_mapping(self.resource_hints))
