# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM diffusion config bound to the shared FlashDreams diffusion model."""

from __future__ import annotations

from dataclasses import dataclass, field

from flashdreams.infra.diffusion.model import DiffusionModel, DiffusionModelConfig


@dataclass(kw_only=True)
class SanaWMDiffusionModelConfig(DiffusionModelConfig):
    """Diffusion model config for SANA-WM's Stage-1 sampler.

    Sana-specific behavior lives in the configured transformer and scheduler;
    this config intentionally instantiates the common FlashDreams
    :class:`DiffusionModel` instead of a custom ``generate`` implementation.
    """

    _target: type[DiffusionModel] = field(default_factory=lambda: DiffusionModel)


__all__ = ["SanaWMDiffusionModelConfig"]
