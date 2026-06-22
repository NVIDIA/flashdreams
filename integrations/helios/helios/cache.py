# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

from flashdreams.infra.pipeline import StreamInferencePipelineCache
from helios.encoder import HeliosConditionings


@dataclass(kw_only=True)
class HeliosPipelineCache(StreamInferencePipelineCache):
    """State shared across AR steps."""

    cond: Optional[HeliosConditionings] = None
    decoded_chunks: list[Tensor] = field(default_factory=list)
    history_frames: Optional[Tensor] = None
    pending_history: Optional[Tensor] = None

    transformer_cache: Any = field(default_factory=dict)
    encoder_cache: Any = None
    decoder_cache: Any = None
