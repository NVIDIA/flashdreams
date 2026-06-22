# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

from flashdreams.infra.pipeline import StreamInferencePipelineCache

from ltx_video.encoder import LTXConditionings
from ltx_video.kv_cache import LTXKVCache, PastKV


@dataclass(kw_only=True)
class LTXPipelineCache(StreamInferencePipelineCache):
    """Mutable state shared across AR steps."""

    cond: Optional[LTXConditionings] = None
    kv: LTXKVCache = field(default_factory=LTXKVCache)
    decoded_chunks: list[Tensor] = field(default_factory=list)
    pending_kv: Optional[PastKV] = None

    transformer_cache: Any = field(default_factory=dict)
    encoder_cache: Any = None
    decoder_cache: Any = None
