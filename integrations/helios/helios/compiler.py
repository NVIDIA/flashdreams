# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""torch.compile and FlashAttention helpers for Helios."""

from __future__ import annotations

import torch
import torch.nn as nn


def enable_flash_attention() -> None:
    """Enable FlashAttention via cuDNN SDPA backend."""
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    print("[Helios compiler] Flash attention (cuDNN) enabled")


def compile_transformer(transformer: nn.Module) -> nn.Module:
    """Compile the Helios DiT for repeated AR chunk calls."""
    compiled = torch.compile(
        transformer,
        mode="default",
        fullgraph=False,
        dynamic=True,
    )
    print("[Helios compiler] torch.compile applied to DiT transformer (dynamic=True)")
    return compiled
