# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton FlashAttention2 kernels."""

from flashdreams.accelerated.multi_head_attention.triton.flash_attention_2_kernel import (
    flash_attention_2,
)
from flashdreams.accelerated.multi_head_attention.triton.flash_attention_2_tma_kernel import (
    flash_attention_2_tma,
    is_tma_flash_attention_supported,
)

__all__ = [
    "flash_attention_2",
    "flash_attention_2_tma",
    "is_tma_flash_attention_supported",
]
