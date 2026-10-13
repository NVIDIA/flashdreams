# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch and native cuDNN scaled-dot-product attention."""

from flashdreams.accelerated.multi_head_attention.cudnn.native_fp8 import (
    native_cudnn_fp8_sdpa,
)
from flashdreams.accelerated.multi_head_attention.cudnn.torch_sdpa import (
    torch_cudnn_sdpa,
)

__all__ = ["native_cudnn_fp8_sdpa", "torch_cudnn_sdpa"]
