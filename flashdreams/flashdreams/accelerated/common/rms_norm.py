# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RMS normalization for accelerated inference paths."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def rms_norm(input: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Apply RMSNorm through PyTorch's fused implementation.

    The fused BF16 reduction can round differently from a decomposed module, so
    integrations must quality-qualify model output before enabling this path.
    """
    if weight.ndim != 1 or weight.shape[0] != input.shape[-1]:
        raise ValueError(
            f"RMSNorm weight {tuple(weight.shape)} does not match input "
            f"dimension {input.shape[-1]}."
        )
    return F.rms_norm(input, (input.shape[-1],), weight, eps)
