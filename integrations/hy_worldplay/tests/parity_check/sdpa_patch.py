# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patch vendor sageattention to use cuDNN SDPA (parity diagnostic).

Vendor uses ``sageattention.sageattn`` (INT8/FP8 matmuls); native uses
``F.scaled_dot_product_attention`` (bf16 + fp32 accumulation). Set
``HY_VENDOR_SDPA=1`` and install this patch to force vendor onto the
same kernel so vendor-vs-native diffs isolate non-attention drift.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F


def enabled() -> bool:
    return os.environ.get("HY_VENDOR_SDPA", "") == "1"


def _sdpa_replacement(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    **_unused: Any,
) -> torch.Tensor:
    # ``HND`` (``[batch, num_heads, seqlen, head_dim]``) matches SDPA's
    # expected layout directly; ``NHD`` would need a transpose but
    # vendor's dits never use it.
    if tensor_layout not in ("HND",):
        raise NotImplementedError(
            f"sdpa_patch only supports tensor_layout='HND'; got {tensor_layout!r}."
        )
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)


def install_sdpa_patch() -> None:
    if not enabled():
        return
    import sageattention

    sageattention.sageattn = _sdpa_replacement
    # Also rebind any ``from sageattention import sageattn`` already
    # captured by the vendor dit module.
    try:
        from wan.models.dits import arwan_w_action_w_mem_relative_rope as _vendor_mod

        _vendor_mod.sageattn = _sdpa_replacement
    except ImportError:
        pass
    print(
        "[sdpa_patch] sageattention.sageattn -> F.scaled_dot_product_attention",
        flush=True,
    )
