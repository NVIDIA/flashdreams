# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 2b.6.2 attention-impl parity probe.

Monkey-patches vendor's ``sageattention.sageattn`` import to delegate
to ``torch.nn.functional.scaled_dot_product_attention``. SageAttention
uses INT8 / FP8 quantized matmuls; cudnn SDPA uses bf16 with fp32
accumulation. The two produce slightly different outputs per attention
call, which compounds across 30 transformer blocks and dominates the
residual chunk-0 + chunk-1+ numerical drift once the three structural
bugs (CFG, RNG, prefill block-forward) are closed.

Setting ``HY_VENDOR_SDPA=1`` together with this patch installed gives
us a vendor baseline that uses the same attention kernel as native
(both go through SDPA / cudnn) so we can isolate any *remaining*
non-attention divergence in the native port.
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
    # ``HND`` in SageAttention's API == ``[batch, num_heads, seqlen,
    # head_dim]`` which is exactly SDPA's expected layout. The
    # ``NHD`` layout would need a transpose; we don't see that in
    # vendor's dits so the simple passthrough below is sufficient.
    if tensor_layout not in ("HND",):
        raise NotImplementedError(
            f"sdpa_patch only supports tensor_layout='HND'; got {tensor_layout!r}."
        )
    return F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=is_causal
    )


def install_sdpa_patch() -> None:
    if not enabled():
        return
    # Rebind the vendor module's ``sageattn`` symbol *before*
    # the dit module imports finish wiring -- the function is
    # captured at call time via attribute lookup on the
    # ``sageattention`` package, so replacing the package's
    # exported function is enough.
    import sageattention

    sageattention.sageattn = _sdpa_replacement
    # Also rebind any direct ``from sageattention import sageattn``
    # imports the vendor dit module has already done.
    try:
        from wan.models.dits import arwan_w_action_w_mem_relative_rope as _vendor_mod

        _vendor_mod.sageattn = _sdpa_replacement
    except ImportError:
        pass
    print(
        "[sdpa_patch] sageattention.sageattn -> F.scaled_dot_product_attention",
        flush=True,
    )
