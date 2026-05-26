# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context-parallel smoke test for FlashVSR's dense full-attention mode.

Two-invocation protocol:

.. code-block:: bash

    uv run --extra dev pytest \
        integrations/flashvsr/tests/test_flashvsr_context_parallel.py::test_flashvsr_full_attention_cp_equivalence -v
    uv run --extra dev torchrun --nproc_per_node=2 -m pytest \
        integrations/flashvsr/tests/test_flashvsr_context_parallel.py::test_flashvsr_full_attention_cp_equivalence -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.distributed as dist
from flashvsr.transformer import FlashVSRTransformer, FlashVSRTransformerConfig
from flashvsr.transformer.network import FlashVSRBlock, FlashVSRDiTNetworkConfig

from flashdreams.core.attention.native import NativeAttention
from flashdreams.core.distributed.context_parallel import (
    cat_outputs_cp,
    split_inputs_cp,
)

pytestmark = pytest.mark.manual

_CP_REFERENCE_PATH = Path(
    os.environ.get(
        "FLASHVSR_FULL_ATTN_CP_REF_PATH",
        str(Path(tempfile.gettempdir()) / "flashdreams" / "flashvsr_full_cp.pt"),
    )
)


def _force_math_sdpa(transformer: FlashVSRTransformer) -> None:
    """Avoid cuDNN SDPA tiny-shape NaNs in this synthetic smoke test."""
    for block in transformer.network.blocks:
        block = cast(FlashVSRBlock, block)
        qkv_format = block.self_attn.attn_op.qkv_format
        block.self_attn.attn_op = NativeAttention(qkv_format=qkv_format, backend="math")
        block.self_attn.set_context_parallel_group(transformer._cp_group)

        qkv_format = block.cross_attn.attn_op.qkv_format
        block.cross_attn.attn_op = NativeAttention(
            qkv_format=qkv_format, backend="math"
        )


def _tiny_full_attention_transformer() -> FlashVSRTransformerConfig:
    return FlashVSRTransformerConfig(
        network=FlashVSRDiTNetworkConfig(
            # Match the stable cuDNN SDPA test shape used by the template
            # integration. Smaller head_dims (16/8) can silently produce NaNs.
            dim=128,
            ffn_dim=256,
            num_heads=2,
            num_layers=1,
            in_dim=4,
            out_dim=4,
            text_dim=32,
            freq_dim=32,
            text_len=4,
            attention_mode="full",
        ),
        dtype=torch.float32,
        checkpoint_path=None,
        batch_shape=(1,),
        len_t=2,
        guidance_scale=1.0,
        topk_ratio=2.0,
        kv_ratio=1,
        local_range=11,
        attention_mode="full",
        compile_network=False,
        use_cuda_graph=False,
    )


@torch.no_grad()
def _cp_one_predict_flow(device: torch.device, *, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    transformer = _tiny_full_attention_transformer().setup().to(device).eval()
    assert isinstance(transformer, FlashVSRTransformer)
    _force_math_sdpa(transformer)

    cfg = transformer.config
    net_cfg = cfg.network
    assert isinstance(net_cfg, FlashVSRDiTNetworkConfig)
    kt, kh, kw = net_cfg.patch_size
    height, width = 4, 4
    pT, pH, pW = cfg.len_t // kt, height // kh, width // kw
    L = pT * pH * pW
    patch_volume = kt * kh * kw

    gen = torch.Generator(device=device).manual_seed(seed + 17)
    text_embeddings = torch.randn(
        1,
        net_cfg.text_len,
        net_cfg.text_dim,
        device=device,
        generator=gen,
        dtype=cfg.dtype,
    )
    noisy_global = torch.randn(
        1,
        L,
        net_cfg.in_dim * patch_volume,
        device=device,
        generator=gen,
        dtype=cfg.dtype,
    )
    lq_global = [
        torch.randn(
            1,
            L,
            net_cfg.dim,
            device=device,
            generator=gen,
            dtype=cfg.dtype,
        )
        for _ in range(net_cfg.num_layers)
    ]

    cache = transformer.initialize_autoregressive_cache(
        height=height,
        width=width,
        text_embeddings=text_embeddings,
    )
    noisy_local = split_inputs_cp(
        noisy_global, seq_dim=1, cp_group=transformer._cp_group
    )
    lq_local = transformer.patchify_and_maybe_split_cp(lq_global)

    cache.start(0)
    flow_local = transformer.predict_flow(
        noisy_latent=noisy_local,
        timestep=torch.tensor(1000.0, device=device, dtype=cfg.dtype),
        cache=cache,
        input=lq_local,
    )
    cache.finalize(0)
    return cat_outputs_cp(flow_local, seq_dim=1, cp_group=transformer._cp_group)


def test_flashvsr_full_attention_cp_equivalence() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if not torch.cuda.is_available():
        pytest.skip("FlashVSR full-attention CP equivalence requires CUDA.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size == 1:
        # First invocation: run the dense full-attention path without CP and
        # persist the global output. The torchrun branch below treats this as
        # the reference tensor for equivalence.
        flow_global = _cp_one_predict_flow(device, seed=0)
        _CP_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"flow_global": flow_global.detach().cpu()}, _CP_REFERENCE_PATH)
        assert torch.isfinite(flow_global).all()

    else:
        # Second invocation: run under torchrun, gather the per-rank CP output,
        # and compare rank 0 against the reference produced above. A missing
        # reference is a usage error, not an environment skip.
        if not _CP_REFERENCE_PATH.exists():
            raise FileNotFoundError(
                f"CP reference {_CP_REFERENCE_PATH} not found; run the single-GPU "
                "branch first."
            )

        assert dist.is_nccl_available()
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
            )

        try:
            flow_global = _cp_one_predict_flow(device, seed=0)
            if rank == 0:
                reference = torch.load(_CP_REFERENCE_PATH, weights_only=True)[
                    "flow_global"
                ]
                torch.testing.assert_close(
                    flow_global.detach().cpu(),
                    reference,
                    rtol=2e-2,
                    atol=2e-2,
                )
        finally:
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
