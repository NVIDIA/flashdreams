# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA parity for grouped-query attention over a changing valid cache prefix."""

from functools import partial

import pytest
import torch
from cosmos3_nano_sim_bimanual import attention
from cosmos3_nano_sim_bimanual.attention import (
    valid_length_cached_attention,
    varlen_attention_available,
)
from torch.nn import functional as F


@pytest.mark.ci_gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or not varlen_attention_available(),
    reason="CUDA and PyTorch varlen FlashAttention are required",
)
@pytest.mark.parametrize("mode", ["eager", "compiled", "cuda_graph"])
@pytest.mark.parametrize("query_heads", [8, 32], ids=["mha", "gqa"])
@pytest.mark.parametrize("causal_block_tokens", [None, 16], ids=["chunk", "causal"])
@torch.inference_mode()
def test_valid_length_attention_matches_expanded_sdpa(
    mode: str, query_heads: int, causal_block_tokens: int | None
) -> None:
    """Match expanded-head SDPA through cache fill and CUDA-graph replay."""
    torch.manual_seed(730)
    q = torch.randn(1, query_heads, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    valid = torch.zeros(1, device="cuda", dtype=torch.int32)
    attend = partial(
        valid_length_cached_attention,
        current_tokens=32,
        causal_block_tokens=causal_block_tokens,
    )
    if mode != "eager":
        attend = torch.compile(attend, fullgraph=True)
    graph = None
    if mode == "cuda_graph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                attend(q, k, v, valid)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = attend(q, k, v, valid)

    for cached in (0, 32, 96):
        valid.fill_(cached)
        if graph is None:
            actual = attend(q, k, v, valid)
        else:
            graph.replay()
        keys = torch.cat((k[..., :cached, :], k[..., -32:, :]), dim=2)
        values = torch.cat((v[..., :cached, :], v[..., -32:, :]), dim=2)
        mask = None
        if causal_block_tokens is not None:
            frame = torch.arange(32, device=q.device) // causal_block_tokens
            mask = torch.cat(
                (
                    torch.ones(32, cached, device=q.device, dtype=torch.bool),
                    frame[None, :] <= frame[:, None],
                ),
                dim=1,
            )
        expected = F.scaled_dot_product_attention(
            q,
            keys.repeat_interleave(query_heads // 8, dim=1),
            values.repeat_interleave(query_heads // 8, dim=1),
            attn_mask=mask,
        )
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.ci_cpu
@pytest.mark.parametrize("accepts_gqa_keyword", [False, True])
def test_varlen_gqa_keyword_compatibility(
    monkeypatch: pytest.MonkeyPatch, accepts_gqa_keyword: bool
) -> None:
    """Support both implicit legacy GQA and the explicit keyword API."""

    def legacy(query: torch.Tensor, *_args: object) -> torch.Tensor:
        return query

    def modern(
        query: torch.Tensor, *_args: object, enable_gqa: bool = False
    ) -> torch.Tensor:
        assert enable_gqa
        return query

    monkeypatch.setattr(
        attention, "varlen_attn", modern if accepts_gqa_keyword else legacy
    )
    monkeypatch.setattr(attention, "_VARLEN_ACCEPTS_GQA", accepts_gqa_keyword)
    q = torch.zeros(1, 32, 8, 16)
    k = torch.zeros(1, 8, 16, 16)
    actual = attention.valid_length_cached_attention(
        q, k, k, torch.zeros(1, dtype=torch.int32), 8
    )
    torch.testing.assert_close(actual, q)
