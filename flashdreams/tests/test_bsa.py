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

"""Unit tests for :class:`BlockSparseAttention`.

Run on a single GPU::

    PYTHONPATH=. pytest tests/test_bsa.py -v
"""

import math

import pytest
import torch
from einops import rearrange

pytest.importorskip("block_sparse_attn")

from flashdreams.core.attention import (
    BlockSparseAttention,
    NativeAttention,
    build_local_spatial_block_mask,
    build_topk_block_mask,
)


def _make_qkv(
    *,
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    qkv_format: str,
    dtype: torch.dtype,
    device: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    if qkv_format == "bhsd":
        shape = (batch, heads, seq, dim)
    else:
        shape = (batch, seq, heads, dim)
    q = torch.randn(shape, generator=g, dtype=dtype, device=device)
    k = torch.randn(shape, generator=g, dtype=dtype, device=device)
    v = torch.randn(shape, generator=g, dtype=dtype, device=device)
    return q, k, v


@pytest.mark.parametrize("qkv_format", ["bhsd", "bshd"])
@pytest.mark.parametrize("blockmask_mode", ["auto_dense", "explicit_ones"])
def test_full_blockmask_matches_sdpa(qkv_format: str, blockmask_mode: str) -> None:
    """Both the auto-mask path with ``topk_ratio=1.0`` and an explicit
    all-ones override must reproduce dense SDPA."""
    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.bfloat16

    batch, heads, seq, dim = 2, 4, 256, 128

    q, k, v = _make_qkv(
        batch=batch,
        heads=heads,
        seq=seq,
        dim=dim,
        qkv_format=qkv_format,
        dtype=dtype,
        device=device,
    )

    sdpa = NativeAttention(qkv_format=qkv_format, backend="cudnn").to(device)
    # topk_ratio defaults to 1.0; with seq=256, q_thw=(4, 8, 8) gives
    # L = 256 tokens and a (2, 1, 1) block grid.
    bsa = BlockSparseAttention(qkv_format=qkv_format).to(device)

    q_thw = (4, 8, 8)
    if blockmask_mode == "auto_dense":
        kwargs = {"q_thw": q_thw}
    else:
        # Block size is 128 in BSA; seq=256 → 2 q-blocks and 2 k-blocks.
        n_blocks = seq // 128
        blockmask = torch.ones(
            (batch, heads, n_blocks, n_blocks), dtype=torch.uint8, device=device
        )
        kwargs = {"blockmask": blockmask}

    with torch.no_grad():
        out_sdpa = sdpa(q, k, v)
        out_bsa = bsa(q, k, v, **kwargs)

    assert out_sdpa.shape == out_bsa.shape
    torch.testing.assert_close(out_bsa, out_sdpa, atol=5e-3, rtol=5e-3)


def test_partial_blockmask_differs_from_sdpa() -> None:
    """A partial blockmask should change the output (sanity check)."""
    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.bfloat16

    batch, heads, seq, dim = 1, 4, 256, 64

    q, k, v = _make_qkv(
        batch=batch,
        heads=heads,
        seq=seq,
        dim=dim,
        qkv_format="bshd",
        dtype=dtype,
        device=device,
    )

    sdpa = NativeAttention(qkv_format="bshd", backend="cudnn").to(device)
    bsa = BlockSparseAttention(qkv_format="bshd").to(device)

    # Drop the (q_block=0, k_block=1) tile so block 0 of Q sees only
    # block 0 of K — clearly different from dense attention.
    n_blocks = seq // 128
    blockmask = torch.ones(
        (batch, heads, n_blocks, n_blocks), dtype=torch.uint8, device=device
    )
    blockmask[:, :, 0, 1] = 0

    with torch.no_grad():
        out_sdpa = sdpa(q, k, v)
        out_bsa = bsa(q, k, v, blockmask=blockmask)

    diff = (out_bsa.float() - out_sdpa.float()).abs().max().item()
    assert diff > 1e-2, (
        f"Partial blockmask should diverge from SDPA, but max abs diff "
        f"was only {diff:.2e}."
    )


def test_diagonal_blockmask_matches_blockwise_sdpa() -> None:
    """Diagonal-only blockmask matches running SDPA per (q_block, k_block)
    aligned tile (each q-block only attends to its own k-block)."""
    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.bfloat16

    batch, heads, seq, dim = 1, 2, 256, 64
    block = 128
    n_blocks = seq // block

    q, k, v = _make_qkv(
        batch=batch,
        heads=heads,
        seq=seq,
        dim=dim,
        qkv_format="bshd",
        dtype=dtype,
        device=device,
    )

    bsa = BlockSparseAttention(qkv_format="bshd").to(device)
    blockmask = torch.eye(n_blocks, dtype=torch.uint8, device=device)
    blockmask = blockmask.expand(batch, heads, n_blocks, n_blocks).contiguous()

    with torch.no_grad():
        out_bsa = bsa(q, k, v, blockmask=blockmask)

    # Reference: independent SDPA per diagonal tile, concatenated.
    ref = torch.zeros_like(out_bsa)
    for i in range(n_blocks):
        s = i * block
        e = s + block
        # SDPA in bhsd; reshape then transpose to call F.scaled_dot_product_attention.
        q_t = q[:, s:e].transpose(1, 2)
        k_t = k[:, s:e].transpose(1, 2)
        v_t = v[:, s:e].transpose(1, 2)
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION
        ):
            out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
        ref[:, s:e] = out.transpose(1, 2)

    torch.testing.assert_close(out_bsa, ref, atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------------
# Mask helpers — verified against verbatim ports of the FlashVSR reference.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _ref_local_mask(
    block_h: int,
    block_w: int,
    win_h: int,
    win_w: int,
    *,
    include_self: bool,
    device: torch.device,
) -> torch.Tensor:
    """Verbatim port of build_local_block_mask_shifted_vec_normal_slide."""
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r = start_r + win_h - 1
    start_c = c_all - c_half
    end_c = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


@torch.no_grad()
def _ref_topk_mask(
    batch_size: int,
    nheads: int,
    seqlen: int,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    *,
    topk: int,
    local_attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Verbatim port of generate_draft_block_mask (B=1 path)."""
    assert batch_size == 1
    avgpool_q = torch.mean(q_w, dim=1)
    avgpool_k = torch.mean(k_w, dim=1)
    avgpool_q = rearrange(avgpool_q, "s (h d) -> s h d", h=nheads)
    avgpool_k = rearrange(avgpool_k, "s (h d) -> s h d", h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = scores.shape[2] // local_attn_mask.shape[1]
    lm = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(
        repeat_len, 1, repeat_num, 1
    )
    lm = rearrange(lm, "x a y b -> (x a) (y b)")
    lm = lm.unsqueeze(0).repeat(repeat_head, 1, 1)
    lm = lm.to(torch.float32)
    lm = lm.masked_fill(lm == False, -float("inf"))  # noqa: E712
    lm = lm.masked_fill(lm == True, 0)  # noqa: E712
    scores = scores + lm
    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, "h (it s1) s2 -> (h it) s1 s2", it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1] - 1, topk)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
    mask_new = rearrange(mask_new, "(h it) s1 s2 -> h (it s1) s2", it=seqlen)
    return mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)


def test_build_local_spatial_block_mask_matches_reference() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for spatial_h, spatial_w, win_h, win_w, include_self in [
        (8, 8, 6, 6, True),
        (4, 6, 3, 5, False),
        (5, 5, 1, 1, True),
    ]:
        ours = build_local_spatial_block_mask(
            spatial_h, spatial_w, win_h, win_w, include_self=include_self, device=device
        )
        ref = _ref_local_mask(
            spatial_h, spatial_w, win_h, win_w, include_self=include_self, device=device
        )
        assert torch.equal(ours, ref), (
            f"local mask mismatch for "
            f"({spatial_h}, {spatial_w}, {win_h}, {win_w}, {include_self})"
        )


def test_build_topk_block_mask_matches_reference() -> None:
    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.float32

    nheads = 4
    head_dim = 32
    win_size = 128

    spatial_h, spatial_w = 4, 4
    spatial_q = spatial_h * spatial_w
    spatial_k = spatial_q
    nf_q = 3
    nf_k = 5
    N_q = nf_q * spatial_q
    N_k = nf_k * spatial_k

    g = torch.Generator(device=device).manual_seed(7)
    q_w = torch.randn(
        (N_q, win_size, nheads * head_dim), generator=g, dtype=dtype, device=device
    )
    k_w = torch.randn(
        (N_k, win_size, nheads * head_dim), generator=g, dtype=dtype, device=device
    )

    local_mask = build_local_spatial_block_mask(
        spatial_h, spatial_w, 3, 3, include_self=True, device=device
    )

    for topk in [1, 5, 32, 1000]:
        ref_mask = _ref_topk_mask(
            batch_size=1,
            nheads=nheads,
            seqlen=nf_q,
            q_w=q_w,
            k_w=k_w,
            topk=topk,
            local_attn_mask=local_mask,
        )

        # Convert (N_q, win, H*D) → (1, N_q, H, D) by averaging over win.
        q_desc = rearrange(
            q_w.mean(dim=1), "s (h d) -> 1 s h d", h=nheads
        ).contiguous()
        k_desc = rearrange(
            k_w.mean(dim=1), "s (h d) -> 1 s h d", h=nheads
        ).contiguous()

        ours = build_topk_block_mask(
            q_desc,
            k_desc,
            num_temporal_q_blocks=nf_q,
            num_temporal_k_blocks=nf_k,
            topk=topk,
            local_spatial_block_mask=local_mask,
        )

        assert ours.shape == (1, nheads, N_q, N_k)
        assert ours.dtype == torch.uint8
        assert torch.equal(ours.bool(), ref_mask), f"topk mask mismatch at topk={topk}"


def test_topk_mask_with_huge_topk_keeps_local_mask() -> None:
    """When ``topk`` exceeds the in-window count, the kept entries
    should be exactly the local mask broadcast over the temporal axes."""
    assert torch.cuda.is_available()
    device = "cuda"

    nheads = 2
    head_dim = 16
    spatial_h, spatial_w = 4, 4
    spatial_q = spatial_h * spatial_w
    nf_q, nf_k = 2, 2
    N_q = nf_q * spatial_q
    N_k = nf_k * spatial_q

    g = torch.Generator(device=device).manual_seed(0)
    q_desc = torch.randn(
        (1, N_q, nheads, head_dim), generator=g, dtype=torch.float32, device=device
    )
    k_desc = torch.randn(
        (1, N_k, nheads, head_dim), generator=g, dtype=torch.float32, device=device
    )

    local_mask = build_local_spatial_block_mask(
        spatial_h, spatial_w, 3, 3, include_self=True, device=device
    )

    # Within each (head, temporal_q) slab there are
    # spatial_q * (nf_k * spatial_q) entries; only spatial_q * (nf_k * in_window_count)
    # of them are non-zero after softmax. With topk = (entries - 1), the
    # threshold sits at the smallest positive entry → kept ≈ local mask.
    apply_topk = spatial_q * N_k - 1
    mask = build_topk_block_mask(
        q_desc,
        k_desc,
        num_temporal_q_blocks=nf_q,
        num_temporal_k_blocks=nf_k,
        topk=apply_topk,
        local_spatial_block_mask=local_mask,
    )

    # Expand local mask to (1, 1, N_q, N_k) and compare.
    expected = local_mask[None, :, None, :].expand(nf_q, spatial_q, nf_k, spatial_q)
    expected = rearrange(expected, "tq sq tk sk -> (tq sq) (tk sk)").contiguous()
    expected = expected[None, None, :, :].expand(1, nheads, N_q, N_k)
    assert torch.equal(mask.bool(), expected)


def test_bsa_with_topk_mask_end_to_end() -> None:
    """End-to-end: build a topk mask, run BSA, sanity-check shape and
    that it differs from dense SDPA (mask is doing real work)."""
    assert torch.cuda.is_available()
    device = "cuda"

    nheads = 4
    head_dim = 32
    spatial_h, spatial_w = 4, 4
    spatial_q = spatial_h * spatial_w
    win_size = 128  # must equal BSA block size
    nf_q = nf_k = 2
    N_q = nf_q * spatial_q
    seq_q = N_q * win_size
    seq_k = N_q * win_size

    g = torch.Generator(device=device).manual_seed(1)
    q = torch.randn(
        (1, seq_q, nheads, head_dim), generator=g, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        (1, seq_k, nheads, head_dim), generator=g, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(
        (1, seq_k, nheads, head_dim), generator=g, dtype=torch.bfloat16, device=device
    )

    # Slab size = spatial_q * (Tb * spatial_q) = 16 * 32 = 512;
    # ratio = 4/512 = 0.0078125 → topk_int = round(0.0078125 * 512) = 4.
    bsa = BlockSparseAttention(
        qkv_format="bshd", topk_ratio=4 / 512, local_range=3
    ).to(device)
    sdpa = NativeAttention(qkv_format="bshd", backend="cudnn").to(device)
    # Token-grid: (Tb*wf, Hb*wh, Wb*ww) = (2*2, 4*8, 4*8) = (4, 32, 32);
    # L = 4*32*32 = 4096 = seq_q ✓.
    q_thw = (nf_q * 2, spatial_h * 8, spatial_w * 8)
    with torch.no_grad():
        out_bsa = bsa(q, k, v, q_thw=q_thw)
        out_sdpa = sdpa(q, k, v)

    assert out_bsa.shape == (1, seq_q, nheads, head_dim)
    assert torch.isfinite(out_bsa).all()
    diff = (out_bsa.float() - out_sdpa.float()).abs().max().item()
    assert diff > 1e-2, (
        f"topk-derived mask should sparsify attention; got max abs diff {diff:.2e}"
    )


def test_bsa_local_attn_mask_is_cached_and_invalidates_on_layout_change() -> None:
    """The lazily-built local mask should be reused across forward
    calls with the same (spatial_h, spatial_w) and rebuilt otherwise."""
    assert torch.cuda.is_available()
    device = "cuda"

    nheads, head_dim = 2, 16
    win_size = 128
    spatial_h, spatial_w = 4, 4
    spatial_q = spatial_h * spatial_w
    nf_q = nf_k = 2
    seq = nf_q * spatial_q * win_size

    g = torch.Generator(device=device).manual_seed(2)
    q = torch.randn(
        (1, seq, nheads, head_dim), generator=g, dtype=torch.bfloat16, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    bsa = BlockSparseAttention(
        qkv_format="bshd", topk_ratio=0.05, local_range=3
    ).to(device)

    # First spatial: (Hb=4, Wb=4) → token (H=32, W=32); T=nf_q*wf=4.
    q_thw_1 = (nf_q * 2, spatial_h * 8, spatial_w * 8)

    assert bsa._local_attn_mask is None
    with torch.no_grad():
        bsa(q, k, v, q_thw=q_thw_1)
    first_mask = bsa._local_attn_mask
    assert first_mask is not None

    with torch.no_grad():
        bsa(q, k, v, q_thw=q_thw_1)
    assert bsa._local_attn_mask is first_mask, "Cached mask should be reused."

    # Different spatial block grid (2, 8) → token (H=16, W=64); seq
    # length matches because Hb*Wb stays at 16.
    spatial_h2, spatial_w2 = 2, 8
    q_thw_2 = (nf_q * 2, spatial_h2 * 8, spatial_w2 * 8)
    seq2 = nf_q * (spatial_h2 * spatial_w2) * win_size
    q2 = torch.randn(
        (1, seq2, nheads, head_dim), generator=g, dtype=torch.bfloat16, device=device
    )
    k2 = torch.randn_like(q2)
    v2 = torch.randn_like(q2)
    with torch.no_grad():
        bsa(q2, k2, v2, q_thw=q_thw_2)
    assert bsa._local_attn_mask is not first_mask, (
        "Mask should be rebuilt when the block grid changes."
    )
    assert bsa._local_attn_mask_key == (spatial_h2, spatial_w2)


def _flashvsr_partition_3d(
    x: torch.Tensor, win: tuple[int, int, int]
) -> torch.Tensor:
    """Verbatim port of WindowPartition3D.partition (5-D in, 3-D out)."""
    B, F, H, W, C = x.shape
    wf, wh, ww = win
    assert F % wf == 0 and H % wh == 0 and W % ww == 0
    x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wf * wh * ww, C)


def test_bsa_module_matches_flashvsr_full_flow() -> None:
    """Compare ``BlockSparseAttention(topk_ratio=R, local_range=L)``
    end-to-end against the FlashVSR reference: same window-partitioned
    q/k/v, same pooled descriptors, same mask (verbatim ports), same
    BSA call. Outputs must match bit-exactly."""
    assert torch.cuda.is_available()
    device = "cuda"

    B = 1
    nheads = 4
    head_dim = 32
    D_total = nheads * head_dim

    F_dim, H_dim, W_dim = 4, 32, 32
    win = (2, 8, 8)
    wf, wh, ww = win
    win_size = wf * wh * ww
    assert win_size == 128, "FlashVSR uses 2x8x8=128 to match the BSA block size."

    nf_q = F_dim // wf
    sp_h = H_dim // wh
    sp_w = W_dim // ww
    spatial_q = sp_h * sp_w

    g = torch.Generator(device=device).manual_seed(3)
    x_q = torch.randn(
        (B, F_dim, H_dim, W_dim, D_total),
        generator=g,
        dtype=torch.bfloat16,
        device=device,
    )
    x_k = torch.randn(
        (B, F_dim, H_dim, W_dim, D_total),
        generator=g,
        dtype=torch.bfloat16,
        device=device,
    )
    x_v = torch.randn(
        (B, F_dim, H_dim, W_dim, D_total),
        generator=g,
        dtype=torch.bfloat16,
        device=device,
    )

    # FlashVSR window-partition + reorder to (B, S, D_total) flat-block-major.
    q_w = _flashvsr_partition_3d(x_q, win)
    k_w = _flashvsr_partition_3d(x_k, win)
    v_w = _flashvsr_partition_3d(x_v, win)
    block_n = q_w.shape[0] // B
    block_n_kv = k_w.shape[0] // B
    block_s = q_w.shape[1]
    reorder_q = rearrange(
        q_w,
        "(b block_n) (block_s) d -> b (block_n block_s) d",
        block_n=block_n,
        block_s=block_s,
    )
    reorder_k = rearrange(
        k_w,
        "(b block_n) (block_s) d -> b (block_n block_s) d",
        block_n=block_n_kv,
        block_s=block_s,
    )
    reorder_v = rearrange(
        v_w,
        "(b block_n) (block_s) d -> b (block_n block_s) d",
        block_n=block_n_kv,
        block_s=block_s,
    )

    q_bshd = rearrange(reorder_q, "b s (h d) -> b s h d", h=nheads)
    k_bshd = rearrange(reorder_k, "b s (h d) -> b s h d", h=nheads)
    v_bshd = rearrange(reorder_v, "b s (h d) -> b s h d", h=nheads)

    topk = 8
    local_range = 3

    # Reference path: build the local mask + topk mask via the
    # verbatim FlashVSR ports and pass that as an explicit blockmask.
    ref_local = _ref_local_mask(
        sp_h, sp_w, local_range, local_range, include_self=True, device=device
    )
    ref_blockmask = _ref_topk_mask(
        batch_size=B,
        nheads=nheads,
        seqlen=nf_q,
        q_w=q_w,
        k_w=k_w,
        topk=topk,
        local_attn_mask=ref_local,
    )
    ref_blockmask_u8 = ref_blockmask.to(torch.uint8)

    # Module-under-test path: same q/k/v, same (topk, local_range),
    # but the mask is synthesized inside forward via the auto path.
    # Convert the integer top-K used by the reference into the ratio
    # the module accepts: slab_size = spatial_q * (n_temp_k * spatial_q),
    # and the module recomputes int(round(ratio * slab_size)) which
    # must round back to ``topk``.
    slab_size = spatial_q * (nf_q * spatial_q)
    topk_ratio = topk / slab_size
    bsa_auto = BlockSparseAttention(
        qkv_format="bshd", topk_ratio=topk_ratio, local_range=local_range
    ).to(device)
    bsa_explicit = BlockSparseAttention(qkv_format="bshd").to(device)

    with torch.no_grad():
        out_auto = bsa_auto(
            q_bshd, k_bshd, v_bshd, q_thw=(F_dim, H_dim, W_dim)
        )
        out_explicit = bsa_explicit(
            q_bshd, k_bshd, v_bshd, blockmask=ref_blockmask_u8
        )

    # Both runs use the same BSA kernel on the same q/k/v; if the mask
    # synthesis is correct the outputs are bit-identical.
    torch.testing.assert_close(out_auto, out_explicit, atol=0.0, rtol=0.0)

    # Also check the cached local mask matches the reference.
    assert torch.equal(bsa_auto._local_attn_mask, ref_local)
    assert bsa_auto._local_attn_mask_key == (sp_h, sp_w)

    # And that the auto path uses some sparsity (otherwise the test
    # wouldn't be exercising the topk logic).
    n_q_blocks = nf_q * spatial_q
    n_k_blocks = nf_q * spatial_q
    assert ref_blockmask_u8.shape == (B, nheads, n_q_blocks, n_k_blocks)
    sparsity = 1.0 - ref_blockmask_u8.float().mean().item()
    assert sparsity > 0.5, (
        f"Expected at least 50% sparsity, got {sparsity:.2%}."
    )


def test_topk_ratio_one_matches_sdpa() -> None:
    """``topk_ratio = 1.0`` must short-circuit to a full-ones mask
    (regardless of ``local_range``) and reproduce dense SDPA.

    Uses a layout whose local mask would otherwise gate out most
    pairs (``local_range = 3`` on a 4×4 block grid), so the test
    fails if the short-circuit is missing and the kept set falls back
    to the local mask."""
    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.bfloat16

    # Token-grid (T=4, H=32, W=32) with default window (2, 8, 8) →
    # block grid (2, 4, 4); seq = T*H*W = 4096.
    T_tok, H_tok, W_tok = 4, 32, 32
    seq = T_tok * H_tok * W_tok

    nheads = 4
    head_dim = 64

    g = torch.Generator(device=device).manual_seed(11)
    q = torch.randn(
        (1, seq, nheads, head_dim), generator=g, dtype=dtype, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    bsa = BlockSparseAttention(
        qkv_format="bshd", topk_ratio=1.0, local_range=3
    ).to(device)
    sdpa = NativeAttention(qkv_format="bshd", backend="cudnn").to(device)

    with torch.no_grad():
        out_bsa = bsa(q, k, v, q_thw=(T_tok, H_tok, W_tok))
        out_sdpa = sdpa(q, k, v)

    assert out_bsa.shape == out_sdpa.shape
    torch.testing.assert_close(out_bsa, out_sdpa, atol=5e-3, rtol=5e-3)


def test_forward_requires_q_thw_or_blockmask() -> None:
    """Calling forward without either ``q_thw`` or ``blockmask`` must
    raise — the dense fallback was intentionally removed."""
    assert torch.cuda.is_available()
    device = "cuda"

    bsa = BlockSparseAttention(qkv_format="bshd").to(device)
    q = torch.randn(1, 256, 2, 32, dtype=torch.bfloat16, device=device)
    with pytest.raises(AssertionError, match="q_thw"):
        bsa(q, q, q)


def test_init_rejects_invalid_window() -> None:
    """``window`` product must equal the BSA block size (128)."""
    with pytest.raises(AssertionError, match="window product"):
        BlockSparseAttention(window=(2, 8, 4))  # 2 * 8 * 4 = 64, not 128


def test_asymmetric_q_thw_kv_thw_matches_sdpa() -> None:
    """``T_q != T_k`` (streaming KV cache) with ``topk_ratio=1.0`` must
    still reproduce dense SDPA over the full asymmetric K/V tensor.

    Mirrors the alpadreams call site: query is one chunk's worth
    (``T_q = 2``) and the cached K/V already covers two chunks
    (``T_k = 4``)."""
    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.bfloat16

    # window=(2, 4, 16) → 128-token blocks; spatial block grid (11, 5)
    # is too large for this unit test, so use a smaller grid that
    # still factors into 128 per window: window=(2, 8, 8), token grid
    # (T_q=2, H=16, W=16) → spatial=(2, 2) blocks, slab manageable.
    wf, wh, ww = 2, 8, 8
    H_tok = 16
    W_tok = 16
    T_q = 2
    T_k = 4

    nheads = 4
    head_dim = 64
    Sq = T_q * H_tok * W_tok
    Sk = T_k * H_tok * W_tok

    g = torch.Generator(device=device).manual_seed(13)
    q = torch.randn(
        (1, Sq, nheads, head_dim), generator=g, dtype=dtype, device=device
    )
    k = torch.randn(
        (1, Sk, nheads, head_dim), generator=g, dtype=dtype, device=device
    )
    v = torch.randn(
        (1, Sk, nheads, head_dim), generator=g, dtype=dtype, device=device
    )

    bsa = BlockSparseAttention(
        qkv_format="bshd", topk_ratio=1.0, window=(wf, wh, ww), local_range=3
    ).to(device)
    sdpa = NativeAttention(qkv_format="bshd", backend="cudnn").to(device)

    with torch.no_grad():
        out_bsa = bsa(
            q,
            k,
            v,
            q_thw=(T_q, H_tok, W_tok),
            kv_thw=(T_k, H_tok, W_tok),
        )
        out_sdpa = sdpa(q, k, v)

    assert out_bsa.shape == out_sdpa.shape == (1, Sq, nheads, head_dim)
    torch.testing.assert_close(out_bsa, out_sdpa, atol=5e-3, rtol=5e-3)


def test_kv_thw_must_match_q_thw_spatial() -> None:
    """The local-spatial mask is shared, so spatial dims must agree."""
    assert torch.cuda.is_available()
    device = "cuda"

    bsa = BlockSparseAttention(qkv_format="bshd").to(device)
    # Token grids differ in H: (2, 16, 16) vs (2, 8, 32). Same Sk total
    # but spatial grid differs → must raise.
    Sq = 2 * 16 * 16
    Sk = 2 * 8 * 32
    q = torch.randn(1, Sq, 2, 64, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, Sk, 2, 64, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    with pytest.raises(AssertionError, match="spatial dims"):
        bsa(q, k, v, q_thw=(2, 16, 16), kv_thw=(2, 8, 32))
