# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Cosmos hierarchical CP token ordering.

Multi-GPU test:
    PYTHONPATH=. torchrun --standalone --nproc_per_node=4 -m pytest \
        tests/test_cosmos_cp_ordering.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from einops import rearrange, repeat
from torch.distributed import ProcessGroup

from flashdreams.core.distributed.context_parallel import (
    cat_outputs_cp,
    split_inputs_cp,
)
from flashdreams.recipes.alpadreams.transformer.impl.context_parallel import (
    create_hierarchical_cp_groups,
)
from flashdreams.recipes.alpadreams.transformer.impl.network import (
    CosmosDiTNetwork,
    CosmosDiTNetworkConfig,
)
from flashdreams.recipes.wan.transformer.impl.rope import RotaryPositionEmbedding3D


def _init_distributed() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size == 1:
        return rank, world_size
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if not torch.cuda.is_available():
        pytest.skip("Multi-rank CP tests require CUDA/NCCL.")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size()


def _destroy_distributed() -> None:
    # Do not tear down NCCL between tests. Repeated init/destroy in the same
    # torchrun-launched pytest process can race across ranks when one test
    # exits earlier under ``-x``. Let torchrun clean up at process exit.
    return


def test_cosmos_patchify_unpatchify_cp_roundtrip() -> None:
    """Distributed roundtrip for the actual Cosmos patchify CP plumbing."""

    rank, world_size = _init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        cfg = CosmosDiTNetworkConfig(
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=128,
            num_blocks=1,
            num_heads=4,
        )
        network = CosmosDiTNetwork(cfg).to(device=device)
        T, H, W, C = 21, 8, 16, 16
        groups = create_hierarchical_cp_groups(
            world_size=world_size,
            rank=rank,
            V=1,
            T=T,
            single_group_as_none=True,
        )
        process_groups = [groups.V_group, groups.T_group, groups.HW_group]

        t = torch.arange(T, device=device).view(1, 1, T, 1, 1, 1)
        h = torch.arange(H, device=device).view(1, 1, 1, 1, H, 1)
        w = torch.arange(W, device=device).view(1, 1, 1, 1, 1, W)
        c = torch.arange(C, device=device).view(1, 1, 1, C, 1, 1)
        x = (t * 1_000_000 + h * 1_000 + w + c * 0.01).to(torch.float32)

        patched = network.patchify_and_maybe_split_cp(
            x, process_groups=process_groups, cp_dims=[1, 2, 3]
        )
        roundtrip = network.unpatchify_and_maybe_gather_cp(
            pH=H // cfg.patch_spatial,
            pW=W // cfg.patch_spatial,
            x=patched,
            process_groups=process_groups,
            cp_dims=[1, 2, 3],
        )
        torch.testing.assert_close(roundtrip, x)
    finally:
        _destroy_distributed()


def _split_phantom_pos_buffers(
    T: int,
    H: int,
    W: int,
    device: torch.device,
    t_group: ProcessGroup | None,
    hw_group: ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror RotaryPositionEmbedding3D.set_context_parallel_group on phantom
    index-only buffers, so the test can decode the (t, h, w) index that
    each rank's local RoPE position is supposed to encode.

    Layout matches RoPE's ``(t h w)`` flat row-major order.
    """
    pos_t = repeat(
        torch.arange(T, device=device, dtype=torch.float32),
        "t -> (t h w) 1",
        h=H,
        w=W,
    )
    pos_h = repeat(
        torch.arange(H, device=device, dtype=torch.float32),
        "h -> (t h w) 1",
        t=T,
        w=W,
    )
    pos_w = repeat(
        torch.arange(W, device=device, dtype=torch.float32),
        "w -> (t h w) 1",
        t=T,
        h=H,
    )

    pos_t = rearrange(pos_t, "(t hw) 1 -> t hw 1", t=T)
    pos_h = rearrange(pos_h, "(t hw) 1 -> t hw 1", t=T)
    pos_w = rearrange(pos_w, "(t hw) 1 -> t hw 1", t=T)
    if t_group is not None:
        pos_t = split_inputs_cp(pos_t, seq_dim=0, cp_group=t_group)
        pos_h = split_inputs_cp(pos_h, seq_dim=0, cp_group=t_group)
        pos_w = split_inputs_cp(pos_w, seq_dim=0, cp_group=t_group)
    if hw_group is not None:
        pos_t = split_inputs_cp(pos_t, seq_dim=1, cp_group=hw_group)
        pos_h = split_inputs_cp(pos_h, seq_dim=1, cp_group=hw_group)
        pos_w = split_inputs_cp(pos_w, seq_dim=1, cp_group=hw_group)
    pos_t = rearrange(pos_t, "t hw 1 -> (t hw)")
    pos_h = rearrange(pos_h, "t hw 1 -> (t hw)")
    pos_w = rearrange(pos_w, "t hw 1 -> (t hw)")
    return pos_t, pos_h, pos_w


def _run_rope_data_alignment(T: int, H: int, W: int) -> None:
    """Per-token alignment check: data slab and RoPE freqs must encode
    the same (t, h, w) at each local index after CP.

    Builds a synthetic video whose per-token value uniquely encodes
    ``(t, h, w)``, runs it through the per-axis V/T/HW patchify split,
    decodes ``(t, h, w)`` from each local token, and compares to the
    indices implied by the new per-axis RoPE split (mirrored via
    ``_split_phantom_pos_buffers``). Under the broken flat-via-THW
    split this would fail at any local index whose actual ``(t, h, w)``
    doesn't lie at the matching flat position; under the per-axis split
    they line up exactly.
    """
    rank, world_size = _init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        cfg = CosmosDiTNetworkConfig(
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=128,
            num_blocks=1,
            num_heads=4,
        )
        network = CosmosDiTNetwork(cfg).to(device=device)
        # Pixel dims: H_pix = 2*H (patch_spatial=2), W_pix = 2*W. Channels=2.
        H_pix = H * cfg.patch_spatial
        W_pix = W * cfg.patch_spatial
        C = 2

        groups = create_hierarchical_cp_groups(
            world_size=world_size, rank=rank, V=1, T=T, single_group_as_none=True
        )

        # Build a synthetic [B=1, V=1, T, C, H_pix, W_pix] tensor whose
        # value at every spatial pixel inside a single (h, w) latent
        # token is ``(t*W_M + h)*W_M + w`` — independent of c and of
        # the patch subgrid (kh, kw) — so the patchified token vector
        # stays consistent across (kh, kw, c) and we can decode
        # (t, h, w) from any of its scalars. ``W_M`` is the smallest
        # power-of-10 strictly larger than max(T, H, W) so the encoded
        # ints stay well below float32's 2^24 exact range.
        max_dim = max(T, H, W)
        W_M = 1
        while W_M <= max_dim:
            W_M *= 10
        t_idx = torch.arange(T, device=device).view(1, 1, T, 1, 1, 1)
        h_idx = torch.arange(H, device=device).view(1, 1, 1, 1, H, 1).repeat_interleave(
            cfg.patch_spatial, dim=-2
        )
        w_idx = torch.arange(W, device=device).view(1, 1, 1, 1, 1, W).repeat_interleave(
            cfg.patch_spatial, dim=-1
        )
        c_filler = torch.zeros(1, 1, 1, C, 1, 1, device=device)
        x = (
            (t_idx * W_M + h_idx) * W_M + w_idx + c_filler
        ).to(torch.float32)

        process_groups = [groups.V_group, groups.T_group, groups.HW_group]
        patched = network.patchify_and_maybe_split_cp(
            x, process_groups=process_groups, cp_dims=[1, 2, 3]
        )
        # patched: [B, V_local, T_local, HW_local, D]. After
        # ``normed_x.reshape(B, V, -1, D)`` in Block.forward this is
        # the t-major flat layout fed to self-attention.
        flat = rearrange(patched, "b v t hw d -> b v (t hw) d")

        # Decode (t, h, w) from any single scalar of each token vector.
        # Use index 0 of the patchified D dim (corresponds to the c=0,
        # kh=0, kw=0 slot under the c-major patchify order); the encoder
        # we built above writes the same (t, h, w) value at every entry
        # of the patch subgrid for c=0, so this is well-defined.
        scalars = flat[0, 0, :, 0].to(torch.long)
        decoded_t = scalars // (W_M * W_M)
        decoded_h = (scalars // W_M) % W_M
        decoded_w = scalars % W_M

        pos_t, pos_h, pos_w = _split_phantom_pos_buffers(
            T=T,
            H=H,
            W=W,
            device=device,
            t_group=groups.T_group,
            hw_group=groups.HW_group,
        )
        assert pos_t.shape == decoded_t.shape, (
            f"shape mismatch: pos_t {pos_t.shape} vs data {decoded_t.shape}"
        )
        torch.testing.assert_close(pos_t.to(torch.long), decoded_t)
        torch.testing.assert_close(pos_h.to(torch.long), decoded_h)
        torch.testing.assert_close(pos_w.to(torch.long), decoded_w)

        # And exercise the real rope adapter: same per-axis split, must
        # produce freqs whose length matches the local token count.
        rope = RotaryPositionEmbedding3D(
            head_dim=32, len_t=T, len_h=H, len_w=W, device=device
        )
        rope.set_context_parallel_group(
            t_group=groups.T_group, hw_group=groups.HW_group
        )
        local_freqs = rope.shift_t(offset=0)
        assert local_freqs.shape[0] == flat.shape[2], (
            f"rope local len {local_freqs.shape[0]} != data local seq len {flat.shape[2]}"
        )
    finally:
        _destroy_distributed()


def test_rope_per_axis_alignment_t21_hw_only() -> None:
    """SIL teacher case: len_t=21 (not power of 2), CP only splits HW."""
    _run_rope_data_alignment(T=21, H=4, W=8)


def test_rope_per_axis_alignment_t2_mixed() -> None:
    """alpadreams chunk2-style: len_t=2 (power of 2), CP can mix T+HW."""
    _run_rope_data_alignment(T=2, H=4, W=8)


def test_rope_per_axis_alignment_t3_hw_only() -> None:
    """alpadreams chunk3-style: len_t=3 (not power of 2), CP only splits HW."""
    _run_rope_data_alignment(T=3, H=4, W=8)


def test_split_cat_cp_roundtrip_on_thw_group() -> None:
    """Direct split/cat roundtrip over THW group order."""

    rank, world_size = _init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        groups = create_hierarchical_cp_groups(
            world_size=world_size,
            rank=rank,
            V=1,
            T=21,
            single_group_as_none=True,
        )
        x = torch.arange(21 * 40, device=device).reshape(1, 21 * 40, 1)
        local = split_inputs_cp(x, seq_dim=1, cp_group=groups.THW_group)
        gathered = cat_outputs_cp(local, seq_dim=1, cp_group=groups.THW_group)
        torch.testing.assert_close(gathered, x)
    finally:
        _destroy_distributed()
