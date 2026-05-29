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

"""CPU-only unit tests for the HY-WorldPlay camera conditioner."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

pytestmark = pytest.mark.ci_cpu


## ---------------------------------------------------------------------------
## Control payload surface
## ---------------------------------------------------------------------------


def test_hyworldplay_ctrl_camera_fields_default_to_none() -> None:
    """Camera ctrl fields default ``None`` so action-only callers stay opt-in."""
    from hy_worldplay._action import HyWorldPlayCtrl

    ctrl = HyWorldPlayCtrl(
        latent=torch.zeros(1, 1, 1, 1, 1),
        mask=torch.zeros(1, 1, 1, 1, 1),
    )
    assert ctrl.viewmats is None
    assert ctrl.Ks is None
    assert ctrl.action is None


def test_hyworldplay_transformer_patchify_preserves_camera_fields() -> None:
    """``viewmats`` / ``Ks`` survive the patchify rebuild of the I2V payload.

    The base transformer reconstructs the ctrl via
    ``I2VCtrl(latent=..., mask=...)`` after patchify, which would drop
    subclass fields unless the subclass overrides the rebuild.
    """
    from hy_worldplay._action import HyWorldPlayCtrl, HyWorldPlayWan21Transformer

    fake_self: Any = type("F", (), {})()

    def passthrough(self, x):
        return x

    fake_self.patchify_and_maybe_split_cp = (
        HyWorldPlayWan21Transformer.patchify_and_maybe_split_cp.__get__(fake_self)
    )

    # The method short-circuits the I2VCtrl branch for HyWorldPlayCtrl
    # inputs, so the super() call is never reached on this path.
    latent = torch.randn(1, 4, 16, 8, 8)
    mask = torch.zeros(1, 4, 16, 8, 8)
    viewmats = torch.eye(4).expand(1, 4, 4, 4).contiguous()
    Ks = torch.eye(3).expand(1, 4, 3, 3).contiguous()
    action = torch.zeros(1, 4, dtype=torch.long)
    ctrl = HyWorldPlayCtrl(
        latent=latent,
        mask=mask,
        action=action,
        viewmats=viewmats,
        Ks=Ks,
    )

    # Force the no-op patchify branch (already patchified): the method
    # returns the ctrl as-is, which still exercises the type check and
    # dataclass plumbing.
    ctrl_patched = HyWorldPlayCtrl(
        latent=latent,
        mask=mask,
        action=action,
        viewmats=viewmats,
        Ks=Ks,
        _is_patchified=True,
    )
    out = fake_self.patchify_and_maybe_split_cp(ctrl_patched)
    assert out is ctrl_patched
    assert out.viewmats is viewmats
    assert out.Ks is Ks
    assert out.action is action


## ---------------------------------------------------------------------------
## PRoPE block surface
## ---------------------------------------------------------------------------


def _make_prope_block(*, dim: int = 64, num_heads: int = 2) -> Any:
    """Build a tiny :class:`HyWorldPlayPRoPEBlock` for the structural checks below."""
    from hy_worldplay._camera import HyWorldPlayPRoPEBlock

    return HyWorldPlayPRoPEBlock(
        dim=dim,
        ffn_dim=dim * 2,
        num_heads=num_heads,
        cross_attn_norm=True,
        eps=1e-6,
        i2v=False,
        apply_rope_before_kvcache=True,
    )


def test_prope_block_self_attn_is_dual_branch_subclass() -> None:
    """PRoPE block must replace stock self-attn with the dual-branch subclass."""
    from hy_worldplay._camera import HyWorldPlayPRoPESelfAttention

    block = _make_prope_block()
    assert isinstance(block.self_attn, HyWorldPlayPRoPESelfAttention)
    # Dual-branch subclass adds ``o_prope`` and ``attn_op_prope`` on
    # top of the standard module surface.
    assert hasattr(block.self_attn, "o_prope")
    assert hasattr(block.self_attn, "attn_op_prope")


def test_prope_block_o_prope_is_zero_init() -> None:
    """``o_prope`` is zero-init so the PRoPE branch contributes zero residual.

    Mirrors upstream's ``nn.init.zeros_(block.attn1.to_out_prope[0].weight)``
    invariant -- until the distilled checkpoint loads non-zero weights,
    the PRoPE branch is a strict identity and the dual-branch block
    stays parity-equivalent to the standard one.
    """
    block = _make_prope_block()
    weight = block.self_attn.o_prope.weight
    assert torch.all(weight == 0), "o_prope.weight not zero-init"
    bias = block.self_attn.o_prope.bias
    if bias is not None:
        assert torch.all(bias == 0), "o_prope.bias not zero-init"


def test_prope_block_forward_requires_viewmats() -> None:
    """PRoPE block must raise ``ValueError`` when ``viewmats`` is missing.

    A silent fallback would let the zero-init ``o_prope`` mask the
    missing camera binding -- the explicit raise surfaces the broken
    plumbing at the first block invocation rather than as confusing
    parity drift much later.
    """
    block = _make_prope_block()
    # Required by Block.forward's assertion (no checkpoint load in this test).
    block._parameters_updated_after_loading_checkpoint = True

    # ``cache`` and ``rope_freqs`` are never read on the viewmats=None
    # path (the ValueError fires first), but the block's forward
    # signature still typechecks them, so we pass placeholders.
    x = torch.zeros(1, 4, 64)
    e = torch.zeros(1, 6, 64)
    rope_freqs = torch.zeros(4, 1, 1, 32)

    with pytest.raises(ValueError, match="viewmats"):
        block(
            x=x,
            e=e,
            cache=cast(Any, object()),
            rope_freqs=rope_freqs,
            viewmats=None,
        )


def test_prope_self_attention_rejects_context_parallel() -> None:
    """``forward_dual_branch`` must raise ``NotImplementedError`` when CP > 1."""
    from hy_worldplay._camera import HyWorldPlayPRoPESelfAttention
    from torch.distributed import ProcessGroup  # noqa: F401  (typing-only)

    attn = HyWorldPlayPRoPESelfAttention(query_dim=64, n_heads=2, head_dim=32)

    # Stub ``is_context_parallel_enabled`` so we can hit the gate
    # without setting up a distributed mesh.
    attn.is_context_parallel_enabled = lambda: True  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

    with pytest.raises(NotImplementedError, match="context-parallel"):
        attn.forward_dual_branch(
            x=torch.zeros(1, 4, 64),
            kv_cache=cast(Any, object()),
            prope_kv_cache=cast(Any, object()),
            rope_freqs=cast(Any, None),
            viewmats=torch.zeros(1, 4, 4, 4),
            Ks=None,
        )


def test_hyworldplay_dit_network_use_prope_blocks_swaps_block_class() -> None:
    """``use_prope_blocks`` toggles ``_build_block`` between stock :class:`Block` and PRoPE."""
    from hy_worldplay._action import (
        HyWorldPlayWanDiTNetwork,
        HyWorldPlayWanDiTNetworkConfig,
    )
    from hy_worldplay._camera import HyWorldPlayPRoPEBlock

    from flashdreams.recipes.wan.transformer.impl.network import Block

    # Tiny but valid network config -- mirrors what
    # :class:`Wan21Transformer` would otherwise build at setup time.
    base_cfg = HyWorldPlayWanDiTNetworkConfig(use_prope_blocks=False)
    base_cfg.num_layers = 2
    base_cfg.num_heads = 4
    base_cfg.dim = 32
    base_cfg.ffn_dim = 64
    base_cfg.freq_dim = 32
    base_cfg.in_dim = 4
    base_cfg.out_dim = 4
    base_net = HyWorldPlayWanDiTNetwork(base_cfg)
    for block in base_net.blocks:
        assert isinstance(block, Block)
        assert not isinstance(block, HyWorldPlayPRoPEBlock)

    prope_cfg = HyWorldPlayWanDiTNetworkConfig(use_prope_blocks=True)
    prope_cfg.num_layers = 2
    prope_cfg.num_heads = 4
    prope_cfg.dim = 32
    prope_cfg.ffn_dim = 64
    prope_cfg.freq_dim = 32
    prope_cfg.in_dim = 4
    prope_cfg.out_dim = 4
    prope_net = HyWorldPlayWanDiTNetwork(prope_cfg)
    for block in prope_net.blocks:
        assert isinstance(block, HyWorldPlayPRoPEBlock)


## ---------------------------------------------------------------------------
## Encoder slicing
## ---------------------------------------------------------------------------


def _make_camera_encoder():
    """Build a HY-WorldPlay encoder skeleton, bypassing the heavy VAE init.

    The real :class:`I2VCtrlEncoder` constructor builds an inner
    :class:`WanVAEEncoder` that pulls in diffusers; the shape-validation
    tests only need the bookkeeping fields, so we sidestep ``__init__``.
    """
    from hy_worldplay._action import HyWorldPlayWanCtrlEncoder

    encoder = HyWorldPlayWanCtrlEncoder.__new__(HyWorldPlayWanCtrlEncoder)
    encoder._action_labels = None
    encoder._viewmats = None
    encoder._intrinsics = None
    return encoder


def test_encoder_set_camera_data_validates_shapes() -> None:
    """``set_camera_data`` rejects mismatched shape contracts up front."""
    encoder = _make_camera_encoder()

    # viewmats trailing shape must be (..., 4, 4).
    with pytest.raises(ValueError, match="viewmats"):
        encoder.set_camera_data(torch.zeros(1, 4, 3, 4), torch.zeros(1, 4, 3, 3))

    # Ks trailing shape must be (..., 3, 3).
    with pytest.raises(ValueError, match="Ks"):
        encoder.set_camera_data(torch.zeros(1, 4, 4, 4), torch.zeros(1, 4, 2, 3))

    # Leading dims must match.
    with pytest.raises(ValueError, match="leading dims"):
        encoder.set_camera_data(torch.zeros(2, 4, 4, 4), torch.zeros(1, 4, 3, 3))


def test_encoder_clear_camera_data_drops_state() -> None:
    """``clear_camera_data`` resets the per-rollout binding."""
    encoder = _make_camera_encoder()
    encoder.set_camera_data(torch.zeros(1, 4, 4, 4), torch.zeros(1, 4, 3, 3))
    assert encoder._viewmats is not None
    assert encoder._intrinsics is not None
    encoder.clear_camera_data()
    assert encoder._viewmats is None
    assert encoder._intrinsics is None
