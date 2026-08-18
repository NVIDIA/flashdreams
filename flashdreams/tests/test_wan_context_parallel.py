# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch

from flashdreams.accelerated.multi_head_attention import AttentionType
from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
    TritonMultiHeadAttention,
)
from flashdreams.core.attention.kvcache import BlockKVCache
from flashdreams.recipes.wan.transformer.impl import modules as wan_modules
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetworkConfig
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerConfig,
)

pytestmark = pytest.mark.ci_cpu


class _FakeProcessGroup:
    def __init__(self, world_size: int, rank: int) -> None:
        self._world_size = world_size
        self._rank = rank

    def size(self) -> int:
        return self._world_size

    def rank(self) -> int:
        return self._rank


class _DummyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cp_group = None
        self.parameters_updated = False

    def set_context_parallel_group(self, cp_group=None) -> None:
        self.cp_group = cp_group

    def update_parameters_after_loading_checkpoint(self) -> None:
        self.parameters_updated = True


class _IdentityAttention(torch.nn.Module):
    def forward(self, q, k, v):
        return q


@dataclass
class _DummyNetworkConfig:
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_dim: int = 16
    apply_rope_before_kvcache: bool = True

    def setup(self) -> _DummyNetwork:
        return _DummyNetwork()


def _mock_distributed(
    monkeypatch, world_size: int = 2, rank: int = 0
) -> _FakeProcessGroup:
    fake_group = _FakeProcessGroup(world_size=world_size, rank=rank)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        torch.distributed,
        "group",
        types.SimpleNamespace(WORLD=fake_group),
        raising=False,
    )
    return fake_group


def test_wan21_uses_world_cp_group_when_distributed(monkeypatch) -> None:
    fake_group = _mock_distributed(monkeypatch, world_size=2, rank=0)
    transformer = Wan21Transformer(
        Wan21TransformerConfig(
            network=cast(WanDiTNetworkConfig, _DummyNetworkConfig()),
            batch_shape=(1,),
            len_t=2,
            window_size_t=2,
            sink_size_t=0,
            compile_network=False,
        )
    )

    assert transformer._cp_group is fake_group
    assert transformer._cp_size == 2
    assert isinstance(transformer.network, _DummyNetwork)
    assert transformer.network.cp_group is fake_group
    assert transformer.network.parameters_updated


def test_kvcache_relative_rope_does_not_mutate_cached_keys(monkeypatch) -> None:
    """Read-time K RoPE is in-place, so it must rotate a temporary copy."""
    attn = wan_modules.MultiHeadAttention(
        query_dim=4,
        n_heads=1,
        head_dim=4,
        apply_rope_before_kvcache=False,
    )
    cast(Any, attn).attn_op = _IdentityAttention()

    cache_k = torch.randn(1, 3, 1, 4)
    cache_v = torch.randn(1, 3, 1, 4)
    cache = BlockKVCache.from_tensor(cache_k.clone(), cache_v, seq_dim=1)
    before = cache._k.clone()

    def _fake_apply_rope_freqs(x, freqs, interleaved=False):
        return x.add_(1.0)

    monkeypatch.setattr(wan_modules, "apply_rope_freqs", _fake_apply_rope_freqs)
    attn.query_kv(
        torch.randn(1, 3, 4),
        cache,
        rope_freqs_q=torch.zeros(3, 1, 1, 4),
        rope_freqs_k=torch.zeros(3, 1, 1, 4),
    )

    torch.testing.assert_close(cache._k, before)


def test_wan21_uses_no_cp_group_when_not_distributed(monkeypatch) -> None:
    """Without ``torch.distributed.init``, the transformer auto-detects cp_size=1."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    transformer = Wan21Transformer(
        Wan21TransformerConfig(
            network=cast(WanDiTNetworkConfig, _DummyNetworkConfig()),
            batch_shape=(1,),
            len_t=2,
            window_size_t=4,
            sink_size_t=0,
            compile_network=False,
        )
    )

    assert transformer._cp_size == 1
    assert transformer._cp_group is None


def test_wan21_requires_tokens_divisible_by_cp_size(monkeypatch) -> None:
    """Per-rollout ``(height, width)`` is checked at cache-init time."""
    _mock_distributed(monkeypatch, world_size=2, rank=0)
    transformer = Wan21Transformer(
        Wan21TransformerConfig(
            network=cast(WanDiTNetworkConfig, _DummyNetworkConfig()),
            batch_shape=(1,),
            len_t=1,
            window_size_t=1,
            sink_size_t=0,
            compile_network=False,
        )
    )
    with pytest.raises(AssertionError, match="must be divisible by cp_size=2"):
        transformer.initialize_autoregressive_cache(
            height=2,
            width=2,
            text_embeddings=torch.zeros(1, 1, 1),
        )


@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
def test_wan_network_propagates_sdpa_backend(
    sdpa_backend: SDPABackend,
) -> None:
    """Propagate the configured SDPA implementation through every DiT block."""
    config = WanDiTNetworkConfig(
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=1,
        patch_embedding_type="linear",
        attention_backend=wan_modules.AttentionBackend.TRITON,
        sdpa_backend=sdpa_backend,
    )

    network = config.setup()
    block = network.blocks[0]

    assert config.sdpa_backend is sdpa_backend
    assert network.sdpa_backend is sdpa_backend
    assert block.sdpa_backend is sdpa_backend
    assert isinstance(block.self_attn, TritonMultiHeadAttention)
    assert block.self_attn.qkv_fusion_option is QKVFusionOption.FULL
    assert block.self_attn.sdpa_backend is sdpa_backend
    assert isinstance(block.cross_attn, wan_modules.TritonCrossAttention)
    assert block.cross_attn.attention_type is AttentionType.CROSS_ATTENTION
    assert block.cross_attn.qkv_fusion_option is QKVFusionOption.FUSE_KV
    assert block.cross_attn.use_fp8 is True
    assert block.cross_attn.sdpa_backend is SDPABackend.TRITON


def test_triton_attention_registers_only_wan_checkpoint_names() -> None:
    """Expose logical Triton modules without adding generic checkpoint aliases."""
    reference = wan_modules.SelfAttention(64, n_heads=4, head_dim=16)
    actual = wan_modules.TritonSelfAttention(64, n_heads=4, head_dim=16)
    actual.load_state_dict(reference.state_dict(), strict=True)

    assert actual.query_projection is actual.q
    assert actual.key_projection is actual.k
    assert actual.value_projection is actual.v
    assert actual.output_projection is actual.o
    assert actual.query_norm is actual.norm_q
    assert actual.key_norm is actual.norm_k

    generic_names = {"q_proj", "k_proj", "v_proj", "output_proj", "q_norm", "k_norm"}
    assert generic_names.isdisjoint(actual._modules)
    assert set(actual.state_dict()) == set(reference.state_dict())
    assert actual._derived_weights.fused_qkv_weight is not None


def test_triton_cross_attention_registers_only_wan_checkpoint_names() -> None:
    """Load Wan T2V cross-attention weights without generic checkpoint aliases."""
    reference = wan_modules.CrossAttention(query_dim=64, n_heads=4, head_dim=16)
    actual = wan_modules.TritonCrossAttention(query_dim=64, n_heads=4, head_dim=16)
    actual.load_state_dict(reference.state_dict(), strict=True)

    assert actual.query_projection is actual.q
    assert actual.key_projection is actual.k
    assert actual.value_projection is actual.v
    assert actual.output_projection is actual.o
    assert actual.query_norm is actual.norm_q
    assert actual.key_norm is actual.norm_k

    generic_names = {"q_proj", "k_proj", "v_proj", "output_proj", "q_norm", "k_norm"}
    assert generic_names.isdisjoint(actual._modules)
    assert set(actual.state_dict()) == set(reference.state_dict())
    assert actual._derived_weights.fused_kv_weight is not None


def test_triton_backend_keeps_native_i2v_cross_attention() -> None:
    """Keep Wan's dual text/image cross-attention when I2V is enabled."""
    reference = wan_modules.Block(
        dim=64,
        ffn_dim=128,
        num_heads=4,
        i2v=True,
    )
    actual = wan_modules.Block(
        dim=64,
        ffn_dim=128,
        num_heads=4,
        i2v=True,
        attention_backend=wan_modules.AttentionBackend.TRITON,
    )
    actual.load_state_dict(reference.state_dict(), strict=True)

    assert isinstance(actual.self_attn, wan_modules.TritonSelfAttention)
    assert isinstance(actual.cross_attn, wan_modules.CrossAttention)
    assert not isinstance(actual.cross_attn, wan_modules.TritonCrossAttention)
    assert actual.cross_attn.i2v is True


def test_wan_patchify_unpatchify_round_trip_without_cp() -> None:
    network = WanDiTNetworkConfig(
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=1,
        patch_embedding_type="linear",
    ).setup()
    latent = torch.randn(1, 2, 16, 4, 4)

    patched = network.patchify_and_maybe_split_cp(
        latent,
        process_groups=[None],
        cp_dims=[-2],
    )
    restored = network.unpatchify_and_maybe_gather_cp(
        pH=2,
        pW=2,
        x=patched,
        process_groups=[None],
        cp_dims=[-2],
    )

    assert patched.shape == (1, 8, 64)
    torch.testing.assert_close(restored, latent)
