# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from dataclasses import dataclass
from typing import Any, Literal, cast

import pytest
import torch

import flashdreams.core.attention.native as native_attention
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
    self_attention_backend: Literal["cudnn", "sage2"] = "cudnn"

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
    attn.apply_kv(
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


def test_wan_attention_backend_defaults_to_cudnn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load() -> None:
        raise AssertionError("default WAN config attempted to load SageAttention")

    monkeypatch.setattr(native_attention, "_load_sage2_op", fail_load)
    config = WanDiTNetworkConfig(
        dim=64,
        ffn_dim=128,
        num_heads=1,
        num_layers=1,
        cross_attn_enable_img=True,
    )

    network = config.setup()
    block = cast(wan_modules.Block, network.blocks[0])

    assert config.self_attention_backend == "cudnn"
    assert block.self_attn.attn_op.backend == "cudnn"
    assert block.cross_attn.attn_op.backend == "cudnn"
    assert block.cross_attn.attn_op_image.backend == "cudnn"


def test_wan_sage2_routes_only_to_self_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_attention,
        "_load_sage2_op",
        lambda: lambda *args, **kwargs: args[0],
    )
    transformer = Wan21Transformer(
        Wan21TransformerConfig(
            network=WanDiTNetworkConfig(
                dim=64,
                ffn_dim=128,
                num_heads=1,
                num_layers=1,
                cross_attn_enable_img=True,
                self_attention_backend="sage2",
            ),
            compile_network=False,
            use_cuda_graph=False,
        )
    )
    block = cast(wan_modules.Block, transformer.network.blocks[0])

    assert block.self_attn.attn_op.backend == "sage2"
    assert block.cross_attn.attn_op.backend == "cudnn"
    assert block.cross_attn.attn_op_image.backend == "cudnn"


def test_wan_sage2_rejects_cuda_graph_before_network_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_setup(config: Wan21TransformerConfig) -> _DummyNetwork:
        raise AssertionError("network construction should not start")

    monkeypatch.setattr(Wan21Transformer, "_setup_network", staticmethod(fail_setup))
    config = Wan21TransformerConfig(
        network=cast(
            WanDiTNetworkConfig,
            _DummyNetworkConfig(self_attention_backend="sage2"),
        ),
        compile_network=False,
        use_cuda_graph=True,
    )

    with pytest.raises(ValueError, match="incompatible with use_cuda_graph=True"):
        Wan21Transformer(config)
