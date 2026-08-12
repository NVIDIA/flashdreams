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

"""CPU-only unit tests for the mid-stream text-edit path.

Covers the invariants a live prompt swap depends on:

* ``BlockKVCache.overwrite_kv_`` replaces contents without moving storage
  (CUDA-graph safety) and rejects shape drift.
* Same-index cache rewrites (the ReCache primitive) overwrite only the last
  chunk's slots and leave bookkeeping untouched.
* ``CosmosDiTNetwork.replace_text_embeddings`` reproduces exactly the
  cross-attn K/V a fresh ``initialize_cache`` would build for the new
  prompt, in place, without touching self-attention history.
* ``CosmosTransformer.replace_text_embeddings`` snapshots old/new K/V for
  text-edit guidance, and ``predict_flow`` combines the two branches
  CFG-style, leaving the buffers on the new prompt.
* The guidance countdown clears after the requested number of chunks and
  ignores same-index re-opens.
"""

from __future__ import annotations

import pytest
import torch
from omnidreams.transformer import (
    CosmosTransformer,
    CosmosTransformerConfig,
    TextEditGuidance,
)
from omnidreams.transformer.impl.network import (
    CosmosDiTNetwork,
    CosmosDiTNetworkConfig,
)

from flashdreams.core.attention.kvcache import BlockKVCache

pytestmark = pytest.mark.ci_cpu


## BlockKVCache primitives


def _make_cross_attn_cache(L: int = 6, n: int = 2, d: int = 4) -> BlockKVCache:
    k = torch.randn(1, L, n, d)
    v = torch.randn(1, L, n, d)
    return BlockKVCache.from_tensor(k, v, seq_dim=-3)


def test_overwrite_kv_preserves_addresses_and_content():
    torch.manual_seed(0)
    cache = _make_cross_attn_cache()
    k_ptr = cache._k.data_ptr()
    v_ptr = cache._v.data_ptr()

    new_k = torch.randn_like(cache._k)
    new_v = torch.randn_like(cache._v)
    cache.overwrite_kv_(new_k, new_v)

    assert cache._k.data_ptr() == k_ptr
    assert cache._v.data_ptr() == v_ptr
    assert torch.equal(cache.cached_k(), new_k)
    assert torch.equal(cache.cached_v(), new_v)


def test_overwrite_kv_rejects_shape_mismatch():
    cache = _make_cross_attn_cache(L=6)
    bad_k = torch.randn(1, 5, 2, 4)
    bad_v = torch.randn(1, 5, 2, 4)
    with pytest.raises(AssertionError, match="shape mismatch"):
        cache.overwrite_kv_(bad_k, bad_v)


def test_clone_kv_returns_detached_copies():
    torch.manual_seed(0)
    cache = _make_cross_attn_cache()
    k_clone, v_clone = cache.clone_kv()
    assert k_clone.data_ptr() != cache._k.data_ptr()
    k_before = cache.cached_k().clone()
    k_clone.fill_(0.0)
    v_clone.fill_(0.0)
    assert torch.equal(cache.cached_k(), k_before)


def test_same_index_rewrite_overwrites_last_chunk_only():
    """ReCache primitive: re-opening the just-committed chunk index rewrites
    the same physical slots without rolling the window or advancing
    bookkeeping."""
    torch.manual_seed(0)
    chunk, n_chunks = 4, 4
    cache = BlockKVCache(
        k_shape=(1, chunk * n_chunks, 2, 4),
        v_shape=(1, chunk * n_chunks, 2, 4),
        seq_dim=-3,
        chunk_size=chunk,
        window_size=chunk * n_chunks,
        sink_size=0,
        device="cpu",
        dtype=torch.float32,
    )
    chunks = [torch.randn(1, chunk, 2, 4) for _ in range(3)]
    for idx, c in enumerate(chunks):
        cache.before_update(idx)
        cache.update(c, c)
        cache.after_update(idx)
    n_cached, prev_idx = cache._n_cached, cache._prev_chunk_idx

    replacement = torch.randn(1, chunk, 2, 4)
    cache.before_update(2)
    cache.update(replacement, replacement)
    cache.after_update(2)

    assert cache._n_cached == n_cached
    assert cache._prev_chunk_idx == prev_idx
    got_k = cache._k[:, : 3 * chunk]
    assert torch.equal(got_k[:, :chunk], chunks[0])
    assert torch.equal(got_k[:, chunk : 2 * chunk], chunks[1])
    assert torch.equal(got_k[:, 2 * chunk :], replacement)

    # The rollout continues normally afterwards.
    cache.before_update(3)
    cache.update(chunks[0], chunks[0])
    cache.after_update(3)
    assert cache._prev_chunk_idx == 3


## Network-level replace


def _tiny_network(seed: int = 0) -> CosmosDiTNetwork:
    torch.manual_seed(seed)
    config = CosmosDiTNetworkConfig(
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=64,
        num_blocks=2,
        num_heads=4,
        adaln_lora_dim=8,
        crossattn_proj_in_channels=32,
        crossattn_emb_channels=16,
        additional_concat_ch=0,
        enable_cross_view_attn=False,
    )
    return CosmosDiTNetwork(config)


def test_network_replace_matches_fresh_init_and_keeps_self_attn():
    torch.manual_seed(0)
    network = _tiny_network()
    ctx1 = torch.randn(1, 1, 10, 32)
    ctx2 = torch.randn(1, 1, 10, 32)

    cache = network.initialize_cache(
        chunk_size=32, window_size=96, sink_size=0, context=ctx1
    )
    reference = network.initialize_cache(
        chunk_size=32, window_size=96, sink_size=0, context=ctx2
    )

    cross_ptrs = [bc.cross_attn._k.data_ptr() for bc in cache.block_caches]
    self_ptrs = [bc.self_attn._k.data_ptr() for bc in cache.block_caches]
    self_snapshot = [bc.self_attn.clone_kv() for bc in cache.block_caches]

    network.replace_text_embeddings(cache, ctx2)

    for bc, ref, cross_ptr, self_ptr, (self_k, self_v) in zip(
        cache.block_caches, reference.block_caches, cross_ptrs, self_ptrs, self_snapshot
    ):
        assert torch.equal(bc.cross_attn._k, ref.cross_attn._k)
        assert torch.equal(bc.cross_attn._v, ref.cross_attn._v)
        assert bc.cross_attn._k.data_ptr() == cross_ptr
        assert bc.self_attn._k.data_ptr() == self_ptr
        assert torch.equal(bc.self_attn._k, self_k)
        assert torch.equal(bc.self_attn._v, self_v)


## Transformer-level replace + guidance


def _tiny_transformer(seed: int = 0) -> CosmosTransformer:
    torch.manual_seed(seed)
    config = CosmosTransformerConfig(
        network=CosmosDiTNetworkConfig(
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=64,
            num_blocks=2,
            num_heads=4,
            adaln_lora_dim=8,
            crossattn_proj_in_channels=32,
            crossattn_emb_channels=16,
            additional_concat_ch=0,
            enable_cross_view_attn=False,
        ),
        checkpoint_path=None,
        batch_shape=(1,),
        num_views=1,
        len_t=2,
        window_size_t=6,
        sink_size_t=0,
        compile_network=False,
        use_cuda_graph=False,
        guidance_scale=1.0,
    )
    return CosmosTransformer(config)


def _init_cache(transformer: CosmosTransformer, seed: int = 1):
    torch.manual_seed(seed)
    text = torch.randn(1, 1, 10, 32)
    image = torch.randn(1, 1, 1, 16, 8, 8)
    cache = transformer.initialize_autoregressive_cache(
        height=8, width=8, text_embeddings=text, image_embeddings=image
    )
    return cache, text


def test_transformer_replace_snapshots_old_and_new_kv():
    transformer = _tiny_transformer()
    cache, _ = _init_cache(transformer)
    old_kv = [bc.cross_attn.clone_kv() for bc in cache.network_cache.block_caches]

    new_text = torch.randn(1, 1, 10, 32)
    transformer.replace_text_embeddings(
        cache, new_text, guidance_scale=2.0, guidance_chunks=3
    )

    guidance = cache.text_edit_guidance
    assert guidance is not None
    assert guidance.scale == 2.0 and guidance.chunks_remaining == 3
    for (k_old, v_old), (k_ref, v_ref) in zip(guidance.kv_old, old_kv):
        assert torch.equal(k_old, k_ref)
        assert torch.equal(v_old, v_ref)
    # Buffers and the "new" snapshot both hold the new prompt's K/V.
    for (k_new, v_new), bc in zip(guidance.kv_new, cache.network_cache.block_caches):
        assert torch.equal(k_new, bc.cross_attn.cached_k())
        assert torch.equal(v_new, bc.cross_attn.cached_v())
        assert not torch.equal(k_new, guidance.kv_old[0][0])

    # A follow-up plain swap (no guidance) clears the guidance state.
    transformer.replace_text_embeddings(cache, torch.randn(1, 1, 10, 32))
    assert cache.text_edit_guidance is None


def test_predict_flow_guidance_combines_and_lands_on_new_kv():
    transformer = _tiny_transformer()
    cache, _ = _init_cache(transformer)
    block_caches = cache.network_cache.block_caches

    kv_old = [
        (torch.zeros_like(bc.cross_attn._k), torch.zeros_like(bc.cross_attn._v))
        for bc in block_caches
    ]
    kv_new = [
        (torch.ones_like(bc.cross_attn._k), torch.ones_like(bc.cross_attn._v))
        for bc in block_caches
    ]
    cache.text_edit_guidance = TextEditGuidance(
        scale=3.0, chunks_remaining=1, kv_old=kv_old, kv_new=kv_new
    )

    # Stub the branch forward: report the current block-0 cross-K content so
    # the test observes which prompt each branch ran under (old=0, new=1).
    def fake_branch(**kwargs):
        return block_caches[0].cross_attn.cached_k().mean() * torch.ones(4)

    transformer._predict_branch = fake_branch  # ty: ignore[invalid-assignment]

    flow = transformer.predict_flow(
        noisy_latent=torch.zeros(4),
        timestep=torch.tensor(1000.0),
        cache=cache,
    )
    # flow_old + scale * (flow_new - flow_old) = 0 + 3 * (1 - 0)
    assert torch.allclose(flow, torch.full((4,), 3.0))
    for bc, (k_new, v_new) in zip(block_caches, kv_new):
        assert torch.equal(bc.cross_attn._k, k_new)
        assert torch.equal(bc.cross_attn._v, v_new)

    # The KV-commit forward must run single-branch under the new prompt.
    transformer._finalizing_kv_cache = True
    flow = transformer.predict_flow(
        noisy_latent=torch.zeros(4),
        timestep=torch.tensor(128.0),
        cache=cache,
    )
    assert torch.allclose(flow, torch.ones(4))


def test_guidance_countdown_clears_after_n_chunks():
    transformer = _tiny_transformer()
    cache, _ = _init_cache(transformer)
    transformer.replace_text_embeddings(
        cache,
        torch.randn(1, 1, 10, 32),
        guidance_scale=2.0,
        guidance_chunks=2,
    )
    assert cache.text_edit_guidance is not None

    cache.start(0)
    assert cache.text_edit_guidance is not None  # guided chunk 1 of 2
    assert cache.text_edit_guidance.chunks_remaining == 1
    cache.finalize(0)

    # A same-index re-open (ReCache of chunk 0) must not consume a chunk.
    cache.start(0)
    assert cache.text_edit_guidance.chunks_remaining == 1
    cache.finalize(0)

    cache.start(1)
    assert cache.text_edit_guidance is not None  # guided chunk 2 of 2
    assert cache.text_edit_guidance.chunks_remaining == 0
    cache.finalize(1)

    cache.start(2)
    assert cache.text_edit_guidance is None  # guidance expired
    cache.finalize(2)


def test_replace_rejects_native_dit_and_cfg_guidance_combination():
    transformer = _tiny_transformer()
    cache, _ = _init_cache(transformer)

    transformer._optimized_dit_executor = object()
    with pytest.raises(NotImplementedError):
        transformer.replace_text_embeddings(cache, torch.randn(1, 1, 10, 32))
    transformer._optimized_dit_executor = None

    cache.network_cache_uncond = cache.network_cache  # any non-None sentinel
    with pytest.raises(AssertionError, match="mutually exclusive"):
        transformer.replace_text_embeddings(
            cache,
            torch.randn(1, 1, 10, 32),
            guidance_scale=2.0,
            guidance_chunks=1,
        )
    # A plain swap (no guidance) is still fine with CFG configs.
    cache.network_cache_uncond = None
    transformer.replace_text_embeddings(cache, torch.randn(1, 1, 10, 32))


## ReCache RNG neutrality


def test_recache_uses_dedicated_rng_and_restores_model_stream():
    """ReCache draws its context noise from a per-index seeded generator and
    leaves the model RNG stream exactly where it was."""
    from omnidreams.pipeline import OmnidreamsPipeline

    pipe = OmnidreamsPipeline.__new__(OmnidreamsPipeline)

    class FakeCache:
        autoregressive_index = 7
        started = None

        def start(self, idx):
            self.started = idx

    class FakeFinalState:
        autoregressive_index = 7
        cache = FakeCache()

    class FakeDM:
        device = torch.device("cpu")

        def __init__(self):
            self._rng = torch.Generator().manual_seed(42)
            self.seen_seed = None

        @property
        def rng(self):
            return self._rng

        def finalize(self, final_state):
            self.seen_seed = self._rng.initial_seed()

    dm = FakeDM()
    rollout_rng = dm._rng
    state_before = rollout_rng.get_state().clone()
    pipe.diffusion_model = dm

    class FakePipelineCache:
        final_state = FakeFinalState()

    pipe.recache_last_chunk(FakePipelineCache())
    assert dm.seen_seed == OmnidreamsPipeline._RECACHE_NOISE_SEED + 7
    assert dm._rng is rollout_rng  # restored, same object
    assert torch.equal(rollout_rng.get_state(), state_before)  # untouched
    assert FakeFinalState.cache.started == 7

    # No final state -> no-op.
    class EmptyCache:
        final_state = None

    pipe.recache_last_chunk(EmptyCache())
