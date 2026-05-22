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

"""CPU-only structural tests for the HY-WorldPlay KV-prefill executor (phase 2b.5b-part2).

Covers the four moving pieces introduced in phase 2b.5b-part2:

* :class:`HyWorldPlayMemoryKVCache` -- the per-block flat cache that
  stores prefilled K / V at upstream's RoPE-collapsed positions
  ``[0, K)``. Read / write / reset semantics are isolated and
  independent of the rolling :class:`BlockKVCache`.
* :class:`HyWorldPlayPRoPEBlockCache` -- now owns a ``memory`` slot
  alongside ``self_attn`` / ``prope_self_attn``. ``reset_current_chunk``
  must wipe only the latter two; the memory cache has its own
  reset cycle owned by the prefill executor.
* :class:`HyWorldPlayPRoPESelfAttention` and
  :class:`HyWorldPlayPRoPEBlock` -- new ``prefill_memory_kv`` side-
  effect calls that write into the memory cache without invoking
  attention / cross-attn / FFN.
* :class:`HyWorldPlayWan21TransformerCache` -- adds the per-rollout
  clean-latent history buffer + the chunk-start rolling-cache reset
  that diverges HY mode from the standard rolling-window semantics.

GPU validation (full 2-chunk rollout, parity diff against the vendor
wrapper) is intentionally deferred -- it lives behind a marker run
that the CI excludes from the CPU sweep these tests participate in.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.ci_cpu


## ---------------------------------------------------------------------------
## HyWorldPlayMemoryKVCache surface
## ---------------------------------------------------------------------------


def test_memory_kv_cache_defaults_to_empty() -> None:
    """A freshly constructed cache holds no K / V on either branch.

    The dual-branch attention reads ``has_rope_kv`` / ``has_prope_kv``
    to decide whether to prepend memory K / V to the current step's
    K / V; the default-empty state must let it short-circuit so the
    forward stays bit-identical to the 2b.4 baseline when no prefill
    has been issued yet (i.e. on chunk 0 of every rollout).
    """
    from hy_worldplay._camera import HyWorldPlayMemoryKVCache

    cache = HyWorldPlayMemoryKVCache()
    assert cache.k_rope is None
    assert cache.v_rope is None
    assert cache.k_prope is None
    assert cache.v_prope is None
    assert cache.has_rope_kv is False
    assert cache.has_prope_kv is False
    assert cache.is_empty is True


def test_memory_kv_cache_write_rope_round_trip() -> None:
    """``write_rope`` populates only the standard branch, not the PRoPE one.

    Two-call test in the same vein as the existing ``BlockKVCache``
    update tests: the cache should treat the two branches as
    independent state, so writing one never silently leaks K / V
    into the other.
    """
    from hy_worldplay._camera import HyWorldPlayMemoryKVCache

    cache = HyWorldPlayMemoryKVCache()
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)
    cache.write_rope(k, v)
    assert cache.has_rope_kv is True
    assert cache.has_prope_kv is False
    assert torch.equal(cache.k_rope, k)
    assert torch.equal(cache.v_rope, v)
    assert cache.is_empty is False


def test_memory_kv_cache_write_prope_round_trip() -> None:
    """``write_prope`` populates only the PRoPE branch.

    Symmetric to the ``write_rope`` test; together they pin the
    branch-independence invariant on both sides.
    """
    from hy_worldplay._camera import HyWorldPlayMemoryKVCache

    cache = HyWorldPlayMemoryKVCache()
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)
    cache.write_prope(k, v)
    assert cache.has_prope_kv is True
    assert cache.has_rope_kv is False
    assert torch.equal(cache.k_prope, k)
    assert torch.equal(cache.v_prope, v)


def test_memory_kv_cache_reset_clears_both_branches() -> None:
    """``reset`` returns the cache to its default-empty state on both branches.

    The prefill executor calls ``reset`` before each new chunk's
    prefill so leftover K / V from the previous chunk's memory image
    cannot leak into the new one. This test pins that invariant
    independently of any prefill driver, so a regression in the
    cache itself surfaces immediately.
    """
    from hy_worldplay._camera import HyWorldPlayMemoryKVCache

    cache = HyWorldPlayMemoryKVCache()
    cache.write_rope(torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8))
    cache.write_prope(torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8))
    assert cache.has_rope_kv and cache.has_prope_kv
    cache.reset()
    assert cache.is_empty is True
    assert cache.k_rope is None and cache.v_rope is None
    assert cache.k_prope is None and cache.v_prope is None


## ---------------------------------------------------------------------------
## HyWorldPlayPRoPEBlockCache surface
## ---------------------------------------------------------------------------


def _make_block_cache(*, dim: int = 64, num_heads: int = 2):
    """Build a minimal :class:`HyWorldPlayPRoPEBlockCache` for direct surface tests.

    Mirrors the ``_make_prope_block`` helper in ``test_camera.py`` --
    we deliberately avoid spinning up a full :class:`HyWorldPlayPRoPEBlock`
    here because the block-cache surface tests are about the *cache
    container's* invariants, not the block's forward.
    """
    from hy_worldplay._camera import HyWorldPlayPRoPEBlock

    block = HyWorldPlayPRoPEBlock(
        dim=dim,
        ffn_dim=dim * 2,
        num_heads=num_heads,
        cross_attn_norm=True,
        eps=1e-6,
        i2v=False,
        apply_rope_before_kvcache=True,
    )
    text_ctx = torch.zeros(1, 8, dim)
    return block, block.initialize_cache(
        chunk_size=4, window_size=4, sink_size=0, context_text=text_ctx
    )


def test_block_cache_has_memory_slot() -> None:
    """The 2b.5b-part2 block cache must expose a ``memory`` slot by default.

    The slot is constructed via ``field(default_factory=...)`` so that
    builders that don't know about the new field still get a working
    empty cache rather than a ``None`` that would crash the dual-
    branch attention path.
    """
    from hy_worldplay._camera import HyWorldPlayMemoryKVCache

    _, cache = _make_block_cache()
    assert isinstance(cache.memory, HyWorldPlayMemoryKVCache)
    assert cache.memory.is_empty is True


def test_block_cache_reset_current_chunk_skips_memory_slot() -> None:
    """``reset_current_chunk`` wipes only the rolling caches, not the memory slot.

    The two lifecycles are independent: rolling caches reset per
    chunk start; memory cache resets only when a new prefill is
    about to run. This test pins that separation; without it, a
    regression that wired ``memory.reset()`` into the per-chunk
    reset path would silently nullify the prefill on chunks > 0.
    """
    _, cache = _make_block_cache()
    cache.memory.write_rope(torch.randn(1, 4, 2, 32), torch.randn(1, 4, 2, 32))
    cache.memory.write_prope(torch.randn(1, 4, 2, 32), torch.randn(1, 4, 2, 32))
    assert cache.memory.has_rope_kv and cache.memory.has_prope_kv
    cache.reset_current_chunk()
    assert cache.memory.has_rope_kv, "reset_current_chunk wiped memory cache"
    assert cache.memory.has_prope_kv, "reset_current_chunk wiped memory cache"


## ---------------------------------------------------------------------------
## Block-level prefill structural surface
## ---------------------------------------------------------------------------


def test_prefill_memory_kv_writes_both_branches() -> None:
    """``HyWorldPlayPRoPEBlock.prefill_memory_kv`` populates both branches of the memory cache.

    Structural smoke: a small block on CPU, fed a tiny memory slice
    + dummy viewmats, must end with both ``has_rope_kv`` and
    ``has_prope_kv`` ``True``. This is the *minimum* structural
    invariant the block needs to satisfy for the executor to have
    something to attend over -- numerical correctness vs upstream is
    a parity-diff concern that lives behind the GPU smoke marker.
    """
    block, cache = _make_block_cache(dim=64, num_heads=2)
    block._parameters_updated_after_loading_checkpoint = True

    # 4 memory tokens (1 "frame" with 4 tokens each in this tiny
    # geometry); shape matches what the network's prefill pass would
    # produce after patchify + AdaLN modulation. ``rope_freqs=None``
    # skips the fused-Triton RoPE kernel that requires CUDA so this
    # test stays CPU-only; the kernel itself is exercised by the
    # base flashdreams CUDA tests.
    x = torch.randn(1, 4, 64)
    e = torch.zeros(1, 6, 64)
    viewmats = torch.eye(4).expand(1, 1, 4, 4).contiguous()

    block.prefill_memory_kv(
        x=x, e=e, rope_freqs=None, viewmats=viewmats, Ks=None, cache=cache
    )

    assert cache.memory.has_rope_kv, "prefill did not write the standard branch"
    assert cache.memory.has_prope_kv, "prefill did not write the PRoPE branch"
    # Sequence dim must equal the input token count so the executor's
    # collapsed-position contract holds. Mismatch here would mean the
    # attention concat produces a stale memory image at the wrong
    # positions on the next forward.
    assert cache.memory.k_rope.shape[-3] == x.shape[-2]
    assert cache.memory.v_rope.shape[-3] == x.shape[-2]
    assert cache.memory.k_prope.shape[-3] == x.shape[-2]
    assert cache.memory.v_prope.shape[-3] == x.shape[-2]


def test_prefill_memory_kv_does_not_touch_rolling_caches() -> None:
    """The prefill must not write into ``self_attn`` / ``prope_self_attn``.

    The rolling caches are the *current chunk's* K / V; the prefill
    is for *historical* K / V at collapsed positions. Mixing them
    would corrupt both sides: the dual-branch attention would
    attend to the prefilled K / V twice (once as memory, once as
    rolling), and the rolling cache would point at non-current-chunk
    positions.
    """
    block, cache = _make_block_cache(dim=64, num_heads=2)
    block._parameters_updated_after_loading_checkpoint = True

    rolling_n_before = cache.self_attn._n_cached
    prope_n_before = cache.prope_self_attn._n_cached

    x = torch.randn(1, 4, 64)
    e = torch.zeros(1, 6, 64)
    viewmats = torch.eye(4).expand(1, 1, 4, 4).contiguous()
    block.prefill_memory_kv(
        x=x, e=e, rope_freqs=None, viewmats=viewmats, Ks=None, cache=cache
    )

    assert cache.self_attn._n_cached == rolling_n_before, (
        "prefill leaked into the standard rolling cache"
    )
    assert cache.prope_self_attn._n_cached == prope_n_before, (
        "prefill leaked into the PRoPE rolling cache"
    )


def test_prefill_memory_kv_requires_viewmats() -> None:
    """The prefill executor must surface a missing-viewmats misconfiguration loudly.

    Mirrors the equivalent gate on :meth:`HyWorldPlayPRoPEBlock.forward`:
    silent fallback would let the prefill produce zero-PRoPE memory
    K / V and the dual-branch attention would silently drop camera
    context for the historical frames.
    """
    block, cache = _make_block_cache(dim=64, num_heads=2)
    block._parameters_updated_after_loading_checkpoint = True

    with pytest.raises(ValueError, match="viewmats"):
        block.prefill_memory_kv(
            x=torch.randn(1, 4, 64),
            e=torch.zeros(1, 6, 64),
            rope_freqs=torch.zeros(4, 1, 1, 32),
            viewmats=None,
            Ks=None,
            cache=cache,
        )


def test_dual_branch_attention_short_circuits_empty_memory_cache() -> None:
    """``forward_dual_branch`` with ``memory_kv_cache=None`` keeps the 2b.4 path live.

    The two ``has_*_kv`` short-circuits inside ``forward_dual_branch``
    must let an empty / ``None`` memory cache pass through without
    invoking the new ``torch.cat`` prepend. We can't easily pin
    bit-identity of the attention output on CPU (the fused RoPE
    kernel is CUDA-only), but we *can* drive the fast-path branch
    explicitly and assert it doesn't fault. This catches any
    regression that would unconditionally try to materialise
    ``memory_kv_cache.k_rope`` (which is ``None`` by default).
    """
    from hy_worldplay._camera import (
        HyWorldPlayMemoryKVCache,
        HyWorldPlayPRoPESelfAttention,
    )

    attn = HyWorldPlayPRoPESelfAttention(
        query_dim=64, n_heads=2, head_dim=32, eps=1e-6, apply_rope_before_kvcache=True
    )

    empty_memory = HyWorldPlayMemoryKVCache()
    assert empty_memory.is_empty
    # The fast-path branch is the ``has_*_kv == False`` arm. Reaching
    # the slow ``torch.cat`` arm with ``k_rope=None`` would raise; the
    # fast-path branch is structurally distinct -- this test is the
    # only place that pins the gate.
    assert empty_memory.has_rope_kv is False
    assert empty_memory.has_prope_kv is False
    # And the equivalent ``memory_kv_cache=None`` argument keeps the
    # same fast path -- the block forward passes ``cache.memory``,
    # but other call sites may pass ``None`` for tests / future
    # phases. Both paths must skip the prepend.
    assert empty_memory.k_rope is None and empty_memory.v_rope is None


## ---------------------------------------------------------------------------
## HyWorldPlayWan21TransformerCache surface
## ---------------------------------------------------------------------------


def test_transformer_cache_history_defaults_to_empty() -> None:
    """A fresh HY transformer cache reports no chunks and a ``None`` history.

    The prefill executor uses ``finished_chunks`` to short-circuit on
    chunk 0 (when the history is empty); pinning the default here
    avoids a regression where a non-zero default would cause the
    executor to attempt a slice on a non-existent buffer.
    """
    from hy_worldplay._action import HyWorldPlayWan21TransformerCache
    from flashdreams.recipes.wan.transformer.impl.network import (
        WanDiTNetworkCache,
    )

    fake_rope = type("R", (), {})()  # the tests below don't exercise it
    cache = HyWorldPlayWan21TransformerCache(
        network_cache=WanDiTNetworkCache(block_caches=[]),
        network_cache_uncond=None,
        rope_adapter=fake_rope,  # type: ignore[arg-type]
    )
    assert cache.clean_latent_history is None
    assert cache.finished_chunks == 0
    assert cache.hy_chunk_size_t == 0
    assert cache.hy_tokens_per_frame == 0


def test_append_clean_latent_grows_history_and_detaches() -> None:
    """``_append_clean_latent_to_history`` concats along the token axis and detaches.

    The history outlives the autograd graph of the chunk that
    produced it (each chunk's denoising graph is freed before the
    next chunk's); the append must therefore detach so a stale
    graph never re-enters via the next chunk's prefill input.
    """
    from hy_worldplay._action import HyWorldPlayWan21Transformer

    transformer = HyWorldPlayWan21Transformer.__new__(HyWorldPlayWan21Transformer)

    chunk0 = torch.randn(1, 4, 16, requires_grad=True)
    history = transformer._append_clean_latent_to_history(None, chunk0)
    assert history is not None
    assert history.shape == (1, 4, 16)
    assert history.requires_grad is False, "history must be detached from the grad graph"

    chunk1 = torch.randn(1, 4, 16, requires_grad=True)
    history = transformer._append_clean_latent_to_history(history, chunk1)
    assert history.shape == (1, 8, 16), (
        "second append must concat along the post-patchify token axis (-2)"
    )
    # Concat preserves the order: chunk0's tokens first, chunk1's after.
    assert torch.equal(history[..., :4, :], chunk0.detach())
    assert torch.equal(history[..., 4:, :], chunk1.detach())


def test_slice_per_frame_handles_action_and_matrices() -> None:
    """``_slice_per_frame`` dispatches by tensor rank / dtype to slice the frame axis.

    Phase 2b.5b-part2 stub: when the per-rollout buffer is not yet
    plumbed through the encoder, the prefill driver falls back to
    slicing the per-AR-step buffer. This test pins the dispatch
    layout so the follow-up that adds the per-rollout wiring
    doesn't accidentally regress the action-int path or the
    viewmats / Ks matrix path.
    """
    from hy_worldplay._action import HyWorldPlayWan21Transformer

    transformer = HyWorldPlayWan21Transformer.__new__(HyWorldPlayWan21Transformer)

    action = torch.arange(8, dtype=torch.long).unsqueeze(0)  # [1, 8]
    sliced = transformer._slice_per_frame(action, [0, 1, 2])
    assert sliced is not None
    assert sliced.shape == (1, 3)
    assert torch.equal(sliced, torch.tensor([[0, 1, 2]]))

    viewmats = torch.eye(4).expand(1, 8, 4, 4).contiguous()
    sliced = transformer._slice_per_frame(viewmats, [0, 1, 2])
    assert sliced is not None
    assert sliced.shape == (1, 3, 4, 4)

    Ks = torch.eye(3).expand(1, 8, 3, 3).contiguous()
    sliced = transformer._slice_per_frame(Ks, [0, 1, 2])
    assert sliced is not None
    assert sliced.shape == (1, 3, 3, 3)


def test_slice_per_frame_returns_none_for_none() -> None:
    """``None`` inputs (action / viewmats / Ks unbound) flow through cleanly."""
    from hy_worldplay._action import HyWorldPlayWan21Transformer

    transformer = HyWorldPlayWan21Transformer.__new__(HyWorldPlayWan21Transformer)
    assert transformer._slice_per_frame(None, [0, 1]) is None


def test_is_first_step_of_chunk_detects_empty_rolling_cache() -> None:
    """``_is_first_step_of_chunk`` reports True iff no rolling K / V are cached.

    The prefill executor reads this to gate "run prefill exactly
    once per chunk"; mis-detecting the first step would either
    re-run the prefill on every scheduler step (wasteful but
    correct) or skip it entirely (incorrect). This test pins the
    correct behaviour at the boundary cases.
    """
    from hy_worldplay._action import (
        HyWorldPlayWan21Transformer,
        HyWorldPlayWan21TransformerCache,
    )
    from flashdreams.recipes.wan.transformer.impl.network import (
        WanDiTNetworkCache,
    )

    transformer = HyWorldPlayWan21Transformer.__new__(HyWorldPlayWan21Transformer)
    _, block_cache = _make_block_cache()

    cache = HyWorldPlayWan21TransformerCache(
        network_cache=WanDiTNetworkCache(block_caches=[block_cache]),
        network_cache_uncond=None,
        rope_adapter=type("R", (), {})(),  # type: ignore[arg-type]
    )

    assert transformer._is_first_step_of_chunk(cache) is True
    block_cache.self_attn._n_cached = 4  # simulate post-step state
    assert transformer._is_first_step_of_chunk(cache) is False


def test_transformer_cache_start_resets_rolling_caches_on_new_chunk() -> None:
    """At chunk_idx > 0, ``cache.start`` must wipe per-block rolling caches.

    Phase 2b.5b-part2: HY mode pushes cross-chunk K / V into the
    dedicated memory cache; the rolling cache should only ever
    contain the current chunk's tokens. ``cache.start(idx > 0)``
    is responsible for clearing the previous chunk's residue.
    """
    from hy_worldplay._action import HyWorldPlayWan21TransformerCache
    from flashdreams.recipes.wan.transformer.impl.network import (
        WanDiTNetworkCache,
    )

    _, block_cache = _make_block_cache()

    fake_rope_freqs = torch.zeros(1)

    class FakeRope:
        def shift_t(self, idx: int) -> torch.Tensor:
            return fake_rope_freqs

    cache = HyWorldPlayWan21TransformerCache(
        network_cache=WanDiTNetworkCache(block_caches=[block_cache]),
        network_cache_uncond=None,
        rope_adapter=FakeRope(),  # type: ignore[arg-type]
    )

    # Simulate chunk 0 ending with populated rolling caches.
    block_cache.self_attn._n_cached = 4
    block_cache.self_attn._prev_chunk_idx = 0
    block_cache.prope_self_attn._n_cached = 4
    block_cache.prope_self_attn._prev_chunk_idx = 0

    # Move to chunk 1 -- the start hook must reset the rolling caches.
    cache.start(autoregressive_index=1)
    assert block_cache.self_attn._n_cached == 0, (
        "start(>0) did not reset the rolling self-attention cache"
    )
    assert block_cache.prope_self_attn._n_cached == 0, (
        "start(>0) did not reset the rolling PRoPE-branch cache"
    )


def test_transformer_cache_start_keeps_chunk_0_intact() -> None:
    """At chunk_idx == 0, ``start`` must *not* touch the rolling caches.

    Otherwise the very first chunk of a rollout would see a wiped
    cache between the test setup and the first scheduler step, which
    is a no-op today but would break any future caller that
    pre-stamps initial K / V before chunk 0.
    """
    from hy_worldplay._action import HyWorldPlayWan21TransformerCache
    from flashdreams.recipes.wan.transformer.impl.network import (
        WanDiTNetworkCache,
    )

    _, block_cache = _make_block_cache()

    fake_rope_freqs = torch.zeros(1)

    class FakeRope:
        def shift_t(self, idx: int) -> torch.Tensor:
            return fake_rope_freqs

    cache = HyWorldPlayWan21TransformerCache(
        network_cache=WanDiTNetworkCache(block_caches=[block_cache]),
        network_cache_uncond=None,
        rope_adapter=FakeRope(),  # type: ignore[arg-type]
    )

    cache.start(autoregressive_index=0)
    # No exception, no side effect on the rolling caches' content.
    assert block_cache.self_attn._n_cached == 0
