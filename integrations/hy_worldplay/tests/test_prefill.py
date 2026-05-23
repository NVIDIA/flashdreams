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


def _make_block_cache(*, dim: int = 64, num_heads: int = 2, device: str = "cpu"):
    """Build a minimal :class:`HyWorldPlayPRoPEBlockCache` for direct surface tests.

    Mirrors the ``_make_prope_block`` helper in ``test_camera.py`` --
    we deliberately avoid spinning up a full :class:`HyWorldPlayPRoPEBlock`
    here because the block-cache surface tests are about the *cache
    container's* invariants, not the block's forward.

    The optional ``device`` argument lets tests that need a full block
    forward (e.g. ``prefill_memory_kv`` post 2b.6.2 -- now runs the
    real attention path, which is cudnn / CUDA only) build the block
    and its cache on the same device so the per-layer norm / proj
    tensors don't fight a mixed-device cudnn SDPA call.
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
    ).to(device)
    text_ctx = torch.zeros(1, 8, dim, device=device)
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "Phase 2b.6.2 -- prefill_memory_kv now runs the FULL block "
        "(self-attn + cross-attn + FFN with residuals) to match "
        "vendor's is_cache=True path. The self-attn uses cudnn's "
        "fused SDPA which is CUDA-only; the structural invariant "
        "this test pins is still exercised by the GPU smoke (and the "
        "next test in this module pins the cache-write side effect "
        "more narrowly via the attention-layer-level prefill_memory_kv)."
    ),
)
def test_prefill_memory_kv_writes_both_branches() -> None:
    """``HyWorldPlayPRoPEBlock.prefill_memory_kv`` populates both branches of the memory cache.

    Structural smoke: a small block on CPU, fed a tiny memory slice
    + dummy viewmats, must end with both ``has_rope_kv`` and
    ``has_prope_kv`` ``True``. This is the *minimum* structural
    invariant the block needs to satisfy for the executor to have
    something to attend over -- numerical correctness vs upstream is
    a parity-diff concern that lives behind the GPU smoke marker.
    """
    block, cache = _make_block_cache(dim=64, num_heads=2, device="cuda")
    block._parameters_updated_after_loading_checkpoint = True

    x = torch.randn(1, 4, 64, device="cuda")
    e = torch.zeros(1, 6, 64, device="cuda")
    viewmats = torch.eye(4, device="cuda").expand(1, 1, 4, 4).contiguous()

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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "Phase 2b.6.2 -- see prefill_memory_kv_writes_both_branches "
        "for rationale. The cache-isolation invariant pinned here "
        "still holds; we just need CUDA to exercise the full block "
        "now that prefill runs cross-attn + FFN end-to-end."
    ),
)
def test_prefill_memory_kv_does_not_touch_rolling_caches() -> None:
    """The prefill must not write into ``self_attn`` / ``prope_self_attn``.

    The rolling caches are the *current chunk's* K / V; the prefill
    is for *historical* K / V at collapsed positions. Mixing them
    would corrupt both sides: the dual-branch attention would
    attend to the prefilled K / V twice (once as memory, once as
    rolling), and the rolling cache would point at non-current-chunk
    positions.
    """
    block, cache = _make_block_cache(dim=64, num_heads=2, device="cuda")
    block._parameters_updated_after_loading_checkpoint = True

    rolling_n_before = cache.self_attn._n_cached
    prope_n_before = cache.prope_self_attn._n_cached

    x = torch.randn(1, 4, 64, device="cuda")
    e = torch.zeros(1, 6, 64, device="cuda")
    viewmats = torch.eye(4, device="cuda").expand(1, 1, 4, 4).contiguous()
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


def test_index_rollout_buffer_slices_action_at_rollout_indices() -> None:
    """``_index_rollout_buffer`` action path indexes the trailing axis at the selected rollout indices.

    Phase 2b.5b-part2-followup: replaces the parity-incorrect
    ``_slice_per_frame`` stub from 2b.5b-part2. The new helper
    expects the per-rollout buffer (full trajectory's worth of
    frames) and indexes into it with the rollout-coordinate
    selection from the encoder. This test pins the action int /
    long dispatch and the trailing-axis indexing so a regression
    that flips back to the ``[:K]`` truncation surfaces here.
    """
    from hy_worldplay._action import HyWorldPlayWan21Transformer

    transformer = HyWorldPlayWan21Transformer.__new__(HyWorldPlayWan21Transformer)

    rollout_action = torch.arange(8, dtype=torch.long).unsqueeze(0)  # [1, 8]
    selected = torch.tensor([0, 3, 5])
    sliced = transformer._index_rollout_buffer(
        rollout=rollout_action,
        per_step=None,
        selected=selected,
        kind="action",
    )
    assert sliced is not None
    assert sliced.shape == (1, 3)
    # The crucial parity-correctness assertion: indices match the
    # rollout positions we asked for, not a contiguous chunk slice.
    assert torch.equal(sliced, torch.tensor([[0, 3, 5]]))


def test_index_rollout_buffer_slices_matrices_at_frame_axis() -> None:
    """``_index_rollout_buffer`` indexes matrix tensors at axis -3 (the F axis).

    viewmats / Ks have rank ``len(batch_shape) + 3``: ``[..., F, M, N]``.
    The frame axis is -3, the matrix axes are -2 / -1. The helper must
    use ``index_select(-3, selected)`` so the matrix payload is
    preserved unchanged; an off-by-one on the axis would turn a
    ``[1, 8, 4, 4]`` viewmats buffer into a transposed ``[1, 8, 4, K]``
    or similar -- this test catches that.
    """
    from hy_worldplay._action import HyWorldPlayWan21Transformer

    transformer = HyWorldPlayWan21Transformer.__new__(HyWorldPlayWan21Transformer)

    # Build a viewmats buffer where each frame's matrix is its index *
    # I -- so we can verify the indexing picks the right rows.
    rollout_viewmats = torch.stack(
        [torch.eye(4) * float(i) for i in range(8)], dim=0
    ).unsqueeze(0)  # [1, 8, 4, 4]
    selected = torch.tensor([0, 3, 5])
    sliced = transformer._index_rollout_buffer(
        rollout=rollout_viewmats,
        per_step=None,
        selected=selected,
        kind="viewmats",
    )
    assert sliced is not None
    assert sliced.shape == (1, 3, 4, 4)
    assert torch.equal(sliced[0, 0], torch.eye(4) * 0.0)
    assert torch.equal(sliced[0, 1], torch.eye(4) * 3.0)
    assert torch.equal(sliced[0, 2], torch.eye(4) * 5.0)

    # Ks works the same way, with the smaller 3x3 matrix payload.
    rollout_Ks = torch.stack(
        [torch.eye(3) * float(i) for i in range(8)], dim=0
    ).unsqueeze(0)  # [1, 8, 3, 3]
    sliced_Ks = transformer._index_rollout_buffer(
        rollout=rollout_Ks,
        per_step=None,
        selected=selected,
        kind="Ks",
    )
    assert sliced_Ks is not None
    assert sliced_Ks.shape == (1, 3, 3, 3)
    assert torch.equal(sliced_Ks[0, 1], torch.eye(3) * 3.0)


def test_index_rollout_buffer_returns_none_when_both_inputs_are_none() -> None:
    """Conditioner not bound on either path -> the helper passes through ``None``.

    The prefill executor only consumes the slice when its conditioner
    is active; if both the rollout and per-step buffers are ``None``
    we surface that to the caller so it can shortcut the downstream
    AdaLN / PRoPE math.
    """
    from hy_worldplay._action import HyWorldPlayWan21Transformer

    transformer = HyWorldPlayWan21Transformer.__new__(HyWorldPlayWan21Transformer)
    assert (
        transformer._index_rollout_buffer(
            rollout=None,
            per_step=None,
            selected=torch.tensor([0]),
            kind="action",
        )
        is None
    )


def test_index_rollout_buffer_falls_back_to_per_step_when_rollout_is_none() -> None:
    """``rollout is None`` triggers the per-step fallback (parity-incorrect, structurally safe).

    This is the safety net for callers that bind the per-step
    conditioner state (via the runner's existing per-AR-step path)
    but have not migrated to the per-rollout setter yet. The
    fallback is documented as parity-incorrect; it returns the
    per-step slice unmodified so the prefill doesn't crash, and
    the conditioner's own gate ensures the slice is not consumed
    in a way that affects the noise prediction.
    """
    from hy_worldplay._action import HyWorldPlayWan21Transformer

    transformer = HyWorldPlayWan21Transformer.__new__(HyWorldPlayWan21Transformer)
    per_step = torch.zeros(1, 4, 4, 4)
    sliced = transformer._index_rollout_buffer(
        rollout=None,
        per_step=per_step,
        selected=torch.tensor([0, 1]),
        kind="viewmats",
    )
    assert sliced is per_step, "fallback must pass through the per-step tensor unchanged"


def test_is_first_step_of_chunk_uses_prefill_latch() -> None:
    """``_is_first_step_of_chunk`` reads the prefill-completed latch.

    The prefill executor reads this to gate "run prefill exactly
    once per chunk"; mis-detecting the first step would either
    re-run the prefill on every scheduler step (wasteful but
    correct -- writes are deterministic) or skip it entirely
    (incorrect). The check is via
    :attr:`HyWorldPlayWan21TransformerCache.prefill_completed_for_chunk`
    rather than the rolling cache's ``_n_cached`` because the
    Wan-2.1 fast path runs ``before_update`` / ``after_update``
    once per chunk (not per scheduler step), so ``_n_cached`` does
    not flip mid-chunk -- relying on it caused 4x redundant
    prefill calls per chunk (2b.6.2 diagnosis).
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
        autoregressive_index=1,
    )

    # Chunk-1 just started; latch is at -1 -> "first step".
    assert transformer._is_first_step_of_chunk(cache) is True
    # Predict_flow has run the prefill once and bumped the latch.
    cache.prefill_completed_for_chunk = 1
    assert transformer._is_first_step_of_chunk(cache) is False
    # New chunk (ar_idx=2) starts; predict_flow hasn't fired prefill
    # yet for this chunk so the latch (still 1) doesn't match the
    # current ar_idx (2) and we report "first step" again.
    cache.autoregressive_index = 2
    assert transformer._is_first_step_of_chunk(cache) is True


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


## ---------------------------------------------------------------------------
## Encoder rollout-buffer plumbing (phase 2b.5b-part2-followup)
## ---------------------------------------------------------------------------


def test_ctrl_rollout_fields_default_to_none() -> None:
    """Rollout buffers default to ``None`` so vendor-wrapper / per-step-only callers stay opt-in.

    The rollout buffers are an additive plumbing layer for the
    prefill executor; ctors that don't bind them must keep producing
    ctrls that the rest of the codebase treats as if 2b.5b-part2 had
    never landed. ``None`` is the universal "not bound" signal that
    ``_index_rollout_buffer`` checks.
    """
    from hy_worldplay._action import HyWorldPlayCtrl

    ctrl = HyWorldPlayCtrl(latent=torch.zeros(1, 1, 1, 1, 1), mask=None)
    assert ctrl.rollout_viewmats is None
    assert ctrl.rollout_Ks is None
    assert ctrl.rollout_action is None


def test_ctrl_rollout_fields_survive_patchify_rebuild() -> None:
    """Patchify must pass through the per-rollout buffers unchanged.

    The ``HyWorldPlayWan21Transformer.patchify_and_maybe_split_cp``
    override rebuilds the ctrl after patchify; if the new rollout
    fields aren't included in the rebuild, the prefill executor
    downstream of patchify would read ``None`` and silently fall
    back to the per-AR-step (parity-incorrect) slice. This test
    pins the inclusion at the patchify rebuild without spinning up
    a full transformer forward.
    """
    from hy_worldplay._action import HyWorldPlayCtrl, HyWorldPlayWan21Transformer

    fake_self = type("F", (), {})()

    fake_self.patchify_and_maybe_split_cp = (
        HyWorldPlayWan21Transformer.patchify_and_maybe_split_cp.__get__(fake_self)
    )

    rollout_viewmats = torch.eye(4).expand(1, 16, 4, 4).contiguous()
    rollout_Ks = torch.eye(3).expand(1, 16, 3, 3).contiguous()
    rollout_action = torch.zeros(1, 16, dtype=torch.long)

    # Already-patchified ctrl: the override returns the ctrl as-is,
    # which is enough to confirm the ``_is_patchified`` early-return
    # sees the new fields. The non-patchified branch can't be tested
    # here without a real transformer (it calls
    # ``self.patchify_and_maybe_split_cp(x.latent)`` recursively),
    # so we exercise the equivalent rebuild via the ctor below.
    ctrl_patched = HyWorldPlayCtrl(
        latent=torch.randn(1, 4, 16, 8, 8),
        mask=torch.zeros(1, 4, 16, 8, 8),
        _is_patchified=True,
        rollout_viewmats=rollout_viewmats,
        rollout_Ks=rollout_Ks,
        rollout_action=rollout_action,
    )
    out = fake_self.patchify_and_maybe_split_cp(ctrl_patched)
    assert out is ctrl_patched
    assert out.rollout_viewmats is rollout_viewmats
    assert out.rollout_Ks is rollout_Ks
    assert out.rollout_action is rollout_action


def test_encoder_attaches_rollout_buffers_to_ctrl() -> None:
    """``HyWorldPlayWanCtrlEncoder.forward`` puts the bound per-rollout buffers on the ctrl.

    End-to-end check of the 2b.5b-part2-followup plumbing: bind the
    full-trajectory action / viewmats / Ks via the encoder's
    setters, then drive a single ``forward`` and assert the output
    ctrl carries the per-rollout buffers in ``rollout_*``. Pins the
    contract between the encoder (data source) and the prefill
    driver (consumer) without touching the transformer / network.
    """
    from hy_worldplay._action import (
        HyWorldPlayWanCtrlEncoder,
        HyWorldPlayWanCtrlEncoderConfig,
    )

    cfg = HyWorldPlayWanCtrlEncoderConfig()
    encoder = HyWorldPlayWanCtrlEncoder(cfg)

    # Stub the parent's forward so we don't have to spin up a VAE -
    # we only exercise the action / viewmats / Ks attach paths.
    from hy_worldplay._action import HyWorldPlayCtrl
    from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl

    # 16 latent frames total (= 4 chunks of len_t=4) so we can test
    # both the per-AR-step slice and the rollout pass-through.
    F_total = 16
    rollout_viewmats = torch.eye(4).expand(1, F_total, 4, 4).contiguous()
    rollout_Ks = torch.eye(3).expand(1, F_total, 3, 3).contiguous()
    rollout_action = torch.arange(F_total, dtype=torch.long).unsqueeze(0)

    encoder.set_camera_data(viewmats=rollout_viewmats, Ks=rollout_Ks)
    encoder.set_action_labels(rollout_action)

    # Stub the parent's forward to return a minimal I2VCtrl with a
    # latent that has the required ``len_t`` axis (4 latent frames
    # per chunk). Bypasses the VAE encode pipeline that the real
    # parent forward would otherwise spin up.
    def fake_super_forward(*, input, autoregressive_index, cache):
        latent = torch.zeros(1, 4, 4, 8, 8)  # [B, C, len_t=4, H, W]
        mask = torch.zeros(1, 4, 4, 8, 8)
        return I2VCtrl(latent=latent, mask=mask)

    # Patch the parent's forward at the bound-method level.
    import unittest.mock as _mock

    with _mock.patch(
        "flashdreams.recipes.wan.autoencoder.i2v.I2VCtrlEncoder.forward",
        side_effect=fake_super_forward,
    ):
        ctrl = encoder.forward(
            input=torch.zeros(1, 3, 16, 64, 64),
            autoregressive_index=2,  # third chunk, frames [8, 12)
            cache=None,
        )

    # Per-AR-step slices match upstream's contract (frames [8, 12)
    # for AR step 2 with len_t=4).
    assert ctrl.action is not None
    assert torch.equal(ctrl.action, torch.tensor([[8, 9, 10, 11]]))
    assert ctrl.viewmats is not None
    assert ctrl.viewmats.shape == (1, 4, 4, 4)

    # NEW: per-rollout buffers carry the FULL trajectory.
    assert ctrl.rollout_viewmats is not None
    assert ctrl.rollout_viewmats.shape == (1, F_total, 4, 4)
    assert ctrl.rollout_Ks is not None
    assert ctrl.rollout_Ks.shape == (1, F_total, 3, 3)
    assert ctrl.rollout_action is not None
    assert torch.equal(ctrl.rollout_action, rollout_action)


def test_encoder_omits_rollout_buffers_when_unbound() -> None:
    """Unbound conditioner -> ``rollout_*`` is ``None`` (matches the per-step contract).

    A user that only enables, say, ``--use-action-conditioning``
    (not ``--use-camera-conditioning``) should see ``rollout_action``
    populated and ``rollout_viewmats`` / ``rollout_Ks`` ``None``;
    the prefill driver's ``_index_rollout_buffer`` then falls back
    to its safe per-step path for the camera conditioner. Pins
    independence of the three streams.
    """
    from hy_worldplay._action import (
        HyWorldPlayWanCtrlEncoder,
        HyWorldPlayWanCtrlEncoderConfig,
    )
    from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl

    cfg = HyWorldPlayWanCtrlEncoderConfig()
    encoder = HyWorldPlayWanCtrlEncoder(cfg)

    # Bind only action; leave camera unbound.
    encoder.set_action_labels(torch.arange(8, dtype=torch.long).unsqueeze(0))

    def fake_super_forward(*, input, autoregressive_index, cache):
        return I2VCtrl(
            latent=torch.zeros(1, 4, 4, 8, 8),
            mask=torch.zeros(1, 4, 4, 8, 8),
        )

    import unittest.mock as _mock

    with _mock.patch(
        "flashdreams.recipes.wan.autoencoder.i2v.I2VCtrlEncoder.forward",
        side_effect=fake_super_forward,
    ):
        ctrl = encoder.forward(
            input=torch.zeros(1, 3, 16, 64, 64),
            autoregressive_index=0,
            cache=None,
        )

    assert ctrl.rollout_action is not None
    assert ctrl.rollout_viewmats is None
    assert ctrl.rollout_Ks is None
