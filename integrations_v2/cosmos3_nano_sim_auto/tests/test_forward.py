# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the per-chunk layer forward.

Real weights are 15.2B parameters and live on the GB300, so these run a tiny
randomly-initialised ``_MoTLayer`` instead. That is enough for what can actually
go wrong here, which is wiring rather than arithmetic: the key stream assembled
in the wrong order, the mask not lining up with it, RoPE applied to the wrong
positions, or the chunk's keys and values coming back mangled.

The anchor is ``gen_compute``. It is the same generation-half block, already
validated for single-view generation, and it orders its keys ``[text | history | current]``
against our ``[text | current | memory]``. With nothing masked out the two must
agree exactly, because attention over a set of keys does not care what order
they arrive in. That single equality catches a surprising share of the ways this
can be got wrong.
"""

from __future__ import annotations

import pytest
import torch
from cosmos3_nano_sim_auto.impl.transformer import forward as forward_module
from cosmos3_nano_sim_auto.impl.transformer.forward import (
    chunk_layer_forward,
    run_chunk_layers,
    run_text_layers,
)
from cosmos3_nano_sim_auto.impl.transformer.rope import caption_mrope_ids
from cosmos_interactive._network import _MoTLayer
from cosmos_interactive._rope import mrope_cos_sin, text_mrope_ids
from cosmos_interactive._spec import LLMSpec

from flashdreams.accelerated.multi_head_attention.kv_arena import allocate_kv_arena

pytestmark = pytest.mark.ci_cpu

# Small enough to be instant, but keeps grouped-query attention (4 query heads
# over 2 key/value heads) so the head expansion is genuinely exercised.
TINY = LLMSpec(
    hidden_size=32,
    num_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    intermediate_size=16,
    vocab_size=128,
    mrope_section=(2, 1, 1),
)
BATCH, CHUNK, TEXT, MEMORY = 1, 3, 2, 5
DTYPE = torch.float32


def build(seed: int = 0):
    torch.manual_seed(seed)
    layer = _MoTLayer(TINY, DTYPE).eval()
    h = torch.randn(BATCH, CHUNK, TINY.hidden_size)
    position_ids = torch.rand(3, CHUNK) * 10
    cos, sin = mrope_cos_sin(
        position_ids, TINY.head_dim, TINY.rope_theta, TINY.mrope_section
    )
    return layer, h, position_ids, cos[None], sin[None]


def cache(length: int, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    shape = (BATCH, TINY.num_key_value_heads, length, TINY.head_dim)
    return (
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
    )


def all_visible() -> torch.Tensor:
    return torch.ones(CHUNK, TEXT + CHUNK + MEMORY, dtype=torch.bool)


@torch.no_grad()
def test_matches_gen_compute_when_nothing_is_masked() -> None:
    """Same block, same keys, different order -- attention cannot tell."""
    layer, h, _, cos, sin = build()
    text_kv, memory_kv = cache(TEXT), cache(MEMORY, seed=2)

    ours, _k, _v = chunk_layer_forward(
        layer, h, cos, sin, text_kv=text_kv, memory_kv=memory_kv, mask=all_visible()
    )
    theirs = layer.gen_compute(
        h,
        cos,
        sin,
        text_kv[0],
        text_kv[1],
        memory_kv[0],
        memory_kv[1],
        within_mask=None,
    )

    assert (ours - theirs).abs().max().item() < 1e-5


@torch.no_grad()
def test_accelerated_rms_norm_reaches_every_generation_norm(monkeypatch) -> None:
    layer, h, position_ids, _cos, _sin = build()
    calls: list[int] = []

    def accelerated(input, weight, eps):
        calls.append(weight.shape[0])
        return torch.nn.functional.rms_norm(input, (weight.shape[0],), weight, eps)

    monkeypatch.setattr(forward_module, "rms_norm", accelerated)
    run_chunk_layers(
        [layer],
        TINY,
        h,
        position_ids,
        text_kv=[cache(TEXT)],
        memory_kv=[cache(MEMORY, seed=2)],
        mask=all_visible(),
        final_norm=torch.nn.RMSNorm(TINY.hidden_size),
        accelerated_rms_norm=True,
    )

    assert calls == [
        TINY.hidden_size,
        TINY.head_dim,
        TINY.head_dim,
        TINY.hidden_size,
        TINY.hidden_size,
    ]


@torch.no_grad()
def test_attention_arena_is_bitwise_equal_to_concatenation() -> None:
    layer, h, _, cos, sin = build()
    text_kv, memory_kv = cache(TEXT), cache(MEMORY, seed=2)
    arena, layout = allocate_kv_arena(
        text_kv,
        memory_kv,
        current_tokens=CHUNK,
        memory_tokens=MEMORY,
    )
    arena_text = (arena[0][:, :, :TEXT], arena[1][:, :, :TEXT])
    arena_memory = (
        arena[0][:, :, layout.memory_start :],
        arena[1][:, :, layout.memory_start :],
    )

    reference = chunk_layer_forward(
        layer, h, cos, sin, text_kv=text_kv, memory_kv=memory_kv, mask=all_visible()
    )
    actual = chunk_layer_forward(
        layer,
        h,
        cos,
        sin,
        text_kv=arena_text,
        memory_kv=arena_memory,
        mask=all_visible(),
        attention_arena=arena,
        attention_arena_layout=layout,
    )

    assert all(
        torch.equal(got, want) for got, want in zip(actual, reference, strict=True)
    )


@torch.no_grad()
def test_attention_arena_rejects_same_shaped_stale_segments() -> None:
    layer, h, _, cos, sin = build()
    text_kv, memory_kv = cache(TEXT), cache(MEMORY, seed=2)
    arena, layout = allocate_kv_arena(
        cache(TEXT, seed=8),
        cache(MEMORY, seed=9),
        current_tokens=CHUNK,
        memory_tokens=MEMORY,
    )

    with pytest.raises(ValueError, match="must alias"):
        chunk_layer_forward(
            layer,
            h,
            cos,
            sin,
            text_kv=text_kv,
            memory_kv=memory_kv,
            mask=all_visible(),
            attention_arena=arena,
            attention_arena_layout=layout,
        )


REGIONS = {
    "text": (0, TEXT),
    "chunk": (TEXT, TEXT + CHUNK),
    "memory": (TEXT + CHUNK, TEXT + CHUNK + MEMORY),
}


@torch.no_grad()
@pytest.mark.parametrize("region", list(REGIONS))
def test_mask_columns_address_the_part_of_the_key_stream_they_name(region: str) -> None:
    """Open one region of the mask at a time and see which cache reaches the output.

    The mask is built by the packer against a promised key order, and assembled
    here. Nothing checks that the two agree -- a mask off by the text length still
    has the right shape and still runs. So open exactly one region and confirm the
    output moves only when the tensor that region is supposed to name moves.
    """
    layer, h, _, cos, sin = build()
    text_kv, memory_kv = cache(TEXT), cache(MEMORY, seed=2)
    start, stop = REGIONS[region]
    mask = torch.zeros(CHUNK, TEXT + CHUNK + MEMORY, dtype=torch.bool)
    mask[:, start:stop] = True

    def run(text, memory):
        out, _k, _v = chunk_layer_forward(
            layer, h, cos, sin, text_kv=text, memory_kv=memory, mask=mask
        )
        return out

    base = run(text_kv, memory_kv)
    text_reaches = not torch.allclose(
        base, run(cache(TEXT, seed=98), memory_kv), atol=1e-6
    )
    memory_reaches = not torch.allclose(
        base, run(text_kv, cache(MEMORY, seed=99)), atol=1e-6
    )

    assert text_reaches == (region == "text")
    assert memory_reaches == (region == "memory")


@torch.no_grad()
def test_masked_out_memory_cannot_reach_the_output() -> None:
    layer, h, _, cos, sin = build()
    text_kv, memory_kv = cache(TEXT), cache(MEMORY, seed=2)
    mask = all_visible()
    mask[:, TEXT + CHUNK :] = False

    with_memory, _k, _v = chunk_layer_forward(
        layer, h, cos, sin, text_kv=text_kv, memory_kv=memory_kv, mask=mask
    )
    empty = torch.zeros(BATCH, TINY.num_key_value_heads, 0, TINY.head_dim)
    without, _k2, _v2 = chunk_layer_forward(
        layer,
        h,
        cos,
        sin,
        text_kv=text_kv,
        memory_kv=(empty, empty),
        mask=mask[:, : TEXT + CHUNK],
    )

    assert (with_memory - without).abs().max().item() < 1e-6


@torch.no_grad()
def test_returned_kv_is_the_chunk_carrying_its_own_rope() -> None:
    """What comes back is what the caller commits to memory once the chunk is clean."""
    layer, h, _, cos, sin = build()
    text_kv, memory_kv = cache(TEXT), cache(MEMORY, seed=2)

    _h, k, v = chunk_layer_forward(
        layer, h, cos, sin, text_kv=text_kv, memory_kv=memory_kv, mask=all_visible()
    )

    assert k.shape == (BATCH, TINY.num_key_value_heads, CHUNK, TINY.head_dim)
    assert v.shape == k.shape
    # RoPE touches keys and not values, so the two must have diverged.
    assert not torch.allclose(k, v)
    # Different positions, different keys -- the chunk is not being placed at the origin.
    other_cos, other_sin = mrope_cos_sin(
        torch.rand(3, CHUNK) * 10 + 50,
        TINY.head_dim,
        TINY.rope_theta,
        TINY.mrope_section,
    )
    _h2, k2, v2 = chunk_layer_forward(
        layer,
        h,
        other_cos[None],
        other_sin[None],
        text_kv=text_kv,
        memory_kv=memory_kv,
        mask=all_visible(),
    )
    assert not torch.allclose(k, k2)
    assert torch.allclose(v, v2)


@torch.no_grad()
def test_run_chunk_layers_threads_one_cache_entry_per_layer() -> None:
    layers = [_MoTLayer(TINY, DTYPE).eval() for _ in range(TINY.num_layers)]
    _layer, h, position_ids, _cos, _sin = build()
    text_kv = [cache(TEXT, seed=10 + i) for i in range(TINY.num_layers)]
    memory_kv = [cache(MEMORY, seed=20 + i) for i in range(TINY.num_layers)]

    out, chunk_kv = run_chunk_layers(
        layers,
        TINY,
        h,
        position_ids,
        text_kv=text_kv,
        memory_kv=memory_kv,
        mask=all_visible(),
    )

    assert out.shape == h.shape
    assert not torch.allclose(out, h)
    assert len(chunk_kv) == TINY.num_layers
    expected = (BATCH, TINY.num_key_value_heads, CHUNK, TINY.head_dim)
    assert all(k.shape == expected for k, _ in chunk_kv)
    # Each layer sees a different cache, so no two layers can emit the same keys.
    assert not torch.allclose(chunk_kv[0][0], chunk_kv[1][0])


@torch.no_grad()
def test_run_chunk_layers_can_avoid_retaining_uncommitted_kv() -> None:
    layers = [_MoTLayer(TINY, DTYPE).eval()]
    _layer, h, position_ids, _cos, _sin = build()

    out, chunk_kv = run_chunk_layers(
        layers,
        TINY,
        h,
        position_ids,
        text_kv=[cache(TEXT)],
        memory_kv=[cache(MEMORY, seed=2)],
        mask=all_visible(),
        collect_kv=False,
    )

    assert out.shape == h.shape
    assert chunk_kv == []


@torch.no_grad()
def test_run_chunk_layers_applies_the_generation_final_norm() -> None:
    layers = [_MoTLayer(TINY, DTYPE).eval()]
    _layer, h, position_ids, _cos, _sin = build()
    kv = [cache(TEXT)], [cache(MEMORY, seed=2)]
    norm = torch.nn.LayerNorm(TINY.hidden_size, dtype=DTYPE)

    plain, _ = run_chunk_layers(
        layers,
        TINY,
        h,
        position_ids,
        text_kv=kv[0],
        memory_kv=kv[1],
        mask=all_visible(),
    )
    normed, _ = run_chunk_layers(
        layers,
        TINY,
        h,
        position_ids,
        text_kv=kv[0],
        memory_kv=kv[1],
        mask=all_visible(),
        final_norm=norm,
    )

    assert torch.allclose(normed, norm(plain))


def test_run_chunk_layers_rejects_a_cache_that_does_not_cover_every_layer() -> None:
    layers = [_MoTLayer(TINY, DTYPE).eval() for _ in range(2)]
    _layer, h, position_ids, _cos, _sin = build()

    with pytest.raises(ValueError, match="one cache entry per layer"):
        run_chunk_layers(
            layers,
            TINY,
            h,
            position_ids,
            text_kv=[cache(TEXT)],
            memory_kv=[cache(MEMORY), cache(MEMORY)],
            mask=all_visible(),
        )


@torch.no_grad()
def test_each_camera_s_caption_is_its_own_document() -> None:
    """A caption cannot read another camera's, so it reads the same either way.

    Run one camera's caption on its own and then again with a second camera's
    appended: the first camera's keys and values have to come out identical.
    No caption was written knowing what another says, and the reference implementation keeps them apart
    the same way, as separate documents inside one causal split.
    """
    torch.manual_seed(3)
    layers = [_MoTLayer(TINY, DTYPE).eval() for _ in range(2)]
    first, second = 3, 4
    h = torch.randn(1, first + second, TINY.hidden_size)
    positions, _used = caption_mrope_ids([first, second])

    _out, alone = run_text_layers(
        layers, TINY, h[:, :first], positions[:, :first], view_text_tokens=[first]
    )
    _out, both = run_text_layers(
        layers, TINY, h, positions, view_text_tokens=[first, second]
    )

    for (k_alone, v_alone), (k_both, v_both) in zip(alone, both, strict=True):
        assert torch.allclose(k_alone, k_both[:, :, :first], atol=1e-6)
        assert torch.allclose(v_alone, v_both[:, :, :first], atol=1e-6)


@torch.no_grad()
def test_a_rig_level_prompt_stays_one_document() -> None:
    """Otherwise the older layout's prompt would lose its own history."""
    torch.manual_seed(4)
    layers = [_MoTLayer(TINY, DTYPE).eval()]
    h = torch.randn(1, 5, TINY.hidden_size)
    positions, _next = text_mrope_ids(5, 0.0)

    merged, _kv = run_text_layers(layers, TINY, h, positions)
    split, _kv = run_text_layers(layers, TINY, h, positions, view_text_tokens=[2, 3])

    # The third token is the first that can reach across the split, so it is
    # the first whose output can differ; the two before it read the same keys.
    assert torch.allclose(merged[:, :2], split[:, :2], atol=1e-6)
    assert not torch.allclose(merged[:, 2], split[:, 2], atol=1e-4)
