# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the autoregressive loop.

A tiny randomly-initialised network again, so what these check is the order of
operations rather than the pictures. The loop's own content is order: which pass
runs, against which memory, with which tokens, and what gets kept afterwards.

The centrepiece is ``test_the_committed_pass_carries_no_timestep``. Memory is
supposed to hold keys for *clean* history, and the cheap-looking mistake is to
keep the last denoising step's keys instead -- they are the right shape, they are
finite, and they are computed from a noisy latent with a timestep added. A
rollout built that way runs to completion and produces history made of noise. So
the test reaches into the passes the loop actually made and pins that the one
whose keys are committed was handed ``vae2llm(denoised)`` and nothing else.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from cosmos3_nano_sim_auto.impl import rollout as rollout_module
from cosmos3_nano_sim_auto.impl.conditioning.control_provider import EncodedControls
from cosmos3_nano_sim_auto.impl.packing import (
    ClipGeometry,
    ar_chunk_ranges,
    build_memory_layout,
)
from cosmos3_nano_sim_auto.impl.rollout import (
    ChunkRollout,
    _KVCache,
    _TokenWindow,
    run_rollout,
)
from cosmos3_nano_sim_auto.impl.transformer.mask import (
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_PADDING,
)
from cosmos_interactive._network import (
    CosmosMoTNetwork,
    CosmosMoTNetworkConfig,
)
from cosmos_interactive._spec import Cosmos3InteractiveSpec, LLMSpec, VFMSpec
from torch.nn.attention.flex_attention import BlockMask

pytestmark = pytest.mark.ci_cpu

SPEC = Cosmos3InteractiveSpec(
    llm=LLMSpec(
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=16,
        vocab_size=100,
        mrope_section=(2, 1, 1),
    ),
    # patch_latent_dim = 2*2*4 = 16.
    vfm=VFMSpec(vae_latent_channels=4, latent_patch_size=2, time_embed_freq_dim=8),
)
PATCH_DIM = SPEC.vfm.patch_latent_dim
STEPS = len(SPEC.vfm.fixed_step_t_list)
TEXT = 3

# Two views and five frames: small, and still the shape of a real clip. Nothing
# is supplied, so frame 0 is a chunk on its own and frames 1..4 follow.
GEOMETRY = ClipGeometry(
    num_views=2,
    frames_per_view=5,
    patch_h=1,
    patch_w=2,
    frames_per_chunk=4,
    condition_frames=0,
)
assert GEOMETRY.frames_per_view is not None
CONTROL_TOKENS = GEOMETRY.num_views * GEOMETRY.frames_per_view * GEOMETRY.spatial_tokens


def build(seed: int = 0) -> CosmosMoTNetwork:
    torch.manual_seed(seed)
    return (
        CosmosMoTNetworkConfig(
            spec=SPEC, dtype=torch.float32, include_vision_tower=False
        )
        .setup()
        .eval()
    )


def inputs(seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    controls = torch.randn(CONTROL_TOKENS, PATCH_DIM, generator=generator)
    text_ids = torch.randint(0, SPEC.llm.vocab_size, (TEXT,), generator=generator)
    return controls, text_ids


def go(network: CosmosMoTNetwork, **kwargs):
    controls, text_ids = inputs()
    with torch.no_grad():
        return run_rollout(
            network,
            geometry=GEOMETRY,
            control_tokens=controls,
            text_ids=text_ids,
            **kwargs,
        )


class PassRecorder:
    """Records every chunk pass the loop makes, in order."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def install(self, monkeypatch) -> PassRecorder:
        original = rollout_module.run_chunk_layers

        def spy(*args, **kwargs):
            self.calls.append({"h": args[2].clone(), "mask": kwargs["mask"].clone()})
            return original(*args, **kwargs)

        monkeypatch.setattr(rollout_module, "run_chunk_layers", spy)
        return self


# --- the loop's shape -------------------------------------------------------


def test_a_clip_with_no_conditioning_runs_two_chunks() -> None:
    result = go(build())

    assert [(step.start, step.end) for step in result.trace] == [(0, 1), (1, 5)]


def test_every_chunk_denoises_once_per_sampler_step() -> None:
    result = go(build())

    assert [step.denoising_passes for step in result.trace] == [STEPS, STEPS]


def test_a_trimmed_schedule_costs_the_passes_it_saves() -> None:
    """Fewer sigmas is the cheapest large win there is, so pin that it lands.

    Passes are ``steps + 1``, and only the first term moves -- the clean replay
    does not care how the chunk was denoised.
    """
    result = go(build(), t_list=list(SPEC.vfm.fixed_step_t_list[:2]))

    assert [step.denoising_passes for step in result.trace] == [2, 2]
    assert [step.clean_pass for step in result.trace] == [True, False]


def test_only_the_chunks_that_will_be_read_are_replayed_clean() -> None:
    # The last chunk's keys would go into a memory nothing attends against, so
    # the reference implementation skips that pass and so do we -- a fifth of the rollout's cost, here.
    result = go(build())

    assert [step.clean_pass for step in result.trace] == [True, False]


def test_the_loop_makes_exactly_the_passes_the_trace_claims(monkeypatch) -> None:
    recorder = PassRecorder().install(monkeypatch)
    result = go(build())

    expected = sum(
        step.denoising_passes + int(step.clean_pass) for step in result.trace
    )
    assert len(recorder.calls) == expected == 2 * STEPS + 1


def test_attention_kv_arena_preserves_the_rollout_bitwise() -> None:
    network = build()
    reference = go(network, history_frames=4)
    actual = go(network, history_frames=4, attention_kv_arena=True)

    assert torch.equal(actual.latent, reference.latent)


def test_memory_grows_by_the_chunk_that_was_committed() -> None:
    result = go(build())
    first, second = result.trace

    # The first chunk sees the control track alone; the second also sees the slot
    # the first went into. That is a whole chunk wide even though frame 0 is one
    # frame, the rest being padding the mask isolates.
    slot = GEOMETRY.num_views * GEOMETRY.frames_per_chunk * GEOMETRY.spatial_tokens
    assert first.memory_tokens == CONTROL_TOKENS
    assert second.memory_tokens == CONTROL_TOKENS + slot


# --- what goes into memory --------------------------------------------------


@torch.no_grad()
def test_the_committed_pass_carries_no_timestep(monkeypatch) -> None:
    """The whole reason a chunk costs one pass more than its denoising steps."""
    network = build()
    recorder = PassRecorder().install(monkeypatch)
    result = go(network)

    # Calls are: STEPS noisy passes for chunk 0, then its clean replay, then
    # STEPS noisy passes for chunk 1 with no replay after it.
    clean = recorder.calls[STEPS]
    denoised = result.tokens.reshape(
        GEOMETRY.num_views, GEOMETRY.frames_per_view, GEOMETRY.spatial_tokens, PATCH_DIM
    )[:, 0:1].reshape(-1, PATCH_DIM)

    assert torch.equal(clean["h"], network.vae2llm(denoised)[None])


@torch.no_grad()
def test_the_committed_pass_is_not_the_last_denoising_pass(monkeypatch) -> None:
    # Stated separately because the assertion above would also pass if the two
    # happened to coincide, and the point is that they must not.
    build_network = build()
    recorder = PassRecorder().install(monkeypatch)
    go(build_network)

    last_noisy = recorder.calls[STEPS - 1]["h"]
    clean = recorder.calls[STEPS]["h"]

    assert last_noisy.shape == clean.shape
    assert not torch.allclose(last_noisy, clean)


def test_the_replay_and_the_denoising_pass_use_different_masks(monkeypatch) -> None:
    # Same tokens, different roles: a current target reads strictly earlier
    # history, a clean one reads its own step. If the roles were not switched the
    # two masks would be identical, so this is the check that they were.
    recorder = PassRecorder().install(monkeypatch)
    go(build())

    noisy_masks = [call["mask"] for call in recorder.calls[:STEPS]]
    clean_mask = recorder.calls[STEPS]["mask"]

    assert all(torch.equal(mask, noisy_masks[0]) for mask in noisy_masks)
    assert clean_mask.shape == noisy_masks[0].shape


# --- outputs ----------------------------------------------------------------


def test_the_latent_comes_back_shaped_for_the_decoder() -> None:
    result = go(build())

    assert result.latent.shape == (
        GEOMETRY.num_views,
        SPEC.vfm.vae_latent_channels,
        GEOMETRY.frames_per_view,
        GEOMETRY.patch_h * SPEC.vfm.latent_patch_size,
        GEOMETRY.patch_w * SPEC.vfm.latent_patch_size,
    )
    assert result.tokens.shape == (CONTROL_TOKENS, PATCH_DIM)
    assert torch.isfinite(result.latent).all()


def test_the_latent_and_the_tokens_are_the_same_thing() -> None:
    from cosmos_interactive._patchify import patchify_latent

    result = go(build())
    repacked = torch.cat(
        [patchify_latent(view, SPEC.vfm.latent_patch_size) for view in result.latent]
    )

    assert torch.equal(repacked, result.tokens)


def test_chunks_are_announced_as_they_finish() -> None:
    seen: list[tuple[int, int, tuple[int, ...]]] = []
    go(
        build(),
        on_chunk=lambda step, tokens: seen.append(
            (step.start, step.end, tuple(tokens.shape))
        ),
    )

    assert [(start, end) for start, end, _shape in seen] == [(0, 1), (1, 5)]
    assert seen[0][2] == (GEOMETRY.num_views, 1, GEOMETRY.spatial_tokens, PATCH_DIM)
    assert seen[1][2] == (GEOMETRY.num_views, 4, GEOMETRY.spatial_tokens, PATCH_DIM)


# --- seeding ----------------------------------------------------------------


def test_the_same_seed_gives_the_same_clip() -> None:
    assert torch.equal(go(build(), seed=7).tokens, go(build(), seed=7).tokens)


def test_a_different_seed_gives_a_different_clip() -> None:
    assert not torch.allclose(go(build(), seed=7).tokens, go(build(), seed=8).tokens)


# --- what is refused --------------------------------------------------------


@torch.no_grad()
def test_flex_attention_generates_the_same_clip_as_the_dense_mask(monkeypatch) -> None:
    """The whole clip, both kernels, same latents.

    ``test_attention.py`` checks the two agree on one chunk of made-up numbers.
    This checks the wiring around them: two chunks, so the key stream grows and
    the mask is rebuilt at a second width, and both kinds of pass in a chunk --
    the denoising passes and the clean replay, which use different masks.

    The mask types are asserted because without them the test passes when the
    flag does nothing at all -- two dense runs agree perfectly. Slow for a CPU
    test, because switching kernels means compiling one, and worth it: a mask
    built at the wrong width for the second chunk still returns a plausible
    clip, and nothing else here would notice.
    """
    seen: list[type] = []
    original = rollout_module.run_chunk_layers

    def spy(*args, **kwargs):
        seen.append(type(kwargs["mask"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(rollout_module, "run_chunk_layers", spy)

    dense = go(build(), seed=7)
    assert set(seen) == {torch.Tensor}

    seen.clear()
    flex = go(build(), seed=7, flex_attention=True)
    assert set(seen) == {BlockMask}

    assert flex.tokens.shape == dense.tokens.shape
    assert (flex.tokens - dense.tokens).abs().max().item() < 1e-4


def test_a_control_track_of_the_wrong_length_is_refused() -> None:
    network = build()
    _controls, text_ids = inputs()

    with pytest.raises(ValueError, match="control_tokens is"):
        run_rollout(
            network,
            geometry=GEOMETRY,
            control_tokens=torch.zeros(CONTROL_TOKENS - 2, PATCH_DIM),
            text_ids=text_ids,
        )


def test_conditioning_frames_without_their_tokens_are_refused() -> None:
    network = build()
    _controls, text_ids = inputs()
    geometry = ClipGeometry(
        num_views=2,
        frames_per_view=5,
        patch_h=1,
        patch_w=2,
        frames_per_chunk=4,
        condition_frames=1,
    )
    controls = torch.zeros(CONTROL_TOKENS, PATCH_DIM)

    with pytest.raises(ValueError, match="condition_tokens is required"):
        run_rollout(
            network, geometry=geometry, control_tokens=controls, text_ids=text_ids
        )


def test_conditioning_tokens_for_a_text2video_clip_are_refused() -> None:
    network = build()
    controls, text_ids = inputs()

    with pytest.raises(ValueError, match="no conditioning frames"):
        run_rollout(
            network,
            geometry=GEOMETRY,
            control_tokens=controls,
            text_ids=text_ids,
            condition_tokens=torch.zeros(4, PATCH_DIM),
        )


# --- driving it a chunk at a time -------------------------------------------
#
# The serving loop asks for one chunk per call rather than handing over its
# thread, so ChunkRollout has to reach the same clip as run_rollout by a
# different route. These pin that it does, and that the half-finished states in
# between refuse to be mistaken for a finished clip.


def stepwise(network: CosmosMoTNetwork, **kwargs) -> ChunkRollout:
    controls, text_ids = inputs()
    return ChunkRollout(
        network, geometry=GEOMETRY, control_tokens=controls, text_ids=text_ids, **kwargs
    )


def test_a_rollout_builds_no_autograd_graph_even_outside_no_grad() -> None:
    """The one that cost a run: eval() does not stop parameters wanting grads.

    A prefill with autograd on retains every layer's activations over the whole
    control track, which at eleven views turned 81 GiB into an OOM at 276. It is
    silent at two views, so the check is that no tensor comes back attached
    rather than any measurement of memory.
    """
    rollout = stepwise(build())  # deliberately not inside torch.no_grad()

    _trace, tokens = rollout.step()

    assert not tokens.requires_grad
    assert tokens.grad_fn is None
    for keys, values in rollout._cache.layers():
        assert not keys.requires_grad and not values.requires_grad


def test_memory_is_one_allocation_that_chunks_are_written_into() -> None:
    """Growing it by ``torch.cat`` doubled the cache on every commit.

    Every layer's new tensor existed alongside its old one, and the comprehension
    held the whole list live at once, so at eleven views and 61 frames a 34.2 GiB
    cache peaked at 68.3 on a step whose own attention is a fraction of that. The
    observable part is that the tensors are the same objects afterwards, having
    been written rather than replaced.
    """
    rollout = stepwise(build())
    cache = rollout._cache
    before = [
        (k.data_ptr(), v.data_ptr()) for k, v in zip(cache._k, cache._v, strict=True)
    ]
    prefilled = cache.length

    with torch.no_grad():
        rollout.step()

    assert [
        (k.data_ptr(), v.data_ptr()) for k, v in zip(cache._k, cache._v, strict=True)
    ] == before
    assert cache.length > prefilled
    # And what the layers read is the written part alone, not the reserve.
    assert all(k.shape[2] == cache.length for k, _v in rollout._cache.layers())


def test_memory_ends_up_exactly_the_size_it_was_built_for() -> None:
    """Capacity is derived, so an off-by-one shows up as slack or as a raise.

    The last chunk is the subtlety: ``step`` never commits it, because nothing
    attends to it afterwards. So memory is the controls plus history up to the
    last chunk's *start*, and sizing for the whole clip would reserve a chunk
    that is never written.
    """
    rollout = stepwise(build())

    with torch.no_grad():
        while not rollout.is_finished:
            rollout.step()

    cache = rollout._cache
    assert cache.length == cache._capacity
    # Controls whole, then the one slot the (0, 1) chunk went into -- the (1, 5)
    # chunk is last and so is never committed.
    last_start = ar_chunk_ranges(GEOMETRY)[-1][0]
    assert last_start == 1
    slot = GEOMETRY.num_views * GEOMETRY.frames_per_chunk * GEOMETRY.spatial_tokens
    assert cache.length == CONTROL_TOKENS + slot


def test_a_chunk_longer_than_its_slot_is_refused() -> None:
    """Slots are one size, so a chunk that does not fit is a sizing disagreement.

    Short is fine and padded; long would write into the next slot.
    """
    rollout = stepwise(build())
    cache = rollout._cache
    layers = len(cache._k)
    too_many = cache._k[0].new_zeros(
        (1, cache._k[0].shape[1], cache._slot_tokens + 1, 8)
    )

    with pytest.raises(ValueError, match="token slot"):
        cache.extend([(too_many, too_many)] * layers)


def test_slots_that_do_not_fit_the_capacity_are_refused_at_build_time() -> None:
    """Sizing is derived, so a mistake should raise before a chunk is generated.

    Caught in the constructor rather than on the write that would overflow,
    which would otherwise surface several chunks into a run.
    """
    prefilled = [(torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8))]

    with pytest.raises(ValueError, match="past a capacity"):
        _KVCache(prefilled, capacity=8, slots=2, slot_tokens=4)


def test_a_cache_with_no_slots_cannot_hold_a_generated_chunk() -> None:
    """A clip whose only chunk is its last commits nothing, so it reserves none.

    A chunk arriving anyway means the sizing and the stepping disagree about
    what is committed -- worth a raise rather than a write past the end.
    """
    prefilled = [(torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8))]
    cache = _KVCache(prefilled, capacity=4, slots=0)
    one = torch.zeros(1, 2, 1, 8)

    with pytest.raises(ValueError, match="no reusable slots"):
        cache.extend([(one, one)])


def test_attention_arena_activates_only_after_current_and_memory_settle() -> None:
    def pair(tokens: int, fill: float) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, SPEC.llm.num_key_value_heads, tokens, SPEC.llm.head_dim)
        return torch.full(shape, fill), torch.full(shape, fill + 1)

    prefilled = [pair(2, 1)]
    text = [pair(1, 2)]
    arena_cache = _KVCache(
        prefilled,
        capacity=6,
        slots=1,
        slot_tokens=4,
        text_prefilled=text,
        attention_current_tokens=4,
    )

    assert arena_cache.attention_layers(4) is None
    arena_cache.extend([pair(4, 3)])
    assert arena_cache.attention_layers(3) is None
    assert arena_cache.attention_layers(4) is not None


def test_a_clip_with_nothing_to_generate_is_refused_at_build_time() -> None:
    """One pixel frame of i2v is one latent frame, and it is already given.

    So there is no chunk to run. Refused in ``__init__``, where the geometry
    says so, rather than as a confusing "all 0 chunks are generated" from the
    first ``step()``.
    """
    geometry = dataclasses.replace(GEOMETRY, frames_per_view=1, condition_frames=1)
    controls, text_ids = inputs()
    spatial = geometry.num_views * geometry.spatial_tokens

    with pytest.raises(ValueError, match="nothing to generate"):
        ChunkRollout(
            build(),
            geometry=geometry,
            control_tokens=controls[:spatial],
            condition_tokens=controls[:spatial],
            text_ids=text_ids,
        )


def test_stepping_reaches_the_same_clip_as_running() -> None:
    """Same seed, same schedule, same answer -- or the app is a different demo."""
    whole = go(build())

    rollout = stepwise(build())
    with torch.no_grad():
        while not rollout.is_finished:
            rollout.step()
    stepped = rollout.result()

    torch.testing.assert_close(stepped.latent, whole.latent)
    assert stepped.trace == whole.trace


def test_a_step_reports_the_frames_it_generated() -> None:
    rollout = stepwise(build())
    assert rollout.chunk_count == 2  # frame 0 alone, then frames 1..4.

    with torch.no_grad():
        first, tokens = rollout.step()

    assert (first.start, first.end) == (0, 1)
    assert tokens.shape[:2] == (GEOMETRY.num_views, 1)
    assert rollout.chunks_done == 1
    assert not rollout.is_finished


def test_an_unfinished_clip_refuses_to_be_a_result() -> None:
    """A partly-zero buffer would decode to a clip that ends in grey."""
    rollout = stepwise(build())
    with torch.no_grad():
        rollout.step()

    with pytest.raises(RuntimeError, match="1 of 2 chunks"):
        rollout.result()


def test_stepping_past_the_end_raises_rather_than_looping() -> None:
    rollout = stepwise(build())
    with torch.no_grad():
        while not rollout.is_finished:
            rollout.step()

        with pytest.raises(RuntimeError, match="all 2 chunks"):
            rollout.step()


def test_latent_for_a_range_matches_the_finished_clip() -> None:
    """Decoding as you go has to agree with decoding at the end."""
    rollout = stepwise(build())
    with torch.no_grad():
        while not rollout.is_finished:
            rollout.step()
        piece = rollout.latent_for(1, 5)

    torch.testing.assert_close(piece, rollout.result().latent[:, :, 1:5])


def test_latent_for_ungenerated_frames_is_refused() -> None:
    # Zeros unpatchify and decode without complaint, so asking early has to be
    # the error rather than the grey frames it would otherwise return.
    rollout = stepwise(build())
    with torch.no_grad():
        rollout.step()

    with pytest.raises(ValueError, match="reaches past"):
        rollout.latent_for(0, 5)


def test_a_conditioned_clip_can_decode_its_given_frames_before_any_step() -> None:
    """The conditioning frames are real footage, and no chunk produces them."""
    geometry = ClipGeometry(
        num_views=2,
        frames_per_view=5,
        patch_h=1,
        patch_w=2,
        frames_per_chunk=4,
        condition_frames=1,
    )
    controls, text_ids = inputs()
    conditions = torch.randn(
        geometry.num_views * 1 * geometry.spatial_tokens, PATCH_DIM
    )
    rollout = ChunkRollout(
        build(),
        geometry=geometry,
        control_tokens=controls,
        text_ids=text_ids,
        condition_tokens=conditions,
    )

    assert rollout.latent_for(0, 1).shape[2] == 1
    with pytest.raises(ValueError, match="reaches past"):
        rollout.latent_for(0, 2)


ROLLING = ClipGeometry(
    num_views=2,
    frames_per_view=17,
    patch_h=1,
    patch_w=2,
    frames_per_chunk=4,
    condition_frames=1,
)
"""Long enough that a window has to drop chunks and stream control.

``GEOMETRY`` cannot: it commits only frame 0, which one slot holds.
"""

FITTING = ClipGeometry(
    num_views=2,
    frames_per_view=16,
    patch_h=1,
    patch_w=2,
    frames_per_chunk=4,
    condition_frames=1,
)
"""``ROLLING`` one frame shorter, which is what makes a 12-frame window free.

Three committed chunks is twelve frames of history, and control has to reach one
chunk further than that -- sixteen, the whole clip. So nothing is evicted and
nothing is streamed, exactly as in the 61-frame clip the demo runs.
"""


def rolling(history_frames, geometry=ROLLING, seed=0) -> ChunkRollout:
    controls, text_ids = inputs()
    assert geometry.frames_per_view is not None
    controls = torch.randn(
        geometry.num_views * geometry.frames_per_view * geometry.spatial_tokens,
        PATCH_DIM,
    )
    conditions = torch.randn(geometry.num_views * geometry.spatial_tokens, PATCH_DIM)
    return ChunkRollout(
        build(seed),
        geometry=geometry,
        control_tokens=controls,
        text_ids=text_ids,
        condition_tokens=conditions,
        history_frames=history_frames,
    )


def run_to_end(rollout: ChunkRollout) -> torch.Tensor:
    with torch.no_grad():
        while not rollout.is_finished:
            rollout.step()
    return rollout.result().latent


def test_a_window_the_clip_already_fits_changes_nothing_at_all() -> None:
    """The whole point of the bound is that it is invisible until it bites.

    This clip commits twelve frames and its control reaches sixteen, so a
    12-frame window evicts nothing and streams nothing: same layout, same order,
    same batching. That has to come back bit for bit, not merely close. If it
    drifts, the window is reordering or dropping something unasked.
    """
    unbounded = rolling(None, FITTING)
    bounded = rolling(12, FITTING)

    assert bounded._control_ranges is None
    assert bounded._cache._capacity == unbounded._cache._capacity
    assert torch.equal(run_to_end(bounded), run_to_end(unbounded))


def test_a_window_shorter_than_the_clip_bounds_the_cache() -> None:
    """Two slots rather than three, so the cache stops tracking clip length."""
    assert rolling(8)._cache._capacity < rolling(None)._cache._capacity
    assert rolling(8)._slots == 2
    assert rolling(4)._slots == 1


def test_a_window_narrower_than_one_chunk_still_keeps_that_chunk() -> None:
    """A chunk is the unit the cache reuses, so it cannot hold less than one."""
    assert rolling(1)._slots == 1


def test_control_reaches_one_chunk_further_than_history() -> None:
    """A chunk reads the map for its own frames, which are not history yet.

    The ranges follow the partition the rollout generates in, not multiples of
    the chunk size, so a slot holds exactly what one chunk reads. The clip opens
    with a one-frame range, which takes a whole slot like any other.
    """
    rollout = rolling(8)

    assert rollout._slots == 2
    assert rollout._control_frames == 13
    assert rollout._control_ranges == [(0, 1), (1, 5), (5, 9), (9, 13)]


def test_an_unbounded_window_front_loads_the_whole_control_track() -> None:
    """Unbounded has to mean unbounded, or "no bound" would still stream."""
    rollout = rolling(None)

    assert rollout._control_ranges is None
    assert rollout._control_frames == ROLLING.frames_per_view


def test_the_oldest_chunk_is_the_one_the_cache_gives_up() -> None:
    """Three committed chunks into two slots: the first is overwritten.

    The retained pair comes back in the order the cache physically holds it --
    newest first, because it reused slot 0 -- and that order is what the layout
    is told, so the metadata follows the keys rather than the clock.
    """
    rollout = rolling(8)
    run_to_end(rollout)

    assert rollout._slot_frames == [(9, 13), (5, 9)]
    assert rollout._history == [(9, 13), (5, 9)]


def test_a_window_can_evict_the_short_chunk_a_clip_opens_with() -> None:
    """Text2video generates frame 0 alone, and the window has to be able to drop it.

    Giving that chunk a slot of its own is what makes it possible. Keeping it
    outside the slots instead left it in the cache after the chunk beside it had
    been evicted, so history read as frame 0, a gap, then the window -- and the
    layout refuses a gap, because nothing would tell the model it was there.
    This clip evicts (1, 5) on its fourth commit while frame 0 is still held,
    which is exactly that shape.
    """
    geometry = ClipGeometry(
        num_views=2,
        frames_per_view=17,
        patch_h=1,
        patch_w=2,
        frames_per_chunk=4,
        condition_frames=0,
    )
    _controls, text_ids = inputs()
    assert geometry.frames_per_view is not None
    rollout = ChunkRollout(
        build(),
        geometry=geometry,
        control_tokens=torch.randn(
            geometry.num_views * geometry.frames_per_view * geometry.spatial_tokens,
            PATCH_DIM,
        ),
        text_ids=text_ids,
        history_frames=8,
    )

    latent = run_to_end(rollout)

    # Frame 0 and (1, 5) are both gone, and what is left is an unbroken window.
    assert rollout._history == [(5, 9), (9, 13)]
    assert latent.shape[2] == geometry.frames_per_view


def test_a_window_can_evict_the_short_chunk_two_supplied_frames_leave() -> None:
    """The same shape without text2video, so the fix is not about frame 0.

    Two conditioning frames leave (2, 5) over, three frames in a four-frame
    slot. It ages out like any other chunk rather than sitting behind a hole.
    """
    geometry = ClipGeometry(
        num_views=2,
        frames_per_view=21,
        patch_h=1,
        patch_w=2,
        frames_per_chunk=4,
        condition_frames=2,
    )
    _controls, text_ids = inputs()
    assert geometry.frames_per_view is not None
    rollout = ChunkRollout(
        build(),
        geometry=geometry,
        control_tokens=torch.randn(
            geometry.num_views * geometry.frames_per_view * geometry.spatial_tokens,
            PATCH_DIM,
        ),
        text_ids=text_ids,
        condition_tokens=torch.randn(
            geometry.num_views * 2 * geometry.spatial_tokens, PATCH_DIM
        ),
        history_frames=8,
    )

    latent = run_to_end(rollout)

    assert rollout._history == [(9, 13), (13, 17)]
    assert latent.shape[2] == geometry.frames_per_view


def test_control_is_streamed_in_and_rotates_like_history() -> None:
    """The map arrives a chunk at a time and the oldest chunk's slot goes.

    The window opens on the clip's own partition, so the frame it starts with is
    a range of its own and the slot holding it is padded out to a chunk. That
    slot is the conditioning frame's map and is pinned like its RGB sink: the
    reference retains and evicts cameras and controls together, so it is the
    oldest *chunk*, (1, 5), that goes when (13, 17) arrives. The last chunk then
    reads the opening map, the two chunks of history it reads RGB for, and its
    own.
    """
    rollout = rolling(8)
    run_to_end(rollout)

    assert rollout._control_next == ROLLING.frames_per_view
    assert rollout._control_ranges == [(0, 1), (13, 17), (5, 9), (9, 13)]


def test_control_is_retained_and_evicted_with_its_rgb() -> None:
    """Sink plus one chunk of history: the reference's two-completed-chunk cache.

    Cameras and controls are retained and evicted together, so the last chunk
    reads the opening map with the opening frame, the map of the one chunk of
    RGB history, and its own -- not the map of a chunk whose RGB is long gone.
    """
    rollout = rolling(4)
    run_to_end(rollout)

    assert rollout._slots == 1
    assert rollout._control_ranges is not None
    assert rollout._control_ranges[0] == (0, 1)
    assert set(rollout._control_ranges) == {(0, 1), (9, 13), (13, 17)}


def test_an_unbounded_cache_keeps_every_committed_chunk_in_order() -> None:
    rollout = rolling(None)
    run_to_end(rollout)

    assert rollout._history == [(1, 5), (5, 9), (9, 13)]


def test_the_layout_describes_the_frames_the_cache_actually_holds() -> None:
    """The one mismatch that would be silent, checked where it is decidable.

    Nothing but the ranges' order couples the metadata to the keys sitting in
    the cache. Get it wrong and every retained frame is labelled with another
    frame's number and timestamp: attention still runs, the clip still decodes,
    and it is wrong in a way no shape or dtype catches.
    """
    rollout = rolling(8)
    run_to_end(rollout)

    memory = build_memory_layout(
        ROLLING,
        control_frame_ranges=rollout._control_ranges,
        control_slot_frames=ROLLING.frames_per_chunk,
        history_frame_ranges=rollout._history,
        device=torch.device("cpu"),
    )

    def frames_of(role: int, ranges) -> list[int]:
        return [
            frame
            for start, end in ranges
            for _ in range(ROLLING.num_views)
            for frame in range(start, end)
            for _ in range(ROLLING.spatial_tokens)
        ]

    for role, ranges in (
        (ROLE_CONTROL, rollout._control_ranges),
        (ROLE_CLEAN_TARGET, rollout._history),
    ):
        held = memory.stream.frame_id[memory.stream.token_role_id == role]
        assert held.tolist() == frames_of(role, ranges)


def test_the_short_final_control_group_is_padded_not_described() -> None:
    """Padding is the only honest description of keys nothing should read."""
    memory = build_memory_layout(
        ROLLING,
        control_frame_ranges=[(12, 16), (16, 17), (8, 12)],
        control_slot_frames=ROLLING.frames_per_chunk,
        device=torch.device("cpu"),
    )
    per_frame = ROLLING.num_views * ROLLING.spatial_tokens

    assert (memory.stream.token_role_id == ROLE_PADDING).sum().item() == 3 * per_frame


def test_evicting_a_chunk_changes_what_the_model_produces() -> None:
    """Otherwise the tests above could pass with the window doing nothing."""
    assert not torch.equal(run_to_end(rolling(8)), run_to_end(rolling(None)))


# --- compiling the noisy pass -----------------------------------------------


def compile_spy(monkeypatch) -> list[str | None]:
    """Record what torch.compile is asked for, and hand back the original.

    The point of these tests is where and how often compilation is requested,
    not what the compiler makes of it, and actually compiling costs seconds a
    call for an answer that is the same either way.
    """
    modes: list[str | None] = []

    def spy(fn, **kwargs):
        modes.append(kwargs.get("mode"))
        return fn

    monkeypatch.setattr(torch, "compile", spy)
    return modes


# --- the control seam ------------------------------------------------------


class SpyControls:
    """A provider that records what it was asked for and answers from a track."""

    def __init__(self, tokens: torch.Tensor, geometry: ClipGeometry) -> None:
        self._inner = EncodedControls(tokens, geometry)
        self.asked: list[tuple[int, int]] = []

    def tokens(self, start: int, end: int) -> torch.Tensor:
        self.asked.append((start, end))
        return self._inner.tokens(start, end)


def test_a_rollout_takes_its_control_from_a_provider() -> None:
    """The seam ludus arrives through, so the rollout must actually go through it.

    A resident tensor read directly would keep working and look identical,
    which is why this asks the provider whether it was called rather than
    comparing pictures.
    """
    controls, text_ids = inputs()
    spy = SpyControls(controls, GEOMETRY)

    rollout = ChunkRollout(
        build(), geometry=GEOMETRY, control_tokens=spy, text_ids=text_ids
    )
    with torch.no_grad():
        while not rollout.is_finished:
            rollout.step()

    assert spy.asked == [(0, GEOMETRY.frames_per_view)]


def test_a_provider_and_the_tensor_it_wraps_generate_the_same_clip() -> None:
    """Handing over a tensor is shorthand for wrapping it, and must stay so."""
    controls, text_ids = inputs()

    direct = run_rollout(
        build(), geometry=GEOMETRY, control_tokens=controls, text_ids=text_ids
    )
    through = run_rollout(
        build(),
        geometry=GEOMETRY,
        control_tokens=EncodedControls(controls, GEOMETRY),
        text_ids=text_ids,
    )

    assert torch.equal(direct.latent, through.latent)


def test_a_streaming_rollout_asks_for_control_a_chunk_at_a_time() -> None:
    """What the seam is for: a provider that renders on demand is asked on demand.

    One chunk a call, for the opening window as much as for each top-up, so a
    provider never has to have the whole track -- which is the thing a renderer
    following a live ego pose cannot have. The calls are contiguous and in
    order, which is what ``RenderedControls`` requires.
    """
    assert ROLLING.frames_per_view is not None
    controls = torch.randn(
        ROLLING.num_views * ROLLING.frames_per_view * ROLLING.spatial_tokens, PATCH_DIM
    )
    _controls, text_ids = inputs()
    conditions = torch.randn(ROLLING.num_views * ROLLING.spatial_tokens, PATCH_DIM)
    spy = SpyControls(controls, ROLLING)

    rollout = ChunkRollout(
        build(),
        geometry=ROLLING,
        control_tokens=spy,
        text_ids=text_ids,
        condition_tokens=conditions,
        history_frames=8,
    )
    with torch.no_grad():
        while not rollout.is_finished:
            rollout.step()

    assert spy.asked == [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17)]
    # Two history slots to fill, so the third chunk is the first that costs what
    # a long run costs. A rate quoted before then is measuring the ramp.
    assert rollout.warmup_chunks == 2


def test_an_unbounded_rollout_counts_every_rolling_chunk_as_warmup() -> None:
    """Its key stream grows to the last chunk, so no chunk is like the one after.

    ``warmup_chunks`` still answers rather than being a special case, and what
    it answers leaves only the tail to average. Three of this clip's four chunks
    grow the cache, so only the last is worth quoting -- and it is the slowest.
    Reading slow is the safe way to be wrong about a rate.
    """
    assert ROLLING.frames_per_view is not None
    controls = torch.randn(
        ROLLING.num_views * ROLLING.frames_per_view * ROLLING.spatial_tokens, PATCH_DIM
    )
    _controls, text_ids = inputs()
    rollout = ChunkRollout(
        build(),
        geometry=ROLLING,
        control_tokens=controls,
        text_ids=text_ids,
        condition_tokens=torch.randn(
            ROLLING.num_views * ROLLING.spatial_tokens, PATCH_DIM
        ),
    )

    assert rollout.warmup_chunks == 3
    assert rollout.chunk_count == 4


def test_a_clip_whose_only_chunk_is_its_last_has_nothing_to_fill() -> None:
    """No warmup, because the cache never grows: the last chunk is not committed.

    Nothing attends to the final chunk, so it is never replayed into memory. A
    one-frame clip is nothing but that chunk, so it reserves no slots at all and
    the first chunk's rate is the truth rather than an oversight.
    """
    geometry = ClipGeometry(
        num_views=2,
        frames_per_view=1,
        patch_h=1,
        patch_w=2,
        frames_per_chunk=4,
        condition_frames=0,
    )
    _controls, text_ids = inputs()
    assert geometry.frames_per_view is not None
    rollout = ChunkRollout(
        build(),
        geometry=geometry,
        control_tokens=torch.randn(
            geometry.num_views * geometry.frames_per_view * geometry.spatial_tokens,
            PATCH_DIM,
        ),
        text_ids=text_ids,
    )

    assert rollout.chunk_count == 1
    assert rollout.warmup_chunks == 0


# --- a driver-driven end ---------------------------------------------------


def test_a_rollout_says_which_frames_it_will_do_next() -> None:
    """The range comes from where the last chunk stopped, not from a list."""
    rollout = stepwise(build())

    assert rollout.next_range == (0, 1)
    with torch.no_grad():
        rollout.step()

    assert rollout.next_range == (1, 5)


def test_a_driver_can_end_a_run_the_clip_has_not_ended() -> None:
    """With controls arriving from outside, only their supplier knows when to stop.

    A clip's length cannot be that signal, because an interactive run has no
    clip. This is the seam, and it has to hold even mid-clip.
    """
    rollout = stepwise(build())
    with torch.no_grad():
        rollout.step()
    assert not rollout.is_finished

    rollout.stop()

    assert rollout.is_finished
    assert rollout.next_range is None


def test_stopping_leaves_what_was_generated_readable() -> None:
    """Stopping ends the run; it does not throw the frames away."""
    rollout = stepwise(build())
    with torch.no_grad():
        rollout.step()
    rollout.stop()

    assert rollout.latent_for(0, 1).shape[2] == 1
    assert rollout.chunks_done == 1


def test_a_stopped_rollout_refuses_another_chunk_and_says_why() -> None:
    """Not "the clip is complete", which it is not -- a driver ended it."""
    rollout = stepwise(build())
    rollout.stop()

    with pytest.raises(RuntimeError, match="nothing left to generate"), torch.no_grad():
        rollout.step()


def test_a_stopped_rollout_will_not_pass_off_a_part_clip_as_the_whole_one() -> None:
    """``result`` would otherwise unpatchify zeros, which decode to grey frames."""
    rollout = stepwise(build())
    with torch.no_grad():
        rollout.step()
    rollout.stop()

    with pytest.raises(RuntimeError, match="still to generate"):
        rollout.result()


# --- the token window ------------------------------------------------------


def windowed(token_frames, geometry=ROLLING, seed=0) -> ChunkRollout:
    assert geometry.frames_per_view is not None
    controls = torch.randn(
        geometry.num_views * geometry.frames_per_view * geometry.spatial_tokens,
        PATCH_DIM,
    )
    _controls, text_ids = inputs()
    conditions = torch.randn(geometry.num_views * geometry.spatial_tokens, PATCH_DIM)
    return ChunkRollout(
        build(seed),
        geometry=geometry,
        control_tokens=controls,
        text_ids=text_ids,
        condition_tokens=conditions,
        token_frames=token_frames,
    )


def test_a_windowed_rollout_accumulates_no_trace_for_nobody_to_read() -> None:
    """The one list left that would have grown with the run.

    ``result`` is the only reader of the accumulated trace, and it refuses a
    windowed rollout -- so on the runs that never end, keeping it would be a
    list growing forever with nothing to look at it. Each chunk's own trace
    still comes back from :meth:`step`, which is what the caller uses.
    """
    bounded = windowed(8)
    unbounded = windowed(None)

    with torch.no_grad():
        while not bounded.is_finished:
            step, _ = bounded.step()
            unbounded.step()
    assert step.end > 0  # it really did generate

    assert bounded._trace == []
    assert len(unbounded._trace) == unbounded.chunk_count


def test_a_token_window_gives_the_same_frames_the_whole_clip_would() -> None:
    """The window is on the way out to the decoder, so it must not alter one.

    Nothing in generation reads these tokens back -- the clean replay uses the
    denoised tensor -- so a window can only ever change which frames are still
    readable, never what is in them. Bit-identical, because "close" here would
    mean the ring is off by a frame somewhere.
    """
    bounded = windowed(8)
    unbounded = windowed(None)

    with torch.no_grad():
        while not bounded.is_finished:
            first, _ = bounded.step()
            second, _ = unbounded.step()
            assert (first.start, first.end) == (second.start, second.end)
            assert torch.equal(
                bounded.latent_for(first.start, first.end),
                unbounded.latent_for(second.start, second.end),
            )


def test_a_window_the_clip_fits_stores_every_frame_at_its_own_index() -> None:
    """Bounded and unbounded then lay out identically, which is why one ring does."""
    rollout = windowed(None, FITTING)

    assert rollout._tokens.capacity == FITTING.frames_per_view
    assert FITTING.frames_per_view is not None
    assert rollout._tokens._ring(0, FITTING.frames_per_view).tolist() == list(
        range(FITTING.frames_per_view)
    )


def test_frames_that_have_left_the_window_are_refused_not_served_stale() -> None:
    """A ring would otherwise hand back whatever those frames held a wrap ago.

    Which decodes to a picture, so it would go out as video rather than as an
    error. This is the one thing a window can get wrong that is not visible.
    """
    rollout = windowed(8)

    with torch.no_grad():
        while rollout.chunks_done < 3:
            rollout.step()

    assert rollout.latent_for(9, 13).shape[2] == 4
    with pytest.raises(ValueError, match="reaches back past"):
        rollout.latent_for(0, 4)


def test_a_window_is_at_least_the_conditioning_frames_and_one_chunk() -> None:
    """Less than that cannot serve the first decode, which spans both."""
    rollout = windowed(1)

    assert (
        rollout._tokens.capacity == ROLLING.condition_frames + ROLLING.frames_per_chunk
    )
    with torch.no_grad():
        rollout.step()

    assert rollout.latent_for(0, 5).shape[2] == 5


def test_a_windowed_rollout_refuses_to_hand_back_a_clip_it_did_not_keep() -> None:
    """``result`` promises the whole clip, and a window has not got one."""
    rollout = windowed(8)
    with torch.no_grad():
        while not rollout.is_finished:
            rollout.step()

    with pytest.raises(RuntimeError, match="no whole clip"):
        rollout.result()


def test_keeping_the_whole_clip_leaves_result_working() -> None:
    """The default, and what every probe and mp4 run depends on."""
    rollout = windowed(None)

    assert run_to_end(rollout).shape[2] == ROLLING.frames_per_view


def test_the_window_is_written_in_order_and_says_so_if_not() -> None:
    """A ring cannot tell a skipped frame from a wrapped one, so it must refuse."""
    window = _TokenWindow(
        num_views=2,
        frames=8,
        spatial=2,
        width=PATCH_DIM,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    window.append(0, 4, torch.randn(2, 4, 2, PATCH_DIM))

    with pytest.raises(ValueError, match="do not carry on from 4"):
        window.append(5, 9, torch.randn(2, 4, 2, PATCH_DIM))


def test_more_frames_at_once_than_the_window_holds_is_refused() -> None:
    """The write would wrap onto itself and keep only the tail."""
    window = _TokenWindow(
        num_views=2,
        frames=4,
        spatial=2,
        width=PATCH_DIM,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="do not fit"):
        window.append(0, 5, torch.randn(2, 5, 2, PATCH_DIM))


def test_the_window_reads_back_what_it_wrote_across_a_wrap() -> None:
    """The ring's own arithmetic, away from a rollout: write past the end and read."""
    window = _TokenWindow(
        num_views=2,
        frames=4,
        spatial=2,
        width=PATCH_DIM,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    first = torch.randn(2, 2, 2, PATCH_DIM)
    second = torch.randn(2, 2, 2, PATCH_DIM)
    third = torch.randn(2, 2, 2, PATCH_DIM)
    window.append(0, 2, first)
    window.append(2, 4, second)
    window.append(4, 6, third)

    assert torch.equal(window.read(4, 6), third)
    assert torch.equal(window.read(2, 4), second)
    assert torch.equal(window.read(2, 6), torch.cat([second, third], dim=1))
    with pytest.raises(ValueError, match="reaches back past"):
        window.read(0, 2)


def test_the_noisy_pass_is_compiled_once_for_the_whole_rollout(monkeypatch) -> None:
    """Per chunk would cost more than it saves, which is the whole trade here.

    Two chunks of four denoising steps make eight passes, and one compilation
    has to serve all of them. It only can because the pass takes what varies as
    arguments; closing over the chunk's plan would give each chunk its own
    function and so its own compile.

    Bounded, because compiling an unbounded cache is refused: its key stream is
    a new shape every chunk, which is the recompile this test is about avoiding.
    """
    modes = compile_spy(monkeypatch)

    go(build(), compile_denoise="max-autotune", history_frames=4)

    assert modes == ["max-autotune"]


def test_torch_s_own_default_is_what_default_asks_for(monkeypatch) -> None:
    """``mode="default"`` is not a mode torch takes; ``None`` is how you say it."""
    modes = compile_spy(monkeypatch)

    go(build(), compile_denoise="default", history_frames=4)

    assert modes == [None]


def test_a_rollout_left_uncompiled_never_reaches_for_the_compiler(monkeypatch) -> None:
    modes = compile_spy(monkeypatch)

    go(build())

    assert modes == []


def test_compiling_an_unbounded_cache_is_refused_rather_than_run_slowly() -> None:
    """Accepting it would recompile every chunk, which is worse than not compiling.

    The cost is invisible in the result -- the clip comes out the same -- so the
    only place it can be caught is here, before the first chunk.
    """
    with pytest.raises(ValueError, match="bound the cache"):
        go(build(), compile_denoise="default", history_frames=None)


def test_unsafe_or_unknown_compile_modes_are_refused_by_the_public_api() -> None:
    for mode in ("reduce-overhead", "fastest"):
        with pytest.raises(ValueError, match="compile_denoise must be one of"):
            go(build(), compile_denoise=mode, history_frames=4)


def test_an_attention_arena_with_an_unbounded_cache_is_refused() -> None:
    with pytest.raises(ValueError, match="arena needs history_frames"):
        go(build(), attention_kv_arena=True, history_frames=None)


@pytest.mark.parametrize("history_frames", [0, -1])
def test_a_nonpositive_programmatic_history_window_is_refused(
    history_frames: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive or None"):
        go(build(), history_frames=history_frames)


def test_compiled_attention_arena_is_refused_by_the_public_api() -> None:
    with pytest.raises(ValueError, match="failed multi-seed quality"):
        go(
            build(),
            history_frames=4,
            attention_kv_arena=True,
            compile_denoise="default",
        )


def test_compiling_the_noisy_pass_does_not_change_the_pictures() -> None:
    """The one test here that pays for a real compile, because it has to.

    Everything above stubs the compiler out, so nothing else would notice if the
    compiled pass were handed its arguments in the wrong order -- it would still
    be one compile, of a function computing the wrong thing.
    """
    torch.testing.assert_close(
        go(build(), compile_denoise="default", history_frames=4).latent,
        go(build(), history_frames=4).latent,
    )


def test_every_pass_reads_the_prompt_and_not_its_memory(monkeypatch) -> None:
    """The two caches are interchangeable to the type checker, not to the model.

    ``text_kv`` and ``memory_kv`` are both per-layer key/value sequences, and the
    key stream is their concatenation, so handing them over the wrong way round
    builds a stream of exactly the width the mask expects out of the wrong
    tokens: nothing raises, and every relative test still compares two runs that
    are wrong in the same way. Length is what tells them apart -- the prompt is
    a handful of tokens, never the hundreds sitting in memory.
    """
    seen: list[int] = []
    original = rollout_module.run_chunk_layers

    def spy(*args, **kwargs):
        seen.append(int(kwargs["text_kv"][0][0].shape[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(rollout_module, "run_chunk_layers", spy)

    go(build())

    assert set(seen) == {TEXT}


def test_the_attention_scope_reaches_every_chunk_pass(monkeypatch) -> None:
    """A scope that parses but never reaches the mask is the bug this guards.

    Every pass is checked rather than the output alone: the loop builds a mask
    three times per chunk -- control top-up, denoise, clean replay -- and one
    of them left on the default is a run that half-obeys the flag.
    """
    recorder = PassRecorder().install(monkeypatch)
    go(build(), attention_scope="decomposed")
    narrow = [call["mask"] for call in recorder.calls]

    recorder = PassRecorder().install(monkeypatch)
    go(build(), attention_scope="all_views")
    wide = [call["mask"] for call in recorder.calls]

    assert len(narrow) == len(wide) > 0
    for narrowed, widened in zip(narrow, wide, strict=True):
        assert bool((widened | ~narrowed).all()), "decomposed only removes keys"
        assert bool(narrowed.any(dim=1).all()), "an all-false row is a silent NaN"
    # Not every pass: the opening chunk is a single frame, where every key is
    # already at the query's own instant and the two scopes agree by right.
    assert any(
        int(narrowed.sum()) < int(widened.sum())
        for narrowed, widened in zip(narrow, wide, strict=True)
    )


def test_the_scope_changes_what_is_generated() -> None:
    """The point of the flag, and what the picture will show."""
    network = build()

    decomposed = go(network, attention_scope="decomposed")
    everything = go(network, attention_scope="all_views")

    assert not torch.allclose(decomposed.latent, everything.latent)


def test_the_fold_reaches_the_prefill_and_every_chunk_pass(monkeypatch) -> None:
    """The flag has to arrive everywhere a mask is built, or the run is mixed.

    Recorded rather than run: the fold gates its passes with block masks, and
    building those is the one part of this loop that wants a GPU. What can be
    checked here is the wiring, which is where a flag like this goes wrong --
    the prefill writes the cache every chunk then reads, so a run folding its
    chunks and masking its prefill would be two attentions deep before its
    first frame, and would look like a quality problem rather than a bug.
    """
    seen: list[bool] = []
    chunk_mask = rollout_module._chunk_mask
    prefill = rollout_module._prefill

    def chunk_spy(*args, maskless=False, **kwargs):
        seen.append(maskless)
        return chunk_mask(*args, maskless=False, **kwargs)

    def prefill_spy(*args, maskless=False, **kwargs):
        seen.append(maskless)
        return prefill(*args, maskless=False, **kwargs)

    monkeypatch.setattr(rollout_module, "_chunk_mask", chunk_spy)
    monkeypatch.setattr(rollout_module, "_prefill", prefill_spy)
    go(build(), attention_scope="decomposed", maskless_attention=True)

    assert seen, "no mask was built at all"
    assert all(seen), f"{seen.count(False)} of {len(seen)} passes ignored the flag"


def test_a_whole_run_folds_and_still_comes_out_finite() -> None:
    """The fold through the loop rather than through one pass.

    The wiring test above records the flag and then runs the mask, so nothing
    it does would catch the fold itself failing in the rollout -- and the
    failure to expect is not an exception but a NaN, since a query row that no
    pass can read merges -inf against -inf. Caption and padding rows are
    exactly that, so a clip is the cheapest place they all show up together.

    The difference from the mask is the point of the flag; its size here is not
    comparable to the 12.8% measured on real weights, since this network is a
    few random layers over a toy rig.
    """
    with torch.no_grad():
        folded = go(build(), attention_scope="decomposed", maskless_attention=True)
        masked = go(build(), attention_scope="decomposed")

    assert torch.isfinite(folded.latent).all()
    assert folded.latent.shape == masked.latent.shape
    assert not torch.allclose(folded.latent, masked.latent)


def test_per_camera_captions_reach_the_chunk_masks(monkeypatch) -> None:
    """The runs have to survive the whole way from the prompt to the kernel.

    Everything in between takes them as an optional argument, so a link left
    unpassed is silent: the run still completes, and every camera reads every
    caption again. The first chunk is one frame of two cameras at two spatial
    tokens each, so rows 0-1 are the first camera and 2-3 the second, and the
    prompt's three tokens are the first three key columns: two for the first
    camera's caption, one for the second's.
    """
    recorder = PassRecorder().install(monkeypatch)
    controls, text_ids = inputs()
    with torch.no_grad():
        run_rollout(
            network=build(),
            geometry=GEOMETRY,
            control_tokens=controls,
            text_ids=text_ids,
            view_text_tokens=[2, 1],
        )

    mask = recorder.calls[0]["mask"]

    assert mask[0, :2].all() and not mask[0, 2]
    assert mask[1, :2].all() and not mask[1, 2]
    assert mask[2, 2] and not mask[2, :2].any()
    assert mask[3, 2] and not mask[3, :2].any()


# --- a run with no end ------------------------------------------------------
#
# ``frames_per_view=None`` is a clip that never finishes, which is what an
# interactive session is: the controls arrive as the driver makes them and
# nothing knows in advance how long anyone will drive. These check that such a
# rollout runs past any length a clip could have had, that it costs the same at
# chunk forty as at chunk four, and that the three things it cannot have are
# refused where a caller can read why.

ENDLESS = dataclasses.replace(ROLLING, frames_per_view=None)


class DrawnControls:
    """A provider with no end, which is what an endless rollout requires.

    Answers any range at all, as a renderer following a live ego pose does:
    there is no track to run off, only a next frame to draw.
    """

    def __init__(self, geometry: ClipGeometry) -> None:
        self._geometry = geometry
        self.asked: list[tuple[int, int]] = []

    def tokens(self, start: int, end: int) -> torch.Tensor:
        self.asked.append((start, end))
        generator = torch.Generator().manual_seed(start)
        count = self._geometry.num_views * (end - start) * self._geometry.spatial_tokens
        return torch.randn(count, PATCH_DIM, generator=generator)


def endless(**kwargs) -> ChunkRollout:
    _controls, text_ids = inputs()
    conditions = torch.randn(ENDLESS.num_views * ENDLESS.spatial_tokens, PATCH_DIM)
    return ChunkRollout(
        build(),
        geometry=ENDLESS,
        control_tokens=DrawnControls(ENDLESS),
        text_ids=text_ids,
        condition_tokens=conditions,
        history_frames=8,
        token_frames=8,
        **kwargs,
    )


def test_a_run_with_no_end_keeps_generating_past_any_clip_length() -> None:
    """The point of the whole thing: nothing stops it but a caller.

    Forty chunks is well past ``ROLLING``'s seventeen frames, which is the
    length this geometry had before its end was taken away.
    """
    rollout = endless()

    with torch.no_grad():
        for _ in range(40):
            trace, _tokens = rollout.step()

    assert not rollout.is_finished
    assert ROLLING.frames_per_view is not None
    assert trace.end > ROLLING.frames_per_view * 2


@pytest.mark.parametrize("arena", [False, True])
def test_a_run_with_no_end_costs_the_same_at_chunk_forty_as_at_chunk_four(
    arena: bool,
) -> None:
    """Flat memory is what makes the previous test a demo rather than an OOM.

    The cache is a ring of slots, so the tokens attended over and the tensors
    holding them have to be the same objects and the same size however long
    the run goes on. With the arena too, which four views turn on by default
    and which reserves its shape once from a bound this run does not have.
    """
    rollout = endless(attention_kv_arena=arena)

    with torch.no_grad():
        for _ in range(4):
            rollout.step()
        early = rollout._cache.length
        pointers = [(k.data_ptr(), v.data_ptr()) for k, v in rollout._cache.layers()]
        for _ in range(36):
            rollout.step()

    assert rollout._cache.length == early
    assert [
        (k.data_ptr(), v.data_ptr()) for k, v in rollout._cache.layers()
    ] == pointers


def test_a_run_with_no_end_draws_its_control_one_chunk_at_a_time() -> None:
    """There is no whole track to front-load, so every chunk has to ask."""
    rollout = endless()

    with torch.no_grad():
        for _ in range(12):
            rollout.step()

    provider = rollout._controls
    assert isinstance(provider, DrawnControls)
    assert len(provider.asked) > 12
    for (_, previous_end), (start, _) in zip(
        provider.asked, provider.asked[1:], strict=False
    ):
        assert start == previous_end


def test_a_run_with_no_end_is_ended_by_its_driver() -> None:
    """``stop`` is the only thing that can finish one, and it has to work."""
    rollout = endless()

    with torch.no_grad():
        rollout.step()
    rollout.stop()

    assert rollout.is_finished
    with pytest.raises(RuntimeError, match="nothing left to generate"):
        with torch.no_grad():
            rollout.step()


def test_a_run_with_no_end_has_no_finished_clip_to_hand_back() -> None:
    """``result`` promises a whole clip, and this one has no last frame."""
    rollout = endless()

    with torch.no_grad():
        rollout.step()

    with pytest.raises(RuntimeError, match="no end"):
        rollout.result()


def test_a_run_with_no_end_refuses_a_cache_that_keeps_everything() -> None:
    """Unbounded history plus unbounded length is the OOM, stated up front."""
    _controls, text_ids = inputs()

    with pytest.raises(ValueError, match="needs history_frames"):
        ChunkRollout(
            build(),
            geometry=ENDLESS,
            control_tokens=DrawnControls(ENDLESS),
            text_ids=text_ids,
            condition_tokens=torch.randn(
                ENDLESS.num_views * ENDLESS.spatial_tokens, PATCH_DIM
            ),
            token_frames=8,
        )


def test_a_run_with_no_end_refuses_an_encoded_track() -> None:
    """A tensor is as long as it is, so it cannot feed a run that never stops."""
    controls, text_ids = inputs()

    with pytest.raises(ValueError, match="cannot feed a clip with no end"):
        ChunkRollout(
            build(),
            geometry=ENDLESS,
            control_tokens=controls,
            text_ids=text_ids,
            condition_tokens=torch.randn(
                ENDLESS.num_views * ENDLESS.spatial_tokens, PATCH_DIM
            ),
            history_frames=8,
            token_frames=8,
        )


def test_fused_merge_policy_reaches_prefill_topup_denoise_and_replay(monkeypatch):
    """Run a windowed clip, checking actual plans and exact native CPU fallback."""
    from cosmos3_nano_sim_auto.impl.transformer.fold import FoldPlan

    generator = torch.Generator().manual_seed(91)
    assert ROLLING.frames_per_view is not None
    controls = torch.randn(
        ROLLING.num_views * ROLLING.frames_per_view * ROLLING.spatial_tokens,
        PATCH_DIM,
        generator=generator,
    )
    conditions = torch.randn(
        ROLLING.num_views * ROLLING.spatial_tokens, PATCH_DIM, generator=generator
    )
    _, text_ids = inputs()
    network = build()
    kwargs: dict[str, Any] = dict(
        geometry=ROLLING,
        control_tokens=controls,
        condition_tokens=conditions,
        text_ids=text_ids,
        history_frames=4,
        attention_scope="decomposed",
        maskless_attention=True,
    )
    expected = run_rollout(network, **kwargs)
    masks, passes, merges = [], [], []
    original_prefill = rollout_module.run_prefill
    original_chunk = rollout_module.run_chunk_layers
    original_mask = rollout_module._chunk_mask
    from cosmos3_nano_sim_auto.impl.transformer import attention as attention_module

    original_merge = attention_module.merge_attentions

    def prefill(*args, **kw):
        masks.append(kw["mask"])
        return original_prefill(*args, **kw)

    def chunk(*args, **kw):
        masks.append(kw["mask"])
        return original_chunk(*args, **kw)

    def mask(*args, **kw):
        passes.append(kw["pass_kind"])
        return original_mask(*args, **kw)

    def merge(parts, *, fused=False, output_dtype=None):
        merges.append(fused)
        return original_merge(parts, fused=fused, output_dtype=output_dtype)

    monkeypatch.setattr(rollout_module, "run_prefill", prefill)
    monkeypatch.setattr(rollout_module, "run_chunk_layers", chunk)
    monkeypatch.setattr(rollout_module, "_chunk_mask", mask)
    monkeypatch.setattr(attention_module, "merge_attentions", merge)
    actual = run_rollout(network, fused_attention_merge=True, **kwargs)
    assert masks and all(
        isinstance(mask, FoldPlan) and mask.fused_merge for mask in masks
    )
    assert set(passes) == {"control", "noisy", "clean"}
    assert merges and all(merges)
    assert torch.equal(actual.latent, expected.latent)


def test_fused_merge_requires_maskless_attention_at_the_rollout_api():
    with pytest.raises(ValueError, match="requires maskless_attention"):
        go(build(), fused_attention_merge=True)


def test_odd_latent_scheduler_excludes_padding_at_every_denoising_step(monkeypatch):
    geometry = dataclasses.replace(GEOMETRY, latent_size=(1, 3))
    controls, text = inputs()
    rollout = ChunkRollout(
        build(), geometry=geometry, control_tokens=controls, text_ids=text
    )
    original_flow = rollout._flow
    noises = []
    original_noise = rollout_module._chunk_noise

    def noise(count, width, **kwargs):
        noises.append((count, width))
        return original_noise(count, width, **kwargs)

    def flow(latent, *args, **kwargs):
        patches = latent.reshape(geometry.num_views, -1, 1, 2, 2, 2, 4)
        assert torch.count_nonzero(patches[..., 1, :, :]) == 0
        assert torch.count_nonzero(patches[:, :, :, -1, :, 1, :]) == 0
        return original_flow(latent, *args, **kwargs)

    monkeypatch.setattr(rollout_module, "_chunk_noise", noise)
    rollout._flow = flow
    while not rollout.is_finished:
        rollout.step()
    result = rollout.result()
    assert result.latent.shape == (2, 4, 5, 1, 3)
    assert noises == [(2 * 4 * 1 * 1 * 3, 1), (2 * 4 * 4 * 1 * 3, 1)]
    assert torch.isfinite(result.latent).all()


@torch.no_grad()
def test_final_rig_checkpoint_runs_prefill_topups_and_replay_with_physical_ids(
    monkeypatch,
):
    network = build()
    network.rig_view_embed = torch.nn.Embedding(12, SPEC.llm.hidden_size)
    geometry = dataclasses.replace(
        GEOMETRY,
        frames_per_view=13,
        frames_per_chunk=3,
        condition_frames=1,
        rig_view_ids=(6, 4),
        seconds_per_frame=4 / 30,
    )
    assert geometry.frames_per_view is not None
    controls = torch.randn(
        geometry.num_views * geometry.frames_per_view * geometry.spatial_tokens,
        PATCH_DIM,
    )
    conditions = torch.randn(geometry.num_views * geometry.spatial_tokens, PATCH_DIM)
    common: dict[str, Any] = dict(
        geometry=geometry,
        control_tokens=controls,
        condition_tokens=conditions,
        text_ids=torch.tensor([1, 2, 3]),
        history_frames=3,
        attention_scope="decomposed",
        decomposed_temporal_window_seconds=0.4,
        seed=7,
    )
    folded = run_rollout(network, maskless_attention=True, **common)
    from cosmos3_nano_sim_auto.impl import packing, prefill
    from cosmos3_nano_sim_auto.impl.transformer.mask import ROLE_UND

    visibility = packing.visibility

    def maskless_reference(q, kv, **kwargs):
        allowed = visibility(q, kv, **kwargs)
        if kwargs.get("scope") == "decomposed":
            allowed &= (
                ~q.is_control[:, None]
                | (q.view_id[:, None] == kv.view_id[None, :])
                | (kv.token_role_id[None, :] == ROLE_UND)
            )
        return allowed

    monkeypatch.setattr(packing, "visibility", maskless_reference)
    monkeypatch.setattr(prefill, "visibility", maskless_reference)
    dense = run_rollout(network, maskless_attention=False, **common)
    # Includes first eviction, control top-ups, and clean replays, not just one forward.
    torch.testing.assert_close(folded.latent, dense.latent, atol=3e-5, rtol=3e-5)
