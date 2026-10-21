# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the clean pass that fills memory.

Two things are being checked, and they are quite different in character.

The layout half asks whether the rows we gather out of the prefill pack are the
rows the rollout will later look for. That is settled against
``build_memory_layout``, which is already tested: gather the pack and the
metadata you get back has to be memory's, field for field. If those two ever
drift the rollout reads keys stamped with the wrong frame or view and nothing
raises.

The other half is about the target. This stops it at the supplied frames where
the reference implementation runs it over the whole clip, because a demo does not have the frames the reference implementation
assumes it has. The tests below pin both halves of why that is safe: the control
keys come out identical either way, and the supplied frames' keys do not.
"""

from __future__ import annotations

import pytest
import torch
from cosmos3_nano_sim_auto.impl.packing import (
    ClipGeometry,
    build_chunk_metadata,
    build_memory_layout,
)
from cosmos3_nano_sim_auto.impl.prefill import (
    build_prefill_pack,
    pad_control_slots,
    run_control_topup,
    run_prefill,
)
from cosmos3_nano_sim_auto.impl.transformer.mask import (
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_TARGET_CONDITION,
    ROLE_UND,
)
from cosmos3_nano_sim_auto.impl.transformer.rope import (
    chunk_mrope_ids,
    clip_mrope_ids,
    vision_temporal_offset,
)
from cosmos_interactive._network import _MoTLayer
from cosmos_interactive._rope import text_mrope_ids
from cosmos_interactive._spec import LLMSpec

pytestmark = pytest.mark.ci_cpu

CLIP = ClipGeometry(
    num_views=3,
    frames_per_view=5,
    patch_h=2,
    patch_w=1,
    frames_per_chunk=2,
    condition_frames=1,
)
TEXT_TOKENS = 4
FPS = 24.0

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


def tagged_kv(tags: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One layer of ``(k, v)`` whose every row carries the tag of its token."""
    rows = tags.reshape(1, 1, -1, 1).to(torch.float32)
    return [(rows.expand(1, 2, len(tags), 8).clone(), rows.clone())]


def test_the_prefilled_control_rows_hold_the_frames_the_layout_claims() -> None:
    """The mismatch that would be silent, decided row by row.

    Nothing but agreement between two orderings couples the control keys to the
    metadata describing them. The pass lays the control out one section per
    range; the cache reuses fixed slots and pads the short ones. Lay it out any
    other way -- one block over every control frame, say -- and the rows still
    have the right shape, attention still runs, and every key is labelled with
    another camera's frame.

    So tag each row with the ``(view, frame)`` that produced it and read the
    tags back through the layout, which is the only place the two orderings can
    be compared.
    """
    ranges = [(0, 1), (1, 3), (3, 5)]
    slot_frames = CLIP.frames_per_chunk
    per_frame = CLIP.num_views * CLIP.spatial_tokens

    tags = torch.tensor(
        [
            view * 100 + frame
            for start, end in ranges
            for view in range(CLIP.num_views)
            for frame in range(start, end)
            for _ in range(CLIP.spatial_tokens)
        ],
        dtype=torch.float32,
    )
    padded = pad_control_slots(
        tagged_kv(tags),
        control_ranges=ranges,
        tokens_per_frame=per_frame,
        slot_frames=slot_frames,
    )

    memory = build_memory_layout(
        CLIP,
        control_frame_ranges=ranges,
        control_slot_frames=slot_frames,
        history_frame_ranges=(),
        history_slot_frames=slot_frames,
    )
    # The control slots sit at the front of both, ahead of the supplied frames
    # the layout goes on to describe and this buffer does not carry.
    written = padded[0][1].reshape(-1)
    region = slice(0, len(written))
    control = (memory.stream.token_role_id == ROLE_CONTROL)[region]

    claimed = (
        memory.stream.view_id[region][control] * 100
        + memory.stream.frame_id[region][control]
    )
    assert len(claimed) == sum(end - start for start, end in ranges) * per_frame
    assert torch.equal(written[control], claimed.to(torch.float32))


def test_a_short_control_range_leaves_its_slot_padded_rather_than_shifting() -> None:
    """A one-frame opening in a four-frame slot must not pull the rest forward.

    Packing the sections back to back would fit, and would put the next slot's
    keys where this slot's padding belongs, so every slot after it is off by
    three frames.
    """
    ranges = [(0, 1), (1, 3)]
    per_frame = CLIP.num_views * CLIP.spatial_tokens
    real = sum((end - start) * per_frame for start, end in ranges)

    padded = pad_control_slots(
        tagged_kv(torch.ones(real)),
        control_ranges=ranges,
        tokens_per_frame=per_frame,
        slot_frames=CLIP.frames_per_chunk,
    )

    rows = padded[0][1].reshape(-1)
    assert len(rows) == len(ranges) * CLIP.frames_per_chunk * per_frame
    # The opening frame, then its padding, then the second range.
    assert torch.equal(rows[:per_frame], torch.ones(per_frame))
    assert not rows[per_frame : CLIP.frames_per_chunk * per_frame].any()


def test_the_gather_walks_the_target_item_s_own_stride() -> None:
    """The trimmed multi-view case, written out, because it is where strides part.

    A target spanning fewer frames than the clip is only ``target_frames`` deep
    per view, where the control track ahead of it is the whole clip. An
    index derived from the geometry rather than from the pack picks the control
    item's stride and lands past the end of the target for every view but the
    first, which is what a second copy of this gather used to do.
    """
    pack = build_prefill_pack(CLIP, text_tokens=TEXT_TOKENS, target_frames=2)

    assert CLIP.frames_per_view is not None
    controls = CLIP.num_views * CLIP.frames_per_view * CLIP.spatial_tokens
    assert controls == 30
    # The controls whole, and then the target's conditioning frames. Target rows
    # are view-outer, frame-inner, so view v's first frame starts at
    # v * 2 frames * 2 spatial -- the conditioning frames are 4 apart, not 10.
    assert pack.memory_token_indexes.tolist() == [
        *range(controls),
        30,
        31,
        34,
        35,
        38,
        39,
    ]


def test_trimming_the_target_makes_the_gather_the_identity() -> None:
    """Stopping at the supplied frames leaves nothing to throw away."""
    pack = build_prefill_pack(CLIP, text_tokens=TEXT_TOKENS)
    memory = build_memory_layout(CLIP)

    assert len(pack.gen) == memory.real_token_count
    assert torch.equal(pack.memory_token_indexes, torch.arange(len(pack.gen)))


@pytest.mark.parametrize("target_frames", [None, CLIP.frames_per_view])
def test_the_gathered_tokens_carry_memory_s_own_metadata(
    target_frames: int | None,
) -> None:
    """The rollout keys on these fields, so they have to survive the gather."""
    pack = build_prefill_pack(
        CLIP, text_tokens=TEXT_TOKENS, target_frames=target_frames
    )
    memory = build_memory_layout(CLIP)
    indexes = pack.memory_token_indexes

    for field in (
        "frame_id",
        "view_id",
        "is_noisy",
        "is_control",
        "token_role_id",
        "causal_step_id",
        "timestamp",
        "sample_id",
    ):
        gathered = getattr(pack.gen, field)[indexes]
        expected = getattr(memory.stream, field)[: memory.real_token_count]
        assert torch.equal(gathered, expected), field


def test_the_pack_holds_the_three_roles_it_should() -> None:
    pack = build_prefill_pack(
        CLIP, text_tokens=TEXT_TOKENS, target_frames=CLIP.frames_per_view
    )
    roles = pack.gen.token_role_id
    assert CLIP.frames_per_view is not None
    per_view = CLIP.frames_per_view * CLIP.spatial_tokens
    conditions = CLIP.condition_frames * CLIP.spatial_tokens

    assert int((roles == ROLE_CONTROL).sum()) == CLIP.num_views * per_view
    assert int((roles == ROLE_TARGET_CONDITION).sum()) == CLIP.num_views * conditions
    assert int((roles == ROLE_CLEAN_TARGET).sum()) == CLIP.num_views * (
        per_view - conditions
    )
    assert torch.all(pack.und.token_role_id == ROLE_UND)


def test_every_query_can_reach_something() -> None:
    """An all-false row is a silent NaN, not an error."""
    pack = build_prefill_pack(
        CLIP, text_tokens=TEXT_TOKENS, target_frames=CLIP.frames_per_view
    )
    mask = pack.mask()

    assert mask.shape == (len(pack.gen), pack.num_und + len(pack.gen))
    assert bool(mask.any(dim=1).all())


def test_controls_look_only_at_strictly_earlier_clean_targets() -> None:
    """the model's ``controls_read_strict_past_clean_rgb``, which the joint checkpoints train with.

    A control reads the clean RGB of earlier steps -- the conditioning frame,
    then each committed chunk -- and never its own step's target or a later one.
    """
    pack = build_prefill_pack(
        CLIP, text_tokens=TEXT_TOKENS, target_frames=CLIP.frames_per_view
    )
    rows = pack.gen.is_control
    targets = ~pack.gen.is_control
    seen = pack.mask()[rows][:, pack.num_und :][:, targets]
    earlier = pack.gen.causal_step_id[targets].unsqueeze(0) < pack.gen.causal_step_id[
        rows
    ].unsqueeze(1)

    assert bool(seen.any())
    assert not bool((seen & ~earlier).any())


def test_conditions_do_look_at_the_target_item() -> None:
    """And the reason it cannot be left untrimmed either."""
    pack = build_prefill_pack(
        CLIP, text_tokens=TEXT_TOKENS, target_frames=CLIP.frames_per_view
    )
    mask = pack.mask()
    condition_rows = pack.gen.token_role_id == ROLE_TARGET_CONDITION
    generated_columns = torch.cat(
        [
            torch.zeros(pack.num_und, dtype=torch.bool),
            pack.gen.token_role_id == ROLE_CLEAN_TARGET,
        ]
    )

    assert bool(mask[condition_rows][:, generated_columns].all())


assert CLIP.frames_per_view is not None


@pytest.mark.parametrize(
    ("target_frames", "message"),
    [(0, "conditioning frames"), (CLIP.frames_per_view + 1, "fit the clip")],
)
def test_a_target_item_outside_the_clip_is_refused(
    target_frames: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_prefill_pack(CLIP, text_tokens=TEXT_TOKENS, target_frames=target_frames)


# ---- running it ----------------------------------------------------------


def build_layers(seed: int = 0) -> list[_MoTLayer]:
    torch.manual_seed(seed)
    return [_MoTLayer(TINY, torch.float32).eval() for _ in range(TINY.num_layers)]


def prefill(layers, *, target_frames=None, target_fill=0.0, flex=False):
    """Run the pass, filling the target's generated frames with a constant."""
    pack = build_prefill_pack(
        CLIP, text_tokens=TEXT_TOKENS, target_frames=target_frames
    )
    offset = vision_temporal_offset(TEXT_TOKENS)
    text_ids, _ = text_mrope_ids(TEXT_TOKENS, 0.0)

    control_ids = clip_mrope_ids(CLIP, fps=FPS, temporal_offset=offset)
    frames = CLIP.condition_frames if target_frames is None else target_frames
    target_ids = chunk_mrope_ids(
        CLIP, chunk_start=0, chunk_frames=frames, fps=FPS, temporal_offset=offset
    )

    # Separate generators, so that changing the pack's length cannot quietly
    # change the prompt as well and make two runs incomparable.
    vision_h = torch.randn(
        1, len(pack.gen), TINY.hidden_size, generator=torch.Generator().manual_seed(7)
    )
    text_h = torch.randn(
        1, TEXT_TOKENS, TINY.hidden_size, generator=torch.Generator().manual_seed(8)
    )
    # Whatever occupies the not-yet-generated frames, the caller does not have it.
    vision_h[:, pack.gen.token_role_id == ROLE_CLEAN_TARGET] = target_fill

    return pack, run_prefill(
        layers,
        TINY,
        text_h=text_h,
        text_position_ids=text_ids,
        vision_h=vision_h,
        vision_position_ids=torch.cat([control_ids, target_ids], dim=1),
        mask=pack.block_mask(block_size=16) if flex else pack.mask(),
        memory_token_indexes=pack.memory_token_indexes,
    )


@torch.no_grad()
def test_flex_attention_fills_the_cache_with_the_same_keys() -> None:
    """Prefill is the pass that matters most here, and the least visibly.

    A chunk that comes out wrong shows up in the picture. Prefill only writes
    the cache, so an error here surfaces later as every chunk being subtly off,
    with nothing pointing back at the cause.

    Prefill's mask is also the emptier of the two by a long way -- 94% of blocks
    against a chunk's 53%, since a control reads only its own view -- so it is
    the one where the two kernels have the most room to disagree.
    """
    # A CPU FlexAttention kernel compiled by an earlier test file can be reused
    # here for a block layout it does not fit, and crashes with an index out of
    # bounds inside inductor's generated code. Compile this one fresh.
    torch._dynamo.reset()
    dense_text, dense_memory = prefill(build_layers())[1]
    flex_text, flex_memory = prefill(build_layers(), flex=True)[1]

    for (dk, dv), (fk, fv) in zip(dense_memory, flex_memory, strict=True):
        assert (fk - dk).abs().max().item() < 1e-5
        assert (fv - dv).abs().max().item() < 1e-5
    # The prompt is keys-only and never gated by this mask, so it must be exact
    # rather than merely close.
    for (dk, dv), (fk, fv) in zip(dense_text, flex_text, strict=True):
        assert torch.equal(fk, dk)
        assert torch.equal(fv, dv)


@torch.no_grad()
def test_run_prefill_returns_a_cache_shaped_for_the_rollout() -> None:
    layers = build_layers()
    pack, (text_kv, memory_kv) = prefill(layers)
    memory = build_memory_layout(CLIP)

    assert len(text_kv) == len(memory_kv) == TINY.num_layers
    for (text_k, text_v), (memory_k, memory_v) in zip(text_kv, memory_kv):
        assert text_k.shape == (1, TINY.num_key_value_heads, TEXT_TOKENS, TINY.head_dim)
        assert text_v.shape == text_k.shape
        assert memory_k.shape == (
            1,
            TINY.num_key_value_heads,
            memory.real_token_count,
            TINY.head_dim,
        )
        assert memory_v.shape == memory_k.shape
    assert len(pack.memory_token_indexes) == memory.real_token_count


@torch.no_grad()
def test_controls_after_the_opening_frame_depend_on_the_clean_target() -> None:
    """Controls read the target's clean frames, but never at their own step.

    The opening frame's map reads no RGB at all, so its keys are untouched by
    whatever fills the target all the way up. Every later control reads clean
    RGB from before it, so a trimmed pack -- which holds only the conditioning
    frame -- gives those keys a different history. That is why the rollout
    computes them once the frames they read are committed.
    """
    layers = build_layers()
    _pack, (_, trimmed) = prefill(layers)
    pack, (_, full) = prefill(
        layers, target_frames=CLIP.frames_per_view, target_fill=3.0
    )

    assert CLIP.frames_per_view is not None
    controls = CLIP.num_views * CLIP.frames_per_view * CLIP.spatial_tokens
    first = pack.gen.causal_step_id[:controls] == 0
    moved = []
    for (trimmed_k, trimmed_v), (full_k, full_v) in zip(trimmed, full):
        for trim, whole in ((trimmed_k, full_k), (trimmed_v, full_v)):
            assert torch.allclose(
                trim[:, :, :controls][:, :, first],
                whole[:, :, :controls][:, :, first],
                atol=1e-6,
            )
            moved.append(
                not torch.allclose(
                    trim[:, :, :controls][:, :, ~first],
                    whole[:, :, :controls][:, :, ~first],
                    atol=1e-4,
                )
            )
    assert any(moved)


@torch.no_grad()
def test_the_conditioning_cache_does_depend_on_the_target_item() -> None:
    """Which is exactly why a demo cannot run the model's pack over frames it does not have."""
    layers = build_layers()
    _pack, (_, quiet) = prefill(
        layers, target_frames=CLIP.frames_per_view, target_fill=0.0
    )
    _pack, (_, loud) = prefill(
        layers, target_frames=CLIP.frames_per_view, target_fill=3.0
    )

    assert CLIP.frames_per_view is not None
    controls = CLIP.num_views * CLIP.frames_per_view * CLIP.spatial_tokens
    moved = [
        not torch.allclose(quiet_k[:, :, controls:], loud_k[:, :, controls:], atol=1e-4)
        for (quiet_k, _), (loud_k, _) in zip(quiet, loud)
    ]

    assert any(moved)


@torch.no_grad()
def test_prefill_memory_drops_straight_into_a_chunk() -> None:
    """The two halves have to agree on width, or the chunk mask will not fit."""
    from cosmos3_nano_sim_auto.impl.packing import build_chunk_metadata
    from cosmos3_nano_sim_auto.impl.transformer.forward import run_chunk_layers

    layers = build_layers()
    _pack, (text_kv, memory_kv) = prefill(layers)

    memory = build_memory_layout(CLIP)
    chunk = build_chunk_metadata(
        CLIP,
        memory,
        chunk_start=CLIP.condition_frames,
        chunk_frames=CLIP.frames_per_chunk,
        text_tokens=TEXT_TOKENS,
    )
    tokens = len(chunk.query)
    position_ids = chunk_mrope_ids(
        CLIP,
        chunk_start=CLIP.condition_frames,
        chunk_frames=CLIP.frames_per_chunk,
        fps=FPS,
        temporal_offset=vision_temporal_offset(TEXT_TOKENS),
    )

    out, chunk_kv = run_chunk_layers(
        layers,
        TINY,
        torch.randn(1, tokens, TINY.hidden_size),
        position_ids,
        text_kv=text_kv,
        memory_kv=memory_kv,
        mask=chunk.mask(),
    )

    assert out.shape == (1, tokens, TINY.hidden_size)
    assert len(chunk_kv) == TINY.num_layers


def control_rows(frames: range) -> list[int]:
    """Rows of a whole-clip control track holding ``frames``, view-outer."""
    assert CLIP.frames_per_view is not None
    spatial, per_view = CLIP.spatial_tokens, CLIP.frames_per_view * CLIP.spatial_tokens
    return [
        view * per_view + frame * spatial + index
        for view in range(CLIP.num_views)
        for frame in frames
        for index in range(spatial)
    ]


def windowed_prefill(layers, *, control_frames: int):
    """A prefill that front-loads only the first ``control_frames`` of control."""
    pack = build_prefill_pack(
        CLIP, text_tokens=TEXT_TOKENS, control_ranges=[(0, control_frames)]
    )
    offset = vision_temporal_offset(TEXT_TOKENS)
    text_ids, _ = text_mrope_ids(TEXT_TOKENS, 0.0)
    control_ids = chunk_mrope_ids(
        CLIP,
        chunk_start=0,
        chunk_frames=control_frames,
        fps=FPS,
        temporal_offset=offset,
    )
    target_ids = chunk_mrope_ids(
        CLIP,
        chunk_start=0,
        chunk_frames=CLIP.condition_frames,
        fps=FPS,
        temporal_offset=offset,
    )

    vision_h = torch.cat(
        [
            CONTROL_H[:, control_rows(range(control_frames))],
            CONDITION_H,
        ],
        dim=1,
    )
    return pack, run_prefill(
        layers,
        TINY,
        text_h=TEXT_H,
        text_position_ids=text_ids,
        vision_h=vision_h,
        vision_position_ids=torch.cat([control_ids, target_ids], dim=1),
        mask=pack.mask(),
        memory_token_indexes=pack.memory_token_indexes,
    )


assert CLIP.frames_per_view is not None
CONTROL_H = torch.randn(
    1,
    CLIP.num_views * CLIP.frames_per_view * CLIP.spatial_tokens,
    TINY.hidden_size,
    generator=torch.Generator().manual_seed(11),
)
"""Embedded control for the whole clip, so a frame is the same either way."""

CONDITION_H = torch.randn(
    1,
    CLIP.num_views * CLIP.condition_frames * CLIP.spatial_tokens,
    TINY.hidden_size,
    generator=torch.Generator().manual_seed(12),
)
TEXT_H = torch.randn(
    1, TEXT_TOKENS, TINY.hidden_size, generator=torch.Generator().manual_seed(8)
)


@torch.no_grad()
def test_streamed_control_gives_the_keys_front_loading_would_have() -> None:
    """The correctness claim behind a bounded cache, checked directly.

    Front-loading the whole track and adding it a chunk at a time have to fill
    the cache with the same keys, or a bounded run is not the same model. It
    holds because control reads only the prompt and control in its own view:
    frames 3 and 4 see frames 0 to 2 either way, and see each other either way.

    Not exact, because the key stream is in a different order -- the top-up puts
    its own frames before the cache, a whole-clip prefill has them after -- and
    a reordered sum of floats is not the same sum.
    """
    layers = build_layers()
    whole = build_prefill_pack(CLIP, text_tokens=TEXT_TOKENS)
    offset = vision_temporal_offset(TEXT_TOKENS)
    text_ids, _ = text_mrope_ids(TEXT_TOKENS, 0.0)
    _text_kv, full_memory = run_prefill(
        layers,
        TINY,
        text_h=TEXT_H,
        text_position_ids=text_ids,
        vision_h=torch.cat([CONTROL_H, CONDITION_H], dim=1),
        vision_position_ids=torch.cat(
            [
                clip_mrope_ids(CLIP, fps=FPS, temporal_offset=offset),
                chunk_mrope_ids(
                    CLIP,
                    chunk_start=0,
                    chunk_frames=CLIP.condition_frames,
                    fps=FPS,
                    temporal_offset=offset,
                ),
            ],
            dim=1,
        ),
        mask=whole.mask(),
        memory_token_indexes=whole.memory_token_indexes,
    )

    window = 3
    _pack, (text_kv, memory_kv) = windowed_prefill(layers, control_frames=window)
    assert CLIP.frames_per_view is not None
    added = range(window, CLIP.frames_per_view)
    metadata = build_chunk_metadata(
        CLIP,
        build_memory_layout(CLIP, control_frame_ranges=[(0, window)]),
        chunk_start=window,
        chunk_frames=len(added),
        text_tokens=TEXT_TOKENS,
        pass_kind="control",
    )
    topped_up = run_control_topup(
        layers,
        TINY,
        control_h=CONTROL_H[:, control_rows(added)],
        control_position_ids=chunk_mrope_ids(
            CLIP,
            chunk_start=window,
            chunk_frames=len(added),
            fps=FPS,
            temporal_offset=offset,
        ),
        text_kv=text_kv,
        memory_kv=memory_kv,
        mask=metadata.mask(),
    )

    rows = torch.tensor(control_rows(added))
    for (topped_k, topped_v), (full_k, full_v) in zip(
        topped_up, full_memory, strict=True
    ):
        assert torch.allclose(topped_k, full_k.index_select(2, rows), atol=1e-5)
        assert torch.allclose(topped_v, full_v.index_select(2, rows), atol=1e-5)


def test_the_prefill_narrows_when_the_scope_does() -> None:
    """The scope has to reach the prefill, not just the chunks after it.

    The cache this pass writes is what every chunk reads, so a prefill left on
    the default while the rollout runs decomposed would poison the run from
    its first frame -- and do it quietly, since both masks are valid.
    """
    pack = build_prefill_pack(
        CLIP, text_tokens=TEXT_TOKENS, target_frames=CLIP.frames_per_view
    )

    wide = pack.mask()
    narrow = pack.mask(scope="decomposed")

    assert int(narrow.sum()) < int(wide.sum())
    assert bool((wide | ~narrow).all()), "decomposed must only ever remove keys"
    assert bool(narrow.any(dim=1).all()), "an all-false row is a silent NaN"
