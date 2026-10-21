# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the gathered folds -- the model's passes with no mask anywhere.

The rectangles are the risk in this module. Whether each is dense, whether they
cover a pass exactly, and whether they double-cover anything are all decided by
:func:`~cosmos3_nano_sim_auto.impl.transformer.fold.build_fold_plan` reading the partition masks, and
every one of those is a silent fault: a rectangle a few keys too wide generates
a clip and reports nothing.

So the tests here are against the block-masked fold in
:mod:`cosmos3_nano_sim_auto.impl.transformer.attention`, which carries the same arithmetic on a
different kernel and so has to agree to rounding. That is a stronger check than
anything written out by hand, and it is the reason the masked fold still exists
after the gathered one replaced it.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from cosmos3_nano_sim_auto.impl.packing import (
    ClipGeometry,
    build_chunk_metadata,
    build_memory_layout,
)
from cosmos3_nano_sim_auto.impl.prefill import build_prefill_pack
from cosmos3_nano_sim_auto.impl.transformer import fold
from cosmos3_nano_sim_auto.impl.transformer.attention import chunk_attention
from cosmos3_nano_sim_auto.impl.transformer.fold import (
    FoldPlan,
    Span,
    build_fold_plan,
    gathered_attention,
)
from cosmos3_nano_sim_auto.impl.transformer.mask import maskless_partitions

pytestmark = pytest.mark.ci_cpu

CLIP = ClipGeometry(
    num_views=2,
    frames_per_view=4,
    patch_h=1,
    patch_w=1,
    frames_per_chunk=1,
    condition_frames=1,
    seconds_per_frame=0.5,
)
HEADS, HEAD_DIM = 4, 8
SCOPES = ("same_view", "decomposed")


def chunk_at(frame: int = 2, views: int = 2, view_text_tokens=None):
    clip = CLIP if views == CLIP.num_views else _with_views(views)
    memory = build_memory_layout(
        clip,
        history_frame_ranges=[(f, f + 1) for f in range(clip.condition_frames, frame)],
    )
    return build_chunk_metadata(
        clip,
        memory,
        chunk_start=frame,
        chunk_frames=1,
        text_tokens=sum(view_text_tokens) if view_text_tokens else 1,
        view_text_tokens=view_text_tokens,
    )


def _with_views(views: int) -> ClipGeometry:
    return ClipGeometry(
        num_views=views,
        frames_per_view=CLIP.frames_per_view,
        patch_h=CLIP.patch_h,
        patch_w=CLIP.patch_w,
        frames_per_chunk=CLIP.frames_per_chunk,
        condition_frames=CLIP.condition_frames,
        seconds_per_frame=CLIP.seconds_per_frame,
    )


def qkv(queries: int, keys: int, *, kv_heads: int = HEADS, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)

    def draw(heads: int, length: int) -> torch.Tensor:
        return torch.randn(1, heads, length, HEAD_DIM, generator=generator)

    return draw(HEADS, queries), draw(kv_heads, keys), draw(kv_heads, keys)


def spread(plan: FoldPlan, index: int, shape) -> torch.Tensor:
    """Paint one pass's rectangles back onto a ``[Q, KV]`` grid."""
    grid = torch.zeros(shape, dtype=torch.bool)
    for group in plan.passes[index][1]:
        queries = torch.cat([torch.arange(s.start, s.stop) for s in group.queries])
        keys = torch.cat([torch.arange(s.start, s.stop) for s in group.keys])
        grid[queries.unsqueeze(1), keys.unsqueeze(0)] = True
    return grid


@pytest.mark.parametrize("scope", SCOPES)
def test_the_rectangles_cover_each_pass_exactly(scope) -> None:
    """Neither a key short nor a key over, which is the whole risk here.

    A rectangle too wide lets a token read what the scope forbids; too narrow
    and it reads less than the checkpoint was trained on. Both generate a
    perfectly plausible clip, so this is checked against the mask the
    rectangles were derived from rather than against a written-out expectation.
    """
    metadata = chunk_at()
    plan = build_fold_plan(metadata.query, metadata.kv, scope=scope)
    partitions = maskless_partitions(metadata.query, metadata.kv, scope=scope)

    assert len(plan.passes) == len(partitions)
    for index, (name, dense) in enumerate(partitions):
        assert plan.passes[index][0] == name
        assert torch.equal(spread(plan, index, dense.shape), dense), name


@pytest.mark.parametrize("scope", SCOPES)
def test_every_rectangle_is_dense(scope) -> None:
    """The property that lets the kernel drop the mask.

    Each group claims every one of its queries reads every one of its keys. If
    that is ever false the kernel has no way to know, because there is no mask
    left to tell it -- it would simply read the extra keys.
    """
    metadata = chunk_at()
    plan = build_fold_plan(metadata.query, metadata.kv, scope=scope)

    for index, (_name, dense) in enumerate(
        maskless_partitions(metadata.query, metadata.kv, scope=scope)
    ):
        for group in plan.passes[index][1]:
            queries = torch.cat([torch.arange(s.start, s.stop) for s in group.queries])
            keys = torch.cat([torch.arange(s.start, s.stop) for s in group.keys])
            assert bool(dense[queries.unsqueeze(1), keys.unsqueeze(0)].all())


@pytest.mark.parametrize("scope", SCOPES)
def test_a_pass_sends_each_query_to_one_group(scope) -> None:
    """Groups partition the queries, which is why a pass needs no merge.

    Outputs are written straight into place. A query in two groups of one pass
    would have the second silently overwrite the first rather than merge with
    it, and the merge across passes would never know.
    """
    metadata = chunk_at()
    plan = build_fold_plan(metadata.query, metadata.kv, scope=scope)

    for _name, groups in plan.passes:
        seen: list[int] = []
        for group in groups:
            for span in group.queries:
                seen.extend(range(span.start, span.stop))
        assert len(seen) == len(set(seen))


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("enable_gqa", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("deduplicate_cross_view", [False, True])
@pytest.mark.parametrize("window", [None, CLIP.seconds_per_frame])
def test_the_gathered_fold_is_the_masked_fold(
    scope, enable_gqa, prefill, deduplicate_cross_view, window
) -> None:
    """Same arithmetic, different kernels, so the outputs must agree.

    Compare the gathered and block-masked kernels with identical pass weighting,
    for both legacy overlap and deduplicated attention. Full-horizon prefill
    metadata exercises cross-view keys as well as control and caption rows.
    """
    metadata = (
        build_prefill_pack(CLIP, text_tokens=1, target_frames=CLIP.frames_per_view)
        if prefill
        else chunk_at()
    )
    settings = dict(
        scope=scope,
        deduplicate_cross_view=deduplicate_cross_view,
        decomposed_temporal_window_seconds=window,
    )
    plan = metadata.fold_plan(**settings)
    kv_heads = HEADS // 2 if enable_gqa else HEADS
    q, k, v = qkv(plan.query_length, plan.key_length, kv_heads=kv_heads)

    gathered = chunk_attention(q, k, v, plan, enable_gqa=enable_gqa)
    masked = chunk_attention(
        q,
        k,
        v,
        metadata.maskless_fold(**settings, block_size=16),
        enable_gqa=enable_gqa,
    )

    torch.testing.assert_close(gathered, masked, atol=1e-5, rtol=1e-5)


def test_one_camera_gathers_to_exactly_its_mask() -> None:
    """With no cross pass to overlap, the fold and the mask are one attention.

    Held against the dense mask rather than the block mask, so that the only
    FlexAttention in the comparison is none: this is the strongest statement in
    the module -- gathered rectangles reproducing an ordinary masked softmax to
    rounding -- and it should not be able to fail because of an unrelated
    kernel.
    """
    metadata = chunk_at(views=1)
    q, k, v = qkv(len(metadata.query), len(metadata.kv))

    gathered = chunk_attention(q, k, v, metadata.fold_plan(scope="decomposed"))
    masked = chunk_attention(q, k, v, metadata.mask(scope="decomposed"))

    torch.testing.assert_close(gathered, masked, atol=1e-5, rtol=1e-5)


def test_the_runs_collapse_so_the_launches_stay_countable() -> None:
    """A launch a frame a camera would cost more than the mask it replaces.

    The planner works on runs of equal metadata and joins spans that abut, so a
    camera's keys arrive as one rectangle rather than one a frame. Pinned
    because nothing else would notice it regressing: the output would stay
    correct and only the frame rate would move.
    """
    plan = build_fold_plan(*_streams(), scope="decomposed")

    assert plan.group_count <= 6, [(name, len(groups)) for name, groups in plan.passes]


def _streams():
    metadata = chunk_at()
    return metadata.query, metadata.kv


def test_a_plan_built_for_another_shape_is_refused() -> None:
    """Cheap to check, and the alternative is reading past a tensor."""
    metadata = chunk_at()
    plan = metadata.fold_plan()
    q, k, v = qkv(len(metadata.query), len(metadata.kv))

    with pytest.raises(ValueError, match="does not match"):
        gathered_attention(q, k[:, :, :-1], v[:, :, :-1], plan)


def test_spans_that_abut_are_one_span() -> None:
    """The join is what keeps the launch count down; assert it rather than it."""
    metadata = chunk_at()
    plan = build_fold_plan(metadata.query, metadata.kv, scope="decomposed")

    for _name, groups in plan.passes:
        for group in groups:
            for earlier, later in zip(group.keys, group.keys[1:], strict=False):
                assert earlier.stop < later.start, (earlier, later)


def test_a_span_knows_its_width() -> None:
    assert len(Span(3, 9)) == 6


@pytest.mark.parametrize("scope", SCOPES)
def test_per_camera_captions_gather_to_the_same_attention_as_the_mask(scope) -> None:
    """Scoping the captions gives the caption pass a group per camera.

    The pass used to be one rectangle every query shared. It is now one per
    camera, and the rectangles are what the fold is: if they were wrong the
    output would still be finite and plausible, so this equality is the only
    cheap way to see it.
    """
    metadata = chunk_at(view_text_tokens=[2, 3])
    q, k, v = qkv(len(metadata.query), len(metadata.kv))

    gathered = chunk_attention(q, k, v, metadata.fold_plan(scope=scope))
    masked = chunk_attention(
        q, k, v, metadata.maskless_fold(scope=scope, block_size=16)
    )

    torch.testing.assert_close(gathered, masked, atol=1e-5, rtol=1e-5)
    caption_groups = dict(metadata.fold_plan(scope=scope).passes)["caption"]
    assert len(caption_groups) == CLIP.num_views


def test_the_rectangles_cover_each_windowed_pass_exactly() -> None:
    """The cross-view window changes which keys a pass reads, not the folding.

    And it does change them: one frame back (0.5 s in this clip) the window
    reaches the other camera's history that the same-instant pass never reads.
    """
    metadata = chunk_at()
    window: dict[str, Any] = {
        "decomposed_temporal_window_seconds": CLIP.seconds_per_frame
    }
    plan = build_fold_plan(metadata.query, metadata.kv, scope="decomposed", **window)
    partitions = maskless_partitions(
        metadata.query, metadata.kv, scope="decomposed", **window
    )

    assert len(plan.passes) == len(partitions)
    for index, (name, dense) in enumerate(partitions):
        assert torch.equal(spread(plan, index, dense.shape), dense), name

    instant = dict(maskless_partitions(metadata.query, metadata.kv, scope="decomposed"))
    assert bool((dict(partitions)["cross_instant"] & ~instant["cross_instant"]).any())


def _equal_shape_bucket(
    plan: FoldPlan, pass_index: int
) -> tuple[fold._Bucket, list[int]]:
    """The first set of two or more groups of a pass with the same query and key count."""
    groups = plan.passes[pass_index][1]
    by_shape: dict[tuple[int, int], list[int]] = {}
    for number, group in enumerate(groups):
        by_shape.setdefault((group.query_count, group.key_count), []).append(number)
    numbers = next(n for n in by_shape.values() if len(n) >= 2)
    members = [groups[n] for n in numbers]
    device = torch.device("cpu")
    query_index = torch.cat([fold._span_index(g.queries, device) for g in members])
    key_index = torch.cat([fold._span_index(g.keys, device) for g in members])
    bucket = fold._Bucket(
        len(numbers),
        members[0].query_count,
        members[0].key_count,
        query_index,
        key_index,
    )
    return bucket, numbers


def test_the_batched_pass_is_the_per_group_pass() -> None:
    """One launch for equal-shape groups must give each group what its own launch gives."""
    metadata = chunk_at()
    plan = metadata.fold_plan(scope="same_view")
    q, k, v = qkv(len(metadata.query), len(metadata.kv))
    bucket, numbers = _equal_shape_bucket(plan, 0)
    out, lse = fold._batched_pass(q, k, v, bucket)
    at = 0
    for number in numbers:
        group = plan.passes[0][1][number]
        expected, expected_lse = fold._dense_attention(
            fold._take(q, group.queries),
            fold._take(k, group.keys),
            fold._take(v, group.keys),
        )
        width = group.query_count
        torch.testing.assert_close(out[:, :, at : at + width], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            lse[:, :, at : at + width], expected_lse, rtol=0, atol=0
        )
        at += width


@pytest.mark.parametrize("window", [None, 0.0, 0.5, 2.0])
@torch.no_grad()
def test_deduplicated_fold_matches_union_attention(window):
    from cosmos3_nano_sim_auto.impl.transformer.mask import ROLE_UND, visibility

    metadata = chunk_at(frame=3, views=3, view_text_tokens=(1, 2, 3))
    plan = build_fold_plan(
        metadata.query,
        metadata.kv,
        decomposed_temporal_window_seconds=window,
        deduplicate_cross_view=True,
    )
    union = visibility(
        metadata.query,
        metadata.kv,
        scope="decomposed",
        decomposed_temporal_window_seconds=window,
    )
    # Maskless replay limits control queries to their own sensor's clean history.
    union &= (
        ~metadata.query.is_control[:, None]
        | (metadata.query.view_id[:, None] == metadata.kv.view_id[None, :])
        | (metadata.kv.token_role_id[None, :] == ROLE_UND)
    )
    coverage = sum(spread(plan, i, union.shape).int() for i in range(len(plan.passes)))
    assert isinstance(coverage, torch.Tensor)
    assert torch.equal(
        coverage, union.int()
    )  # Every allowed edge appears exactly once.
    q, k, v = qkv(len(metadata.query), len(metadata.kv), kv_heads=2)
    actual = chunk_attention(q, k, v, plan, enable_gqa=True)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=union, enable_gqa=True
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
