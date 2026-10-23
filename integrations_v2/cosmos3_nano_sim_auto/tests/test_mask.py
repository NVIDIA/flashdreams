# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for multiview attention visibility.

Hand-built metadata isolates visibility rules from token layout construction.
"""

from __future__ import annotations

import pytest
import torch
from cosmos3_nano_sim_auto.impl.transformer.mask import (
    LIDAR_VIEW_ID,
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_PADDING,
    ROLE_TARGET_CONDITION,
    ROLE_UND,
    StreamFields,
    _compiled_create_block_mask,
    build_block_mask,
    maskless_partitions,
    visibility,
    visibility_mask_mod,
)

pytestmark = pytest.mark.ci_cpu

SECONDS_PER_FRAME = 0.5


def build_stream(
    tokens: list[tuple[int, int, int, int]], sample: int = 0
) -> StreamFields:
    """Build a stream from ``(role, frame, view, causal_step)`` rows.

    ``is_noisy`` follows the role the way the packer will set it: current targets
    are the noisy RGB being denoised, everything else is clean or conditioning.
    Timestamps derive from the frame index at a fixed rate, matching the model's
    ``seconds_per_frame``.
    """
    roles = torch.tensor([t[0] for t in tokens])
    frames = torch.tensor([t[1] for t in tokens])
    views = torch.tensor([t[2] for t in tokens])
    steps = torch.tensor([t[3] for t in tokens])
    return StreamFields(
        sample_id=torch.full((len(tokens),), sample, dtype=torch.long),
        frame_id=frames,
        view_id=views,
        is_noisy=roles == ROLE_CURRENT_TARGET,
        is_control=roles == ROLE_CONTROL,
        timestamp=frames.to(torch.float32) * SECONDS_PER_FRAME,
        token_role_id=roles,
        causal_step_id=steps,
    )


# One two-view, three-frame sample at one frame per chunk, so causal_step == frame.
# Mirrors the model's `_multiview_transfer_items`.
def _sample_tokens() -> list[tuple[int, int, int, int]]:
    tokens: list[tuple[int, int, int, int]] = [(ROLE_UND, -1, -1, 0)]
    for frame in range(3):
        for view in range(2):
            tokens.append((ROLE_CONTROL, frame, view, frame))
            tokens.append((ROLE_CURRENT_TARGET, frame, view, frame))
            tokens.append((ROLE_CLEAN_TARGET, frame, view, frame))
    for view in range(2):
        tokens.append((ROLE_TARGET_CONDITION, 0, view, 0))
    return tokens


def index_of(
    tokens: list[tuple[int, int, int, int]], role: int, frame: int, view: int
) -> int:
    matches = [
        i
        for i, t in enumerate(tokens)
        if t[0] == role and t[1] == frame and t[2] == view
    ]
    assert len(matches) == 1, (
        f"expected exactly one {role=} {frame=} {view=}, got {matches}"
    )
    return matches[0]


@pytest.fixture
def tokens() -> list[tuple[int, int, int, int]]:
    return _sample_tokens()


def dense_from_mask_mod(stream: StreamFields, **kwargs) -> torch.Tensor:
    """Expand the FlexAttention predicate back to a ``[Q, KV]`` tensor.

    Only a test does this -- expanding is the cost the predicate exists to
    avoid -- but it is the one way to compare the two against each other.
    """
    flex = pytest.importorskip(
        "torch.nn.attention.flex_attention", reason="needs FlexAttention"
    )
    mod = visibility_mask_mod(stream, stream, **kwargs)
    tokens = len(stream)
    expanded = flex.create_mask(
        mod, B=None, H=None, Q_LEN=tokens, KV_LEN=tokens, device=stream.sample_id.device
    )
    return expanded[0, 0]


@pytest.mark.parametrize("pattern", ["causal", "current_control"])
@pytest.mark.parametrize("scope", ["same_view", "decomposed", "all_views"])
def test_the_mask_mod_says_exactly_what_the_dense_mask_says(
    tokens, pattern, scope
) -> None:
    """The FlexAttention predicate and the dense tensor are one set of rules.

    They are written twice, in the same file and the same order, because the
    kernel wants one token pair at a time and cuDNN wants the matrix. Nothing
    but this test stops the two from drifting, and a drift would not raise --
    it would quietly change what the model can see.
    """
    stream = build_stream(tokens)
    dense = visibility(stream, stream, pattern=pattern, scope=scope)

    assert torch.equal(dense_from_mask_mod(stream, pattern=pattern, scope=scope), dense)


def test_the_mask_mod_agrees_with_the_dense_mask_on_padding() -> None:
    """Padding is the one rule that ignores the others, so it is checked apart.

    A padded key is not merely invisible: pad attends pad, or a query whose row
    is entirely false makes softmax produce NaN rather than zero.
    """
    tokens = [
        (ROLE_UND, -1, -1, 0),
        (ROLE_CURRENT_TARGET, 0, 0, 0),
        (ROLE_PADDING, -1, -1, -1),
    ]
    stream = build_stream(tokens)

    assert torch.equal(dense_from_mask_mod(stream), visibility(stream, stream))


@pytest.mark.parametrize(
    ("pattern", "sees_control_history"), [("causal", True), ("current_control", False)]
)
def test_current_target_visibility(tokens, pattern, sees_control_history) -> None:
    """Current targets read same-step noise and strictly earlier clean history.

    Mirrors the model's ``test_teacher_forcing_mask_matches_replay_visibility``.
    """
    stream = build_stream(tokens)
    mask = visibility(stream, stream, pattern=pattern, scope="all_views")
    q = index_of(tokens, ROLE_CURRENT_TARGET, frame=2, view=0)

    assert mask[q, 0].item() is True  # the understanding token
    assert mask[q, index_of(tokens, ROLE_CONTROL, 1, 0)].item() is sees_control_history
    assert mask[q, index_of(tokens, ROLE_CONTROL, 2, 0)].item() is True
    assert mask[q, index_of(tokens, ROLE_CURRENT_TARGET, 2, 1)].item() is True
    assert mask[q, index_of(tokens, ROLE_CURRENT_TARGET, 1, 0)].item() is False
    assert mask[q, index_of(tokens, ROLE_CLEAN_TARGET, 1, 1)].item() is True
    assert mask[q, index_of(tokens, ROLE_CLEAN_TARGET, 2, 0)].item() is False


def test_control_queries_are_causal_and_view_local(tokens) -> None:
    """A control token sees its own view's control history, never other views or its own step."""
    stream = build_stream(tokens)
    mask = visibility(stream, stream, pattern="causal", scope="all_views")
    q = index_of(tokens, ROLE_CONTROL, frame=1, view=0)

    assert mask[q, index_of(tokens, ROLE_CONTROL, 0, 0)].item() is True
    assert mask[q, index_of(tokens, ROLE_CONTROL, 1, 0)].item() is True
    assert mask[q, index_of(tokens, ROLE_CONTROL, 2, 0)].item() is False
    assert mask[q, index_of(tokens, ROLE_CONTROL, 1, 1)].item() is False


def test_target_condition_reads_noisy_rgb_but_not_clean_memory(tokens) -> None:
    """Mirrors the model's ``test_conditioning_rgb_attends_current_noisy_rgb_but_not_clean_replay_memory``."""
    stream = build_stream(tokens)
    mask = visibility(stream, stream, pattern="causal", scope="all_views")
    q = index_of(tokens, ROLE_TARGET_CONDITION, frame=0, view=0)

    assert mask[q, index_of(tokens, ROLE_CURRENT_TARGET, 2, 1)].item() is True
    assert mask[q, index_of(tokens, ROLE_CLEAN_TARGET, 1, 1)].item() is False


@pytest.mark.parametrize(
    ("scope", "sees_same_frame_other_view", "sees_other_view_same_step"),
    [
        ("same_view", False, False),
        ("decomposed", True, False),
        ("all_views", True, True),
    ],
)
def test_scope_restricts_cross_view_relations(
    scope, sees_same_frame_other_view, sees_other_view_same_step
) -> None:
    """Replay causality intersects with the configured view scope.

    Mirrors the model's ``test_teacher_forcing_scope_restricts_current_and_clean_target_relations``.
    Two frames per chunk here, so ``causal_step = frame // 2`` and frames 2 and 3
    share a step while frame 1 sits in the one before.
    """
    tokens: list[tuple[int, int, int, int]] = [(ROLE_UND, -1, -1, 0)]
    for frame in range(4):
        for view in range(2):
            tokens.append((ROLE_CURRENT_TARGET, frame, view, frame // 2))
            tokens.append((ROLE_CLEAN_TARGET, frame, view, frame // 2))
    stream = build_stream(tokens)
    mask = visibility(stream, stream, pattern="causal", scope=scope)

    q = index_of(tokens, ROLE_CURRENT_TARGET, frame=2, view=0)
    same_frame_other_view = index_of(tokens, ROLE_CURRENT_TARGET, 3, 1)
    other_view_earlier_step = index_of(tokens, ROLE_CURRENT_TARGET, 1, 1)

    # Frame 3 view 1 shares the causal step with the query but is a different
    # frame and view, so only all_views admits it; decomposed needs same frame.
    assert mask[q, same_frame_other_view].item() is sees_other_view_same_step
    assert mask[q, index_of(tokens, ROLE_CURRENT_TARGET, 2, 1)].item() is (
        sees_same_frame_other_view or sees_other_view_same_step
    )
    assert mask[q, other_view_earlier_step].item() is False
    assert mask[q, index_of(tokens, ROLE_CURRENT_TARGET, 2, 0)].item() is True


# Not covered here: the timestamp-window branch of ``decomposed`` scope. It only
# becomes observable when replayed clean history sits at a strictly lower causal
# step than the current chunk, and that step assignment comes from the memory
# layout rather than from the frame index -- hand-writing it here would be
# inventing the packer's output and testing the invention. It is covered against
# real metadata in ``test_packing.py``.


def test_cross_sample_attention_is_blocked(tokens) -> None:
    """Two packed samples never see each other, whatever their roles say."""
    assert not visibility(
        build_stream(tokens, sample=0),
        build_stream(tokens, sample=1),
        scope="all_views",
    ).any()


def test_padding_attends_only_padding() -> None:
    """Padding rows must stay valid so softmax never sees an all-false row."""
    tokens = [
        (ROLE_PADDING, 0, 0, 0),
        (ROLE_CURRENT_TARGET, 0, 0, 0),
        (ROLE_PADDING, 0, 0, 0),
    ]
    stream = build_stream(tokens)
    mask = visibility(stream, stream, scope="all_views")
    assert mask[0].tolist() == [True, False, True]
    assert mask[1, 0].item() is False


def test_the_block_mask_builder_stays_compiled() -> None:
    """Correctness does not depend on this. Memory does, by about 9x.

    Eager ``create_block_mask`` calls ``create_mask``, which materializes the
    whole ``[Q, KV]`` bool and runs the predicate's dozen boolean terms over it
    -- the cost the block mask exists to avoid. Measured at four views: 7.2 GiB
    eager against 0.77 GiB compiled, above a 0.66 GiB dense mask.

    Dropping the ``torch.compile`` would therefore pass every other test in this
    file and every parity test, and quietly cost tens of GiB at eleven views.
    Hence a test that asserts nothing about behaviour.

    ``_torchdynamo_orig_callable`` is private, and the public-looking
    alternative does not apply: ``torch.compile`` returns an ``OptimizedModule``
    only for an ``nn.Module``, and ``create_block_mask`` is a function, so the
    wrapper's type is plain ``function``. If a future release renames the
    attribute this test fails rather than passing quietly, which is the
    direction to fail in -- it would be read as the compile having gone.
    """
    assert hasattr(_compiled_create_block_mask(), "_torchdynamo_orig_callable")


def test_the_block_mask_builder_compiles_once() -> None:
    """A fresh predicate closure per call must not mean a fresh compile.

    The rollout builds one per chunk per branch, so recompiling each time would
    cost more than the kernel saves.
    """
    assert _compiled_create_block_mask() is _compiled_create_block_mask()


def test_the_block_mask_builder_preserves_asymmetric_geometry(tokens) -> None:
    """Query and key tiles may differ without changing the visibility predicate."""
    stream = build_stream(tokens)

    mask = build_block_mask(stream, stream, block_size=(16, 32))

    assert mask.BLOCK_SIZE == (16, 32)
    assert mask.shape == (1, 1, len(stream), len(stream))


def _union(partitions) -> torch.Tensor:
    merged = torch.zeros_like(partitions[0][1])
    for _name, part in partitions:
        merged = merged | part
    return merged


def test_the_folds_cover_exactly_what_the_mask_allows(tokens) -> None:
    """The passes are a rearrangement of the mask's keys, not a second rule set.

    This is the property that makes the fold safe to reason about: whatever the
    merge does with weights, no pass may reach a key the scope forbids, and none
    of the scope's keys may go unread. Checked against `visibility` rather than
    a written-out expectation so the two cannot drift apart.
    """
    stream = build_stream(tokens)

    for scope in ("same_view", "decomposed"):
        folded = _union(maskless_partitions(stream, stream, scope=scope))
        masked = visibility(stream, stream, scope=scope)

        assert torch.equal(folded, masked), scope


def test_only_the_decomposed_fold_overlaps_and_only_on_the_shared_cell(
    tokens,
) -> None:
    """Where the fold stops being the mask, and exactly how far.

    `same_view` is a partition, so its passes are disjoint and the merge is
    exact. `decomposed` is two overlapping partitions, and every pair counted
    twice has to be a sensor pair at both the query's own view and its own
    instant -- that set is the whole of the difference from the mask, and the
    reason the two measure 12.8% apart.
    """
    stream = build_stream(tokens)

    exact = maskless_partitions(stream, stream, scope="same_view")
    for left in range(len(exact)):
        for right in range(left + 1, len(exact)):
            shared = exact[left][1] & exact[right][1]
            assert not shared.any(), (exact[left][0], exact[right][0])

    passes = dict(maskless_partitions(stream, stream, scope="decomposed"))
    overlap = passes["same_view"] & passes["cross_instant"]

    assert overlap.any(), "decomposed with no overlap is same_view by another name"
    same_view = stream.view_id.unsqueeze(1) == stream.view_id.unsqueeze(0)
    same_frame = stream.frame_id.unsqueeze(1) == stream.frame_id.unsqueeze(0)
    sensor = ~stream.is_control.unsqueeze(1) & ~stream.is_control.unsqueeze(0)
    # Captions are read once, by their own pass, so they are not in the overlap
    # however their view and frame compare.
    caption = (stream.token_role_id == ROLE_UND).unsqueeze(0)
    assert torch.equal(
        overlap,
        visibility(stream, stream, scope="decomposed")
        & same_view
        & same_frame
        & sensor
        & ~caption,
    )
    assert not (overlap & passes["caption"]).any(), "captions are never double-read"


def test_every_row_the_fold_keeps_has_a_key_in_some_pass(tokens) -> None:
    """A row with no keys anywhere is a -inf merge, which is a NaN frame.

    Padding is the case that nearly slips: it is excluded from the cross-instant
    pass and reaches no caption, so it rides with the same-view pass or it
    reaches nothing at all.
    """
    stream = build_stream(tokens)

    for scope in ("same_view", "decomposed"):
        folded = _union(maskless_partitions(stream, stream, scope=scope))
        rows = visibility(stream, stream, scope=scope).any(dim=1)

        assert torch.equal(folded.any(dim=1), rows), scope


def test_the_fold_refuses_the_scope_it_cannot_express(tokens) -> None:
    """`all_views` has no partition, and guessing one would be a silent change."""
    stream = build_stream(tokens)

    with pytest.raises(ValueError, match="all_views"):
        maskless_partitions(stream, stream, scope="all_views")


def _one_view_tokens() -> list[tuple[int, int, int, int]]:
    """The same sample as :func:`_sample_tokens`, with the second camera gone."""
    return [
        token for token in _sample_tokens() if token[2] == 0 or token[0] == ROLE_UND
    ]


def test_one_camera_builds_no_cross_pass_and_so_folds_exactly() -> None:
    """the reference implementation gates its cross pass on the rig having a second camera, and so must this.

    With one view, "every view at my instant" is a subset of "my view at every
    frame": the pass adds no key and merges the ones it repeats at twice their
    weight. That is not the fold of the mask, it is a reweighting of it, and the reference implementation
    builds no such pass -- so a one-view run has to come out identical to its
    mask, which is also what makes the one-view timing worth comparing.
    """
    stream = build_stream(_one_view_tokens())

    names = [name for name, _part in maskless_partitions(stream, stream)]

    assert names == ["same_view", "caption"], names
    folded = _union(maskless_partitions(stream, stream, scope="decomposed"))
    assert torch.equal(folded, visibility(stream, stream, scope="decomposed"))


def test_two_cameras_still_build_the_cross_pass() -> None:
    """The guard above must key on the rig, not remove the pass outright."""
    stream = build_stream(_sample_tokens())

    names = [name for name, _part in maskless_partitions(stream, stream)]

    assert names == ["same_view", "cross_instant", "caption"], names


def _per_view_caption_tokens() -> list[tuple[int, int, int, int]]:
    """:func:`_sample_tokens` with one caption per camera instead of one for the rig.

    The layout the delivered checkpoint was trained with: each camera's caption
    is tokenized on its own and carries that camera's view id, where a caption
    covering the whole rig carries the ``-1`` sentinel.
    """
    tokens = [(ROLE_UND, -1, view, 0) for view in range(2)]
    return tokens + [token for token in _sample_tokens() if token[0] != ROLE_UND]


def test_a_camera_reads_its_own_caption_and_not_the_other_camera_s() -> None:
    """The rule the delivered checkpoint needs, and the whole point of the change.

    Under the merged layout every camera read every caption, which is what the
    previous checkpoint was trained with. Reading another camera's caption here
    is reading a description of a scene pointing the other way.
    """
    tokens = _per_view_caption_tokens()
    stream = build_stream(tokens)
    mask = visibility(stream, stream, scope="all_views")

    own, other = 0, 1
    for role in (ROLE_CURRENT_TARGET, ROLE_CLEAN_TARGET, ROLE_CONTROL):
        q = index_of(tokens, role, frame=2 if role != ROLE_CONTROL else 1, view=0)
        assert mask[q, own].item() is True, role
        assert mask[q, other].item() is False, role

    condition = index_of(tokens, ROLE_TARGET_CONDITION, frame=0, view=1)
    assert mask[condition, other].item() is True
    assert mask[condition, own].item() is False


def test_a_caption_written_for_the_whole_rig_is_still_read_by_every_camera() -> None:
    """The older layout has to keep working, since the two are worth comparing.

    A rig-level caption carries no view, and the rule turns on the caption's
    own view rather than on a flag, so the same code serves both layouts.
    """
    tokens = _sample_tokens()
    stream = build_stream(tokens)
    mask = visibility(stream, stream, scope="all_views")

    caption = index_of(tokens, ROLE_UND, frame=-1, view=-1)
    for view in range(2):
        q = index_of(tokens, ROLE_CURRENT_TARGET, frame=2, view=view)
        assert mask[q, caption].item() is True, view


def test_the_mask_mod_scopes_captions_the_way_the_dense_mask_does() -> None:
    """Two statements of the caption rule, checked against each other."""
    stream = build_stream(_per_view_caption_tokens())

    assert torch.equal(
        dense_from_mask_mod(stream, scope="all_views"),
        visibility(stream, stream, scope="all_views"),
    )


def test_the_fold_keys_each_camera_against_its_own_caption() -> None:
    """The fold is the path we run, so scoping the mask alone would change nothing.

    Its caption partition is derived from :func:`visibility`, so narrowing the
    rule there narrows this too. What is worth pinning is that the result is
    still a partition: no camera reads a caption twice, and every row that has
    a key keeps one.
    """
    stream = build_stream(_per_view_caption_tokens())

    for scope in ("same_view", "decomposed"):
        passes = maskless_partitions(stream, stream, scope=scope)
        for left in range(len(passes)):
            for right in range(left + 1, len(passes)):
                if scope == "decomposed" and {passes[left][0], passes[right][0]} == {
                    "same_view",
                    "cross_instant",
                }:
                    continue
                shared = passes[left][1] & passes[right][1]
                assert not shared.any(), (scope, passes[left][0], passes[right][0])

        folded = _union(passes)
        rows = visibility(stream, stream, scope=scope)
        assert torch.equal(folded.any(dim=1), rows.any(dim=1)), scope


# 0.4 s is the model's window; at this fixture's 0.5 s a frame it stays inside one
# instant, so 1.0 s reaches earlier frames the way 0.4 s does at 30 fps.
@pytest.mark.parametrize("window", [0.4, 2 * SECONDS_PER_FRAME])
@pytest.mark.parametrize("joint", [False, True])
def test_windowed_fold_preserves_targets_and_keeps_control_history_in_own_view(
    tokens, window, joint
) -> None:
    """Upstream maskless replay excludes control queries from its cross pass."""
    if joint:
        tokens = tokens + [
            (role, frame, LIDAR_VIEW_ID, step)
            for role, frame, view, step in tokens
            if view == 0
        ]
    stream = build_stream(tokens)
    passes = dict(
        maskless_partitions(
            stream,
            stream,
            scope="decomposed",
            decomposed_temporal_window_seconds=window,
            joint=joint,
        )
    )
    masked = visibility(
        stream,
        stream,
        scope="decomposed",
        decomposed_temporal_window_seconds=window,
        joint=joint,
    )
    folded = _union(tuple(passes.items()))
    controls = stream.is_control
    assert torch.equal(folded[~controls], masked[~controls])
    assert not passes["cross_instant"][controls].any()
    own_view = stream.view_id[:, None] == stream.view_id[None, :]
    clean = (stream.token_role_id == ROLE_TARGET_CONDITION) | (
        stream.token_role_id == ROLE_CLEAN_TARGET
    )
    earlier = stream.causal_step_id[None, :] < stream.causal_step_id[:, None]
    expected = own_view & clean[None, :] & earlier
    assert expected[controls].any()
    assert torch.equal(
        passes["same_view"][controls] & clean[None, :], expected[controls]
    )
