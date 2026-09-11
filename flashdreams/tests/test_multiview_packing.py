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

"""CPU tests for multi-view token packing."""

from __future__ import annotations

import pytest
import torch

from flashdreams.core.attention.multiview.mask import (
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_TARGET_CONDITION,
    ROLE_UND,
    StreamFields,
)
from flashdreams.core.attention.multiview.packing import (
    ClipGeometry,
    ar_chunk_plan,
    ar_chunk_range,
    ar_chunk_ranges,
    build_chunk_metadata,
    build_memory_layout,
    causal_steps,
    pack_cross_view_attention,
    unpack_cross_view_attention,
)

pytestmark = pytest.mark.ci_cpu

# Two views, three latent frames at 2 Hz, one spatial token per frame,
# one frame per chunk, and frame 0 supplied.
CLIP = ClipGeometry(
    num_views=2,
    frames_per_view=3,
    patch_h=1,
    patch_w=1,
    frames_per_chunk=1,
    condition_frames=1,
    seconds_per_frame=0.5,
)


@pytest.mark.parametrize(
    "seconds_per_frame", [0.0, -1.0, float("nan"), float("inf"), float("-inf")]
)
def test_clip_geometry_rejects_nonpositive_or_nonfinite_timing(
    seconds_per_frame: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ClipGeometry(
            num_views=2,
            frames_per_view=3,
            patch_h=1,
            patch_w=1,
            frames_per_chunk=1,
            condition_frames=1,
            seconds_per_frame=seconds_per_frame,
        )


def test_pack_cross_view_attention_shares_all_views_at_each_frame() -> None:
    """Give each view its own queries and the same rig-wide context."""
    tokens = torch.arange(1 * 4 * 2 * 3).reshape(1, 4, 2, 3, 1)

    query, context = pack_cross_view_attention(tokens)

    assert query.shape == (1, 2, 4, 3, 1)
    assert context.shape == (1, 2, 4, 12, 1)
    for frame in range(2):
        expected = tokens[:, :, frame].reshape(1, 12, 1)
        for view in range(4):
            assert torch.equal(query[:, frame, view], tokens[:, view, frame])
            assert torch.equal(context[:, frame, view], expected)


def test_pack_cross_view_attention_validates_layout() -> None:
    """Refuse tensors that cannot describe a non-empty five-axis clip."""
    with pytest.raises(ValueError, match=r"\[B, V, T, S, D\]"):
        pack_cross_view_attention(torch.zeros(1, 2, 3, 4))
    with pytest.raises(ValueError, match="empty axis"):
        pack_cross_view_attention(torch.zeros(1, 4, 0, 3, 8))


def test_cross_view_attention_layout_round_trips_to_view_major_tokens() -> None:
    """Restore attention output to the layout expected by a multi-view block."""
    tokens = torch.arange(2 * 4 * 3 * 5 * 7).reshape(2, 4, 3, 5, 7)

    query, _context = pack_cross_view_attention(tokens)

    assert torch.equal(
        unpack_cross_view_attention(query),
        tokens.reshape(2, 4, 15, 7),
    )


def test_unpack_cross_view_attention_validates_layout() -> None:
    """Refuse tensors that cannot be per-frame, per-view attention output."""
    with pytest.raises(ValueError, match=r"\[B, T, V, S, D\]"):
        unpack_cross_view_attention(torch.zeros(1, 2, 3, 4))
    with pytest.raises(ValueError, match="empty axis"):
        unpack_cross_view_attention(torch.zeros(1, 0, 4, 3, 8))


def index_of(stream: StreamFields, role: int, frame: int, view: int) -> int:
    matches = (
        (
            (stream.token_role_id == role)
            & (stream.frame_id == frame)
            & (stream.view_id == view)
        )
        .nonzero()
        .flatten()
    )
    assert matches.numel() == 1, (
        f"expected exactly one {role=} {frame=} {view=}, got {matches.tolist()}"
    )
    return int(matches[0])


def rolled_out_chunk(**kwargs):
    """Memory holding frame 1, and the chunk generating frame 2."""
    memory = build_memory_layout(CLIP, history_frame_ranges=[(1, 2)])
    chunk = build_chunk_metadata(
        CLIP, memory, chunk_start=2, chunk_frames=1, text_tokens=1, **kwargs
    )
    return memory, chunk


def test_causal_partition_makes_frame_zero_a_singleton() -> None:
    """The ``[1, C, C, ...]`` partition, and the ``frame // C`` answer it is not."""
    frames = torch.tensor([-1, 0, 1, 2, 3, 4, 5])

    assert causal_steps(frames, 2).tolist() == [-1, 0, 1, 1, 2, 2, 3]
    assert causal_steps(frames, 1).tolist() == [-1, 0, 1, 2, 3, 4, 5]
    # The singleton at the front pushes every later block one frame along, so the
    # obvious reading disagrees from frame 2 on.
    assert causal_steps(frames, 2).tolist() != (frames // 2).tolist()


def test_memory_orders_controls_then_conditions_then_history() -> None:
    """Keep controls, conditions, and generated chunks in cache order."""
    memory = build_memory_layout(CLIP, history_frame_ranges=[(1, 2)], capacity=128)
    stream = memory.stream

    assert memory.real_token_count == 10
    assert memory.capacity == 128
    assert stream.token_role_id[:6].tolist() == [ROLE_CONTROL] * 6
    assert stream.token_role_id[6:8].tolist() == [ROLE_TARGET_CONDITION] * 2
    assert stream.token_role_id[8:10].tolist() == [ROLE_CLEAN_TARGET] * 2
    assert stream.frame_id[8:10].tolist() == [1, 1]
    assert stream.view_id[8:10].tolist() == [0, 1]
    assert stream.timestamp[8:10].tolist() == [0.5, 0.5]
    # Controls run view-outer, frame-inner -- not frame-outer.
    assert stream.frame_id[:6].tolist() == [0, 1, 2, 0, 1, 2]
    assert stream.view_id[:6].tolist() == [0, 0, 0, 1, 1, 1]
    # Nothing cached is noisy, and only the controls are control.
    assert not stream.is_noisy[: memory.real_token_count].any()
    assert (
        stream.is_control[: memory.real_token_count].tolist()
        == [True] * 6 + [False] * 4
    )


def test_completed_chunk_appends_without_disturbing_the_prefix() -> None:
    """Each chunk lands at the previous ``real_token_count``, which is the write offset."""
    before = build_memory_layout(CLIP)
    after = build_memory_layout(CLIP, history_frame_ranges=[(1, 2)])

    assert before.real_token_count == 8
    assert after.real_token_count == 10
    assert (
        after.stream.token_role_id[:8].tolist() == before.stream.token_role_id.tolist()
    )
    assert after.stream.frame_id[:8].tolist() == before.stream.frame_id.tolist()


WIDE = ClipGeometry(
    num_views=2,
    frames_per_view=5,
    patch_h=1,
    patch_w=1,
    frames_per_chunk=1,
    condition_frames=1,
    seconds_per_frame=0.5,
)
"""``CLIP``, long enough to drop a chunk off the front and still hold two."""


def test_history_may_start_after_the_conditioning_frames() -> None:
    """A bounded cache drops its oldest chunk, so history need not start at the front."""
    memory = build_memory_layout(CLIP, history_frame_ranges=[(2, 3)])

    assert (memory.stream.token_role_id == ROLE_CLEAN_TARGET).sum().item() == (
        CLIP.num_views * CLIP.spatial_tokens
    )


def test_history_ranges_may_not_leave_a_gap() -> None:
    """Frames missing from the middle would be attended across with nothing saying so."""
    with pytest.raises(ValueError, match="gap"):
        build_memory_layout(WIDE, history_frame_ranges=[(1, 2), (3, 4)])


def test_history_ranges_may_not_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        build_memory_layout(WIDE, history_frame_ranges=[(1, 3), (2, 4)])


def test_history_ranges_need_not_be_in_chronological_order() -> None:
    """Once the cache reuses its oldest slot it holds chunks rotated, not in order.

    Attention does not care, as long as the metadata follows the order the
    cache is actually in -- which is why this is allowed rather than sorted.
    """
    rotated = build_memory_layout(WIDE, history_frame_ranges=[(3, 4), (1, 3)])
    per_frame = WIDE.num_views * WIDE.spatial_tokens

    history = rotated.stream.frame_id[rotated.stream.token_role_id == ROLE_CLEAN_TARGET]
    assert history[:per_frame].tolist() == [3] * per_frame


def test_chunk_rejects_generating_over_conditioning_frames() -> None:
    memory = build_memory_layout(CLIP)
    with pytest.raises(ValueError, match="conditioning frames"):
        build_chunk_metadata(CLIP, memory, chunk_start=0, chunk_frames=1, text_tokens=1)


def test_chunk_numbers_its_frames_globally() -> None:
    """A chunk's frames are numbered where they fall in the clip, not from zero."""
    _memory, chunk = rolled_out_chunk()

    assert chunk.query.frame_id.tolist() == [2, 2]
    assert chunk.query.view_id.tolist() == [0, 1]
    assert chunk.query.timestamp.tolist() == [1.0, 1.0]
    assert chunk.query.causal_step_id.tolist() == [2, 2]
    assert chunk.query.token_role_id.tolist() == [ROLE_CURRENT_TARGET] * 2
    assert chunk.query.is_noisy.all()
    # [text | current chunk | memory], and the chunk carries no controls of its own.
    assert chunk.num_und == 1
    assert chunk.memory_offset == 3
    assert chunk.kv.token_role_id[0].item() == ROLE_UND
    assert len(chunk.kv) == 1 + 2 + 10
    assert not chunk.query.is_control.any()


def test_chunk_reads_control_up_to_its_own_step_in_its_own_view() -> None:
    """Controls are causal and view-local even though all of them are cached."""
    clip = ClipGeometry(
        num_views=2,
        frames_per_view=4,
        patch_h=1,
        patch_w=1,
        frames_per_chunk=1,
        condition_frames=1,
        seconds_per_frame=0.5,
    )
    memory = build_memory_layout(clip, history_frame_ranges=[(1, 2)])
    chunk = build_chunk_metadata(
        clip, memory, chunk_start=2, chunk_frames=1, text_tokens=1
    )
    mask = chunk.mask()
    q = index_of(chunk.query, ROLE_CURRENT_TARGET, frame=2, view=0)

    def sees(frame: int, view: int) -> bool:
        return bool(mask[q, index_of(chunk.kv, ROLE_CONTROL, frame, view)].item())

    assert sees(frame=1, view=0) is True
    assert sees(frame=2, view=0) is True
    assert sees(frame=3, view=0) is False
    assert sees(frame=2, view=1) is False


def test_chunk_reads_the_prompt_and_its_own_view_partners() -> None:
    _memory, chunk = rolled_out_chunk()
    mask = chunk.mask()
    q = index_of(chunk.query, ROLE_CURRENT_TARGET, frame=2, view=0)

    assert mask[q, 0].item() is True  # the understanding token
    assert mask[q, index_of(chunk.kv, ROLE_CURRENT_TARGET, 2, 1)].item() is True
    assert mask[q, index_of(chunk.kv, ROLE_TARGET_CONDITION, 0, 0)].item() is True


@pytest.mark.parametrize(
    ("scope", "sees_other_view_history"), [("same_view", False), ("all_views", True)]
)
def test_scope_restricts_clean_history(scope, sees_other_view_history) -> None:
    """Apply the configured attention scope to clean history."""
    _memory, chunk = rolled_out_chunk()
    mask = chunk.mask(scope=scope)
    q = index_of(chunk.query, ROLE_CURRENT_TARGET, frame=2, view=0)

    assert mask[q, index_of(chunk.kv, ROLE_CLEAN_TARGET, 1, 0)].item() is True
    assert (
        mask[q, index_of(chunk.kv, ROLE_CLEAN_TARGET, 1, 1)].item()
        is sees_other_view_history
    )


@pytest.mark.parametrize(("window", "reaches"), [(0.5, True), (0.25, False)])
def test_decomposed_temporal_window_reaches_bounded_history(window, reaches) -> None:
    """Reach cross-view history only within the decomposed temporal window.

    The chunk and the cached history compare global capture timestamps, so a window
    of exactly the frame interval admits the previous frame of another view and a
    shorter one does not. This is the branch ``test_mask.py`` had to leave out: the
    step numbering it turns on comes from the memory layout, not the frame index.
    """
    _memory, chunk = rolled_out_chunk()
    mask = chunk.mask(scope="decomposed", decomposed_temporal_window_seconds=window)
    q = index_of(chunk.query, ROLE_CURRENT_TARGET, frame=2, view=0)
    history = index_of(chunk.kv, ROLE_CLEAN_TARGET, frame=1, view=1)

    assert chunk.query.timestamp[q].item() == pytest.approx(1.0)
    assert chunk.kv.timestamp[history].item() == pytest.approx(0.5)
    assert mask[q, history].item() is reaches


def test_padded_capacity_leaves_no_empty_softmax_row() -> None:
    """Every query, real or padding, must reach at least one key."""
    memory = build_memory_layout(CLIP, history_frame_ranges=[(1, 2)], capacity=64)
    chunk = build_chunk_metadata(
        CLIP,
        memory,
        chunk_start=2,
        chunk_frames=1,
        text_tokens=1,
        text_capacity=8,
        chunk_capacity=16,
    )
    mask = chunk.mask()

    assert mask.shape == (16, 8 + 16 + 64)
    assert mask.any(dim=1).all()


# --- the chunk partition ----------------------------------------------------

BRING_UP = ClipGeometry(
    num_views=11,
    frames_per_view=5,
    patch_h=15,
    patch_w=26,
    frames_per_chunk=4,
    condition_frames=0,
)
"""The 17-pixel-frame target: eleven views, text2video, so nothing is given."""


def test_a_text2video_clip_generates_frame_zero_on_its_own() -> None:
    # Not an artifact of the block size: the partition pins frame 0 apart to
    # match the training first-frame pin, so 5 frames is two steps, not one.
    assert ar_chunk_ranges(BRING_UP) == [(0, 1), (1, 5)]


def test_a_conditioning_prefix_skips_the_singleton() -> None:
    assert ar_chunk_ranges(CLIP) == [(1, 2), (2, 3)]


@pytest.mark.parametrize(
    ("frames", "chunk", "expected"),
    [
        (5, 4, [(0, 1), (1, 5)]),
        (9, 4, [(0, 1), (1, 5), (5, 9)]),
        (7, 4, [(0, 1), (1, 5), (5, 7)]),  # clipped tail
        (4, 1, [(0, 1), (1, 2), (2, 3), (3, 4)]),  # framewise
        (1, 4, [(0, 1)]),
    ],
)
def test_the_partition_covers_the_clip_exactly(frames, chunk, expected) -> None:
    geometry = ClipGeometry(
        num_views=1,
        frames_per_view=frames,
        patch_h=1,
        patch_w=1,
        frames_per_chunk=chunk,
        condition_frames=0,
    )
    assert ar_chunk_ranges(geometry) == expected


@pytest.mark.parametrize(
    ("frames", "chunk", "condition"),
    [
        (5, 4, 0),
        (9, 4, 0),
        (7, 4, 0),
        (4, 1, 0),
        (1, 4, 0),
        (17, 4, 1),
        (17, 4, 2),
        (33, 4, 1),
    ],
)
def test_asking_one_range_at_a_time_walks_the_same_partition(
    frames, chunk, condition
) -> None:
    """The step path asks for one range; the list is only the same rule iterated.

    Two implementations of the partition would be two chances to get it wrong,
    and the failure is quiet -- a chunk that spans two causal steps reads part
    of itself as history, which the mask allows.
    """
    geometry = ClipGeometry(
        num_views=1,
        frames_per_view=frames,
        patch_h=1,
        patch_w=1,
        frames_per_chunk=chunk,
        condition_frames=condition,
    )

    walked = []
    frame = condition
    while (found := ar_chunk_range(geometry, frame)) is not None:
        walked.append(found)
        frame = found[1]

    assert walked == ar_chunk_ranges(geometry)


def test_there_is_no_chunk_past_the_end_of_the_clip() -> None:
    """How a rollout of known length learns it is done, without holding a list."""
    assert ar_chunk_range(BRING_UP, BRING_UP.frames_per_view) is None
    assert ar_chunk_range(BRING_UP, BRING_UP.frames_per_view - 1) is not None


@pytest.mark.parametrize(
    ("frames", "chunk", "condition"),
    [
        (5, 4, 0),
        (9, 4, 0),
        (7, 4, 0),
        (4, 1, 0),
        (1, 4, 0),
        (17, 4, 1),
        (17, 4, 2),
        (33, 4, 1),
    ],
)
def test_the_plan_summarizes_the_ranges_without_keeping_them(
    frames, chunk, condition
) -> None:
    """What the cache is sized from, so it has to agree with the ranges exactly.

    Every committed range counts, short ones included, because each takes a
    slot of its own. The last range is left out: nothing attends to it, so it is
    never committed.
    """
    geometry = ClipGeometry(
        num_views=1,
        frames_per_view=frames,
        patch_h=1,
        patch_w=1,
        frames_per_chunk=chunk,
        condition_frames=condition,
    )
    ranges = ar_chunk_ranges(geometry)
    committed = [(s, e) for s, e in ranges if e < frames]

    plan = ar_chunk_plan(geometry)

    assert plan.count == len(ranges)
    assert plan.committed_frames == sum(e - s for s, e in committed)


@pytest.mark.parametrize("condition", [0, 1, 2])
def test_a_slot_each_is_enough_for_every_chunk_a_clip_commits(condition) -> None:
    """The count the cache is sized by has to cover the chunks it will hold.

    A short leading chunk -- frame 0 alone, or what two supplied frames leave
    over -- takes a whole slot and wastes the rest, so slots are counted from
    frames rather than from ranges. That has to round up to at least as many
    slots as there are chunks to put in them, or a chunk would have nowhere to
    go once the cache wrapped.
    """
    geometry = ClipGeometry(
        num_views=1,
        frames_per_view=50,
        patch_h=1,
        patch_w=1,
        frames_per_chunk=4,
        condition_frames=condition,
    )
    plan = ar_chunk_plan(geometry)
    committed = [(s, e) for s, e in ar_chunk_ranges(geometry) if e < 50]

    slots = -(-plan.committed_frames // 4)

    assert slots >= len(committed)


def test_no_chunk_straddles_a_causal_step() -> None:
    # `causal_steps` labels frames and `ar_chunk_ranges` groups them, and the two
    # have to be the same partition -- a chunk spanning two steps would read part
    # of itself as history, which the mask permits for clean targets and so would
    # not raise.
    for geometry in (BRING_UP, CLIP):
        for start, end in ar_chunk_ranges(geometry):
            frames = torch.arange(start, end)
            steps = causal_steps(frames, geometry.frames_per_chunk)
            assert steps.unique().numel() == 1, (
                f"chunk [{start}, {end}) spans {steps.tolist()}"
            )


# --- the clean replay -------------------------------------------------------


def test_the_clean_pass_relabels_the_chunk_as_history() -> None:
    _memory, noisy = rolled_out_chunk()
    _memory_again, clean = rolled_out_chunk(pass_kind="clean")

    assert (noisy.query.token_role_id == ROLE_CURRENT_TARGET).all()
    assert noisy.query.is_noisy.all()
    assert (clean.query.token_role_id == ROLE_CLEAN_TARGET).all()
    assert not clean.query.is_noisy.any()


def test_the_clean_pass_leaves_the_layout_alone() -> None:
    # Only the roles change. Same tokens, same order, same key stream, so the
    # keys it produces drop into memory where the noisy pass's would have.
    _m, noisy = rolled_out_chunk()
    _m2, clean = rolled_out_chunk(pass_kind="clean")

    assert torch.equal(noisy.query.frame_id, clean.query.frame_id)
    assert torch.equal(noisy.query.view_id, clean.query.view_id)
    assert torch.equal(noisy.query.causal_step_id, clean.query.causal_step_id)
    assert noisy.num_und == clean.num_und
    assert noisy.memory_offset == clean.memory_offset


def test_a_clean_query_reads_its_own_step_and_a_current_one_does_not() -> None:
    # The asymmetry the replay exists for. Frame 1 is in memory as clean history
    # at the same causal step the chunk generating frame 1 would occupy.
    memory = build_memory_layout(CLIP, history_frame_ranges=[(1, 2)])
    noisy = build_chunk_metadata(
        CLIP, memory, chunk_start=1, chunk_frames=1, text_tokens=1
    )
    clean = build_chunk_metadata(
        CLIP,
        memory,
        chunk_start=1,
        chunk_frames=1,
        text_tokens=1,
        pass_kind="clean",
    )

    history = index_of(noisy.kv, ROLE_CLEAN_TARGET, frame=1, view=0)
    q = 0

    assert noisy.mask()[q, history].item() is False
    assert clean.mask()[q, history].item() is True


def test_an_unknown_pass_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="pass_kind must be"):
        rolled_out_chunk(pass_kind="teacher_forcing")
