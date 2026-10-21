# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for multiview mRoPE positions.

The best thing to check against is the code already in the tree. One camera's
positions have to be *identical* to what
``cosmos_interactive._rope.vae_mrope_ids`` gives for a track of
``frames_per_view`` frames, and the rig is then that track repeated: cameras at
one instant share one coordinate. That equality is why the model's prefill needs no
positions of its own, and pinning it here keeps this port from drifting away
from an implementation that is already in use.

Chunks are where the two part company, and where the bugs live.
"""

from __future__ import annotations

import pytest
import torch
from cosmos3_nano_sim_auto.impl.packing import (
    ClipGeometry,
    build_chunk_metadata,
    build_memory_layout,
)
from cosmos3_nano_sim_auto.impl.transformer.rope import (
    BASE_FPS,
    MODALITY_MARGIN,
    caption_mrope_ids,
    chunk_mrope_ids,
    clip_mrope_ids,
    temporal_positions,
    vision_temporal_offset,
)
from cosmos_interactive._rope import text_mrope_ids, vae_mrope_ids

pytestmark = pytest.mark.ci_cpu

FPS = 12.0
OFFSET = 15_007.0


def geometry(
    *,
    num_views: int = 3,
    frames_per_view: int | None = 4,
    patch_h: int = 1,
    patch_w: int = 1,
):
    return ClipGeometry(
        num_views=num_views,
        frames_per_view=frames_per_view,
        patch_h=patch_h,
        patch_w=patch_w,
        frames_per_chunk=2,
        condition_frames=1,
    )


def test_every_camera_shares_the_instant_it_was_filmed_at() -> None:
    """The convention these weights were trained under: time, not identity.

    The camera a token came from is carried by the attention routing and by
    the caption scoped to it. Spelling identity into the position instead is
    the bug this test exists to keep out, and it is a quiet one: the run still
    generates, each view still follows its own controls, and only the things
    the views have to agree on between themselves come out wrong.
    """
    clip = geometry()
    positions = temporal_positions(clip, torch.arange(clip.frames_per_view))

    assert positions.tolist() == [0, 1, 2, 3] * 3


def test_chunks_keep_global_positions() -> None:
    """A chunk carries each camera's place in the clip, not a count from zero."""
    clip = geometry()
    positions = temporal_positions(clip, torch.arange(2, 4))

    assert positions.tolist() == [2, 3, 2, 3, 2, 3]
    assert positions.tolist() != [0, 1, 0, 1, 0, 1]


def test_the_rig_is_one_camera_s_track_repeated() -> None:
    """One camera matches the stock packer, and the rig is that, tiled."""
    clip = geometry(num_views=3, frames_per_view=4, patch_h=2, patch_w=3)

    ours = clip_mrope_ids(clip, fps=FPS, temporal_offset=OFFSET)
    one_camera, _ = vae_mrope_ids(
        clip.frames_per_view,
        clip.patch_h,
        clip.patch_w,
        OFFSET,
        fps=FPS,
        base_fps=BASE_FPS,
    )

    assert torch.equal(ours, one_camera.repeat(1, clip.num_views))


def test_chunk_is_the_clip_restricted_to_its_frames() -> None:
    clip = geometry()
    whole = clip_mrope_ids(clip, fps=FPS, temporal_offset=OFFSET)
    chunk = chunk_mrope_ids(
        clip, chunk_start=2, chunk_frames=2, fps=FPS, temporal_offset=OFFSET
    )

    keep = [
        view * clip.frames_per_view + frame
        for view in range(clip.num_views)
        for frame in range(2, 4)
    ]
    assert torch.equal(chunk, whole[:, keep])


def test_positions_scale_with_the_clip_frame_rate() -> None:
    """Half the base rate puts frames twice as far apart on the temporal axis."""
    clip = geometry()
    at_base = clip_mrope_ids(clip, fps=BASE_FPS, temporal_offset=0.0)
    at_half = clip_mrope_ids(clip, fps=BASE_FPS / 2, temporal_offset=0.0)

    assert at_base[0].tolist() == [0, 1, 2, 3] * 3
    assert torch.equal(at_half[0], at_base[0] * 2)


def test_offset_shifts_only_the_temporal_axis() -> None:
    clip = geometry(patch_h=2, patch_w=3)
    at_zero = clip_mrope_ids(clip, fps=FPS, temporal_offset=0.0)
    shifted = clip_mrope_ids(clip, fps=FPS, temporal_offset=OFFSET)

    assert torch.equal(shifted[0], at_zero[0] + OFFSET)
    assert torch.equal(shifted[1:], at_zero[1:])


def test_spatial_axes_restart_every_frame() -> None:
    """``unified_3d_mrope_reset_spatial_ids=True``: absolute position within a frame."""
    clip = geometry(num_views=2, frames_per_view=1, patch_h=2, patch_w=3)
    ids = clip_mrope_ids(clip, fps=FPS, temporal_offset=0.0)

    assert ids[1].tolist() == [0, 0, 0, 1, 1, 1] * 2
    assert ids[2].tolist() == [0, 1, 2, 0, 1, 2] * 2


def test_token_count_matches_what_the_packer_emits() -> None:
    """One position per token, or the two disagree about which token is which."""
    clip = geometry(num_views=3, frames_per_view=4, patch_h=2, patch_w=3)
    memory = build_memory_layout(clip, history_frame_ranges=[(1, 3)])
    chunk = build_chunk_metadata(
        clip, memory, chunk_start=3, chunk_frames=1, text_tokens=4
    )

    ids = chunk_mrope_ids(
        clip, chunk_start=3, chunk_frames=1, fps=FPS, temporal_offset=OFFSET
    )

    assert ids.shape == (3, len(chunk.query))


def test_chunk_rejects_frames_outside_the_clip() -> None:
    clip = geometry()
    with pytest.raises(ValueError, match="outside the clip"):
        chunk_mrope_ids(
            clip, chunk_start=3, chunk_frames=2, fps=FPS, temporal_offset=OFFSET
        )


def test_vision_starts_after_the_text_plus_the_modality_margin() -> None:
    assert vision_temporal_offset(7) == 7 + MODALITY_MARGIN
    assert MODALITY_MARGIN == 15_000.0


def test_a_chunk_still_cannot_reach_outside_the_clip() -> None:
    """The one guard left now that cameras cannot alias onto each other.

    Sharing a coordinate removes the aliasing this used to also protect
    against, so what remains is the plain bound: a chunk covers frames the
    clip has.
    """
    clip = geometry(num_views=3, frames_per_view=4)

    assert temporal_positions(clip, torch.tensor([3])).tolist() == [3, 3, 3]
    with pytest.raises(ValueError, match="outside the clip"):
        chunk_mrope_ids(
            clip, chunk_start=4, chunk_frames=1, fps=FPS, temporal_offset=OFFSET
        )


def test_the_run_s_length_does_not_move_the_frames_it_shares() -> None:
    """A longer run leaves the shared prefix's coordinates exactly alone.

    Under the old convention the declared length was the spacing between
    cameras, so asking for more frames moved every camera but the first. It is
    what made a 601-frame run a different run from a 197-frame one over the
    frames they had in common.
    """
    short = temporal_positions(geometry(frames_per_view=4), torch.arange(4))
    long = temporal_positions(geometry(frames_per_view=400), torch.arange(4))

    assert torch.equal(short, long)


def test_every_camera_s_caption_is_laid_from_zero() -> None:
    """Captions describe one instant of one rig, so they share the grid.

    the model's packer rewinds its cursor between captions. Laying them end to end
    instead would leave each camera's caption at a different place on the
    temporal axis, which is what the positions being shared says they are not.
    """
    ids, used = caption_mrope_ids([3, 2, 4])

    assert ids.shape == (3, 9)
    assert ids[0].tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]
    assert torch.equal(ids[1], ids[0]) and torch.equal(ids[2], ids[0])
    assert used == 4


def test_vision_follows_the_longest_caption_and_not_their_total() -> None:
    """Otherwise a clip's vision positions would depend on its camera count.

    Eleven captions of a few hundred tokens each would push vision thousands of
    positions further along than the one caption training measured from.
    """
    _ids, used = caption_mrope_ids([300, 280, 310])

    assert vision_temporal_offset(used) == 310 + MODALITY_MARGIN
    assert vision_temporal_offset(used) < vision_temporal_offset(300 + 280 + 310)


def test_a_lone_caption_lands_where_a_merged_prompt_would() -> None:
    """The per-view path has to be a generalisation, not a separate convention."""
    ids, used = caption_mrope_ids([5])
    merged, _next = text_mrope_ids(5, 0.0)

    assert torch.equal(ids, merged)
    assert vision_temporal_offset(used) == vision_temporal_offset(5)


@pytest.mark.parametrize(
    ("runs", "match"),
    [([], "at least one camera"), ([3, -1], "non-negative")],
)
def test_caption_positions_reject_runs_they_cannot_lay(
    runs: list[int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        caption_mrope_ids(runs)


def test_a_clip_with_no_end_positions_a_chunk_but_not_a_whole_clip() -> None:
    """A run that never finishes has positions for each chunk and no clip-wide set.

    Taking the end off changes nothing about where a frame sits: frame 9 has
    the coordinate it would have had in a clip long enough to hold it, which
    is what lets a cached chunk and a new one agree.
    """
    endless = geometry(frames_per_view=None)

    positions = chunk_mrope_ids(
        endless, chunk_start=9, chunk_frames=2, fps=FPS, temporal_offset=OFFSET
    )
    bounded = chunk_mrope_ids(
        geometry(frames_per_view=40),
        chunk_start=9,
        chunk_frames=2,
        fps=FPS,
        temporal_offset=OFFSET,
    )

    assert torch.equal(positions, bounded)
    with pytest.raises(ValueError, match="no whole-clip positions"):
        clip_mrope_ids(endless, fps=FPS, temporal_offset=OFFSET)
