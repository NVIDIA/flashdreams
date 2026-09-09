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

"""CPU tests for multi-view rotary position IDs."""

from __future__ import annotations

import pytest
import torch

from flashdreams.core.attention.multiview.packing import (
    ClipGeometry,
    build_chunk_metadata,
    build_memory_layout,
)
from flashdreams.core.attention.multiview.rope import (
    BASE_FPS,
    MODALITY_MARGIN,
    chunk_mrope_ids,
    clip_mrope_ids,
    temporal_positions,
    vision_temporal_offset,
)

pytestmark = pytest.mark.ci_cpu

FPS = 12.0
OFFSET = 15_007.0


def geometry(
    *,
    num_views: int = 3,
    frames_per_view: int = 4,
    patch_h: int = 1,
    patch_w: int = 1,
    position_stride: int | None = None,
):
    return ClipGeometry(
        num_views=num_views,
        frames_per_view=frames_per_view,
        patch_h=patch_h,
        patch_w=patch_w,
        frames_per_chunk=2,
        condition_frames=1,
        position_stride=position_stride,
    )


def test_views_occupy_consecutive_temporal_bands() -> None:
    clip = geometry()
    positions = temporal_positions(clip, torch.arange(clip.frames_per_view))

    assert positions.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_chunks_keep_global_positions() -> None:
    """A chunk carries each camera's place in the clip, not a count from zero."""
    clip = geometry()
    positions = temporal_positions(clip, torch.arange(2, 4))

    assert positions.tolist() == [2, 3, 6, 7, 10, 11]
    assert positions.tolist() != [0, 1, 2, 3, 4, 5]


def test_whole_clip_uses_camera_major_temporal_positions() -> None:
    """Keep temporal positions in ``view * stride + frame`` order."""
    clip = geometry(num_views=3, frames_per_view=4, patch_h=2, patch_w=3)

    ids = clip_mrope_ids(clip, fps=FPS, temporal_offset=OFFSET)
    temporal = (
        torch.arange(clip.num_views * clip.frames_per_view, dtype=torch.float32)
        * (BASE_FPS / FPS)
        + OFFSET
    ).repeat_interleave(clip.spatial_tokens)

    assert torch.equal(ids[0], temporal)


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

    assert at_base[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
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


def test_a_reserved_stride_spreads_the_cameras_further_apart() -> None:
    """The gap between cameras is the stride, and only the stride."""
    clip = geometry(position_stride=10)
    positions = temporal_positions(clip, torch.arange(clip.frames_per_view))

    assert positions.tolist() == [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]


def test_the_clip_s_own_length_is_what_no_reservation_means() -> None:
    """The default has to stay exactly what training used, or every run moves."""
    clip = geometry()

    assert clip.stride == clip.frames_per_view
    assert geometry(position_stride=4).stride == 4
    assert torch.equal(
        temporal_positions(clip, torch.arange(4)),
        temporal_positions(geometry(position_stride=4), torch.arange(4)),
    )


def test_frames_within_the_reservation_never_reach_the_next_camera() -> None:
    """What the reservation buys: room to keep generating without aliasing.

    A frame at or past the stride would carry the position of the next camera's
    frame, and since that position is the only thing naming the camera, the two
    tokens would be indistinguishable. Reserving room is what lets a rollout of
    unknown length keep counting.
    """
    clip = geometry(num_views=2, frames_per_view=4, position_stride=100)
    far = temporal_positions(clip, torch.arange(90, 94))

    assert far.tolist() == [90, 91, 92, 93, 190, 191, 192, 193]
    assert not set(far[:4].tolist()) & set(far[4:].tolist())


def test_a_stride_shorter_than_the_clip_is_refused() -> None:
    """Views would overlap, which is silent: eleven cameras become fewer."""
    with pytest.raises(ValueError, match="shorter than"):
        geometry(frames_per_view=8, position_stride=7)


def test_no_chunk_can_reach_the_next_camera_s_band() -> None:
    """The two guards together make aliasing unreachable, which is the policy.

    The stride is a reserve rather than something a run renumbers around, and
    that only works if a run cannot walk out of its reserve. Two checks hold it:
    a stride is never shorter than the clip, and a chunk never runs past the
    clip. So the horizon a run may reach is the thing to raise, and the picture
    can never quietly lose a camera to it.
    """
    clip = geometry(num_views=3, frames_per_view=4, position_stride=4)
    last = temporal_positions(clip, torch.tensor([clip.frames_per_view - 1]))

    assert last.tolist() == [3, 7, 11]
    with pytest.raises(ValueError, match="outside the clip"):
        chunk_mrope_ids(
            clip, chunk_start=4, chunk_frames=1, fps=FPS, temporal_offset=OFFSET
        )
