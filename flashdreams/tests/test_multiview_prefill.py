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

"""CPU tests for multi-view prefill metadata and fixed-slot packing."""

from __future__ import annotations

import pytest
import torch

from flashdreams.core.attention.multiview import (
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_TARGET_CONDITION,
    ROLE_UND,
    ClipGeometry,
    StreamFields,
    build_memory_layout,
    build_prefill_pack,
    pad_control_slots,
)

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


def tagged_kv(tags: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build one K/V layer whose rows carry their source-token tag."""
    rows = tags.reshape(1, 1, -1, 1).to(torch.float32)
    return [(rows.expand(1, 2, len(tags), 8).clone(), rows.clone())]


def test_prefilled_control_rows_match_the_memory_layout() -> None:
    ranges = [(0, 1), (1, 3), (3, 5)]
    slot_frames = CLIP.frames_per_chunk
    per_frame = CLIP.tokens_per_frame
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

    written = padded[0][1].reshape(-1)
    region = slice(0, len(written))
    control = (memory.stream.token_role_id == ROLE_CONTROL)[region]
    claimed = (
        memory.stream.view_id[region][control] * 100
        + memory.stream.frame_id[region][control]
    )
    assert len(claimed) == sum(end - start for start, end in ranges) * per_frame
    assert torch.equal(written[control], claimed.to(torch.float32))


def test_short_control_range_leaves_its_slot_tail_zeroed() -> None:
    ranges = [(0, 1), (1, 3)]
    per_frame = CLIP.tokens_per_frame
    real = sum((end - start) * per_frame for start, end in ranges)

    padded = pad_control_slots(
        tagged_kv(torch.ones(real)),
        control_ranges=ranges,
        tokens_per_frame=per_frame,
        slot_frames=CLIP.frames_per_chunk,
    )

    rows = padded[0][1].reshape(-1)
    assert len(rows) == len(ranges) * CLIP.frames_per_chunk * per_frame
    assert torch.equal(rows[:per_frame], torch.ones(per_frame))
    assert not rows[per_frame : CLIP.frames_per_chunk * per_frame].any()


def test_prefill_gather_uses_the_target_section_stride() -> None:
    pack = build_prefill_pack(CLIP, text_tokens=TEXT_TOKENS, target_frames=2)
    controls = CLIP.num_views * CLIP.frames_per_view * CLIP.spatial_tokens

    assert controls == 30
    assert pack.memory_token_indexes.tolist() == [
        *range(controls),
        30,
        31,
        34,
        35,
        38,
        39,
    ]


def test_condition_only_target_makes_the_gather_an_identity() -> None:
    pack = build_prefill_pack(CLIP, text_tokens=TEXT_TOKENS)
    memory = build_memory_layout(CLIP)

    assert len(pack.gen) == memory.real_token_count
    assert torch.equal(pack.memory_token_indexes, torch.arange(len(pack.gen)))


@pytest.mark.parametrize("target_frames", [None, CLIP.frames_per_view])
def test_gathered_prefill_metadata_matches_memory(
    target_frames: int | None,
) -> None:
    pack = build_prefill_pack(
        CLIP,
        text_tokens=TEXT_TOKENS,
        target_frames=target_frames,
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


def test_prefill_pack_contains_control_condition_and_clean_target_roles() -> None:
    pack = build_prefill_pack(
        CLIP,
        text_tokens=TEXT_TOKENS,
        target_frames=CLIP.frames_per_view,
    )
    roles = pack.gen.token_role_id
    per_view = CLIP.frames_per_view * CLIP.spatial_tokens
    conditions = CLIP.condition_frames * CLIP.spatial_tokens

    assert int((roles == ROLE_CONTROL).sum()) == CLIP.num_views * per_view
    assert int((roles == ROLE_TARGET_CONDITION).sum()) == (CLIP.num_views * conditions)
    assert int((roles == ROLE_CLEAN_TARGET).sum()) == (
        CLIP.num_views * (per_view - conditions)
    )
    assert torch.all(pack.und.token_role_id == ROLE_UND)


def test_prefill_pack_preserves_view_caption_runs() -> None:
    pack = build_prefill_pack(
        CLIP,
        text_tokens=6,
        view_text_tokens=(2, 1, 3),
    )

    assert pack.und.view_id.tolist() == [0, 0, 1, 2, 2, 2]
    text_visibility = pack.mask()[:, :6]
    for view in range(CLIP.num_views):
        rows = pack.gen.view_id == view
        expected = pack.und.view_id == view
        assert torch.equal(text_visibility[rows], expected.expand(int(rows.sum()), -1))


def test_every_prefill_query_can_reach_a_key() -> None:
    pack = build_prefill_pack(
        CLIP,
        text_tokens=TEXT_TOKENS,
        target_frames=CLIP.frames_per_view,
    )
    mask = pack.mask()

    assert mask.shape == (len(pack.gen), pack.num_und + len(pack.gen))
    assert bool(mask.any(dim=1).all())


def test_prefill_block_mask_accepts_asymmetric_tiles() -> None:
    pack = build_prefill_pack(CLIP, text_tokens=TEXT_TOKENS)

    mask = pack.block_mask(block_size=(16, 32))

    assert tuple(mask.shape[-2:]) == (len(pack.gen), pack.num_und + len(pack.gen))
    assert mask.BLOCK_SIZE == (16, 32)


def test_control_queries_cannot_see_the_target_item() -> None:
    pack = build_prefill_pack(
        CLIP,
        text_tokens=TEXT_TOKENS,
        target_frames=CLIP.frames_per_view,
    )
    mask = pack.mask()
    control_rows = pack.gen.is_control
    target_columns = torch.cat(
        [torch.zeros(pack.num_und, dtype=torch.bool), ~pack.gen.is_control]
    )

    assert not bool(mask[control_rows][:, target_columns].any())


def test_condition_queries_can_see_the_clean_target_item() -> None:
    pack = build_prefill_pack(
        CLIP,
        text_tokens=TEXT_TOKENS,
        target_frames=CLIP.frames_per_view,
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


@pytest.mark.parametrize(
    ("target_frames", "message"),
    [(0, "conditioning frames"), (CLIP.frames_per_view + 1, "fit the clip")],
)
def test_prefill_rejects_a_target_outside_the_clip(
    target_frames: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_prefill_pack(
            CLIP,
            text_tokens=TEXT_TOKENS,
            target_frames=target_frames,
        )


def test_prefill_rejects_invalid_text_or_slot_geometry() -> None:
    with pytest.raises(ValueError, match="text_tokens"):
        build_prefill_pack(CLIP, text_tokens=-1)
    with pytest.raises(ValueError, match="tokens_per_frame"):
        pad_control_slots(
            tagged_kv(torch.ones(CLIP.tokens_per_frame)),
            control_ranges=[(0, 1)],
            tokens_per_frame=0,
            slot_frames=1,
        )
    with pytest.raises(ValueError, match="slot_frames"):
        pad_control_slots(
            tagged_kv(torch.ones(CLIP.tokens_per_frame)),
            control_ranges=[(0, 1)],
            tokens_per_frame=CLIP.tokens_per_frame,
            slot_frames=0,
        )


def test_control_slot_padding_validates_prefill_geometry() -> None:
    with pytest.raises(ValueError, match="shape"):
        pad_control_slots(
            [(torch.zeros(3), torch.zeros(3))],
            control_ranges=[(0, 1)],
            tokens_per_frame=1,
            slot_frames=2,
        )
    with pytest.raises(ValueError, match="control ranges need"):
        pad_control_slots(
            [(torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))],
            control_ranges=[(0, 2)],
            tokens_per_frame=1,
            slot_frames=2,
        )


def test_prefill_pack_fields_are_stream_metadata() -> None:
    pack = build_prefill_pack(CLIP, text_tokens=TEXT_TOKENS)

    assert isinstance(pack.und, StreamFields)
    assert isinstance(pack.gen, StreamFields)
