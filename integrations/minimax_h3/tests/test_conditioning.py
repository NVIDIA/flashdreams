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

"""CPU parity tests for native MiniMax H3 conditioning."""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest
import torch
from minimax_h3.conditioning import (
    audio_latent_num_frames,
    build_packed_layout,
    build_ref2va_packed_layout,
    patchify_video_latents,
    prepare_denoise_state,
    prepare_ref2va_denoise_state,
    resolve_canvas_size,
    video_latent_num_frames,
)
from minimax_h3.model import MiniMaxH3DiffusionModel
from minimax_h3.reference_conditioning import (
    MiniMaxH3AudioReference,
    MiniMaxH3EncodedReferences,
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
)
from minimax_h3.transformer import MiniMaxH3TransformerConfig

pytestmark = pytest.mark.ci_cpu


def _layout_digest(*values: torch.Tensor | int) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, torch.Tensor):
            digest.update(value.contiguous().numpy().tobytes())
        else:
            digest.update(struct.pack("<q", value))
    return digest.hexdigest()


def test_geometry_matches_pinned_h3_oracle() -> None:
    """Freeze default canvas and both duration-boundary latent counts."""
    assert resolve_canvas_size(16, 9) == (768, 1344)
    assert video_latent_num_frames(124) == 37
    assert video_latent_num_frames(362) == 107
    assert audio_latent_num_frames(124) == 207
    assert audio_latent_num_frames(362) == 603
    with pytest.raises(ValueError, match="aligned"):
        video_latent_num_frames(360)
    with pytest.raises(ValueError, match="between 1:4 and 4:1"):
        resolve_canvas_size(5, 1)


def test_patchify_video_latents_uses_frame_major_rows() -> None:
    """Pack spatial patches frame-major and row-major within each frame."""
    latents = torch.arange(32, dtype=torch.float32).reshape(1, 1, 2, 4, 4)
    expected = torch.tensor(
        [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15],
            [16, 17, 20, 21],
            [18, 19, 22, 23],
            [24, 25, 28, 29],
            [26, 27, 30, 31],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(patchify_video_latents(latents), expected)
    with pytest.raises(ValueError, match="not divisible"):
        patchify_video_latents(torch.zeros(1, 1, 1, 3, 4))


def test_packed_layout_matches_pinned_oracle_digest() -> None:
    """Match Diffusers ``175fe6b2419a`` including float64 rotary ulps."""
    layout = build_packed_layout(
        torch.tensor([1, 0, 1]),
        num_latent_frames=2,
        latent_height=4,
        latent_width=8,
        num_audio_latents=4,
        keyframe_anchors=("first", "last"),
    )
    assert layout.position_ids.dtype == torch.float64
    assert layout.token_tags.tolist()[:3] == [1, 0, 1]
    assert layout.num_condition_video_rows == 16
    assert layout.num_condition_audio_rows == 0
    assert (
        _layout_digest(
            layout.position_ids,
            layout.token_tags,
            layout.video_indices,
            layout.audio_indices,
            layout.text_indices,
            layout.num_condition_video_rows,
            layout.num_condition_audio_rows,
        )
        == "debfd788961ee5645377f8f2c4202ec15e3480e6b9c558b3050b9e0875d03785"
    )
    with pytest.raises(ValueError, match="first.*last"):
        build_packed_layout(
            torch.tensor([1]),
            num_latent_frames=2,
            latent_height=4,
            latent_width=4,
            num_audio_latents=4,
            keyframe_anchors=("middle",),
        )


def test_prepare_denoise_state_matches_oracle_rng_order() -> None:
    """Draw keyframe, video, then audio noise from one request generator."""
    prompt_embeds = torch.zeros((1, 3, 5120), dtype=torch.bfloat16)
    state = prepare_denoise_state(
        prompt_embeds,
        torch.tensor([1, 0, 1]),
        num_frames=124,
        height=32,
        width=32,
        generator=torch.Generator().manual_seed(123),
        condition_latents=(torch.zeros(1, 24, 1, 2, 2),),
        keyframe_anchors=("first",),
    )
    digest = hashlib.sha256()
    digest.update(state.latents.contiguous().numpy().tobytes())
    digest.update(state.audio_latents.contiguous().numpy().tobytes())
    assert digest.hexdigest() == (
        "d3be029186106d48975bcbc78b6dde61e8ddea98f40a12c0844bc7833a549024"
    )
    assert tuple(state.latents.shape) == (38, 96)
    assert tuple(state.audio_latents.shape) == (414, 32)
    assert (state.num_latent_frames, state.num_audio_latents) == (37, 207)
    assert state.num_condition_video_rows == 1
    MiniMaxH3DiffusionModel._validate_state(
        state,
        MiniMaxH3TransformerConfig(
            checkpoint_path=None,
            device="meta",
            execution_device="cpu",
            attention_backend="math",
        ),
    )


def test_prepare_denoise_state_rejects_mismatched_inputs() -> None:
    """Reject presentation and condition rows before a transformer forward."""
    with pytest.raises(ValueError, match="every prompt-embedding row"):
        prepare_denoise_state(
            torch.zeros(1, 2, 5120),
            torch.tensor([1]),
            num_frames=124,
            height=32,
            width=32,
        )
    with pytest.raises(ValueError, match="one keyframe anchor"):
        prepare_denoise_state(
            torch.zeros(1, 1, 5120),
            torch.tensor([1]),
            num_frames=124,
            height=32,
            width=32,
            condition_latents=(torch.zeros(1, 24, 1, 2, 2),),
        )
    with pytest.raises(ValueError, match="target latent canvas"):
        prepare_denoise_state(
            torch.zeros(1, 1, 5120),
            torch.tensor([1]),
            num_frames=124,
            height=32,
            width=32,
            condition_latents=(torch.zeros(1, 24, 1, 4, 4),),
            keyframe_anchors=("first",),
        )
    with pytest.raises(ValueError, match="124 through 362"):
        prepare_denoise_state(
            torch.zeros(1, 1, 5120),
            torch.tensor([1]),
            num_frames=107,
            height=32,
            width=32,
        )


def _reference_fixture() -> tuple[list[MiniMaxH3Reference], MiniMaxH3EncodedReferences]:
    """Build the mixed-modality geometry used by pinned REF2VA parity tests."""
    references = [
        MiniMaxH3ImageReference(np.zeros((1, 1, 3), dtype=np.uint8)),
        MiniMaxH3VideoReference(
            np.zeros((1, 1, 1, 3), dtype=np.uint8),
            24.0,
            torch.zeros(2, 1),
            32000,
        ),
        MiniMaxH3AudioReference(torch.zeros(2, 1), 32000),
    ]
    encoded = MiniMaxH3EncodedReferences(
        video=(
            torch.zeros(1, 24, 1, 4, 8),
            torch.ones(1, 24, 2, 4, 4),
        ),
        audio=(torch.full((6, 32), 3.0), torch.full((4, 32), 4.0)),
    )
    return references, encoded


def test_ref2va_layout_matches_pinned_oracle_digest() -> None:
    """Match Diffusers ``175fe6b2419a`` shared-clock float64 layout exactly."""
    references, encoded = _reference_fixture()
    layout = build_ref2va_packed_layout(
        torch.tensor([1, 0, 1]),
        references,
        encoded,
        num_latent_frames=2,
        latent_height=4,
        latent_width=8,
        num_audio_latents=4,
    )

    assert tuple(layout.position_ids.shape) == (53, 3)
    assert layout.num_condition_video_rows == 16
    assert layout.num_condition_audio_rows == 10
    assert (
        _layout_digest(
            layout.position_ids,
            layout.token_tags,
            layout.video_indices,
            layout.audio_indices,
            layout.text_indices,
            layout.num_condition_video_rows,
            layout.num_condition_audio_rows,
        )
        == "19bfa28898063150168746bbd26fa88b619b6d4656723f0e215a2123455d5be3"
    )


def test_ref2va_layout_preserves_long_video_rotary_sum_order() -> None:
    """Freeze the sequential float64 span that differs after 16 latent frames."""
    references = [
        MiniMaxH3VideoReference(np.zeros((1, 1, 1, 3), dtype=np.uint8), fps=24.0)
    ]
    layout = build_ref2va_packed_layout(
        torch.tensor([1, 1]),
        references,
        MiniMaxH3EncodedReferences(video=(torch.zeros(1, 24, 16, 4, 4),), audio=()),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=4,
    )

    assert (
        _layout_digest(
            layout.position_ids,
            layout.token_tags,
            layout.video_indices,
            layout.audio_indices,
            layout.text_indices,
            layout.num_condition_video_rows,
            layout.num_condition_audio_rows,
        )
        == "9cfaff1dab99498789d4f13eb355638870182ba3371deebeffb59883ef825439"
    )


def test_prepare_ref2va_state_preserves_rng_and_clean_audio_order() -> None:
    """Draw visual conditions, target video, then target audio in that order."""
    references, _ = _reference_fixture()
    encoded = MiniMaxH3EncodedReferences(
        video=(
            torch.zeros(1, 24, 1, 2, 4),
            torch.ones(1, 24, 2, 2, 2),
        ),
        audio=(torch.full((6, 32), 3.0), torch.full((4, 32), 4.0)),
    )
    state = prepare_ref2va_denoise_state(
        torch.zeros(1, 3, 5120, dtype=torch.bfloat16),
        torch.tensor([1, 0, 1]),
        references,
        encoded,
        num_frames=124,
        height=32,
        width=32,
        generator=torch.Generator().manual_seed(123),
    )

    oracle = torch.Generator().manual_seed(123)
    first_noise = torch.randn(encoded.video[0].shape, generator=oracle)
    second_noise = torch.randn(encoded.video[1].shape, generator=oracle)
    target_video = torch.randn((1, 24, 37, 2, 2), generator=oracle)
    target_audio = torch.randn((414, 32), generator=oracle)
    noise_level = torch.tensor(0.999, dtype=torch.float32)
    expected_video = torch.cat(
        [
            patchify_video_latents(
                noise_level * encoded.video[0] + (1.0 - noise_level) * first_noise
            ),
            patchify_video_latents(
                noise_level * encoded.video[1] + (1.0 - noise_level) * second_noise
            ),
            patchify_video_latents(target_video),
        ]
    )
    expected_audio = torch.cat([*encoded.audio, target_audio])
    torch.testing.assert_close(state.latents, expected_video, rtol=0, atol=0)
    torch.testing.assert_close(state.audio_latents, expected_audio, rtol=0, atol=0)
    assert state.num_condition_video_rows == 4
    assert state.num_condition_audio_rows == 10
    assert tuple(state.latents.shape) == (41, 96)
    assert tuple(state.audio_latents.shape) == (424, 32)
    MiniMaxH3DiffusionModel._validate_state(
        state,
        MiniMaxH3TransformerConfig(
            checkpoint_path=None,
            device="meta",
            execution_device="cpu",
            attention_backend="math",
        ),
    )


def test_ref2va_layout_rejects_reference_encoder_mismatch() -> None:
    """Reject a missing visual latent before consuming the request generator."""
    references, encoded = _reference_fixture()
    with pytest.raises(ValueError, match="video references do not match"):
        build_ref2va_packed_layout(
            torch.tensor([1]),
            references,
            MiniMaxH3EncodedReferences(video=encoded.video[:1], audio=encoded.audio),
            num_latent_frames=2,
            latent_height=4,
            latent_width=8,
            num_audio_latents=4,
        )
