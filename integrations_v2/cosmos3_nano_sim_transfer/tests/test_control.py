# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control frames: direct resize per control kind, range, and latent spans."""

from __future__ import annotations

import pytest
import torch
from cosmos3_nano_sim_transfer.impl.control import (
    latent_count,
    latent_frame_spans,
    padded_frame_indices,
    resize_control,
    to_model_range,
)

pytestmark = pytest.mark.ci_cpu


def _palette_frames(height: int, width: int) -> torch.Tensor:
    """``[2, 3, H, W]`` uint8 frames of four flat label colors."""
    colors = torch.tensor(
        [[0, 0, 0], [255, 0, 0], [12, 200, 77], [90, 90, 250]], dtype=torch.uint8
    )
    labels = (
        torch.arange(height)[:, None] // 7 + torch.arange(width)[None, :] // 11
    ) % 4
    frame = colors[labels].permute(2, 0, 1)
    return torch.stack((frame, frame.roll(3, dims=-1)))


def test_frames_at_model_size_pass_through() -> None:
    frames = torch.randint(0, 256, (2, 3, 480, 832), dtype=torch.uint8)
    for hint in ("seg", "edge", "depth", "blur"):
        assert torch.equal(resize_control(frames, hint), frames.float())


@pytest.mark.parametrize("hint", ["seg", "edge"])
def test_label_controls_keep_their_palette(hint: str) -> None:
    frames = _palette_frames(720, 1280)
    resized = resize_control(frames, hint)
    assert resized.shape == (2, 3, 480, 832)
    before = {tuple(c) for c in frames.permute(0, 2, 3, 1).reshape(-1, 3).tolist()}
    after = {
        tuple(int(v) for v in c)
        for c in resized.permute(0, 2, 3, 1).reshape(-1, 3).tolist()
    }
    assert after <= before


def test_label_resize_is_nearest_neighbor() -> None:
    frames = _palette_frames(240, 416)
    resized = resize_control(frames, "seg")
    # Exact 2x: every source pixel becomes a 2x2 block.
    assert torch.equal(
        resized, frames.float().repeat_interleave(2, -2).repeat_interleave(2, -1)
    )


def test_depth_stays_fractional_and_blur_is_rounded() -> None:
    frames = torch.randint(0, 256, (1, 3, 300, 500), dtype=torch.uint8)
    depth = resize_control(frames, "depth")
    blur = resize_control(frames, "blur")
    assert depth.shape == blur.shape == (1, 3, 480, 832)
    assert not torch.equal(depth, depth.round())
    assert torch.equal(blur, blur.round()) and blur.min() >= 0 and blur.max() <= 255


def test_invalid_inputs_are_rejected() -> None:
    frames = torch.zeros(1, 3, 480, 832, dtype=torch.uint8)
    with pytest.raises(ValueError, match="unknown control"):
        resize_control(frames, "normals")
    with pytest.raises(ValueError, match="uint8"):
        resize_control(frames.float(), "seg")
    with pytest.raises(ValueError, match="uint8"):
        resize_control(frames[:, :1], "seg")


def test_model_range() -> None:
    assert torch.equal(
        to_model_range(torch.tensor([0.0, 127.5, 255.0])),
        torch.tensor([-1.0, 0.0, 1.0]),
    )


def test_latent_spans_are_one_then_four_frames() -> None:
    spans = latent_frame_spans(201)
    assert len(spans) == 51
    assert spans[0] == slice(0, 1)
    assert all(s.stop - s.start == 4 for s in spans[1:])
    assert spans[-1] == slice(197, 201)
    assert latent_frame_spans(7) == [slice(0, 1), slice(1, 5)]
    assert latent_frame_spans(1) == [slice(0, 1)]
    with pytest.raises(ValueError):
        latent_frame_spans(0)


# cosmos-framework 86e3528 ``vision.pad_temporal_frames`` applied to frame
# indices, padded to whole latents (and two other targets).
COSMOS_FRAMEWORK_PADDING = [
    (1, None, [0]),
    (2, None, [0, 1, 1, 1, 1]),
    (3, None, [0, 1, 2, 2, 1]),
    (4, None, [0, 1, 2, 3, 3]),
    (6, None, [0, 1, 2, 3, 4, 5, 5, 4, 3]),
    (7, None, [0, 1, 2, 3, 4, 5, 6, 6, 5]),
    (9, None, list(range(9))),
    (10, None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7]),
    (1, 5, [0, 0, 0, 0, 0]),
    (3, 13, [0, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1]),
]


@pytest.mark.parametrize(("frames", "target", "expected"), COSMOS_FRAMEWORK_PADDING)
def test_clip_is_padded_as_cosmos_framework_pads_it(
    frames: int, target: int | None, expected: list[int]
) -> None:
    assert padded_frame_indices(frames, target) == expected


def test_latent_count_completes_the_last_latent() -> None:
    assert [latent_count(t) for t in (1, 2, 5, 6, 9, 201, 600, 601, 602)] == [
        1, 2, 2, 3, 3, 51, 151, 151, 152,
    ]  # fmt: skip
    for frames in range(1, 30):
        padded = padded_frame_indices(frames)
        assert len(latent_frame_spans(len(padded))) == latent_count(frames)
    with pytest.raises(ValueError):
        latent_count(0)
