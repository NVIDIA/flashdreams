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

"""CPU contracts for native MiniMax H3 keyframe conditioning."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from minimax_h3.keyframes import encode_keyframes, prepare_keyframes
from PIL import Image

pytestmark = pytest.mark.ci_cpu


def _keyframe_fixtures() -> tuple[Image.Image, Image.Image]:
    first = Image.fromarray(np.arange(13 * 21 * 3, dtype=np.uint8).reshape(13, 21, 3))
    last = Image.fromarray(
        np.arange(23 * 11 * 3, dtype=np.uint8).reshape(23, 11, 3)[::-1].copy()
    )
    return first, last


def test_prepare_keyframes_matches_released_resize_and_crop() -> None:
    """Stretch the geometry anchor and cover-crop its follower exactly."""
    first, last = _keyframe_fixtures()
    batch = prepare_keyframes(
        first_image=first,
        last_image=last,
        height=32,
        width=64,
    )
    digest = hashlib.sha256()
    for image in batch.images:
        digest.update(np.array(image).tobytes())

    assert batch.anchors == ("first", "last")
    assert (batch.height, batch.width) == (32, 64)
    assert [image.size for image in batch.images] == [(64, 32), (64, 32)]
    assert digest.hexdigest() == (
        "c43289b3090ecfefe2413e811a01bc12c9594f757acd0d81c68b4650ff9721ff"
    )


def test_last_only_keyframe_remains_last_anchor() -> None:
    """Use a lone last image as geometry source without moving its anchor."""
    _, last = _keyframe_fixtures()
    batch = prepare_keyframes(last_image=last, height=32, width=32)
    assert batch.anchors == ("last",)
    assert batch.images[0].size == (32, 32)


class _VideoVAE:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, int]] = []

    def encode_condition_pixels(
        self, pixels: torch.Tensor, *, seed: int
    ) -> torch.Tensor:
        self.calls.append((pixels, seed))
        return torch.full((1, 24, 1, 2, 4), float(len(self.calls)))


def test_encode_keyframes_passes_base_range_pixels_and_fixed_seed() -> None:
    """Send one RGB frame per keyframe through the VAE condition adapter."""
    first, last = _keyframe_fixtures()
    batch = prepare_keyframes(
        first_image=first,
        last_image=last,
        height=32,
        width=64,
    )
    vae = _VideoVAE()
    encoded = encode_keyframes(vae, batch)

    assert [tuple(value.shape) for value in encoded] == [
        (1, 24, 1, 2, 4),
        (1, 24, 1, 2, 4),
    ]
    assert [seed for _, seed in vae.calls] == [42, 42]
    for pixels, _ in vae.calls:
        assert tuple(pixels.shape) == (1, 3, 1, 32, 64)
        assert pixels.dtype == torch.float32
        assert bool((pixels >= 0.0).all())
        assert bool((pixels <= 1.0).all())


def test_prepare_keyframes_rejects_incomplete_requests() -> None:
    """Reject missing media and partial explicit canvas dimensions."""
    first, _ = _keyframe_fixtures()
    with pytest.raises(ValueError, match="requires"):
        prepare_keyframes()
    with pytest.raises(ValueError, match="passed together"):
        prepare_keyframes(first_image=first, height=32)
    with pytest.raises(ValueError, match="PIL"):
        prepare_keyframes(first_image=np.zeros((32, 32, 3)))  # ty: ignore[invalid-argument-type]
