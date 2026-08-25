# SPDX-FileCopyrightText: Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

"""Native MiniMax H3 first/last-keyframe preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from minimax_h3.conditioning import resolve_canvas_size
from minimax_h3.constants import KEYFRAME_ENCODE_SEED, validate_canvas


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3KeyframeBatch:
    """Prepared RGB keyframes and their positions on one target canvas."""

    images: tuple[Image.Image, ...]
    anchors: tuple[str, ...]
    height: int
    width: int

    def __post_init__(self) -> None:
        if not self.images or len(self.images) > 2:
            raise ValueError("a keyframe batch must contain one or two images")
        if len(self.images) != len(self.anchors):
            raise ValueError("every keyframe requires one anchor")
        if any(anchor not in ("first", "last") for anchor in self.anchors):
            raise ValueError("keyframe anchors must be 'first' or 'last'")
        validate_canvas(self.width, self.height)
        for image in self.images:
            if image.mode != "RGB" or image.size != (self.width, self.height):
                raise ValueError("prepared keyframes must be RGB target-canvas images")


def prepare_keyframes(
    *,
    first_image: Image.Image | None = None,
    last_image: Image.Image | None = None,
    height: int | None = None,
    width: int | None = None,
) -> MiniMaxH3KeyframeBatch:
    """Put first/last keyframes on H3's target canvas.

    The first supplied image is the geometry anchor and is stretched to the
    canvas. A second image is proportionally enlarged and center-cropped with
    the released MiniMax integer arithmetic.
    """
    for image in (first_image, last_image):
        if image is not None and not isinstance(image, Image.Image):
            raise ValueError("keyframes must be PIL RGB-compatible images")
    supplied = [
        (anchor, image.convert("RGB"))
        for anchor, image in (("first", first_image), ("last", last_image))
        if image is not None
    ]
    if not supplied:
        raise ValueError("FL2VA requires a first image, a last image, or both")
    if (height is None) != (width is None):
        raise ValueError("height and width must be passed together")
    if height is None or width is None:
        height, width = resolve_canvas_size(*supplied[0][1].size)
    validate_canvas(width, height)

    prepared = []
    for index, (_, image) in enumerate(supplied):
        if image.size == (width, height):
            prepared.append(image)
        elif index == 0:
            prepared.append(
                image.resize((width, height), Image.Resampling.LANCZOS)
            )
        else:
            scale = max(width / image.size[0], height / image.size[1])
            resized_size = (
                max(width, round(image.size[0] * scale)),
                max(height, round(image.size[1] * scale)),
            )
            left = max(0, (resized_size[0] - width) // 2)
            top = max(0, (resized_size[1] - height) // 2)
            resized = image.resize(resized_size, Image.Resampling.LANCZOS)
            prepared.append(
                resized.crop((left, top, left + width, top + height))
            )
    return MiniMaxH3KeyframeBatch(
        images=tuple(prepared),
        anchors=tuple(anchor for anchor, _ in supplied),
        height=height,
        width=width,
    )


@torch.no_grad()
def encode_keyframes(
    video_vae: Any,
    keyframes: MiniMaxH3KeyframeBatch,
    *,
    seed: int = KEYFRAME_ENCODE_SEED,
) -> tuple[Tensor, ...]:
    """Encode prepared keyframes into normalized CPU conditioning latents."""
    device = video_vae.device
    encoded = []
    for image in keyframes.images:
        pixels = (
            torch.from_numpy(np.array(image, dtype=np.uint8))
            .to(device=device, dtype=torch.float32)
            .permute(2, 0, 1)[None, :, None]
            .div(255.0)
        )
        encoded.append(video_vae.encode_condition_pixels(pixels, seed=seed))
    return tuple(encoded)


__all__ = [
    "MiniMaxH3KeyframeBatch",
    "encode_keyframes",
    "prepare_keyframes",
]
