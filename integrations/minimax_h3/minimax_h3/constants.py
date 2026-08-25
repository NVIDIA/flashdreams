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

"""MiniMax H3 geometry, timing, and model constants."""

from __future__ import annotations

import math

MODEL_ID = "MiniMaxAI/MiniMax-H3"
FPS = 24
MIN_DURATION = 5.0
MAX_DURATION = 15.0
FRAME_CHUNK = 17
FRAME_REMAINDER = 5
CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
VAE_SPATIAL_COMPRESSION_RATIO = 16
VAE_LATENT_CHANNELS = 24
AUDIO_SAMPLE_RATE = 32000
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_LATENT_CHANNELS = 32
AUDIO_CHANNELS = 2
PATCH_SIZE = (1, 2, 2)
KEYFRAME_NOISE_AUG = 0.999
TEXT_ENCODER_LAYER = 50
TEXT_EMBED_DIM = 5120
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2


def align_num_frames(duration: float) -> int:
    """Convert a requested duration to the next H3 frame count.

    Validate the user-facing duration before alignment. The aligned model
    timeline may extend beyond ``MAX_DURATION`` by less than one frame chunk;
    for example, 15 seconds aligns from 360 to 362 frames.

    Args:
        duration: Requested duration in seconds.

    Returns:
        Smallest frame count on the H3 ``17k + 5`` grid covering the request.

    Raises:
        ValueError: ``duration`` is non-finite or outside the advertised range.
    """
    if not math.isfinite(duration) or not MIN_DURATION <= duration <= MAX_DURATION:
        raise ValueError(
            f"duration must be between {MIN_DURATION:g} and {MAX_DURATION:g} seconds"
        )
    frames = math.ceil(duration * FPS)
    while frames % FRAME_CHUNK != FRAME_REMAINDER:
        frames += 1
    return frames


def validate_canvas(width: int, height: int) -> None:
    """Validate an H3 output canvas."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError(f"width and height must be multiples of {CANVAS_MULTIPLE}")
    if not 0.25 <= width / height <= 4:
        raise ValueError("aspect ratio must be between 1:4 and 4:1")
