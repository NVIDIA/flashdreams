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


def align_num_frames(duration: float) -> int:
    """Convert seconds to H3's next decodable frame count."""
    if not math.isfinite(duration) or not MIN_DURATION <= duration <= MAX_DURATION:
        raise ValueError(
            f"duration must be between {MIN_DURATION:g} and {MAX_DURATION:g} seconds"
        )
    frames = math.ceil(duration * FPS)
    while frames % FRAME_CHUNK != FRAME_REMAINDER:
        frames += 1
    if frames / FPS > MAX_DURATION:
        raise ValueError("duration aligns beyond MiniMax H3's 15-second maximum")
    return frames


def validate_canvas(width: int, height: int) -> None:
    """Validate an H3 output canvas."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError(f"width and height must be multiples of {CANVAS_MULTIPLE}")
    if not 0.25 <= width / height <= 4:
        raise ValueError("aspect ratio must be between 1:4 and 4:1")
