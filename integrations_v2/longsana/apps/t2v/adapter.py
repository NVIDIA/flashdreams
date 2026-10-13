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

"""LongSana text-to-video Runtime V2 application."""

from __future__ import annotations

import dataclasses
from typing import Any

from t2v import T2VApplication, T2VApplicationDefaults

from flashdreams.api_v2.application import IApplication
from flashdreams.runtime_v2.session_desc import SessionDesc
from longsana.config import PIPELINE_LONGSANA_2B_480P
from longsana.impl.constants import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    MAX_ROLLOUT_BLOCKS,
)

LONGSANA_T2V_DEFAULTS = T2VApplicationDefaults(
    pipeline_config=PIPELINE_LONGSANA_2B_480P,
    total_blocks=26,
    pixel_width=DEFAULT_VIDEO_WIDTH,
    pixel_height=DEFAULT_VIDEO_HEIGHT,
    fps=DEFAULT_VIDEO_FPS,
)
"""A 26-block rollout emits 1,041 frames, about 65 seconds at 16 FPS."""


class LongSanaT2VApplication(T2VApplication):
    """LongSana 2B constant-memory text-to-video application."""

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """
        Args:
            pipeline_config: Optional stand-in used by tests.
        """
        defaults = LONGSANA_T2V_DEFAULTS
        if pipeline_config is not None:
            defaults = dataclasses.replace(
                defaults,
                pipeline_config=pipeline_config,
            )
        super().__init__(defaults=defaults)

    def _validate_total_blocks(self, total_blocks: int) -> None:
        """Reject rollouts that exceed the released absolute RoPE table."""
        super()._validate_total_blocks(total_blocks)
        if total_blocks > MAX_ROLLOUT_BLOCKS:
            raise ValueError(
                "LongSana supports at most "
                f"{MAX_ROLLOUT_BLOCKS} blocks, got {total_blocks}."
            )

    def _validate_frame_size(self, session_desc: SessionDesc, pipeline: Any) -> None:
        """Require the native 832 by 480 release dimensions."""
        del pipeline
        requested = (session_desc.video_width, session_desc.video_height)
        expected = (DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT)
        if requested != expected:
            raise ValueError(
                f"LongSana 2B 480p requires {expected[0]}x{expected[1]} output, "
                f"got {requested[0]}x{requested[1]}."
            )


def create_app() -> IApplication:
    """Return a new LongSana text-to-video application."""
    return LongSanaT2VApplication()
