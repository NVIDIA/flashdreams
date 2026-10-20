# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.1 adapter for the Minimal T2V application."""

from minimal_t2v import MinimalT2VApplication

from flashdreams.runtime_v2.session_desc import SessionDesc
from wan21.config import PIPELINE_WAN21_T2V_1PT3B_480P


def create_app() -> MinimalT2VApplication:
    """Return Minimal T2V with Wan 2.1's single-block pipeline."""
    return MinimalT2VApplication(
        pipeline_config=PIPELINE_WAN21_T2V_1PT3B_480P,
        desc=SessionDesc(
            video_width=832,
            video_height=480,
            frames_per_second_for_step=16,
        ),
        total_blocks=1,
    )
