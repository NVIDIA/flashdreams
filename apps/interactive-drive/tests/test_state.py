# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch
from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import StepResult
from interactive_drive_app.overlays import BEV_OVERLAY_KEY
from interactive_drive_app.state import DRIVING_METADATA_KEY, DrivingViewState

pytestmark = pytest.mark.ci_cpu


def test_result_metadata_updates_app_chrome_state() -> None:
    bev = np.ones((4, 4, 3), dtype=np.uint8)
    video = VideoStepResult.from_video_chunk(
        chunk_index=0,
        video_chunk=torch.zeros((1, 1, 1, 3, 4, 4)),
        layout="bvtchw",
        metadata={
            DRIVING_METADATA_KEY: {
                "speed_mps": 12.5,
                "steering": -0.25,
                "throttle": 0.75,
                "brake": 0.0,
                "reverse": False,
                "bev": bev,
            }
        },
    )
    state = DrivingViewState()

    frame = state.project_frame(
        StepResult(step_index=0, output=video),
        video,
        0,
        np.zeros((4, 4, 3), dtype=np.uint8),
        10,
    )

    assert state.speed_mps == 12.5
    assert state.steering == -0.25
    assert state.throttle == 0.75
    assert frame.overlay_data[BEV_OVERLAY_KEY] is bev
