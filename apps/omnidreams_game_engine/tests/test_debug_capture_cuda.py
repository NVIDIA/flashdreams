# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA capture must not wait for future generation on the default stream."""

from pathlib import Path

import numpy as np
import pytest
import torch
from omnidreams_game_engine.debug_capture import CaptureRecorder

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_gpu


def test_capture_copies_ready_snapshot_without_waiting_for_next_chunk(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    recorder = CaptureRecorder(tmp_path, SessionDesc(), images=False, snapshot_every=1)
    try:
        with torch.cuda.stream(torch.cuda.default_stream()):
            result = StepResult(
                step_index=0,
                output=torch.full((1, 3, 4, 6), 42, dtype=torch.uint8, device="cuda"),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
            result.wait_until_ready()
            # Warm the capture stream and transfer path before timing isolation.
            recorder._pixels(result)
            torch.cuda._sleep(1_000_000_000)
            next_chunk_ready = torch.cuda.Event()
            next_chunk_ready.record()
            pixels = recorder._pixels(result)
            assert not next_chunk_ready.query(), "Capture waited for the next chunk"
            np.testing.assert_array_equal(pixels, np.full((1, 4, 6, 3), 42))
    finally:
        torch.cuda.synchronize()
        recorder.close()
