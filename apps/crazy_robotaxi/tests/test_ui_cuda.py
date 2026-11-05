# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA checks for responsive minimap composition under queued GPU work."""

import pytest
import torch
from crazy_robotaxi.ui import _composite_bev_ego_car

pytestmark = pytest.mark.ci_gpu


def test_minimap_glyph_does_not_wait_for_pending_gpu_work() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    expected = torch.zeros(3, 64, 64)
    _composite_bev_ego_car(expected)
    panel = expected.new_zeros(expected.shape, device="cuda")
    stream = torch.cuda.Stream()
    finished = torch.cuda.Event()
    with torch.cuda.stream(stream):
        # Warm the fill kernels before testing the deliberately queued delay.
        _composite_bev_ego_car(panel)
        stream.synchronize()
        panel.zero_()
        torch.cuda._sleep(1_000_000_000)
        finished.record(stream)
        assert not finished.query()
        _composite_bev_ego_car(panel)
        assert not finished.query(), "Drawing the glyph blocked on queued GPU work"
    stream.synchronize()
    torch.testing.assert_close(panel.cpu(), expected)
