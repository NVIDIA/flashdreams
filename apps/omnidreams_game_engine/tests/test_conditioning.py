# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for model versus UI conditioning tensor contracts."""

import pytest
import torch
from omnidreams_game_engine.conditioning import _bev_presentation_frames

pytestmark = pytest.mark.ci_cpu


def test_bev_presentation_preserves_renderer_bytes_in_tchw_layout() -> None:
    source = torch.arange(2 * 3 * 4 * 3, dtype=torch.uint8).reshape(2, 3, 4, 3)

    result = _bev_presentation_frames(source)

    assert result.shape == (2, 3, 3, 4)
    assert result.dtype is torch.uint8
    assert result.is_contiguous()
    assert torch.equal(result.permute(0, 2, 3, 1), source)


@pytest.mark.parametrize(
    "source",
    [
        torch.zeros(1, 3, 4, 3, dtype=torch.float32),
        torch.zeros(1, 3, 4, 4, dtype=torch.uint8),
        torch.zeros(3, 4, 3, dtype=torch.uint8),
    ],
)
def test_bev_presentation_rejects_non_renderer_contract(source: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="uint8 THWC RGB"):
        _bev_presentation_frames(source)
