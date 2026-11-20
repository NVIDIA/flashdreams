# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spatial RoPE coordinates across resolutions and temporal cache modes."""

from typing import Any

import pytest
import torch

from flashdreams.core.attention.rope import (
    KVCacheRelativeRotaryPositionEmbedding3D,
    RotaryPositionEmbedding3D,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("relative", [False, True])
def test_spatial_scale_matches_reference_coordinates(relative: bool) -> None:
    rope_type = (
        KVCacheRelativeRotaryPositionEmbedding3D
        if relative
        else RotaryPositionEmbedding3D
    )
    kwargs: dict[str, Any] = dict(head_dim=24, len_t=2, device=torch.device("cpu"))
    if relative:
        kwargs.update(sink_size_t=2, window_size_t=4)
    reference = rope_type(len_h=8, len_w=12, **kwargs)
    scaled = rope_type(len_h=4, len_w=4, spatial_scale=(2.0, 3.0), **kwargs)
    for step in (0, 1, 3, 9):
        expected = reference.shift_t(step).reshape(-1, 8, 12, 1, 1, 24)
        expected = expected[:, ::2, ::3].reshape(-1, 1, 1, 24)
        torch.testing.assert_close(scaled.shift_t(step), expected, rtol=2e-7, atol=1e-9)
    unit = rope_type(len_h=4, len_w=4, **kwargs)
    explicit_unit = rope_type(len_h=4, len_w=4, spatial_scale=(1.0, 1.0), **kwargs)
    assert torch.equal(unit.shift_t(3), explicit_unit.shift_t(3))


@pytest.mark.parametrize(
    "scale", [(0, 1), (-1, 1), (1, float("nan")), (float("inf"), 1)]
)
def test_spatial_scale_rejects_invalid_coordinates(scale: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        RotaryPositionEmbedding3D(
            head_dim=24,
            len_t=2,
            len_h=4,
            len_w=4,
            spatial_scale=scale,
            device=torch.device("cpu"),
        )
