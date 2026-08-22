# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from omnidreams_game_engine.motion_conformance import compare_motion

pytestmark = pytest.mark.ci_cpu


def _translated_frames(step_px: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, size=(88, 160, 3), dtype=np.uint8)
    return tuple(
        cv2.warpAffine(
            base,
            np.asarray([[1.0, 0.0, float(index * step_px)], [0.0, 1.0, 0.0]]),
            (160, 88),
            borderMode=cv2.BORDER_REFLECT,
        )
        for index in range(8)
    )


def _zoom_frames(scale_step: float) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, size=(88, 160, 3), dtype=np.uint8)
    center = (79.5, 43.5)
    return tuple(
        cv2.warpAffine(
            base,
            cv2.getRotationMatrix2D(center, 0.0, 1.0 + index * scale_step),
            (160, 88),
            borderMode=cv2.BORDER_REFLECT,
        )
        for index in range(8)
    )


def test_motion_conformance_accepts_matching_turn_flow() -> None:
    frames = _translated_frames(2)

    result = compare_motion(
        frames,
        frames,
        yaw_delta_rad=0.1,
        longitudinal_delta_m=0.0,
    )

    assert result.axis == "turn"
    assert result.mismatched is False


def test_motion_conformance_rejects_opposite_turn_flow() -> None:
    result = compare_motion(
        _translated_frames(2),
        _translated_frames(-2),
        yaw_delta_rad=0.1,
        longitudinal_delta_m=0.0,
    )

    assert result.axis == "turn"
    assert result.mismatched is True


def test_motion_conformance_rejects_missing_turn_flow() -> None:
    stationary = _translated_frames(0)
    result = compare_motion(
        _translated_frames(2),
        stationary,
        yaw_delta_rad=0.1,
        longitudinal_delta_m=0.0,
    )

    assert result.mismatched is True


def test_motion_conformance_rejects_opposite_longitudinal_flow() -> None:
    result = compare_motion(
        _zoom_frames(0.02),
        _zoom_frames(-0.02),
        yaw_delta_rad=0.0,
        longitudinal_delta_m=1.0,
    )

    assert result.axis == "longitudinal"
    assert result.mismatched is True


def test_motion_conformance_skips_low_motion_chunks() -> None:
    result = compare_motion(
        _translated_frames(0),
        _translated_frames(-2),
        yaw_delta_rad=0.0,
        longitudinal_delta_m=0.0,
    )

    assert result.axis == "none"
    assert result.mismatched is False


def test_motion_conformance_downsamples_lazy_tensor_without_host_materialization() -> (
    None
):
    class _LazyFrame:
        def __init__(self, value: np.ndarray) -> None:
            self.tensor = torch.from_numpy(value)
            self.host_materialized = False

        def to_cuda_tensor(self) -> torch.Tensor:
            return self.tensor

        def to_cuda_event(self) -> None:
            return None

        def to_numpy(self) -> np.ndarray:
            self.host_materialized = True
            return self.tensor.numpy()

    condition = tuple(_LazyFrame(frame) for frame in _translated_frames(2))
    generated = tuple(_LazyFrame(frame) for frame in _translated_frames(2))

    result = compare_motion(
        condition,
        generated,
        yaw_delta_rad=0.1,
        longitudinal_delta_m=0.0,
    )

    assert result.mismatched is False
    assert not any(frame.host_materialized for frame in condition + generated)
