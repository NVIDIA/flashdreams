# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from alpadreams.webrtc.session import (
    AlpadreamsInferenceRuntime,
    AlpadreamsRuntimeConfig,
)

from flashdreams.serving.webrtc.controls import CameraPoseIntegrator

pytestmark = pytest.mark.ci_cpu


@dataclass
class _FakeOutput:
    state: Any
    rgb_frames: torch.Tensor
    finalization_state: dict[str, int]


class _FakeWrapper:
    initial_frame_chunk_size = 2
    frame_chunk_size = 3

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...], list[int]]] = []
        self.finalized: list[dict[str, int]] = []

    def start_generation(self, **kwargs: Any) -> _FakeOutput:
        poses = kwargs["camera_poses_per_view"]["camera_front_wide_120fov"]
        timestamps = kwargs["frame_timestamps_us"]
        self.calls.append(("start", tuple(poses.shape), timestamps))
        return _FakeOutput(
            state=SimpleNamespace(pipeline_cache=object()),
            rgb_frames=torch.zeros((1, 1, 2, 3, 4, 5), dtype=torch.uint8),
            finalization_state={"autoregressive_index": 0},
        )

    def continue_generation(self, **kwargs: Any) -> _FakeOutput:
        poses = kwargs["camera_poses_per_view"]["camera_front_wide_120fov"]
        timestamps = kwargs["frame_timestamps_us"]
        self.calls.append(("continue", tuple(poses.shape), timestamps))
        return _FakeOutput(
            state=kwargs["state"],
            rgb_frames=torch.zeros((1, 1, 3, 3, 4, 5), dtype=torch.uint8),
            finalization_state={"autoregressive_index": 1},
        )

    def finalize_block_generation(
        self, pipeline_cache: object, finalization_state: dict[str, int]
    ) -> None:
        del pipeline_cache
        self.finalized.append(finalization_state)


def _build_fake_runtime() -> tuple[AlpadreamsInferenceRuntime, _FakeWrapper]:
    runtime = AlpadreamsInferenceRuntime(
        config=AlpadreamsRuntimeConfig(device="cpu", fps=30)
    )
    wrapper = _FakeWrapper()
    runtime._wrapper = wrapper  # ty:ignore[invalid-assignment]
    runtime._renderer = object()
    runtime._initial_rgb_frames = torch.zeros((1, 1, 3, 4, 5), dtype=torch.uint8)
    runtime._text_prompts = []
    runtime._camera_to_rig = torch.eye(4)
    runtime._device = torch.device("cpu")
    runtime._next_timestamp_us = 1000
    runtime.pose_integrator = CameraPoseIntegrator()
    runtime.pose_integrator.reset()
    return runtime, wrapper


def test_generate_chunk_dispatches_start_then_continue() -> None:
    runtime, wrapper = _build_fake_runtime()

    result0 = runtime._generate_one_chunk_sync(
        segments=[(0.0, 2 / 30, frozenset({"w"}))],
        frame_times=[1 / 30, 2 / 30],
    )
    result1 = runtime._generate_one_chunk_sync(
        segments=[(2 / 30, 5 / 30, frozenset())],
        frame_times=[3 / 30, 4 / 30, 5 / 30],
    )

    assert result0.chunk_index == 0
    assert result0.num_frames == 2
    assert result1.chunk_index == 1
    assert result1.num_frames == 3
    assert wrapper.calls[0][0] == "start"
    assert wrapper.calls[0][1] == (2, 4, 4)
    assert wrapper.calls[0][2] == [1000, 34333]
    assert wrapper.calls[1][0] == "continue"
    assert wrapper.calls[1][1] == (3, 4, 4)
    assert len(wrapper.finalized) == 2


def test_prepare_clipgt_dir_stages_unprefixed_parquets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clipgt = tmp_path / "clipgt"
    clipgt.mkdir()
    (clipgt / "calibration_estimate.parquet").touch()
    (clipgt / "egomotion_estimate.parquet").touch()
    (clipgt / "lane.parquet").touch()
    runtime = AlpadreamsInferenceRuntime(
        config=AlpadreamsRuntimeConfig(device="cpu", fps=30)
    )

    staged = runtime._prepare_clipgt_dir(clipgt)

    assert staged != clipgt
    assert (staged / "clip.calibration_estimate.parquet").exists()
    assert (staged / "clip.egomotion_estimate.parquet").exists()
    assert (staged / "clip.lane.parquet").exists()

    monkeypatch.chdir(tmp_path)
    staged_from_relative = runtime._prepare_clipgt_dir(Path("clipgt"))
    assert (staged_from_relative / "clip.calibration_estimate.parquet").exists()
