# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from lingbot import runtime as runtime_module
from lingbot.runtime import (
    LINGBOT_MODEL_ID,
    LingbotReplayInputs,
    LingbotReplayRuntime,
    LingbotReplayRuntimeOptions,
    inference_input_from_replay_inputs,
)

from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import InferenceConfig, InferenceInput

pytestmark = pytest.mark.ci_gpu


def test_lingbot_replay_runtime_accepts_direct_inputs_on_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the migrated Lingbot runtime API path with CUDA tensors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")

    image = tmp_path / "image.jpg"
    poses = tmp_path / "poses.npy"
    intrinsics = tmp_path / "intrinsics.npy"
    image.write_bytes(b"fake")
    np.save(poses, np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)))
    np.save(intrinsics, np.ones((2, 4), dtype=np.float32))
    pipeline = _FakeCudaLingbotPipeline()

    def _fake_load_first_frame_tensor(
        path: Path,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert path == image
        return torch.zeros(
            (1, 3, 2, 2),
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )

    monkeypatch.setattr(
        runtime_module,
        "load_first_frame_tensor",
        _fake_load_first_frame_tensor,
    )
    monkeypatch.setattr(
        runtime_module,
        "get_Ks_transformed",
        lambda intrinsics_t, **_kwargs: intrinsics_t,
    )
    monkeypatch.setattr(
        runtime_module,
        "preprocess_example_poses",
        lambda c2ws: (c2ws, 2.5),
    )

    runtime = LingbotReplayRuntime(
        config=InferenceConfig(model_id=LINGBOT_MODEL_ID, device="cuda"),
        options=LingbotReplayRuntimeOptions(
            pipeline_config=object(),
            pipeline_factory=lambda _pipeline_config, _device: pipeline,
        ),
    )
    replay_inputs = LingbotReplayInputs(
        prompt="drive",
        first_frame_path=image,
        camera_poses_path=poses,
        camera_intrinsics_path=intrinsics,
        total_blocks=1,
        pixel_height=2,
        pixel_width=2,
        fps=16,
    )
    session = runtime.start_session(inference_input_from_replay_inputs(replay_inputs))
    try:
        result = session.step(InferenceInput())
        torch.cuda.synchronize()
    finally:
        session.close()
        runtime.close()

    assert result.frame_count == 1
    assert isinstance(result.output, VideoStepResult)
    assert result.output.video_chunk.is_cuda
    assert result.output.video_chunk.shape == (1, 3, 2, 2)
    assert pipeline.initialize_cache_devices == ["cuda"]
    assert pipeline.generate_world_scales == [2.5]


class _FakeCudaLingbotPipeline:
    def __init__(self) -> None:
        self.initialize_cache_devices: list[str] = []
        self.generate_world_scales: list[float] = []

    def initialize_cache(self, *, text: list[str], image: torch.Tensor) -> object:
        assert text == ["drive"]
        assert image.is_cuda
        self.initialize_cache_devices.append(image.device.type)
        return object()

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        assert autoregressive_index == 0
        return 1

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: object,
        input: Any,
    ) -> torch.Tensor:
        del cache
        assert autoregressive_index == 0
        assert input.intrinsics.is_cuda
        assert input.poses.is_cuda
        self.generate_world_scales.append(input.world_scale)
        return torch.zeros(
            (1, 3, 2, 2),
            device=input.intrinsics.device,
            dtype=torch.bfloat16,
        )

    def finalize(self, *, autoregressive_index: int, cache: object) -> dict[str, float]:
        del autoregressive_index, cache
        return {"denoise_s": 0.25}
