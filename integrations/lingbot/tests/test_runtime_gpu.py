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
    LingbotModelAdapter,
    LingbotReplayInputs,
    LingbotReplayRuntime,
    LingbotReplayRuntimeOptions,
    inference_input_from_replay_inputs,
)

from flashdreams.runtime import (
    CanonicalInputs,
    InferenceConfig,
    InferenceInput,
    StepResult,
)

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
    trajectory = np.tile(np.eye(4, dtype=np.float32), (32, 1, 1))
    trajectory[:, 2, 3] = np.arange(32, dtype=np.float32)
    np.save(poses, trajectory)
    np.save(
        intrinsics,
        np.tile(np.array([416.0, 416.0, 416.0, 240.0], dtype=np.float32), (32, 1)),
    )
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
    # Camera inputs now reach the session per step through the mapping, so the
    # GPU path has to be driven the same way the standard loop drives it.
    mapping = LingbotModelAdapter().create_input_mapping(replay_inputs)
    session = runtime.start_session(
        mapping.map_global_conditioning_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=inference_input_from_replay_inputs(replay_inputs),
        )
    )
    try:
        request = session.next_step_request()
        assert request is not None
        result = session.step(
            mapping.map_step_inputs(
                canonical_inputs=CanonicalInputs(),
                inference_input=InferenceInput(),
                request=request,
            )
        )
        torch.cuda.synchronize()
    finally:
        session.close()
        runtime.close()

    assert result.frame_count == 1
    assert isinstance(result, StepResult)
    assert result.video_chunk.is_cuda
    assert result.video_chunk.shape == (1, 3, 2, 2)
    assert pipeline.initialize_cache_devices == ["cuda"]
    assert pipeline.generate_world_scales == [mapping.camera_trace.world_scale]


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


def test_event_driven_camera_control_on_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the CUDA session from key events instead of a fixed pose trace."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")

    from lingbot.input_mapping import (
        FIELD_CAMERA_TRAJECTORY,
        KeyboardToCameraCommand,
    )

    from flashdreams.runtime import (
        InputCanonicalizer,
        UserInputCapability,
        UserInputEvent,
        UserInputs,
        UserInputSchema,
    )

    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    pipeline = _FakeCudaLingbotPipeline()

    def _fake_load_first_frame_tensor(path: Path, **kwargs: Any) -> torch.Tensor:
        del path
        return torch.zeros((1, 3, 2, 2), device=kwargs["device"], dtype=kwargs["dtype"])

    monkeypatch.setattr(
        runtime_module,
        "load_first_frame_tensor",
        _fake_load_first_frame_tensor,
    )

    runtime = LingbotReplayRuntime(
        config=InferenceConfig(model_id=LINGBOT_MODEL_ID, device="cuda"),
        options=LingbotReplayRuntimeOptions(
            pipeline_config=object(),
            pipeline_factory=lambda _pipeline_config, _device: pipeline,
        ),
    )
    adapter = LingbotModelAdapter()
    mapping = adapter.create_live_input_mapping(
        fps=16,
        base_intrinsics=torch.tensor([416.0, 416.0, 416.0, 240.0]),
        world_scale=1.0,
        prompt="drive",
    )
    initial_inputs = mapping.map_global_conditioning_inputs(
        canonical_inputs=CanonicalInputs(),
        inference_input=InferenceInput(
            global_conditioning={
                "prompt": "drive",
                "first_frame_path": image,
                "total_blocks": 1,
                "pixel_height": 2,
                "pixel_width": 2,
                "fps": 16,
            }
        ),
    )
    canonicalizer = InputCanonicalizer([KeyboardToCameraCommand()])
    source = UserInputSchema(
        capabilities=(
            UserInputCapability(
                event_type="key_down", payload_fields=frozenset({"key"})
            ),
            UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"})),
        )
    )
    user_inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.0, event_type="key_down", payload={"key": "w"}
            ),
        )
    )

    session = runtime.start_session(initial_inputs)
    try:
        request = session.next_step_request()
        assert request is not None
        assert request.user_input_window is not None
        step_inputs = mapping.map_step_inputs(
            canonical_inputs=canonicalizer.canonicalize(
                user_inputs,
                window=request.user_input_window,
                source_schema=source,
            ),
            inference_input=InferenceInput(),
            request=request,
        )
        # Holding forward must produce real motion before it reaches the model.
        # This chunk is one frame, so compare against the identity start pose
        # rather than across frames: 0.8 m/s at 16fps advances 0.05 along +z.
        poses = step_inputs.step[FIELD_CAMERA_TRAJECTORY]
        assert poses[-1][2, 3].item() == pytest.approx(0.05, abs=1e-4)
        result = session.step(step_inputs)
        torch.cuda.synchronize()
    finally:
        session.close()
        runtime.close()

    assert isinstance(result, StepResult)
    assert result.video_chunk.is_cuda
    assert pipeline.generate_world_scales == [1.0]
