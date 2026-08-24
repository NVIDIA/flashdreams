# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the shared camera-to-video v2 application."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from cam2v import (
    Cam2VApplication,
    Cam2VApplicationDefaults,
    Cam2VConditioning,
    Cam2VModelState,
    Cam2VModelThread,
    Cam2VSessionConfig,
    CameraControlInput,
)
from numpy import uint64

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


class _Decoder:
    spatial_compression_ratio = 1


class _Pipeline:
    """Small CPU pipeline recording shared Cam2V inputs and lifecycle calls."""

    def __init__(self) -> None:
        self.decoder = _Decoder()
        self.camera_input: CameraControlInput | None = None
        self.device: str | None = None
        self.closed = False

    def to(self, device: str) -> "_Pipeline":
        """Record application-owned device placement."""
        self.device = device
        return self

    def eval(self) -> "_Pipeline":
        """Match the real pipeline construction chain."""
        return self

    def get_num_output_frames(self, step_index: int) -> int:
        """Return a two-frame chunk for each model-generation step."""
        del step_index
        return 2

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: object,
        input: CameraControlInput,
    ) -> torch.Tensor:
        """Record camera conditioning and return a deterministic chunk."""
        del autoregressive_index, cache
        self.camera_input = input
        return torch.zeros((2, 3, 1, 1), dtype=torch.float32)

    def finalize(
        self,
        *,
        autoregressive_index: int,
        cache: object,
    ) -> dict[str, float]:
        """Return one model-provided timing metric."""
        del autoregressive_index, cache
        return {"model_step_s": 1.0}

    def close(self) -> None:
        """Record application cleanup."""
        self.closed = True


class _PipelineConfig:
    """Return one retained stand-in pipeline from ``setup``."""

    def __init__(self) -> None:
        self.pipeline = _Pipeline()

    def setup(self) -> _Pipeline:
        """Return the application-owned pipeline."""
        return self.pipeline


def _conditioning() -> Cam2VConditioning:
    return Cam2VConditioning(
        prompt="camera demo",
        first_frame_path=Path("first.jpg"),
        base_intrinsics=torch.tensor([1.0, 1.0, 0.5, 0.5]),
        world_scale=1.0,
    )


def test_model_thread_maps_wasd_to_shared_camera_input_and_metrics() -> None:
    """Keep keyboard-to-pose conversion outside concrete integrations."""
    pipeline = _Pipeline()
    state = Cam2VModelState(
        pipeline=pipeline,
        session_desc=SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            frames_per_second_for_step=16,
            video_width=1,
            video_height=1,
        ),
        config=Cam2VSessionConfig(
            conditioning=_conditioning(),
            total_blocks=1,
            device=torch.device("cpu"),
            log_every_blocks=1,
            warmup_blocks=0,
        ),
        cache=object(),
    )
    thread = Cam2VModelThread(state=state, frequency=16)
    events = UserInputEvents(
        [
            UserInputEvent(
                timestamp=uint64(0),
                event_data=KeyboardUserInputEventData(
                    key="w",
                    state=KeyboardInputState.PRESSED,
                ),
            )
        ]
    )

    result = thread.step(0, events)[0]

    assert result.frame_count == 2
    assert result.metrics["model_step_s"] == 1.0
    assert result.metrics["steady_state_fps"] > 0
    assert result.metrics["model_step_wall_s"] > 0
    assert pipeline.camera_input is not None
    assert pipeline.camera_input.poses.shape == (2, 4, 4)
    assert pipeline.camera_input.poses[-1, 2, 3] > 0
    assert thread.is_finished()


def test_application_owns_pipeline_and_resolves_inputs_per_session_desc() -> None:
    """Keep the loaded model application-scoped and rollout inputs session-scoped."""
    pipeline_config = _PipelineConfig()
    seen: list[Mapping[str, Any]] = []

    def resolve(values: Mapping[str, Any]) -> Cam2VConditioning:
        seen.append(values)
        return _conditioning()

    app = Cam2VApplication(
        defaults=Cam2VApplicationDefaults(
            pipeline_config=pipeline_config,
            input_resolver=resolve,
            total_blocks=3,
            pixel_width=8,
            pixel_height=4,
            device="cpu",
            fps=16,
        )
    )
    app.init(["--total-blocks", "2", "--warmup-blocks", "0"])

    session = app.create_session(app.session_desc())

    assert session.session_desc.video_width == 8
    assert seen[0]["pixel_width"] == 8
    assert seen[0]["pixel_height"] == 4
    assert seen[0]["fps"] == 16
    assert pipeline_config.pipeline.device == "cpu"
    app.close()
    assert pipeline_config.pipeline.closed


def test_defaults_reject_invalid_timing_configuration() -> None:
    """Fail before model construction when timing defaults are invalid."""
    with pytest.raises(ValueError, match="warmup_blocks"):
        Cam2VApplicationDefaults(
            pipeline_config=object(),
            input_resolver=lambda values: _conditioning(),
            total_blocks=1,
            pixel_width=1,
            pixel_height=1,
            warmup_blocks=-1,
        )
