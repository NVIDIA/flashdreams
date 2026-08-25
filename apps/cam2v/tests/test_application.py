# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the shared camera-to-video v2 application."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import cam2v.ui as cam2v_ui
import pytest
import tomli as tomllib
import torch
from cam2v import (
    Cam2VApplication,
    Cam2VApplicationDefaults,
    Cam2VConditioning,
    Cam2VImGUIThread,
    Cam2VModelState,
    Cam2VModelThread,
    Cam2VSession,
    Cam2VSessionConfig,
    Cam2VUIState,
    Cam2VUIStatus,
    CameraControlInput,
)
from cam2v.dummy import DummyCam2VPipelineConfig
from cam2v.dummy import create_app as create_dummy_app
from numpy import uint64

from flashdreams.api_v2.thread import BlitModelOutputToScreenThread
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
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
    ui_state = Cam2VUIState(total_blocks=1, target_fps=16, warmup_blocks=0)
    ui_thread = Cam2VImGUIThread(
        state=ui_state,
        frequency=60,
        output_layout=VideoTensorLayout.tchw,
        presentation_manager=PresentationManager(),
        renderer=Mock(),
    )
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
        ui_thread=ui_thread,
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
    ui_thread._run_message_batch()
    assert ui_state.status is not None
    assert ui_state.status.completed_blocks == 1
    assert ui_state.status.frames_generated == 2
    assert pipeline.camera_input is not None
    assert pipeline.camera_input.poses.shape == (2, 4, 4)
    assert pipeline.camera_input.poses[-1, 2, 3] > 0
    assert thread.is_finished()


def test_imgui_overlay_tracks_controls_and_model_status(monkeypatch: Any) -> None:
    """Keep immediate input display in UI-thread-owned state."""
    logger = Mock()
    monkeypatch.setattr(cam2v_ui, "logger", logger)
    state = Cam2VUIState(total_blocks=4, target_fps=16, warmup_blocks=1)
    presentation_manager = PresentationManager()
    presentation_manager.publish(
        0,
        [
            StepResult(
                step_index=0,
                output=torch.zeros((1, 3, 2, 2), dtype=torch.bfloat16),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
        ],
    )
    assert presentation_manager.advance(0)[0]
    thread = Cam2VImGUIThread(
        state=state,
        frequency=60,
        output_layout=VideoTensorLayout.tchw,
        presentation_manager=presentation_manager,
        renderer=Mock(),
    )
    imgui = Mock()
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
    state.update_status(
        Cam2VUIStatus(
            completed_blocks=2,
            frames_generated=24,
            chunk_fps=13.5,
            steady_state_fps=13.25,
            model_step_wall_s=0.89,
        )
    )

    back_buffer = thread.draw_ui(imgui, 0, events)

    assert back_buffer is not None
    assert back_buffer.dtype is torch.float32
    displayed = [call.args[0] for call in imgui.text.call_args_list]
    assert "Rollout: 2/4 blocks" in displayed
    assert "Latest model rate: 13.50 FPS" in displayed
    assert "Active keys: W" in displayed
    logger.info.assert_called_once_with(
        "Cam2V ImGui UI-thread processed keyboard event "
        "key={} state={} timestamp_us={} held_keys={}",
        "w",
        "Pressed",
        0,
        "w",
    )


def test_cam2v_session_registers_the_shared_imgui_thread() -> None:
    """Construct the overlay at the shared session boundary."""
    session = Cam2VSession(
        pipeline=_Pipeline(),
        session_desc=SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            frames_per_second_for_step=16,
            video_width=8,
            video_height=4,
        ),
        config=Cam2VSessionConfig(
            conditioning=_conditioning(),
            total_blocks=2,
            device=torch.device("cpu"),
            log_every_blocks=1,
            warmup_blocks=0,
        ),
    )

    session.init()

    assert isinstance(session.ui_thread, Cam2VImGUIThread)
    assert session.ui_thread.state.total_blocks == 2


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
    app.init(["--total-blocks", "2", "--warmup-blocks", "0", "--no-ui"])

    session = app.create_session(app.session_desc())

    assert isinstance(session, Cam2VSession)
    session.init()
    ui_thread, _ = session._take_threads()
    assert session.session_desc.video_width == 8
    assert isinstance(ui_thread, BlitModelOutputToScreenThread)
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


def test_dummy_cam2v_pipeline_simulates_generation_without_a_model() -> None:
    """Exercise camera-dependent dummy output without image or GPU dependencies."""
    pipeline = DummyCam2VPipelineConfig(
        step_wait_seconds=0.0,
        frames_per_chunk=2,
    ).setup()
    cache = pipeline.initialize_cache(
        text=["dummy"],
        image=torch.zeros((1, 3, 4, 8), dtype=torch.float32),
    )
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[:, 0, 3] = 1.0

    frames = pipeline.generate(
        autoregressive_index=0,
        cache=cache,
        input=CameraControlInput(
            intrinsics=torch.zeros((2, 4)),
            poses=poses,
            world_scale=1.0,
        ),
    )

    assert frames.shape == (2, 3, 4, 8)
    assert frames[:, 0].mean() > frames[:, 1].mean()


def test_dummy_cam2v_application_exposes_slow_step_controls() -> None:
    """Keep synthetic latency configurable through application arguments."""
    app = create_dummy_app()

    app.init(
        [
            "--step-wait-seconds",
            "0.25",
            "--frames-per-chunk",
            "3",
            "--total-blocks",
            "1",
        ]
    )

    assert isinstance(app, Cam2VApplication)
    assert app.pipeline_config == DummyCam2VPipelineConfig(
        step_wait_seconds=0.25,
        frames_per_chunk=3,
    )


def test_shared_cam2v_package_registers_the_dummy_application() -> None:
    """Expose the slow dummy through the v2 application registry."""
    manifest = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert (
        manifest["project"]["entry-points"]["flashdreams.applications_v2"][
            "cam2v-dummy"
        ]
        == "cam2v.dummy:create_app"
    )
