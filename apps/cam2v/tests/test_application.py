# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the shared camera-to-video v2 application."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import cam2v.session as cam2v_session
import cam2v.ui as cam2v_ui
import pytest
import tomli as tomllib
import torch
from cam2v import (
    Cam2VApplication,
    Cam2VApplicationDefaults,
    Cam2VConditioning,
    Cam2VModelLoop,
    Cam2VModelState,
    Cam2VSession,
    Cam2VSessionConfig,
    Cam2VSlangPyUILoop,
    Cam2VUIState,
    Cam2VUIStatus,
    CameraControlInput,
    KeyboardResampler,
)
from cam2v.dummy import DummyCam2VPipelineConfig
from cam2v.dummy import create_app as create_dummy_app
from numpy import uint64

from flashdreams.runtime_v2.blit_model_output_to_screen_loop import (
    BlitModelOutputToScreenLoop,
)
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.recent_frame_rate import (
    RecentFrameRateSnapshot,
    RecentFrameRateTracker,
)
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _recent_model_rate_snapshot(
    frame_count: int = 13,
) -> RecentFrameRateSnapshot:
    tracker = RecentFrameRateTracker(window_seconds=2.0)
    tracker.observe(
        completed_at=time.perf_counter(),
        frame_count=frame_count,
        elapsed_s=1.0,
    )
    return tracker.snapshot()


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


def test_model_loop_maps_wasd_to_shared_camera_input_and_metrics() -> None:
    """Keep keyboard-to-pose conversion outside concrete integrations."""
    pipeline = _Pipeline()
    ui_state = Cam2VUIState(total_blocks=1, target_fps=16, warmup_blocks=0)
    shutdown_event = threading.Event()
    failure_queue: queue.Queue[BaseException] = queue.Queue()
    presentation_manager = PresentationManager()
    ui_loop = Cam2VSlangPyUILoop(renderer=Mock())
    ui_loop.register_session_loop_objects(
        state=ui_state,
        frequency=60,
        shutdown_event=shutdown_event,
        failure_queue=failure_queue,
    )
    ui_loop.register_session_ui_loop_objects(
        output_layout=VideoTensorLayout.tchw,
        presentation_manager=presentation_manager,
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
            first_frame_dtype=torch.float32,
            first_frame_interpolation="linear",
            warmup_blocks=0,
        ),
        keyboard_resampler=KeyboardResampler(fps=16),
        cache=object(),
        ui_loop=ui_loop,
    )
    model_loop = Cam2VModelLoop()
    model_loop.register_session_loop_objects(
        state=state,
        frequency=16,
        shutdown_event=shutdown_event,
        failure_queue=failure_queue,
    )
    events = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key="w",
                state=KeyboardInputState.PRESSED,
            )
        ]
    )

    result = model_loop.step(0, events)[0]

    assert result.frame_count == 2
    assert result.metrics["model_step_s"] == 1.0
    assert result.metrics["steady_state_fps"] > 0
    assert result.metrics["recent_model_fps"] == pytest.approx(
        result.metrics["chunk_fps"]
    )
    assert result.metrics["model_step_wall_s"] > 0
    ui_loop._run_message_batch()
    assert ui_state.status is not None
    assert ui_state.status.completed_blocks == 1
    assert ui_state.status.frames_generated == 2
    assert ui_state.status.recent_model_rate_snapshot is not None
    assert ui_state.status.recent_model_fps() == pytest.approx(
        result.metrics["recent_model_fps"]
    )
    assert pipeline.camera_input is not None
    assert pipeline.camera_input.poses.shape == (2, 4, 4)
    assert pipeline.camera_input.poses[-1, 2, 3] > 0
    assert model_loop.is_finished()


def _input_test_model_loop(
    *, log_model_timing: bool = False
) -> tuple[Cam2VModelLoop, Cam2VModelState, _Pipeline]:
    """Return a registered CPU model loop for camera-input tests."""
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
            total_blocks=4,
            device=torch.device("cpu"),
            first_frame_dtype=torch.float32,
            first_frame_interpolation="linear",
            warmup_blocks=4,
            log_model_timing=log_model_timing,
        ),
        keyboard_resampler=KeyboardResampler(fps=16),
        cache=object(),
    )
    model_loop = Cam2VModelLoop()
    model_loop.register_session_loop_objects(
        state=state,
        frequency=16,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    return model_loop, state, pipeline


def test_model_loop_logs_each_ar_step_wall_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep useful model timing visible without the synchronous stage profiler."""
    timing_logger = Mock()
    monkeypatch.setattr(cam2v_session, "logger", timing_logger)
    quiet_model_loop, _, _ = _input_test_model_loop()
    quiet_model_loop.step(0, UserInputEvents([]))
    timing_logger.info.assert_not_called()

    model_loop, _, _ = _input_test_model_loop(log_model_timing=True)

    model_loop.step(0, UserInputEvents([]))

    timing_logger.info.assert_called_once()
    message, step_index, phase, frame_count, wall_ms, chunk_fps = (
        timing_logger.info.call_args.args
    )
    assert message == (
        "Cam2V AR {} [{}] | {} frames | step wall {:.1f} ms | {:.2f} fps"
    )
    assert step_index == 0
    assert phase == "warmup"
    assert frame_count == 2
    assert wall_ms > 0.0
    assert chunk_fps == pytest.approx(2_000.0 / wall_ms)


def test_keyboard_resampler_keeps_an_aliased_key_held_until_all_sources_release() -> (
    None
):
    resampler = KeyboardResampler(fps=10)
    resampler.on_edge(arrival_t=0.0, event="keydown", key="w")
    resampler.on_edge(arrival_t=0.01, event="keydown", key="ArrowUp")
    resampler.on_edge(arrival_t=0.02, event="keyup", key="ArrowUp")

    segments, _ = resampler.sample_chunk(1)

    assert segments[-1][2] == frozenset({"w"})


def test_ui_recent_model_frame_rate_reaches_zero_during_a_stall() -> None:
    rate = RecentFrameRateTracker(window_seconds=2.0)
    rate.observe(completed_at=0.5, frame_count=5, elapsed_s=0.5)
    rate.observe(completed_at=1.5, frame_count=20, elapsed_s=1.0)
    status = Cam2VUIStatus(
        completed_blocks=2,
        frames_generated=25,
        chunk_fps=20.0,
        recent_model_rate_snapshot=rate.snapshot(),
        model_step_wall_s=1.0,
    )

    assert status.recent_model_fps(now=1.5) == pytest.approx(50.0 / 3.0)
    assert status.recent_model_fps(now=2.0) == pytest.approx(50.0 / 3.0)
    assert status.recent_model_fps(now=3.6) == 0.0


def test_slangpy_continuous_redraw_expires_recent_model_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refresh the recent model rate on every continuous UI step."""
    rate = RecentFrameRateTracker(window_seconds=2.0)
    rate.observe(completed_at=0.0, frame_count=13, elapsed_s=1.0)
    ui_loop = Cam2VSlangPyUILoop(renderer=Mock())
    state = Cam2VUIState(total_blocks=4, target_fps=16, warmup_blocks=0)
    state.update_status(
        Cam2VUIStatus(
            completed_blocks=1,
            frames_generated=13,
            chunk_fps=13.0,
            recent_model_rate_snapshot=rate.snapshot(),
            model_step_wall_s=1.0,
        )
    )
    ui_loop.register_session_loop_objects(
        state=state,
        frequency=60,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    ui_loop.register_session_ui_loop_objects(
        output_layout=VideoTensorLayout.tchw,
        presentation_manager=PresentationManager(),
    )
    ui = SimpleNamespace(
        screen=object(),
        Window=Mock(return_value=object()),
        Text=Mock(side_effect=lambda parent, text: SimpleNamespace(text=text)),
    )
    monkeypatch.setattr(cam2v_ui.time, "perf_counter", lambda: 1.9)

    ui_loop.step_ui(ui, 0, UserInputEvents([]))

    assert "Recent model rate (2 s): 13.00 FPS" in (
        widget.text for widget in state.status_widgets
    )
    monkeypatch.setattr(cam2v_ui.time, "perf_counter", lambda: 2.4)
    ui_loop.step_ui(ui, 1, UserInputEvents([]))
    assert "Recent model rate (2 s): 0.00 FPS" in (
        widget.text for widget in state.status_widgets
    )


def test_model_loop_preserves_a_quick_tap_after_wall_clock_stall() -> None:
    """Apply a short press in the next chunk even when the model clock lags."""
    model_loop, state, pipeline = _input_test_model_loop()
    events = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(10_000_000),
                key="w",
                state=KeyboardInputState.PRESSED,
            ),
            KeyboardUserInputEvent(
                timestamp=uint64(10_080_000),
                key="w",
                state=KeyboardInputState.RELEASED,
            ),
        ]
    )

    model_loop.step(0, events)

    assert pipeline.camera_input is not None
    assert pipeline.camera_input.poses[-1, 2, 3].item() == pytest.approx(0.064)
    assert state.input_timeline.next_window_start_s == pytest.approx(10.125)


def test_unsupported_key_does_not_advance_the_camera_timeline() -> None:
    model_loop, state, _ = _input_test_model_loop()
    start = state.input_timeline.next_window_start_s

    model_loop.step(
        0,
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=uint64(10_000_000),
                    key="Escape",
                    state=KeyboardInputState.PRESSED,
                )
            ]
        ),
    )

    assert state.input_timeline.next_window_start_s == pytest.approx(start + 0.125)


def test_model_loop_releases_camera_controls_when_browser_loses_focus() -> None:
    """Stop retained movement at the timestamped browser focus-loss edge."""
    model_loop, _, pipeline = _input_test_model_loop()
    events = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key="w",
                state=KeyboardInputState.PRESSED,
            ),
            FocusUserInputEvent(
                timestamp=uint64(50_000),
                focused=False,
            ),
        ]
    )

    model_loop.step(0, events)

    assert pipeline.camera_input is not None
    poses = pipeline.camera_input.poses
    assert poses[0, 2, 3].item() == pytest.approx(0.04)
    assert poses[1, 2, 3].item() == pytest.approx(0.04)


def test_model_loop_normalizes_browser_arrow_keys() -> None:
    """Use the shared keyboard normalization without an input registry."""
    model_loop, _, pipeline = _input_test_model_loop()
    events = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key="ArrowUp",
                state=KeyboardInputState.PRESSED,
            )
        ]
    )

    model_loop.step(0, events)

    assert pipeline.camera_input is not None
    assert pipeline.camera_input.poses[-1, 2, 3].item() == pytest.approx(0.1)


def test_slangpy_overlay_tracks_controls_and_model_status() -> None:
    """Keep immediate input display in UI-loop-owned state."""
    state = Cam2VUIState(total_blocks=4, target_fps=16, warmup_blocks=1)
    presentation_manager = PresentationManager()
    presentation_manager.configure(
        backpressure_mode=BackpressureMode.BLOCK,
        stop=threading.Event(),
        put_timeout=0.01,
    )
    presentation_manager.publish(
        0,
        [
            StepResult(
                step_index=0,
                output=torch.zeros((3, 3, 2, 2), dtype=torch.bfloat16),
                frame_count=3,
                output_layout=VideoTensorLayout.tchw,
            )
        ],
    )
    assert presentation_manager.advance(0, now=1.0)[0]
    ui = SimpleNamespace(
        screen=object(),
        Window=Mock(return_value=object()),
        Text=Mock(side_effect=lambda parent, text: SimpleNamespace(text=text)),
    )
    renderer = Mock()

    def render(
        step_index: int,
        events: UserInputEvents,
        draw: Any,
    ) -> torch.Tensor:
        draw(ui, step_index, events)
        return torch.zeros((4, 2, 2), dtype=torch.float32)

    renderer.render.side_effect = render
    ui_loop = Cam2VSlangPyUILoop(renderer=renderer)
    ui_loop.register_session_loop_objects(
        state=state,
        frequency=60,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    ui_loop.register_session_ui_loop_objects(
        output_layout=VideoTensorLayout.tchw,
        presentation_manager=presentation_manager,
    )
    pressed = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key="ArrowUp",
                state=KeyboardInputState.PRESSED,
            )
        ]
    )
    state.update_status(
        Cam2VUIStatus(
            completed_blocks=2,
            frames_generated=24,
            chunk_fps=13.5,
            recent_model_rate_snapshot=_recent_model_rate_snapshot(),
            model_step_wall_s=0.89,
        )
    )

    result = ui_loop.step(0, pressed)
    output = result.read_output()

    assert output.shape == (1, 3, 2, 2)
    assert output.dtype is torch.bfloat16
    assert state.held_keys == {"w"}
    displayed = [widget.text for widget in state.status_widgets]
    assert "Rollout: 2/4 blocks" in displayed
    assert "Presented: 1 frames (24 generated)" in displayed
    assert "Latest model rate: 13.50 FPS" in displayed
    assert any(line.startswith("Recent model rate (2 s):") for line in displayed)
    assert state.active_keys_widget is not None
    assert state.active_keys_widget.text == "Active keys: W"

    ui_loop.step(
        1,
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=uint64(1),
                    key="e",
                    state=KeyboardInputState.PRESSED,
                ),
                KeyboardUserInputEvent(
                    timestamp=uint64(2),
                    key="e",
                    state=KeyboardInputState.RELEASED,
                ),
            ]
        ),
    )
    assert state.held_keys == {"w"}

    ui_loop.step(
        2,
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=uint64(3),
                    key="w",
                    state=KeyboardInputState.PRESSED,
                ),
                KeyboardUserInputEvent(
                    timestamp=uint64(4),
                    key="ArrowUp",
                    state=KeyboardInputState.RELEASED,
                ),
            ]
        ),
    )
    assert state.held_keys == {"w"}
    ui_loop.step(
        3,
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=uint64(5),
                    key="w",
                    state=KeyboardInputState.RELEASED,
                )
            ]
        ),
    )
    assert not state.held_keys

    ui_loop.step(
        4,
        UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=uint64(6),
                    key="e",
                    state=KeyboardInputState.PRESSED,
                ),
                KeyboardUserInputEvent(
                    timestamp=uint64(7),
                    key="i",
                    state=KeyboardInputState.PRESSED,
                ),
            ]
        ),
    )
    assert state.held_keys == {"e", "i"}
    ui_loop.step(
        5,
        UserInputEvents([FocusUserInputEvent(timestamp=uint64(8), focused=False)]),
    )
    assert not state.held_keys
    assert state.active_keys_widget.text == "Active keys: none"

    assert presentation_manager.advance(0, now=1.1)[0]
    ui_loop.step(6, UserInputEvents([]))
    displayed = [widget.text for widget in state.status_widgets]
    assert "Presented: 2 frames (24 generated)" in displayed


def test_cam2v_session_registers_the_shared_slangpy_ui_loop() -> None:
    """Construct the overlay at the shared session boundary."""
    session = Cam2VSession(
        pipeline=_Pipeline(),
        session_desc=SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            backpressure_mode=BackpressureMode.BLOCK,
            presentation_mode=PresentationMode.CONTINUOUS,
            frames_per_second_for_step=16,
            video_width=8,
            video_height=4,
        ),
        config=Cam2VSessionConfig(
            conditioning=_conditioning(),
            total_blocks=2,
            device=torch.device("cpu"),
            first_frame_dtype=torch.float32,
            first_frame_interpolation="linear",
            warmup_blocks=0,
        ),
    )

    session.init()

    assert isinstance(session.ui_loop, Cam2VSlangPyUILoop)
    assert session.ui_loop.state.total_blocks == 2
    assert session.model_loop.state.input_timeline.samples_per_second == 16
    assert session.model_loop.state.keyboard_resampler is not None
    assert session.model_loop.state.keyboard_resampler.fps == 16
    assert session.session_desc.backpressure_mode is BackpressureMode.BLOCK
    assert session.session_desc.presentation_mode is PresentationMode.CONTINUOUS


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
            first_frame_dtype=torch.float64,
            first_frame_interpolation="nearest",
            device="cpu",
            fps=16,
        )
    )
    app.init(["--total-blocks", "2", "--warmup-blocks", "0", "--no-ui"])

    session = app.create_session(app.session_desc())

    assert isinstance(session, Cam2VSession)
    session.init()
    ui_loop, model_loop = session._take_loops()
    assert session.session_desc.video_width == 8
    assert isinstance(ui_loop, BlitModelOutputToScreenLoop)
    assert seen[0]["pixel_width"] == 8
    assert seen[0]["pixel_height"] == 4
    assert seen[0]["fps"] == 16
    assert pipeline_config.pipeline.device == "cpu"
    assert model_loop.state.config.first_frame_dtype is torch.float64
    assert model_loop.state.config.first_frame_interpolation == "nearest"
    app.close()
    assert pipeline_config.pipeline.closed


def test_defaults_reject_invalid_timing_configuration() -> None:
    """Fail before model construction when timing defaults are invalid."""
    invalid_log_model_timing: Any = 1
    with pytest.raises(ValueError, match="warmup_blocks"):
        Cam2VApplicationDefaults(
            pipeline_config=object(),
            input_resolver=lambda values: _conditioning(),
            total_blocks=1,
            pixel_width=1,
            pixel_height=1,
            first_frame_dtype=torch.float32,
            first_frame_interpolation="default",
            warmup_blocks=-1,
        )
    with pytest.raises(TypeError, match="log_model_timing"):
        Cam2VApplicationDefaults(
            pipeline_config=object(),
            input_resolver=lambda values: _conditioning(),
            total_blocks=1,
            pixel_width=1,
            pixel_height=1,
            first_frame_dtype=torch.float32,
            first_frame_interpolation="default",
            log_model_timing=invalid_log_model_timing,
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
    assert app.session_desc().backpressure_mode is BackpressureMode.BLOCK
    assert app.session_desc().presentation_mode is PresentationMode.CONTINUOUS
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
