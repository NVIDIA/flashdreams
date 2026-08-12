# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lingbot.demo.app as demo_app_module
import numpy as np
import pytest
import torch
from aiohttp import web
from lingbot.config import RUNNER_LINGBOT_WORLD_FAST
from lingbot.demo import (
    DEFAULT_LINGBOT_PRESET,
    LINGBOT_MODEL_ID,
    LingbotDemoAdapter,
    LingbotInputProvider,
    LingbotReplayInputs,
    LingbotWebRTCScenario,
)
from lingbot.demo.app import _replay_spec, _webrtc_spec, parse_args
from lingbot.demo.providers import PROVIDER_INPUTS_METADATA_KEY
from lingbot.demo.replay import (
    LingbotReplayRuntime,
    LingbotReplayRuntimeOptions,
)
from lingbot.demo.webrtc import serve_lingbot_webrtc_demo
from lingbot.input_mapping import (
    FIELD_CAMERA_INTRINSICS,
    FIELD_CAMERA_TRAJECTORY,
)
from lingbot.runtime import (
    FIELD_FIRST_FRAME_PATH,
    FIELD_FPS,
    FIELD_PIXEL_HEIGHT,
    FIELD_PIXEL_WIDTH,
    FIELD_PROMPT,
    FIELD_TOTAL_BLOCKS,
    inference_input_from_replay_inputs,
)
from lingbot.webrtc.session import (
    LingbotRuntimeConfig,
    LingbotSessionInput,
    TextEventSpec,
)

from flashdreams.runtime import (
    CanonicalInputs,
    InferenceConfig,
    InferenceInput,
    OutputArtifact,
    OutputTarget,
    StepRequest,
    StepRequirements,
    StepResult,
    UserInputEvent,
    UserInputs,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    UserInputWindow,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.server import SESSION_MANAGER_KEY

pytestmark = pytest.mark.ci_cpu


def _write_camera_assets(poses: Path, intrinsics: Path, *, frames: int = 64) -> None:
    """Write real .npy camera assets; the input mapping loads them for real."""
    trajectory = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    trajectory[:, 2, 3] = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    np.save(poses, trajectory)
    np.save(
        intrinsics,
        np.tile(np.array([416.0, 416.0, 416.0, 240.0], dtype=np.float32), (frames, 1)),
    )


def _patch_lingbot_webrtc_example(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    example_idx: int = 0,
) -> Path:
    """Provide a local example-data directory for shared WebRTC preparation."""
    import lingbot.runtime as runtime_module

    example_dir = tmp_path / f"example-{example_idx:02d}"
    example_dir.mkdir()
    (example_dir / "image.jpg").write_bytes(b"fake")
    _write_camera_assets(example_dir / "poses.npy", example_dir / "intrinsics.npy")
    (example_dir / "prompt.txt").write_text("drive through a forest\n")

    def fake_download(*, is_rank_zero: bool, example_idx: int) -> Path:
        del is_rank_zero, example_idx
        return example_dir

    monkeypatch.setattr(
        runtime_module,
        "ensure_example_data_downloaded",
        fake_download,
    )
    return example_dir


def test_lingbot_demo_defaults_to_interactive_preset() -> None:
    args = parse_args(["replay", "--output", "demo.mp4"])

    assert args.preset_id == "lingbot-world-fast-taehv-window15-sink3"


def test_lingbot_direct_runner_launch_builds_mp4_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[DemoSpec] = []

    def fake_run_replay_demo(*, spec: DemoSpec, adapter: object) -> str:
        del adapter
        captured.append(spec)
        return "completed"

    monkeypatch.setattr(demo_app_module, "run_replay_demo", fake_run_replay_demo)

    result = demo_app_module.launch_from_runner(
        config=RUNNER_LINGBOT_WORLD_FAST,
        mode="mp4",
        scenario={"example_idx": 2, "total_blocks": 3},
        output={"path": tmp_path / "demo.mp4", "fps": 12},
    )

    assert result == "completed"
    assert captured[0].preset_id == RUNNER_LINGBOT_WORLD_FAST.runner_name
    assert isinstance(captured[0].output, Mp4OutputSpec)
    assert captured[0].output.path == tmp_path / "demo.mp4"


def test_lingbot_direct_runner_launch_builds_null_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[DemoSpec] = []

    def fake_run_replay_demo(*, spec: DemoSpec, adapter: object) -> str:
        del adapter
        captured.append(spec)
        return "completed"

    monkeypatch.setattr(demo_app_module, "run_replay_demo", fake_run_replay_demo)

    result = demo_app_module.launch_from_runner(
        config=RUNNER_LINGBOT_WORLD_FAST,
        mode="null",
        scenario={"example_idx": 2, "total_blocks": 3},
        output={"fps": 12},
    )

    assert result == "completed"
    assert captured[0].preset_id == RUNNER_LINGBOT_WORLD_FAST.runner_name
    assert isinstance(captured[0].output, NullOutputSpec)
    scenario = captured[0].scenario
    assert scenario is not None
    assert scenario[FIELD_TOTAL_BLOCKS] == 3
    assert scenario[FIELD_FPS] == 12


def test_lingbot_demo_adapter_declares_shared_demo_modes() -> None:
    adapter = LingbotDemoAdapter()

    assert adapter.model_id == LINGBOT_MODEL_ID
    assert adapter.supported_input_modes() == ("replay", "keyboard-driving")
    assert adapter.supported_output_modes() == ("mp4", "null", "webrtc")
    fields = {
        field.name
        for field in adapter.inference_input_schema.global_conditioning_fields
    }
    assert "scenario" not in fields
    assert {
        FIELD_PROMPT,
        FIELD_FIRST_FRAME_PATH,
        FIELD_TOTAL_BLOCKS,
        FIELD_PIXEL_HEIGHT,
        FIELD_PIXEL_WIDTH,
        FIELD_FPS,
    }.issubset(fields)
    # Camera control is per-step model input, not session-global scenario data.
    step_fields = {field.name for field in adapter.inference_input_schema.step_fields}
    assert step_fields == {FIELD_CAMERA_TRAJECTORY, FIELD_CAMERA_INTRINSICS}


def test_lingbot_replay_cli_builds_null_output_spec() -> None:
    args = parse_args(["replay", "--output-mode", "null", "--total-blocks", "1"])

    spec = _replay_spec(args)

    assert spec.model_id == LINGBOT_MODEL_ID
    assert spec.input_mode == "replay"
    assert isinstance(spec.output, NullOutputSpec)
    assert isinstance(spec.scenario, dict)
    assert spec.scenario[FIELD_TOTAL_BLOCKS] == 1


def test_lingbot_replay_cli_requires_output_only_for_mp4(tmp_path: Path) -> None:
    parse_args(["replay", "--output", str(tmp_path / "demo.mp4")])

    with pytest.raises(SystemExit):
        parse_args(["replay"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "replay",
                "--output-mode",
                "null",
                "--output",
                str(tmp_path / "demo.mp4"),
            ]
        )


def test_lingbot_replay_adapter_accepts_null_output(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    poses = tmp_path / "poses.npy"
    intrinsics = tmp_path / "intrinsics.npy"
    image.write_bytes(b"fake")
    _write_camera_assets(poses, intrinsics)
    spec = DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        preset_id=DEFAULT_LINGBOT_PRESET,
        input_mode="replay",
        scenario={
            "prompt": "drive through a city",
            "image_path": image,
            "pose_path": poses,
            "intrinsic_path": intrinsics,
            "total_blocks": 1,
        },
        output=NullOutputSpec(),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            preset_id=DEFAULT_LINGBOT_PRESET,
            runtime_options={"pipeline_config": object()},
        ),
    )

    prepared = LingbotDemoAdapter().prepare_scenario(spec)

    assert prepared.initial_inputs.global_conditioning[FIELD_FIRST_FRAME_PATH] == image
    assert prepared.initial_inputs.global_conditioning[FIELD_TOTAL_BLOCKS] == 1
    assert prepared.mapping is None
    assert prepared.canonicalizer.converters == ()
    assert PROVIDER_INPUTS_METADATA_KEY in prepared.metadata


def test_lingbot_replay_demo_rejects_compat_runner_without_mapping(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.jpg"
    poses = tmp_path / "poses.npy"
    intrinsics = tmp_path / "intrinsics.npy"
    image.write_bytes(b"fake")
    _write_camera_assets(poses, intrinsics)
    pipeline_config = object()
    adapter = LingbotDemoAdapter()
    output = _RecordingOutputTarget()
    calls: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> Sequence[OutputArtifact]:
        calls.append(kwargs)
        return (OutputArtifact(kind="video/mp4", uri="memory://lingbot"),)

    spec = DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        preset_id=DEFAULT_LINGBOT_PRESET,
        input_mode="replay",
        scenario={
            "prompt": "drive through a city",
            "image_path": image,
            "pose_path": poses,
            "intrinsic_path": intrinsics,
            "total_blocks": 1,
        },
        output=Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=16, output_layout="tchw"),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            preset_id=DEFAULT_LINGBOT_PRESET,
            runtime_options={"pipeline_config": pipeline_config},
        ),
    )

    with pytest.raises(ValueError, match="Compatibility replay runners require"):
        run_replay_demo(
            spec=spec,
            adapter=adapter,
            output_target_factory=lambda output_spec: output,
            runner=fake_runner,
        )

    assert calls == []


def test_lingbot_replay_demo_run_mode_uses_model_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lingbot.runtime as runtime_module

    import flashdreams.runtime.demo.replay as replay_module

    image = tmp_path / "image.jpg"
    poses = tmp_path / "poses.npy"
    intrinsics = tmp_path / "intrinsics.npy"
    image.write_bytes(b"fake")
    _write_camera_assets(poses, intrinsics, frames=16)
    pipeline = _FakeLingbotPipeline()
    driver_calls: list[dict[str, Any]] = []
    pipeline_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        runtime_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )

    class RecordingBatchSessionDriver(replay_module.BatchSessionDriver):
        def run_one_session(self, **kwargs: Any) -> Any:
            driver_calls.append(kwargs)
            return super().run_one_session(**kwargs)

    class RecordingStepPipeline(replay_module.StepPipeline):
        def execute_step(self, **kwargs: Any) -> Any:
            pipeline_calls.append(kwargs)
            return super().execute_step(**kwargs)

    monkeypatch.setattr(
        replay_module,
        "BatchSessionDriver",
        RecordingBatchSessionDriver,
    )
    monkeypatch.setattr(replay_module, "StepPipeline", RecordingStepPipeline)

    spec = DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        preset_id=DEFAULT_LINGBOT_PRESET,
        input_mode="replay",
        scenario={
            "prompt": "drive through a city",
            "image_path": image,
            "pose_path": poses,
            "intrinsic_path": intrinsics,
            "total_blocks": 1,
        },
        output=Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=16, output_layout="tchw"),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            preset_id=DEFAULT_LINGBOT_PRESET,
            device="cpu",
            runtime_options={"pipeline_config": object()},
        ),
    )
    expected_mapping = LingbotDemoAdapter().create_input_mapping(
        LingbotReplayInputs(
            prompt="drive through a city",
            first_frame_path=image,
            camera_poses_path=poses,
            camera_intrinsics_path=intrinsics,
            total_blocks=1,
        )
    )

    def pipeline_factory(pipeline_config: object, device: str) -> _FakeLingbotPipeline:
        del pipeline_config, device
        return pipeline

    adapter = LingbotDemoAdapter(pipeline_factory=pipeline_factory)

    result = run_replay_demo(
        spec=spec,
        adapter=adapter,
        output_target_factory=lambda output_spec: _RecordingOutputTarget(),
    )

    assert result.status == "completed"
    assert len(driver_calls) == 1
    assert len(pipeline_calls) == 1
    assert isinstance(pipeline_calls[0]["provider"], LingbotInputProvider)
    assert pipeline.generate_calls == [
        {
            "autoregressive_index": 0,
            "intrinsics_shape": (1, 4),
            "poses_shape": (1, 4, 4),
            "world_scale": pytest.approx(expected_mapping.camera_trace.world_scale),
        }
    ]


def test_lingbot_replay_demo_runs_with_null_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lingbot.runtime as runtime_module

    image = tmp_path / "image.jpg"
    poses = tmp_path / "poses.npy"
    intrinsics = tmp_path / "intrinsics.npy"
    image.write_bytes(b"fake")
    _write_camera_assets(poses, intrinsics, frames=16)
    pipeline = _FakeLingbotPipeline()
    monkeypatch.setattr(
        runtime_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )

    spec = DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        preset_id=DEFAULT_LINGBOT_PRESET,
        input_mode="replay",
        scenario={
            "prompt": "drive through a city",
            "image_path": image,
            "pose_path": poses,
            "intrinsic_path": intrinsics,
            "total_blocks": 1,
            "pixel_height": 2,
            "pixel_width": 2,
        },
        output=NullOutputSpec(),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            preset_id=DEFAULT_LINGBOT_PRESET,
            device="cpu",
            runtime_options={"pipeline_config": object()},
        ),
    )

    def pipeline_factory(pipeline_config: object, device: str) -> _FakeLingbotPipeline:
        del pipeline_config, device
        return pipeline

    result = run_replay_demo(
        spec=spec,
        adapter=LingbotDemoAdapter(pipeline_factory=pipeline_factory),
    )

    assert result.status == "completed"
    assert result.artifacts == ()
    assert len(pipeline.generate_calls) == 1


def test_lingbot_replay_invalid_scenario_fails_before_runtime_creation(
    tmp_path: Path,
) -> None:
    adapter = LingbotDemoAdapter(
        replay_runtime_factory=lambda **kwargs: pytest.fail(
            f"runtime should not be created: {kwargs}"
        )
    )
    output_factory_calls = 0

    def output_factory(output_spec: object) -> OutputTarget:
        nonlocal output_factory_calls
        del output_spec
        output_factory_calls += 1
        return _RecordingOutputTarget()

    spec = DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        input_mode="replay",
        scenario={
            "prompt": "drive",
            "image_path": tmp_path / "missing.jpg",
            "pose_path": tmp_path / "missing-poses.npy",
            "intrinsic_path": tmp_path / "missing-intrinsics.npy",
        },
        output=Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=16),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            runtime_options={"pipeline_config": object()},
        ),
    )

    with pytest.raises(FileNotFoundError, match=f"missing {FIELD_FIRST_FRAME_PATH}"):
        run_replay_demo(
            spec=spec,
            adapter=adapter,
            output_target_factory=output_factory,
        )

    assert output_factory_calls == 0


def test_lingbot_replay_cli_defaults_to_example_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lingbot.runtime as runtime_module

    example_dir = tmp_path / "example"
    example_dir.mkdir()
    (example_dir / "image.jpg").write_bytes(b"fake")
    _write_camera_assets(example_dir / "poses.npy", example_dir / "intrinsics.npy")
    (example_dir / "prompt.txt").write_text("drive through a forest\n")
    downloaded: list[int] = []

    def fake_download(*, is_rank_zero: bool, example_idx: int) -> Path:
        assert is_rank_zero is True
        downloaded.append(example_idx)
        return example_dir

    monkeypatch.setattr(
        runtime_module,
        "ensure_example_data_downloaded",
        fake_download,
    )
    args = parse_args(["replay", "--output", str(tmp_path / "demo.mp4")])
    spec = _replay_spec(args)

    prepared = LingbotDemoAdapter().prepare_scenario(spec)

    inputs = prepared.initial_inputs.global_conditioning
    assert downloaded == [0]
    assert inputs[FIELD_FIRST_FRAME_PATH] == example_dir / "image.jpg"
    assert inputs[FIELD_PROMPT] == "drive through a forest"


def test_lingbot_replay_cli_can_disable_example_data(tmp_path: Path) -> None:
    args = parse_args(
        ["replay", "--no-example-data", "--output", str(tmp_path / "demo.mp4")]
    )
    spec = _replay_spec(args)

    with pytest.raises(ValueError, match=f"require {FIELD_FIRST_FRAME_PATH}"):
        LingbotDemoAdapter().prepare_scenario(spec)


def test_lingbot_replay_runtime_generates_video_step_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lingbot.runtime as runtime_module

    image = tmp_path / "image.jpg"
    poses = tmp_path / "poses.npy"
    intrinsics = tmp_path / "intrinsics.npy"
    image.write_bytes(b"fake")
    _write_camera_assets(poses, intrinsics, frames=16)
    pipeline = _FakeLingbotPipeline()
    monkeypatch.setattr(
        runtime_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )

    runtime = LingbotReplayRuntime(
        config=InferenceConfig(model_id=LINGBOT_MODEL_ID, device="cpu"),
        options=LingbotReplayRuntimeOptions(
            pipeline_config=object(),
            pipeline_factory=lambda pipeline_config, device: pipeline,
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
    adapter = LingbotDemoAdapter()
    mapping = adapter.create_input_mapping(replay_inputs)
    initial_inputs = mapping.map_global_conditioning_inputs(
        canonical_inputs=CanonicalInputs(),
        inference_input=inference_input_from_replay_inputs(replay_inputs),
    )
    session = runtime.start_session(initial_inputs)

    request = session.next_step_request()
    assert request is not None
    assert request.step_index == 0
    # The session asks for exactly this chunk's slice of the input timeline.
    assert request.user_input_window is not None
    assert request.user_input_window.start_s == 0.0
    assert request.user_input_window.end_s == 1 / 16
    assert request.metadata["num_frames"] == 1

    step_inputs = mapping.map_step_inputs(
        canonical_inputs=CanonicalInputs(),
        inference_input=InferenceInput(),
        request=request,
    )
    assert step_inputs.step[FIELD_CAMERA_TRAJECTORY].shape == (1, 4, 4)
    assert step_inputs.step[FIELD_CAMERA_INTRINSICS].shape == (1, 4)
    result = session.step(step_inputs)

    assert result.step_index == 0
    assert result.frame_count == 1
    assert isinstance(result, StepResult)
    assert result.layout == "tchw"
    assert result.video_chunk.shape == (1, 3, 2, 2)
    assert result.output_window is not None
    assert result.output_window.start_s == 0.0
    assert result.output_window.end_s == 1 / 16
    assert result.metrics["denoise_s"] == 0.25
    assert session.next_step_request() is None
    assert pipeline.initialize_cache_calls == [
        {"text": ["drive"], "image_shape": (1, 3, 2, 2)}
    ]
    assert pipeline.generate_calls == [
        {
            "autoregressive_index": 0,
            "intrinsics_shape": (1, 4),
            "poses_shape": (1, 4, 4),
            "world_scale": pytest.approx(mapping.camera_trace.world_scale),
        }
    ]
    runtime.close()


def test_lingbot_webrtc_cli_builds_keyboard_driving_spec() -> None:
    args = parse_args(
        [
            "webrtc",
            "--host",
            "127.0.0.1",
            "--port",
            "9090",
            "--device",
            "cuda:2",
            "--seed",
            "123",
            "--no-compile",
            "--fps",
            "12",
            "--video-height",
            "32",
            "--video-width",
            "64",
            "--warmup-chunks",
            "0",
            "--warmup-timeout-s",
            "1.5",
            "--client-liveness-timeout-s",
            "2.5",
            "--prefer-sw-encoder",
            "--example-idx",
            "2",
        ]
    )

    spec = _webrtc_spec(args, device="cuda:3", context_parallel_size=4)

    assert spec.model_id == LINGBOT_MODEL_ID
    assert spec.preset_id == DEFAULT_LINGBOT_PRESET
    assert spec.input_mode == "keyboard-driving"
    assert isinstance(spec.scenario, LingbotWebRTCScenario)
    assert spec.scenario.example_idx == 2
    assert spec.scenario.prefer_sw_encoder is True
    assert isinstance(spec.output, WebRTCOutputSpec)
    assert spec.output.host == "127.0.0.1"
    assert spec.output.port == 9090
    assert spec.output.fps == 12
    assert spec.output.video_width == 64
    assert spec.output.video_height == 32
    assert spec.output.warmup_chunks == 0
    assert spec.output.warmup_timeout_s == 1.5
    assert spec.output.client_liveness_timeout_s == 2.5
    assert spec.config is not None
    assert spec.config.device == "cuda:3"
    assert spec.config.compile is False
    assert spec.config.runtime_options["seed"] == 123
    assert spec.config.runtime_options["context_parallel_size"] == 4
    assert spec.config.runtime_options["example_idx"] == 2


def test_lingbot_adapter_prepares_public_webrtc_scenario_as_live_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_lingbot_webrtc_example(monkeypatch, tmp_path)
    adapter = LingbotDemoAdapter()
    spec = DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        preset_id=DEFAULT_LINGBOT_PRESET,
        input_mode="keyboard-driving",
        scenario=LingbotWebRTCScenario(example_idx=0),
        output=WebRTCOutputSpec(fps=16, video_width=64, video_height=32),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            preset_id=DEFAULT_LINGBOT_PRESET,
            runtime_options={
                "text_events": [
                    {
                        "event_id": "storm",
                        "label": "Storm",
                        "prompt": "A storm moves through the scene.",
                    }
                ]
            },
        ),
    )

    prepared = adapter.prepare_scenario(spec)
    provider = adapter.create_model_input_provider(spec, prepared)

    assert isinstance(provider, LingbotInputProvider)
    assert provider.capabilities.supports_realtime_clock is True
    assert {"key_down", "key_up", "text_event"}.issubset(
        prepared.source_schema.declared_event_types()
    )


def test_lingbot_webrtc_demo_uses_shared_manager_with_model_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_lingbot_webrtc_example(monkeypatch, tmp_path, example_idx=2)
    pipeline_config = object()
    spec = DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        preset_id=DEFAULT_LINGBOT_PRESET,
        input_mode="keyboard-driving",
        scenario=LingbotWebRTCScenario(example_idx=2, prefer_sw_encoder=True),
        output=WebRTCOutputSpec(
            host="0.0.0.0",
            port=8080,
            fps=24,
            video_width=64,
            video_height=32,
            warmup_chunks=0,
            warmup_timeout_s=1.0,
        ),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            preset_id=DEFAULT_LINGBOT_PRESET,
            device="cuda:7",
            runtime_options={"pipeline_config": pipeline_config, "seed": 123},
        ),
    )

    calls: list[dict[str, Any]] = []
    serve_lingbot_webrtc_demo(
        spec=spec,
        world_rank=1,
        runtime_factory=_FakeWebRTCRuntime,
        server_runner=lambda **kwargs: calls.append(kwargs),
    )

    manager = calls[0]["session_manager"]
    runtime = manager._runtime
    assert isinstance(runtime, _FakeWebRTCRuntime)
    assert type(manager) is BaseWebRTCSessionManager
    assert isinstance(manager._shared_adapter, LingbotDemoAdapter)
    assert manager._shared_spec is not None
    assert manager._shared_spec.input_mode == "keyboard-driving"
    assert isinstance(manager._shared_spec.output, WebRTCOutputSpec)
    assert manager._shared_scenario is not None
    assert manager._shared_scenario.mapping is None
    assert manager._shared_scenario.canonicalizer.converters == ()
    assert manager._needs_legacy_segment_metadata() is False
    provider = manager._shared_adapter.create_model_input_provider(
        manager._shared_spec,
        manager._shared_scenario,
    )
    assert isinstance(provider, LingbotInputProvider)
    assert provider.capabilities.supports_realtime_clock is True
    assert manager.runtime_config is runtime.config
    assert runtime.config.pipeline_config is pipeline_config
    assert runtime.config.config_name == DEFAULT_LINGBOT_PRESET
    assert runtime.config.seed == 123
    assert runtime.config.device == "cuda:7"
    assert runtime.config.video_width == 64
    assert runtime.config.video_height == 32
    assert runtime.config.fps == 24
    assert runtime.config.warmup_chunks == 0
    assert runtime.config.warmup_timeout_s == 1.0
    assert runtime.config.encoder_backend == "default"
    assert runtime.config.example_data_dir.name == "02"
    assert isinstance(spec.output, WebRTCOutputSpec)
    assert manager.client_liveness_timeout_s == spec.output.client_liveness_timeout_s
    assert manager.identity == DEFAULT_LINGBOT_PRESET
    assert calls[0]["host"] == "0.0.0.0"
    assert calls[0]["port"] == 8080


def test_lingbot_webrtc_shared_provider_reflects_pending_session_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_lingbot_webrtc_example(monkeypatch, tmp_path)
    pipeline_config = object()
    spec = DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        preset_id=DEFAULT_LINGBOT_PRESET,
        input_mode="keyboard-driving",
        scenario=LingbotWebRTCScenario(example_idx=0),
        output=WebRTCOutputSpec(
            fps=16,
            video_width=64,
            video_height=32,
            warmup_chunks=8,
            warmup_timeout_s=2.5,
        ),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            preset_id=DEFAULT_LINGBOT_PRESET,
            runtime_options={"pipeline_config": pipeline_config},
        ),
    )
    calls: list[dict[str, Any]] = []
    pending = LingbotSessionInput(
        prompt="drive through a custom city",
        text_events=(
            TextEventSpec(
                event_id="rain",
                label="Rain",
                prompt="heavy rain falls on the road",
            ),
        ),
    )

    serve_lingbot_webrtc_demo(
        spec=spec,
        runtime_factory=_FakeWebRTCRuntime,
        server_runner=lambda **kwargs: calls.append(kwargs),
    )

    manager = calls[0]["session_manager"]
    assert callable(manager._shared_spec_factory)
    session_spec = manager._shared_spec_factory(pending)
    assert isinstance(session_spec.output, WebRTCOutputSpec)
    assert session_spec.output.video_width == 64
    assert session_spec.output.video_height == 32
    assert session_spec.output.warmup_chunks == 8
    prepared = manager._shared_adapter.prepare_scenario(session_spec)
    provider = manager._shared_adapter.create_model_input_provider(
        session_spec,
        prepared,
    )

    initial = provider.prepare_initial_input()
    assert initial.global_conditioning[FIELD_PROMPT] == "drive through a custom city"
    assert initial.global_conditioning[FIELD_PIXEL_HEIGHT] == 32
    assert initial.global_conditioning[FIELD_PIXEL_WIDTH] == 64
    assert {"key_down", "key_up", "text_event"}.issubset(
        prepared.source_schema.declared_event_types()
    )
    prepared_step = provider.prepare_step(
        request=StepRequirements(
            step_index=0,
            input_frame_count=4,
            metadata={"frame_start": 0, "num_frames": 4},
        ),
        user_window=UserInputWindow(
            start_s=0.0,
            end_s=0.25,
            inputs=UserInputs(
                events=(
                    UserInputEvent(
                        timestamp_s=0.0,
                        event_type="text_event",
                        payload={"event_id": "rain"},
                    ),
                )
            ),
        ),
    )
    assert prepared_step.inference_input is not None
    assert (
        prepared_step.inference_input.global_conditioning[FIELD_PROMPT]
        == "heavy rain falls on the road"
    )


def test_lingbot_webrtc_demo_uses_shared_viewer_shell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import flashdreams.serving.webrtc.demo as shared_webrtc_module

    _patch_lingbot_webrtc_example(monkeypatch, tmp_path)
    app_calls: list[dict[str, Any]] = []

    def fake_create_packaged_app(**kwargs: Any) -> web.Application:
        app_calls.append(kwargs)
        app = web.Application()
        app[SESSION_MANAGER_KEY] = kwargs["session_manager"]
        kwargs["configure_app"](app)
        return app

    monkeypatch.setattr(
        shared_webrtc_module, "create_packaged_webrtc_app", fake_create_packaged_app
    )
    spec = DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        preset_id=DEFAULT_LINGBOT_PRESET,
        input_mode="keyboard-driving",
        scenario=LingbotWebRTCScenario(),
        output=WebRTCOutputSpec(
            host="0.0.0.0",
            port=8080,
            warmup_timeout_s=1.0,
            preload_name="Test Lingbot",
        ),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            preset_id=DEFAULT_LINGBOT_PRESET,
            runtime_options={"pipeline_config": object()},
        ),
    )

    app = serve_lingbot_webrtc_demo(
        spec=spec,
        runtime_factory=_FakeWebRTCRuntime,
        create_app_fn=lambda **kwargs: fake_create_packaged_app(**kwargs),
        server_runner=lambda **kwargs: None,
    )

    assert isinstance(app, web.Application)
    assert app_calls[0]["session_manager"] is app[SESSION_MANAGER_KEY]
    assert app_calls[0]["request_session_url"] == (
        "http://127.0.0.1:8080/request_session"
    )
    assert app_calls[0]["preload_name"] == "Test Lingbot"
    assert str(app_calls[0]["web_resource"]).endswith("serving/webrtc/web")
    assert str(app_calls[0]["model_web_resource"]).endswith("lingbot/webrtc/web")
    assert callable(app_calls[0]["configure_app"])
    route_paths = {resource.canonical for resource in app.router.resources()}
    assert "/api/session/initial_scene" in route_paths
    assert "/api/session/first_frame" in route_paths
    assert "/api/session/input" in route_paths


def test_lingbot_webrtc_demo_serves_through_shared_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import flashdreams.serving.webrtc.demo as shared_webrtc_module

    _patch_lingbot_webrtc_example(monkeypatch, tmp_path)
    server_calls: list[dict[str, Any]] = []

    def fake_create_packaged_app(**kwargs: Any) -> web.Application:
        app = web.Application()
        app[SESSION_MANAGER_KEY] = kwargs["session_manager"]
        kwargs["configure_app"](app)
        return app

    def fake_server_runner(**kwargs: Any) -> None:
        server_calls.append(kwargs)

    monkeypatch.setattr(
        shared_webrtc_module, "create_packaged_webrtc_app", fake_create_packaged_app
    )
    spec = DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        preset_id=DEFAULT_LINGBOT_PRESET,
        input_mode="keyboard-driving",
        scenario={"example_idx": 0},
        output=WebRTCOutputSpec(
            host="0.0.0.0",
            port=8080,
            warmup_timeout_s=1.0,
        ),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            preset_id=DEFAULT_LINGBOT_PRESET,
            runtime_options={"pipeline_config": object()},
        ),
    )

    app = serve_lingbot_webrtc_demo(
        spec=spec,
        world_rank=0,
        runtime_factory=_FakeWebRTCRuntime,
        server_runner=fake_server_runner,
    )

    assert len(server_calls) == 1
    assert server_calls[0]["world_rank"] == 0
    assert server_calls[0]["app"] is app
    assert server_calls[0]["host"] == "0.0.0.0"
    assert server_calls[0]["port"] == 8080
    assert type(server_calls[0]["session_manager"]) is BaseWebRTCSessionManager


class _RecordingOutputTarget:
    def open(self) -> None:
        return None

    def write(self, result: StepResult) -> None:
        del result

    def close(self) -> Sequence[OutputArtifact]:
        return ()


class _FakeLingbotPipeline:
    def __init__(self) -> None:
        self.initialize_cache_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []

    def initialize_cache(self, *, text: list[str], image: torch.Tensor) -> object:
        self.initialize_cache_calls.append(
            {
                "text": text,
                "image_shape": tuple(image.shape),
            }
        )
        return object()

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        del autoregressive_index
        return 1

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: object,
        input: Any,
    ) -> torch.Tensor:
        del cache
        self.generate_calls.append(
            {
                "autoregressive_index": autoregressive_index,
                "intrinsics_shape": tuple(input.intrinsics.shape),
                "poses_shape": tuple(input.poses.shape),
                "world_scale": input.world_scale,
            }
        )
        return torch.full((1, 3, 2, 2), float(autoregressive_index))

    def finalize(self, *, autoregressive_index: int, cache: object) -> dict[str, float]:
        del autoregressive_index, cache
        return {"denoise_s": 0.25}


class _FakeWebRTCRuntime:
    def __init__(self, config: LingbotRuntimeConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        return None

    async def reset_for_new_session(self, *args: Any, **kwargs: Any) -> None:
        return None

    def peek_input_fps(self) -> float:
        return 16.0

    def peek_steady_output_num_frames(self) -> int:
        return 1

    def next_step_request(self) -> StepRequest:
        return StepRequest(step_index=0, metadata={"input_frame_count": 1})

    async def step(
        self,
        *,
        request: StepRequest,
        segments: list[Any],
        frame_times: list[float],
    ) -> Any:
        del request, segments, frame_times
        return None

    async def close(self) -> None:
        return None

    def send_exit_signal(self) -> None:
        return None

    def wait_for_termination(self) -> None:
        return None
