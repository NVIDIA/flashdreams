# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from aiohttp import web
from lingbot.demo import (
    DEFAULT_LINGBOT_PRESET,
    LINGBOT_MODEL_ID,
    LingbotDemoAdapter,
    LingbotReplayScenario,
    LingbotWebRTCScenario,
)
from lingbot.demo.cli import _replay_spec, _webrtc_spec, parse_args
from lingbot.demo.replay import (
    LingbotReplayRuntime,
    LingbotReplayRuntimeOptions,
)
from lingbot.demo.webrtc import LingbotDemoWebRTCSessionManager
from lingbot.webrtc.session import LingbotRuntimeConfig

from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import (
    InferenceConfig,
    InferenceInput,
    OutputArtifact,
    OutputTarget,
    StepResult,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    WebRTCOutputSpec,
    serve_flashdreams_demo,
)
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.runtime.demo.webrtc import WebRTCDemo, build_webrtc_demo
from flashdreams.serving.webrtc.server import SESSION_MANAGER_KEY

pytestmark = pytest.mark.ci_cpu


def test_lingbot_demo_defaults_to_interactive_preset() -> None:
    args = parse_args(["replay", "--output", "demo.mp4"])

    assert args.preset_id == "lingbot-world-fast-taehv-window15-sink3"


def test_lingbot_demo_adapter_declares_mp4_and_webrtc_modes() -> None:
    adapter = LingbotDemoAdapter()

    assert adapter.model_id == LINGBOT_MODEL_ID
    assert adapter.supported_input_modes() == ("replay", "keyboard-driving")
    assert adapter.supported_output_modes() == ("mp4", "webrtc")


def test_lingbot_replay_demo_uses_shared_runner(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    poses = tmp_path / "poses.npy"
    intrinsics = tmp_path / "intrinsics.npy"
    image.write_bytes(b"fake")
    poses.write_bytes(b"fake")
    intrinsics.write_bytes(b"fake")
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

    artifacts = run_replay_demo(
        spec=spec,
        adapter=adapter,
        output_target_factory=lambda output_spec: output,
        runner=fake_runner,
    )

    assert artifacts == (OutputArtifact(kind="video/mp4", uri="memory://lingbot"),)
    assert len(calls) == 1
    assert calls[0]["adapter"] is adapter
    assert calls[0]["config"] == spec.config
    scenario = calls[0]["initial_inputs"].global_conditioning["scenario"]
    assert isinstance(scenario, LingbotReplayScenario)
    assert scenario.prompt == "drive through a city"
    assert scenario.image_path == image
    assert scenario.pose_path == poses
    assert scenario.intrinsic_path == intrinsics


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

    with pytest.raises(FileNotFoundError, match="missing image_path"):
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
    import lingbot.demo.spec as spec_module

    example_dir = tmp_path / "example"
    example_dir.mkdir()
    (example_dir / "image.jpg").write_bytes(b"fake")
    (example_dir / "poses.npy").write_bytes(b"fake")
    (example_dir / "intrinsics.npy").write_bytes(b"fake")
    (example_dir / "prompt.txt").write_text("drive through a forest\n")
    downloaded: list[int] = []

    def fake_download(*, is_rank_zero: bool, example_idx: int) -> Path:
        assert is_rank_zero is True
        downloaded.append(example_idx)
        return example_dir

    monkeypatch.setattr(
        spec_module,
        "ensure_example_data_downloaded",
        fake_download,
    )
    args = parse_args(["replay", "--output", str(tmp_path / "demo.mp4")])
    spec = _replay_spec(args)

    prepared = LingbotDemoAdapter().prepare_scenario(spec)

    scenario = prepared.initial_inputs.global_conditioning["scenario"]
    assert isinstance(scenario, LingbotReplayScenario)
    assert downloaded == [0]
    assert scenario.image_path == example_dir / "image.jpg"
    assert scenario.pose_path == example_dir / "poses.npy"
    assert scenario.intrinsic_path == example_dir / "intrinsics.npy"
    assert scenario.prompt == "drive through a forest"


def test_lingbot_replay_cli_can_disable_example_data(tmp_path: Path) -> None:
    args = parse_args(
        ["replay", "--no-example-data", "--output", str(tmp_path / "demo.mp4")]
    )
    spec = _replay_spec(args)

    with pytest.raises(ValueError, match="requires image_path"):
        LingbotDemoAdapter().prepare_scenario(spec)


def test_lingbot_replay_runtime_generates_video_step_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lingbot.demo.replay as replay_module

    image = tmp_path / "image.jpg"
    poses = tmp_path / "poses.npy"
    intrinsics = tmp_path / "intrinsics.npy"
    image.write_bytes(b"fake")
    np.save(poses, np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)))
    np.save(intrinsics, np.ones((2, 4), dtype=np.float32))
    pipeline = _FakeLingbotPipeline()
    monkeypatch.setattr(
        replay_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )
    monkeypatch.setattr(
        replay_module,
        "get_Ks_transformed",
        lambda intrinsics_t, **kwargs: intrinsics_t,
    )
    monkeypatch.setattr(
        replay_module,
        "preprocess_example_poses",
        lambda c2ws: (c2ws, 2.5),
    )

    runtime = LingbotReplayRuntime(
        config=InferenceConfig(model_id=LINGBOT_MODEL_ID, device="cpu"),
        options=LingbotReplayRuntimeOptions(
            pipeline_config=object(),
            pipeline_factory=lambda pipeline_config, device: pipeline,
        ),
    )
    scenario = LingbotReplayScenario(
        prompt="drive",
        image_path=image,
        pose_path=poses,
        intrinsic_path=intrinsics,
        total_blocks=1,
        pixel_height=2,
        pixel_width=2,
        fps=16,
    )
    session = runtime.start_session(
        InferenceInput(global_conditioning={"scenario": scenario})
    )

    request = session.next_step_request()
    assert request is not None
    assert request.step_index == 0
    result = session.step(InferenceInput())

    assert result.step_index == 0
    assert result.frame_count == 1
    assert isinstance(result.output, VideoStepResult)
    assert result.output.layout == "tchw"
    assert result.output.video_chunk.shape == (1, 3, 2, 2)
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
            "world_scale": 2.5,
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


def test_lingbot_webrtc_demo_uses_existing_manager_with_model_config() -> None:
    pipeline_config = object()
    adapter = LingbotDemoAdapter(webrtc_runtime_factory=_FakeWebRTCRuntime)
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

    demo = build_webrtc_demo(spec=spec, adapter=adapter)

    assert isinstance(demo.runtime, _FakeWebRTCRuntime)
    assert isinstance(demo.session_manager, LingbotDemoWebRTCSessionManager)
    assert demo.session_manager._runtime is demo.runtime
    assert demo.session_manager.runtime_config is demo.runtime.config
    assert demo.runtime_config is demo.runtime.config
    assert demo.runtime_config.pipeline_config is pipeline_config
    assert demo.runtime_config.config_name == DEFAULT_LINGBOT_PRESET
    assert demo.runtime_config.seed == 123
    assert demo.runtime_config.device == "cuda:7"
    assert demo.runtime_config.video_width == 64
    assert demo.runtime_config.video_height == 32
    assert demo.runtime_config.fps == 24
    assert demo.runtime_config.encoder_backend == "default"
    assert demo.runtime_config.example_data_dir.name == "02"
    assert demo.session_manager._model_name() == DEFAULT_LINGBOT_PRESET
    assert demo.host == "0.0.0.0"
    assert demo.port == 8080


def test_lingbot_webrtc_demo_installs_model_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lingbot.demo.webrtc as demo_webrtc_module

    app_calls: list[dict[str, Any]] = []

    async def _ok(request: web.Request) -> web.Response:
        del request
        return web.Response(text="ok")

    def fake_create_app(**kwargs: Any) -> web.Application:
        app_calls.append(kwargs)
        app = web.Application()
        app[SESSION_MANAGER_KEY] = kwargs["session_manager"]
        app.router.add_get("/api/session/initial_scene", _ok)
        app.router.add_get("/api/session/first_frame", _ok)
        app.router.add_post("/api/session/input", _ok)
        return app

    monkeypatch.setattr(demo_webrtc_module, "create_app", fake_create_app)
    adapter = LingbotDemoAdapter(webrtc_runtime_factory=_FakeWebRTCRuntime)
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

    demo = build_webrtc_demo(spec=spec, adapter=adapter, create_app=True)

    assert demo.app is not None
    assert app_calls[0]["session_manager"] is demo.session_manager
    assert app_calls[0]["request_session_url"] == (
        "http://127.0.0.1:8080/request_session"
    )
    route_paths = {resource.canonical for resource in demo.app.router.resources()}
    assert "/api/session/initial_scene" in route_paths
    assert "/api/session/first_frame" in route_paths
    assert "/api/session/input" in route_paths


def test_lingbot_webrtc_demo_serves_through_shared_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lingbot.demo.webrtc as demo_webrtc_module

    server_calls: list[dict[str, Any]] = []

    def fake_create_app(**kwargs: Any) -> web.Application:
        app = web.Application()
        app[SESSION_MANAGER_KEY] = kwargs["session_manager"]
        return app

    def fake_server_runner(**kwargs: Any) -> None:
        server_calls.append(kwargs)

    monkeypatch.setattr(demo_webrtc_module, "create_app", fake_create_app)
    adapter = LingbotDemoAdapter(webrtc_runtime_factory=_FakeWebRTCRuntime)
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

    demo = cast(
        WebRTCDemo,
        serve_flashdreams_demo(
            spec=spec,
            adapter=adapter,
            world_rank=0,
            server_runner=fake_server_runner,
        ),
    )

    assert len(server_calls) == 1
    assert server_calls[0]["world_rank"] == 0
    assert server_calls[0]["session_manager"] is demo.session_manager
    assert server_calls[0]["app"] is demo.app
    assert server_calls[0]["host"] == "0.0.0.0"
    assert server_calls[0]["port"] == 8080
    assert isinstance(demo.session_manager, LingbotDemoWebRTCSessionManager)


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

    def peek_steady_chunk_num_frames(self) -> int:
        return 1

    def peek_next_chunk_num_frames(self) -> int:
        return 1

    async def generate_chunk(
        self,
        *,
        segments: list[Any],
        frame_times: list[float],
    ) -> Any:
        del segments, frame_times
        return None

    async def close(self) -> None:
        return None

    def send_exit_signal(self) -> None:
        return None

    def wait_for_termination(self) -> None:
        return None
