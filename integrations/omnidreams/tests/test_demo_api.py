# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import omnidreams.demo.spec as spec_module
import omnidreams.demo.webrtc as demo_webrtc_module
import pytest
import torch
from aiohttp import web
from omnidreams.config import OMNIDREAMS_RUNNERS
from omnidreams.demo import (
    DEFAULT_OMNIDREAMS_PRESET,
    OMNIDREAMS_MODEL_ID,
    OmnidreamsDemoAdapter,
    OmnidreamsReplayScenario,
    OmnidreamsWebRTCScenario,
)
from omnidreams.demo.cli import _replay_spec, _webrtc_spec, parse_args
from omnidreams.demo.replay import (
    OmnidreamsReplayRuntime,
    OmnidreamsReplayRuntimeOptions,
)
from omnidreams.demo.webrtc import OmnidreamsDemoWebRTCSessionManager

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


def test_omnidreams_demo_defaults_to_stable_non_perf_preset() -> None:
    args = parse_args(["replay", "--output", "demo.mp4"])

    assert args.preset_id == "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
    assert not args.preset_id.endswith("-perf")


def test_omnidreams_demo_adapter_declares_mp4_and_webrtc_modes() -> None:
    adapter = OmnidreamsDemoAdapter()

    assert adapter.model_id == OMNIDREAMS_MODEL_ID
    assert adapter.supported_input_modes() == ("replay", "keyboard-driving")
    assert adapter.supported_output_modes() == ("mp4", "webrtc")


def test_omnidreams_replay_demo_uses_shared_runner(tmp_path: Path) -> None:
    hdmap = tmp_path / "hdmap.mp4"
    first_frame = tmp_path / "first.png"
    hdmap.write_bytes(b"fake")
    first_frame.write_bytes(b"fake")
    pipeline_config = object()
    adapter = OmnidreamsDemoAdapter()
    output = _RecordingOutputTarget()
    calls: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> Sequence[OutputArtifact]:
        calls.append(kwargs)
        return (OutputArtifact(kind="video/mp4", uri="memory://omnidreams"),)

    spec = DemoSpec(
        model_id=OMNIDREAMS_MODEL_ID,
        preset_id=DEFAULT_OMNIDREAMS_PRESET,
        input_mode="replay",
        scenario={
            "prompt": "drive through a city",
            "hdmap_video_paths": (hdmap,),
            "first_frame_paths": (first_frame,),
            "camera_names": ("camera_front_wide_120fov",),
            "total_blocks": 1,
        },
        output=Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=30),
        config=InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=DEFAULT_OMNIDREAMS_PRESET,
            runtime_options={"pipeline_config": pipeline_config},
        ),
    )

    artifacts = run_replay_demo(
        spec=spec,
        adapter=adapter,
        output_target_factory=lambda output_spec: output,
        runner=fake_runner,
    )

    assert artifacts == (OutputArtifact(kind="video/mp4", uri="memory://omnidreams"),)
    assert len(calls) == 1
    assert calls[0]["adapter"] is adapter
    assert calls[0]["config"] == spec.config
    scenario = calls[0]["initial_inputs"].global_conditioning["scenario"]
    assert isinstance(scenario, OmnidreamsReplayScenario)
    assert scenario.prompts == ("drive through a city",)
    assert scenario.hdmap_video_paths == (hdmap,)
    assert scenario.first_frame_paths == (first_frame,)
    assert scenario.camera_names == ("camera_front_wide_120fov",)


def test_omnidreams_replay_invalid_scenario_fails_before_runtime_creation(
    tmp_path: Path,
) -> None:
    adapter = OmnidreamsDemoAdapter(
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
        model_id=OMNIDREAMS_MODEL_ID,
        input_mode="replay",
        scenario={
            "prompt": "drive",
            "hdmap_video_paths": (tmp_path / "missing-hdmap.mp4",),
            "first_frame_paths": (tmp_path / "missing-first.png",),
        },
        output=Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=30),
        config=InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            runtime_options={"pipeline_config": object()},
        ),
    )

    with pytest.raises(FileNotFoundError, match="missing hdmap_video_paths"):
        run_replay_demo(
            spec=spec,
            adapter=adapter,
            output_target_factory=output_factory,
        )

    assert output_factory_calls == 0


def test_omnidreams_replay_cli_defaults_to_hf_example_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hdmap = tmp_path / "hf-hdmap.mp4"
    first_frame = tmp_path / "hf-first.png"
    hdmap.write_bytes(b"fake")
    first_frame.write_bytes(b"fake")
    synced_uuids: list[str] = []

    def fake_sync(uuid: str) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        synced_uuids.append(uuid)
        return (hdmap,), (first_frame,)

    monkeypatch.setattr(
        spec_module,
        "_ensure_hf_single_view_example_data_synced",
        fake_sync,
    )
    args = parse_args(["replay", "--output", str(tmp_path / "demo.mp4")])
    spec = _replay_spec(args)

    prepared = OmnidreamsDemoAdapter().prepare_scenario(spec)

    scenario = prepared.initial_inputs.global_conditioning["scenario"]
    assert isinstance(scenario, OmnidreamsReplayScenario)
    assert synced_uuids == ["239560dc-33d1-11ef-9720-00044bcbccac"]
    assert scenario.hdmap_video_paths == (hdmap,)
    assert scenario.first_frame_paths == (first_frame,)
    assert scenario.camera_names == ("camera_front_wide_120fov",)
    assert scenario.prompts == (
        str(getattr(OMNIDREAMS_RUNNERS[DEFAULT_OMNIDREAMS_PRESET], "prompt")),
    )


def test_omnidreams_replay_cli_can_disable_example_data(tmp_path: Path) -> None:
    args = parse_args(
        ["replay", "--no-example-data", "--output", str(tmp_path / "demo.mp4")]
    )
    spec = _replay_spec(args)

    with pytest.raises(ValueError, match="requires hdmap_video_paths"):
        OmnidreamsDemoAdapter().prepare_scenario(spec)


def test_omnidreams_replay_runtime_generates_video_step_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import omnidreams.demo.replay as replay_module

    hdmap = tmp_path / "hdmap.mp4"
    first_frame = tmp_path / "first.png"
    hdmap.write_bytes(b"fake")
    first_frame.write_bytes(b"fake")
    pipeline = _FakeOmnidreamsPipeline()
    monkeypatch.setattr(
        replay_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )
    monkeypatch.setattr(
        replay_module,
        "_load_video",
        lambda *args, **kwargs: torch.zeros(2, 3, 2, 2),
    )

    runtime = OmnidreamsReplayRuntime(
        config=InferenceConfig(model_id=OMNIDREAMS_MODEL_ID, device="cpu"),
        options=OmnidreamsReplayRuntimeOptions(
            pipeline_config=object(),
            pipeline_factory=lambda pipeline_config, device: pipeline,
        ),
    )
    scenario = OmnidreamsReplayScenario(
        prompts=("drive",),
        hdmap_video_paths=(hdmap,),
        first_frame_paths=(first_frame,),
        camera_names=("camera_front_wide_120fov",),
        total_blocks=1,
        pixel_height=2,
        pixel_width=2,
        fps=30,
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
    assert result.output.layout == "bvtchw"
    assert result.output.video_chunk.shape == (1, 1, 1, 3, 2, 2)
    assert result.metrics["denoise_s"] == 0.25
    assert session.next_step_request() is None
    assert pipeline.initialize_cache_calls == [
        {
            "text": [["drive"]],
            "image_shape": (1, 1, 1, 3, 2, 2),
            "view_names": ["camera_front_wide_120fov"],
        }
    ]
    runtime.close()


def test_omnidreams_webrtc_cli_builds_keyboard_driving_spec(tmp_path: Path) -> None:
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
            "--scene-dir",
            str(tmp_path / "scene"),
            "--scene-uuid",
            "scene-1",
            "--scene-variant",
            "rain",
            "--camera-name",
            "camera_front_wide_120fov",
            "--fps",
            "24",
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
            "--debug-serve-hdmaps",
            "--prefer-sw-encoder",
        ]
    )

    spec = _webrtc_spec(args, device="cuda:3")

    assert spec.model_id == OMNIDREAMS_MODEL_ID
    assert spec.preset_id == DEFAULT_OMNIDREAMS_PRESET
    assert spec.input_mode == "keyboard-driving"
    assert isinstance(spec.scenario, OmnidreamsWebRTCScenario)
    assert spec.scenario.scene_dir == tmp_path / "scene"
    assert spec.scenario.scene_uuid == "scene-1"
    assert spec.scenario.scene_variant == "rain"
    assert spec.scenario.camera_name == "camera_front_wide_120fov"
    assert spec.scenario.debug_serve_hdmaps is True
    assert spec.scenario.prefer_sw_encoder is True
    assert isinstance(spec.output, WebRTCOutputSpec)
    assert spec.output.host == "127.0.0.1"
    assert spec.output.port == 9090
    assert spec.output.fps == 24
    assert spec.output.video_width == 64
    assert spec.output.video_height == 32
    assert spec.output.warmup_chunks == 0
    assert spec.output.warmup_timeout_s == 1.5
    assert spec.output.client_liveness_timeout_s == 2.5
    assert spec.config is not None
    assert spec.config.device == "cuda:3"
    assert spec.config.runtime_options["seed"] == 123


def test_omnidreams_webrtc_demo_uses_shared_manager_with_model_config() -> None:
    pipeline_config = object()
    adapter = OmnidreamsDemoAdapter(webrtc_runtime_factory=_FakeWebRTCRuntime)
    spec = DemoSpec(
        model_id=OMNIDREAMS_MODEL_ID,
        preset_id=DEFAULT_OMNIDREAMS_PRESET,
        input_mode="keyboard-driving",
        scenario=OmnidreamsWebRTCScenario(
            scene_uuid="scene-1",
            scene_variant="rain",
            camera_name="camera_front_wide_120fov",
            debug_serve_hdmaps=True,
            prefer_sw_encoder=True,
        ),
        output=WebRTCOutputSpec(
            host="0.0.0.0",
            port=8082,
            fps=24,
            video_width=64,
            video_height=32,
            warmup_chunks=0,
            warmup_timeout_s=1.0,
        ),
        config=InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=DEFAULT_OMNIDREAMS_PRESET,
            device="cuda:7",
            runtime_options={"pipeline_config": pipeline_config, "seed": 123},
        ),
    )

    demo = build_webrtc_demo(spec=spec, adapter=adapter)

    assert isinstance(demo.runtime, _FakeWebRTCRuntime)
    assert isinstance(demo.session_manager, OmnidreamsDemoWebRTCSessionManager)
    assert demo.session_manager._runtime is demo.runtime
    assert demo.session_manager.runtime_config is demo.runtime.config
    assert demo.runtime_config is demo.runtime.config
    assert demo.runtime_config.pipeline_config is pipeline_config
    assert demo.runtime_config.pipeline_config_name == DEFAULT_OMNIDREAMS_PRESET
    assert demo.runtime_config.scene_uuid == "scene-1"
    assert demo.runtime_config.scene_variant == "rain"
    assert demo.runtime_config.seed == 123
    assert demo.runtime_config.device == "cuda:7"
    assert demo.runtime_config.video_width == 64
    assert demo.runtime_config.video_height == 32
    assert demo.runtime_config.fps == 24
    assert demo.runtime_config.debug_serve_hdmaps is True
    assert demo.runtime_config.encoder_backend == "default"
    assert demo.session_manager._model_name() == DEFAULT_OMNIDREAMS_PRESET
    assert demo.host == "0.0.0.0"
    assert demo.port == 8082


def test_omnidreams_webrtc_demo_installs_model_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_calls: list[dict[str, Any]] = []

    def fake_create_packaged_webrtc_app(**kwargs: Any) -> web.Application:
        app_calls.append(kwargs)
        app = web.Application()
        app[SESSION_MANAGER_KEY] = kwargs["session_manager"]
        kwargs["configure_app"](app)
        return app

    monkeypatch.setattr(
        demo_webrtc_module,
        "create_packaged_webrtc_app",
        fake_create_packaged_webrtc_app,
    )
    adapter = OmnidreamsDemoAdapter(webrtc_runtime_factory=_FakeWebRTCRuntime)
    spec = DemoSpec(
        model_id=OMNIDREAMS_MODEL_ID,
        preset_id=DEFAULT_OMNIDREAMS_PRESET,
        input_mode="keyboard-driving",
        scenario=OmnidreamsWebRTCScenario(),
        output=WebRTCOutputSpec(
            host="0.0.0.0",
            port=8082,
            warmup_timeout_s=1.0,
            preload_name="Test Omnidreams",
        ),
        config=InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=DEFAULT_OMNIDREAMS_PRESET,
            runtime_options={"pipeline_config": object()},
        ),
    )

    demo = build_webrtc_demo(spec=spec, adapter=adapter, create_app=True)

    assert demo.app is not None
    assert app_calls[0]["session_manager"] is demo.session_manager
    assert app_calls[0]["request_session_url"] == (
        "http://127.0.0.1:8082/request_session"
    )
    assert app_calls[0]["preload_name"] == "Test Omnidreams"
    assert str(app_calls[0]["model_web_resource"]).endswith("omnidreams/webrtc/web")
    route_paths = {resource.canonical for resource in demo.app.router.resources()}
    assert "/api/postprocess/options" in route_paths
    assert "/api/session/input" in route_paths


def test_omnidreams_webrtc_demo_serves_through_shared_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_calls: list[dict[str, Any]] = []

    def fake_create_packaged_webrtc_app(**kwargs: Any) -> web.Application:
        app = web.Application()
        app[SESSION_MANAGER_KEY] = kwargs["session_manager"]
        kwargs["configure_app"](app)
        return app

    def fake_server_runner(**kwargs: Any) -> None:
        server_calls.append(kwargs)

    monkeypatch.setattr(
        demo_webrtc_module,
        "create_packaged_webrtc_app",
        fake_create_packaged_webrtc_app,
    )
    adapter = OmnidreamsDemoAdapter(webrtc_runtime_factory=_FakeWebRTCRuntime)
    spec = DemoSpec(
        model_id=OMNIDREAMS_MODEL_ID,
        preset_id=DEFAULT_OMNIDREAMS_PRESET,
        input_mode="keyboard-driving",
        scenario={"scene_uuid": "scene-1"},
        output=WebRTCOutputSpec(
            host="0.0.0.0",
            port=8082,
            warmup_timeout_s=1.0,
        ),
        config=InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=DEFAULT_OMNIDREAMS_PRESET,
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
    assert server_calls[0]["port"] == 8082
    assert isinstance(demo.session_manager, OmnidreamsDemoWebRTCSessionManager)


class _RecordingOutputTarget:
    def open(self) -> None:
        return None

    def write(self, result: StepResult) -> None:
        del result

    def close(self) -> Sequence[OutputArtifact]:
        return ()


class _FakeOmnidreamsPipeline:
    def __init__(self) -> None:
        self.initialize_cache_calls: list[dict[str, Any]] = []
        self.released_encoders = False

    def initialize_cache(
        self,
        *,
        text: list[list[str]],
        image: torch.Tensor,
        view_names: list[str],
    ) -> object:
        self.initialize_cache_calls.append(
            {
                "text": text,
                "image_shape": tuple(image.shape),
                "view_names": view_names,
            }
        )
        return object()

    def release_oneshot_encoders(self) -> None:
        self.released_encoders = True

    def get_num_frames(self, autoregressive_index: int) -> int:
        del autoregressive_index
        return 1

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: object,
        hdmap: torch.Tensor,
    ) -> torch.Tensor:
        del cache, hdmap
        return torch.full((1, 1, 1, 3, 2, 2), float(autoregressive_index))

    def finalize(self, *, autoregressive_index: int, cache: object) -> dict[str, float]:
        del autoregressive_index, cache
        return {"denoise_s": 0.25}


class _FakeWebRTCRuntime:
    def __init__(self, config: Any) -> None:
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
