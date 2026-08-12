# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import omnidreams.demo as demo_package
import omnidreams.demo.app as demo_app_module
import omnidreams.demo.spec as spec_module
import pytest
import tomli as tomllib
import torch
from aiohttp import web
from omnidreams.config import OMNIDREAMS_RUNNERS
from omnidreams.demo import (
    DEFAULT_OMNIDREAMS_PRESET,
    OMNIDREAMS_CONDITIONING_LUDUS,
    OMNIDREAMS_CONDITIONING_PRECOMPUTED,
    OMNIDREAMS_MODEL_ID,
    LudusSceneConditioningProvider,
    OmnidreamsDemoAdapter,
    OmnidreamsLudusReplayScenario,
    OmnidreamsReplayScenario,
    OmnidreamsWebRTCScenario,
    PrecomputedHDMapProvider,
)
from omnidreams.demo.app import _replay_spec, _webrtc_spec, parse_args
from omnidreams.demo.controls import SPARSE_KEY_SEGMENTS_METADATA_KEY
from omnidreams.demo.replay import (
    OmnidreamsReplayRuntime,
    OmnidreamsReplayRuntimeOptions,
    OmnidreamsReplaySession,
)
from omnidreams.demo.runtime import (
    OmnidreamsRuntime,
    OmnidreamsRuntimeOptions,
    OmnidreamsSession,
)
from omnidreams.demo.webrtc import (
    OmnidreamsWebRTCModelRuntimeConfig,
    _should_use_legacy_webrtc_path,
    serve_omnidreams_webrtc_demo,
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
    OutputDecision,
    PreparedScenario,
    RuntimeHost,
    SessionInfo,
    UserInputWindow,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.serving.webrtc.manager import (
    BaseWebRTCSessionManager,
    ManagedWebRTCSession,
)
from flashdreams.serving.webrtc.server import SESSION_MANAGER_KEY
from flashdreams.serving.webrtc.services import (
    WEBRTC_SKIPPED_INPUTS_METADATA_KEY,
    WEBRTC_SKIPPED_WINDOW_METADATA_KEY,
    WebRTCInputSource,
    WebRTCTransportService,
)

pytestmark = pytest.mark.ci_cpu


def test_omnidreams_demo_defaults_to_stable_non_perf_preset() -> None:
    args = parse_args(["replay", "--output", "demo.mp4"])

    assert args.preset_id == "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
    assert not args.preset_id.endswith("-perf")


def test_omnidreams_direct_runner_launch_builds_null_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[DemoSpec] = []

    def fake_run_replay_demo(*, spec: DemoSpec, adapter: object) -> str:
        del adapter
        captured.append(spec)
        return "completed"

    monkeypatch.setattr(demo_app_module, "run_replay_demo", fake_run_replay_demo)
    config = OMNIDREAMS_RUNNERS["omnidreams"]

    result = demo_app_module.launch_from_runner(
        config=config,
        mode="null",
        scenario={"example_data": True, "total_blocks": 2},
        output={},
    )

    assert result == "completed"
    assert captured[0].preset_id == config.pipeline.name
    assert isinstance(captured[0].output, NullOutputSpec)


def test_omnidreams_replay_cli_builds_null_output_spec() -> None:
    args = parse_args(["replay", "--output-mode", "null"])

    spec = _replay_spec(args)

    assert spec.input_mode == "replay"
    assert isinstance(spec.output, NullOutputSpec)
    assert spec.config is not None
    assert spec.config.model_id == OMNIDREAMS_MODEL_ID


def test_omnidreams_replay_cli_builds_ludus_conditioning_spec(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "events": [
                    {"timestamp_s": 0.0, "event": "keydown", "key": "w"},
                    {"timestamp_s": 0.5, "event": "keyup", "key": "w"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "demo.mp4"

    args = parse_args(
        [
            "replay",
            "--conditioning-mode",
            OMNIDREAMS_CONDITIONING_LUDUS,
            "--keyboard-trace",
            str(trace_path),
            "--scene-uuid",
            "scene-1",
            "--scene-variant",
            "rain",
            "--camera-name",
            "camera_front_wide_120fov",
            "--seed",
            "123",
            "--total-blocks",
            "3",
            "--output",
            str(output_path),
        ]
    )

    spec = _replay_spec(args)

    assert spec.input_mode == "replay"
    assert isinstance(spec.scenario, dict)
    scenario = spec.scenario
    assert scenario["conditioning_mode"] == OMNIDREAMS_CONDITIONING_LUDUS
    assert scenario["keyboard_trace_path"] == trace_path
    assert scenario["scene_uuid"] == "scene-1"
    assert scenario["scene_variant"] == "rain"
    assert scenario["total_blocks"] == 3
    assert isinstance(spec.output, Mp4OutputSpec)
    assert spec.output.path == output_path
    assert spec.config is not None
    assert spec.config.seed == 123
    assert spec.config.runtime_options["seed"] == 123


def test_omnidreams_demo_adapter_declares_shared_modes() -> None:
    adapter = OmnidreamsDemoAdapter()

    assert adapter.model_id == OMNIDREAMS_MODEL_ID
    assert adapter.supported_input_modes() == ("replay", "keyboard-driving")
    assert adapter.supported_output_modes() == ("mp4", "null", "webrtc")
    assert adapter.supported_conditioning_modes() == (
        OMNIDREAMS_CONDITIONING_PRECOMPUTED,
        OMNIDREAMS_CONDITIONING_LUDUS,
    )
    assert [
        field.name
        for field in adapter.inference_input_schema.global_conditioning_fields
    ] == ["prompt", "first_frame", "scenario"]
    assert [field.name for field in adapter.inference_input_schema.step_fields] == [
        "hdmap"
    ]


def test_omnidreams_runtime_keeps_replay_aliases() -> None:
    assert OmnidreamsReplayRuntime is OmnidreamsRuntime
    assert OmnidreamsReplayRuntimeOptions is OmnidreamsRuntimeOptions
    assert OmnidreamsReplaySession is OmnidreamsSession


def test_omnidreams_demo_adapter_accepts_shared_runtime_factory() -> None:
    runtime = _FactoryRuntime()
    pipeline_config = object()
    calls: list[dict[str, Any]] = []

    def pipeline_factory(config_value: Any, device: str) -> Any:
        del config_value, device
        return object()

    def runtime_factory(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return runtime

    adapter = OmnidreamsDemoAdapter(
        runtime_factory=runtime_factory,
        pipeline_factory=pipeline_factory,
    )
    config = InferenceConfig(
        model_id=OMNIDREAMS_MODEL_ID,
        runtime_options={"pipeline_config": pipeline_config},
    )

    assert adapter.create_runtime(config) is runtime
    assert len(calls) == 1
    assert calls[0]["config"] == config
    options = calls[0]["options"]
    assert isinstance(options, OmnidreamsRuntimeOptions)
    assert options.pipeline_config is pipeline_config
    assert options.pipeline_factory is pipeline_factory


def test_omnidreams_demo_adapter_rejects_ambiguous_runtime_factories() -> None:
    def runtime_factory(**kwargs: Any) -> _FactoryRuntime:
        del kwargs
        return _FactoryRuntime()

    with pytest.raises(ValueError, match="runtime_factory"):
        OmnidreamsDemoAdapter(
            runtime_factory=runtime_factory,
            replay_runtime_factory=runtime_factory,
        )


def test_omnidreams_demo_does_not_import_legacy_webrtc_package() -> None:
    demo_dir = Path(demo_package.__file__).parent

    for path in demo_dir.glob("*.py"):
        assert "omnidreams.webrtc" not in path.read_text(encoding="utf-8"), path


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

    result = run_replay_demo(
        spec=spec,
        adapter=adapter,
        output_target_factory=lambda output_spec: output,
        runner=fake_runner,
    )

    assert result.status == "completed"
    assert result.artifacts == (
        OutputArtifact(kind="video/mp4", uri="memory://omnidreams"),
    )
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
        str(getattr(OMNIDREAMS_RUNNERS["omnidreams"], "prompt")),
    )


def test_omnidreams_replay_cli_can_disable_example_data(tmp_path: Path) -> None:
    args = parse_args(
        ["replay", "--no-example-data", "--output", str(tmp_path / "demo.mp4")]
    )
    spec = _replay_spec(args)

    with pytest.raises(ValueError, match="requires hdmap_video_paths"):
        OmnidreamsDemoAdapter().prepare_scenario(spec)


def test_omnidreams_precomputed_hdmap_provider_prepares_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import omnidreams.demo.providers as providers_module

    hdmap = tmp_path / "hdmap.mp4"
    first_frame = tmp_path / "first.png"
    hdmap.write_bytes(b"fake")
    first_frame.write_bytes(b"fake")
    loaded_hdmap = torch.arange(3 * 3 * 2 * 2).reshape(3, 3, 2, 2)
    monkeypatch.setattr(
        providers_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.ones(1, 3, 2, 2),
    )
    monkeypatch.setattr(
        providers_module,
        "_load_video",
        lambda *args, **kwargs: loaded_hdmap,
    )
    adapter = OmnidreamsDemoAdapter()
    spec = _replay_demo_spec(
        tmp_path=tmp_path,
        hdmap=hdmap,
        first_frame=first_frame,
        total_blocks=2,
    )
    prepared = adapter.prepare_scenario(spec)

    provider = adapter.create_model_input_provider(spec, prepared)

    assert isinstance(provider, PrecomputedHDMapProvider)
    initial = provider.prepare_initial_input()
    scenario = initial.global_conditioning["scenario"]
    assert isinstance(scenario, OmnidreamsReplayScenario)
    assert initial.global_conditioning["prompt"] == [["drive"]]
    assert initial.global_conditioning["first_frame"].shape == (1, 1, 1, 3, 2, 2)
    assert initial.metadata["view_names"] == ("camera_front_wide_120fov",)

    step = provider.prepare_step(
        request=StepRequirements(step_index=0, input_frame_count=2),
        user_window=UserInputWindow(start_s=0.0, end_s=1.0),
    )

    assert step.inference_input is not None
    hdmap_chunk = step.inference_input.step["hdmap"]
    assert isinstance(hdmap_chunk, torch.Tensor)
    assert hdmap_chunk.shape == (1, 1, 2, 3, 2, 2)
    torch.testing.assert_close(hdmap_chunk[0, 0], loaded_hdmap[:2])

    exhausted = provider.prepare_step(
        request=StepRequirements(step_index=1, input_frame_count=2),
        user_window=UserInputWindow(start_s=1.0, end_s=2.0),
    )

    assert exhausted.inference_input is None
    assert exhausted.control.close_session is True
    provider.reset()
    reset_step = provider.prepare_step(
        request=StepRequirements(step_index=0, input_frame_count=1),
        user_window=UserInputWindow(start_s=0.0, end_s=1.0),
    )
    assert reset_step.inference_input is not None
    torch.testing.assert_close(
        reset_step.inference_input.step["hdmap"][0, 0],
        loaded_hdmap[:1],
    )
    provider.close()


def test_omnidreams_ludus_provider_prepares_deterministic_hdmaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scene, rasterizers = _install_fake_ludus_provider_dependencies(monkeypatch)
    scene_path = tmp_path / "scene.usdz"
    scene_path.write_bytes(b"fake")
    adapter = OmnidreamsDemoAdapter()
    spec = _ludus_replay_demo_spec(
        tmp_path=tmp_path,
        scene_path=scene_path,
        total_blocks=2,
    )
    prepared = adapter.prepare_scenario(spec)

    provider = adapter.create_model_input_provider(spec, prepared)

    assert isinstance(provider, LudusSceneConditioningProvider)
    initial = provider.prepare_initial_input()
    scenario = initial.global_conditioning["scenario"]
    assert isinstance(scenario, OmnidreamsLudusReplayScenario)
    assert scenario.camera_names == ("camera_front_wide_120fov",)
    assert initial.global_conditioning["prompt"] == [["city scene"]]
    assert initial.global_conditioning["first_frame"].shape == (1, 1, 1, 3, 2, 2)
    assert initial.metadata["view_names"] == ("camera_front_wide_120fov",)

    first = provider.prepare_step(
        request=StepRequirements(step_index=0, input_frame_count=2),
        user_window=UserInputWindow(
            start_s=0.0,
            end_s=2 / 30,
            frame_times=(1 / 30, 2 / 30),
        ),
    )

    assert first.inference_input is not None
    first_hdmap = first.inference_input.step["hdmap"]
    assert isinstance(first_hdmap, torch.Tensor)
    assert first_hdmap.shape == (1, 1, 2, 3, 2, 2)
    assert first.inference_input.metadata["frame_timestamps_us"] == (1_000, 34_333)
    assert first.inference_input.metadata["keyboard_segments"] == (
        (0.0, 2 / 30, ("w",)),
    )
    assert len(rasterizers) == 1
    assert rasterizers[0].calls[0]["timestamps_us"] == (1_000, 34_333)
    assert rasterizers[0].calls[0]["rig_poses_world"].shape == (2, 4, 4)
    assert rasterizers[0].calls[0]["rig_poses_world"][0, 0, 3] > 0

    provider.reset()
    reset_first = provider.prepare_step(
        request=StepRequirements(step_index=0, input_frame_count=2),
        user_window=UserInputWindow(
            start_s=0.0,
            end_s=2 / 30,
            frame_times=(1 / 30, 2 / 30),
        ),
    )

    assert reset_first.inference_input is not None
    torch.testing.assert_close(reset_first.inference_input.step["hdmap"], first_hdmap)

    provider.reset()
    realtime_first = provider.prepare_step(
        request=StepRequirements(step_index=0, input_frame_count=2),
        user_window=UserInputWindow(
            start_s=0.0,
            end_s=2 / 30,
            frame_times=(1 / 30, 2 / 30),
            inputs=UserInputs(
                events=(
                    UserInputEvent(
                        timestamp_s=0.0,
                        event_type="key_down",
                        payload={"key": "w"},
                    ),
                )
            ),
        ),
    )

    assert realtime_first.inference_input is not None
    torch.testing.assert_close(
        realtime_first.inference_input.step["hdmap"],
        first_hdmap,
    )
    provider.close()
    assert rasterizers[0].closed is True


def test_omnidreams_ludus_provider_advances_replay_trace_without_frame_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scene, _rasterizers = _install_fake_ludus_provider_dependencies(monkeypatch)
    scene_path = tmp_path / "scene.usdz"
    scene_path.write_bytes(b"fake")
    adapter = OmnidreamsDemoAdapter()
    spec = _ludus_replay_demo_spec(
        tmp_path=tmp_path,
        scene_path=scene_path,
        total_blocks=2,
        keyboard_events=(
            {"timestamp_s": 0.0, "event": "keydown", "key": "w"},
            {"timestamp_s": 0.08, "event": "keydown", "key": "d"},
        ),
    )
    prepared = adapter.prepare_scenario(spec)
    provider = adapter.create_model_input_provider(spec, prepared)
    assert isinstance(provider, LudusSceneConditioningProvider)
    provider.prepare_initial_input()
    replay_window = UserInputWindow(start_s=0.0, end_s=3600.0)

    first = provider.prepare_step(
        request=StepRequirements(step_index=0, input_frame_count=2),
        user_window=replay_window,
    )
    second = provider.prepare_step(
        request=StepRequirements(step_index=1, input_frame_count=2),
        user_window=replay_window,
    )

    assert first.inference_input is not None
    assert second.inference_input is not None
    assert first.inference_input.metadata["keyboard_segments"] == (
        (0.0, 2 / 30, ("w",)),
    )
    assert second.inference_input.metadata["keyboard_segments"] == (
        (2 / 30, 0.08, ("w",)),
        (0.08, 4 / 30, ("d", "w")),
    )


def test_omnidreams_ludus_provider_folds_webrtc_skipped_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scene, _rasterizers = _install_fake_ludus_provider_dependencies(monkeypatch)
    scene_path = tmp_path / "scene.usdz"
    scene_path.write_bytes(b"fake")
    adapter = OmnidreamsDemoAdapter()
    spec = _ludus_replay_demo_spec(
        tmp_path=tmp_path,
        scene_path=scene_path,
        total_blocks=1,
    )
    prepared = adapter.prepare_scenario(spec)
    provider = adapter.create_model_input_provider(spec, prepared)
    assert isinstance(provider, LudusSceneConditioningProvider)
    provider.prepare_initial_input()

    skipped_release = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.1,
                event_type="key_up",
                payload={"key": "w"},
            ),
        )
    )
    step = provider.prepare_step(
        request=StepRequirements(step_index=0, input_frame_count=2),
        user_window=UserInputWindow(
            start_s=0.25,
            end_s=0.25 + 2 / 30,
            frame_times=(0.25 + 1 / 30, 0.25 + 2 / 30),
            metadata={
                WEBRTC_SKIPPED_INPUTS_METADATA_KEY: skipped_release,
                WEBRTC_SKIPPED_WINDOW_METADATA_KEY: (0.0, 0.25),
            },
        ),
    )

    assert step.inference_input is not None
    assert step.inference_input.metadata["keyboard_segments"] == (
        (0.25, 0.25 + 2 / 30, ()),
    )


def test_omnidreams_replay_run_mode_uses_precomputed_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import omnidreams.demo.providers as providers_module

    hdmap = tmp_path / "hdmap.mp4"
    first_frame = tmp_path / "first.png"
    hdmap.write_bytes(b"fake")
    first_frame.write_bytes(b"fake")
    loaded_hdmap = torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
    pipeline = _FakeOmnidreamsPipeline()
    sink = _RecordingOutputSink()
    monkeypatch.setattr(
        providers_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )
    monkeypatch.setattr(
        providers_module,
        "_load_video",
        lambda *args, **kwargs: loaded_hdmap,
    )
    adapter = OmnidreamsDemoAdapter(
        pipeline_factory=lambda pipeline_config, device: pipeline,
    )
    spec = _replay_demo_spec(
        tmp_path=tmp_path,
        hdmap=hdmap,
        first_frame=first_frame,
        total_blocks=2,
    )

    result = run_replay_demo(
        spec=spec,
        adapter=adapter,
        output_sink_factory=lambda output_spec: sink,
    )

    assert result.status == "completed"
    assert result.artifacts == (
        OutputArtifact(kind="video/mp4", uri="memory://omnidreams"),
    )
    assert [result.step_index for result in sink.results] == [0, 1]
    assert pipeline.initialize_cache_calls == [
        {
            "text": [["drive"]],
            "image_shape": (1, 1, 1, 3, 2, 2),
            "view_names": ["camera_front_wide_120fov"],
        }
    ]
    assert len(pipeline.generated_hdmaps) == 2
    torch.testing.assert_close(pipeline.generated_hdmaps[0][0, 0], loaded_hdmap[:1])
    torch.testing.assert_close(pipeline.generated_hdmaps[1][0, 0], loaded_hdmap[1:2])


def test_omnidreams_replay_run_mode_uses_ludus_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ludus_provider_dependencies(monkeypatch)
    scene_path = tmp_path / "scene.usdz"
    scene_path.write_bytes(b"fake")
    pipeline = _FakeOmnidreamsPipeline()
    sink = _RecordingOutputSink()
    adapter = OmnidreamsDemoAdapter(
        pipeline_factory=lambda pipeline_config, device: pipeline,
    )
    spec = _ludus_replay_demo_spec(
        tmp_path=tmp_path,
        scene_path=scene_path,
        total_blocks=2,
    )

    result = run_replay_demo(
        spec=spec,
        adapter=adapter,
        output_sink_factory=lambda output_spec: sink,
    )

    assert result.status == "completed"
    assert result.artifacts == (
        OutputArtifact(kind="video/mp4", uri="memory://omnidreams"),
    )
    assert [result.step_index for result in sink.results] == [0, 1]
    assert pipeline.initialize_cache_calls == [
        {
            "text": [["city scene"]],
            "image_shape": (1, 1, 1, 3, 2, 2),
            "view_names": ["camera_front_wide_120fov"],
        }
    ]
    assert len(pipeline.generated_hdmaps) == 2
    assert pipeline.generated_hdmaps[0].shape == (1, 1, 1, 3, 2, 2)
    assert pipeline.generated_hdmaps[1].shape == (1, 1, 1, 3, 2, 2)
    assert not torch.equal(pipeline.generated_hdmaps[0], pipeline.generated_hdmaps[1])


def test_omnidreams_replay_null_output_uses_precomputed_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import omnidreams.demo.providers as providers_module

    hdmap = tmp_path / "hdmap.mp4"
    first_frame = tmp_path / "first.png"
    hdmap.write_bytes(b"fake")
    first_frame.write_bytes(b"fake")
    loaded_hdmap = torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
    pipeline = _FakeOmnidreamsPipeline()
    monkeypatch.setattr(
        providers_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )
    monkeypatch.setattr(
        providers_module,
        "_load_video",
        lambda *args, **kwargs: loaded_hdmap,
    )
    adapter = OmnidreamsDemoAdapter(
        pipeline_factory=lambda pipeline_config, device: pipeline,
    )
    spec = _replay_demo_spec(
        tmp_path=tmp_path,
        hdmap=hdmap,
        first_frame=first_frame,
        total_blocks=2,
        output=NullOutputSpec(),
    )

    result = run_replay_demo(spec=spec, adapter=adapter)

    assert result.status == "completed"
    assert result.artifacts == ()
    assert pipeline.initialize_cache_calls == [
        {
            "text": [["drive"]],
            "image_shape": (1, 1, 1, 3, 2, 2),
            "view_names": ["camera_front_wide_120fov"],
        }
    ]
    assert len(pipeline.generated_hdmaps) == 2
    torch.testing.assert_close(pipeline.generated_hdmaps[0][0, 0], loaded_hdmap[:1])
    torch.testing.assert_close(pipeline.generated_hdmaps[1][0, 0], loaded_hdmap[1:2])


def test_omnidreams_replay_output_target_path_uses_precomputed_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import omnidreams.demo.providers as providers_module

    hdmap = tmp_path / "hdmap.mp4"
    first_frame = tmp_path / "first.png"
    hdmap.write_bytes(b"fake")
    first_frame.write_bytes(b"fake")
    loaded_hdmap = torch.arange(1 * 3 * 2 * 2).reshape(1, 3, 2, 2)
    pipeline = _FakeOmnidreamsPipeline()
    output = _RecordingOutputTarget()
    monkeypatch.setattr(
        providers_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )
    monkeypatch.setattr(
        providers_module,
        "_load_video",
        lambda *args, **kwargs: loaded_hdmap,
    )
    adapter = OmnidreamsDemoAdapter(
        pipeline_factory=lambda pipeline_config, device: pipeline,
    )
    spec = _replay_demo_spec(
        tmp_path=tmp_path,
        hdmap=hdmap,
        first_frame=first_frame,
        total_blocks=1,
    )

    result = run_replay_demo(
        spec=spec,
        adapter=adapter,
        output_target_factory=lambda output_spec: output,
    )

    assert result.status == "completed"
    assert [result.step_index for result in output.results] == [0]
    assert len(pipeline.generated_hdmaps) == 1
    torch.testing.assert_close(pipeline.generated_hdmaps[0][0, 0], loaded_hdmap)


def test_omnidreams_replay_runtime_generates_video_step_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import omnidreams.demo.runtime as runtime_module

    hdmap = tmp_path / "hdmap.mp4"
    first_frame = tmp_path / "first.png"
    hdmap.write_bytes(b"fake")
    first_frame.write_bytes(b"fake")
    pipeline = _FakeOmnidreamsPipeline()
    monkeypatch.setattr(
        runtime_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )

    runtime = OmnidreamsRuntime(
        config=InferenceConfig(model_id=OMNIDREAMS_MODEL_ID, device="cpu"),
        options=OmnidreamsRuntimeOptions(
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
    assert isinstance(session, OmnidreamsSession)

    requirements = session.next_step_requirements()
    assert isinstance(requirements, StepRequirements)
    assert requirements.step_index == 0
    assert requirements.input_frame_count == 1
    request = session.next_step_request()
    assert request is not None
    assert request.step_index == 0
    assert request.metadata["input_frame_count"] == 1
    result = session.step(InferenceInput(step={"hdmap": torch.zeros(1, 1, 1, 3, 2, 2)}))

    assert result.step_index == 0
    assert result.frame_count == 1
    assert isinstance(result, StepResult)
    assert result.layout == "bvtchw"
    assert result.video_chunk.shape == (1, 1, 1, 3, 2, 2)
    assert result.metrics["denoise_s"] == 0.25
    assert session.next_step_request() is None
    assert pipeline.released_encoders is True
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
    legacy_module_name = "omnidreams.demo.webrtc_legacy"
    sys.modules.pop(legacy_module_name, None)
    pipeline_config = object()
    runtime = _FactoryRuntime()
    spec = DemoSpec(
        model_id=OMNIDREAMS_MODEL_ID,
        preset_id=DEFAULT_OMNIDREAMS_PRESET,
        input_mode="keyboard-driving",
        scenario=OmnidreamsWebRTCScenario(
            scene_uuid="scene-1",
            scene_variant="rain",
            camera_name="camera_front_wide_120fov",
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

    calls: list[dict[str, Any]] = []
    runtime_calls: list[dict[str, Any]] = []

    def shared_runtime_factory(**kwargs: Any) -> Any:
        runtime_calls.append(kwargs)
        return runtime

    serve_omnidreams_webrtc_demo(
        spec=spec,
        world_rank=1,
        shared_runtime_factory=shared_runtime_factory,
        server_runner=lambda **kwargs: calls.append(kwargs),
    )

    manager = calls[0]["session_manager"]
    assert type(manager) is BaseWebRTCSessionManager
    assert manager._runtime is runtime
    assert isinstance(manager._shared_host, RuntimeHost)
    assert isinstance(manager._shared_adapter, OmnidreamsDemoAdapter)
    assert isinstance(manager._shared_scenario, PreparedScenario)
    assert manager.runtime_config.pipeline_config is pipeline_config
    assert manager.runtime_config.pipeline_config_name == DEFAULT_OMNIDREAMS_PRESET
    assert manager.runtime_config.scene_uuid == "scene-1"
    assert manager.runtime_config.scene_variant == "rain"
    assert manager.runtime_config.seed == 123
    assert manager.runtime_config.device == "cuda:7"
    assert manager.runtime_config.video_width == 64
    assert manager.runtime_config.video_height == 32
    assert manager.runtime_config.fps == 24
    assert manager.runtime_config.debug_serve_hdmaps is False
    assert manager.runtime_config.encoder_backend == "default"
    assert manager.identity == DEFAULT_OMNIDREAMS_PRESET
    assert len(runtime_calls) == 1
    runtime_config = runtime_calls[0]["config"]
    assert runtime_config.seed == 123
    assert runtime_config.runtime_options["pipeline_config"] is pipeline_config
    assert (
        runtime_config.runtime_options["release_oneshot_encoders_after_cache_init"]
        is False
    )
    options = runtime_calls[0]["options"]
    assert isinstance(options, OmnidreamsRuntimeOptions)
    assert options.release_oneshot_encoders_after_cache_init is False
    scenario = manager._shared_scenario.initial_inputs.global_conditioning["scenario"]
    assert isinstance(scenario, OmnidreamsLudusReplayScenario)
    assert scenario.scene_uuid == "scene-1"
    assert scenario.scene_variant == "rain"
    assert scenario.pixel_width == 64
    assert scenario.pixel_height == 32
    assert scenario.fps == 24
    assert calls[0]["host"] == "0.0.0.0"
    assert calls[0]["port"] == 8082
    assert legacy_module_name not in sys.modules


def test_omnidreams_webrtc_demo_keeps_legacy_runtime_factory_path() -> None:
    spec = DemoSpec(
        model_id=OMNIDREAMS_MODEL_ID,
        preset_id=DEFAULT_OMNIDREAMS_PRESET,
        input_mode="keyboard-driving",
        scenario=OmnidreamsWebRTCScenario(),
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
            runtime_options={"pipeline_config": object(), "seed": 123},
        ),
    )

    calls: list[dict[str, Any]] = []
    serve_omnidreams_webrtc_demo(
        spec=spec,
        world_rank=1,
        runtime_factory=_FakeWebRTCRuntime,
        server_runner=lambda **kwargs: calls.append(kwargs),
    )

    manager = calls[0]["session_manager"]
    runtime = manager._runtime
    assert isinstance(runtime, _FakeWebRTCRuntime)
    assert manager.runtime_config is runtime.config
    assert runtime.config.debug_serve_hdmaps is False


def test_omnidreams_webrtc_demo_keeps_legacy_fallback_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _should_use_legacy_webrtc_path(
        scenario=OmnidreamsWebRTCScenario(),
        runtime_factory=_FakeWebRTCRuntime,
    )
    assert _should_use_legacy_webrtc_path(
        scenario=OmnidreamsWebRTCScenario(debug_serve_hdmaps=True),
        runtime_factory=None,
    )

    monkeypatch.setenv("WORLD_SIZE", "2")
    assert _should_use_legacy_webrtc_path(
        scenario=OmnidreamsWebRTCScenario(),
        runtime_factory=None,
    )

    monkeypatch.setenv("WORLD_SIZE", "not-an-int")
    assert not _should_use_legacy_webrtc_path(
        scenario=OmnidreamsWebRTCScenario(),
        runtime_factory=None,
    )


def test_omnidreams_webrtc_demo_installs_model_assets_without_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flashdreams.serving.webrtc.demo as shared_webrtc_module

    app_calls: list[dict[str, Any]] = []

    def fake_create_packaged_webrtc_app(**kwargs: Any) -> web.Application:
        app_calls.append(kwargs)
        app = web.Application()
        app[SESSION_MANAGER_KEY] = kwargs["session_manager"]
        if configure_app := kwargs["configure_app"]:
            configure_app(app)
        return app

    monkeypatch.setattr(
        shared_webrtc_module,
        "create_packaged_webrtc_app",
        fake_create_packaged_webrtc_app,
    )
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

    app = serve_omnidreams_webrtc_demo(
        spec=spec,
        shared_runtime_factory=lambda **kwargs: _FactoryRuntime(),
        server_runner=lambda **kwargs: None,
    )

    assert isinstance(app, web.Application)
    assert app_calls[0]["session_manager"] is app[SESSION_MANAGER_KEY]
    assert app_calls[0]["request_session_url"] == (
        "http://127.0.0.1:8082/request_session"
    )
    assert app_calls[0]["preload_name"] == "Test Omnidreams"
    assert str(app_calls[0]["model_web_resource"]).endswith("omnidreams/demo/web")
    assert app_calls[0]["configure_app"] is None


def test_omnidreams_webrtc_adapter_caps_video_display_size() -> None:
    web_dir = Path(demo_package.__file__).resolve().parent / "web"
    adapter_js = (web_dir / "adapter.js").read_text(encoding="utf-8")
    adapter_css = (web_dir / "adapter.css").read_text(encoding="utf-8")

    assert 'stylesheet: "/model-static/adapter.css?v=model-ui-v2"' in adapter_js
    assert ".stageVideo" in adapter_css
    assert "1280px" in adapter_css
    assert "704px" in adapter_css
    assert "calc(" not in adapter_css
    assert "object-fit: contain" in adapter_css

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        meta = tomllib.load(fh)
    package_data = meta["tool"]["setuptools"]["package-data"]["omnidreams.demo"]
    assert "web/adapter.js" in package_data
    assert "web/adapter.css" in package_data


def test_omnidreams_webrtc_demo_serves_through_shared_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flashdreams.serving.webrtc.demo as shared_webrtc_module

    server_calls: list[dict[str, Any]] = []

    def fake_create_packaged_webrtc_app(**kwargs: Any) -> web.Application:
        app = web.Application()
        app[SESSION_MANAGER_KEY] = kwargs["session_manager"]
        if configure_app := kwargs["configure_app"]:
            configure_app(app)
        return app

    def fake_server_runner(**kwargs: Any) -> None:
        server_calls.append(kwargs)

    monkeypatch.setattr(
        shared_webrtc_module,
        "create_packaged_webrtc_app",
        fake_create_packaged_webrtc_app,
    )
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

    app = serve_omnidreams_webrtc_demo(
        spec=spec,
        world_rank=0,
        shared_runtime_factory=lambda **kwargs: _FactoryRuntime(),
        server_runner=fake_server_runner,
    )

    assert len(server_calls) == 1
    assert server_calls[0]["world_rank"] == 0
    assert server_calls[0]["app"] is app
    assert server_calls[0]["host"] == "0.0.0.0"
    assert server_calls[0]["port"] == 8082
    assert type(server_calls[0]["session_manager"]) is BaseWebRTCSessionManager


@pytest.mark.asyncio
async def test_omnidreams_webrtc_runtime_uses_shared_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnidreams.demo.webrtc_legacy import OmnidreamsWebRTCModelRuntime

    _scene, rasterizers = _install_fake_ludus_provider_dependencies(monkeypatch)
    scene_path = tmp_path / "scene.usdz"
    scene_path.write_bytes(b"fake")
    pipeline = _VariableFrameOmnidreamsPipeline((2, 3))
    config = OmnidreamsWebRTCModelRuntimeConfig(
        pipeline_config_name="fake",
        pipeline_config=object(),
        pipeline_factory=lambda pipeline_config, device: pipeline,
        scene_dir=scene_path,
        device="cpu",
        fps=30,
        video_height=2,
        video_width=2,
        warmup_chunks=0,
    )
    runtime = OmnidreamsWebRTCModelRuntime(config=config)
    await runtime.initialize()
    await runtime.reset_for_new_session()
    session = await runtime.start_inference_session()

    first_request = session.next_step_request()
    assert first_request is not None
    assert first_request.metadata["input_frame_count"] == 2
    first = session.step(
        runtime.input_mapping.map_step_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=InferenceInput(
                metadata={
                    SPARSE_KEY_SEGMENTS_METADATA_KEY: (
                        (0.0, 2 / 30, frozenset({"w"})),
                    ),
                    "frame_times": (1 / 30, 2 / 30),
                    "window_start_s": 0.0,
                    "window_end_s": 2 / 30,
                }
            ),
            request=first_request,
        )
    )
    second_request = session.next_step_request()
    assert second_request is not None
    assert second_request.metadata["input_frame_count"] == 3
    second = session.step(
        runtime.input_mapping.map_step_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=InferenceInput(
                metadata={
                    SPARSE_KEY_SEGMENTS_METADATA_KEY: (
                        (2 / 30, 5 / 30, frozenset({"d"})),
                    ),
                    "frame_times": (3 / 30, 4 / 30, 5 / 30),
                    "window_start_s": 2 / 30,
                    "window_end_s": 5 / 30,
                }
            ),
            request=second_request,
        )
    )

    assert (first.step_index, first.frame_count) == (0, 2)
    assert (second.step_index, second.frame_count) == (1, 3)
    assert isinstance(session, OmnidreamsSession) is False
    assert pipeline.initialize_cache_calls == [
        {
            "text": [["city scene"]],
            "image_shape": (1, 1, 1, 3, 2, 2),
            "view_names": ["camera_front_wide_120fov"],
        }
    ]
    assert [tuple(hdmap.shape) for hdmap in pipeline.generated_hdmaps] == [
        (1, 1, 2, 3, 2, 2),
        (1, 1, 3, 3, 2, 2),
    ]
    assert rasterizers[0].calls[0]["timestamps_us"] == (1_000, 34_333)
    assert rasterizers[0].calls[1]["timestamps_us"] == (67_666, 100_999, 134_332)
    session.close()
    await runtime.close()
    assert rasterizers[0].closed is True


@pytest.mark.asyncio
async def test_omnidreams_webrtc_runtime_keeps_encoders_after_warmup_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnidreams.demo.webrtc_legacy import OmnidreamsWebRTCModelRuntime

    _install_fake_ludus_provider_dependencies(monkeypatch)
    scene_path = tmp_path / "scene.usdz"
    scene_path.write_bytes(b"fake")
    pipeline = _FailsIfEncodersReleasedOmnidreamsPipeline()
    config = OmnidreamsWebRTCModelRuntimeConfig(
        pipeline_config_name="fake",
        pipeline_config=object(),
        pipeline_factory=lambda pipeline_config, device: pipeline,
        scene_dir=scene_path,
        device="cpu",
        fps=30,
        video_height=2,
        video_width=2,
        warmup_chunks=0,
    )
    runtime = OmnidreamsWebRTCModelRuntime(config=config)
    await runtime.initialize()

    await runtime.reset_for_new_session()
    warmup_session = await runtime.start_inference_session()
    warmup_session.close()
    await runtime.reset_for_new_session()
    browser_session = await runtime.start_inference_session()

    assert pipeline.released_encoders is False
    assert len(pipeline.initialize_cache_calls) == 2
    browser_session.close()
    await runtime.close()


@pytest.mark.asyncio
async def test_omnidreams_webrtc_manager_drives_shared_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scene, rasterizers = _install_fake_ludus_provider_dependencies(monkeypatch)
    scene_path = tmp_path / "scene.usdz"
    scene_path.write_bytes(b"fake")
    pipeline = _VariableFrameOmnidreamsPipeline((2, 3))
    adapter = OmnidreamsDemoAdapter(
        pipeline_factory=lambda pipeline_config, device: pipeline,
    )
    spec = DemoSpec(
        model_id=OMNIDREAMS_MODEL_ID,
        preset_id=DEFAULT_OMNIDREAMS_PRESET,
        input_mode="keyboard-driving",
        scenario=OmnidreamsWebRTCScenario(
            scene_dir=scene_path,
            scene_uuid="scene-1",
            camera_name="camera_front_wide_120fov",
        ),
        output=WebRTCOutputSpec(
            fps=30,
            video_width=2,
            video_height=2,
            warmup_chunks=0,
            warmup_timeout_s=1.0,
        ),
        config=InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=DEFAULT_OMNIDREAMS_PRESET,
            device="cpu",
            seed=123,
            runtime_options={
                "pipeline_config": object(),
                "seed": 123,
                "release_oneshot_encoders_after_cache_init": False,
            },
        ),
    )
    prepared = adapter.prepare_scenario(spec)
    assert spec.config is not None
    runtime = adapter.create_runtime(spec.config)
    runtime_config = OmnidreamsWebRTCModelRuntimeConfig(
        pipeline_config_name=DEFAULT_OMNIDREAMS_PRESET,
        pipeline_config=object(),
        scene_dir=scene_path,
        scene_uuid="scene-1",
        device="cpu",
        fps=30,
        video_height=2,
        video_width=2,
        warmup_chunks=0,
    )
    host = RuntimeHost(runtime)
    manager = BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=runtime_config,
        fps=runtime_config.fps,
        identity=runtime_config.pipeline_config_name,
        supported_control_keys=frozenset({"w", "a", "s", "d"}),
        shared_host=host,
        shared_adapter=adapter,
        shared_spec=spec,
        shared_scenario=prepared,
    )
    manager._runtime_ready = True
    loop = asyncio.get_running_loop()
    context = manager._shared_run_context(loop)
    reservation = context.admission.try_reserve()
    assert reservation is not None
    resampler = _FakeWebRTCResampler(start_v=loop.time(), fps=runtime_config.fps)
    input_source = WebRTCInputSource(resampler=resampler)
    input_source.handle_browser_payload(
        {"type": "action", "action": {"event": "step"}},
        timestamp_s=loop.time(),
    )
    transport = WebRTCTransportService(loop=loop)
    channel = _FakeWebRTCChannel()
    managed_session = ManagedWebRTCSession(
        runtime=runtime,
        video_track=_FakeWebRTCVideoTrack(fps=runtime_config.fps),  # ty:ignore[invalid-argument-type]
        video_encoder=_FakeWebRTCVideoEncoder(),  # ty:ignore[invalid-argument-type]
        peer_connection=_FakeWebRTCPeerConnection(),
        resampler=resampler,  # ty:ignore[invalid-argument-type]
        control_channel=channel,
        input_source=input_source,
        transport=transport,
        reservation=reservation,
        last_client_message_at=loop.time(),
    )
    manager._active_session = managed_session
    try:
        managed_session.generation_task = asyncio.create_task(
            manager._run_realtime_driver_session(
                managed_session=managed_session,
                context=context,
                session_input=None,
            )
        )
        chunk = await _wait_for_chunk_done(channel)

        assert chunk["type"] == "chunk_done"
        assert chunk["model"] == DEFAULT_OMNIDREAMS_PRESET
        assert chunk["num_frames"] == 2
        assert [tuple(hdmap.shape) for hdmap in pipeline.generated_hdmaps][:1] == [
            (1, 1, 2, 3, 2, 2)
        ]
        assert rasterizers[0].calls[0]["timestamps_us"] == (1_000, 34_333)
        assert isinstance(
            managed_session.input_source,
            WebRTCInputSource,
        )
    finally:
        transport.close("test complete")
        await manager.shutdown()


class _RecordingOutputTarget:
    def __init__(self) -> None:
        self.results: list[StepResult] = []

    def open(self) -> None:
        return None

    def write(self, result: StepResult) -> None:
        self.results.append(result)

    def close(self) -> Sequence[OutputArtifact]:
        return ()


def _replay_demo_spec(
    *,
    tmp_path: Path,
    hdmap: Path,
    first_frame: Path,
    total_blocks: int,
    output: Mp4OutputSpec | NullOutputSpec | None = None,
) -> DemoSpec:
    return DemoSpec(
        model_id=OMNIDREAMS_MODEL_ID,
        preset_id=DEFAULT_OMNIDREAMS_PRESET,
        input_mode="replay",
        scenario={
            "prompt": "drive",
            "hdmap_video_paths": (hdmap,),
            "first_frame_paths": (first_frame,),
            "camera_names": ("camera_front_wide_120fov",),
            "total_blocks": total_blocks,
            "pixel_height": 2,
            "pixel_width": 2,
            "fps": 30,
        },
        output=output or Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=30),
        config=InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=DEFAULT_OMNIDREAMS_PRESET,
            device="cpu",
            runtime_options={"pipeline_config": object()},
        ),
    )


def _ludus_replay_demo_spec(
    *,
    tmp_path: Path,
    scene_path: Path,
    total_blocks: int,
    keyboard_events: Sequence[Mapping[str, object]] = (
        {"timestamp_s": 0.0, "event": "keydown", "key": "w"},
        {"timestamp_s": 0.5, "event": "keyup", "key": "w"},
    ),
    output: Mp4OutputSpec | NullOutputSpec | None = None,
) -> DemoSpec:
    return DemoSpec(
        model_id=OMNIDREAMS_MODEL_ID,
        preset_id=DEFAULT_OMNIDREAMS_PRESET,
        input_mode="replay",
        scenario={
            "conditioning_mode": OMNIDREAMS_CONDITIONING_LUDUS,
            "keyboard_events": tuple(keyboard_events),
            "scene_path": scene_path,
            "scene_variant": "default",
            "camera_name": "camera_front_wide_120fov",
            "total_blocks": total_blocks,
            "pixel_height": 2,
            "pixel_width": 2,
            "fps": 30,
        },
        output=output or Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=30),
        config=InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=DEFAULT_OMNIDREAMS_PRESET,
            device="cpu",
            seed=123,
            runtime_options={"pipeline_config": object(), "seed": 123},
        ),
    )


def _install_fake_ludus_provider_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, list["_FakeLudusRasterizer"]]:
    import omnidreams.demo.providers as providers_module

    scene = SimpleNamespace(
        scene_id="fake-scene",
        prompt="city scene",
        initial_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        initial_rig_to_world=np.eye(4, dtype=np.float32),
        initial_timestamp_us=1_000,
    )
    rasterizers: list[_FakeLudusRasterizer] = []

    def fake_load_scene_bundle(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args, kwargs
        return scene

    def fake_new_rasterizer(*args: Any, **kwargs: Any) -> "_FakeLudusRasterizer":
        del args, kwargs
        rasterizer = _FakeLudusRasterizer()
        rasterizers.append(rasterizer)
        return rasterizer

    monkeypatch.setattr(
        providers_module,
        "_load_ludus_scene_bundle",
        fake_load_scene_bundle,
    )
    monkeypatch.setattr(
        providers_module,
        "_new_ludus_rasterizer",
        fake_new_rasterizer,
    )
    return scene, rasterizers


class _RecordingOutputSink:
    produces_artifacts = True

    def __init__(self) -> None:
        self.session_info: SessionInfo | None = None
        self.results: list[StepResult] = []
        self.closed = False

    def open(self, session_info: SessionInfo) -> None:
        self.session_info = session_info

    def begin_generation(self, generation: int) -> None:
        del generation

    def write(self, result: StepResult) -> OutputDecision:
        self.results.append(result)
        return OutputDecision()

    def close(self) -> Sequence[OutputArtifact]:
        self.closed = True
        return (OutputArtifact(kind="video/mp4", uri="memory://omnidreams"),)


class _FactoryRuntime:
    def start_session(self, inputs: InferenceInput) -> Any:
        del inputs
        raise NotImplementedError

    def close(self) -> None:
        return None


class _FakeOmnidreamsPipeline:
    def __init__(self) -> None:
        self.initialize_cache_calls: list[dict[str, Any]] = []
        self.generated_hdmaps: list[torch.Tensor] = []
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
        del cache
        self.generated_hdmaps.append(hdmap.detach().clone())
        return torch.full((1, 1, 1, 3, 2, 2), float(autoregressive_index))

    def finalize(self, *, autoregressive_index: int, cache: object) -> dict[str, float]:
        del autoregressive_index, cache
        return {"denoise_s": 0.25}


class _VariableFrameOmnidreamsPipeline(_FakeOmnidreamsPipeline):
    def __init__(self, frame_counts: tuple[int, ...]) -> None:
        super().__init__()
        self._frame_counts = frame_counts

    def get_num_frames(self, autoregressive_index: int) -> int:
        return self._frame_counts[
            min(autoregressive_index, len(self._frame_counts) - 1)
        ]

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: object,
        hdmap: torch.Tensor,
    ) -> torch.Tensor:
        del cache
        self.generated_hdmaps.append(hdmap.detach().clone())
        frame_count = self.get_num_frames(autoregressive_index)
        return torch.full((1, 1, frame_count, 3, 2, 2), float(autoregressive_index))


class _FailsIfEncodersReleasedOmnidreamsPipeline(_FakeOmnidreamsPipeline):
    def initialize_cache(
        self,
        *,
        text: list[list[str]],
        image: torch.Tensor,
        view_names: list[str],
    ) -> object:
        if self.released_encoders:
            raise AssertionError("encoders were released before the next session")
        return super().initialize_cache(
            text=text,
            image=image,
            view_names=view_names,
        )


class _FakeLudusRasterizer:
    def __init__(self) -> None:
        self.loaded_scene: object | None = None
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def load_scene(self, scene: object) -> None:
        self.loaded_scene = scene

    def render_chunk(
        self,
        *,
        rig_poses_world: np.ndarray,
        timestamps_us: np.ndarray,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "rig_poses_world": np.array(rig_poses_world, copy=True),
                "timestamps_us": tuple(int(t) for t in timestamps_us),
            }
        )
        frames = []
        for timestamp_us in timestamps_us:
            value = int(timestamp_us % 251)
            frames.append(
                SimpleNamespace(
                    rgb_host_uint8=np.full((2, 2, 3), value, dtype=np.uint8)
                )
            )
        return SimpleNamespace(frames=tuple(frames))

    def cleanup(self) -> None:
        self.closed = True


class _FakeWebRTCResampler:
    def __init__(self, *, start_v: float, fps: int) -> None:
        self.dt = 1.0 / fps
        self.next_chunk_start_v = start_v

    def reset(self, *, start_v: float) -> None:
        self.next_chunk_start_v = start_v

    def sample_chunk(self, num_frames: int) -> list[float]:
        start = self.next_chunk_start_v
        frame_times = [start + (index + 1) * self.dt for index in range(num_frames)]
        end = frame_times[-1]
        self.next_chunk_start_v = end
        return frame_times


class _FakeWebRTCVideoTrack:
    def __init__(self, *, fps: int) -> None:
        self.fps = fps
        self.closed = False
        self.enqueued: list[StepResult] = []

    async def enqueue_result(self, result: StepResult) -> int:
        self.enqueued.append(result)
        return result.frame_count

    def qsize(self) -> int:
        return 0

    async def close(self) -> None:
        self.closed = True


class _FakeWebRTCVideoEncoder:
    backend = "fake"
    prefers_codec: str | None = None

    def prepare_chunk_payload(self, result: StepResult, track: Any) -> StepResult:
        del track
        return result

    async def deliver_prepared_chunk(
        self,
        payload: object,
        track: Any,
        *,
        force_keyframe: bool = False,
    ) -> SimpleNamespace:
        del force_keyframe
        if not isinstance(payload, StepResult):
            raise TypeError("Fake WebRTC encoder expected a StepResult payload.")
        return SimpleNamespace(
            num_frames=await track.enqueue_result(payload),
            encode_ms=0.0,
        )


class _FakeWebRTCPeerConnection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeWebRTCChannel:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str) -> None:
        self.messages.append(message)


async def _wait_for_chunk_done(channel: _FakeWebRTCChannel) -> dict[str, Any]:
    for _ in range(100):
        chunk_done = [
            json.loads(message)
            for message in channel.messages
            if json.loads(message).get("type") == "chunk_done"
        ]
        if chunk_done:
            return chunk_done[0]
        await asyncio.sleep(0.01)
    pytest.fail("Timed out waiting for WebRTC chunk_done.")


class _FakeWebRTCRuntime:
    def __init__(self, config: Any) -> None:
        self.config = config

    async def initialize(self) -> None:
        return None

    async def reset_for_new_session(self, *args: Any, **kwargs: Any) -> None:
        return None

    def peek_input_fps(self) -> float:
        return 30.0

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
