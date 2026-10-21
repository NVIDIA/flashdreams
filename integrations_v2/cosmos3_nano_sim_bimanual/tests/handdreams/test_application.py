# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the Cosmos HandDreams adapter, model options, and scene binding."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tomllib
from cosmos3_nano_sim_bimanual.apps.handdreams import adapter as app_module
from cosmos3_nano_sim_bimanual.apps.handdreams.adapter import (
    CosmosHandDreamsV2Application as HandDreamsV2Application,
)
from cosmos3_nano_sim_bimanual.apps.handdreams.config import (
    DEFAULT_HAND_PIPELINE_CONFIG,
)
from cosmos3_nano_sim_bimanual.config import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
)
from cosmos3_nano_sim_bimanual.decoder import HandDecoderConfig
from cosmos3_nano_sim_bimanual.pipeline import (
    validate_pipeline_config,
)
from cosmos3_nano_sim_bimanual.transformer import HandTransformerConfig
from flashdreams_controller_input.isaac_teleop_source import DEFAULT_ISAAC_TELEOP_PYTHON
from handdreams import app as shared_app_module
from handdreams.config import (
    HanddreamsScenario,
)
from PIL import Image

from apps.handdreams.tests.helpers import (  # ty: ignore[unresolved-import]  # pytest exposes these shared helpers via the repository root.
    _FakeInferencePipeline,
    _FakePipeline,
    _FakePostprocessStream,
    _ScalePostprocessConfig,
    _scene,
)
from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig
from flashdreams.infra.postprocess import (
    VideoSpec,
)
from flashdreams.runtime_v2 import cli as runtime_cli
from flashdreams.runtime_v2 import client_window_factory
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
)

pytestmark = pytest.mark.ci_cpu


def _application(**kwargs):
    kwargs.setdefault("inference_pipeline_factory", lambda _: _FakeInferencePipeline())
    kwargs.setdefault("pipeline_factory", lambda **_: _FakePipeline())
    kwargs.setdefault("scene_factory", lambda scenario, _: _scene(scenario.sample))
    return HandDreamsV2Application(**kwargs)


def _pipeline_config(*, chunk_size: int = 1, **transformer_options: Any):
    return derive_config(
        DEFAULT_HAND_PIPELINE_CONFIG,
        decoder={"device": "cpu"},
        diffusion_model={
            "transformer": {
                "device": "cpu",
                "chunk_size": chunk_size,
                "overlap_cache_update": False,
                "compile_denoise": False,
                "valid_length_cache_attention": False,
                "cuda_graph_denoise": False,
                **transformer_options,
            }
        },
    )


def _scene_directory(root: Path, name: str, *, shade: int = 32) -> Path:
    directory = root / name
    directory.mkdir()
    metadata = {
        "schema_version": 1,
        "title": f"External {name}",
        "prompt": "Move the object",
        "fps": 30.0,
        "image_size": [9, 16, 9, 16],
        "wrist_poses": {side: np.eye(4).tolist() for side in ("left", "right")},
        "gripper_open": {"left": 0.5, "right": 0.5},
    }
    (directory / "scene.json").write_text(json.dumps(metadata), encoding="utf-8")
    Image.new("RGB", (16, 9), (shade, shade, shade)).save(directory / "first_frame.jpg")
    return directory


@pytest.fixture
def scene_arguments(tmp_path: Path) -> list[str]:
    source = _scene_directory(tmp_path, "synthetic-scene")
    return ["--scene-path", str(source)]


@pytest.fixture
def controller_modes(monkeypatch: pytest.MonkeyPatch):
    # Load this checkout's declared plugins, not a developer's installed extras.
    metadata_path = (
        Path(shared_app_module.__file__).parents[1]
        / "controller_input"
        / "pyproject.toml"
    )
    metadata = tomllib.loads(metadata_path.read_text(encoding="utf-8"))
    group = client_window_factory.CLIENT_WINDOW_ENTRY_POINT_GROUP
    entries = tuple(
        EntryPoint(name=name, value=value, group=group)
        for name, value in metadata["project"]["entry-points"][group].items()
    )
    monkeypatch.setattr(client_window_factory, "entry_points", lambda **_: entries)
    monkeypatch.setattr(
        runtime_cli, "registered_application_slugs", lambda: ("handdreams",)
    )
    return client_window_factory.client_window_modes()


def test_default_scene_resolution_does_not_download(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "default_scene_directory", lambda: tmp_path)
    monkeypatch.setattr(
        app_module, "prepare_default_scenes", lambda: pytest.fail("lazy download")
    )
    with pytest.raises(ValueError, match="no scene.json"):
        app_module._resolve_scene_source(HanddreamsScenario())


@pytest.mark.parametrize("name", ["idx_scene_b", "fold,shirt", " fold shirt "])
def test_scene_source_selects_one_scene_from_an_explicit_directory(
    tmp_path: Path,
    name: str,
) -> None:
    _scene_directory(tmp_path, "idx_scene_a")
    selected = _scene_directory(tmp_path, name)
    assert app_module._resolve_scene_source(
        HanddreamsScenario(sample=name, sample_dir=tmp_path)
    ) == (name, str(selected.resolve()))


def test_scene_source_preserves_an_explicit_directory_and_logical_name(
    tmp_path: Path,
) -> None:
    selected = _scene_directory(tmp_path, "arbitrary_directory")
    assert app_module._resolve_scene_source(
        HanddreamsScenario(sample="logical-scene-name", scene_path=selected)
    ) == ("logical-scene-name", str(selected.resolve()))


def test_default_scene_fetch_failure_precedes_model_and_teleop(monkeypatch) -> None:
    def fail_fetch():
        raise RuntimeError("Hugging Face access required")

    monkeypatch.setattr(app_module, "prepare_default_scenes", fail_fetch)
    app = _application(
        inference_pipeline_factory=lambda _: pytest.fail("unknown scene loaded model"),
        pipeline_factory=lambda **_: pytest.fail("unknown scene started Teleop"),
    )
    with pytest.raises(RuntimeError, match="Hugging Face access required"):
        app.init([])


def test_default_scenes_prepare_only_during_initialization(tmp_path, monkeypatch):
    selected = _scene_directory(tmp_path, "synthetic-scene")
    prepared = []
    monkeypatch.setattr(app_module, "default_scene_directory", lambda: tmp_path)
    monkeypatch.setattr(
        app_module, "prepare_default_scenes", lambda: prepared.append(True) or tmp_path
    )
    app = _application()
    app.init([])
    assert prepared == [True]
    assert app.scene_catalog == ("synthetic-scene",)
    assert app_module._resolve_scene_source(app.launch_config.scenario) == (
        "synthetic-scene",
        str(selected),
    )
    app_module._scene_previews(app.launch_config.scenario, app.scene_catalog)
    assert prepared == [True]
    app.close()


@pytest.mark.parametrize(
    "sample", ["idx_scene", "custom-scene", "fold,shirt", " fold shirt "]
)
def test_application_accepts_explicit_directory_scene_names(
    tmp_path: Path, sample: str, monkeypatch
) -> None:
    monkeypatch.setattr(
        app_module,
        "prepare_default_scenes",
        lambda: pytest.fail("external source fetched defaults"),
    )
    _scene_directory(tmp_path, sample)
    app = _application()
    app.init(["--sample", sample, "--sample-dir", str(tmp_path)])
    assert sample in app.scene_catalog
    app.close()


@pytest.mark.parametrize("source_kind", ["catalog", "single"])
@pytest.mark.parametrize("name", ["arbitrary-scene", "fold,shirt", " fold shirt "])
def test_external_scene_previews_use_selected_assets(
    tmp_path: Path, source_kind: str, name: str
) -> None:
    selected = _scene_directory(tmp_path, name)
    expected = {name: 32}
    if source_kind == "catalog":
        _scene_directory(tmp_path, "other-scene", shade=96)
        expected["other-scene"] = 96
        arguments = ["--sample-dir", str(tmp_path)]
    else:
        arguments = ["--scene-path", str(selected)]

    app = _application()
    try:
        app.init(arguments)
        assert (
            tuple(preview.name for preview in app._scene_previews) == app.scene_catalog
        )
        assert set(app.scene_catalog) == set(expected)
        for preview in app._scene_previews:
            assert preview.title == f"External {preview.name}"
            assert preview.thumbnail is not None
            assert preview.thumbnail.shape == (352, 640, 3)
            assert not preview.thumbnail.flags.writeable
            np.testing.assert_array_equal(preview.thumbnail, expected[preview.name])
        assert app_module._resolve_scene_source(app.launch_config.scenario) == (
            name,
            str(selected.resolve()),
        )
    finally:
        app.close()


def test_application_rejects_absent_directory_scene_before_starting(
    tmp_path: Path,
) -> None:
    _scene_directory(tmp_path, "idx_scene")
    app = _application(
        inference_pipeline_factory=lambda _: pytest.fail(
            "invalid scene loaded the model"
        ),
        pipeline_factory=lambda **_: pytest.fail("invalid scene started Teleop"),
    )
    with pytest.raises(ValueError, match="absent.*idx_missing"):
        app.init(["--sample", "idx_missing", "--sample-dir", str(tmp_path)])


def test_application_rejects_conflicting_scene_sources(tmp_path: Path) -> None:
    source = _scene_directory(tmp_path, "idx_scene")
    app = _application(
        inference_pipeline_factory=lambda _: pytest.fail(
            "invalid scene loaded the model"
        ),
        pipeline_factory=lambda **_: pytest.fail("invalid scene started Teleop"),
    )
    with pytest.raises(SystemExit) as error:
        app.init(["--scene-path", str(source), "--sample-dir", str(tmp_path)])
    assert error.value.code == 2


@pytest.mark.parametrize("prewarm_all", [False, True])
def test_application_defaults_live_and_reuses_app_owned_resources(
    tmp_path: Path,
    prewarm_all: bool,
) -> None:
    _scene_directory(tmp_path, "idx_scene_a")
    _scene_directory(tmp_path, "idx_scene_b")
    inference_pipeline = _FakeInferencePipeline()
    controller_pipeline = _FakePipeline()
    resolved_configs = []
    selected: list[str] = []
    scene_models: list[Any] = []
    app = _application(
        inference_pipeline_factory=lambda config: (
            resolved_configs.append(config) or inference_pipeline
        ),
        pipeline_factory=lambda **_kwargs: controller_pipeline,
        scene_factory=lambda scenario, pipeline: (
            selected.append(scenario.sample)
            or scene_models.append(pipeline)
            or _scene(scenario.sample)
        ),
    )
    caller_metadata = {"sample": "idx_scene_b", "caller": "untouched"}
    parser = argparse.ArgumentParser()
    runtime_cli._add_session_arguments(parser)
    requested = replace(
        runtime_cli._session_desc(
            app,
            parser.parse_args(
                [
                    "--fps",
                    "24",
                    "--presentation-mode",
                    "on_demand",
                    "--backpressure-mode",
                    "drop_oldest",
                ]
            ),
        ),
        frames_per_second_for_ui=72,
        metadata=caller_metadata,
    )
    app.init(
        [
            "--device",
            "cpu",
            "--control-fps",
            "15",
            "--sample",
            "idx_scene_a",
            "--sample-dir",
            str(tmp_path),
            "--compile-denoise",
            "--compile-prewarm-steps",
            "2",
            *(["--prewarm-all-scenes"] if prewarm_all else []),
        ]
    )
    assert app.launch_config.duration_seconds is None
    assert app.launch_config.control_fps == 15
    assert selected == ["idx_scene_a", "idx_scene_b"]
    assert inference_pipeline.prepared_scenes[0].name == "idx_scene_a"
    first = app.create_session(requested)
    second = app.create_session(replace(requested, metadata={"sample": "idx_scene_a"}))
    assert selected == ["idx_scene_a", "idx_scene_b"]
    assert [scene.name for scene in inference_pipeline.prepared_scenes] == selected
    assert caller_metadata == {"sample": "idx_scene_b", "caller": "untouched"}
    assert first.session_desc.metadata is not caller_metadata
    assert first.session_desc.metadata["scene_catalog"] == (
        "idx_scene_a",
        "idx_scene_b",
    )
    assert second.session_desc.metadata["sample"] == "idx_scene_a"
    assert first.session_desc.frames_per_second_for_ui == 72
    assert first.session_desc.frames_per_second_for_step == 24
    assert first.session_desc.presentation_mode is PresentationMode.ON_DEMAND
    assert first.session_desc.backpressure_mode is BackpressureMode.DROP_OLDEST
    assert controller_pipeline.starts == 1
    assert len(resolved_configs) == 1
    assert resolved_configs[0].diffusion_model.transformer.device == "cpu"
    assert resolved_configs[0].diffusion_model.transformer.model_fps == 15
    assert resolved_configs[0].decoder.device == "cpu"
    assert resolved_configs[0].compile_prewarm_steps == 2
    assert scene_models == [inference_pipeline, inference_pipeline]
    app.close()
    app.close()
    assert controller_pipeline.closes == 1
    assert inference_pipeline.close_count == 1


def test_application_prepares_postprocess_with_model_and_reports_output_size(
    monkeypatch: pytest.MonkeyPatch,
    scene_arguments: list[str],
) -> None:
    events: list[str] = []
    processor = _ScalePostprocessConfig(device="cpu")
    _FakePostprocessStream.created.clear()

    class _RecordingPostprocessStream(_FakePostprocessStream):
        def prepare(self, spec: VideoSpec) -> None:
            events.append("postprocess")
            super().prepare(spec)

    monkeypatch.setattr(
        shared_app_module,
        "discover_postprocess_presets",
        lambda: {"test-2x": processor},
    )
    monkeypatch.setattr(
        "flashdreams.plugins.registry.resolve_postprocess_preset",
        lambda name: processor,
    )
    monkeypatch.setattr(
        shared_app_module,
        "create_video_postprocess_stream",
        lambda **kwargs: _RecordingPostprocessStream(**kwargs),
    )
    app = _application(
        inference_pipeline_factory=lambda _config: (
            events.append("model") or _FakeInferencePipeline()
        ),
        scene_factory=lambda scenario, _pipeline: (
            events.append("scene") or _scene(scenario.sample)
        ),
    )
    app.init(
        [
            "--device",
            "cpu",
            "--postprocess-preset",
            "test-2x",
            "--postprocess-device",
            "cpu",
            *scene_arguments,
        ]
    )

    session = app.create_session(app.session_desc())

    assert events == ["model", "scene", "postprocess"]
    assert len(_FakePostprocessStream.created) == 1
    stream = _FakePostprocessStream.created[0]
    assert stream.prepared == [VideoSpec(width=640, height=352, fps=36)]
    assert session.session_desc.video_width == 1280
    assert session.session_desc.video_height == 920
    session.init()
    assert session.ui_loop.state.stage_width == 1280
    assert session.ui_loop.state.stage_height == 704
    assert session.ui_loop.state.show_postprocess_toggle
    assert session.ui_loop.state.postprocess_enabled
    assert session.model_loop.state.postprocess_enabled
    assert session._shutdown_registered_loops() == []
    session.close()
    assert stream.finished == 1
    app.close()


def test_default_launch_uses_packaged_teleop_and_demo_options(scene_arguments) -> None:
    app = _application()
    app.init(scene_arguments)
    desc = app.session_desc()
    assert (desc.video_width, desc.video_height) == (
        DEFAULT_VIDEO_WIDTH,
        DEFAULT_VIDEO_HEIGHT,
    )
    assert app.default_client_window_mode() == "webrtc"
    assert app.launch_config.control_fps is None
    assert desc.frames_per_second_for_ui == 36
    assert app.launch_config.isaac_teleop_python == Path(DEFAULT_ISAAC_TELEOP_PYTHON)
    assert app.launch_config.pipeline_config == replace(
        DEFAULT_HAND_PIPELINE_CONFIG, compile_prewarm_steps=1
    )
    pipeline = app.launch_config.pipeline_config
    transformer = pipeline.diffusion_model.transformer
    scheduler = pipeline.diffusion_model.scheduler
    assert isinstance(transformer, HandTransformerConfig)
    assert transformer.model_fps is None
    assert isinstance(scheduler, FlowMatchSchedulerConfig)
    assert isinstance(pipeline.decoder, HandDecoderConfig)
    assert (
        transformer.checkpoint,
        pipeline.decoder.backend,
        transformer.chunk_size,
        scheduler.num_inference_steps,
        transformer.window_frames,
        transformer.cache_mode,
    ) == (
        "cosmos3_nano_sim_bimanual_ga",
        "taehv2super",
        2,
        4,
        64,
        "block",
    )
    assert transformer.cache_refresh_mode == "chunk"
    assert transformer.overlap_cache_update
    assert transformer.compile_denoise
    assert transformer.compile_denoise_mode == "default"
    assert transformer.valid_length_cache_attention
    assert transformer.cuda_graph_denoise
    assert transformer.cuda_graph_warmup_iters == 2
    assert app.scene_catalog == ("synthetic-scene",)
    assert app.launch_config.scenario.sample == "synthetic-scene"
    app.close()


@pytest.mark.parametrize("mode", [None, "webrtc", "native-window"])
def test_standard_cli_selects_controller_plugin_before_initializing(
    monkeypatch: pytest.MonkeyPatch,
    controller_modes,
    scene_arguments: list[str],
    mode: str | None,
) -> None:
    app = _application()
    output_mode = mode or "webrtc"
    selected = client_window_factory.client_window_mode(
        f"quest-remote-{output_mode}", modes=controller_modes
    )
    captured = []

    def create_window(args):
        assert app._pipeline is None
        captured.append(args)
        return None  # No sockets, subprocesses, or OS window in this CPU test.

    def run(_runner, session_desc, commandline_args, **_kwargs):
        assert session_desc.frames_per_second_for_step == 24
        assert list(commandline_args) == ["--device", "cpu", *scene_arguments]
        app.init(commandline_args)
        app.close()

    monkeypatch.delenv("FLASHDREAMS_CONTROLLER_TOKEN", raising=False)
    monkeypatch.delenv("TELEOP_STREAM_SERVER_IP", raising=False)
    monkeypatch.setattr(runtime_cli, "create_application", lambda _slug: app)
    monkeypatch.setattr(selected, "create", create_window)
    monkeypatch.setattr(runtime_cli.ApplicationRunner, "run", run)
    monkeypatch.setattr(runtime_cli, "shutdown_distributed", lambda **_: None)
    runtime_cli.entrypoint(
        [
            "handdreams",
            *([] if mode is None else ["--mode", mode]),
            "--host",
            "0.0.0.0",
            "--port",
            "8081",
            "--window-title",
            "HandDreams test",
            "--fps",
            "24",
            "--isaac-teleop-accept-eula",
            "--",
            "--device",
            "cpu",
            *scene_arguments,
        ]
    )

    assert len(captured) == 1
    args = captured[0]
    assert args.mode == selected.name
    assert hasattr(selected, "output_mode")
    assert selected.output_mode == output_mode
    assert (args.host, args.port, args.window_title) == (
        "0.0.0.0",
        8081,
        "HandDreams test",
    )
    assert args.controller_host == "127.0.0.1"
    assert args.controller_cert is None and args.controller_key is None
    assert args.quest_device is None
    assert args.quest_require_physical_wake
    assert args.isaac_teleop_accept_eula


def test_standard_cli_application_help_does_not_prepare_controllers(
    monkeypatch: pytest.MonkeyPatch,
    controller_modes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = _application(
        inference_pipeline_factory=lambda _: pytest.fail("help loaded a model"),
        pipeline_factory=lambda **_: pytest.fail("help started controllers"),
    )
    monkeypatch.setattr(runtime_cli, "create_application", lambda _slug: app)
    monkeypatch.setattr(
        app, "resolve_client_window_mode", lambda _: pytest.fail("help resolved input")
    )
    for mode in controller_modes:
        monkeypatch.setattr(
            mode, "check_arguments", lambda _: pytest.fail("help checked a window")
        )
        monkeypatch.setattr(
            mode, "create", lambda _: pytest.fail("help opened a window")
        )
    with pytest.raises(SystemExit) as error:
        runtime_cli.entrypoint(["handdreams", "--", "--help"])
    assert error.value.code == 0
    assert "--sample-dir" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["mp4", "null", "isaac-teleop-webrtc"])
def test_noninteractive_or_explicit_input_mode_is_unchanged(mode: str) -> None:
    assert _application().resolve_client_window_mode(mode) == mode


def test_optimized_defaults_can_be_disabled(scene_arguments: list[str]) -> None:
    app = _application()
    app.init(
        [
            "--no-compile-denoise",
            "--no-valid-length-cache-attention",
            "--no-cuda-graph-denoise",
            "--no-overlap-cache-update",
            *scene_arguments,
        ]
    )
    pipeline = app.launch_config.pipeline_config
    transformer = pipeline.diffusion_model.transformer
    assert not transformer.compile_denoise
    assert not transformer.valid_length_cache_attention
    assert not transformer.cuda_graph_denoise
    assert not transformer.overlap_cache_update
    assert pipeline.compile_prewarm_steps == 0
    app.close()


def test_model_configuration_preserves_control_cadence() -> None:
    parser = argparse.ArgumentParser()
    app_module._add_model_arguments(parser)
    arguments = vars(parser.parse_args(["--device", "cpu"]))
    arguments["control_fps"] = 60
    config = app_module._resolve_pipeline_config(arguments)
    assert isinstance(config.diffusion_model.transformer, HandTransformerConfig)
    assert config.diffusion_model.transformer.model_fps == 60


@pytest.mark.parametrize("device", ["cpu", "cpu:0", "cuda", "cuda:1"])
def test_implicit_optimizations_follow_selected_device(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    parser = argparse.ArgumentParser()
    app_module._add_model_arguments(parser)
    arguments = vars(parser.parse_args(["--device", device]))
    original_arguments = arguments.copy()
    original_preset = deepcopy(DEFAULT_HAND_PIPELINE_CONFIG)
    monkeypatch.setattr(
        "torch.cuda.is_available",
        lambda: pytest.fail("config resolution queried CUDA availability"),
    )

    config = app_module._resolve_pipeline_config(arguments)
    transformer = config.diffusion_model.transformer
    optimized = device.startswith("cuda")
    for name in (
        "compile_denoise",
        "overlap_cache_update",
        "valid_length_cache_attention",
        "cuda_graph_denoise",
    ):
        assert getattr(transformer, name) is optimized
    assert isinstance(transformer, HandTransformerConfig)
    assert isinstance(config.decoder, HandDecoderConfig)
    assert transformer.device == config.decoder.device == device
    assert config.compile_prewarm_steps == (1 if optimized else 0)
    if not optimized:
        validate_pipeline_config(config, check_runtime=True)
    assert arguments == original_arguments
    assert DEFAULT_HAND_PIPELINE_CONFIG == original_preset


def test_cpu_application_reaches_real_runtime_validation(
    scene_arguments: list[str],
) -> None:
    received = []

    def create_pipeline(config):
        validate_pipeline_config(config, check_runtime=True)
        received.append(config)
        return _FakeInferencePipeline()

    app = _application(inference_pipeline_factory=create_pipeline)
    try:
        app.init(["--device", "cpu", *scene_arguments])
        assert len(received) == 1
        assert received[0].compile_prewarm_steps == 0
    finally:
        app.close()


@pytest.mark.parametrize("option", ["compile-denoise", "overlap-cache-update"])
def test_cpu_explicit_optimization_is_preserved(option: str) -> None:
    parser = argparse.ArgumentParser()
    app_module._add_model_arguments(parser)
    config = app_module._resolve_pipeline_config(
        vars(parser.parse_args(["--device", "cpu", f"--{option}"]))
    )
    transformer = config.diffusion_model.transformer
    assert getattr(transformer, option.replace("-", "_"))
    assert isinstance(transformer, HandTransformerConfig)
    assert not transformer.valid_length_cache_attention
    assert not transformer.cuda_graph_denoise
    assert config.compile_prewarm_steps == (33 if option == "compile-denoise" else 0)
    validate_pipeline_config(config, check_runtime=True)


@pytest.mark.parametrize(
    "option", ["valid-length-cache-attention", "cuda-graph-denoise"]
)
def test_cpu_explicit_cuda_option_still_fails_runtime_validation(
    monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    parser = argparse.ArgumentParser()
    app_module._add_model_arguments(parser)
    config = app_module._resolve_pipeline_config(
        vars(parser.parse_args(["--device", "cpu", "--compile-denoise", f"--{option}"]))
    )
    assert getattr(config.diffusion_model.transformer, option.replace("-", "_"))
    monkeypatch.setattr(
        "cosmos3_nano_sim_bimanual.transformer.config.varlen_attention_available",
        lambda: True,
    )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        validate_pipeline_config(config, check_runtime=True)


def test_application_forwards_component_configs_without_session_metrics(
    scene_arguments,
) -> None:
    expected = {
        "checkpoint": "local-checkpoint",
        "credential_path": "local-credential",
        "device": "cpu",
        "seed": 17,
        "decoder": "wan",
        "chunk_size": 4,
        "denoise_steps": 4,
        "window_frames": 32,
        "native_gqa_attention": True,
        "sdpa_backend": "math",
        "cache_mode": "block",
        "cache_refresh_mode": "framewise",
        "overlap_cache_update": True,
        "compile_denoise": True,
        "compile_denoise_mode": "default",
        "fixed_shape_cache_fill": True,
        "valid_length_cache_attention": False,
        "cuda_graph_denoise": True,
        "cuda_graph_warmup_iters": 3,
        "profile": True,
    }
    arguments = ["--synchronize-metrics", *scene_arguments]
    for name, value in expected.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            arguments.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            arguments.extend((flag, str(value)))
    app = _application()
    app.init(arguments)
    pipeline = app.launch_config.pipeline_config
    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, HandTransformerConfig)
    assert isinstance(pipeline.decoder, HandDecoderConfig)
    assert isinstance(pipeline.diffusion_model.scheduler, FlowMatchSchedulerConfig)
    actual = {
        "checkpoint": transformer.checkpoint,
        "credential_path": transformer.credential_path,
        "device": transformer.device,
        "seed": pipeline.diffusion_model.seed,
        "decoder": pipeline.decoder.backend,
        "chunk_size": transformer.chunk_size,
        "denoise_steps": pipeline.diffusion_model.scheduler.num_inference_steps,
        "window_frames": transformer.window_frames,
        "native_gqa_attention": transformer.native_gqa_attention,
        "sdpa_backend": transformer.sdpa_backend,
        "cache_mode": transformer.cache_mode,
        "cache_refresh_mode": transformer.cache_refresh_mode,
        "overlap_cache_update": transformer.overlap_cache_update,
        "compile_denoise": transformer.compile_denoise,
        "compile_denoise_mode": transformer.compile_denoise_mode,
        "fixed_shape_cache_fill": transformer.fixed_shape_cache_fill,
        "valid_length_cache_attention": transformer.valid_length_cache_attention,
        "cuda_graph_denoise": transformer.cuda_graph_denoise,
        "cuda_graph_warmup_iters": transformer.cuda_graph_warmup_iters,
        "profile": transformer.profile,
    }
    assert actual == expected
    assert pipeline.decoder.device == expected["device"]
    assert pipeline.decoder.credential_path == expected["credential_path"]
    assert app.launch_config.synchronize_metrics
    app.close()


def test_application_rejects_pre_ga_two_step_override(scene_arguments, capsys) -> None:
    with pytest.raises(SystemExit):
        _application().init(["--denoise-steps", "2", *scene_arguments])
    assert "choose from 4" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag",
    [
        "--duration-seconds",
        "--isaac-teleop-startup-timeout-s",
        "--position-gain",
        "--grip-threshold",
        "--stale-ms",
        "--gripper-slew-per-action",
        "--open-claw-threshold",
    ],
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_application_rejects_nonfinite_config_before_starting(
    flag, value, scene_arguments
) -> None:
    app = _application(
        inference_pipeline_factory=lambda _: pytest.fail(
            "invalid config loaded the model"
        ),
        pipeline_factory=lambda **_: pytest.fail("invalid config started Teleop"),
    )
    with pytest.raises(ValueError, match="finite"):
        app.init([f"{flag}={value}", *scene_arguments])


@pytest.mark.parametrize(
    "options",
    [
        {"chunk_size": 0},
        {"cuda_graph_warmup_iters": -1},
        {"cache_mode": "cat", "compile_denoise": True},
        {"cuda_graph_denoise": True},
    ],
)
def test_programmatic_pipeline_config_validation(options) -> None:
    with pytest.raises(ValueError):
        validate_pipeline_config(_pipeline_config(**options), check_runtime=False)


def test_mvp_imports_do_not_load_legacy_launchers_or_observer(
    tmp_path: Path, scene_arguments
) -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = (
            "internal",
            "cosmos_interactive_internal",
            "handdreams.shadow",
        )

        class RejectLegacyImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".")
                       for name in blocked):
                    raise AssertionError("MVP imported retired component: " + fullname)
                return None

        sys.meta_path.insert(0, RejectLegacyImport())
        from cosmos3_nano_sim_bimanual.apps.handdreams.adapter import CosmosHandDreamsV2Application
        from apps.handdreams.tests.helpers import _FakeInferencePipeline, _FakePipeline, _scene
        from flashdreams_controller_input.quest_remote_v2_window import (
            QUEST_REMOTE_WEBRTC_MODE,
            QUEST_REMOTE_NATIVE_WINDOW_MODE,
        )

        app = CosmosHandDreamsV2Application(
            inference_pipeline_factory=lambda _: _FakeInferencePipeline(),
            pipeline_factory=lambda **_: _FakePipeline(),
            scene_factory=lambda scenario, _: _scene(scenario.sample),
        )
        app.init(sys.argv[1:])
        assert app.default_client_window_mode() == "webrtc"
        assert app.resolve_client_window_mode("webrtc") == QUEST_REMOTE_WEBRTC_MODE.name
        assert app.resolve_client_window_mode("native-window") == QUEST_REMOTE_NATIVE_WINDOW_MODE.name
        app.close()
        """
    )
    # This subprocess runs outside the checkout but uses repository test helpers.
    script = (
        f"import sys\nsys.path.insert(0, {str(Path(__file__).resolve().parents[4])!r})\n"
        + script
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *scene_arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "argument",
    [
        "--live",
        "--auto-live",
        "--no-auto-live",
        "--ui",
        "--no-ui",
        "--full-length",
        "--xr-trace",
        "--isaac-shadow",
        "--ui-fps",
    ],
)
def test_application_rejects_removed_routes(argument: str) -> None:
    with pytest.raises(SystemExit) as error:
        _application().init([argument])
    assert error.value.code == 2


def test_inference_start_failure_releases_preflighted_controller_pipeline(
    scene_arguments,
) -> None:
    controller_pipeline = _FakePipeline()

    def fail_inference(_config):
        assert controller_pipeline.starts == 1
        raise RuntimeError("checkpoint unavailable")

    app = _application(
        inference_pipeline_factory=fail_inference,
        pipeline_factory=lambda **_kwargs: controller_pipeline,
    )
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        app.init(["--device", "cpu", *scene_arguments])
    app.close()
    app.close()
    assert controller_pipeline.closes == 1


def test_application_owns_pipeline_before_partial_start_failure(
    scene_arguments,
) -> None:
    inference_pipeline = _FakeInferencePipeline()
    inference_configs = []

    class _PartiallyStartedPipeline(_FakePipeline):
        def start(self) -> None:
            self.starts += 1
            raise RuntimeError("partial start failed")

    pipeline = _PartiallyStartedPipeline()
    app = _application(
        inference_pipeline_factory=lambda config: (
            inference_configs.append(config) or inference_pipeline
        ),
        pipeline_factory=lambda **_kwargs: pipeline,
        scene_factory=lambda scenario, _model: _scene(scenario.sample),
    )
    with pytest.raises(RuntimeError, match="partial start failed"):
        app.init(["--device", "cpu", *scene_arguments])
    assert app._pipeline is pipeline
    assert app._inference_pipeline is None
    app.close()
    assert pipeline.starts == 1
    assert pipeline.closes == 1
    assert inference_configs == []
    assert inference_pipeline.close_count == 0
