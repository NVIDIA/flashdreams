# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for finite GA action replay without controller services."""

import argparse
import time
from dataclasses import replace
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tomllib
import torch
from cosmos3_nano_sim_bimanual.apps.handdreams import replay
from cosmos3_nano_sim_bimanual.transformer import HandTransformerConfig
from flashdreams_controller_input.isaac_teleop_source import IsaacTeleopSource
from handdreams.teleop import IsaacTeleopControllerPipeline

from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.null_client_window import NullClientWindow
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("size", ["0", "-1"])
def test_replay_rejects_invalid_chunk_before_dataset_preparation(monkeypatch, size):
    monkeypatch.setattr(
        replay, "prepare_dataset_replays", lambda: pytest.fail("must not fetch")
    )
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        replay.create_app().init(["--chunk-size", size])


def test_replay_model_config_keeps_scene_conditioning_rate() -> None:
    parser = argparse.ArgumentParser()
    replay._add_model_arguments(parser)
    config = replay._resolve_pipeline_config(
        vars(parser.parse_args(["--device", "cpu"]))
    )
    transformer = config.diffusion_model.transformer
    assert isinstance(transformer, HandTransformerConfig)
    assert transformer.model_fps is None


def test_replay_entry_point_loads_the_controller_free_application() -> None:
    project = Path(replay.__file__).parents[2] / "pyproject.toml"
    metadata = tomllib.loads(project.read_text(encoding="utf-8"))
    group = "flashdreams.applications_v2"
    entries = metadata["project"]["entry-points"][group]
    entry = EntryPoint(
        name="handdreams-replay", value=entries["handdreams-replay"], group=group
    )
    assert entry.load() is replay.create_app
    assert entries["handdreams"] == (
        "cosmos3_nano_sim_bimanual.apps.handdreams.adapter:create_app"
    )


def _actions(rows: int = 8) -> np.ndarray:
    actions = np.arange(rows * 59, dtype=np.float32).reshape(rows, 59) / 100
    actions[:, 18:33] = 0
    actions[:, 43:58] = 0
    return actions


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_load_recorded_actions_preserves_values_and_pads_on_cpu(
    tmp_path: Path, dtype: Any
) -> None:
    source = _actions().astype(dtype)
    path = tmp_path / "actions.npy"
    np.save(path, source)

    result = replay.load_recorded_actions(path, actions_per_step=4)

    assert result.shape == (8, 64)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert result.is_contiguous()
    torch.testing.assert_close(result[:, :59], torch.from_numpy(source).float())
    assert torch.count_nonzero(result[:, 59:]) == 0


@pytest.mark.parametrize(
    "source",
    [
        np.zeros((8, 29), dtype=np.float32),
        np.zeros((8, 64), dtype=np.float32),
        np.zeros((1, 8, 59), dtype=np.float32),
        np.zeros((0, 59), dtype=np.float32),
        _actions(7),
        np.zeros((8, 59), dtype=np.int64),
        np.zeros((8, 59), dtype=object),
    ],
    ids=["pre-ga", "padded-ambiguous", "rank", "empty", "tail", "integer", "object"],
)
def test_load_recorded_actions_rejects_incompatible_arrays(
    tmp_path: Path, source: np.ndarray
) -> None:
    path = tmp_path / "actions.npy"
    np.save(path, source)
    with pytest.raises(ValueError):
        replay.load_recorded_actions(path, actions_per_step=4)


@pytest.mark.parametrize(
    ("column", "value"),
    [(0, np.nan), (0, np.inf), (18, 1.0), (32, 1.0), (43, 1.0), (57, 1.0)],
)
def test_load_recorded_actions_rejects_nonfinite_and_unmasked_values(
    tmp_path: Path, column: int, value: float
) -> None:
    source = _actions()
    source[0, column] = value
    path = tmp_path / "actions.npy"
    np.save(path, source)
    with pytest.raises(ValueError):
        replay.load_recorded_actions(path, actions_per_step=4)


def test_load_recorded_actions_rejects_npz_archives(tmp_path: Path) -> None:
    path = tmp_path / "actions.npz"
    np.savez(path, actions=_actions())
    with pytest.raises(ValueError):
        replay.load_recorded_actions(path, actions_per_step=4)


class _Pipeline:
    """Record replay calls without loading model weights or using CUDA."""

    actions_per_step = 4
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.prepared: list[Any] = []
        self.contexts: list[dict[str, Any]] = []
        self.caches: list[object] = []
        self.generated: list[
            tuple[int, object, torch.Tensor | replay.HandActionInput]
        ] = []
        self.finalized: list[tuple[int, object]] = []
        self.released: list[object] = []
        self.close_count = 0
        self.frames = torch.zeros(4, 3, 2, 2)

    def prepare_scene(self, scene: Any) -> None:
        self.prepared.append(scene)

    def initialize_cache(self, **contexts: Any) -> object:
        cache = object()
        self.contexts.append(contexts)
        self.caches.append(cache)
        return cache

    def generate(
        self,
        autoregressive_index: int,
        cache: object,
        input: torch.Tensor | replay.HandActionInput,
    ) -> torch.Tensor:
        copied = (
            replay.HandActionInput(input.actions.clone(), input.domain_ids.clone())
            if isinstance(input, replay.HandActionInput)
            else input.clone()
        )
        self.generated.append((autoregressive_index, cache, copied))
        return self.frames

    def finalize(self, autoregressive_index: int, cache: object) -> dict[str, int]:
        self.finalized.append((autoregressive_index, cache))
        return {"fake_step": autoregressive_index}

    def release_session(self, cache: object) -> None:
        self.released.append(cache)

    def close(self) -> None:
        self.close_count += 1


def _input_arguments(tmp_path: Path, source: np.ndarray) -> list[str]:
    actions_path = tmp_path / "actions.npy"
    np.save(actions_path, source)
    scene_path = tmp_path / "scene"
    scene_path.mkdir()
    (scene_path / "first_frame.jpg").touch()
    (scene_path / "scene.json").write_text("{}", encoding="utf-8")
    return [
        "--scene-path",
        str(scene_path),
        "--recorded-actions",
        str(actions_path),
        "--chunk-size",
        "1",
    ]


def _application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, initialize: bool = True
) -> tuple[Any, _Pipeline, SimpleNamespace, np.ndarray]:
    source = _actions()
    arguments = _input_arguments(tmp_path, source)
    scene = SimpleNamespace(
        name="scene", fps=24.0, content_height=360, content_width=640
    )
    pipeline = _Pipeline()
    monkeypatch.setattr(replay, "_resolve_pipeline_config", lambda _: object())
    monkeypatch.setattr(replay, "_setup_inference_pipeline", lambda _: pipeline)
    monkeypatch.setattr(replay, "_load_scene", lambda _scenario, _pipeline: scene)
    monkeypatch.setattr(
        replay, "prepare_dataset_replays", lambda: pytest.fail("dataset downloaded")
    )
    monkeypatch.setattr(
        IsaacTeleopControllerPipeline,
        "start",
        lambda *_args, **_kwargs: pytest.fail("replay started calibration"),
    )
    monkeypatch.setattr(
        IsaacTeleopSource,
        "start",
        lambda *_args, **_kwargs: pytest.fail("replay started CloudXR"),
    )
    app = replay.create_app()
    if initialize:
        app.init(arguments)
    return app, pipeline, scene, source


def test_application_uses_standard_windows_and_prepares_scene_without_controllers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uninitialized = replay.create_app()
    for mode in ("webrtc", "native-window", "null", "mp4"):
        assert uninitialized.resolve_client_window_mode(mode) == mode
    app, pipeline, scene, _ = _application(tmp_path, monkeypatch)
    try:
        for mode in ("webrtc", "native-window", "null", "mp4"):
            assert app.resolve_client_window_mode(mode) == mode
        assert pipeline.prepared == [scene]
        desc = app.session_desc()
        assert desc.video_width == 640
        assert desc.video_height == 352
        assert desc.frames_per_second_for_step == scene.fps
        assert desc.frames_per_second_for_ui == scene.fps
    finally:
        app.close()
        app.close()
    assert pipeline.close_count == 1


def test_replay_passes_exact_chunks_and_restarts_from_a_fresh_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, pipeline, scene, source = _application(tmp_path, monkeypatch)
    try:
        session = app.create_session(app.session_desc())
        session.init()
        loop = session.model_loop
        first_cache = pipeline.caches[0]
        assert pipeline.contexts == [
            {
                "transformer_context": {"scene": scene},
                "decoder_context": {"scene": scene},
            }
        ]
        assert not loop.is_finished()
        for index in range(2):
            (result,) = loop.step(index, UserInputEvents([]))
            actual_index, cache, actions = pipeline.generated[-1]
            assert isinstance(actions, torch.Tensor)
            assert actual_index == index
            assert cache is first_cache
            torch.testing.assert_close(
                actions[:, :59], torch.from_numpy(source[index * 4 : (index + 1) * 4])
            )
            assert torch.count_nonzero(actions[:, 59:]) == 0
            assert pipeline.finalized[-1] == (index, first_cache)
            assert result.step_index == index
            assert result.frame_count == 4
            torch.testing.assert_close(result.read_output(), pipeline.frames)
            assert result.metrics == {"fake_step": index}
            assert loop.is_finished() is (index == 1)

        assert loop.step(2, UserInputEvents([])) == []
        assert len(pipeline.generated) == 2
        loop.reset()
        assert pipeline.released == [first_cache]
        assert not loop.is_finished()
        loop.step(0, UserInputEvents([]))
        assert pipeline.generated[-1][1] is pipeline.caches[1]
        torch.testing.assert_close(pipeline.generated[-1][2], pipeline.generated[0][2])
        loop.close()
        loop.close()
        session.close()
        session.close()
        assert pipeline.released == pipeline.caches
    finally:
        app.close()
    assert pipeline.close_count == 1


def _dataset_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...] = ("fold-tshirt", "fold-shorts", "open-tissue-box"),
):
    source = replay.action_format.pad_action(_actions(10))
    domains = np.array([15, 15, 2, 2, 2, 15, 15, 2, 2, 15], dtype=np.int64)
    app, pipeline, _, _ = _application(tmp_path, monkeypatch, initialize=False)
    prepared = []
    scenes = {name: SimpleNamespace(name=name, fps=24.0) for name in names}
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / "first_frame.jpg").touch()
        (tmp_path / name / "scene.json").write_text("{}", encoding="utf-8")

    def prepare():
        prepared.append(True)
        return {name: (tmp_path / name, source, domains) for name in names}

    monkeypatch.setattr(replay, "prepare_dataset_replays", prepare)
    monkeypatch.setattr(
        replay, "_load_scene", lambda scenario, _: scenes[scenario.scene_path.name]
    )
    monkeypatch.setattr(
        replay,
        "scene_previews",
        lambda sources: tuple(replay.ScenePreview(name, name) for name, _ in sources),
    )
    return app, pipeline, scenes, source, domains, prepared


def _keys(*keys: str) -> UserInputEvents:
    return UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=np.uint64(index), key=key, state=KeyboardInputState.PRESSED
            )
            for index, key in enumerate(keys)
        ]
    )


def test_dataset_catalog_selects_replays_and_returns_to_picker_without_controllers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app, pipeline, scenes, source, domains, prepared = _dataset_application(
        tmp_path, monkeypatch
    )
    app.init(["--chunk-size", "1"])
    try:
        assert prepared == [True]
        assert pipeline.prepared == list(scenes.values())
        assert capsys.readouterr().out.count("dropping 2 trailing actions") == 3
        session = app.create_session(app.session_desc())
        session.init()
        assert isinstance(session.ui_loop, replay.HandDreamsUILoop)
        assert session.ui_loop.state.status.replay
        assert session.ui_loop.state.status.phase is replay.HandDreamsPhase.SELECTING
        assert session.model_loop.is_finished()
        assert session.model_loop.step(0, UserInputEvents([])) == []
        session.model_loop.reset()
        assert not pipeline.caches
        session.ui_loop.step(0, _keys("ArrowRight"))
        assert session.ui_loop.flush_ui_loop_requests() is None
        session.ui_loop.step(1, _keys("Enter"))
        request = session.ui_loop.flush_ui_loop_requests()
        assert request is not None and request.new_session is not None
        assert request.new_session.metadata["sample"] == "fold-tshirt"
        session.close()

        session = app.create_session(request.new_session)
        session.init()
        assert session.ui_loop.state.status.phase is replay.HandDreamsPhase.LIVE
        assert (
            pipeline.contexts[-1]["transformer_context"]["scene"]
            is scenes["fold-tshirt"]
        )
        loop = session.model_loop
        for index in range(2):
            loop.step(index, UserInputEvents([]))
            inputs = pipeline.generated[-1][2]
            assert isinstance(inputs, replay.HandActionInput)
            rows = slice(index * 4, (index + 1) * 4)
            torch.testing.assert_close(inputs.actions, torch.from_numpy(source[rows]))
            torch.testing.assert_close(
                inputs.domain_ids, torch.from_numpy(domains[rows])
            )
        assert loop.is_finished()
        assert not session.ui_loop.is_finished()
        assert loop.step(2, UserInputEvents([])) == []
        loop.reset()
        session.ui_loop.reset()
        assert session.ui_loop.state.status.phase is replay.HandDreamsPhase.LIVE
        assert session.ui_loop.state.status.replay
        loop.step(0, UserInputEvents([]))
        inputs = pipeline.generated[-1][2]
        assert isinstance(inputs, replay.HandActionInput)
        torch.testing.assert_close(inputs.domain_ids, torch.from_numpy(domains[:4]))
        session.ui_loop.step(2, _keys("ArrowRight", "Enter"))
        request = session.ui_loop.flush_ui_loop_requests()
        assert request is not None and request.new_session is not None
        assert request.new_session.metadata["sample"] == "fold-shorts"
        session.close()

        session = app.create_session(request.new_session)
        session.init()
        assert (
            pipeline.contexts[-1]["transformer_context"]["scene"]
            is scenes["fold-shorts"]
        )
        assert session.model_loop.state.row == 0
        session.ui_loop.step(0, _keys("Escape"))
        request = session.ui_loop.flush_ui_loop_requests()
        assert request is not None and request.new_session is not None
        assert request.new_session.metadata["select_scene"] is True
        session.close()

        count = len(pipeline.caches)
        session = app.create_session(request.new_session)
        session.init()
        assert session.ui_loop.state.status.phase is replay.HandDreamsPhase.SELECTING
        assert session.model_loop.is_finished()
        assert len(pipeline.caches) == count
        assert prepared == [True]
    finally:
        app.close()
    assert pipeline.released == pipeline.caches
    assert pipeline.close_count == 1
    assert not app._recordings


def test_single_replay_scene_keeps_picker_and_escape(tmp_path, monkeypatch):
    app, pipeline, _, _, _, _ = _dataset_application(
        tmp_path, monkeypatch, names=("fold-tshirt",)
    )
    app.init(["--chunk-size", "1"])
    try:
        session = app.create_session(app.session_desc())
        session.init()
        assert session.ui_loop.state.status.scene_catalog == ("fold-tshirt",)
        assert session.ui_loop.state.status.phase is replay.HandDreamsPhase.SELECTING
        assert not pipeline.caches
        session.ui_loop.step(0, _keys("ArrowRight", "Enter"))
        request = session.ui_loop.flush_ui_loop_requests()
        assert request.new_session.metadata["sample"] == "fold-tshirt"
        session.close()
        session = app.create_session(request.new_session)
        session.init()
        assert session.ui_loop.state.status.phase is replay.HandDreamsPhase.LIVE
        session.ui_loop.step(0, _keys("Escape"))
        request = session.ui_loop.flush_ui_loop_requests()
        assert request.new_session.metadata["select_scene"] is True
        session.close()
        session = app.create_session(request.new_session)
        session.init()
        assert session.ui_loop.state.status.phase is replay.HandDreamsPhase.SELECTING
    finally:
        app.close()


@pytest.mark.parametrize("sample", ["missing", "../fold-tshirt", None, [], 3])
def test_dataset_catalog_rejects_unknown_scene_metadata(
    tmp_path, monkeypatch, sample
) -> None:
    app, pipeline, _, _, _, _ = _dataset_application(tmp_path, monkeypatch)
    app.init(["--chunk-size", "1"])
    try:
        with pytest.raises(ValueError, match="unknown replay scene"):
            app.create_session(replace(app.session_desc(), metadata={"sample": sample}))
        assert not pipeline.caches
    finally:
        app.close()


@pytest.mark.parametrize("rows", [0, 3])
def test_dataset_replay_rejects_less_than_one_chunk_before_loading_model(
    monkeypatch: pytest.MonkeyPatch, rows: int
) -> None:
    monkeypatch.setattr(replay, "_resolve_pipeline_config", lambda _: object())
    monkeypatch.setattr(
        replay,
        "prepare_dataset_replays",
        lambda: {
            "fold-tshirt": (
                Path("unused"),
                np.zeros((rows, 64), dtype=np.float32),
                np.full(rows, 15, dtype=np.int64),
            )
        },
    )
    monkeypatch.setattr(
        replay, "_setup_inference_pipeline", lambda _: pytest.fail("model loaded")
    )
    app = replay.create_app()
    with pytest.raises(ValueError, match="at least 4 action rows"):
        app.init(["--chunk-size", "1"])
    app.close()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--scene-path", "scene"],
        ["--recorded-actions", "actions.npy"],
    ],
)
def test_replay_rejects_ambiguous_or_incomplete_source_arguments(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(
        replay, "prepare_dataset_replays", lambda: pytest.fail("dataset downloaded")
    )
    monkeypatch.setattr(
        replay, "_setup_inference_pipeline", lambda _: pytest.fail("model loaded")
    )
    with pytest.raises(SystemExit) as error:
        replay.create_app().init(arguments)
    assert error.value.code == 2


def test_application_rejects_concurrent_sessions_and_closes_active_resources_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, pipeline, _, _ = _application(tmp_path, monkeypatch)
    first = app.create_session(app.session_desc())
    first.init()
    with pytest.raises(RuntimeError):
        app.create_session(app.session_desc())
    first.close()
    second = app.create_session(app.session_desc())
    second.init()
    app.close()
    app.close()
    second.close()
    assert pipeline.released == pipeline.caches
    assert pipeline.close_count == 1


def test_fractional_scene_fps_rejects_replay_and_closes_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _input_arguments(tmp_path, _actions())
    pipeline = _Pipeline()
    scene = SimpleNamespace(fps=29.97)
    monkeypatch.setattr(replay, "_resolve_pipeline_config", lambda _: object())
    monkeypatch.setattr(replay, "_setup_inference_pipeline", lambda _: pipeline)
    monkeypatch.setattr(replay, "_load_scene", lambda _scenario, _pipeline: scene)
    app = replay.create_app()
    with pytest.raises(ValueError, match="whole-number scene FPS"):
        app.init(arguments)
    app.close()
    assert pipeline.close_count == 1
    assert not pipeline.prepared
    with pytest.raises(RuntimeError):
        app.create_session(app.session_desc())  # ty: ignore[invalid-argument-type]  # Closed apps have no session description.


@pytest.mark.parametrize("failure", ["actions", "config"])
def test_invalid_inputs_fail_before_loading_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    arguments = _input_arguments(tmp_path, _actions(7 if failure == "actions" else 8))
    if failure == "config":
        arguments.extend(["--chunk-size", "0"])
    monkeypatch.setattr(
        replay, "_setup_inference_pipeline", lambda _: pytest.fail("model loaded")
    )
    app = replay.create_app()
    with pytest.raises(ValueError):
        app.init(arguments)
    app.close()


def test_scene_preparation_failure_closes_the_loaded_pipeline_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _input_arguments(tmp_path, _actions())
    pipeline = _Pipeline()
    monkeypatch.setattr(replay, "_resolve_pipeline_config", lambda _: object())
    monkeypatch.setattr(replay, "_setup_inference_pipeline", lambda _: pipeline)

    def fail_scene(_scenario: Any, _pipeline: Any) -> Any:
        raise RuntimeError("scene encoding failed")

    monkeypatch.setattr(replay, "_load_scene", fail_scene)
    app = replay.create_app()
    with pytest.raises(RuntimeError, match="scene encoding failed"):
        app.init(arguments)
    app.close()
    assert pipeline.close_count == 1
    with pytest.raises(RuntimeError):
        app.create_session(app.session_desc())  # ty: ignore[invalid-argument-type]  # Closed apps have no session description.


def test_generation_failure_preserves_action_cursor_and_allows_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, pipeline, _, _ = _application(tmp_path, monkeypatch)
    try:
        session = app.create_session(app.session_desc())
        session.init()
        loop = session.model_loop
        loop.step(0, UserInputEvents([]))
        assert loop.state.row == 4

        def fail_generate(*_args: Any, **_kwargs: Any) -> torch.Tensor:
            raise RuntimeError("generation failed")

        monkeypatch.setattr(pipeline, "generate", fail_generate)
        with pytest.raises(RuntimeError, match="generation failed"):
            loop.step(1, UserInputEvents([]))
        assert loop.state.row == 4
        assert len(pipeline.finalized) == 1
        assert not loop.is_finished()
    finally:
        app.close()
        app.close()
    assert pipeline.released == pipeline.caches
    assert pipeline.close_count == 1


def test_runtime_drains_finite_replay_and_closes_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, pipeline, scene, _ = _application(tmp_path, monkeypatch, initialize=False)
    pipeline.frames = torch.zeros(4, 3, 352, 640)
    presented_frame_counts = []
    window = NullClientWindow()
    monkeypatch.setattr(
        window,
        "write",
        lambda result: presented_frame_counts.append(result.frame_count),
    )
    arguments = [
        "--scene-path",
        str(tmp_path / "scene"),
        "--recorded-actions",
        str(tmp_path / "actions.npy"),
    ]
    started = time.monotonic()
    ApplicationRunner(app, window).run(
        app.session_desc(),
        [*arguments, "--chunk-size", "1"],
        timeout_seconds=5,
    )

    assert time.monotonic() - started < 5
    assert pipeline.prepared == [scene]
    assert [index for index, _, _ in pipeline.generated] == [0, 1]
    assert sum(presented_frame_counts) >= 8
    assert pipeline.released == pipeline.caches
    assert pipeline.close_count == 1


def test_runtime_keeps_dataset_picker_open_after_recording_ends(tmp_path, monkeypatch):
    app, pipeline, scenes, _, _, _ = _dataset_application(tmp_path, monkeypatch)
    pipeline.frames = torch.zeros(4, 3, 352, 640)
    presented = []
    window = NullClientWindow()
    monkeypatch.setattr(window, "write", lambda result: presented.append(result))
    monkeypatch.setattr(
        window,
        "get_user_input_events",
        lambda: UserInputEvents(
            [CloseUserInputEvent(timestamp=np.uint64(0))]
            if len(presented) >= 12
            else []
        ),
    )
    ApplicationRunner(app, window).run(
        replace(app.session_desc(), metadata={"sample": "fold-tshirt"}),
        ["--chunk-size", "1"],
        timeout_seconds=5,
    )
    assert len(presented) >= 12
    assert pipeline.prepared == list(scenes.values())
    assert [index for index, _, _ in pipeline.generated] == [0, 1]
    assert pipeline.released == pipeline.caches
    assert pipeline.close_count == 1
