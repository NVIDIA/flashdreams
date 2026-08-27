# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
import types
from collections.abc import Callable
from pathlib import Path

import pytest
from crazy_robotaxi import cli as demo_mod
from crazy_robotaxi.cli import (
    SceneOption,
    _resolve_scene_variant,
    _validate_presenter_mode,
    build_parser,
)
from omnidreams_game_engine.demo import build_parser as build_engine_demo_parser

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "removed_flag",
    ("--auto-start", "--no-auto-start", "--autoload-scene", "--no-autoload-scene"),
)
def test_removed_auto_start_flags_are_rejected(removed_flag: str) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([removed_flag])


@pytest.mark.parametrize("parser_factory", [build_parser, build_engine_demo_parser])
def test_map_directory_flag_replaces_scene_directory(
    parser_factory: Callable[[], argparse.ArgumentParser],
) -> None:
    parser = parser_factory()

    assert parser.parse_args(["--map-dir", "maps"]).scene_dir == Path("maps")
    with pytest.raises(SystemExit):
        parser.parse_args(["--scene-dir", "scenes"])


def test_bare_native_taxi_mode_is_rejected() -> None:
    args = build_parser().parse_args(["--game-mode", "taxi", "--no-hud"])

    with pytest.raises(SystemExit, match="has no game or BEV overlays"):
        _validate_presenter_mode(args)


def test_browser_taxi_mode_may_imply_no_hud() -> None:
    args = build_parser().parse_args(
        ["--game-mode", "taxi", "--no-hud", "--stream-mjpeg", "8080"]
    )

    _validate_presenter_mode(args)


def test_resolve_scene_variant_uses_default_map_variant(tmp_path: Path) -> None:
    game_map = (tmp_path / "city.robotaxi.yaml").resolve()
    option = SceneOption(
        label="City",
        path=game_map,
        variants=("default", "rain", "snow"),
        variant_paths={
            "default": game_map,
            "rain": game_map,
            "snow": game_map,
        },
    )

    assert _resolve_scene_variant((option,), game_map, "default") == "default"


def test_resolve_scene_variant_keeps_explicit_choice(tmp_path: Path) -> None:
    game_map = (tmp_path / "city.robotaxi.yaml").resolve()
    option = SceneOption(
        label="City",
        path=game_map,
        variants=("default", "rain", "snow"),
        variant_paths={variant: game_map for variant in ("default", "rain", "snow")},
    )

    assert _resolve_scene_variant((option,), game_map, "rain") == "rain"


def test_resolve_scene_variant_falls_back_to_first_authored_variant(
    tmp_path: Path,
) -> None:
    game_map = (tmp_path / "city.robotaxi.yaml").resolve()
    option = SceneOption(
        label="City",
        path=game_map,
        variants=("default", "rain"),
        variant_paths={"default": game_map, "rain": game_map},
    )

    assert _resolve_scene_variant((option,), game_map, "missing") == "default"


class _FakePresenter:
    """Records the scene lifecycle calls ``_run_streaming`` makes."""

    def __init__(self, **_kwargs: object) -> None:
        self.acknowledged: list[tuple[Path, str]] = []
        # Probe callables passed to wait_while_preloading, plus an ordered
        # call log so a test can assert the preload wait happens *before* the
        # scene is acknowledged.
        self.wait_while_preloading_probes: list[object] = []
        self.calls: list[str] = []
        self.closed = False

    def set_scene_selection_locked(self, *_args: object) -> None: ...

    def wait_while_preloading(self, probe: object) -> None:
        self.wait_while_preloading_probes.append(probe)
        self.calls.append("wait_while_preloading")

    def acknowledge_scene_change(self, scene_path: Path, variant: str) -> None:
        self.acknowledged.append((scene_path, variant))
        self.calls.append("acknowledge_scene_change")

    @property
    def pending_scene_change(self) -> tuple[Path, str] | None:
        return None  # one rollout, then the loop exits

    def close(self) -> None:
        self.closed = True


class _FakeApp:
    can_prewarm = False

    def __init__(
        self, preload_states: tuple[bool, ...] = (False,), **_kwargs: object
    ) -> None:
        self.loaded: list[tuple[Path, str]] = []
        self.ran = 0
        # Successive return values for preload_in_progress().
        self._preload_states = list(preload_states)

    def model_ready(self) -> bool:
        return True

    def preload_in_progress(self) -> bool:
        if self._preload_states:
            return self._preload_states.pop(0)
        return False

    def load_scene(self, scene_path: Path, variant: str, _prompt: object) -> bool:
        self.loaded.append((scene_path, variant))
        return True

    def run_scene(self) -> None:
        self.ran += 1

    def shutdown(self) -> None: ...


@pytest.mark.parametrize("preloading", [False, True])
def test_run_streaming_starts_command_line_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preloading: bool,
) -> None:
    scene = tmp_path / "scene.robotaxi.yaml"
    scene.write_bytes(b"")
    option = SceneOption(label="scene", path=scene, variants=("default",))

    presenter = _FakePresenter()
    # When preloading, preload_in_progress() reports True on the first check.
    app = _FakeApp(preload_states=(True, False) if preloading else (False,))

    monkeypatch.setattr(
        demo_mod, "_apply_cuda_visible_devices_inplace", lambda _v: None
    )
    monkeypatch.setattr(demo_mod, "_resolve_demo_paths", lambda _a: None)
    monkeypatch.setattr(demo_mod, "_discover_scene_options", lambda *_a: (option,))
    monkeypatch.setattr(
        demo_mod._cli,
        "prepare_config_and_backend",
        lambda _a: (
            types.SimpleNamespace(scene_path=scene, variant="default"),
            object(),
        ),
    )
    monkeypatch.setattr(demo_mod, "_build_application", lambda *_a, **_k: app)
    # The streaming presenter is imported lazily inside _run_streaming
    import crazy_robotaxi.streaming_presenter as sp_mod
    import omnidreams_game_engine.input.keyboard as kbd_mod

    monkeypatch.setattr(sp_mod, "MJPEGStreamingPresenter", lambda **_k: presenter)
    monkeypatch.setattr(sp_mod, "parse_bind", lambda _v: ("127.0.0.1", 8080))
    monkeypatch.setattr(kbd_mod, "KeyboardState", lambda *_a, **_k: object())

    args = argparse.Namespace(
        cuda_visible_devices="",
        scene_dir=tmp_path,
        scene=scene,
        backend="placeholder",
        manifest=None,
        stream_mjpeg="8080",
        preload_scenes=False,
        prompt=None,
    )

    demo_mod._run_streaming(args)

    assert app.loaded == [(scene, "default")]
    assert app.ran == 1
    assert presenter.acknowledged[0] == (scene, "default")
    assert presenter.closed

    if preloading:
        # The preload wait fires with the app's own probe, before the scene
        # is acknowledged/loaded.
        assert presenter.wait_while_preloading_probes == [app.preload_in_progress]
        assert presenter.calls.index("wait_while_preloading") < presenter.calls.index(
            "acknowledge_scene_change"
        )
    else:
        assert presenter.wait_while_preloading_probes == []
