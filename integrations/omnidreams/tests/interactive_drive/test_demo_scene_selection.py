# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
import types
from pathlib import Path

import pytest

from omnidreams.interactive_drive import demo as demo_mod
from omnidreams.interactive_drive.demo import (
    SceneOption,
    _resolve_scene_variant,
    build_parser,
)


def test_auto_start_flag_and_deprecated_alias() -> None:
    parser = build_parser()

    assert parser.parse_args([]).auto_start is False
    assert parser.parse_args(["--auto-start"]).auto_start is True
    # --autoload-scene is kept as a backward-compatible alias for --auto-start.
    assert parser.parse_args(["--autoload-scene"]).auto_start is True
    assert parser.parse_args(["--no-autoload-scene"]).auto_start is False


def test_resolve_scene_variant_prefers_weather_archive_path_for_default(
    tmp_path: Path,
) -> None:
    scene_uuid = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
    base = (tmp_path / f"clipgt-{scene_uuid}.usdz").resolve()
    snow = (tmp_path / f"clipgt-{scene_uuid}-snow.usdz").resolve()
    option = SceneOption(
        label="Quiet Suburban Boulevard",
        path=base,
        variants=("default", "rain", "snow"),
        variant_paths={"default": base, "snow": snow},
    )

    assert _resolve_scene_variant((option,), snow, "default") == "snow"


def test_resolve_scene_variant_keeps_explicit_weather_choice(tmp_path: Path) -> None:
    scene_uuid = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
    base = (tmp_path / f"clipgt-{scene_uuid}.usdz").resolve()
    snow = (tmp_path / f"clipgt-{scene_uuid}-snow.usdz").resolve()
    rain = (tmp_path / f"clipgt-{scene_uuid}-rain.usdz").resolve()
    option = SceneOption(
        label="Quiet Suburban Boulevard",
        path=base,
        variants=("default", "rain", "snow"),
        variant_paths={"default": base, "rain": rain, "snow": snow},
    )

    assert _resolve_scene_variant((option,), snow, "rain") == "rain"


def test_resolve_scene_variant_legacy_option_without_variant_paths(
    tmp_path: Path,
) -> None:
    scene = (tmp_path / "legacy.usdz").resolve()
    option = SceneOption(label="legacy", path=scene, variants=("1", "2"))

    assert _resolve_scene_variant((option,), scene, "default") == "1"
    assert _resolve_scene_variant((option,), scene, "2") == "2"


class _FakePresenter:
    """Records the scene-selection calls ``_run_streaming`` makes."""

    def __init__(self, **_kwargs: object) -> None:
        self.wait_for_scene_selection_calls = 0
        self.acknowledged: list[tuple[Path, str]] = []
        self.closed = False

    def set_model_status(self, **_kwargs: object) -> None: ...

    def set_scene_selection_locked(self, *_args: object) -> None: ...

    def wait_while_preloading(self, *_args: object) -> None: ...

    def wait_for_scene_selection(self) -> tuple[Path, str] | None:
        # The whole point of --auto-start is that this never runs; record any
        # call so the test can assert the picker was skipped.
        self.wait_for_scene_selection_calls += 1
        return None

    def acknowledge_scene_change(self, scene_path: Path, variant: str) -> None:
        self.acknowledged.append((scene_path, variant))

    @property
    def pending_scene_change(self) -> None:
        return None  # one rollout, then the loop exits

    def close(self) -> None:
        self.closed = True


class _FakeApp:
    can_prewarm = False

    def __init__(self, **_kwargs: object) -> None:
        self.loaded: list[tuple[Path, str]] = []
        self.ran = 0

    def model_ready(self) -> bool:
        return True

    def preload_in_progress(self) -> bool:
        return False

    def load_scene(self, scene_path: Path, variant: str, _prompt: object) -> bool:
        self.loaded.append((scene_path, variant))
        return True

    def run_scene(self) -> None:
        self.ran += 1

    def shutdown(self) -> None: ...


def test_run_streaming_auto_start_skips_scene_picker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene.usdz"
    scene.write_bytes(b"")
    option = SceneOption(label="scene", path=scene, variants=("default",))

    presenter = _FakePresenter()
    app = _FakeApp()

    monkeypatch.setattr(demo_mod, "_apply_cuda_visible_devices_inplace", lambda _v: None)
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
    monkeypatch.setattr(demo_mod, "InteractiveDriveApp", lambda **_k: app)
    # The streaming presenter is imported lazily inside _run_streaming
    import omnidreams.interactive_drive.input.keyboard as kbd_mod
    import omnidreams.interactive_drive.streaming_presenter as sp_mod

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
        auto_start=True,
    )

    demo_mod._run_streaming(args)

    assert presenter.wait_for_scene_selection_calls == 0
    assert app.loaded == [(scene, "default")]
    assert app.ran == 1
    assert presenter.acknowledged[0] == (scene, "default")
    assert presenter.closed
