from __future__ import annotations

from pathlib import Path

import pytest
from interactive_drive import InteractiveDriveApplication

pytestmark = pytest.mark.ci_cpu


def test_interactive_drive_uses_regular_application_contract() -> None:
    app = InteractiveDriveApplication()
    assert app.session_desc().video_width == 1280
    assert app.session_desc().video_height == 704


def test_interactive_drive_resolves_default_scene_when_omitted(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "default.usdz"
    scene.touch()
    calls: list[None] = []

    def resolve_default_scene() -> Path:
        calls.append(None)
        return scene

    app = InteractiveDriveApplication(default_scene_resolver=resolve_default_scene)
    app.init(["--backend", "raster", "--total-blocks", "0"])

    assert calls == [None]
    assert app._config is not None
    assert app._config.app.scene_path == scene


def test_interactive_drive_prefers_explicit_scene(tmp_path: Path) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()

    def unexpected_default_scene() -> Path:
        raise AssertionError("default scene should not be resolved")

    app = InteractiveDriveApplication(default_scene_resolver=unexpected_default_scene)
    app.init(["--scene", str(scene), "--backend", "raster"])

    assert app._config is not None
    assert app._config.app.scene_path == scene
