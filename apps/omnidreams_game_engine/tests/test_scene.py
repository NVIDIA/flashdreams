# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for authored scene prompt selection."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from omnidreams_game_engine import scene as scene_module
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.game_map import GameMapError, load_game_map_header
from omnidreams_game_engine.scene import SceneRequest, load_scene

pytestmark = pytest.mark.ci_cpu

_MAP_HEADER_DOCUMENT = """\
schema_version: 1
id: shared-map
name: Shared Map
compiler: {}
nodes: []
roads: []
spawns:
  - {id: start, prompt: A road scene.}
"""
"""Minimal map metadata for filename validation."""


@pytest.mark.parametrize("suffix", [".game-map.yaml", ".robotaxi.yaml"])
def test_game_map_header_accepts_shared_and_legacy_suffixes(
    tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"shared{suffix}"
    path.write_text(_MAP_HEADER_DOCUMENT)

    header = load_game_map_header(path)

    assert header.map_id == "shared-map"
    assert header.source_path == path


def test_game_map_header_rejects_unrelated_yaml(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(_MAP_HEADER_DOCUMENT)

    with pytest.raises(GameMapError, match="suffixes"):
        load_game_map_header(path)


@pytest.mark.parametrize("use_prompt_context", [False, True])
def test_scene_forwards_context_selection_to_map_compiler(
    monkeypatch: pytest.MonkeyPatch,
    use_prompt_context: bool,
) -> None:
    compiled = SimpleNamespace(archive_path=Path("compiled.usdz"))
    compiler_options: list[dict[str, object]] = []
    loader_options: list[dict[str, object]] = []

    def compile_map(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        compiler_options.append(kwargs)
        return compiled

    monkeypatch.setattr(
        scene_module,
        "compile_game_map",
        compile_map,
    )
    monkeypatch.setattr(
        scene_module,
        "load_scene_bundle",
        lambda **kwargs: loader_options.append(kwargs),
    )

    load_scene(
        SceneRequest(
            map_path=Path("map.game-map.yaml"),
            use_prompt_context=use_prompt_context,
        ),
        RasterConfig(),
    )

    assert compiler_options == [
        {
            "spawn_id": None,
            "use_prompt_context": use_prompt_context,
            "force": False,
        }
    ]
    assert "prompt_override" not in loader_options[0]
