# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for authored scene prompt selection."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from omnidreams_game_engine import scene as scene_module
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.scene import SceneRequest, load_scene

pytestmark = pytest.mark.ci_cpu


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
            map_path=Path("map.robotaxi.yaml"),
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
