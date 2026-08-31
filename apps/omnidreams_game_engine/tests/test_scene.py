# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for authored scene prompt selection."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from omnidreams_game_engine import scene as scene_module
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.game_map.types import GameMapVisualVariant
from omnidreams_game_engine.scene import SceneRequest, load_scene

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    ("use_prompt_context", "explicit_prompt", "context_prompt", "expected"),
    [
        (False, None, "context prompt", None),
        (True, None, "context prompt", "context prompt"),
        (True, None, None, "full prompt"),
        (True, "explicit prompt", "context prompt", "explicit prompt"),
    ],
)
def test_scene_selects_context_prompt_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    use_prompt_context: bool,
    explicit_prompt: str | None,
    context_prompt: str | None,
    expected: str | None,
) -> None:
    variants = (
        GameMapVisualVariant(
            name="default",
            image=None,
            prompt="full prompt",
            prompt_context=context_prompt,
        ),
    )
    compiled = SimpleNamespace(
        archive_path=Path("compiled.usdz"),
        game_map=SimpleNamespace(
            default_spawn=SimpleNamespace(variants=variants),
        ),
    )
    captured: list[str | None] = []
    monkeypatch.setattr(
        scene_module, "compile_game_map", lambda *args, **kwargs: compiled
    )
    monkeypatch.setattr(
        scene_module,
        "load_scene_bundle",
        lambda **kwargs: captured.append(kwargs["prompt_override"]),
    )

    load_scene(
        SceneRequest(
            map_path=Path("map.robotaxi.yaml"),
            prompt=explicit_prompt,
            use_prompt_context=use_prompt_context,
        ),
        RasterConfig(),
    )

    assert captured == [expected]
