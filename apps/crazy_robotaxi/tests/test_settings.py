# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the user-authored Crazy Robotaxi settings document."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from crazy_robotaxi.settings import SettingsDocument, SettingsError

pytestmark = pytest.mark.ci_cpu


@dataclass(frozen=True)
class _Diffusion:
    seed: int = 0


@dataclass(frozen=True)
class _Pipeline:
    name: str
    diffusion_model: _Diffusion = _Diffusion()


def _load(path: Path) -> SettingsDocument:
    return SettingsDocument.load(
        path,
        pipeline_config=_Pipeline("regular"),
        width=1280,
        height=704,
    )


def test_sparse_yaml_overrides_nested_model_config(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """\
schema_version: 1
model:
  pipeline:
    diffusion_model:
      seed: 42
game:
  gamepad_button_style: PlayStation
presentation:
  show_fps: true
""",
        encoding="utf-8",
    )

    document = _load(path)

    assert document.settings.model.pipeline.diffusion_model.seed == 42
    assert document.settings.renderer.raster.resolution_wh == (1280, 704)
    assert document.settings.game.gamepad_button_style == "PlayStation"
    assert document.settings.presentation.show_fps


def test_launch_selections_are_not_user_yaml_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("launch:\n  mode: race\n", encoding="utf-8")

    with pytest.raises(SettingsError, match="unknown keys: launch"):
        _load(path)


def test_save_is_sparse_atomic_and_preserves_retained_comments(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """\
# player preferences
presentation:
  show_fps: true  # keep this explanation
""",
        encoding="utf-8",
    )
    document = _load(path)
    draft = document.update(
        document.settings,
        ("presentation", "hud_enabled"),
        False,
    )

    document.save(draft)

    saved = path.read_text(encoding="utf-8")
    assert "# player preferences" in saved
    assert "show_fps: true  # keep this explanation" in saved
    assert "hud_enabled: false" in saved
    assert "runtime:" not in saved
    assert not tuple(tmp_path.glob(".config.yaml.*.tmp"))
