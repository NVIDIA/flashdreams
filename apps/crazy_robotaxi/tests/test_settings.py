# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the user-authored Crazy Robotaxi settings document."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from crazy_robotaxi.settings import ModelPreset, SettingsDocument

pytestmark = pytest.mark.ci_cpu


@dataclass(frozen=True)
class _Diffusion:
    seed: int = 0


@dataclass(frozen=True)
class _Pipeline:
    name: str
    diffusion_model: _Diffusion = _Diffusion()


def _presets() -> dict[str, ModelPreset]:
    return {
        "regular": ModelPreset(_Pipeline("regular"), 1280, 704),
        "fast": ModelPreset(_Pipeline("fast"), 1168, 640),
    }


def _load(path: Path) -> SettingsDocument:
    return SettingsDocument.load(
        path,
        presets=_presets(),
        default_preset_name="regular",
    )


def test_sparse_yaml_selects_preset_and_overrides_nested_model_config(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """\
schema_version: 1
launch:
  mode: race
  map: maps/test.robotaxi.yaml
model:
  preset: fast
  pipeline:
    diffusion_model:
      seed: 42
presentation:
  show_fps: true
""",
        encoding="utf-8",
    )

    document = _load(path)

    assert document.settings.launch.mode == "race"
    assert (
        document.settings.launch.map == (tmp_path / "maps/test.robotaxi.yaml").resolve()
    )
    assert document.settings.model.preset == "fast"
    assert document.settings.model.pipeline.diffusion_model.seed == 42
    assert document.settings.renderer.raster.resolution_wh == (1168, 640)
    assert document.settings.presentation.show_fps


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


def test_non_default_preset_is_persisted_even_without_other_overrides(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    document = _load(path)
    draft = document.update(document.settings, ("model", "preset"), "fast")

    document.save(draft)

    assert _load(path).settings.model.preset == "fast"
    assert "preset: fast" in path.read_text(encoding="utf-8")
