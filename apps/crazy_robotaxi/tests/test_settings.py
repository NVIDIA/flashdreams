# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the user-authored Crazy Robotaxi settings document."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from crazy_robotaxi.settings import SettingsDocument, SettingsError

pytestmark = pytest.mark.ci_cpu


@dataclass(frozen=True)
class _Diffusion:
    seed: int = 0


@dataclass(frozen=True)
class _Quantization:
    projection: torch.dtype | None = None


@dataclass(frozen=True)
class _Pipeline:
    name: str
    diffusion_model: _Diffusion = _Diffusion()
    quantization: _Quantization = _Quantization()


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


def test_pipeline_name_is_not_a_user_setting(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  pipeline:\n    name: other\n", encoding="utf-8")

    with pytest.raises(SettingsError, match="model.pipeline has unknown keys: name"):
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


def test_save_retains_quoted_string_override(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text('model:\n  device: "cpu"\n', encoding="utf-8")
    document = _load(path)

    document.save(
        document.update(
            document.settings,
            ("presentation", "show_fps"),
            True,
        )
    )

    assert _load(path).settings.model.device == "cpu"


def test_load_can_append_style_skin(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    skins = "\n".join(
        f"      - {{name: skin-{index}, prompt: prompt-{index}}}" for index in range(5)
    )
    path.write_text(
        f"live_edit:\n  style:\n    skins:\n{skins}\n",
        encoding="utf-8",
    )

    document = _load(path)

    assert document.settings.live_edit.style.skins[-1].name == "skin-4"
    assert document.settings.live_edit.style.skins[-1].prompt == "prompt-4"


def test_load_nullable_torch_dtype(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  pipeline:\n    quantization:\n      projection: float8_e4m3fn\n",
        encoding="utf-8",
    )

    document = _load(path)

    assert (
        document.settings.model.pipeline.quantization.projection is torch.float8_e4m3fn
    )


def test_save_rejects_runtime_invalid_taxi_rules(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    document = _load(path)
    draft = document.update(
        document.settings,
        ("game", "taxi", "rules", "pickup_grid_spacing_m"),
        0.0,
    )

    with pytest.raises(SettingsError, match="pickup_grid_spacing_m must be positive"):
        document.save(draft)

    assert not path.exists()
