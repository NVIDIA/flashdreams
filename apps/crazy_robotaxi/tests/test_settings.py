# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the user-authored Crazy Robotaxi settings document."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import pytest
import torch
from crazy_robotaxi.settings import (
    SettingsDocument,
    SettingsError,
    iter_setting_fields,
    parse_editor_value,
    setting_description,
)

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
    synthetic_text_max_length: int = 7
    """Preset-owned value excluded from user overrides."""

    state_dict_transform: Callable[[object], object] | None = None
    """Internal checkpoint hook, including its unset state."""

    optional_boolean: bool | None = None
    """Nullable user preference retained beside the internal hook."""

    python_name: str = "default"
    """Setting whose YAML key is its Python field name."""


@dataclass
class _DocumentedOptions:
    """Options with source documentation and an undocumented field."""

    documented: int = 0
    """Existing help with ``None`` and :attr:`other_value`.

    Additional implementation notes stay out of the tooltip.
    """

    undocumented: int = 0


@dataclass
class _InheritedDocumentedOptions(_DocumentedOptions):
    """Options that inherit their field documentation."""


@pytest.mark.parametrize(
    "config_type", (_DocumentedOptions, _InheritedDocumentedOptions)
)
def test_setting_description_reads_source_docs(
    config_type: type[_DocumentedOptions],
) -> None:
    config = config_type()
    assert {
        item.name: setting_description(config, item) for item in fields(config)
    } == {
        "documented": "Existing help with None and other_value.",
        "undocumented": None,
    }


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
  width: 1920
  height: 1080
  show_fps: true
  show_live_edit_buttons: false
  live_edit_mapping_location: control hints
  show_current_prompt: true
""",
        encoding="utf-8",
    )

    document = _load(path)

    assert document.settings.model.pipeline.diffusion_model.seed == 42
    assert document.settings.renderer.raster.resolution_wh == (1280, 704)
    assert document.settings.game.gamepad_button_style == "PlayStation"
    assert document.settings.presentation.width == 1920
    assert document.settings.presentation.height == 1080
    assert document.settings.presentation.show_fps
    assert not document.settings.presentation.show_live_edit_buttons
    assert document.settings.presentation.live_edit_mapping_location == "control hints"
    assert document.settings.presentation.show_current_prompt


def test_yaml_keys_come_from_python_field_names(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  pipeline:\n    python_name: custom\n", encoding="utf-8")
    document = _load(path)
    assert document.settings.model.pipeline.python_name == "custom"

    draft = document.update(
        document.settings, ("model", "pipeline", "python_name"), "updated"
    )
    document.save(draft)

    saved = path.read_text(encoding="utf-8")
    assert "python_name: updated" in saved
    assert _load(path).settings.model.pipeline.python_name == "updated"


@pytest.mark.parametrize(
    "presentation",
    (
        "width: 1920",
        "height: 1080",
        "width: 0\n  height: 1080",
    ),
)
def test_presentation_resolution_must_be_complete_and_positive(
    tmp_path: Path,
    presentation: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"presentation:\n  {presentation}\n", encoding="utf-8")

    with pytest.raises(SettingsError, match="presentation width and height"):
        _load(path)


def test_launch_selections_are_not_user_yaml_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("launch:\n  mode: race\n", encoding="utf-8")

    with pytest.raises(SettingsError, match="unknown keys: launch"):
        _load(path)


def test_live_edit_prompt_suffix_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """\
live_edit:
  weather:
    enabled: true
    weathers:
      - name: custom
        prompt_suffix: Custom weather conditions.
""",
        encoding="utf-8",
    )

    document = _load(path)
    document.save(document.settings)

    assert document.settings.live_edit.weather.weathers[0].prompt_suffix == (
        "Custom weather conditions."
    )
    assert "prompt_suffix: Custom weather conditions." in path.read_text(
        encoding="utf-8"
    )


def test_pipeline_name_is_not_a_user_setting(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  pipeline:\n    name: other\n", encoding="utf-8")

    with pytest.raises(SettingsError, match="model.pipeline has unknown keys: name"):
        _load(path)


@pytest.mark.parametrize("name", ("synthetic_text_max_length", "state_dict_transform"))
def test_internal_fields_are_excluded_from_yaml_and_drafts(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / "config.yaml"
    document = _load(path)
    names = {
        item.name
        for item, _ in iter_setting_fields(
            document.settings.model.pipeline, ("model", "pipeline")
        )
    }
    assert name not in names
    assert "optional_boolean" in names

    with pytest.raises(SettingsError, match="is not configurable"):
        document.update(document.settings, ("model", "pipeline", name), None)
    assert not path.exists()

    path.write_text(f"model:\n  pipeline:\n    {name}: null\n", encoding="utf-8")
    if name == "state_dict_transform":
        with pytest.raises(
            SettingsError, match=f"model.pipeline has unknown keys: {name}"
        ):
            _load(path)
    else:
        loaded = _load(path)
        assert loaded.settings.model.pipeline.synthetic_text_max_length == 7
        loaded.save(loaded.settings)
        assert name not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name",
    (
        "compute_device",
        "sync_gpu_timing",
        "perf_log_interval_frames",
        "near_plane_m",
        "far_plane_m",
        "fog_start_m",
        "fog_end_m",
        "fog_power",
        "triangle_raytrace_distance_m",
        "triangle_raytrace_edge_samples",
        "depth_clear_m",
    ),
)
def test_unused_raster_fields_are_not_user_settings(tmp_path: Path, name: str) -> None:
    path = tmp_path / "config.yaml"
    document = _load(path)
    names = {
        item.name
        for item, _ in iter_setting_fields(
            document.settings.renderer.raster, ("renderer", "raster")
        )
    }
    assert name not in names

    with pytest.raises(SettingsError, match="is not configurable"):
        document.update(document.settings, ("renderer", "raster", name), None)
    legacy_yaml = f"renderer:\n  raster:\n    {name}: null\n    width: 1024 # keep\n"
    path.write_text(legacy_yaml, encoding="utf-8")
    loaded = _load(path)
    assert getattr(loaded.settings.renderer.raster, name) == getattr(
        document.defaults.renderer.raster, name
    )
    assert loaded.settings.renderer.raster.width == 1024
    assert path.read_text(encoding="utf-8") == legacy_yaml

    loaded.save(loaded.settings)
    saved = path.read_text(encoding="utf-8")
    assert name not in saved
    assert "width: 1024 # keep" in saved
    assert _load(path).settings.renderer.raster.width == 1024


def test_new_dataclass_values_ignore_deprecated_settings(tmp_path: Path) -> None:
    raster = _load(tmp_path / "config.yaml").defaults.renderer.raster
    parsed = parse_editor_value(
        "{near_plane_m: null, width: 1024}",
        type(raster),
        None,
        ("renderer", "raster"),
        base_dir=tmp_path,
    )
    assert isinstance(parsed, type(raster))
    assert parsed.near_plane_m == raster.near_plane_m
    assert parsed.width == 1024

    with pytest.raises(SettingsError, match="unknown keys: near_plnae_m"):
        parse_editor_value(
            "{near_plane_m: null, near_plnae_m: 2}",
            type(raster),
            None,
            ("renderer", "raster"),
            base_dir=tmp_path,
        )


def test_deprecated_settings_do_not_hide_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "renderer:\n  raster:\n    near_plane_m: 2\n    near_plnae_m: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(SettingsError, match="unknown keys: near_plnae_m"):
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
