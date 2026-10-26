# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe configuration and application binding checks for OmniDreams."""

import re
from collections.abc import Callable
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, cast

import pytest
from crazy_robotaxi.application import CrazyRobotaxiApplication
from crazy_robotaxi.settings import (
    SettingsDocument,
    SettingsError,
    setting_value,
)
from interactive_drive import InteractiveDriveApplication, InteractiveDriveConfig
from omnidreams.apps.crazy_robotaxi.adapter import (
    OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS,
)
from omnidreams.apps.interactive_drive.adapter import (
    create_app as create_interactive_drive_app,
)
from omnidreams.apps.interactive_drive.adapter import (
    create_fast_perf_app as create_interactive_drive_fast_perf_app,
)
from omnidreams.apps.interactive_drive.adapter import (
    create_optimized_gb300_app as create_interactive_drive_gb300_app,
)
from omnidreams.apps.interactive_drive.adapter import (
    create_optimized_rtx_pro_6000_app as create_interactive_drive_rtx_pro_6000_app,
)
from omnidreams.apps.interactive_drive.adapter import (
    create_perf_app as create_interactive_drive_perf_app,
)
from omnidreams.config import (
    OMNIDREAMS_CONFIGS,
    OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_FAST_PERF_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_OPTIMIZED_GB300_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_PERF_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
    OMNIDREAMS_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_RTX_5090_FAST_PIPELINE_CONFIG,
    OMNIDREAMS_RTX_5090_PIPELINE_CONFIG,
)
from omnidreams.impl.pipeline import OmnidreamsPipelineConfig
from omnidreams.impl.transformer import CosmosTransformerConfig
from omnidreams.impl.vae_native import OmnidreamsWanVAEEncoderConfig

from flashdreams.api_v2.application import IApplication

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "key",
    (
        "synthetic_text_max_length",
        "diffusion_model.transformer.batch_shape",
        "diffusion_model.transformer.num_views",
        "diffusion_model.transformer.network.model_channels",
        "diffusion_model.transformer.network.enable_cross_view_attn",
        "diffusion_model.transformer.state_dict_transform",
        "text_encoder.embedding_concat_strategy",
        "text_encoder.n_layers_per_group",
        "image_encoder.z_dim",
        "image_encoder.native_vae_backend",
        "image_encoder.state_dict_transform",
        "encoder.latent_std",
        "encoder.native_vae_backend",
        "encoder.state_dict_transform",
    ),
)
def test_preset_owned_model_fields_are_not_user_overrides(
    tmp_path: Path, key: str
) -> None:
    path = tmp_path / "config.yaml"
    document = SettingsDocument.load(
        path, pipeline_config=OMNIDREAMS_PIPELINE_CONFIG, width=1280, height=704
    )
    names = ("model", "pipeline", *key.split("."))
    with pytest.raises(SettingsError, match="is not configurable"):
        document.update(document.settings, names, None)

    yaml = "\n".join(
        f"{'  ' * depth}{name}:" + (" null" if depth == len(names) - 1 else "")
        for depth, name in enumerate(names)
    )
    path.write_text(yaml + "\n", encoding="utf-8")
    if names[-1] == "state_dict_transform":
        with pytest.raises(
            SettingsError,
            match=re.escape(f"{'.'.join(names[:-1])} has unknown keys: {names[-1]}"),
        ):
            SettingsDocument.load(
                path, pipeline_config=OMNIDREAMS_PIPELINE_CONFIG, width=1280, height=704
            )
    else:
        loaded = SettingsDocument.load(
            path, pipeline_config=OMNIDREAMS_PIPELINE_CONFIG, width=1280, height=704
        )
        assert setting_value(loaded.settings, names) == setting_value(
            document.defaults, names
        )
        loaded.save(loaded.settings)
        assert "model:" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "config", OMNIDREAMS_CONFIGS.values(), ids=OMNIDREAMS_CONFIGS.keys()
)
def test_build_and_tuning_controls_round_trip(
    tmp_path: Path, config: OmnidreamsPipelineConfig
) -> None:
    path = tmp_path / "config.yaml"
    document = SettingsDocument.load(
        path, pipeline_config=config, width=1280, height=704
    )
    overrides: dict[str, Any] = {
        "model.pipeline.diffusion_model.transformer.window_size_t": 8,
        "model.pipeline.diffusion_model.scheduler.shift": 3.5,
        "game.taxi.vehicle.max_speed_mps": 25.0,
        "game.taxi.rules.global_time_s": 90.0,
    }
    for component, prefix in (
        ("diffusion_model.transformer", "native_dit"),
        ("image_encoder", "native_vae"),
        ("encoder", "native_vae"),
    ):
        overrides.update(
            {
                f"model.pipeline.{component}.{prefix}_build_root": str(
                    tmp_path / component
                ),
                f"model.pipeline.{component}.{prefix}_max_jobs": 2,
                f"model.pipeline.{component}.{prefix}_verbose_build": True,
            }
        )
    draft = document.settings
    for key, value in overrides.items():
        draft = document.update(draft, tuple(key.split(".")), value)
    document.save(draft)

    reloaded = SettingsDocument.load(
        path, pipeline_config=config, width=1280, height=704
    )
    for key, value in overrides.items():
        assert setting_value(reloaded.settings, tuple(key.split("."))) == value
    transformer = reloaded.settings.model.pipeline.diffusion_model.transformer
    expected_transformer = config.diffusion_model.transformer
    assert isinstance(expected_transformer, CosmosTransformerConfig)
    assert transformer.network == expected_transformer.network
    assert transformer.batch_shape == expected_transformer.batch_shape
    assert transformer.num_views == expected_transformer.num_views


@pytest.mark.parametrize(
    "config",
    [
        OMNIDREAMS_RESPONSIVE_PIPELINE_CONFIG,
        OMNIDREAMS_PERF_RESPONSIVE_PIPELINE_CONFIG,
        OMNIDREAMS_FAST_PERF_RESPONSIVE_PIPELINE_CONFIG,
        OMNIDREAMS_OPTIMIZED_GB300_RESPONSIVE_PIPELINE_CONFIG,
        OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_PIPELINE_CONFIG,
    ],
)
def test_responsive_presets_use_compatible_short_history_caches(
    config: OmnidreamsPipelineConfig,
) -> None:
    transformer = cast(CosmosTransformerConfig, config.diffusion_model.transformer)
    assert transformer.early_short_history_block_count is not None
    assert transformer.early_short_history_block_count > 0
    assert not transformer.network.apply_rope_before_kvcache
    assert transformer.native_dit_acceleration == "disabled"


def test_fast_perf_configures_native_fp8_vae_and_dit() -> None:
    config = OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG
    for encoder in (config.image_encoder, config.encoder):
        encoder = cast(OmnidreamsWanVAEEncoderConfig, encoder)
        assert encoder.native_vae_acceleration == "required"
        assert encoder.native_vae_backend == "fp8"
        assert encoder.native_vae_fp8_state_path or encoder.native_vae_fp8_auto_export
    transformer = cast(CosmosTransformerConfig, config.diffusion_model.transformer)
    assert transformer.native_dit_acceleration == "required"


def test_rtx_5090_presets_place_text_encoder_on_host() -> None:
    for pipeline in (
        OMNIDREAMS_RTX_5090_PIPELINE_CONFIG,
        OMNIDREAMS_RTX_5090_FAST_PIPELINE_CONFIG,
    ):
        assert pipeline.text_encoder is not None
        assert pipeline.text_encoder.run_on_cpu


def test_crazy_robotaxi_fast_perf_honors_explicit_pipeline_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "apps/crazy_robotaxi/tests/maps/race_course.robotaxi.yaml"
    )
    map_path = tmp_path / source.name
    map_path.write_bytes(source.read_bytes())
    monkeypatch.setattr("crazy_robotaxi.application._DEFAULT_MAP", map_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path / "cache"))
    app = CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS,
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: cast(Any, object()),
        native_preparer=lambda: None,
    )

    app.init(
        [
            "--seed",
            "7",
            "--no-compile",
            "--profile-pipeline",
        ],
    )

    pipeline = cast(Any, app._pipeline_config)
    transformer = pipeline.diffusion_model.transformer
    assert pipeline.diffusion_model.seed == 7
    assert transformer.compile_network is False
    assert app._config is not None
    assert app._config.pipeline_profiling


@pytest.mark.parametrize(
    ("factory", "resolution_wh"),
    [
        (create_interactive_drive_app, (1280, 704)),
        (create_interactive_drive_gb300_app, (1280, 704)),
        (create_interactive_drive_rtx_pro_6000_app, (1280, 704)),
        (create_interactive_drive_perf_app, (1168, 640)),
        (create_interactive_drive_fast_perf_app, (1168, 640)),
    ],
)
def test_each_application_owns_its_parsed_config(
    factory: Callable[[], IApplication],
    resolution_wh: tuple[int, int],
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene.usdz"
    scene.touch()
    app = factory()
    assert isinstance(app, InteractiveDriveApplication)
    app._backend_factory = cast(Any, lambda _: object())

    app.init(["--scene", str(scene)])

    assert app._config is not None
    assert app._config.app.raster.resolution_wh == resolution_wh
    assert type(app._config) is InteractiveDriveConfig


def test_installed_application_entry_points_construct_apps() -> None:
    entry_points = distribution("flashdreams-omnidreams").entry_points.select(
        group="flashdreams.applications_v2"
    )
    assert "crazy-robotaxi-omnidreams" in entry_points.names
    for entry_point in entry_points:
        factory = entry_point.load()
        assert isinstance(factory(), IApplication), entry_point.name
