# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe configuration and application binding checks for OmniDreams."""

from collections.abc import Callable
from pathlib import Path

import pytest
import tomli as tomllib
from interactive_drive import InteractiveDriveApplication, InteractiveDriveConfig
from omnidreams.apps.interactive_drive.adapter import (
    OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS,
    OMNIDREAMS_INTERACTIVE_DRIVE_FAST_PERF_DEFAULTS,
    OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS,
)
from omnidreams.apps.interactive_drive.adapter import (
    create_app as create_interactive_drive_app,
)
from omnidreams.apps.interactive_drive.adapter import (
    create_fast_perf_app as create_interactive_drive_fast_perf_app,
)
from omnidreams.apps.interactive_drive.adapter import (
    create_perf_app as create_interactive_drive_perf_app,
)
from omnidreams.config import (
    OMNIDREAMS_CONFIGS,
    OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
)
from omnidreams.impl.transformer import CosmosTransformerConfig
from omnidreams.impl.vae_native import OmnidreamsWanVAEEncoderConfig

from flashdreams.api_v2.application import IApplication

pytestmark = pytest.mark.ci_cpu


def test_pipeline_configs_are_keyed_by_name() -> None:
    """Expose every model-owned OmniDreams pipeline config."""
    assert OMNIDREAMS_CONFIGS == {
        "omnidreams": OMNIDREAMS_PIPELINE_CONFIG,
        "omnidreams-perf": OMNIDREAMS_PERF_PIPELINE_CONFIG,
        "omnidreams-fast-perf": OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
    }


def test_fast_perf_uses_native_vae_when_available() -> None:
    config = OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG
    image_encoder = config.image_encoder
    encoder = config.encoder
    transformer = config.diffusion_model.transformer
    assert isinstance(image_encoder, OmnidreamsWanVAEEncoderConfig)
    assert isinstance(encoder, OmnidreamsWanVAEEncoderConfig)
    assert isinstance(transformer, CosmosTransformerConfig)

    assert image_encoder.native_vae_acceleration == "required"
    assert image_encoder.native_vae_backend == "fp8"
    assert image_encoder.native_vae_fp8_state_path is None
    assert image_encoder.native_vae_fp8_auto_export is True
    assert encoder.native_vae_acceleration == "required"
    assert encoder.native_vae_backend == "fp8"
    assert encoder.native_vae_fp8_state_path is None
    assert encoder.native_vae_fp8_auto_export is True
    assert transformer.native_dit_acceleration == "required"


def test_application_defaults_are_owned_by_each_adapter() -> None:
    """Keep demo-specific configuration beside each application factory."""
    assert OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS.slug == "interactive-drive"
    assert OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS.slug == "interactive-drive-perf"
    assert (
        OMNIDREAMS_INTERACTIVE_DRIVE_FAST_PERF_DEFAULTS.slug
        == "interactive-drive-fast-perf"
    )
    assert OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS.width == 1280
    assert OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS.width == 1168
    for defaults, pipeline_config in (
        (OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS, OMNIDREAMS_PIPELINE_CONFIG),
        (
            OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS,
            OMNIDREAMS_PERF_PIPELINE_CONFIG,
        ),
        (
            OMNIDREAMS_INTERACTIVE_DRIVE_FAST_PERF_DEFAULTS,
            OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
        ),
    ):
        assert defaults.pipeline_config is pipeline_config


@pytest.mark.parametrize(
    ("factory", "resolution_wh"),
    [
        (create_interactive_drive_app, (1280, 704)),
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

    app.init(["--scene", str(scene)])

    assert isinstance(app, InteractiveDriveApplication)
    assert app._config is not None
    assert app._config.app.raster.resolution_wh == resolution_wh
    assert type(app._config) is InteractiveDriveConfig


def test_pyproject_registers_model_owned_app_adapters() -> None:
    """Expose applications through nested adapters and no runner entry points."""
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(path.read_text())
    entry_points = project["project"]["entry-points"]

    assert "flashdreams.runner_configs" not in entry_points
    assert entry_points["flashdreams.applications_v2"] == {
        "interactive-drive-omnidreams": (
            "omnidreams.apps.interactive_drive.adapter:create_app"
        ),
        "interactive-drive-omnidreams-perf": (
            "omnidreams.apps.interactive_drive.adapter:create_perf_app"
        ),
        "interactive-drive-omnidreams-fast-perf": (
            "omnidreams.apps.interactive_drive.adapter:create_fast_perf_app"
        ),
        "crazy-robotaxi-omnidreams": (
            "omnidreams.apps.crazy_robotaxi.adapter:create_app"
        ),
        "crazy-robotaxi-omnidreams-perf": (
            "omnidreams.apps.crazy_robotaxi.adapter:create_perf_app"
        ),
        "crazy-robotaxi-omnidreams-fast-perf": (
            "omnidreams.apps.crazy_robotaxi.adapter:create_fast_perf_app"
        ),
    }
