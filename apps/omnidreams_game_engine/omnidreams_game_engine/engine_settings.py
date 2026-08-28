# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layered settings shared by interactive driving applications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.yaml_config import (
    StrictConfigError,
    load_yaml_mapping,
    overlay_dataclass,
    require_version,
)


@dataclass(frozen=True)
class MapLaunchSettings:
    """Map selection and cache behavior."""

    path: Path | None = None
    """Selected game-map path; ``None`` requires a CLI selection."""

    camera: str = "camera_front_wide_120fov"
    """Camera identifier selected from the map's available views."""

    variant: str = "default"
    """Visual variant selected from the map."""

    prompt: str | None = None
    """Base prompt override; ``None`` uses the map variant's prompt."""

    force_recompile: bool = False
    """Whether to rebuild the selected map's compiled cache once."""

    preload_maps: bool = False
    """Whether to parse selectable maps during startup."""


@dataclass(frozen=True)
class WorldModelLaunchSettings:
    """World-model selection and V2 runtime options."""

    backend: Literal["raster", "omnidreams"] = "omnidreams"
    """Main-camera backend; the V2 game requires ``omnidreams``."""

    offload_text_encoder: bool = False
    """Whether one-shot encoders may be released after initialization."""

    device: str = "cuda"
    """Device used to instantiate the V2 pipeline."""

    compile: bool | None = None
    """Optional override for transformer compilation."""

    profile_pipeline: bool = False
    """Whether to collect synchronized pipeline stage timings."""


@dataclass(frozen=True)
class RenderingSettings:
    """Primary-camera and BEV rendering configuration."""

    raster: RasterConfig = field(default_factory=RasterConfig)
    """Primary-camera rasterization settings."""

    bev: BevConfig = field(default_factory=BevConfig)
    """Top-down map rasterization settings."""


@dataclass(frozen=True)
class PresentationSettings:
    """Presentation options retained across V1 and V2 hosts."""

    hud_enabled: bool = True
    """Whether the host presents the game HUD."""

    show_fps: bool = False
    """Whether the HUD displays the measured generated-video frame rate."""

    stream_jpeg_quality: int = 85
    """JPEG quality used by streaming hosts."""

    stream_scale: float = 1.0
    """Streaming output scale."""


@dataclass(frozen=True)
class WheelSettings:
    """Optional steering-wheel configuration."""

    enabled: bool = True
    """Whether steering-wheel input is enabled."""

    profile: str = "auto"
    """Wheel profile name or automatic-selection marker."""


@dataclass(frozen=True)
class EngineRuntimeSettings:
    """Operational and V2 diagnostic controls."""

    cuda_visible_devices: str = "auto"
    """CUDA visibility override retained for configuration compatibility."""

    profile_world_model: bool = False
    """Legacy spelling for pipeline profiling."""

    total_blocks: int | None = None
    """Optional bound on generated blocks."""

    prewarm_blocks: int = 4
    """Hidden neutral blocks generated before presentation."""

    profile_input_latency: bool = False
    """Whether to display input-to-model-frame diagnostics."""


@dataclass(frozen=True)
class EngineSettings:
    """Complete durable engine configuration."""

    map: MapLaunchSettings = field(default_factory=MapLaunchSettings)
    """Map selection and cache behavior."""

    world_model: WorldModelLaunchSettings = field(
        default_factory=WorldModelLaunchSettings
    )
    """World-model selection and runtime behavior."""

    rendering: RenderingSettings = field(default_factory=RenderingSettings)
    """Primary-camera and BEV rendering settings."""

    presentation: PresentationSettings = field(default_factory=PresentationSettings)
    """Presentation settings shared with non-V2 hosts."""

    wheel: WheelSettings = field(default_factory=WheelSettings)
    """Steering-wheel settings shared with non-V2 hosts."""

    runtime: EngineRuntimeSettings = field(default_factory=EngineRuntimeSettings)
    """Operational and diagnostic settings."""


def load_engine_settings(
    path: Path,
    *,
    base: EngineSettings | None = None,
) -> EngineSettings:
    """Overlay a partial engine YAML onto typed settings.

    Args:
        path: Engine configuration path.
        base: Lower-precedence settings; ``None`` uses typed defaults.

    Returns:
        Resolved engine settings.

    Raises:
        StrictConfigError: The YAML or merged settings are invalid.
    """
    config_path = path.expanduser().resolve()
    doc = load_yaml_mapping(config_path)
    require_version(doc, "engine")
    values = dict(doc)
    values.pop("schema_version")
    settings = overlay_dataclass(
        base or EngineSettings(), values, "engine", base_dir=config_path.parent
    )
    _validate_engine_settings(settings)
    return settings


def _validate_engine_settings(settings: EngineSettings) -> None:
    raster = settings.rendering.raster
    bev = settings.rendering.bev
    if raster.width <= 0 or raster.height <= 0:
        raise StrictConfigError("engine.rendering.raster dimensions must be positive")
    if raster.near_plane_m >= raster.far_plane_m:
        raise StrictConfigError(
            "engine.rendering.raster.near_plane_m must be less than far_plane_m"
        )
    if raster.fog_start_m >= raster.fog_end_m:
        raise StrictConfigError(
            "engine.rendering.raster.fog_start_m must be less than fog_end_m"
        )
    if bev.width <= 0 or bev.height <= 0 or bev.height_m <= 0.0:
        raise StrictConfigError("engine.rendering.bev dimensions must be positive")
    if not 0.0 < bev.fov_deg < 180.0:
        raise StrictConfigError(
            "engine.rendering.bev.fov_deg must be between 0 and 180"
        )
    if settings.world_model.backend != "omnidreams":
        raise StrictConfigError(
            "Crazy Robotaxi V2 requires world_model.backend=omnidreams"
        )
    if settings.runtime.total_blocks is not None and settings.runtime.total_blocks <= 0:
        raise StrictConfigError("engine.runtime.total_blocks must be positive")
    if settings.runtime.prewarm_blocks < 0:
        raise StrictConfigError("engine.runtime.prewarm_blocks must be non-negative")
