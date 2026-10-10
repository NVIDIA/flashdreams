# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified user-authored settings for Crazy Robotaxi."""

from __future__ import annotations

import copy
import io
import os
import re
import tempfile
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, Field, dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

import torch
from omnidreams_game_engine.config import BevConfig, RasterConfig
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from tyro._docstrings import get_field_docstring

from crazy_robotaxi.controls import GamepadButtonStyle
from crazy_robotaxi.dynamics import TaxiVehicleConfig
from crazy_robotaxi.live_edit.config import LiveEditConfig
from crazy_robotaxi.rules import TaxiGameConfig

SettingPath = tuple[str, ...]
LiveEditMappingLocation = Literal["buttons", "control hints"]
SETTING_CLI_FLAGS: dict[str, str] = {
    "diagnostics.input_trace_path": "--profile-input-latency [TRACE_PATH]",
    "diagnostics.profile_input_latency": "--profile-input-latency [TRACE_PATH]",
    "diagnostics.profile_pipeline": "--profile-pipeline",
    "game.effects.visual_flare": "--visual-flare, --no-visual-flare",
    "game.race.times_path": "--race-times",
    "game.taxi.high_scores_path": "--high-scores",
    "game.taxi.rules.global_time_s": "--game-time-s",
    "game.taxi.seed": "--game-seed, --seed",
    "live_edit.coins.enabled": "--live-edit-coins, --no-live-edit-coins",
    "live_edit.coins.max_visible_sprites": "--live-edit-coin-max-visible",
    "live_edit.coins.sprite_path": "--live-edit-coin-sprite",
    "live_edit.items.enabled": "--live-edit-items, --no-live-edit-items",
    "live_edit.items.item_types": "--live-edit-item-types",
    "live_edit.items.mystery_seed": "--live-edit-item-mystery-seed",
    "live_edit.items.mystery_sprite_path": "--live-edit-item-mystery-sprite",
    "live_edit.items.nitro_boost": "--live-edit-nitro-boost",
    "live_edit.items.nitro_duration_s": "--live-edit-nitro-duration-s",
    "live_edit.items.nitro_max_speed_mps": "--live-edit-nitro-max-speed",
    "live_edit.items.nitro_sprite_path": "--live-edit-item-nitro-sprite",
    "live_edit.items.rain_sprite_path": "--live-edit-item-rain-sprite",
    "live_edit.items.snow_sprite_path": "--live-edit-item-snow-sprite",
    "live_edit.items.spacing_m": "--live-edit-item-spacing",
    "live_edit.map_context.enabled": "--live-edit-map-context, --no-live-edit-map-context",
    "live_edit.obstacle.active_chunks": "--live-edit-obstacle-chunks",
    "live_edit.obstacle.annotate": "--live-edit-obstacle-annotate, --no-live-edit-obstacle-annotate",
    "live_edit.obstacle.count": "--live-edit-obstacle-count",
    "live_edit.obstacle.enabled": "--live-edit-obstacle, --no-live-edit-obstacle",
    "live_edit.obstacle.guide_scale": "--live-edit-obstacle-guide-scale",
    "live_edit.obstacle.physics": "--live-edit-obstacle-physics, --no-live-edit-obstacle-physics",
    "live_edit.obstacle.placement": "--live-edit-obstacle-placement",
    "live_edit.obstacle.spawn_ahead_m": "--live-edit-obstacle-ahead-m",
    "live_edit.obstacle.stagger_chunks": "--live-edit-obstacle-stagger-chunks",
    "live_edit.obstacle.static_ahead_m": "--live-edit-obstacle-static-ahead-m",
    "live_edit.obstacle.static_count": "--live-edit-obstacle-static-count",
    "live_edit.obstacle.static_lateral_m": "--live-edit-obstacle-static-lateral-m",
    "live_edit.perf_log_every_frames": "--live-edit-perf-log",
    "live_edit.style.base_corrector_checkpoint": "--live-edit-base-corrector",
    "live_edit.style.base_corrector_gain": "--live-edit-base-corrector-gain",
    "live_edit.style.corrector_checkpoint": "--live-edit-style-corrector",
    "live_edit.style.corrector_gain": "--live-edit-style-gain",
    "live_edit.style.corrector_mode": "--live-edit-corrector-mode",
    "live_edit.style.enabled": "--live-edit-style, --no-live-edit-style",
    "live_edit.style.gate_alpha_json": "--live-edit-gate-alpha-json",
    "live_edit.style.guidance_chunks": "--live-edit-skin-guidance-chunks",
    "live_edit.style.lora_checkpoint": "--live-edit-style-lora",
    "live_edit.style.reswap_interval_chunks": "--live-edit-style-reswap-chunks",
    "live_edit.style.skins": "--live-edit-skin-first (cycle order only)",
    "live_edit.weather.clear_guidance_chunks": "--live-edit-weather-clear-guidance-chunks",
    "live_edit.weather.corrector_checkpoint": "--live-edit-weather-corrector",
    "live_edit.weather.corrector_gain": "--live-edit-weather-corrector-gain",
    "live_edit.weather.enabled": "--live-edit-weather, --no-live-edit-weather",
    "live_edit.weather.guidance_chunks": "--live-edit-weather-guidance-chunks",
    "live_edit.weather.guidance_scale": "--live-edit-weather-guidance",
    "live_edit.weather.maintain_chunks": "--live-edit-weather-maintain-chunks",
    "live_edit.weather.maintain_interval_chunks": "--live-edit-weather-maintain-interval",
    "live_edit.weather.weathers": "--live-edit-weather-first (cycle order only)",
    "model.device": "--device",
    "model.pipeline.diffusion_model.seed": "--model-seed, --seed",
    "model.pipeline.diffusion_model.transformer.compile_network": "--compile, --no-compile",
    "presentation.height": "--display-height",
    "presentation.show_fps": "--show-fps, --no-show-fps",
    "presentation.width": "--display-width",
    "renderer.raster.height": "--height",
    "renderer.raster.width": "--width",
    "runtime.prewarm_blocks": "--prewarm-blocks",
    "runtime.total_blocks": "--total-blocks",
}
"""CLI flags that change each setting, keyed by its path in Crazy Robotaxi."""

_NON_USER_SETTING_PATHS = frozenset({("model", "pipeline", "name")})
_DEPRECATED_SETTING_PATHS = frozenset(
    tuple(path.split("."))
    for path in (
        "model.pipeline.diffusion_model.transformer.batch_shape",
        "model.pipeline.diffusion_model.transformer.network.adaln_lora_dim",
        "model.pipeline.diffusion_model.transformer.network.additional_concat_ch",
        "model.pipeline.diffusion_model.transformer.network.concat_padding_mask",
        "model.pipeline.diffusion_model.transformer.network.crossattn_emb_channels",
        "model.pipeline.diffusion_model.transformer.network.crossattn_proj_in_channels",
        "model.pipeline.diffusion_model.transformer.network.enable_cross_view_attn",
        "model.pipeline.diffusion_model.transformer.network.in_channels",
        "model.pipeline.diffusion_model.transformer.network.mlp_ratio",
        "model.pipeline.diffusion_model.transformer.network.model_channels",
        "model.pipeline.diffusion_model.transformer.network.n_cameras_emb",
        "model.pipeline.diffusion_model.transformer.network.num_blocks",
        "model.pipeline.diffusion_model.transformer.network.num_heads",
        "model.pipeline.diffusion_model.transformer.network.out_channels",
        "model.pipeline.diffusion_model.transformer.network.patch_spatial",
        "model.pipeline.diffusion_model.transformer.network.patch_temporal",
        "model.pipeline.diffusion_model.transformer.network.use_adaln_lora",
        "model.pipeline.diffusion_model.transformer.network.use_crossattn_projection",
        "model.pipeline.diffusion_model.transformer.network.view_condition_dim",
        "model.pipeline.diffusion_model.transformer.num_views",
        "model.pipeline.encoder.base_dim",
        "model.pipeline.encoder.is_residual",
        "model.pipeline.encoder.latent_mean",
        "model.pipeline.encoder.latent_std",
        "model.pipeline.encoder.native_vae_backend",
        "model.pipeline.encoder.patch_size",
        "model.pipeline.encoder.z_dim",
        "model.pipeline.image_encoder.base_dim",
        "model.pipeline.image_encoder.is_residual",
        "model.pipeline.image_encoder.latent_mean",
        "model.pipeline.image_encoder.latent_std",
        "model.pipeline.image_encoder.native_vae_backend",
        "model.pipeline.image_encoder.patch_size",
        "model.pipeline.image_encoder.z_dim",
        "model.pipeline.synthetic_text_max_length",
        "model.pipeline.text_encoder.embedding_concat_strategy",
        "model.pipeline.text_encoder.n_layers_per_group",
        "renderer.raster.compute_device",
        "renderer.raster.depth_clear_m",
        "renderer.raster.far_plane_m",
        "renderer.raster.fog_end_m",
        "renderer.raster.fog_power",
        "renderer.raster.fog_start_m",
        "renderer.raster.near_plane_m",
        "renderer.raster.perf_log_interval_frames",
        "renderer.raster.sync_gpu_timing",
        "renderer.raster.triangle_raytrace_distance_m",
        "renderer.raster.triangle_raytrace_edge_samples",
    )
)
"""Settings excluded by Crazy Robotaxi that may remain in older YAML files."""


@dataclass(frozen=True)
class TaxiRulesSettings:
    """Taxi rules without session persistence or vehicle dynamics."""

    waypoint_spacing_m: float = 10.0
    """Distance between candidate points sampled along navigation routes."""

    pickup_grid_spacing_m: float = 60.0
    """Spacing used to spread pickup locations across the map."""

    pickup_min_distance_m: float = 20.0
    """Minimum straight-line distance from the taxi to a new pickup."""

    initial_pickup_max_distance_m: float = 200.0
    """Preferred maximum distance to the first, camera-visible pickup."""

    pickup_radius_m: float = 5.0
    """Distance at which a passenger is collected."""

    dropoff_radius_m: float = 6.0
    """Distance at which a fare is completed."""

    fare_min_route_distance_m: float = 200.0
    """Preferred minimum route length from pickup to dropoff."""

    fare_max_route_distance_m: float = 250.0
    """Preferred maximum straight-line distance between fare endpoints. The minimum may
    not exceed this maximum."""

    target_speed_mps: float = 10.0
    """Nominal speed used to calculate a fare's time limit."""

    grace_s: float = 8.0
    """Extra time added to the distance-based fare limit."""

    min_time_s: float = 12.0
    """Lower bound for a fare's calculated time limit."""

    max_time_s: float = 45.0
    """Upper bound for a fare's calculated time limit; must be at least Min Time S."""

    trip_time_multiplier: float = 2.0
    """Multiplies the fare limit after it is calculated and clamped."""

    base_fare_points: int = 500
    """Points awarded for a completed fare."""

    bonus_points_per_second: int = 100
    """Additional points per whole second remaining on a completed fare."""

    event_banner_s: float = 2.0
    """Duration of pickup, completion, and failure banners in simulation time."""

    global_time_s: float = 60.0
    """Starting game clock; must be positive."""

    dropoff_time_bonus_s: float = 30.0
    """Time added to the game clock after a successful dropoff."""

    ground_snap_max_absolute_rotation_deg: float = 10.0
    """Largest ground rotation accepted when aligning the taxi to the road surface."""

    ground_snap_settle_fraction: float = 0.25
    """Fraction of stale ground attitude removed after an invalid ground sample."""


@dataclass(frozen=True)
class TaxiSettings:
    """Taxi rules, vehicle behavior, and persistence."""

    seed: int | None = None
    """Seed for repeatable taxi gameplay; blank uses fresh randomness. This is
    independent of the model diffusion seed."""

    high_scores_path: Path | None = None
    """CSV file for the taxi leaderboard; blank uses the default high-score location."""

    rules: TaxiRulesSettings = TaxiRulesSettings()
    vehicle: TaxiVehicleConfig = TaxiVehicleConfig()

    def game_config(self, *, default_high_scores_path: Path) -> TaxiGameConfig:
        """Resolve the existing runtime game configuration."""
        return TaxiGameConfig(
            vehicle=self.vehicle,
            seed=self.seed,
            high_scores_path=self.high_scores_path or default_high_scores_path,
            **{
                item.name: getattr(self.rules, item.name) for item in fields(self.rules)
            },
        )


@dataclass(frozen=True)
class RaceSettings:
    """Race persistence settings; course selection belongs to launch."""

    times_path: Path | None = None
    """File for race times; blank uses the default leaderboard location."""


@dataclass(frozen=True)
class GameEffectsSettings:
    """Game-directed presentation effects."""

    visual_flare: bool = False
    """Enables the game-directed visual flare effect."""


@dataclass(frozen=True)
class GameSettings:
    """Complete gameplay configuration."""

    gamepad_button_style: GamepadButtonStyle = "Xbox"
    """Labels shown for gamepad buttons: Xbox, PlayStation, or Nintendo Switch. It does
    not remap controls."""

    taxi: TaxiSettings = TaxiSettings()
    race: RaceSettings = RaceSettings()
    effects: GameEffectsSettings = GameEffectsSettings()


@dataclass(frozen=True)
class ModelSettings:
    """Runner-owned pipeline configuration and device placement."""

    device: str = "cuda"
    """Device used for the world model, normally cuda."""

    pipeline: Any = None


@dataclass(frozen=True)
class RendererSettings:
    """Primary semantic raster and top-down renderer settings."""

    raster: RasterConfig = RasterConfig()
    bev: BevConfig = BevConfig()


@dataclass(frozen=True)
class PresentationSettings:
    """Player-facing HUD settings."""

    width: int | None = None
    """Presentation width in pixels; ``None`` uses the model output width.
    Set together with the height.
    """

    height: int | None = None
    """Presentation height in pixels; ``None`` uses the model output height.
    Set together with the width.
    """

    hud_enabled: bool = True
    """Shows the gameplay HUD."""

    show_fps: bool = False
    """Shows the frame-rate counter."""

    show_current_prompt: bool = False
    """Shows the world-model prompt."""

    show_control_hints: bool = True
    """Shows the control help on the HUD."""

    show_live_edit_buttons: bool = True
    """Shows live-edit ability buttons."""

    live_edit_mapping_location: LiveEditMappingLocation = "buttons"
    """Places live-edit mappings in buttons or control hints."""


@dataclass(frozen=True)
class RuntimeSettings:
    """Operational controls for one application session."""

    total_blocks: int | None = None
    """Optional limit on generated model blocks; blank leaves the run unbounded."""

    prewarm_blocks: int = 8
    """Blocks generated before play to warm the pipeline; must be nonnegative."""


@dataclass(frozen=True)
class DiagnosticsSettings:
    """Opt-in profiling and diagnostic output."""

    profile_pipeline: bool = False
    """Enables pipeline profiling output."""

    profile_input_latency: bool = False
    """Measures input-to-output latency."""

    input_trace_path: Path | None = None
    """Optional file for input trace output."""


@dataclass(frozen=True)
class CrazyRobotaxiUserSettings:
    """The settings tree represented by both YAML and the Options UI."""

    game: GameSettings
    model: ModelSettings
    renderer: RendererSettings
    presentation: PresentationSettings = PresentationSettings()
    live_edit: LiveEditConfig = LiveEditConfig()
    runtime: RuntimeSettings = RuntimeSettings()
    diagnostics: DiagnosticsSettings = DiagnosticsSettings()


def default_config_path() -> Path:
    """Return the platform-style per-user configuration path."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "crazy-robotaxi" / "config.yaml"


def default_settings(
    pipeline_config: Any,
    *,
    width: int,
    height: int,
) -> CrazyRobotaxiUserSettings:
    """Build defaults around the pipeline selected by the runner."""
    return CrazyRobotaxiUserSettings(
        game=GameSettings(),
        model=ModelSettings(
            pipeline=copy.deepcopy(pipeline_config),
        ),
        renderer=RendererSettings(
            raster=RasterConfig(width=width, height=height),
            bev=BevConfig(),
        ),
    )


class SettingsError(ValueError):
    """Invalid user-authored settings."""


def presentation_resolution_wh(
    settings: PresentationSettings,
) -> tuple[int, int] | None:
    """Return the configured presentation resolution when complete."""
    width = settings.width
    height = settings.height
    if (width is None) != (height is None):
        raise SettingsError("presentation width and height must be set together")
    if width is None or height is None:
        return None
    if width <= 0 or height <= 0:
        raise SettingsError("presentation width and height must be positive")
    return width, height


@dataclass
class SettingsDocument:
    """Round-trip YAML document and its resolved typed settings."""

    path: Path
    defaults: CrazyRobotaxiUserSettings
    settings: CrazyRobotaxiUserSettings
    cli_overrides: dict[SettingPath, object]
    _yaml: YAML
    _document: CommentedMap

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        pipeline_config: Any,
        width: int,
        height: int,
    ) -> "SettingsDocument":
        """Load sparse YAML over the runner-selected pipeline defaults."""
        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        config_path = path.expanduser().resolve()
        if config_path.exists():
            try:
                raw = yaml.load(config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise SettingsError(f"Could not parse {config_path}: {exc}") from exc
            if raw is None:
                raw = CommentedMap()
            if not isinstance(raw, CommentedMap):
                raise SettingsError(f"{config_path} must contain a YAML mapping")
            document = raw
        else:
            document = CommentedMap()
            document.yaml_set_start_comment(
                "Crazy Robotaxi user settings. Omitted values inherit preset defaults."
            )
        version = document.get("schema_version", 1)
        if not isinstance(version, int) or isinstance(version, bool) or version != 1:
            raise SettingsError("schema_version must be 1")
        base = default_settings(
            pipeline_config,
            width=width,
            height=height,
        )
        values = {
            key: value for key, value in document.items() if key != "schema_version"
        }
        settings = _overlay_dataclass(base, values, (), base_dir=config_path.parent)
        settings = normalize_settings(settings)
        _validate_settings(settings)
        return cls(
            path=config_path,
            defaults=base,
            settings=settings,
            cli_overrides={},
            _yaml=yaml,
            _document=document,
        )

    def update(
        self,
        settings: CrazyRobotaxiUserSettings,
        path: SettingPath,
        value: object,
    ) -> CrazyRobotaxiUserSettings:
        """Replace one draft value in the typed settings tree."""
        current: object = settings
        for depth, name in enumerate(path):
            known = {
                item.name for item, _ in iter_setting_fields(current, path[:depth])
            }
            if name not in known:
                raise SettingsError(f"{'.'.join(path)} is not configurable")
            current = getattr(current, name)
        return replace_setting(settings, path, value)

    def save(self, settings: CrazyRobotaxiUserSettings) -> None:
        """Validate and atomically save sparse overrides while retaining comments."""
        settings = normalize_settings(settings)
        _validate_settings(settings)
        desired = CommentedMap()
        desired["schema_version"] = 1
        desired.update(_settings_diff(self.defaults, settings))
        _sync_mapping(self._document, desired)
        buffer = io.StringIO()
        self._yaml.dump(self._document, buffer)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(buffer.getvalue())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        self.settings = settings


def normalize_settings(
    settings: CrazyRobotaxiUserSettings,
) -> CrazyRobotaxiUserSettings:
    """Apply declared feature dependencies without mutating preset literals."""
    live_edit = settings.live_edit
    item_types = set(live_edit.items.item_types) if live_edit.items.enabled else set()
    live_edit = replace(
        live_edit,
        style=replace(
            live_edit.style,
            enabled=live_edit.style.enabled or "mystery" in item_types,
        ),
        weather=replace(
            live_edit.weather,
            enabled=live_edit.weather.enabled or bool({"rain", "snow"} & item_types),
        ),
    )
    return replace(settings, live_edit=live_edit)


def _validate_settings(settings: CrazyRobotaxiUserSettings) -> None:
    raster = settings.renderer.raster
    bev = settings.renderer.bev
    if raster.width <= 0 or raster.height <= 0:
        raise SettingsError("renderer.raster width and height must be positive")
    if bev.width <= 0 or bev.height <= 0 or bev.height_m <= 0:
        raise SettingsError("renderer.bev dimensions must be positive")
    presentation_resolution_wh(settings.presentation)
    if settings.runtime.total_blocks is not None and settings.runtime.total_blocks <= 0:
        raise SettingsError("runtime.total_blocks must be positive")
    if settings.runtime.prewarm_blocks < 0:
        raise SettingsError("runtime.prewarm_blocks must be non-negative")
    rules = settings.game.taxi.rules
    if rules.fare_min_route_distance_m > rules.fare_max_route_distance_m:
        raise SettingsError(
            "minimum fare distance must not exceed maximum fare distance"
        )
    if rules.min_time_s > rules.max_time_s:
        raise SettingsError("minimum fare time must not exceed maximum fare time")
    if rules.global_time_s <= 0:
        raise SettingsError("game.taxi.rules.global_time_s must be positive")
    try:
        settings.game.taxi.game_config(default_high_scores_path=Path())
    except ValueError as exc:
        raise SettingsError(f"game.taxi is invalid: {exc}") from exc


def _overlay_dataclass(
    base: Any,
    values: Mapping[str, object],
    path: SettingPath,
    *,
    base_dir: Path,
) -> Any:
    if not is_dataclass(base) or isinstance(base, type):
        raise SettingsError(f"{'.'.join(path) or 'settings'} is not configurable")
    known = {item.name: item for item, _annotation in iter_setting_fields(base, path)}
    deprecated = {
        item.name
        for item in fields(base)
        if (*path, item.name) in _DEPRECATED_SETTING_PATHS
    }
    values = {name: value for name, value in values.items() if name not in deprecated}
    unknown = sorted(set(values) - set(known))
    if unknown:
        context = ".".join(path) or "settings"
        raise SettingsError(f"{context} has unknown keys: {', '.join(unknown)}")
    hints = get_type_hints(type(base))
    updates = {}
    for name, raw in values.items():
        item = known[name]
        current = getattr(base, item.name)
        updates[item.name] = _convert_value(
            raw,
            hints.get(item.name, type(current)),
            current,
            (*path, name),
            base_dir=base_dir,
        )
    try:
        return replace(base, **updates)
    except (TypeError, ValueError) as exc:
        raise SettingsError(
            f"{'.'.join(path) or 'settings'} is invalid: {exc}"
        ) from exc


def _convert_value(
    raw: object,
    expected: Any,
    current: object,
    path: SettingPath,
    *,
    base_dir: Path,
) -> object:
    context = ".".join(path)
    origin = get_origin(expected)
    arguments = get_args(expected)
    if origin in (Union, types.UnionType):
        if raw is None and type(None) in arguments:
            return None
        errors = []
        for candidate in (item for item in arguments if item is not type(None)):
            try:
                return _convert_value(raw, candidate, current, path, base_dir=base_dir)
            except SettingsError as exc:
                errors.append(str(exc))
        raise SettingsError(errors[-1] if errors else f"{context} is invalid")
    if raw is None:
        raise SettingsError(f"{context} cannot be null")
    if is_dataclass(current) and not isinstance(current, type):
        if not isinstance(raw, Mapping):
            raise SettingsError(f"{context} must be a mapping")
        return _overlay_dataclass(
            current,
            cast(Mapping[str, object], raw),
            path,
            base_dir=base_dir,
        )
    if isinstance(expected, type) and is_dataclass(expected):
        if not isinstance(raw, Mapping) or not all(
            isinstance(name, str) for name in raw
        ):
            raise SettingsError(f"{context} must be a mapping with string keys")
        raw_values = {
            name: value
            for name, value in cast(Mapping[str, object], raw).items()
            if (*path, name) not in _DEPRECATED_SETTING_PATHS
        }
        configurable = {
            item.name: item
            for item in fields(expected)
            if item.init
            and item.name != "_target"
            and (*path, item.name) not in _NON_USER_SETTING_PATHS
            and (*path, item.name) not in _DEPRECATED_SETTING_PATHS
        }
        unknown = sorted(set(raw_values) - set(configurable))
        if unknown:
            raise SettingsError(f"{context} has unknown keys: {', '.join(unknown)}")
        missing = sorted(
            name
            for name, item in configurable.items()
            if name not in raw_values
            and item.default is MISSING
            and item.default_factory is MISSING
        )
        if missing:
            raise SettingsError(f"{context} is missing keys: {', '.join(missing)}")
        hints = get_type_hints(expected)
        updates: dict[str, object] = {}
        for name, value in raw_values.items():
            item = configurable[name]
            if item.default is not MISSING:
                default = item.default
            elif item.default_factory is not MISSING:
                default = item.default_factory()
            else:
                default = None
            updates[item.name] = _convert_value(
                value,
                hints.get(item.name, type(default)),
                default,
                (*path, name),
                base_dir=base_dir,
            )
        try:
            return cast(Any, expected)(**updates)
        except (TypeError, ValueError) as exc:
            raise SettingsError(f"{context} is invalid: {exc}") from exc
    if origin is Literal:
        if raw not in arguments:
            raise SettingsError(f"{context} must be one of {arguments}")
        return str(raw) if isinstance(raw, str) else raw
    if expected is Path or isinstance(current, Path):
        if not isinstance(raw, str):
            raise SettingsError(f"{context} must be a path string")
        candidate = Path(raw).expanduser()
        return (
            candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        )
    if expected is torch.dtype or isinstance(current, torch.dtype):
        if not isinstance(raw, str) or not hasattr(torch, raw.removeprefix("torch.")):
            raise SettingsError(f"{context} must name a torch dtype")
        value = getattr(torch, raw.removeprefix("torch."))
        if not isinstance(value, torch.dtype):
            raise SettingsError(f"{context} must name a torch dtype")
        return value
    if origin in (list, tuple) or isinstance(current, (list, tuple)):
        if not isinstance(raw, list):
            raise SettingsError(f"{context} must be a sequence")
        item_types = arguments
        if origin is tuple and len(item_types) == 2 and item_types[1] is Ellipsis:
            item_types = (item_types[0],) * len(raw)
        elif not item_types:
            item_types = (Any,) * len(raw)
        elif origin is list:
            item_types = (item_types[0],) * len(raw)
        elif len(item_types) != len(raw):
            raise SettingsError(f"{context} must contain {len(item_types)} values")
        current_values = (
            cast(Sequence[object], current)
            if isinstance(current, (list, tuple))
            else ()
        )
        converted = [
            _convert_value(
                value,
                item_types[index],
                current_values[index] if index < len(current_values) else None,
                (*path, str(index)),
                base_dir=base_dir,
            )
            for index, value in enumerate(raw)
        ]
        return (
            tuple(converted)
            if origin is tuple or isinstance(current, tuple)
            else converted
        )
    if origin in (dict, Mapping) or isinstance(current, Mapping):
        if not isinstance(raw, Mapping):
            raise SettingsError(f"{context} must be a mapping")
        key_type, value_type = arguments or (Any, Any)
        converted = {}
        for key, value in raw.items():
            existing = current.get(key) if isinstance(current, Mapping) else None
            converted_key = _convert_value(
                key,
                key_type,
                key if key_type is Any else None,
                (*path, "key"),
                base_dir=base_dir,
            )
            converted[converted_key] = _convert_value(
                value,
                value_type,
                existing,
                (*path, str(key)),
                base_dir=base_dir,
            )
        return converted
    if expected is Any:
        if current is None:
            return raw
        expected = type(current)
    if expected is bool:
        if not isinstance(raw, bool):
            raise SettingsError(f"{context} must be a boolean")
        return bool(raw)
    if expected is int:
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise SettingsError(f"{context} must be an integer")
        return int(raw)
    if expected is float:
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise SettingsError(f"{context} must be a number")
        return float(raw)
    if expected is str:
        if not isinstance(raw, str):
            raise SettingsError(f"{context} must be a string")
        return str(raw)
    if isinstance(expected, type) and issubclass(expected, Enum):
        try:
            return expected(raw)
        except ValueError as exc:
            raise SettingsError(f"{context} has invalid value {raw!r}") from exc
    raise SettingsError(f"{context} is read-only")


def replace_setting(root: Any, path: SettingPath, value: object) -> Any:
    """Immutably replace one value in a nested dataclass tree."""
    if not path:
        return value
    name, *remaining = path
    current = getattr(root, name)
    updated = replace_setting(current, tuple(remaining), value)
    return replace(root, **{name: updated})


def setting_value(root: Any, path: SettingPath) -> object:
    """Return one value from a nested settings path."""
    value = root
    for name in path:
        value = getattr(value, name)
    return value


def setting_choices(annotation: Any) -> tuple[object, ...]:
    """Return Literal choices for one field."""
    origin = get_origin(annotation)
    if origin is Literal:
        return get_args(annotation)
    if origin in (Union, types.UnionType):
        literal = next(
            (item for item in get_args(annotation) if get_origin(item) is Literal),
            None,
        )
        if literal is not None:
            return (None, *get_args(literal))
    return ()


def format_editor_value(value: object) -> str:
    """Format a scalar or sequence for the generic Options text editor."""
    serialized = _serialize_value(value)
    if serialized is _READ_ONLY:
        raise SettingsError("value is not configurable")
    if isinstance(serialized, (list, dict)):
        yaml = YAML(typ="safe")
        yaml.default_flow_style = True
        stream = io.StringIO()
        yaml.dump(serialized, stream)
        return stream.getvalue().strip()
    return "" if serialized is None else str(serialized)


def parse_editor_value(
    text: str,
    annotation: Any,
    current: object,
    path: SettingPath,
    *,
    base_dir: Path,
) -> object:
    """Parse one generic Options text field through the YAML type converter."""
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    optional = origin in (Union, types.UnionType) and type(None) in arguments
    if not text.strip() and optional:
        return None
    non_null = tuple(item for item in arguments if item is not type(None))
    scalar_type = non_null[0] if optional and len(non_null) == 1 else annotation
    if scalar_type is str or scalar_type is Path or isinstance(current, (str, Path)):
        raw: object = text
    else:
        try:
            raw = YAML(typ="safe").load(text)
        except Exception as exc:
            raise SettingsError(f"{'.'.join(path)} is invalid: {exc}") from exc
    return _convert_value(raw, annotation, current, path, base_dir=base_dir)


def _settings_diff(
    base: object,
    current: object,
    path: SettingPath = (),
) -> dict[str, object]:
    result: dict[str, object] = {}
    for item, _annotation in iter_setting_fields(current, path):
        before = getattr(base, item.name)
        after = getattr(current, item.name)
        item_path = (*path, item.name)
        if is_dataclass(after) and not isinstance(after, type):
            nested = _settings_diff(before, after, item_path)
            if nested:
                result[item.name] = nested
        elif after != before:
            serialized = _serialize_value(after)
            if serialized is not _READ_ONLY:
                result[item.name] = serialized
    return result


_READ_ONLY = object()


def _serialize_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        result = {}
        for item in fields(value):
            if item.name == "_target":
                continue
            serialized = _serialize_value(getattr(value, item.name))
            if serialized is _READ_ONLY:
                return _READ_ONLY
            result[item.name] = serialized
        return result
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            serialized_key = _serialize_value(key)
            serialized_item = _serialize_value(item)
            if serialized_key is _READ_ONLY or serialized_item is _READ_ONLY:
                return _READ_ONLY
            result[serialized_key] = serialized_item
        return result
    if isinstance(value, tuple):
        serialized = [_serialize_value(item) for item in value]
        return _READ_ONLY if _READ_ONLY in serialized else serialized
    if isinstance(value, list):
        serialized = [_serialize_value(item) for item in value]
        return _READ_ONLY if _READ_ONLY in serialized else serialized
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, type) or callable(value):
        return _READ_ONLY
    return _READ_ONLY


def _sync_mapping(target: CommentedMap, desired: Mapping[str, object]) -> None:
    for key in list(target):
        if key not in desired:
            del target[key]
    for key, value in desired.items():
        if isinstance(value, Mapping):
            existing = target.get(key)
            if not isinstance(existing, CommentedMap):
                existing = CommentedMap()
                target[key] = existing
            _sync_mapping(existing, cast(Mapping[str, object], value))
        else:
            target[key] = value


def clone_settings(settings: CrazyRobotaxiUserSettings) -> CrazyRobotaxiUserSettings:
    """Return an isolated Options draft."""
    return copy.deepcopy(settings)


def restart_required_settings(
    original: CrazyRobotaxiUserSettings,
    draft: CrazyRobotaxiUserSettings,
) -> tuple[str, ...]:
    """Return changed setting paths that apply after restart."""

    # ponytail: The current V2 host keeps menus and gameplay in one session.
    # Remove this policy when separate menu and gameplay sessions let saved
    # settings configure the next gameplay session directly.
    def changed_paths(
        before: object,
        after: object,
        prefix: tuple[str, ...],
    ) -> tuple[str, ...]:
        changed: list[str] = []
        for item, _annotation in iter_setting_fields(before, prefix):
            old_value = getattr(before, item.name)
            new_value = getattr(after, item.name)
            path = (*prefix, item.name)
            if is_dataclass(old_value) and not isinstance(old_value, type):
                changed.extend(changed_paths(old_value, new_value, path))
            elif old_value != new_value:
                changed.append(".".join(path))
        return tuple(changed)

    live_paths = {
        "presentation.width",
        "presentation.height",
        "presentation.hud_enabled",
        "presentation.show_fps",
        "presentation.show_current_prompt",
        "presentation.show_control_hints",
        "presentation.show_live_edit_buttons",
        "presentation.live_edit_mapping_location",
    }
    return tuple(
        path for path in changed_paths(original, draft, ()) if path not in live_paths
    )


def iter_setting_fields(
    value: object,
    path: SettingPath = (),
) -> tuple[tuple[Field[Any], Any], ...]:
    """Return user-authored fields at ``path`` within the app's settings tree."""
    if not is_dataclass(value) or isinstance(value, type):
        return ()
    hints = get_type_hints(type(value))
    return tuple(
        (item, hints.get(item.name, type(getattr(value, item.name))))
        for item in fields(value)
        if _is_user_setting_field(value, item, (*path, item.name), hints.get(item.name))
    )


def setting_description(value: object, item: Field[Any]) -> str | None:
    """Return the first paragraph of a field docstring as plain text."""
    description = get_field_docstring(type(value), item.name, ())
    if not description:
        return None
    summary = " ".join(description.split("\n\n", 1)[0].split())
    return re.sub(r":[a-z]+:`([^`]+)`", r"\1", summary).replace("`", "")


def options_documentation(settings: CrazyRobotaxiUserSettings) -> str:
    """Return a Markdown reference using the Options menu's fields and help."""
    lines = [
        "# Crazy Robotaxi options reference",
        "",
        "Each section follows the menu headings and uses the exact on-screen "
        "labels. An em dash in the CLI column means the setting is configured "
        "through Options or the user YAML file.",
        "",
    ]

    def section(value: object, path: SettingPath) -> None:
        heading = " → ".join(name.replace("_", " ").upper() for name in path)
        lines.extend([f"{'#' * min(len(path) + 1, 6)} {heading}", ""])
        leaves = []
        children = []
        for item, _annotation in iter_setting_fields(value, path):
            current = getattr(value, item.name)
            item_path = (*path, item.name)
            if is_dataclass(current) and not isinstance(current, type):
                children.append((current, item_path))
            else:
                label = f"**{item.name.replace('_', ' ').title()}:**"
                description = (setting_description(value, item) or "—").replace(
                    "|", "\\|"
                )
                declaring_type = next(
                    cls
                    for cls in reversed(type(value).__mro__)
                    if item.name in cls.__dict__.get("__annotations__", {})
                )
                package = declaring_type.__module__.split(".")[0]
                owner = {
                    "crazy_robotaxi": "Crazy Robotaxi",
                    "omnidreams_game_engine": "Game engine",
                    "omnidreams": "OmniDreams",
                    "flashdreams": "FlashDreams",
                }.get(package, package)
                yaml_key = ".".join(item_path)
                flags = SETTING_CLI_FLAGS.get(yaml_key)
                cli = f"`{flags}`" if flags else "—"
                leaves.append(
                    f"| {label} | `{yaml_key}` | {owner} | {cli} | {description} |"
                )
        if leaves:
            lines.extend(
                [
                    "| On-screen label | YAML key | Defined by | CLI flag | What it changes |",
                    "| --- | --- | --- | --- | --- |",
                    *leaves,
                    "",
                ]
            )
        for child, child_path in children:
            section(child, child_path)

    for item, _annotation in iter_setting_fields(settings):
        section(getattr(settings, item.name), (item.name,))
    return "\n".join(lines)


def _is_user_setting_field(
    value: object,
    item: Field[Any],
    path: SettingPath,
    annotation: Any,
) -> bool:
    if (
        item.name == "_target"
        or path in _NON_USER_SETTING_PATHS
        or path in _DEPRECATED_SETTING_PATHS
    ):
        return False
    alternatives = (
        get_args(annotation)
        if get_origin(annotation) in (Union, types.UnionType)
        else (annotation,)
    )
    if any(
        candidate is Callable or get_origin(candidate) is Callable
        for candidate in alternatives
    ):
        return False
    current = getattr(value, item.name)
    if isinstance(current, type) or callable(current):
        return False
    if is_dataclass(current) and not isinstance(current, type):
        return bool(iter_setting_fields(current, path))
    return _serialize_value(current) is not _READ_ONLY


__all__ = [
    "SETTING_CLI_FLAGS",
    "CrazyRobotaxiUserSettings",
    "LiveEditMappingLocation",
    "SettingsDocument",
    "SettingsError",
    "clone_settings",
    "default_config_path",
    "format_editor_value",
    "iter_setting_fields",
    "normalize_settings",
    "options_documentation",
    "parse_editor_value",
    "presentation_resolution_wh",
    "restart_required_settings",
    "setting_choices",
    "setting_description",
    "setting_value",
]
