# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified user-authored settings for Crazy Robotaxi."""

from __future__ import annotations

import copy
import io
import os
import tempfile
import types
from collections.abc import Mapping, Sequence
from dataclasses import Field, dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

import torch
from omnidreams_game_engine.config import BevConfig, RasterConfig
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from crazy_robotaxi.dynamics import TaxiVehicleConfig
from crazy_robotaxi.live_edit.config import LiveEditConfig
from crazy_robotaxi.rules import TaxiGameConfig

SettingPath = tuple[str, ...]


@dataclass(frozen=True)
class TaxiRulesSettings:
    """Taxi rules without session persistence or vehicle dynamics."""

    waypoint_spacing_m: float = 10.0
    pickup_grid_spacing_m: float = 60.0
    pickup_min_distance_m: float = 20.0
    initial_pickup_max_distance_m: float = 200.0
    pickup_radius_m: float = 5.0
    dropoff_radius_m: float = 6.0
    fare_min_route_distance_m: float = 200.0
    fare_max_route_distance_m: float = 250.0
    target_speed_mps: float = 10.0
    grace_s: float = 8.0
    min_time_s: float = 12.0
    max_time_s: float = 45.0
    trip_time_multiplier: float = 2.0
    base_fare_points: int = 500
    bonus_points_per_second: int = 100
    event_banner_s: float = 2.0
    global_time_s: float = 60.0
    dropoff_time_bonus_s: float = 30.0
    ground_snap_max_absolute_rotation_deg: float = 10.0
    ground_snap_settle_fraction: float = 0.25


@dataclass(frozen=True)
class TaxiSettings:
    """Taxi rules, vehicle behavior, and persistence."""

    seed: int | None = None
    high_scores_path: Path | None = None
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


@dataclass(frozen=True)
class GameEffectsSettings:
    """Game-directed presentation effects."""

    visual_flare: bool = False


@dataclass(frozen=True)
class GameSettings:
    """Complete gameplay configuration."""

    taxi: TaxiSettings = TaxiSettings()
    race: RaceSettings = RaceSettings()
    effects: GameEffectsSettings = GameEffectsSettings()


@dataclass(frozen=True)
class ModelSettings:
    """Runner-owned pipeline configuration and device placement."""

    device: str = "cuda"
    pipeline: Any = None


@dataclass(frozen=True)
class RendererSettings:
    """Primary semantic raster and top-down renderer settings."""

    raster: RasterConfig = RasterConfig()
    bev: BevConfig = BevConfig()


@dataclass(frozen=True)
class PresentationSettings:
    """Player-facing HUD settings."""

    hud_enabled: bool = True
    show_fps: bool = False
    show_control_hints: bool = True


@dataclass(frozen=True)
class RuntimeSettings:
    """Operational controls for one application session."""

    total_blocks: int | None = None
    prewarm_blocks: int = 8


@dataclass(frozen=True)
class DiagnosticsSettings:
    """Opt-in profiling and diagnostic output."""

    profile_pipeline: bool = False
    profile_input_latency: bool = False
    input_trace_path: Path | None = None
    alignment_directory: Path | None = None


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
        if type(version) is not int or version != 1:
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
        settings = _normalize_settings(settings)
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
        return replace_setting(settings, path, value)

    def save(self, settings: CrazyRobotaxiUserSettings) -> None:
        """Validate and atomically save sparse overrides while retaining comments."""
        settings = _normalize_settings(settings)
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


def _normalize_settings(
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
    if raster.near_plane_m >= raster.far_plane_m:
        raise SettingsError(
            "renderer.raster.near_plane_m must be less than far_plane_m"
        )
    if raster.fog_start_m >= raster.fog_end_m:
        raise SettingsError("renderer.raster.fog_start_m must be less than fog_end_m")
    if bev.width <= 0 or bev.height <= 0 or bev.height_m <= 0:
        raise SettingsError("renderer.bev dimensions must be positive")
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


def _overlay_dataclass(
    base: Any,
    values: Mapping[str, object],
    path: SettingPath,
    *,
    base_dir: Path,
) -> Any:
    if not is_dataclass(base) or isinstance(base, type):
        raise SettingsError(f"{'.'.join(path) or 'settings'} is not configurable")
    known = {item.name: item for item in fields(base) if item.name != "_target"}
    unknown = sorted(set(values) - set(known))
    if unknown:
        context = ".".join(path) or "settings"
        raise SettingsError(f"{context} has unknown keys: {', '.join(unknown)}")
    hints = get_type_hints(type(base))
    updates = {}
    for name, raw in values.items():
        current = getattr(base, name)
        updates[name] = _convert_value(
            raw,
            hints.get(name, type(current)),
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
    if origin is Literal:
        if raw not in arguments:
            raise SettingsError(f"{context} must be one of {arguments}")
        return raw
    if expected is Path or isinstance(current, Path):
        if not isinstance(raw, str):
            raise SettingsError(f"{context} must be a path string")
        candidate = Path(raw).expanduser()
        return (
            candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        )
    if isinstance(current, torch.dtype):
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
        current_values = cast(Sequence[object], current)
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
        return raw
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


def editable_setting(value: object, path: SettingPath) -> bool:
    """Return whether the generic UI can safely author ``value``."""
    if path[-1:] == ("_target",) or path == ("model", "pipeline", "name"):
        return False
    return (
        not isinstance(value, type)
        and not callable(value)
        and _serialize_value(value) is not _READ_ONLY
    )


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
        return readonly_display(value)
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


def _settings_diff(base: object, current: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in fields(cast(Any, current)):
        if item.name == "_target":
            continue
        before = getattr(base, item.name)
        after = getattr(current, item.name)
        if is_dataclass(after) and not isinstance(after, type):
            nested = _settings_diff(before, after)
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
        for item in fields(cast(Any, before)):
            old_value = getattr(before, item.name)
            new_value = getattr(after, item.name)
            path = (*prefix, item.name)
            if is_dataclass(old_value) and not isinstance(old_value, type):
                changed.extend(changed_paths(old_value, new_value, path))
            elif old_value != new_value:
                changed.append(".".join(path))
        return tuple(changed)

    return tuple(
        path
        for path in changed_paths(original, draft, ())
        if not path.startswith("presentation.")
    )


def iter_setting_fields(value: object) -> tuple[tuple[Field[Any], Any], ...]:
    """Return dataclass fields with resolved annotations in definition order."""
    if not is_dataclass(value) or isinstance(value, type):
        return ()
    hints = get_type_hints(type(value))
    return tuple(
        (item, hints.get(item.name, type(getattr(value, item.name))))
        for item in fields(value)
    )


def readonly_display(value: object) -> str:
    """Format structural settings for read-only UI display."""
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", repr(value))
        return f"{module}.{name}".lstrip(".")
    return str(value)


__all__ = [
    "CrazyRobotaxiUserSettings",
    "SettingsDocument",
    "SettingsError",
    "clone_settings",
    "default_config_path",
    "editable_setting",
    "format_editor_value",
    "iter_setting_fields",
    "parse_editor_value",
    "readonly_display",
    "restart_required_settings",
    "setting_choices",
    "setting_value",
]
