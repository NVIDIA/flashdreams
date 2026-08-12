# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams demo-specific scenario shapes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from omnidreams.runner import (
    DEFAULT_EXAMPLE_DATA_UUID_1V,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _ensure_hf_single_view_example_data_synced,
    _example_camera_names,
)
from omnidreams.scenes import SCENE_VARIANT_DEFAULT

DEFAULT_OMNIDREAMS_PRESET = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
OMNIDREAMS_MODEL_ID = "omnidreams"
DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
OMNIDREAMS_CONDITIONING_PRECOMPUTED = "precomputed-hdmap"
OMNIDREAMS_CONDITIONING_LUDUS = "ludus-scene-driving"
OMNIDREAMS_CONDITIONING_MODES = (
    OMNIDREAMS_CONDITIONING_PRECOMPUTED,
    OMNIDREAMS_CONDITIONING_LUDUS,
)
LudusBackendName: TypeAlias = Literal["cuda", "vulkan"]

_KEY_EVENT_ALIASES = {
    "down": "keydown",
    "key_down": "keydown",
    "keyup": "keyup",
    "up": "keyup",
    "key_up": "keyup",
}


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsKeyboardTraceEvent:
    """One recorded keyboard edge in a finite Ludus replay trace."""

    timestamp_s: float
    event: str
    key: str

    def __post_init__(self) -> None:
        timestamp_s = float(self.timestamp_s)
        if not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError(
                "OmnidreamsKeyboardTraceEvent.timestamp_s must be finite and >= 0."
            )
        key = str(self.key).strip().lower()
        if not key:
            raise ValueError("OmnidreamsKeyboardTraceEvent.key must be non-empty.")
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "event", _normalize_key_event_name(self.event))
        object.__setattr__(self, "key", key)


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsReplayScenario:
    """Resolved replay assets for the shared MP4 demo path."""

    prompts: tuple[str, ...]
    hdmap_video_paths: tuple[Path, ...]
    first_frame_paths: tuple[Path, ...]
    camera_names: tuple[str, ...]
    total_blocks: int = 60
    pixel_height: int = DEFAULT_VIDEO_HEIGHT
    pixel_width: int = DEFAULT_VIDEO_WIDTH
    fps: int = 30

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("OmnidreamsReplayScenario.prompts must be non-empty.")
        num_views = len(self.prompts)
        for name, values in (
            ("hdmap_video_paths", self.hdmap_video_paths),
            ("first_frame_paths", self.first_frame_paths),
            ("camera_names", self.camera_names),
        ):
            if len(values) != num_views:
                raise ValueError(
                    f"OmnidreamsReplayScenario.{name} has {len(values)} "
                    f"entries but prompts has {num_views}."
                )
        if self.total_blocks <= 0:
            raise ValueError("OmnidreamsReplayScenario.total_blocks must be > 0.")
        if self.pixel_height <= 0 or self.pixel_width <= 0:
            raise ValueError("OmnidreamsReplayScenario pixel dimensions must be > 0.")
        if self.fps <= 0:
            raise ValueError("OmnidreamsReplayScenario.fps must be > 0.")
        object.__setattr__(
            self,
            "hdmap_video_paths",
            tuple(Path(path) for path in self.hdmap_video_paths),
        )
        object.__setattr__(
            self,
            "first_frame_paths",
            tuple(Path(path) for path in self.first_frame_paths),
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsLudusReplayScenario:
    """Resolved Ludus scene plus a finite recorded keyboard trace."""

    keyboard_events: tuple[OmnidreamsKeyboardTraceEvent, ...]
    scene_path: Path | None = None
    scene_dir: Path | None = None
    scene_uuid: str | None = DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID
    scene_variant: str = SCENE_VARIANT_DEFAULT
    camera_name: str = "camera_front_wide_120fov"
    prompt: str | None = None
    total_blocks: int = 60
    pixel_height: int = DEFAULT_VIDEO_HEIGHT
    pixel_width: int = DEFAULT_VIDEO_WIDTH
    fps: int = 30
    move_speed_per_s: float = 6.0
    rotate_speed_rad_per_s: float = math.radians(35.0)
    ludus_backend: LudusBackendName = "cuda"

    @property
    def camera_names(self) -> tuple[str, ...]:
        return (self.camera_name,)

    @property
    def prompts(self) -> tuple[str, ...]:
        return () if self.prompt is None else (self.prompt,)

    def __post_init__(self) -> None:
        if self.scene_path is not None:
            object.__setattr__(self, "scene_path", Path(self.scene_path))
        if self.scene_dir is not None:
            object.__setattr__(self, "scene_dir", Path(self.scene_dir))
        if not (self.scene_path or self.scene_dir or self.scene_uuid):
            raise ValueError(
                "OmnidreamsLudusReplayScenario requires scene_path, "
                "scene_dir, or scene_uuid."
            )
        if not self.scene_variant.strip():
            raise ValueError("OmnidreamsLudusReplayScenario.scene_variant is required.")
        if not self.camera_name.strip():
            raise ValueError("OmnidreamsLudusReplayScenario.camera_name is required.")
        if self.total_blocks <= 0:
            raise ValueError("OmnidreamsLudusReplayScenario.total_blocks must be > 0.")
        if self.pixel_height <= 0 or self.pixel_width <= 0:
            raise ValueError(
                "OmnidreamsLudusReplayScenario pixel dimensions must be > 0."
            )
        if self.fps <= 0:
            raise ValueError("OmnidreamsLudusReplayScenario.fps must be > 0.")
        if self.move_speed_per_s <= 0:
            raise ValueError(
                "OmnidreamsLudusReplayScenario.move_speed_per_s must be > 0."
            )
        if self.rotate_speed_rad_per_s <= 0:
            raise ValueError(
                "OmnidreamsLudusReplayScenario.rotate_speed_rad_per_s must be > 0."
            )
        object.__setattr__(
            self,
            "ludus_backend",
            _ludus_backend_name(self.ludus_backend),
        )
        previous_timestamp_s = -math.inf
        normalized_events: list[OmnidreamsKeyboardTraceEvent] = []
        for event in self.keyboard_events:
            normalized = (
                event
                if isinstance(event, OmnidreamsKeyboardTraceEvent)
                else _keyboard_trace_event(event)
            )
            if normalized.timestamp_s < previous_timestamp_s:
                raise ValueError(
                    "OmnidreamsLudusReplayScenario.keyboard_events must be sorted "
                    "by non-decreasing timestamp_s."
                )
            previous_timestamp_s = normalized.timestamp_s
            normalized_events.append(normalized)
        object.__setattr__(self, "keyboard_events", tuple(normalized_events))


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsWebRTCScenario:
    """Scene/options for the shared WebRTC demo path."""

    scene_dir: Path | None = None
    scene_uuid: str | None = DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID
    scene_variant: str = SCENE_VARIANT_DEFAULT
    camera_name: str = "camera_front_wide_120fov"
    debug_serve_hdmaps: bool = False
    prefer_sw_encoder: bool = False

    def __post_init__(self) -> None:
        if self.scene_dir is not None:
            object.__setattr__(self, "scene_dir", Path(self.scene_dir))
        if not self.scene_variant.strip():
            raise ValueError("OmnidreamsWebRTCScenario.scene_variant is required.")
        if not self.camera_name.strip():
            raise ValueError("OmnidreamsWebRTCScenario.camera_name is required.")


def conditioning_mode_from_scenario(value: Any) -> str:
    """Return the resolved OmniDreams replay conditioning mode."""
    if isinstance(value, OmnidreamsLudusReplayScenario):
        return OMNIDREAMS_CONDITIONING_LUDUS
    if isinstance(value, OmnidreamsReplayScenario):
        return OMNIDREAMS_CONDITIONING_PRECOMPUTED
    if value is None or not isinstance(value, Mapping):
        return OMNIDREAMS_CONDITIONING_PRECOMPUTED

    mode = (
        str(value.get("conditioning_mode", OMNIDREAMS_CONDITIONING_PRECOMPUTED))
        .strip()
        .lower()
    )
    if mode in {"precomputed", "hdmap", "precomputed-hdmaps"}:
        mode = OMNIDREAMS_CONDITIONING_PRECOMPUTED
    if mode in {"ludus", "keyboard-driving", "ludus-keyboard"}:
        mode = OMNIDREAMS_CONDITIONING_LUDUS
    if mode not in OMNIDREAMS_CONDITIONING_MODES:
        supported = ", ".join(OMNIDREAMS_CONDITIONING_MODES)
        raise ValueError(
            f"Unsupported OmniDreams conditioning_mode={mode!r}. "
            f"Supported modes: {supported}."
        )
    return mode


def resolve_replay_scenario(
    value: Any,
    *,
    default_prompt: str = "",
) -> OmnidreamsReplayScenario:
    """Normalize a user/demo scenario into a validated replay scenario."""
    if isinstance(value, OmnidreamsReplayScenario):
        _require_existing_paths(value.hdmap_video_paths, label="hdmap_video_paths")
        _require_existing_paths(value.first_frame_paths, label="first_frame_paths")
        return value
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "OmniDreams replay scenario must be an OmnidreamsReplayScenario "
            "a mapping, or None."
        )

    hdmap_paths = _path_tuple(value.get("hdmap_video_paths", ()))
    first_paths = _path_tuple(value.get("first_frame_paths", ()))
    example_data = _resolve_example_data_default(value)
    if example_data and (not hdmap_paths or not first_paths):
        example_hdmaps, example_first_frames = (
            _ensure_hf_single_view_example_data_synced(
                str(value.get("example_data_uuid", DEFAULT_EXAMPLE_DATA_UUID_1V))
            )
        )
        if not hdmap_paths:
            hdmap_paths = example_hdmaps
        if not first_paths:
            first_paths = example_first_frames

    _require_existing_paths(hdmap_paths, label="hdmap_video_paths")
    _require_existing_paths(first_paths, label="first_frame_paths")
    if len(hdmap_paths) != len(first_paths):
        raise ValueError(
            "OmniDreams replay scenario requires one HDMap video and first "
            "frame per view."
        )

    num_views = len(hdmap_paths)
    prompts = _resolve_prompts(value, num_views, default_prompt=default_prompt)
    camera_names = _string_tuple(value.get("camera_names", ()))
    if not camera_names:
        camera_names = (
            _example_camera_names(num_views)
            if example_data
            else tuple(f"view_{i}" for i in range(num_views))
        )

    return OmnidreamsReplayScenario(
        prompts=prompts,
        hdmap_video_paths=hdmap_paths,
        first_frame_paths=first_paths,
        camera_names=camera_names,
        total_blocks=int(value.get("total_blocks", 60)),
        pixel_height=int(value.get("pixel_height", DEFAULT_VIDEO_HEIGHT)),
        pixel_width=int(value.get("pixel_width", DEFAULT_VIDEO_WIDTH)),
        fps=int(value.get("fps", 30)),
    )


def resolve_ludus_replay_scenario(value: Any) -> OmnidreamsLudusReplayScenario:
    """Normalize a user/demo scenario into a Ludus recorded-trace scenario."""
    if isinstance(value, OmnidreamsLudusReplayScenario):
        _require_optional_existing_path(value.scene_path, label="scene_path")
        return value
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "OmniDreams Ludus replay scenario must be an "
            "OmnidreamsLudusReplayScenario, a mapping, or None."
        )
    return OmnidreamsLudusReplayScenario(
        keyboard_events=_keyboard_trace_events(value),
        scene_path=_optional_path(value.get("scene_path")),
        scene_dir=_optional_path(value.get("scene_dir")),
        scene_uuid=_optional_string(
            value.get("scene_uuid", DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID)
        ),
        scene_variant=str(value.get("scene_variant", SCENE_VARIANT_DEFAULT)),
        camera_name=str(value.get("camera_name", "camera_front_wide_120fov")),
        prompt=_optional_string(value.get("prompt")),
        total_blocks=int(value.get("total_blocks", 60)),
        pixel_height=int(value.get("pixel_height", DEFAULT_VIDEO_HEIGHT)),
        pixel_width=int(value.get("pixel_width", DEFAULT_VIDEO_WIDTH)),
        fps=int(value.get("fps", 30)),
        move_speed_per_s=float(value.get("move_speed_per_s", 6.0)),
        rotate_speed_rad_per_s=float(
            value.get("rotate_speed_rad_per_s", math.radians(35.0))
        ),
        ludus_backend=_ludus_backend_name(value.get("ludus_backend", "cuda")),
    )


def resolve_webrtc_scenario(value: Any) -> OmnidreamsWebRTCScenario:
    """Normalize a user/demo scenario into a WebRTC scenario."""
    if value is None:
        return OmnidreamsWebRTCScenario()
    if isinstance(value, OmnidreamsWebRTCScenario):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "OmniDreams WebRTC scenario must be an OmnidreamsWebRTCScenario, "
            "a mapping, or None."
        )
    scene_dir = value.get("scene_dir")
    return OmnidreamsWebRTCScenario(
        scene_dir=Path(scene_dir) if scene_dir is not None else None,
        scene_uuid=value.get("scene_uuid", DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID),
        scene_variant=str(value.get("scene_variant", SCENE_VARIANT_DEFAULT)),
        camera_name=str(value.get("camera_name", "camera_front_wide_120fov")),
        debug_serve_hdmaps=bool(value.get("debug_serve_hdmaps", False)),
        prefer_sw_encoder=bool(value.get("prefer_sw_encoder", False)),
    )


def _resolve_prompts(
    value: Mapping[str, Any],
    num_views: int,
    *,
    default_prompt: str,
) -> tuple[str, ...]:
    prompts = _string_tuple(value.get("prompts", ()))
    if prompts:
        if len(prompts) != num_views:
            raise ValueError(
                f"OmniDreams replay prompts has {len(prompts)} entries but "
                f"there are {num_views} views."
            )
        return prompts
    prompt = str(value.get("prompt", "")).strip()
    if not prompt:
        prompt = default_prompt.strip()
    if not prompt:
        raise ValueError("OmniDreams replay scenario requires prompt or prompts.")
    return (prompt,) * num_views


def _resolve_example_data_default(value: Mapping[str, Any]) -> bool:
    explicit = value.get("example_data")
    if explicit is not None:
        return _bool_value(explicit)
    return not (
        _has_nonempty_value(value, "hdmap_video_paths")
        or _has_nonempty_value(value, "first_frame_paths")
    )


def _keyboard_trace_events(
    value: Mapping[str, Any],
) -> tuple[OmnidreamsKeyboardTraceEvent, ...]:
    events_value = value.get("keyboard_events")
    if events_value is None:
        trace_path = _optional_path(value.get("keyboard_trace_path"))
        if trace_path is None:
            return ()
        if not trace_path.exists():
            raise FileNotFoundError(
                f"OmniDreams keyboard_trace_path missing: {trace_path}"
            )
        loaded = json.loads(trace_path.read_text(encoding="utf-8"))
        events_value = (
            loaded.get("events", ()) if isinstance(loaded, Mapping) else loaded
        )
    if isinstance(events_value, (str, bytes)) or not isinstance(events_value, Sequence):
        raise TypeError("OmniDreams keyboard trace must be a sequence of events.")
    return tuple(_keyboard_trace_event(event) for event in events_value)


def _keyboard_trace_event(value: Any) -> OmnidreamsKeyboardTraceEvent:
    if isinstance(value, OmnidreamsKeyboardTraceEvent):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "OmniDreams keyboard trace events must be mappings or "
            "OmnidreamsKeyboardTraceEvent instances."
        )
    timestamp = _first_present(value, ("timestamp_s", "time_s", "timestamp", "t"))
    if timestamp is None:
        raise ValueError("OmniDreams keyboard trace event missing timestamp_s.")
    event = _first_present(value, ("event", "event_type", "type"))
    if event is None:
        raise ValueError("OmniDreams keyboard trace event missing event.")
    key = value.get("key")
    if key is None:
        raise ValueError("OmniDreams keyboard trace event missing key.")
    return OmnidreamsKeyboardTraceEvent(
        timestamp_s=float(timestamp),
        event=str(event),
        key=str(key),
    )


def _first_present(value: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _normalize_key_event_name(value: str) -> str:
    event = str(value).strip().lower()
    event = _KEY_EVENT_ALIASES.get(event, event)
    if event not in {"keydown", "keyup"}:
        raise ValueError(
            "OmniDreams keyboard trace event must be 'keydown' or 'keyup'."
        )
    return event


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _has_nonempty_value(value: Mapping[str, Any], key: str) -> bool:
    if key not in value:
        return False
    raw = value[key]
    if raw is None or raw == "":
        return False
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return len(raw) > 0
    return True


def _path_tuple(value: Any) -> tuple[Path, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (str, Path)):
        return (Path(value),)
    if isinstance(value, Sequence):
        return tuple(Path(path) for path in value)
    raise TypeError(f"Expected path or path sequence, got {type(value).__name__}.")


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise TypeError(f"Expected string or string sequence, got {type(value).__name__}.")


def _optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ludus_backend_name(value: Any) -> LudusBackendName:
    backend = str(value).strip().lower()
    if backend not in {"cuda", "vulkan"}:
        raise ValueError(
            "OmnidreamsLudusReplayScenario.ludus_backend must be 'cuda' or 'vulkan'."
        )
    return cast(LudusBackendName, backend)


def _require_optional_existing_path(path: Path | None, *, label: str) -> None:
    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(
            f"OmniDreams Ludus replay scenario missing {label}: {path}"
        )


def _require_existing_paths(paths: tuple[Path, ...], *, label: str) -> None:
    if not paths:
        raise ValueError(f"OmniDreams replay scenario requires {label}.")
    missing = tuple(path for path in paths if not path.exists())
    if missing:
        raise FileNotFoundError(
            f"OmniDreams replay scenario missing {label}: "
            + ", ".join(str(path) for path in missing)
        )


__all__ = [
    "DEFAULT_OMNIDREAMS_PRESET",
    "DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID",
    "OMNIDREAMS_CONDITIONING_LUDUS",
    "OMNIDREAMS_CONDITIONING_MODES",
    "OMNIDREAMS_CONDITIONING_PRECOMPUTED",
    "OMNIDREAMS_MODEL_ID",
    "OmnidreamsKeyboardTraceEvent",
    "OmnidreamsLudusReplayScenario",
    "OmnidreamsReplayScenario",
    "OmnidreamsWebRTCScenario",
    "conditioning_mode_from_scenario",
    "resolve_ludus_replay_scenario",
    "resolve_replay_scenario",
    "resolve_webrtc_scenario",
]
