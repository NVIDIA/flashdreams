# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams demo-specific scenario shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnidreams.runner import (
    DEFAULT_EXAMPLE_DATA_UUID_1V,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _ensure_hf_single_view_example_data_synced,
    _example_camera_names,
)
from omnidreams.scenes import SCENE_VARIANT_DEFAULT
from omnidreams.webrtc.session import DEFAULT_WEBRTC_SCENE_UUID

DEFAULT_OMNIDREAMS_PRESET = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
OMNIDREAMS_MODEL_ID = "omnidreams"


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
class OmnidreamsWebRTCScenario:
    """Scene/options for the shared WebRTC demo path."""

    scene_dir: Path | None = None
    scene_uuid: str | None = DEFAULT_WEBRTC_SCENE_UUID
    scene_variant: str = SCENE_VARIANT_DEFAULT
    camera_name: str = "camera_front_wide_120fov"
    debug_serve_hdmaps: bool = False
    postprocess_preset: str = ""
    prefer_sw_encoder: bool = False

    def __post_init__(self) -> None:
        if self.scene_dir is not None:
            object.__setattr__(self, "scene_dir", Path(self.scene_dir))
        if not self.scene_variant.strip():
            raise ValueError("OmnidreamsWebRTCScenario.scene_variant is required.")
        if not self.camera_name.strip():
            raise ValueError("OmnidreamsWebRTCScenario.camera_name is required.")


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
        scene_uuid=value.get("scene_uuid", DEFAULT_WEBRTC_SCENE_UUID),
        scene_variant=str(value.get("scene_variant", SCENE_VARIANT_DEFAULT)),
        camera_name=str(value.get("camera_name", "camera_front_wide_120fov")),
        debug_serve_hdmaps=bool(value.get("debug_serve_hdmaps", False)),
        postprocess_preset=str(value.get("postprocess_preset", "")),
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
    "OMNIDREAMS_MODEL_ID",
    "OmnidreamsReplayScenario",
    "OmnidreamsWebRTCScenario",
    "resolve_replay_scenario",
    "resolve_webrtc_scenario",
]
