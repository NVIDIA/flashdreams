# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot demo-specific scenario shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lingbot.runner import (
    EXAMPLE_DATA_AVAILABLE_IDXS,
    EXAMPLE_DATA_BASE_URL,
    EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS,
    ensure_example_data_downloaded,
    example_data_dirname,
)

DEFAULT_LINGBOT_PRESET = "lingbot-world-fast-taehv-window15-sink3"
LINGBOT_MODEL_ID = "lingbot"
DEFAULT_PIXEL_HEIGHT = 464
DEFAULT_PIXEL_WIDTH = 832
DEFAULT_FPS = 16


@dataclass(frozen=True, kw_only=True, slots=True)
class LingbotReplayScenario:
    """Resolved replay assets for the shared MP4 demo path."""

    prompt: str
    image_path: Path
    pose_path: Path
    intrinsic_path: Path
    total_blocks: int = 20
    pixel_height: int = DEFAULT_PIXEL_HEIGHT
    pixel_width: int = DEFAULT_PIXEL_WIDTH
    fps: int = DEFAULT_FPS

    def __post_init__(self) -> None:
        if self.total_blocks <= 0:
            raise ValueError("LingbotReplayScenario.total_blocks must be > 0.")
        if self.pixel_height <= 0 or self.pixel_width <= 0:
            raise ValueError("LingbotReplayScenario pixel dimensions must be > 0.")
        if self.fps <= 0:
            raise ValueError("LingbotReplayScenario.fps must be > 0.")
        object.__setattr__(self, "prompt", " ".join(self.prompt.split()))
        object.__setattr__(self, "image_path", Path(self.image_path))
        object.__setattr__(self, "pose_path", Path(self.pose_path))
        object.__setattr__(self, "intrinsic_path", Path(self.intrinsic_path))


@dataclass(frozen=True, kw_only=True, slots=True)
class LingbotWebRTCScenario:
    """Example-data and serving options for the shared WebRTC demo path."""

    example_idx: int = 0
    prefer_sw_encoder: bool = False

    def __post_init__(self) -> None:
        if self.example_idx not in EXAMPLE_DATA_AVAILABLE_IDXS:
            raise ValueError(
                "LingbotWebRTCScenario.example_idx must be one of "
                f"{EXAMPLE_DATA_AVAILABLE_IDXS}."
            )


def resolve_replay_scenario(
    value: Any,
    *,
    default_prompt: str = "",
) -> LingbotReplayScenario:
    """Normalize a user/demo scenario into a validated replay scenario."""
    if isinstance(value, LingbotReplayScenario):
        _require_existing_path(value.image_path, label="image_path")
        _require_existing_path(value.pose_path, label="pose_path")
        _require_existing_path(value.intrinsic_path, label="intrinsic_path")
        return value
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "Lingbot replay scenario must be a LingbotReplayScenario, "
            "a mapping, or None."
        )

    example_idx = int(value.get("example_idx", 0))
    if example_idx not in EXAMPLE_DATA_AVAILABLE_IDXS:
        raise ValueError(
            f"Lingbot replay example_idx must be one of {EXAMPLE_DATA_AVAILABLE_IDXS}."
        )

    image_path = _optional_path(value.get("image_path"))
    pose_path = _optional_path(value.get("pose_path"))
    intrinsic_path = _optional_path(value.get("intrinsic_path"))
    prompt_path = _optional_path(value.get("prompt_path"))
    example_data = _resolve_example_data_default(value)

    if example_data and (
        image_path is None
        or pose_path is None
        or intrinsic_path is None
        or (
            prompt_path is None
            and not _has_nonempty_value(value, "prompt")
            and example_idx in EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS
        )
    ):
        example_dir = ensure_example_data_downloaded(
            is_rank_zero=True,
            example_idx=example_idx,
        )
        image_path = image_path or example_dir / "image.jpg"
        pose_path = pose_path or example_dir / "poses.npy"
        intrinsic_path = intrinsic_path or example_dir / "intrinsics.npy"
        if (
            prompt_path is None
            and not _has_nonempty_value(value, "prompt")
            and example_idx in EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS
        ):
            prompt_path = example_dir / "prompt.txt"

    image_path = _require_path_value(image_path, label="image_path")
    pose_path = _require_path_value(pose_path, label="pose_path")
    intrinsic_path = _require_path_value(intrinsic_path, label="intrinsic_path")
    _require_existing_path(image_path, label="image_path")
    _require_existing_path(pose_path, label="pose_path")
    _require_existing_path(intrinsic_path, label="intrinsic_path")
    if prompt_path is not None:
        _require_existing_path(prompt_path, label="prompt_path")

    return LingbotReplayScenario(
        prompt=_resolve_prompt(
            value,
            prompt_path=prompt_path,
            default_prompt=default_prompt,
        ),
        image_path=image_path,
        pose_path=pose_path,
        intrinsic_path=intrinsic_path,
        total_blocks=int(value.get("total_blocks", 20)),
        pixel_height=int(value.get("pixel_height", DEFAULT_PIXEL_HEIGHT)),
        pixel_width=int(value.get("pixel_width", DEFAULT_PIXEL_WIDTH)),
        fps=int(value.get("fps", DEFAULT_FPS)),
    )


def resolve_webrtc_scenario(value: Any) -> LingbotWebRTCScenario:
    """Normalize a user/demo scenario into a WebRTC scenario."""
    if value is None:
        return LingbotWebRTCScenario()
    if isinstance(value, LingbotWebRTCScenario):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "Lingbot WebRTC scenario must be a LingbotWebRTCScenario, "
            "a mapping, or None."
        )
    return LingbotWebRTCScenario(
        example_idx=int(value.get("example_idx", 0)),
        prefer_sw_encoder=bool(value.get("prefer_sw_encoder", False)),
    )


def example_asset_urls(example_idx: int) -> dict[str, str]:
    """Return the canonical upstream asset URLs for a Lingbot example."""
    dirname = example_data_dirname(example_idx)
    return {
        "image": f"{EXAMPLE_DATA_BASE_URL}/{dirname}/image.jpg",
        "intrinsics": f"{EXAMPLE_DATA_BASE_URL}/{dirname}/intrinsics.npy",
        "poses": f"{EXAMPLE_DATA_BASE_URL}/{dirname}/poses.npy",
    }


def _resolve_prompt(
    value: Mapping[str, Any],
    *,
    prompt_path: Path | None,
    default_prompt: str,
) -> str:
    prompt = str(value.get("prompt", "")).strip()
    if prompt:
        return prompt
    if prompt_path is not None:
        lines = prompt_path.read_text(encoding="utf-8").splitlines()
        if lines:
            prompt = lines[0].strip()
            if prompt:
                return prompt
    return default_prompt.strip()


def _resolve_example_data_default(value: Mapping[str, Any]) -> bool:
    explicit = value.get("example_data")
    if explicit is not None:
        return _bool_value(explicit)
    return not (
        _has_nonempty_value(value, "image_path")
        and _has_nonempty_value(value, "pose_path")
        and _has_nonempty_value(value, "intrinsic_path")
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


def _optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def _require_path_value(value: Path | None, *, label: str) -> Path:
    if value is None:
        raise ValueError(f"Lingbot replay scenario requires {label}.")
    return value


def _require_existing_path(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Lingbot replay scenario missing {label}: {path}"
        )


__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_LINGBOT_PRESET",
    "DEFAULT_PIXEL_HEIGHT",
    "DEFAULT_PIXEL_WIDTH",
    "LINGBOT_MODEL_ID",
    "LingbotReplayScenario",
    "LingbotWebRTCScenario",
    "example_asset_urls",
    "resolve_replay_scenario",
    "resolve_webrtc_scenario",
]
