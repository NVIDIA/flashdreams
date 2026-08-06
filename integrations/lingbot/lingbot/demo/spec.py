# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot demo-specific input shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lingbot.example_data import (
    EXAMPLE_DATA_AVAILABLE_IDXS,
    EXAMPLE_DATA_BASE_URL,
    EXAMPLE_DATA_DIR_LOCAL,
    EXAMPLE_DATA_FILENAMES,
    EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS,
    example_asset_urls,
    example_data_dirname,
)
from lingbot.runtime import (
    DEFAULT_FPS,
    DEFAULT_LINGBOT_PRESET,
    DEFAULT_PIXEL_HEIGHT,
    DEFAULT_PIXEL_WIDTH,
    LINGBOT_MODEL_ID,
    LingbotReplayInputs,
    replay_inputs_from_mapping,
)


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


def resolve_replay_inputs(
    value: Any,
    *,
    default_prompt: str = "",
    is_rank_zero: bool = True,
) -> LingbotReplayInputs:
    """Normalize a user/demo scenario into direct Lingbot runtime inputs."""
    return replay_inputs_from_mapping(
        value,
        default_prompt=default_prompt,
        is_rank_zero=is_rank_zero,
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


__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_LINGBOT_PRESET",
    "DEFAULT_PIXEL_HEIGHT",
    "DEFAULT_PIXEL_WIDTH",
    "EXAMPLE_DATA_AVAILABLE_IDXS",
    "EXAMPLE_DATA_BASE_URL",
    "EXAMPLE_DATA_DIR_LOCAL",
    "EXAMPLE_DATA_FILENAMES",
    "EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS",
    "LINGBOT_MODEL_ID",
    "LingbotReplayInputs",
    "LingbotWebRTCScenario",
    "example_asset_urls",
    "example_data_dirname",
    "resolve_replay_inputs",
    "resolve_webrtc_scenario",
]
