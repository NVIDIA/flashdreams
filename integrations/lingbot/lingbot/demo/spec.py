# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot demo-specific input shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flashdreams.runtime import UserInputEvent, UserInputs
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


def resolve_text_event_prompts(value: Any) -> dict[str, str]:
    """Return the scenario's text-event catalog as ``{event_id: prompt}``."""
    if not isinstance(value, Mapping):
        return {}
    catalog = value.get("text_events")
    if not catalog:
        return {}
    if isinstance(catalog, Mapping):
        return {str(key): str(prompt) for key, prompt in catalog.items()}
    prompts: dict[str, str] = {}
    for entry in catalog:
        event_id = getattr(entry, "event_id", None)
        prompt = getattr(entry, "prompt", None)
        if event_id is None and isinstance(entry, Mapping):
            event_id = entry.get("event_id")
            prompt = entry.get("prompt")
        if event_id is None:
            raise ValueError("Lingbot text events require an 'event_id'.")
        prompts[str(event_id)] = "" if prompt is None else str(prompt)
    return prompts


def resolve_user_input_events(value: Any) -> UserInputs:
    """Normalize a scenario's recorded event trace into :class:`UserInputs`.

    Each record is ``{"t": seconds, "type": event_type, ...payload}``, which
    maps one-to-one onto ``UserInputEvent``. Events are sorted by timestamp
    because ``UserInputs`` requires non-decreasing order.
    """
    if not isinstance(value, Mapping):
        return UserInputs()
    records = value.get("events")
    if not records:
        return UserInputs()

    events: list[UserInputEvent] = []
    for record in records:
        if isinstance(record, UserInputEvent):
            events.append(record)
            continue
        if not isinstance(record, Mapping):
            raise TypeError(
                "Lingbot scenario events must be UserInputEvent objects or mappings."
            )
        payload = {
            key: item
            for key, item in record.items()
            if key not in {"t", "timestamp_s", "type", "event_type", "source"}
        }
        timestamp_s = record.get("t", record.get("timestamp_s"))
        event_type = record.get("type", record.get("event_type"))
        if timestamp_s is None or event_type is None:
            raise ValueError(
                "Lingbot scenario events require a timestamp ('t') and a type ('type')."
            )
        events.append(
            UserInputEvent(
                timestamp_s=float(timestamp_s),
                event_type=str(event_type),
                payload=payload,
                source=record.get("source"),
            )
        )
    events.sort(key=lambda event: event.timestamp_s)
    return UserInputs(events=tuple(events))


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
    "resolve_text_event_prompts",
    "resolve_user_input_events",
    "resolve_webrtc_scenario",
]
