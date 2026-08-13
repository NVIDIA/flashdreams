# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot adapter for the shared demo API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from flashdreams.runtime import (
    UserInputCapability,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    BATCH_INPUT_FPS_METADATA_KEY,
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    PreparedScenario,
    WebRTCOutputSpec,
)
from flashdreams.runtime.interfaces import InferenceRuntime
from lingbot.runtime import (
    FIELD_FPS,
    FIELD_PIXEL_HEIGHT,
    FIELD_PIXEL_WIDTH,
    FIELD_TOTAL_BLOCKS,
    LingbotModelAdapter,
    LingbotReplayRuntime,
    PipelineFactory,
    inference_input_from_replay_inputs,
)

from .providers import (
    PROVIDER_INPUTS_METADATA_KEY,
    LingbotInputProvider,
    create_lingbot_provider_inputs,
)
from .spec import (
    resolve_replay_inputs,
    resolve_text_event_prompts,
    resolve_user_input_events,
    resolve_webrtc_scenario,
)

ReplayRuntimeFactory = Callable[..., InferenceRuntime]


class LingbotDemoAdapter(LingbotModelAdapter):
    """Model-owned Lingbot adapter consumed by shared demo launchers."""

    def __init__(
        self,
        *,
        replay_runtime_factory: ReplayRuntimeFactory = LingbotReplayRuntime,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        super().__init__(
            runtime_factory=replay_runtime_factory,
            pipeline_factory=pipeline_factory,
        )

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay", "keyboard-driving")

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("mp4", "null", "webrtc")

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if spec.input_mode == "replay":
            if not isinstance(spec.output, (Mp4OutputSpec, NullOutputSpec)):
                raise ValueError("Lingbot replay demo requires MP4 or null output.")
            scenario = spec.scenario
            live_camera = False
        elif spec.input_mode == "keyboard-driving":
            if not isinstance(spec.output, WebRTCOutputSpec):
                raise ValueError(
                    "Lingbot keyboard-driving demo requires WebRTC output."
                )
            scenario = _keyboard_driving_scenario(spec, output=spec.output)
            live_camera = True
        else:
            raise ValueError(
                "Lingbot prepare_scenario supports input_mode='replay' or "
                f"'keyboard-driving', got {spec.input_mode!r}."
            )

        replay_inputs = resolve_replay_inputs(
            scenario,
            default_prompt=_default_prompt(self, spec),
        )
        text_event_prompts = resolve_text_event_prompts(scenario)
        user_inputs = resolve_user_input_events(scenario)
        provider_inputs = create_lingbot_provider_inputs(
            replay_inputs,
            live_camera=live_camera or _camera_source(scenario) == "events",
            text_event_prompts=text_event_prompts,
        )
        return PreparedScenario(
            initial_inputs=inference_input_from_replay_inputs(replay_inputs),
            user_inputs=user_inputs,
            source_schema=_source_schema(
                user_inputs,
                include_keyboard=live_camera,
                include_text_events=live_camera and bool(text_event_prompts),
            ),
            metadata={
                BATCH_INPUT_FPS_METADATA_KEY: replay_inputs.fps,
                "model_id": self.model_id,
                "preset_id": self.preset_id(spec.config),
                PROVIDER_INPUTS_METADATA_KEY: provider_inputs,
            },
        )

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> LingbotInputProvider:
        del spec
        return LingbotInputProvider(
            scenario=scenario,
            inference_input_schema=self.inference_input_schema,
        )


def _camera_source(scenario: Any) -> str:
    if isinstance(scenario, Mapping):
        return str(scenario.get("camera_source", "trace"))
    return "trace"


def _keyboard_driving_scenario(
    spec: DemoSpec,
    *,
    output: WebRTCOutputSpec,
) -> Mapping[str, Any]:
    webrtc_scenario = resolve_webrtc_scenario(spec.scenario)
    scenario: dict[str, Any] = (
        dict(spec.scenario) if isinstance(spec.scenario, Mapping) else {}
    )
    scenario.setdefault("camera_source", "events")
    scenario.setdefault("example_data", True)
    scenario.setdefault("example_idx", webrtc_scenario.example_idx)
    scenario.setdefault(FIELD_TOTAL_BLOCKS, _total_blocks_default(spec))
    scenario.setdefault(FIELD_PIXEL_HEIGHT, output.video_height)
    scenario.setdefault(FIELD_PIXEL_WIDTH, output.video_width)
    scenario.setdefault(FIELD_FPS, output.fps)
    config = spec.config
    if config is not None and "text_events" not in scenario:
        text_events = config.runtime_options.get("text_events")
        if text_events is not None:
            scenario["text_events"] = text_events
    return scenario


def _total_blocks_default(spec: DemoSpec) -> int:
    config = spec.config
    if config is not None:
        total_blocks = config.runtime_options.get("total_blocks")
        if total_blocks is not None:
            return int(total_blocks)
    return 1_000_000


def _default_prompt(adapter: LingbotDemoAdapter, spec: DemoSpec) -> str:
    config = spec.config
    if config is not None:
        default_prompt = config.runtime_options.get("default_prompt")
        if default_prompt is not None:
            return str(default_prompt)
    return adapter.default_replay_prompt(config)


_KEY_EVENT_TYPES = frozenset({"key_down", "key_up"})

_KEYBOARD_CAPABILITIES = (
    UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"})),
    UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"})),
)

_TEXT_EVENT_CAPABILITY = UserInputCapability(
    event_type="text_event",
    payload_fields=frozenset({"event_id"}),
)


def _source_schema(
    user_inputs: UserInputs,
    *,
    include_keyboard: bool = False,
    include_text_events: bool = False,
) -> UserInputSchema:
    """Declare what this scenario's event source can provide.

    Capabilities describe the source, not the particular trace. A keyboard
    source is declared to provide both key edges even if one recording happens
    to contain no ``key_up`` -- a key held for the whole run is a normal trace.
    Declaring only the observed types would fail the keyboard converter's
    consumed set, and ``converters_for`` would silently drop it, leaving the run
    with no camera control.
    """
    observed = {event.event_type for event in user_inputs.events}
    capabilities: list[UserInputCapability] = []
    if include_keyboard or observed & _KEY_EVENT_TYPES:
        capabilities.extend(_KEYBOARD_CAPABILITIES)
    if include_text_events or "text_event" in observed:
        capabilities.append(_TEXT_EVENT_CAPABILITY)
    for event_type in sorted(observed - _KEY_EVENT_TYPES - {"text_event"}):
        payload_fields: frozenset[str] = frozenset()
        for event in user_inputs.events:
            if event.event_type == event_type:
                payload_fields = frozenset(event.payload)
                break
        capabilities.append(
            UserInputCapability(
                event_type=event_type,
                payload_fields=payload_fields,
            )
        )
    return UserInputSchema(
        capabilities=tuple(capabilities),
        description=(
            "Lingbot replay event trace"
            if capabilities
            else "fixed Lingbot replay input"
        ),
    )


__all__ = [
    "LingbotDemoAdapter",
    "ReplayRuntimeFactory",
]
