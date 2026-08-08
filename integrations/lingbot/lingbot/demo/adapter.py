# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot adapter for the shared demo API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from flashdreams.runtime import (
    InputCanonicalizer,
    UserInputCapability,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    PreparedScenario,
)
from flashdreams.runtime.interfaces import InferenceRuntime
from lingbot.input_mapping import (
    KeyboardToCameraCommand,
    TextEventSelection,
)
from lingbot.runtime import (
    LingbotModelAdapter,
    LingbotReplayRuntime,
    PipelineFactory,
    inference_input_from_replay_inputs,
)

from .spec import (
    resolve_replay_inputs,
    resolve_text_event_prompts,
    resolve_user_input_events,
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
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("mp4",)

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if spec.input_mode != "replay":
            raise ValueError(
                "Lingbot prepare_scenario currently supports only "
                f"input_mode='replay', got {spec.input_mode!r}."
            )
        if not isinstance(spec.output, Mp4OutputSpec):
            raise ValueError("Lingbot replay demo currently requires MP4 output.")

        replay_inputs = resolve_replay_inputs(
            spec.scenario,
            default_prompt=self.default_replay_prompt(spec.config),
        )
        text_event_prompts = resolve_text_event_prompts(spec.scenario)
        user_inputs = resolve_user_input_events(spec.scenario)
        if _camera_source(spec.scenario) == "events":
            # Live control still needs the scenario's calibration, so the trace
            # is loaded for its intrinsics and world scale and then discarded
            # as a trajectory source.
            trace = self.create_input_mapping(replay_inputs).camera_trace
            mapping = self.create_live_input_mapping(
                fps=replay_inputs.fps,
                base_intrinsics=trace.intrinsics[0],
                # A trace's world scale is derived from how far its poses
                # travel, so a stationary example yields 0. Live control has no
                # trajectory to normalize against, so it falls back to the same
                # unit scale the live runtime uses.
                world_scale=trace.world_scale or 1.0,
                prompt=replay_inputs.prompt,
                text_event_prompts=text_event_prompts,
            )
        else:
            mapping = self.create_input_mapping(
                replay_inputs,
                text_event_prompts=text_event_prompts,
            )
        return PreparedScenario(
            initial_inputs=inference_input_from_replay_inputs(replay_inputs),
            user_inputs=user_inputs,
            source_schema=_source_schema(user_inputs),
            canonicalizer=_canonicalizer(text_event_prompts),
            mapping=mapping,
            metadata={
                "model_id": self.model_id,
                "preset_id": self.preset_id(spec.config),
            },
        )


def _camera_source(scenario: Any) -> str:
    if isinstance(scenario, Mapping):
        return str(scenario.get("camera_source", "trace"))
    return "trace"


_KEY_EVENT_TYPES = frozenset({"key_down", "key_up"})

_KEYBOARD_CAPABILITIES = (
    UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"})),
    UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"})),
)

_TEXT_EVENT_CAPABILITY = UserInputCapability(
    event_type="text_event",
    payload_fields=frozenset({"event_id"}),
)


def _source_schema(user_inputs: UserInputs) -> UserInputSchema:
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
    if observed & _KEY_EVENT_TYPES:
        capabilities.extend(_KEYBOARD_CAPABILITIES)
    if "text_event" in observed:
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


def _canonicalizer(text_event_prompts: Mapping[str, str] | None) -> InputCanonicalizer:
    converters: list[Any] = [KeyboardToCameraCommand()]
    if text_event_prompts:
        converters.append(TextEventSelection())
    return InputCanonicalizer(converters)


__all__ = [
    "LingbotDemoAdapter",
    "ReplayRuntimeFactory",
]
