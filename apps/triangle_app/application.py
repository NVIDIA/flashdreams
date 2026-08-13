# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable triangle application contract and input behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeAlias

from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InputField,
    StepRequirements,
    UserInputCapability,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    REALTIME_SKIPPED_INPUTS_METADATA_KEY,
    DemoSpec,
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from flashdreams.runtime.interfaces import InferenceRuntime

DEFAULT_TRIANGLE_COLOR = (255, 160, 32)
TriangleOutputMode: TypeAlias = Literal[
    "mp4",
    "null",
    "webrtc",
    "local-window",
]
TRIANGLE_INPUT_MODES = ("replay", "realtime")
TRIANGLE_OUTPUT_MODES: tuple[TriangleOutputMode, ...] = (
    "mp4",
    "null",
    "webrtc",
    "local-window",
)
TRIANGLE_INPUT_SCHEMA = InferenceInputSchema(
    global_conditioning_fields=(
        InputField(name="width"),
        InputField(name="height"),
        InputField(name="fps"),
        InputField(name="total_frames"),
    ),
    step_fields=(InputField(name="color", input_modality="triangle/rgb"),),
)

_KEYBOARD_SCHEMA = UserInputSchema(
    capabilities=(
        UserInputCapability(
            event_type="key_down",
            input_modality="keyboard",
            payload_fields=frozenset({"key"}),
        ),
    )
)
_COLORS = (
    DEFAULT_TRIANGLE_COLOR,
    (255, 64, 64),
    (64, 255, 128),
    (64, 128, 255),
)
_COLOR_KEYS = {"r": _COLORS[1], "g": _COLORS[2], "b": _COLORS[3]}


@dataclass(frozen=True, slots=True)
class TriangleScenario:
    width: int
    height: int
    fps: int
    total_frames: int

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.fps, self.total_frames) <= 0:
            raise ValueError("Triangle scenario values must be positive.")


class TriangleApp(ABC):
    inference_input_schema = TRIANGLE_INPUT_SCHEMA
    canonical_input_schema = CanonicalInputSchema()

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime: ...

    def supported_input_modes(self) -> tuple[str, ...]:
        return TRIANGLE_INPUT_MODES

    def supported_output_modes(self) -> tuple[str, ...]:
        return TRIANGLE_OUTPUT_MODES

    def default_input_mapping(self) -> IdentityInputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Expected model_id={self.model_id!r}.")

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if spec.input_mode not in self.supported_input_modes():
            raise ValueError(f"Unsupported input mode: {spec.input_mode!r}.")
        if spec.output.mode not in self.supported_output_modes():
            raise ValueError(f"Unsupported output mode: {spec.output.mode!r}.")
        if not isinstance(spec.scenario, TriangleScenario):
            raise TypeError("Triangle application requires TriangleScenario.")
        scenario = spec.scenario
        return PreparedScenario(
            initial_inputs=InferenceInput(
                global_conditioning={
                    "width": scenario.width,
                    "height": scenario.height,
                    "fps": scenario.fps,
                    "total_frames": scenario.total_frames,
                }
            )
        )

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> TriangleInputProvider:
        return TriangleInputProvider(
            initial_inputs=scenario.initial_inputs,
            interactive=spec.input_mode == "realtime",
        )


class TriangleInputProvider:
    def __init__(
        self,
        *,
        initial_inputs: InferenceInput,
        interactive: bool,
    ) -> None:
        self._initial_inputs = initial_inputs
        self._color = DEFAULT_TRIANGLE_COLOR
        self.capabilities = ProviderCapabilities(
            supports_realtime_clock=True,
            supports_recorded_input=True,
            deterministic_given_inputs=True,
            user_input_schema=_KEYBOARD_SCHEMA if interactive else UserInputSchema(),
            inference_input_schema=TRIANGLE_INPUT_SCHEMA,
        )

    def prepare_initial_input(self) -> InferenceInput:
        return self._initial_inputs

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        del request
        skipped = user_window.metadata.get(REALTIME_SKIPPED_INPUTS_METADATA_KEY)
        if isinstance(skipped, UserInputs):
            self._apply_events(skipped)
        self._apply_events(user_window.inputs)
        return PreparedStep(inference_input=InferenceInput(step={"color": self._color}))

    def _apply_events(self, inputs: UserInputs) -> None:
        for event in inputs.events:
            if event.event_type != "key_down":
                continue
            key = str(event.payload.get("key", "")).lower()
            if key == "space":
                self._color = _COLORS[(_COLORS.index(self._color) + 1) % len(_COLORS)]
            elif key in _COLOR_KEYS:
                self._color = _COLOR_KEYS[key]

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if inputs is not None:
            self._initial_inputs = inputs
        self._color = DEFAULT_TRIANGLE_COLOR

    def close(self) -> None:
        return None


__all__ = [
    "DEFAULT_TRIANGLE_COLOR",
    "TRIANGLE_INPUT_MODES",
    "TRIANGLE_INPUT_SCHEMA",
    "TRIANGLE_OUTPUT_MODES",
    "TriangleApp",
    "TriangleInputProvider",
    "TriangleOutputMode",
    "TriangleScenario",
]
