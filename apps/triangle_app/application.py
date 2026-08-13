# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable triangle application contract and input behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast, final

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InputField,
    StepRequest,
    StepRequirements,
    StepResult,
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
    WebRTCAppResources,
)
from flashdreams.runtime.interfaces import InferenceRuntime
from flashdreams.serving.native_window.runner import NativePresenter
from torch import Tensor

DEFAULT_TRIANGLE_COLOR = (255, 160, 32)
_INPUT_SCHEMA = InferenceInputSchema(
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


@dataclass(frozen=True, slots=True)
class TriangleInferenceRequest:
    scenario: TriangleScenario
    step_index: int
    color: tuple[int, int, int]


class TriangleApp(ABC):
    inference_input_schema = _INPUT_SCHEMA
    canonical_input_schema = CanonicalInputSchema()
    output_layout: VideoTensorLayout = "tchw"
    supported_control_keys = frozenset({"r", "g", "b", "space"})
    native_presenter_factory: Callable[..., NativePresenter] | None = None
    native_key_bindings: Mapping[str, Sequence[str]] | None = None

    def __init__(
        self,
        *,
        application_name: str,
        description: str,
        width: int,
        height: int,
        fps: int,
        total_frames: int,
        title: str,
        device: str = "cpu",
        default_io_handler: str = "local-window",
    ) -> None:
        self.application_name = application_name
        self.description = description
        self.scenario = TriangleScenario(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
        )
        self.fps = fps
        self.video_width = width
        self.video_height = height
        self.title = title
        self.default_io_handler = default_io_handler
        self.config = InferenceConfig(model_id=self.model_id, device=device)
        self._runtime_loaded = False
        self._runtime_closed = False

    @property
    def webrtc_app_resources(self) -> WebRTCAppResources:
        return WebRTCAppResources(preload_name=self.application_name)

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def load_checkpoint(self, config: InferenceConfig) -> None: ...

    @abstractmethod
    def run_inference(self, request: TriangleInferenceRequest) -> Tensor: ...

    def unload_checkpoint(self) -> None:
        return None

    @final
    def initialize(self, config: InferenceConfig) -> None:
        self.validate_config(config)
        if self._runtime_closed:
            raise RuntimeError("Cannot reload a closed triangle runtime.")
        if not self._runtime_loaded:
            self.load_checkpoint(config)
            self._runtime_loaded = True

    @final
    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.initialize(config)
        return self

    @final
    def create_session(self, launch_args: object) -> _TriangleSession:
        if not self._runtime_loaded or self._runtime_closed:
            raise RuntimeError("Triangle runtime is not available.")
        if isinstance(launch_args, TriangleScenario):
            scenario = launch_args
        elif isinstance(launch_args, InferenceInput):
            scenario = _scenario_from_inputs(launch_args)
        else:
            raise TypeError("Triangle launch arguments must describe a scenario.")
        return _TriangleSession(self, scenario)

    @final
    def start_session(self, inputs: InferenceInput) -> _TriangleSession:
        return self.create_session(inputs)

    @final
    def close(self) -> None:
        if self._runtime_loaded:
            self.unload_checkpoint()
            self._runtime_loaded = False
        self._runtime_closed = True

    def default_input_mapping(self) -> IdentityInputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Expected model_id={self.model_id!r}.")

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
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
            interactive=bool(spec.metadata.get("realtime", False)),
        )


class _TriangleSession:
    def __init__(self, model: TriangleApp, scenario: TriangleScenario) -> None:
        self._model = model
        self._scenario = scenario
        self._step_index = 0
        self._closed = False
        self._color = DEFAULT_TRIANGLE_COLOR

    def next_event(self) -> StepRequirements | None:
        if self._closed or self._step_index >= self._scenario.total_frames:
            return None
        return StepRequirements(
            step_index=self._step_index,
            metadata={"input_frame_count": 1},
        )

    def next_step_request(self) -> StepRequest | None:
        event = self.next_event()
        if event is None:
            return None
        return StepRequest(
            step_index=event.step_index,
            metadata=event.metadata,
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        if self._closed:
            raise RuntimeError("Cannot step a closed triangle session.")
        if self._step_index >= self._scenario.total_frames:
            raise RuntimeError("Triangle session is complete.")
        self._color = _color(inputs.step.get("color", DEFAULT_TRIANGLE_COLOR))
        return self._generate(self._step_index)

    def generate(
        self,
        event: StepRequirements,
        user_input: UserInputs,
    ) -> StepResult:
        if event.step_index != self._step_index:
            raise ValueError(
                f"Expected triangle step {self._step_index}, got {event.step_index}."
            )
        self._color = _apply_color_events(self._color, user_input)
        return self._generate(event.step_index)

    def _generate(self, step_index: int) -> StepResult:
        if self._closed:
            raise RuntimeError("Cannot generate from a closed triangle session.")
        frame = self._model.run_inference(
            TriangleInferenceRequest(
                scenario=self._scenario,
                step_index=step_index,
                color=self._color,
            )
        )
        expected_shape = (3, self._scenario.height, self._scenario.width)
        if tuple(frame.shape) != expected_shape:
            raise ValueError(
                f"Triangle model returned shape {tuple(frame.shape)}, "
                f"expected {expected_shape}."
            )
        self._step_index += 1
        return StepResult.from_video_chunk(
            step_index=step_index,
            video_chunk=frame.unsqueeze(0),
            layout="tchw",
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if inputs is not None:
            self._scenario = _scenario_from_inputs(inputs)
        self._step_index = 0
        self._closed = False
        self._color = DEFAULT_TRIANGLE_COLOR

    def close(self) -> None:
        self._closed = True


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
            inference_input_schema=_INPUT_SCHEMA,
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
        self._color = _apply_color_events(self._color, inputs)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if inputs is not None:
            self._initial_inputs = inputs
        self._color = DEFAULT_TRIANGLE_COLOR

    def close(self) -> None:
        return None


def _scenario_from_inputs(inputs: InferenceInput) -> TriangleScenario:
    values = inputs.global_conditioning
    return TriangleScenario(
        width=int(values["width"]),
        height=int(values["height"]),
        fps=int(values["fps"]),
        total_frames=int(values["total_frames"]),
    )


def _apply_color_events(
    color: tuple[int, int, int],
    inputs: UserInputs,
) -> tuple[int, int, int]:
    for event in inputs.events:
        if event.event_type != "key_down":
            continue
        key = str(event.payload.get("key", "")).lower()
        if key == "space":
            color = _COLORS[(_COLORS.index(color) + 1) % len(_COLORS)]
        elif key in _COLOR_KEYS:
            color = _COLOR_KEYS[key]
    return color


def _color(value: object) -> tuple[int, int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(type(channel) is not int for channel in value)
    ):
        raise TypeError("Triangle color must contain three integer channels.")
    color = cast(tuple[int, int, int], value)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError("Triangle color channels must be in [0, 255].")
    return color


__all__ = [
    "DEFAULT_TRIANGLE_COLOR",
    "TriangleApp",
    "TriangleInferenceRequest",
    "TriangleInputProvider",
    "TriangleScenario",
]
