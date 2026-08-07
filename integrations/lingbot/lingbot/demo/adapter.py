# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plug-compatible local-window adapter for LingBot World."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import (
    DRIVER_COMMAND,
    RGB_VIDEO,
    CanonicalInputs,
    CanonicalInputSchema,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceOutputSchema,
    InputCanonicalizer,
    InputField,
    InputMappingSchema,
    StepRequest,
    StepResult,
    TimeWindow,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    DemoRoute,
    DemoSpec,
    LocalWindowOutputSpec,
    PreparedSession,
)
from flashdreams.runtime.interfaces import InferenceRuntime

from lingbot.webrtc.session import (
    LingbotInferenceRuntime,
    LingbotRuntimeConfig,
    LingbotSessionInput,
)

LINGBOT_MODEL_ID = "lingbot"

LINGBOT_DRIVING_INPUT_SCHEMA = InferenceInputSchema(
    global_conditioning_fields=(InputField(name="lingbot_scenario"),),
    step_fields=(
        InputField(
            name="driver_command",
            input_modality=DRIVER_COMMAND.name,
        ),
    ),
)


@dataclass(frozen=True, kw_only=True, slots=True)
class LingbotDrivingScenario:
    """Prompt and optional first-frame settings for a LingBot session."""

    prompt: str | None = None
    first_frame_image_url: str | None = None


class LingbotDriverCommandMapping:
    """Map normalized driving intent to LingBot camera-control input."""

    mapping_schema = InputMappingSchema(
        name="lingbot-driver-command",
        consumes=(DRIVER_COMMAND,),
        produces_global_conditioning=(InputField(name="lingbot_scenario"),),
        produces_step=(
            InputField(
                name="driver_command",
                input_modality=DRIVER_COMMAND.name,
            ),
        ),
    )

    def validate(self, **_: Any) -> None:
        return

    def map_global_conditioning_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
    ) -> InferenceInput:
        del canonical_inputs
        return inference_input

    def map_step_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
        request: StepRequest,
    ) -> InferenceInput:
        del request
        command = canonical_inputs.values.get(
            DRIVER_COMMAND.name,
            {
                "throttle": 0.0,
                "brake": 0.0,
                "steer": 0.0,
                "stop": False,
                "reverse": False,
                "steer_is_direct": False,
                "manual_control": False,
            },
        )
        if not isinstance(command, Mapping):
            raise TypeError("driver_command canonical value must be a mapping.")
        return InferenceInput(
            global_conditioning=inference_input.global_conditioning,
            step={"driver_command": dict(command)},
            metadata=inference_input.metadata,
        )


class LingbotStandardRuntime:
    """Synchronous standard-runtime wrapper over LingBot's async runtime."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        runtime_config: LingbotRuntimeConfig,
        runtime_factory: Callable[
            [LingbotRuntimeConfig], Any
        ] = LingbotInferenceRuntime,
    ) -> None:
        self.config = config
        self._runtime = runtime_factory(runtime_config)
        asyncio.run(self._runtime.initialize())
        self._fps = runtime_config.fps
        self._closed = False

    def start_session(self, inputs: InferenceInput) -> LingbotStandardSession:
        if self._closed:
            raise RuntimeError("LingbotStandardRuntime is closed.")
        scenario = inputs.global_conditioning.get("lingbot_scenario")
        if not isinstance(scenario, LingbotDrivingScenario):
            raise TypeError(
                "LingBot runtime requires LingbotDrivingScenario conditioning."
            )
        asyncio.run(
            self._runtime.reset_for_new_session(
                LingbotSessionInput(
                    prompt=scenario.prompt,
                    first_frame_image_url=scenario.first_frame_image_url,
                )
            )
        )
        return LingbotStandardSession(runtime=self._runtime, fps=self._fps)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        asyncio.run(self._runtime.close())


class LingbotStandardSession:
    """One driver-command-controlled LingBot rollout."""

    def __init__(self, *, runtime: Any, fps: int) -> None:
        self._runtime = runtime
        self._fps = fps
        self._step_index = 0
        self._elapsed_s = 0.0
        self._closed = False

    def next_step_request(self) -> StepRequest | None:
        if self._closed:
            return None
        num_frames = int(self._runtime.peek_next_chunk_num_frames())
        duration_s = num_frames / self._fps
        return StepRequest(
            step_index=self._step_index,
            inference_input_schema=LINGBOT_DRIVING_INPUT_SCHEMA,
            user_input_window=TimeWindow(
                start_s=self._elapsed_s,
                end_s=self._elapsed_s + duration_s,
            ),
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        LINGBOT_DRIVING_INPUT_SCHEMA.require_step(inputs)
        command = inputs.step["driver_command"]
        if not isinstance(command, Mapping):
            raise TypeError("driver_command step input must be a mapping.")
        num_frames = int(self._runtime.peek_next_chunk_num_frames())
        duration_s = num_frames / self._fps
        start_s = self._elapsed_s
        end_s = start_s + duration_s
        keys = _keys_from_driver_command(command)
        frame_times = [start_s + (index + 1) / self._fps for index in range(num_frames)]
        result = asyncio.run(
            self._runtime.generate_chunk(
                segments=[(start_s, end_s, keys)],
                frame_times=frame_times,
            )
        )
        output = StepResult(
            step_index=self._step_index,
            output=result,
            frame_count=result.num_frames,
            output_window=TimeWindow(start_s=start_s, end_s=end_s),
            metrics=result.stats or {},
        )
        self._step_index += 1
        self._elapsed_s = end_s
        return output

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        asyncio.run(self._runtime.reset_for_new_session())
        self._step_index = 0
        self._elapsed_s = 0.0

    def close(self) -> None:
        self._closed = True


class LingbotDemoAdapter:
    """LingBot model adapter for plug-compatible driving demos."""

    model_id = LINGBOT_MODEL_ID
    inference_input_schema = LINGBOT_DRIVING_INPUT_SCHEMA
    inference_output_schema = InferenceOutputSchema(
        modality=RGB_VIDEO,
        python_type=VideoStepResult,
        layouts=frozenset({"bvtchw"}),
    )
    canonical_input_schema = CanonicalInputSchema(modalities=(DRIVER_COMMAND,))

    def __init__(
        self,
        *,
        runtime_factory: Callable[
            [LingbotRuntimeConfig], Any
        ] = LingbotInferenceRuntime,
    ) -> None:
        self._mapping = LingbotDriverCommandMapping()
        self._runtime_factory = runtime_factory

    def supported_routes(self) -> tuple[DemoRoute, ...]:
        return (
            DemoRoute(
                input_mode="keyboard-driving",
                output_mode="local-window",
            ),
        )

    def default_input_mapping(self) -> LingbotDriverCommandMapping:
        return self._mapping

    def prepare_session(self, spec: DemoSpec) -> PreparedSession:
        value = spec.scenario
        if value is None:
            scenario = LingbotDrivingScenario()
        elif isinstance(value, LingbotDrivingScenario):
            scenario = value
        elif isinstance(value, Mapping):
            scenario = LingbotDrivingScenario(
                prompt=str(value["prompt"])
                if value.get("prompt") is not None
                else None,
                first_frame_image_url=(
                    str(value["first_frame_image_url"])
                    if value.get("first_frame_image_url") is not None
                    else None
                ),
            )
        else:
            raise TypeError("LingBot driving scenario must be a mapping or None.")
        return PreparedSession(
            initial_inputs=InferenceInput(
                global_conditioning={"lingbot_scenario": scenario}
            ),
            inference_input_schema=LINGBOT_DRIVING_INPUT_SCHEMA,
            source_schema=UserInputSchema(description="live interactive-drive input"),
            canonicalizer=InputCanonicalizer(),
            mapping=self._mapping,
        )

    def list_sessions(self, spec: DemoSpec) -> tuple[DemoSpec, ...]:
        return (spec,)

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"LingBot adapter requires model_id={self.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        return self._create_runtime(config=config, output=LocalWindowOutputSpec())

    def create_demo_runtime(self, spec: DemoSpec) -> InferenceRuntime:
        if not isinstance(spec.output, LocalWindowOutputSpec):
            raise TypeError("LingBot driving requires LocalWindowOutputSpec.")
        config = spec.config
        if config is None:
            raise RuntimeError("DemoSpec.config was not initialized.")
        return self._create_runtime(config=config, output=spec.output)

    def _create_runtime(
        self,
        *,
        config: InferenceConfig,
        output: LocalWindowOutputSpec,
    ) -> LingbotStandardRuntime:
        self.validate_config(config)
        return LingbotStandardRuntime(
            config=config,
            runtime_config=LingbotRuntimeConfig(
                config_name=config.preset_id
                or "lingbot-world-fast-taehv-window15-sink3",
                device=config.device or "cuda:0",
                video_width=output.width,
                video_height=output.height,
            ),
            runtime_factory=self._runtime_factory,
        )


def _keys_from_driver_command(command: Mapping[str, Any]) -> frozenset[str]:
    if bool(command.get("stop", False)):
        return frozenset()
    keys: set[str] = set()
    if float(command.get("throttle", 0.0)) > 0.01:
        keys.add("w")
    if float(command.get("brake", 0.0)) > 0.01 or bool(command.get("reverse", False)):
        keys.add("s")
    steer = float(command.get("steer", 0.0))
    if steer > 0.01:
        keys.add("a")
    elif steer < -0.01:
        keys.add("d")
    return frozenset(keys)


__all__ = [
    "LINGBOT_DRIVING_INPUT_SCHEMA",
    "LINGBOT_MODEL_ID",
    "LingbotDemoAdapter",
    "LingbotDriverCommandMapping",
    "LingbotDrivingScenario",
    "LingbotStandardRuntime",
    "LingbotStandardSession",
]
