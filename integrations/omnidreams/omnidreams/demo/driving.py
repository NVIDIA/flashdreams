# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Standard runtime/session adapter for interactive OmniDreams driving."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import (
    DRIVER_COMMAND,
    CanonicalInputs,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InputField,
    InputMappingSchema,
    StepRequest,
    StepResult,
    TimeWindow,
)
from flashdreams.serving.presentation.frame import as_rgb_host_uint8
from omnidreams.interactive_drive.backends.base import RenderBackend
from omnidreams.interactive_drive.config import AppConfig
from omnidreams.interactive_drive.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
    state_from_initial_pose,
)
from omnidreams.interactive_drive.simulation.ground_snap import GroundSnapper
from omnidreams.interactive_drive.simulation.map_bounds import MapBounds
from omnidreams.interactive_drive.types import SceneBundle
from omnidreams.interactive_drive.video_model.local import LocalVideoModelAdapter

DRIVING_INPUT_SCHEMA = InferenceInputSchema(
    global_conditioning_fields=(InputField(name="driving_scenario"),),
    step_fields=(
        InputField(
            name="driver_command",
            input_modality=DRIVER_COMMAND.name,
        ),
    ),
)
"""Model-facing schema for one interactive driving session."""


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsDrivingScenario:
    """Resolved scene and simulation state for one driving session."""

    app_config: AppConfig
    scene: SceneBundle
    map_bounds: MapBounds | None
    ground_snapper: GroundSnapper | None


class OmnidreamsDriverCommandMapping:
    """Map canonical driver intent into OmniDreams session inputs."""

    mapping_schema = InputMappingSchema(
        name="omnidreams-driver-command",
        consumes=(DRIVER_COMMAND,),
        produces_global_conditioning=(InputField(name="driving_scenario"),),
        produces_step=(
            InputField(
                name="driver_command",
                input_modality=DRIVER_COMMAND.name,
            ),
        ),
    )

    def validate(
        self,
        *,
        canonical_schema=None,
        inference_input_schema=None,
    ) -> None:
        del canonical_schema
        schema = inference_input_schema or DRIVING_INPUT_SCHEMA
        if schema.missing_step(
            InferenceInput(step={"driver_command": _idle_driver_command()})
        ):
            raise ValueError("Driving schema does not accept driver_command.")

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
            _idle_driver_command(),
        )
        if not isinstance(command, Mapping):
            raise TypeError("driver_command canonical value must be a mapping.")
        return InferenceInput(
            global_conditioning=inference_input.global_conditioning,
            step={"driver_command": dict(command)},
            metadata=inference_input.metadata,
        )


class OmnidreamsDrivingRuntime:
    """Reusable warmed render backend for sequential driving scenes."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        app_config: AppConfig,
        backend: RenderBackend,
    ) -> None:
        self.config = config
        self.app_config = app_config
        self._backend = backend
        self._video = LocalVideoModelAdapter(backend)
        self._video.warmup_model()
        self._closed = False

    def start_session(self, inputs: InferenceInput) -> OmnidreamsDrivingSession:
        if self._closed:
            raise RuntimeError("OmnidreamsDrivingRuntime is closed.")
        scenario = inputs.global_conditioning.get("driving_scenario")
        if not isinstance(scenario, OmnidreamsDrivingScenario):
            raise TypeError(
                "Driving runtime requires OmnidreamsDrivingScenario global "
                "conditioning."
            )
        self._video.load_scene(scenario.scene)
        return OmnidreamsDrivingSession(video=self._video, scenario=scenario)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._backend.close()


class OmnidreamsDrivingSession:
    """One interactive scene rollout over a warmed OmniDreams backend."""

    def __init__(
        self,
        *,
        video: LocalVideoModelAdapter,
        scenario: OmnidreamsDrivingScenario,
    ) -> None:
        self._video = video
        self._scenario = scenario
        self._step_index = 0
        self._elapsed_s = 0.0
        self._closed = False
        self._simulation = self._new_simulation()

    def next_step_request(self) -> StepRequest | None:
        if self._closed:
            return None
        chunk_size = self._chunk_size()
        duration_s = chunk_size * self._scenario.app_config.chunk.frame_interval_s
        return StepRequest(
            step_index=self._step_index,
            inference_input_schema=DRIVING_INPUT_SCHEMA,
            user_input_window=TimeWindow(
                start_s=self._elapsed_s,
                end_s=self._elapsed_s + duration_s,
            ),
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        DRIVING_INPUT_SCHEMA.require_step(inputs)
        raw_command = inputs.step["driver_command"]
        if not isinstance(raw_command, Mapping):
            raise TypeError("driver_command step input must be a mapping.")
        command = _DriverCommand.from_mapping(raw_command)
        chunk_size = self._chunk_size()
        window_start_s = self._elapsed_s
        trajectory = self._simulation.pose_chunk(
            cast(Any, command),
            chunk_size,
            self._scenario.app_config.chunk.frame_interval_s,
            0.0,
        )
        render_start = time.perf_counter()
        frame_chunk = self._video.render_chunk(trajectory)
        render_duration_s = time.perf_counter() - render_start
        self._elapsed_s += chunk_size * self._scenario.app_config.chunk.frame_interval_s

        video_result = _video_result(
            frame_chunk=frame_chunk,
            command=command,
            chunk_index=self._step_index,
        )
        result = StepResult(
            step_index=self._step_index,
            output=video_result,
            frame_count=video_result.num_frames,
            output_window=TimeWindow(
                start_s=window_start_s,
                end_s=self._elapsed_s,
            ),
            metrics={"render_s": render_duration_s},
        )
        self._step_index += 1
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._video.reset()
        self._step_index = 0
        self._elapsed_s = 0.0
        self._simulation = self._new_simulation()

    def close(self) -> None:
        self._closed = True

    def _chunk_size(self) -> int:
        chunk = self._scenario.app_config.chunk
        return (
            chunk.initial_chunk_frames if self._step_index == 0 else chunk.chunk_frames
        )

    def _new_simulation(self) -> EgoVehicleKinematics:
        scene = self._scenario.scene
        config = self._scenario.app_config
        return EgoVehicleKinematics(
            initial_state=state_from_initial_pose(
                initial_rig_to_world=scene.initial_rig_to_world,
                initial_yaw_rad=scene.initial_yaw_rad,
                initial_speed_mps=10.0,
            ),
            vehicle_config=config.vehicle,
            ground_snapper=self._scenario.ground_snapper,
            initial_timestamp_us=scene.initial_timestamp_us,
            map_bounds=self._scenario.map_bounds,
            oob_margin_m=config.oob_margin_m,
            oob_warning_zone_m=config.oob_warning_zone_m,
        )


@dataclass(frozen=True, slots=True)
class _DriverCommand:
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    stop: bool = False
    reverse: bool = False
    steer_is_direct: bool = False
    manual_control: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> _DriverCommand:
        return cls(
            throttle=float(value.get("throttle", 0.0)),
            brake=float(value.get("brake", 0.0)),
            steer=float(value.get("steer", 0.0)),
            stop=bool(value.get("stop", False)),
            reverse=bool(value.get("reverse", False)),
            steer_is_direct=bool(value.get("steer_is_direct", False)),
            manual_control=bool(value.get("manual_control", False)),
        )


def _idle_driver_command() -> dict[str, float | bool]:
    return {
        "throttle": 0.0,
        "brake": 0.0,
        "steer": 0.0,
        "stop": False,
        "reverse": False,
        "steer_is_direct": False,
        "manual_control": False,
    }


def _video_result(
    *,
    frame_chunk,
    command: _DriverCommand,
    chunk_index: int,
) -> VideoStepResult:
    frames = frame_chunk.frames
    rgb_frames = [
        as_rgb_host_uint8(
            frame.model_rgb_host_uint8
            if frame.model_rgb_host_uint8 is not None
            else frame.rgb_host_uint8
        )
        for frame in frames
    ]
    video = torch.from_numpy(np.stack(rgb_frames, axis=0)).permute(0, 3, 1, 2)
    boundary = frame_chunk.boundary_state_after_chunk
    bev = frames[-1].bev_host_uint8 if frames else None
    status = frames[-1].status_message if frames else None
    return VideoStepResult.from_video_chunk(
        chunk_index=chunk_index,
        video_chunk=video.unsqueeze(0).unsqueeze(0),
        layout="bvtchw",
        metadata={
            "interactive_drive": {
                "speed_mps": boundary.speed_mps,
                "steering": command.steer,
                "throttle": command.throttle,
                "brake": command.brake,
                "reverse": command.reverse,
                "bev": bev,
                "status_message": status,
            }
        },
    )


__all__ = [
    "DRIVING_INPUT_SCHEMA",
    "OmnidreamsDriverCommandMapping",
    "OmnidreamsDrivingRuntime",
    "OmnidreamsDrivingScenario",
]
