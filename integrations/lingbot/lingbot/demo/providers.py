# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot model-input providers for shared demo run modes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from flashdreams.runtime import (
    InferenceInput,
    InferenceInputSchema,
    StepRequirements,
    UserInputs,
)
from flashdreams.runtime.demo import (
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from flashdreams.runtime.keyboard import DEFAULT_SUPPORTED_KEYS, KeyboardState
from flashdreams.serving.webrtc.services import (
    WEBRTC_SKIPPED_INPUTS_METADATA_KEY,
    WEBRTC_SKIPPED_WINDOW_METADATA_KEY,
)
from lingbot.controls import CameraPoseIntegrator, PoseSegment
from lingbot.runtime import (
    FIELD_PROMPT,
    FIELD_WORLD_SCALE,
    LingbotModelAdapter,
    LingbotReplayInputs,
)

FIELD_CAMERA_TRAJECTORY = "camera_trajectory"
FIELD_CAMERA_INTRINSICS = "camera_intrinsics"
FIELD_TOTAL_CAMERA_FRAMES = "total_camera_frames"
PROVIDER_INPUTS_METADATA_KEY = "lingbot_provider_inputs"

_INTRINSICS_REFERENCE_HEIGHT = 480
_INTRINSICS_REFERENCE_WIDTH = 832
_CLEAR_STATES = frozenset({"clear", "release", "off", "none"})
_TRIGGER_STATES = frozenset({"trigger", "hold", "on"})


@dataclass(frozen=True, kw_only=True, slots=True)
class LingbotCameraTrace:
    """Fixed LingBot camera trajectory loaded for a shared demo scenario."""

    __hash__ = None

    poses: torch.Tensor
    intrinsics: torch.Tensor
    world_scale: float

    def __post_init__(self) -> None:
        if self.poses.ndim != 3 or self.poses.shape[1:] != (4, 4):
            raise ValueError(
                f"LingbotCameraTrace.poses must be [T, 4, 4], got "
                f"{tuple(self.poses.shape)}."
            )
        if self.intrinsics.ndim != 2 or self.intrinsics.shape[1] != 4:
            raise ValueError(
                f"LingbotCameraTrace.intrinsics must be [T, 4], got "
                f"{tuple(self.intrinsics.shape)}."
            )
        if self.world_scale < 0:
            raise ValueError("LingbotCameraTrace.world_scale must be >= 0.")

    @property
    def frame_count(self) -> int:
        return int(self.poses.shape[0])


@dataclass(frozen=True, kw_only=True, slots=True)
class LingbotProviderInputs:
    """Provider-owned LingBot input setup for a migrated shared demo scenario."""

    fps: int
    trace: LingbotCameraTrace | None = None
    base_intrinsics: torch.Tensor | Sequence[float] | None = None
    world_scale: float | None = None
    prompt: str = ""
    text_event_prompts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("LingbotProviderInputs.fps must be > 0.")
        object.__setattr__(
            self,
            "text_event_prompts",
            MappingProxyType(
                {str(key): str(value) for key, value in self.text_event_prompts.items()}
            ),
        )
        if self.trace is not None:
            object.__setattr__(self, "world_scale", self.trace.world_scale)
            object.__setattr__(self, "base_intrinsics", None)
            return
        if self.base_intrinsics is None:
            raise ValueError("Live LingBot provider inputs require base_intrinsics.")
        intrinsics = torch.as_tensor(self.base_intrinsics, dtype=torch.float32).reshape(
            4
        )
        object.__setattr__(self, "base_intrinsics", intrinsics.clone())
        if self.world_scale is None or self.world_scale <= 0:
            raise ValueError("Live LingBot provider inputs require world_scale > 0.")

    @property
    def live_camera(self) -> bool:
        return self.trace is None


def create_lingbot_provider_inputs(
    replay_inputs: LingbotReplayInputs,
    *,
    live_camera: bool,
    text_event_prompts: Mapping[str, str] | None = None,
) -> LingbotProviderInputs:
    """Build provider-owned input setup from resolved LingBot replay inputs."""
    trace = load_camera_trace(
        camera_poses_path=replay_inputs.camera_poses_path,
        camera_intrinsics_path=replay_inputs.camera_intrinsics_path,
        pixel_height=replay_inputs.pixel_height,
        pixel_width=replay_inputs.pixel_width,
        intrinsics_reference_height=_INTRINSICS_REFERENCE_HEIGHT,
        intrinsics_reference_width=_INTRINSICS_REFERENCE_WIDTH,
        world_scale=replay_inputs.world_scale,
    )
    if live_camera:
        return LingbotProviderInputs(
            fps=replay_inputs.fps,
            base_intrinsics=trace.intrinsics[0],
            # A trace's world scale is derived from pose spread, so stationary
            # examples fall back to the unit live-control scale.
            world_scale=trace.world_scale or 1.0,
            prompt=replay_inputs.prompt,
            text_event_prompts=text_event_prompts or {},
        )
    return LingbotProviderInputs(
        fps=replay_inputs.fps,
        trace=trace,
        prompt=replay_inputs.prompt,
        text_event_prompts=text_event_prompts or {},
    )


def load_camera_trace(
    *,
    camera_poses_path: str | Path,
    camera_intrinsics_path: str | Path,
    pixel_height: int,
    pixel_width: int,
    intrinsics_reference_height: int,
    intrinsics_reference_width: int,
    world_scale: float | None = None,
) -> LingbotCameraTrace:
    """Load and preprocess a fixed LingBot camera trajectory from ``.npy`` files."""
    from lingbot.encoder.utils import (  # noqa: PLC0415
        get_Ks_transformed,
        preprocess_example_poses,
    )

    intrinsics = torch.from_numpy(
        np.asarray(np.load(camera_intrinsics_path), dtype=np.float32)
    )
    intrinsics = get_Ks_transformed(
        intrinsics,
        height_org=intrinsics_reference_height,
        width_org=intrinsics_reference_width,
        height_resize=pixel_height,
        width_resize=pixel_width,
        height_final=pixel_height,
        width_final=pixel_width,
    )
    poses, inferred_world_scale = preprocess_example_poses(
        np.asarray(np.load(camera_poses_path))
    )
    return LingbotCameraTrace(
        poses=torch.from_numpy(np.ascontiguousarray(poses)).to(torch.float32),
        intrinsics=intrinsics.to(torch.float32),
        world_scale=float(inferred_world_scale if world_scale is None else world_scale),
    )


class LingbotInputProvider:
    """Convert shared user-input windows directly into LingBot model inputs."""

    def __init__(
        self,
        *,
        scenario: PreparedScenario,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        provider_inputs = scenario.metadata.get(PROVIDER_INPUTS_METADATA_KEY)
        if not isinstance(provider_inputs, LingbotProviderInputs):
            raise TypeError(
                "LingbotInputProvider requires PreparedScenario.metadata"
                f"[{PROVIDER_INPUTS_METADATA_KEY!r}] to be LingbotProviderInputs."
            )
        if inference_input_schema is None:
            inference_input_schema = LingbotModelAdapter().inference_input_schema

        self.capabilities = ProviderCapabilities(
            supports_realtime_clock=provider_inputs.live_camera,
            supports_recorded_input=True,
            supports_reset=True,
            deterministic_given_inputs=True,
            user_input_schema=scenario.source_schema,
            inference_input_schema=inference_input_schema,
        )
        self._scenario = scenario
        self._inputs = provider_inputs
        self._step_base_inputs = InferenceInput(
            step=scenario.initial_inputs.step,
            metadata=scenario.initial_inputs.metadata,
        )
        self._next_frame_start = 0
        self._keyboard_state = KeyboardState(supported_keys=DEFAULT_SUPPORTED_KEYS)
        self._integrator = (
            CameraPoseIntegrator() if provider_inputs.live_camera else None
        )
        self._active_text_event_id: str | None = None
        self._applied_text_event_id: str | None = None
        self._closed = False

    def prepare_initial_input(self) -> InferenceInput:
        self._require_open()
        self._reset_state()
        payload = dict(self._scenario.initial_inputs.global_conditioning)
        payload[FIELD_WORLD_SCALE] = self._inputs.world_scale
        if self._inputs.trace is not None:
            payload[FIELD_TOTAL_CAMERA_FRAMES] = self._inputs.trace.frame_count
        return InferenceInput(
            global_conditioning=payload,
            step=self._scenario.initial_inputs.step,
            metadata=self._scenario.initial_inputs.metadata,
        )

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        self._require_open()
        if user_window.control is not None:
            return PreparedStep(control=user_window.control)

        self._advance_skipped_input_state(user_window)
        metadata: dict[str, Any] = dict(request.metadata)
        num_frames = _metadata_positive_int(
            metadata,
            "num_frames",
            default=request.input_frame_count,
        )
        frame_start = _metadata_int(
            metadata,
            "frame_start",
            default=self._next_frame_start,
        )
        if self._inputs.trace is not None:
            self._consume_window_inputs(
                user_window.inputs,
                start_s=user_window.start_s,
                end_s=user_window.end_s,
                collect_camera_segments=False,
            )
            poses, intrinsics = self._slice_trace(
                frame_start=frame_start,
                num_frames=num_frames,
            )
        else:
            segments = self._consume_window_inputs(
                user_window.inputs,
                start_s=user_window.start_s,
                end_s=user_window.end_s,
                collect_camera_segments=True,
            )
            poses, intrinsics = self._integrate_live_camera(
                segments=segments,
                start_s=user_window.start_s,
                end_s=user_window.end_s,
                num_frames=num_frames,
            )
        step = dict(self._step_base_inputs.step)
        step[FIELD_CAMERA_TRAJECTORY] = poses
        step[FIELD_CAMERA_INTRINSICS] = intrinsics
        inference_input = InferenceInput(
            global_conditioning=self._text_event_update(),
            step=step,
            metadata=self._step_base_inputs.metadata,
        )
        self._next_frame_start = frame_start + num_frames
        return PreparedStep(inference_input=inference_input)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._require_open()
        self._reset_state()

    def close(self) -> None:
        self._closed = True

    def _slice_trace(
        self,
        *,
        frame_start: int,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trace = self._inputs.trace
        if trace is None:
            raise RuntimeError("Cannot slice a missing LingBot camera trace.")
        frame_end = frame_start + num_frames
        if frame_end > trace.frame_count:
            raise ValueError(
                f"Lingbot camera trace has {trace.frame_count} frames, but "
                f"step needs frames [{frame_start}, {frame_end})."
            )
        return (
            trace.poses[frame_start:frame_end],
            trace.intrinsics[frame_start:frame_end],
        )

    def _integrate_live_camera(
        self,
        *,
        segments: list[PoseSegment],
        start_s: float,
        end_s: float,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._integrator is None:
            raise RuntimeError("Cannot integrate LingBot camera without an integrator.")
        base_intrinsics = self._inputs.base_intrinsics
        if not isinstance(base_intrinsics, torch.Tensor):
            raise RuntimeError("Live LingBot camera control requires base_intrinsics.")
        frame_times = [
            start_s + (index + 1) / self._inputs.fps for index in range(num_frames)
        ]
        frame_times[-1] = min(frame_times[-1], end_s)
        poses = self._integrator.integrate_chunk(
            segments=segments,
            frame_times=frame_times,
        )
        poses_t = torch.from_numpy(np.ascontiguousarray(poses)).to(torch.float32)
        poses_t = poses_t.reshape(num_frames, 4, 4)
        intrinsics_t = base_intrinsics.reshape(1, 4).repeat(num_frames, 1)
        return poses_t, intrinsics_t

    def _consume_window_inputs(
        self,
        user_inputs: UserInputs,
        *,
        start_s: float,
        end_s: float,
        collect_camera_segments: bool,
    ) -> list[PoseSegment]:
        segments: list[PoseSegment] = []
        segment_start = start_s
        previous_keys = self._keyboard_state.resolved_effective_keys()

        for event in user_inputs.events:
            if event.event_type in {"key_down", "key_up"} and self._inputs.live_camera:
                key = event.payload.get("key")
                if not isinstance(key, str):
                    continue
                edge_t = min(max(float(event.timestamp_s), start_s), end_s)
                if collect_camera_segments and edge_t > segment_start:
                    segments.append((segment_start, edge_t, previous_keys))
                    segment_start = edge_t
                self._keyboard_state.apply_event(
                    event="keydown" if event.event_type == "key_down" else "keyup",
                    key=key,
                )
                previous_keys = self._keyboard_state.resolved_effective_keys()
                continue
            if event.event_type == "text_event":
                self._apply_text_event(event.payload)

        if collect_camera_segments:
            if end_s > segment_start or not segments:
                segments.append((segment_start, end_s, previous_keys))
            return segments
        return []

    def _apply_text_event(self, payload: Mapping[str, Any]) -> None:
        event_id = payload.get("event_id")
        state = str(payload.get("state", "trigger")).strip().lower()
        if state and state not in _CLEAR_STATES and state not in _TRIGGER_STATES:
            raise ValueError(
                f"Unsupported text event state {state!r}. Supported states: "
                f"{sorted(_CLEAR_STATES | _TRIGGER_STATES)}."
            )
        if event_id is None or state in _CLEAR_STATES:
            self._active_text_event_id = None
            return
        self._active_text_event_id = str(event_id)

    def _text_event_update(self) -> Mapping[str, Any]:
        if not self._inputs.text_event_prompts:
            return {}
        event_id = self._active_text_event_id
        if event_id == self._applied_text_event_id:
            return {}
        if event_id is not None and event_id not in self._inputs.text_event_prompts:
            supported = ", ".join(sorted(self._inputs.text_event_prompts))
            raise ValueError(
                f"Unknown Lingbot text event_id={event_id!r}. Supported: {supported}"
            )
        self._applied_text_event_id = event_id
        prompt = (
            self._inputs.prompt
            if event_id is None
            else self._inputs.text_event_prompts[event_id]
        )
        return {} if prompt is None else {FIELD_PROMPT: prompt}

    def _reset_state(self) -> None:
        self._keyboard_state = KeyboardState(supported_keys=DEFAULT_SUPPORTED_KEYS)
        if self._integrator is not None:
            self._integrator.reset()
        self._active_text_event_id = None
        self._applied_text_event_id = None
        self._next_frame_start = 0

    def _advance_skipped_input_state(self, user_window: UserInputWindow) -> None:
        skipped_inputs = user_window.metadata.get(WEBRTC_SKIPPED_INPUTS_METADATA_KEY)
        skipped_window = user_window.metadata.get(WEBRTC_SKIPPED_WINDOW_METADATA_KEY)
        if not isinstance(skipped_inputs, UserInputs):
            return
        if not isinstance(skipped_window, tuple) or len(skipped_window) != 2:
            return
        start_value, end_value = skipped_window
        if not isinstance(start_value, int | float) or not isinstance(
            end_value,
            int | float,
        ):
            return
        start_s = float(start_value)
        end_s = float(end_value)
        if end_s <= start_s:
            return
        self._consume_window_inputs(
            skipped_inputs,
            start_s=start_s,
            end_s=end_s,
            collect_camera_segments=False,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("LingbotInputProvider is closed.")


def _metadata_int(
    metadata: Mapping[str, Any],
    name: str,
    *,
    default: int,
) -> int:
    if name not in metadata:
        return default
    return _required_int(metadata, name)


def _metadata_positive_int(
    metadata: Mapping[str, Any],
    name: str,
    *,
    default: int,
) -> int:
    if name not in metadata:
        if default <= 0:
            raise ValueError(f"StepRequirements.{name} fallback must be > 0.")
        return default
    return _required_positive_int(metadata, name)


def _required_int(metadata: Mapping[str, Any], name: str) -> int:
    value = metadata[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Step metadata {name!r} must be an integer.")
    return value


def _required_positive_int(metadata: Mapping[str, Any], name: str) -> int:
    value = _required_int(metadata, name)
    if value <= 0:
        raise ValueError(f"Step metadata {name!r} must be > 0.")
    return value


__all__ = [
    "FIELD_CAMERA_INTRINSICS",
    "FIELD_CAMERA_TRAJECTORY",
    "FIELD_TOTAL_CAMERA_FRAMES",
    "PROVIDER_INPUTS_METADATA_KEY",
    "LingbotCameraTrace",
    "LingbotInputProvider",
    "LingbotProviderInputs",
    "create_lingbot_provider_inputs",
    "load_camera_trace",
]
