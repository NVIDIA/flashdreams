# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot user-event canonicalization and canonical-to-model input mapping.

Lingbot's two live controls are a free camera driven from the keyboard and a
catalog of server-owned text events. This module carries both across the
``UserInputs -> CanonicalInputs -> InferenceInput`` boundary:

- :data:`CAMERA_COMMAND` and :class:`KeyboardToCameraCommand` turn raw key
  edges into device-independent camera intent;
- :data:`TEXT_EVENT` and :class:`TextEventSelection` track which text event is
  active;
- :class:`LingbotInputMapping` turns that canonical intent into the per-step
  camera trajectory the session consumes, and requests a session-global prompt
  update when the active text event changes.

The modalities live here rather than in ``flashdreams.runtime.canonical``
because Lingbot is currently their only consumer. Both are plain
``CanonicalModality`` values, so lifting them into the shared canonical layer
later is a move plus an export, with no change to this mapping.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flashdreams.runtime.canonical import DeviceConverterSchema
from flashdreams.runtime.inputs import (
    CanonicalInputs,
    CanonicalInputSchema,
    CanonicalModality,
    InferenceInput,
    InferenceInputSchema,
    InputField,
    TimeWindow,
    UserInputCapability,
    UserInputs,
)
from flashdreams.runtime.keyboard import DEFAULT_SUPPORTED_KEYS, KeyboardState
from flashdreams.runtime.mapping import InputMappingSchema
from flashdreams.runtime.types import StepRequest
from lingbot.controls import (
    CameraPoseIntegrator,
    PoseSegment,
)

FIELD_CAMERA_TRAJECTORY = "camera_trajectory"
FIELD_CAMERA_INTRINSICS = "camera_intrinsics"
FIELD_PROMPT = "prompt"
FIELD_WORLD_SCALE = "world_scale"
FIELD_TOTAL_CAMERA_FRAMES = "total_camera_frames"

_PASSTHROUGH_GLOBAL_FIELDS: tuple[InputField, ...] = (
    InputField(name="first_frame_path", input_modality="image/path"),
    InputField(name="total_blocks", input_modality="count"),
    InputField(name="pixel_height", input_modality="pixel-height"),
    InputField(name="pixel_width", input_modality="pixel-width"),
    InputField(name="fps", input_modality="fps"),
)
"""App-owned session inputs this mapping forwards without interpreting them."""

_CLEAR_STATES = frozenset({"clear", "release", "off", "none"})
_TRIGGER_STATES = frozenset({"trigger", "hold", "on"})

_AXES: tuple[str, ...] = ("move_forward", "move_right", "yaw", "pitch")

_AXIS_KEYS: Mapping[str, tuple[str, str]] = {
    # Axis -> (positive key, negative key) in CameraPoseIntegrator's vocabulary.
    # Both directions of the keyboard/axis conversion are derived from this one
    # table so a rebind cannot make them disagree.
    "move_forward": ("w", "s"),
    "move_right": ("e", "q"),
    "yaw": ("a", "d"),
    "pitch": ("i", "k"),
}

_KEY_ALIASES: Mapping[str, str] = {"j": "a", "l": "d"}
"""Alternate yaw keys accepted by ``KeyboardState``, folded onto ``a``/``d``."""


CAMERA_COMMAND = CanonicalModality(
    name="camera_command",
    payload_fields=frozenset({*_AXES, "segments"}),
    description=(
        "Free-camera intent. move_forward, move_right, yaw, and pitch are in "
        "[-1, 1] and hold the level state at the end of the window. segments "
        "carries the piecewise-constant timeline inside the window as "
        "((start_s, end_s, axes), ...), so a consumer can integrate sub-window "
        "timing instead of quantizing control to the chunk boundary."
    ),
)

TEXT_EVENT = CanonicalModality(
    name="text_event",
    payload_fields=frozenset({"event_id"}),
    description=(
        "Identifier of the active server-owned text event, or None when no "
        "event is active. Level-triggered: the value is held until cleared."
    ),
)


def _axes_from_keys(pressed: Iterable[str]) -> dict[str, float]:
    """Return camera axis values for a resolved set of pressed keys."""
    keys = {_KEY_ALIASES.get(key, key) for key in pressed}
    axes: dict[str, float] = {}
    for axis, (positive, negative) in _AXIS_KEYS.items():
        value = 0.0
        if positive in keys:
            value += 1.0
        if negative in keys:
            value -= 1.0
        axes[axis] = value
    return axes


def _keys_from_axes(axes: Mapping[str, float]) -> frozenset[str]:
    """Return the integrator key set equivalent to ``axes``.

    Pose integration stays the single implementation in
    :class:`CameraPoseIntegrator`, which is expressed over key sets. Converting
    back here keeps live Lingbot trajectories identical to the WebRTC path
    instead of forking the integration math.
    """
    keys: set[str] = set()
    for axis, (positive, negative) in _AXIS_KEYS.items():
        value = float(axes.get(axis, 0.0))
        if value > 0:
            keys.add(positive)
        elif value < 0:
            keys.add(negative)
    return frozenset(keys)


class KeyboardToCameraCommand:
    """Convert keyboard edges into :data:`CAMERA_COMMAND` level state."""

    def __init__(
        self,
        *,
        name: str = "keyboard-to-camera-command",
        supported_keys: frozenset[str] = DEFAULT_SUPPORTED_KEYS,
        priority: int = 0,
    ) -> None:
        self._supported_keys = supported_keys
        self._state = KeyboardState(supported_keys=supported_keys)
        self._schema = DeviceConverterSchema(
            name=name,
            produces=CAMERA_COMMAND,
            device_kind="keyboard",
            priority=priority,
            consumes=(
                UserInputCapability(
                    event_type="key_down",
                    payload_fields=frozenset({"key"}),
                ),
                UserInputCapability(
                    event_type="key_up",
                    payload_fields=frozenset({"key"}),
                ),
            ),
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        return self._schema

    def reset(self) -> None:
        self._state = KeyboardState(supported_keys=self._supported_keys)

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, Any] | None:
        segments: list[tuple[float, float, dict[str, float]]] = []
        segment_start = window.start_s
        axes = _axes_from_keys(self._state.resolved_effective_keys())

        for event in user_inputs.events:
            if event.event_type not in {"key_down", "key_up"}:
                continue
            key = event.payload.get("key")
            if not isinstance(key, str):
                continue
            edge_t = min(max(float(event.timestamp_s), window.start_s), window.end_s)
            if edge_t > segment_start:
                segments.append((segment_start, edge_t, axes))
                segment_start = edge_t
            self._state.apply_event(
                event="keydown" if event.event_type == "key_down" else "keyup",
                key=key,
            )
            axes = _axes_from_keys(self._state.resolved_effective_keys())

        if window.end_s > segment_start or not segments:
            segments.append((segment_start, window.end_s, axes))

        return CAMERA_COMMAND.value({**axes, "segments": tuple(segments)})


class TextEventSelection:
    """Track the active :data:`TEXT_EVENT` id across windows."""

    def __init__(
        self,
        *,
        name: str = "text-event-selection",
        priority: int = 0,
    ) -> None:
        self._active_event_id: str | None = None
        self._schema = DeviceConverterSchema(
            name=name,
            produces=TEXT_EVENT,
            device_kind="text-event",
            priority=priority,
            consumes=(
                UserInputCapability(
                    event_type="text_event",
                    payload_fields=frozenset({"event_id"}),
                ),
            ),
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        return self._schema

    def reset(self) -> None:
        self._active_event_id = None

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, Any] | None:
        del window
        for event in user_inputs.events:
            if event.event_type != "text_event":
                continue
            event_id = event.payload.get("event_id")
            state = str(event.payload.get("state", "trigger")).strip().lower()
            if state and state not in _CLEAR_STATES and state not in _TRIGGER_STATES:
                raise ValueError(
                    f"Unsupported text event state {state!r}. Supported states: "
                    f"{sorted(_CLEAR_STATES | _TRIGGER_STATES)}."
                )
            if event_id is None or state in _CLEAR_STATES:
                self._active_event_id = None
                continue
            self._active_event_id = str(event_id)
        return TEXT_EVENT.value({"event_id": self._active_event_id})


@dataclass(frozen=True, kw_only=True, slots=True)
class LingbotCameraTrace:
    """Fixed camera trajectory resolved from a replay scenario.

    Tensors are CPU float32; the session owns device placement.
    """

    __hash__ = None

    poses: torch.Tensor
    """Camera-to-world poses, shape ``[T, 4, 4]``."""

    intrinsics: torch.Tensor
    """Per-frame intrinsics, shape ``[T, 4]``, already rescaled to output size."""

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
            # Zero is legal: preprocess_example_poses derives the scale from
            # pose spread, so a stationary trace yields 0. That reached the
            # model before this mapping existed, so it still does.
            raise ValueError("LingbotCameraTrace.world_scale must be >= 0.")

    @property
    def frame_count(self) -> int:
        return int(self.poses.shape[0])


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
    """Load and preprocess a fixed Lingbot camera trajectory from ``.npy`` files."""
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


class LingbotInputMapping:
    """Build Lingbot per-step camera inputs from canonical user input.

    Two trajectory sources are supported through one mapping object, because
    ``run_inference_session`` takes a single mapping:

    - a fixed :class:`LingbotCameraTrace`, sliced per step, which consumes no
      canonical modality and keeps MP4/benchmark runs deterministic;
    - live :data:`CAMERA_COMMAND` intent integrated into a trajectory, for
      event-driven runs.

    Text events are mapped to a session-global prompt update rather than a
    per-step field: swapping the rollout's text context is session-global model
    state, so it travels in the ``global_conditioning`` slot of the step
    payload. Whether the model can apply that update is session-owned.
    """

    def __init__(
        self,
        *,
        fps: int,
        trace: LingbotCameraTrace | None = None,
        base_intrinsics: torch.Tensor | Sequence[float] | None = None,
        world_scale: float | None = None,
        text_event_prompts: Mapping[str, str] | None = None,
        integrator: CameraPoseIntegrator | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("LingbotInputMapping.fps must be > 0.")
        if trace is None and base_intrinsics is None:
            raise ValueError(
                "LingbotInputMapping requires either a fixed camera trace or "
                "base_intrinsics for live camera control."
            )
        self._fps = int(fps)
        self._trace = trace
        self._text_event_prompts = dict(text_event_prompts or {})
        self._applied_event_id: str | None = None
        self._base_prompt: str | None = None

        if trace is None:
            intrinsics = torch.as_tensor(base_intrinsics, dtype=torch.float32).reshape(
                4
            )
            if world_scale is None or world_scale <= 0:
                raise ValueError(
                    "Live Lingbot camera control requires a positive world_scale."
                )
            self._base_intrinsics = intrinsics
            self._world_scale = float(world_scale)
            self._integrator = integrator or CameraPoseIntegrator()
        else:
            self._base_intrinsics = None
            self._world_scale = trace.world_scale
            self._integrator = None

        consumes: list[CanonicalModality] = []
        if trace is None:
            consumes.append(CAMERA_COMMAND)
        if self._text_event_prompts:
            consumes.append(TEXT_EVENT)
        self._mapping_schema = InputMappingSchema(
            name="lingbot-input-mapping",
            consumes=tuple(consumes),
            produces_global_conditioning=(
                # map_global_conditioning_inputs returns the app-owned session
                # payload augmented with the fields below, so the pass-through
                # fields are part of what this mapping produces. Declaring them
                # keeps undeclared_inference_inputs() quiet and lets the
                # compatibility check see that required session inputs are
                # reachable; omitting them makes the check reject every run.
                *_PASSTHROUGH_GLOBAL_FIELDS,
                InputField(
                    name=FIELD_WORLD_SCALE,
                    required=False,
                    input_modality="scale",
                    frequency_consumed="once",
                ),
                InputField(
                    name=FIELD_TOTAL_CAMERA_FRAMES,
                    required=False,
                    input_modality="count",
                    frequency_consumed="once",
                ),
                InputField(
                    name=FIELD_PROMPT,
                    required=False,
                    input_modality="text",
                    frequency_consumed="once",
                    description="Text-event prompt update for an active rollout.",
                ),
            ),
            produces_step=(
                InputField(
                    name=FIELD_CAMERA_TRAJECTORY,
                    input_modality="c2w_sequence",
                    frequency_consumed="per_step",
                    metadata={"shape": "[T,4,4]", "frame": "camera_to_world"},
                ),
                InputField(
                    name=FIELD_CAMERA_INTRINSICS,
                    input_modality="intrinsics_vec4_sequence",
                    frequency_consumed="per_step",
                    metadata={"shape": "[T,4]"},
                ),
            ),
        )

    @property
    def mapping_schema(self) -> InputMappingSchema:
        return self._mapping_schema

    @property
    def camera_trace(self) -> LingbotCameraTrace:
        """Return the fixed trace, for callers reusing its calibration."""
        if self._trace is None:
            raise ValueError("This Lingbot mapping has no fixed camera trace.")
        return self._trace

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema:
        """Return the modalities this mapping consumes, for adapter reporting."""
        return CanonicalInputSchema(
            modalities=self._mapping_schema.consumes,
            description="Lingbot live camera and text-event control.",
        )

    def validate(
        self,
        *,
        canonical_schema: CanonicalInputSchema | None = None,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        if canonical_schema is not None:
            for modality in self._mapping_schema.consumes:
                if not canonical_schema.supports(modality):
                    raise ValueError(
                        f"Lingbot input mapping requires canonical modality "
                        f"{modality.name!r}, which the selected input source "
                        f"cannot supply."
                    )
        if inference_input_schema is not None:
            for name in (FIELD_CAMERA_TRAJECTORY, FIELD_CAMERA_INTRINSICS):
                if inference_input_schema.field_for(name=name, phase="step") is None:
                    raise ValueError(
                        f"Lingbot input mapping produces step input {name!r}, "
                        f"which this model does not declare."
                    )

    def map_global_conditioning_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
    ) -> InferenceInput:
        del canonical_inputs
        payload = dict(inference_input.global_conditioning)
        payload[FIELD_WORLD_SCALE] = self._world_scale
        if self._trace is not None:
            payload[FIELD_TOTAL_CAMERA_FRAMES] = self._trace.frame_count
        return InferenceInput(
            global_conditioning=payload,
            step=inference_input.step,
            metadata=inference_input.metadata,
        )

    def map_step_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
        request: StepRequest,
    ) -> InferenceInput:
        num_frames = _required_int(request.metadata, "num_frames")
        frame_start = _required_int(request.metadata, "frame_start")

        if self._trace is not None:
            poses, intrinsics = self._slice_trace(
                frame_start=frame_start,
                num_frames=num_frames,
            )
        else:
            poses, intrinsics = self._integrate(
                canonical_inputs=canonical_inputs,
                request=request,
                frame_start=frame_start,
                num_frames=num_frames,
            )

        step = dict(inference_input.step)
        step[FIELD_CAMERA_TRAJECTORY] = poses
        step[FIELD_CAMERA_INTRINSICS] = intrinsics
        return InferenceInput(
            global_conditioning=self._text_event_update(canonical_inputs),
            step=step,
            metadata=inference_input.metadata,
        )

    def _slice_trace(
        self,
        *,
        frame_start: int,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._trace is not None
        frame_end = frame_start + num_frames
        if frame_end > self._trace.frame_count:
            raise ValueError(
                f"Lingbot camera trace has {self._trace.frame_count} frames, but "
                f"step needs frames [{frame_start}, {frame_end})."
            )
        return (
            self._trace.poses[frame_start:frame_end],
            self._trace.intrinsics[frame_start:frame_end],
        )

    def _integrate(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        request: StepRequest,
        frame_start: int,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._integrator is not None
        assert self._base_intrinsics is not None
        command = canonical_inputs.values.get(CAMERA_COMMAND.name)
        if command is None:
            raise ValueError(
                "Lingbot live camera control requires a 'camera_command' "
                "canonical value for every step; the selected input source "
                "produced none."
            )

        window = request.user_input_window
        start_s = window.start_s if window is not None else frame_start / self._fps
        end_s = (
            window.end_s
            if window is not None
            else (frame_start + num_frames) / self._fps
        )
        segments = _pose_segments(command, start_s=start_s, end_s=end_s)
        frame_times = [start_s + (index + 1) / self._fps for index in range(num_frames)]
        # The integrator rejects frame times outside the segment span, and float
        # accumulation can leave the last one a hair past the window end.
        frame_times[-1] = min(frame_times[-1], end_s)

        poses = self._integrator.integrate_chunk(
            segments=segments,
            frame_times=frame_times,
        )
        poses_t = torch.from_numpy(np.ascontiguousarray(poses)).to(torch.float32)
        poses_t = poses_t.reshape(num_frames, 4, 4)
        intrinsics_t = self._base_intrinsics.reshape(1, 4).repeat(num_frames, 1)
        return poses_t, intrinsics_t

    def _text_event_update(
        self,
        canonical_inputs: CanonicalInputs,
    ) -> Mapping[str, Any]:
        if not self._text_event_prompts:
            return {}
        value = canonical_inputs.values.get(TEXT_EVENT.name)
        if value is None:
            return {}
        event_id = value.get("event_id")
        if event_id == self._applied_event_id:
            return {}
        if event_id is not None and event_id not in self._text_event_prompts:
            supported = ", ".join(sorted(self._text_event_prompts))
            raise ValueError(
                f"Unknown Lingbot text event_id={event_id!r}. Supported: {supported}"
            )
        self._applied_event_id = event_id
        prompt = (
            self._base_prompt
            if event_id is None
            else self._text_event_prompts[event_id]
        )
        return {} if prompt is None else {FIELD_PROMPT: prompt}

    def set_base_prompt(self, prompt: str) -> None:
        """Record the rollout prompt restored when a text event is cleared."""
        self._base_prompt = prompt

    def reset(self) -> None:
        """Reset state accumulated while mapping a rollout."""
        self._applied_event_id = None
        if self._integrator is not None:
            self._integrator.reset()


def _pose_segments(
    command: Mapping[str, Any],
    *,
    start_s: float,
    end_s: float,
) -> list[PoseSegment]:
    """Return integrator-ready segments for one step window."""
    raw = command.get("segments")
    if not raw:
        # A source that supplies only level state still drives the step; the
        # whole window then holds one constant command.
        return [(start_s, end_s, _keys_from_axes(command))]
    segments: list[PoseSegment] = []
    for segment_start, segment_end, axes in raw:
        if float(segment_end) <= float(segment_start):
            continue
        segments.append(
            (float(segment_start), float(segment_end), _keys_from_axes(axes))
        )
    if not segments:
        return [(start_s, end_s, _keys_from_axes(command))]
    return segments


def _required_int(metadata: Mapping[str, Any], name: str) -> int:
    if name not in metadata:
        raise ValueError(
            f"Lingbot input mapping requires StepRequest.metadata[{name!r}]; the "
            f"session did not provide it."
        )
    return int(metadata[name])


__all__ = [
    "CAMERA_COMMAND",
    "FIELD_CAMERA_INTRINSICS",
    "FIELD_CAMERA_TRAJECTORY",
    "FIELD_TOTAL_CAMERA_FRAMES",
    "KeyboardToCameraCommand",
    "LingbotCameraTrace",
    "LingbotInputMapping",
    "TEXT_EVENT",
    "TextEventSelection",
    "load_camera_trace",
]
