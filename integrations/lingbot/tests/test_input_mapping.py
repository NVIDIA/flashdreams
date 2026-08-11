# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from lingbot.demo.providers import PROVIDER_INPUTS_METADATA_KEY, LingbotInputProvider
from lingbot.demo.spec import resolve_text_event_prompts, resolve_user_input_events
from lingbot.input_mapping import (
    CAMERA_COMMAND,
    FIELD_CAMERA_INTRINSICS,
    FIELD_CAMERA_TRAJECTORY,
    TEXT_EVENT,
    KeyboardToCameraCommand,
    LingbotInputMapping,
    TextEventSelection,
    load_camera_trace,
)

from flashdreams.runtime import (
    CanonicalInputs,
    InferenceInput,
    InputCanonicalizer,
    StepRequest,
    StepRequirements,
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import UserInputWindow

pytestmark = pytest.mark.ci_cpu

_KEYBOARD_SOURCE = UserInputSchema(
    capabilities=(
        UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"})),
        UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"})),
        UserInputCapability(
            event_type="text_event", payload_fields=frozenset({"event_id"})
        ),
    )
)


def _step_request(*, step_index: int, frame_start: int, num_frames: int, fps: int = 16):
    return StepRequest(
        step_index=step_index,
        user_input_window=TimeWindow(
            start_s=frame_start / fps,
            end_s=(frame_start + num_frames) / fps,
        ),
        metadata={"num_frames": num_frames, "frame_start": frame_start},
    )


def _live_mapping(**kwargs) -> LingbotInputMapping:
    return LingbotInputMapping(
        fps=16,
        base_intrinsics=torch.tensor([416.0, 416.0, 416.0, 240.0]),
        world_scale=1.0,
        **kwargs,
    )


def test_keyboard_events_become_camera_command_axes() -> None:
    converter = KeyboardToCameraCommand()
    inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.0, event_type="key_down", payload={"key": "w"}
            ),
        )
    )
    window = TimeWindow(start_s=0.0, end_s=1.0)

    value = converter.convert(inputs.window(window), window)

    assert value is not None
    assert value["move_forward"] == 1.0
    assert value["yaw"] == 0.0
    # Level-triggered: a key held across the next window still means forward.
    next_window = TimeWindow(start_s=1.0, end_s=2.0)
    held = converter.convert(UserInputs().window(next_window), next_window)
    assert held is not None
    assert held["move_forward"] == 1.0


def test_camera_command_segments_preserve_sub_window_timing() -> None:
    converter = KeyboardToCameraCommand()
    window = TimeWindow(start_s=0.0, end_s=1.0)
    inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.5, event_type="key_down", payload={"key": "w"}
            ),
        )
    )

    value = converter.convert(inputs.window(window), window)

    assert value is not None
    segments = value["segments"]
    assert [(start, end) for start, end, _ in segments] == [(0.0, 0.5), (0.5, 1.0)]
    assert segments[0][2]["move_forward"] == 0.0
    assert segments[1][2]["move_forward"] == 1.0


def test_key_events_drive_a_camera_trajectory() -> None:
    canonicalizer = InputCanonicalizer([KeyboardToCameraCommand()])
    mapping = _live_mapping()
    user_inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.0, event_type="key_down", payload={"key": "w"}
            ),
        )
    )
    request = _step_request(step_index=0, frame_start=0, num_frames=4)
    assert request.user_input_window is not None

    step_inputs = mapping.map_step_inputs(
        canonical_inputs=canonicalizer.canonicalize(
            user_inputs,
            window=request.user_input_window,
            source_schema=_KEYBOARD_SOURCE,
        ),
        inference_input=InferenceInput(),
        request=request,
    )

    poses = step_inputs.step[FIELD_CAMERA_TRAJECTORY]
    assert poses.shape == (4, 4, 4)
    assert step_inputs.step[FIELD_CAMERA_INTRINSICS].shape == (4, 4)
    # Holding forward has to actually move the camera along the trajectory.
    assert not torch.allclose(poses[0], poses[-1])
    assert poses[-1][:3, 3].abs().sum() > 0


def test_idle_keyboard_leaves_the_camera_stationary() -> None:
    canonicalizer = InputCanonicalizer([KeyboardToCameraCommand()])
    mapping = _live_mapping()
    request = _step_request(step_index=0, frame_start=0, num_frames=4)
    assert request.user_input_window is not None

    step_inputs = mapping.map_step_inputs(
        canonical_inputs=canonicalizer.canonicalize(
            UserInputs(),
            window=request.user_input_window,
            source_schema=_KEYBOARD_SOURCE,
        ),
        inference_input=InferenceInput(),
        request=request,
    )

    poses = step_inputs.step[FIELD_CAMERA_TRAJECTORY]
    assert torch.allclose(poses[0], poses[-1])


def test_text_event_becomes_a_global_conditioning_prompt_update() -> None:
    canonicalizer = InputCanonicalizer(
        [KeyboardToCameraCommand(), TextEventSelection()]
    )
    mapping = _live_mapping(text_event_prompts={"storm": "a violent storm"})
    mapping.set_base_prompt("a calm street")
    user_inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.0,
                event_type="text_event",
                payload={"event_id": "storm"},
            ),
        )
    )
    request = _step_request(step_index=0, frame_start=0, num_frames=4)
    assert request.user_input_window is not None

    step_inputs = mapping.map_step_inputs(
        canonical_inputs=canonicalizer.canonicalize(
            user_inputs,
            window=request.user_input_window,
            source_schema=_KEYBOARD_SOURCE,
        ),
        inference_input=InferenceInput(),
        request=request,
    )

    assert step_inputs.global_conditioning["prompt"] == "a violent storm"

    # The swap is requested once, not re-sent on every later step.
    next_request = _step_request(step_index=1, frame_start=4, num_frames=4)
    assert next_request.user_input_window is not None
    held = mapping.map_step_inputs(
        canonical_inputs=canonicalizer.canonicalize(
            user_inputs,
            window=next_request.user_input_window,
            source_schema=_KEYBOARD_SOURCE,
        ),
        inference_input=InferenceInput(),
        request=next_request,
    )
    assert held.global_conditioning == {}


def test_clearing_a_text_event_restores_the_base_prompt() -> None:
    converter = TextEventSelection()
    window = TimeWindow(start_s=0.0, end_s=1.0)
    triggered = converter.convert(
        UserInputs(
            events=(
                UserInputEvent(
                    timestamp_s=0.0,
                    event_type="text_event",
                    payload={"event_id": "storm"},
                ),
            )
        ),
        window,
    )
    assert triggered is not None and triggered["event_id"] == "storm"

    cleared = converter.convert(
        UserInputs(
            events=(
                UserInputEvent(
                    timestamp_s=0.5,
                    event_type="text_event",
                    payload={"event_id": "storm", "state": "clear"},
                ),
            )
        ),
        window,
    )
    assert cleared is not None and cleared["event_id"] is None


def test_unknown_text_event_is_rejected_by_the_mapping() -> None:
    canonicalizer = InputCanonicalizer(
        [KeyboardToCameraCommand(), TextEventSelection()]
    )
    mapping = _live_mapping(text_event_prompts={"storm": "a violent storm"})
    request = _step_request(step_index=0, frame_start=0, num_frames=4)
    assert request.user_input_window is not None
    canonical = canonicalizer.canonicalize(
        UserInputs(
            events=(
                UserInputEvent(
                    timestamp_s=0.0,
                    event_type="text_event",
                    payload={"event_id": "volcano"},
                ),
            )
        ),
        window=request.user_input_window,
        source_schema=_KEYBOARD_SOURCE,
    )

    with pytest.raises(ValueError, match="Unknown Lingbot text event_id"):
        mapping.map_step_inputs(
            canonical_inputs=canonical,
            inference_input=InferenceInput(),
            request=request,
        )


def test_live_mapping_requires_camera_command_from_the_source() -> None:
    mapping = _live_mapping()

    with pytest.raises(ValueError, match="requires a 'camera_command'"):
        mapping.map_step_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=InferenceInput(),
            request=_step_request(step_index=0, frame_start=0, num_frames=4),
        )


def test_trace_mapping_slices_successive_chunks(tmp_path: Path) -> None:
    poses_path = tmp_path / "poses.npy"
    intrinsics_path = tmp_path / "intrinsics.npy"
    trajectory = np.tile(np.eye(4, dtype=np.float32), (32, 1, 1))
    trajectory[:, 2, 3] = np.arange(32, dtype=np.float32)
    np.save(poses_path, trajectory)
    np.save(
        intrinsics_path,
        np.tile(np.array([416.0, 416.0, 416.0, 240.0], dtype=np.float32), (32, 1)),
    )
    trace = load_camera_trace(
        camera_poses_path=poses_path,
        camera_intrinsics_path=intrinsics_path,
        pixel_height=464,
        pixel_width=832,
        intrinsics_reference_height=480,
        intrinsics_reference_width=832,
    )
    mapping = LingbotInputMapping(fps=16, trace=trace)

    first = mapping.map_step_inputs(
        canonical_inputs=CanonicalInputs(),
        inference_input=InferenceInput(),
        request=_step_request(step_index=0, frame_start=0, num_frames=4),
    )
    second = mapping.map_step_inputs(
        canonical_inputs=CanonicalInputs(),
        inference_input=InferenceInput(),
        request=_step_request(step_index=1, frame_start=4, num_frames=4),
    )

    assert first.step[FIELD_CAMERA_TRAJECTORY].shape == (4, 4, 4)
    # Consecutive steps must advance through the trace, not restart it.
    assert not torch.allclose(
        first.step[FIELD_CAMERA_TRAJECTORY], second.step[FIELD_CAMERA_TRAJECTORY]
    )
    assert mapping.mapping_schema.consumes == ()


def test_trace_mapping_reports_running_past_the_end(tmp_path: Path) -> None:
    poses_path = tmp_path / "poses.npy"
    intrinsics_path = tmp_path / "intrinsics.npy"
    np.save(poses_path, np.tile(np.eye(4, dtype=np.float32), (16, 1, 1)))
    np.save(
        intrinsics_path,
        np.tile(np.array([416.0, 416.0, 416.0, 240.0], dtype=np.float32), (16, 1)),
    )
    mapping = LingbotInputMapping(
        fps=16,
        trace=load_camera_trace(
            camera_poses_path=poses_path,
            camera_intrinsics_path=intrinsics_path,
            pixel_height=464,
            pixel_width=832,
            intrinsics_reference_height=480,
            intrinsics_reference_width=832,
        ),
    )

    with pytest.raises(ValueError, match="camera trace has"):
        mapping.map_step_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=InferenceInput(),
            request=_step_request(step_index=0, frame_start=0, num_frames=999),
        )


def test_scenario_event_trace_resolves_into_user_inputs() -> None:
    user_inputs = resolve_user_input_events(
        {
            "events": [
                {"t": 1.5, "type": "key_down", "key": "a"},
                {"t": 0.0, "type": "key_down", "key": "w"},
                {"t": 2.0, "type": "text_event", "event_id": "storm"},
            ]
        }
    )

    # UserInputs requires non-decreasing timestamps, so resolution must sort.
    assert [event.timestamp_s for event in user_inputs.events] == [0.0, 1.5, 2.0]
    assert user_inputs.events[0].payload == {"key": "w"}
    assert user_inputs.events[2].event_type == "text_event"


def test_scenario_text_event_catalog_resolves() -> None:
    assert resolve_text_event_prompts({"text_events": {"storm": "a storm"}}) == {
        "storm": "a storm"
    }
    assert resolve_text_event_prompts(
        {"text_events": [{"event_id": "portal", "prompt": "a glowing portal"}]}
    ) == {"portal": "a glowing portal"}
    assert resolve_text_event_prompts(None) == {}


def test_declared_modalities_match_what_the_converters_produce() -> None:
    canonicalizer = InputCanonicalizer(
        [KeyboardToCameraCommand(), TextEventSelection()]
    )

    schema = canonicalizer.canonical_schema(_KEYBOARD_SOURCE)

    assert schema.supports(CAMERA_COMMAND)
    assert schema.supports(TEXT_EVENT)
    # A source with no key events cannot feed the keyboard converter.
    empty = canonicalizer.canonical_schema(UserInputSchema())
    assert not empty.supports(CAMERA_COMMAND)


def test_event_driven_scenario_uses_shared_input_provider(tmp_path: Path) -> None:
    """A scenario can drive the camera from events instead of the pose trace."""
    from lingbot.demo.adapter import LingbotDemoAdapter
    from lingbot.runtime import LINGBOT_MODEL_ID

    from flashdreams.runtime import InferenceConfig
    from flashdreams.runtime.demo import DemoSpec, Mp4OutputSpec

    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    poses_path = tmp_path / "poses.npy"
    intrinsics_path = tmp_path / "intrinsics.npy"
    np.save(poses_path, np.tile(np.eye(4, dtype=np.float32), (32, 1, 1)))
    np.save(
        intrinsics_path,
        np.tile(np.array([416.0, 416.0, 416.0, 240.0], dtype=np.float32), (32, 1)),
    )

    spec = DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        input_mode="replay",
        scenario={
            "prompt": "a calm street",
            "image_path": image,
            "pose_path": poses_path,
            "intrinsic_path": intrinsics_path,
            "camera_source": "events",
            "text_events": {"storm": "a violent storm"},
            "events": [
                {"t": 0.0, "type": "key_down", "key": "w"},
                {"t": 0.2, "type": "text_event", "event_id": "storm"},
            ],
        },
        output=Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=16),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            runtime_options={"pipeline_config": object()},
        ),
    )

    prepared = LingbotDemoAdapter().prepare_scenario(spec)

    assert prepared.mapping is None
    assert prepared.canonicalizer.converters == ()
    assert PROVIDER_INPUTS_METADATA_KEY in prepared.metadata
    assert len(prepared.user_inputs.events) == 2

    request = _step_request(step_index=0, frame_start=0, num_frames=4)
    assert request.user_input_window is not None
    provider = LingbotInputProvider(
        scenario=prepared,
        inference_input_schema=LingbotDemoAdapter().inference_input_schema,
    )
    step = provider.prepare_step(
        request=StepRequirements(
            step_index=request.step_index,
            input_frame_count=4,
            metadata=request.metadata,
        ),
        user_window=UserInputWindow(
            start_s=request.user_input_window.start_s,
            end_s=request.user_input_window.end_s,
            inputs=prepared.user_inputs,
        ),
    )
    assert step.inference_input is not None
    step_inputs = step.inference_input
    poses = step_inputs.step[FIELD_CAMERA_TRAJECTORY]
    assert poses.shape == (4, 4, 4)
    assert not torch.allclose(poses[0], poses[-1])
    assert step_inputs.global_conditioning["prompt"] == "a violent storm"
