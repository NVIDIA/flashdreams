# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from lingbot.demo import DEFAULT_LINGBOT_PRESET, LINGBOT_MODEL_ID, LingbotDemoAdapter
from lingbot.demo.providers import (
    FIELD_CAMERA_TRAJECTORY,
    FIELD_TOTAL_CAMERA_FRAMES,
    PROVIDER_INPUTS_METADATA_KEY,
    LingbotInputProvider,
)
from lingbot.input_mapping import (
    KeyboardToCameraCommand,
    LingbotInputMapping,
    TextEventSelection,
)
from lingbot.runtime import (
    FIELD_FIRST_FRAME_PATH,
    FIELD_PROMPT,
    FIELD_TOTAL_BLOCKS,
    LingbotReplayInputs,
)

from flashdreams.runtime import (
    CanonicalInputs,
    InferenceConfig,
    InferenceInput,
    InputCanonicalizer,
    StepRequest,
    StepRequirements,
    TimeWindow,
    UserInputEvent,
    UserInputs,
)
from flashdreams.runtime.demo import DemoSpec, Mp4OutputSpec, PreparedScenario
from flashdreams.runtime.demo.session_inputs import UserInputWindow
from flashdreams.serving.webrtc.services import (
    WEBRTC_SKIPPED_INPUTS_METADATA_KEY,
    WEBRTC_SKIPPED_WINDOW_METADATA_KEY,
)

pytestmark = pytest.mark.ci_cpu


def test_lingbot_provider_initial_input_matches_mapping_path(tmp_path: Path) -> None:
    adapter = LingbotDemoAdapter()
    actual = _prepared_scenario(tmp_path, adapter=adapter)
    legacy = _legacy_bridge(tmp_path, adapter=adapter)
    assert actual.mapping is None
    assert actual.canonicalizer.converters == ()
    assert PROVIDER_INPUTS_METADATA_KEY in actual.metadata

    provider = LingbotInputProvider(
        scenario=actual,
        inference_input_schema=adapter.inference_input_schema,
    )

    expected_initial = legacy.mapping.map_global_conditioning_inputs(
        canonical_inputs=CanonicalInputs(),
        inference_input=actual.initial_inputs,
    )
    actual_initial = provider.prepare_initial_input()

    assert actual_initial.global_conditioning == expected_initial.global_conditioning
    assert actual_initial.step == expected_initial.step
    assert actual_initial.metadata == expected_initial.metadata
    assert actual_initial.global_conditioning[FIELD_PROMPT] == "drive through a city"
    assert actual_initial.global_conditioning[FIELD_FIRST_FRAME_PATH] == (
        tmp_path / "image.jpg"
    )
    assert actual_initial.global_conditioning[FIELD_TOTAL_BLOCKS] == 2
    assert FIELD_TOTAL_CAMERA_FRAMES in actual_initial.global_conditioning


def test_lingbot_provider_trace_steps_match_mapping_path(tmp_path: Path) -> None:
    adapter = LingbotDemoAdapter()
    actual = _prepared_scenario(tmp_path, adapter=adapter)
    legacy = _legacy_bridge(tmp_path, adapter=adapter)
    provider = LingbotInputProvider(
        scenario=actual,
        inference_input_schema=adapter.inference_input_schema,
    )

    provider.prepare_initial_input()
    expected_first = _legacy_step(
        actual,
        legacy,
        step_index=0,
        frame_start=0,
        num_frames=4,
    )
    actual_first = _provider_step(
        provider,
        step_index=0,
        frame_start=0,
        num_frames=4,
        inputs=actual.user_inputs,
    )
    expected_second = _legacy_step(
        actual,
        legacy,
        step_index=1,
        frame_start=4,
        num_frames=4,
    )
    actual_second = _provider_step(
        provider,
        step_index=1,
        frame_start=4,
        num_frames=4,
        inputs=actual.user_inputs,
    )

    assert actual_first.global_conditioning == expected_first.global_conditioning
    assert torch.allclose(
        actual_first.step[FIELD_CAMERA_TRAJECTORY],
        expected_first.step[FIELD_CAMERA_TRAJECTORY],
    )
    assert torch.allclose(
        actual_second.step[FIELD_CAMERA_TRAJECTORY],
        expected_second.step[FIELD_CAMERA_TRAJECTORY],
    )
    assert not torch.allclose(
        actual_first.step[FIELD_CAMERA_TRAJECTORY],
        actual_second.step[FIELD_CAMERA_TRAJECTORY],
    )


def test_lingbot_provider_uses_driver_user_window_inputs(tmp_path: Path) -> None:
    adapter = LingbotDemoAdapter()
    scenario_events = ({"t": 10.0, "type": "key_down", "key": "a"},)
    actual = _prepared_scenario(
        tmp_path,
        adapter=adapter,
        camera_source="events",
        events=scenario_events,
    )
    legacy = _legacy_bridge(
        tmp_path,
        adapter=adapter,
        camera_source="events",
    )
    provider = LingbotInputProvider(
        scenario=actual,
        inference_input_schema=adapter.inference_input_schema,
    )
    window_inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.0,
                event_type="key_down",
                payload={"key": "w"},
            ),
        )
    )

    provider.prepare_initial_input()
    expected_step = _legacy_step(
        actual,
        legacy,
        step_index=0,
        frame_start=0,
        num_frames=4,
        inputs=window_inputs,
    )
    actual_step = _provider_step(
        provider,
        step_index=0,
        frame_start=0,
        num_frames=4,
        inputs=window_inputs,
    )

    poses = actual_step.step[FIELD_CAMERA_TRAJECTORY]
    assert torch.allclose(poses, expected_step.step[FIELD_CAMERA_TRAJECTORY])
    assert not torch.allclose(poses[0], poses[-1])


def test_lingbot_provider_folds_webrtc_skipped_inputs_into_state(
    tmp_path: Path,
) -> None:
    adapter = LingbotDemoAdapter()
    scenario_events = ({"t": 10.0, "type": "key_down", "key": "a"},)
    with_skip = _prepared_scenario(
        tmp_path,
        adapter=adapter,
        camera_source="events",
        events=scenario_events,
    )
    idle = _prepared_scenario(
        tmp_path,
        adapter=adapter,
        camera_source="events",
        events=scenario_events,
    )
    provider = LingbotInputProvider(
        scenario=with_skip,
        inference_input_schema=adapter.inference_input_schema,
    )
    idle_provider = LingbotInputProvider(
        scenario=idle,
        inference_input_schema=adapter.inference_input_schema,
    )
    skipped_inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.0,
                event_type="key_down",
                payload={"key": "w"},
            ),
        )
    )

    provider.prepare_initial_input()
    idle_provider.prepare_initial_input()
    legacy = _legacy_bridge(
        tmp_path,
        adapter=adapter,
        camera_source="events",
    )
    legacy.canonicalizer.canonicalize(
        skipped_inputs,
        window=TimeWindow(start_s=0.0, end_s=0.25),
        source_schema=with_skip.source_schema,
    )
    actual = _provider_step(
        provider,
        step_index=0,
        frame_start=4,
        num_frames=4,
        inputs=UserInputs(),
        metadata={
            WEBRTC_SKIPPED_INPUTS_METADATA_KEY: skipped_inputs,
            WEBRTC_SKIPPED_WINDOW_METADATA_KEY: (0.0, 0.25),
        },
    )
    expected_idle = _provider_step(
        idle_provider,
        step_index=0,
        frame_start=4,
        num_frames=4,
        inputs=UserInputs(),
    )
    expected = _legacy_step(
        with_skip,
        legacy,
        step_index=0,
        frame_start=4,
        num_frames=4,
        inputs=UserInputs(),
    )

    poses = actual.step[FIELD_CAMERA_TRAJECTORY]
    assert torch.allclose(poses, expected.step[FIELD_CAMERA_TRAJECTORY])
    assert not torch.allclose(poses, expected_idle.step[FIELD_CAMERA_TRAJECTORY])
    assert not torch.allclose(poses[0], poses[-1])


def test_lingbot_provider_reset_clears_text_event_state(tmp_path: Path) -> None:
    adapter = LingbotDemoAdapter()
    text_events = {"storm": "a violent storm"}
    actual = _prepared_scenario(
        tmp_path,
        adapter=adapter,
        camera_source="events",
        text_events=text_events,
        events=(
            {"t": 10.0, "type": "key_down", "key": "w"},
            {"t": 10.1, "type": "text_event", "event_id": "storm"},
        ),
    )
    legacy = _legacy_bridge(
        tmp_path,
        adapter=adapter,
        camera_source="events",
        text_events=text_events,
    )
    provider = LingbotInputProvider(
        scenario=actual,
        inference_input_schema=adapter.inference_input_schema,
    )
    first_inputs = _window_inputs_with_text_event(timestamp_s=0.0)

    provider.prepare_initial_input()
    first = _provider_step(
        provider,
        step_index=0,
        frame_start=0,
        num_frames=4,
        inputs=first_inputs,
    )
    expected_first = _legacy_step(
        actual,
        legacy,
        step_index=0,
        frame_start=0,
        num_frames=4,
        inputs=first_inputs,
    )
    repeated = _provider_step(
        provider,
        step_index=1,
        frame_start=4,
        num_frames=4,
        inputs=UserInputs(),
    )
    expected_repeated = _legacy_step(
        actual,
        legacy,
        step_index=1,
        frame_start=4,
        num_frames=4,
        inputs=UserInputs(),
    )
    provider.reset()
    after_reset = _provider_step(
        provider,
        step_index=0,
        frame_start=0,
        num_frames=4,
        inputs=first_inputs,
    )

    assert first.global_conditioning == expected_first.global_conditioning
    assert repeated.global_conditioning == expected_repeated.global_conditioning
    assert first.global_conditioning[FIELD_PROMPT] == "a violent storm"
    assert repeated.global_conditioning == {}
    assert after_reset.global_conditioning[FIELD_PROMPT] == "a violent storm"


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"frame_start": "0", "num_frames": 4}, "frame_start"),
        ({"frame_start": 0.5, "num_frames": 4}, "frame_start"),
        ({"frame_start": 0, "num_frames": "4"}, "num_frames"),
        ({"frame_start": 0, "num_frames": 4.5}, "num_frames"),
    ],
)
def test_lingbot_provider_rejects_non_integer_frame_metadata(
    tmp_path: Path,
    metadata: dict[str, object],
    match: str,
) -> None:
    provider = LingbotInputProvider(
        scenario=_prepared_scenario(tmp_path, adapter=LingbotDemoAdapter())
    )
    provider.prepare_initial_input()

    with pytest.raises(TypeError, match=match):
        provider.prepare_step(
            request=StepRequirements(
                step_index=0,
                input_frame_count=4,
                metadata=metadata,
            ),
            user_window=UserInputWindow(
                start_s=0.0,
                end_s=0.25,
                inputs=UserInputs(),
            ),
        )


def _prepared_scenario(
    tmp_path: Path,
    *,
    adapter: LingbotDemoAdapter,
    camera_source: str = "trace",
    text_events: dict[str, str] | None = None,
    events: tuple[dict[str, Any], ...] = (),
) -> PreparedScenario:
    image = tmp_path / "image.jpg"
    poses = tmp_path / "poses.npy"
    intrinsics = tmp_path / "intrinsics.npy"
    image.write_bytes(b"fake")
    _write_camera_assets(poses, intrinsics)
    scenario: dict[str, Any] = {
        "prompt": "drive through a city",
        "image_path": image,
        "pose_path": poses,
        "intrinsic_path": intrinsics,
        "camera_source": camera_source,
        "total_blocks": 2,
    }
    if text_events is not None:
        scenario["text_events"] = text_events
    if events:
        scenario["events"] = list(events)
    return adapter.prepare_scenario(
        DemoSpec(
            model_id=LINGBOT_MODEL_ID,
            preset_id=DEFAULT_LINGBOT_PRESET,
            input_mode="replay",
            scenario=scenario,
            output=Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=16),
            config=InferenceConfig(
                model_id=LINGBOT_MODEL_ID,
                preset_id=DEFAULT_LINGBOT_PRESET,
                runtime_options={"pipeline_config": object()},
            ),
        )
    )


def _write_camera_assets(poses: Path, intrinsics: Path, *, frames: int = 32) -> None:
    trajectory = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    trajectory[:, 2, 3] = np.arange(frames, dtype=np.float32)
    np.save(poses, trajectory)
    np.save(
        intrinsics,
        np.tile(np.array([416.0, 416.0, 416.0, 240.0], dtype=np.float32), (frames, 1)),
    )


@dataclass(slots=True)
class _LegacyBridge:
    mapping: LingbotInputMapping
    canonicalizer: InputCanonicalizer


def _legacy_bridge(
    tmp_path: Path,
    *,
    adapter: LingbotDemoAdapter,
    camera_source: str = "trace",
    text_events: dict[str, str] | None = None,
) -> _LegacyBridge:
    replay_inputs = LingbotReplayInputs(
        prompt="drive through a city",
        first_frame_path=tmp_path / "image.jpg",
        camera_poses_path=tmp_path / "poses.npy",
        camera_intrinsics_path=tmp_path / "intrinsics.npy",
        total_blocks=2,
    )
    if camera_source == "events":
        trace = adapter.create_input_mapping(replay_inputs).camera_trace
        mapping = adapter.create_live_input_mapping(
            fps=replay_inputs.fps,
            base_intrinsics=trace.intrinsics[0],
            world_scale=trace.world_scale or 1.0,
            prompt=replay_inputs.prompt,
            text_event_prompts=text_events,
        )
    else:
        mapping = adapter.create_input_mapping(
            replay_inputs,
            text_event_prompts=text_events,
        )
    converters: list[Any] = [KeyboardToCameraCommand()]
    if text_events:
        converters.append(TextEventSelection())
    return _LegacyBridge(
        mapping=mapping,
        canonicalizer=InputCanonicalizer(converters),
    )


def _legacy_step(
    scenario: PreparedScenario,
    legacy: _LegacyBridge,
    *,
    step_index: int,
    frame_start: int,
    num_frames: int,
    inputs: UserInputs | None = None,
) -> InferenceInput:
    window = TimeWindow(
        start_s=frame_start / 16,
        end_s=(frame_start + num_frames) / 16,
    )
    return legacy.mapping.map_step_inputs(
        canonical_inputs=legacy.canonicalizer.canonicalize(
            scenario.user_inputs if inputs is None else inputs,
            window=window,
            source_schema=scenario.source_schema,
        ),
        inference_input=InferenceInput(
            step=scenario.initial_inputs.step,
            metadata=scenario.initial_inputs.metadata,
        ),
        request=StepRequest(
            step_index=step_index,
            user_input_window=window,
            metadata={"frame_start": frame_start, "num_frames": num_frames},
        ),
    )


def _provider_step(
    provider: LingbotInputProvider,
    *,
    step_index: int,
    frame_start: int,
    num_frames: int,
    inputs: UserInputs,
    metadata: dict[str, object] | None = None,
) -> InferenceInput:
    prepared = provider.prepare_step(
        request=StepRequirements(
            step_index=step_index,
            input_frame_count=num_frames,
            metadata={"frame_start": frame_start, "num_frames": num_frames},
        ),
        user_window=UserInputWindow(
            start_s=frame_start / 16,
            end_s=(frame_start + num_frames) / 16,
            inputs=inputs,
            metadata={} if metadata is None else metadata,
        ),
    )
    assert prepared.inference_input is not None
    return prepared.inference_input


def _window_inputs_with_text_event(*, timestamp_s: float) -> UserInputs:
    return UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=timestamp_s,
                event_type="key_down",
                payload={"key": "w"},
            ),
            UserInputEvent(
                timestamp_s=timestamp_s,
                event_type="text_event",
                payload={"event_id": "storm"},
            ),
        )
    )
