# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The manager's ``InferenceSession`` branch must preserve camera controls.

The session branch buffers raw events, canonicalizes them over the chunk
window, and maps them into per-step ``InferenceInput``. The resulting camera
trajectory must match the direct resampler/integrator reference, or moving
LingBot's live path onto the runtime API would silently change how it drives.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from lingbot.input_mapping import (
    FIELD_CAMERA_INTRINSICS,
    FIELD_CAMERA_TRAJECTORY,
    KeyboardToCameraCommand,
    LingbotInputMapping,
    TextEventSelection,
)
from lingbot.webrtc.session import LINGBOT_WEBRTC_SOURCE_SCHEMA

from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime.inputs import InferenceInput, TimeWindow
from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.types import StepRequest, StepResult
from flashdreams.serving.realtime.input import KeyboardResampler
from flashdreams.serving.webrtc.controls import CameraPoseIntegrator
from flashdreams.serving.webrtc.manager import (
    BaseWebRTCSessionManager,
    ManagedWebRTCSession,
)

pytestmark = pytest.mark.ci_cpu

_FPS = 16
_NUM_FRAMES = 4
_BASE_INTRINSICS = torch.tensor([416.0, 416.0, 416.0, 240.0])


class _FakeRuntimeConfig:
    video_width = 64
    video_height = 64
    warmup_chunks = 0
    warmup_timeout_s = 1.0


class _FakeSession:
    """Records what the manager hands the model."""

    def __init__(self) -> None:
        self.steps: list[InferenceInput] = []
        self._index = 0

    def next_step_request(self) -> StepRequest:
        return StepRequest(
            step_index=self._index,
            metadata={
                "num_frames": _NUM_FRAMES,
                "frame_start": self._index * _NUM_FRAMES,
            },
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        self.steps.append(inputs)
        index = self._index
        self._index += 1
        return StepResult(
            step_index=index,
            output=VideoStepResult(
                chunk_index=index,
                video_chunk=torch.zeros(_NUM_FRAMES, 3, 4, 4),
                layout="tchw",
                num_frames=_NUM_FRAMES,
                stats={},
            ),
            frame_count=_NUM_FRAMES,
        )


class _FakeRuntime:
    def __init__(self, *, text_event_prompts: dict[str, str] | None = None) -> None:
        self.input_canonicalizer = InputCanonicalizer(
            [KeyboardToCameraCommand(), TextEventSelection()]
        )
        self.input_mapping = LingbotInputMapping(
            fps=_FPS,
            base_intrinsics=_BASE_INTRINSICS,
            world_scale=1.0,
            text_event_prompts=text_event_prompts,
        )
        self.input_mapping.set_base_prompt("a calm street")
        self.input_source_schema = LINGBOT_WEBRTC_SOURCE_SCHEMA
        self.session = _FakeSession()

    async def start_inference_session(self) -> _FakeSession:
        return self.session


class _Manager(BaseWebRTCSessionManager[Any, Any]):
    def _model_name(self) -> str:
        return "fake"


def _managed_session(runtime: _FakeRuntime) -> ManagedWebRTCSession:
    return ManagedWebRTCSession(
        runtime=runtime,
        video_track=None,
        video_encoder=None,
        peer_connection=None,
        resampler=KeyboardResampler(fps=_FPS, start_v=0.0),
        inference_session=runtime.session,
    )


def _manager(runtime: _FakeRuntime) -> _Manager:
    return _Manager(runtime=runtime, runtime_config=_FakeRuntimeConfig(), fps=_FPS)


def _reference_poses(
    edges: list[tuple[float, str, str]], *, chunks: int
) -> np.ndarray:
    resampler = KeyboardResampler(fps=_FPS, start_v=0.0)
    integrator = CameraPoseIntegrator()
    for timestamp_s, event, key in edges:
        resampler.on_edge(arrival_t=timestamp_s, event=event, key=key)
    poses = []
    for _ in range(chunks):
        segments, frame_times = resampler.sample_chunk(_NUM_FRAMES)
        poses.append(
            integrator.integrate_chunk(segments=segments, frame_times=frame_times)
        )
    return np.concatenate(poses)


def _session_branch_poses(
    edges: list[tuple[float, str, str]], *, chunks: int
) -> np.ndarray:
    runtime = _FakeRuntime()
    manager = _manager(runtime)
    managed = _managed_session(runtime)
    for timestamp_s, event, key in edges:
        manager._record_user_event(
            managed_session=managed,
            timestamp_s=timestamp_s,
            event_type="key_down" if event == "keydown" else "key_up",
            payload={"key": key},
        )

    poses = []
    for chunk_index in range(chunks):
        start_s = chunk_index * _NUM_FRAMES / _FPS
        end_s = (chunk_index + 1) * _NUM_FRAMES / _FPS
        window = TimeWindow(start_s=start_s, end_s=end_s)
        request = runtime.session.next_step_request()
        from dataclasses import replace

        step_inputs = manager._build_step_inputs(
            managed_session=managed,
            request=replace(request, user_input_window=window),
            window=window,
        )
        runtime.session.step(step_inputs)
        manager._prune_consumed_user_events(managed, before_s=start_s)
        poses.append(step_inputs.step[FIELD_CAMERA_TRAJECTORY].numpy())
    return np.concatenate(poses)


@pytest.mark.parametrize(
    "edges",
    [
        pytest.param([(0.0, "keydown", "w")], id="hold_forward"),
        pytest.param(
            [(0.0, "keydown", "w"), (0.13, "keydown", "a")], id="mid_chunk_turn"
        ),
        pytest.param(
            [(0.0, "keydown", "w"), (0.25, "keydown", "d")], id="chunk_boundary"
        ),
        pytest.param(
            [(0.01, "keydown", "w"), (0.04, "keyup", "w"), (0.08, "keydown", "w")],
            id="rapid_toggle",
        ),
        pytest.param([], id="idle"),
    ],
)
def test_session_branch_matches_reference_camera_integration(
    edges: list[tuple[float, str, str]],
) -> None:
    reference = _reference_poses(edges, chunks=3)
    session_branch = _session_branch_poses(edges, chunks=3)

    assert reference.shape == session_branch.shape
    np.testing.assert_allclose(session_branch, reference, atol=1e-5)


def test_session_branch_supplies_intrinsics_for_every_step() -> None:
    runtime = _FakeRuntime()
    manager = _manager(runtime)
    managed = _managed_session(runtime)
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.0,
        event_type="key_down",
        payload={"key": "w"},
    )
    window = TimeWindow(start_s=0.0, end_s=_NUM_FRAMES / _FPS)

    step_inputs = manager._build_step_inputs(
        managed_session=managed,
        request=runtime.session.next_step_request(),
        window=window,
    )

    assert step_inputs.step[FIELD_CAMERA_INTRINSICS].shape == (_NUM_FRAMES, 4)
    assert torch.allclose(
        step_inputs.step[FIELD_CAMERA_INTRINSICS][0], _BASE_INTRINSICS
    )


def test_consumed_events_are_pruned() -> None:
    runtime = _FakeRuntime()
    manager = _manager(runtime)
    managed = _managed_session(runtime)
    for index in range(5):
        manager._record_user_event(
            managed_session=managed,
            timestamp_s=index * 0.1,
            event_type="key_down",
            payload={"key": "w"},
        )

    manager._prune_consumed_user_events(managed, before_s=0.25)

    # Held-key state lives in the converter, so consumed events are safe to drop
    # and must be, or a long session's buffer grows without bound.
    assert [event.timestamp_s for event in managed.user_events] == [
        pytest.approx(0.3),
        pytest.approx(0.4),
    ]


@pytest.mark.asyncio
async def test_text_event_becomes_a_buffered_user_event() -> None:
    runtime = _FakeRuntime(text_event_prompts={"storm": "a violent storm"})
    manager = _manager(runtime)
    managed = _managed_session(runtime)
    assert not hasattr(runtime, "trigger_event")

    handled = await manager._handle_event_message(
        managed_session=managed,
        payload={"event_id": "storm", "state": "trigger"},
    )

    assert handled is True
    assert [event.event_type for event in managed.user_events] == ["text_event"]
    assert managed.user_events[0].payload["event_id"] == "storm"

    # A text event can itself be the first interaction, so it is stamped just
    # before the resampler re-anchors its clock. Chunk 0 must still see it.
    anchor = managed.user_events[0].timestamp_s + 0.05
    await manager._step_inference_session(
        managed_session=managed,
        window=TimeWindow(start_s=anchor, end_s=anchor + _NUM_FRAMES / _FPS),
    )

    assert len(runtime.session.steps) == 1
    assert runtime.session.steps[0].global_conditioning["prompt"] == "a violent storm"


def test_real_lingbot_runtime_selects_the_session_branch() -> None:
    """The shipped runtime must be session-capable while retaining segment stepping."""
    from lingbot.webrtc.session import LingbotInferenceRuntime, LingbotRuntimeConfig

    runtime = LingbotInferenceRuntime(config=LingbotRuntimeConfig(device="cpu"))

    assert BaseWebRTCSessionManager._drives_inference_session(runtime) is True
    assert callable(runtime.generate_chunk)


def test_session_start_requires_an_initialized_rollout() -> None:
    """Starting a session before reset must fail loudly, not silently no-op."""
    import asyncio

    from lingbot.webrtc.session import (
        LingbotInferenceRuntime,
        LingbotRuntimeConfig,
        LingbotRuntimeError,
    )

    runtime = LingbotInferenceRuntime(config=LingbotRuntimeConfig(device="cpu"))

    with pytest.raises(LingbotRuntimeError, match="input mapping is not initialized"):
        asyncio.run(runtime.start_inference_session())
