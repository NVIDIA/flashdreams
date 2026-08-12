# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The manager's legacy ``InferenceSession`` branch must preserve camera controls.

The old direct WebRTC session branch buffers raw events, canonicalizes them
over the chunk window, and maps them into per-step ``InferenceInput``. The
resulting camera trajectory must match the direct resampler/integrator
reference while this compatibility path remains available.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import torch
from lingbot.controls import CameraPoseIntegrator, KeyboardResampler
from lingbot.input_mapping import (
    FIELD_CAMERA_INTRINSICS,
    FIELD_CAMERA_TRAJECTORY,
    KeyboardToCameraCommand,
    LingbotInputMapping,
    TextEventSelection,
)
from lingbot.webrtc.session import LINGBOT_WEBRTC_SOURCE_SCHEMA

from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.demo import RealtimeEventResampler
from flashdreams.runtime.inputs import InferenceInput, TimeWindow
from flashdreams.runtime.types import StepRequest, StepResult
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
                "input_frame_count": _NUM_FRAMES,
                "num_frames": _NUM_FRAMES,
                "frame_start": self._index * _NUM_FRAMES,
            },
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        self.steps.append(inputs)
        index = self._index
        self._index += 1
        return StepResult.from_video_chunk(
            step_index=index,
            video_chunk=torch.zeros(_NUM_FRAMES, 3, 4, 4),
            layout="tchw",
        )


class _FakeRuntime:
    def __init__(self, *, text_event_prompts: dict[str, str] | None = None) -> None:
        self.text_event_prompts = dict(text_event_prompts or {})
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

    def validate_user_event(
        self, *, event_type: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        if event_type != "text_event":
            return payload
        state = str(payload.get("state", "trigger")).strip().lower() or "trigger"
        clear_states = {"clear", "release", "off", "none"}
        trigger_states = {"trigger", "hold", "on"}
        if state not in clear_states and state not in trigger_states:
            raise ValueError(f"Unsupported text event state {state!r}.")
        event_id = payload.get("event_id")
        if event_id is None or state in clear_states:
            return {"event_id": None, "state": state}
        event_id = str(event_id)
        if event_id not in self.text_event_prompts:
            raise ValueError(f"Unknown Lingbot text event_id={event_id!r}.")
        return {"event_id": event_id, "state": state}


class _Manager(BaseWebRTCSessionManager[Any, Any]):
    def _model_name(self) -> str:
        return "fake"


class _FakeControlChannel:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def send(self, payload: str) -> None:
        decoded = json.loads(payload)
        assert isinstance(decoded, dict)
        self.messages.append(decoded)


class _FakeCloseable:
    async def close(self) -> None:
        return


class _FakeVideoEncoder:
    fps = _FPS
    backend = "fake"
    prefers_codec: str | None = None

    def close(self) -> None:
        return


def _managed_session(runtime: _FakeRuntime) -> ManagedWebRTCSession:
    return ManagedWebRTCSession(
        runtime=runtime,
        video_track=_FakeCloseable(),  # ty:ignore[invalid-argument-type]
        video_encoder=_FakeVideoEncoder(),  # ty:ignore[invalid-argument-type]
        peer_connection=_FakeCloseable(),
        resampler=RealtimeEventResampler(fps=_FPS, start_v=0.0),
        inference_session=runtime.session,
    )


def _manager(runtime: _FakeRuntime) -> _Manager:
    return _Manager(
        runtime=runtime,
        runtime_config=_FakeRuntimeConfig(),
        fps=_FPS,
        identity="fake",
    )


def _reference_poses(edges: list[tuple[float, str, str]], *, chunks: int) -> np.ndarray:
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
async def test_catch_up_window_clears_release_and_renders_latest_step() -> None:
    runtime = _FakeRuntime()
    manager = _manager(runtime)
    managed = _managed_session(runtime)
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.1,
        event_type="key_down",
        payload={"key": "w"},
    )
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.2,
        event_type="key_up",
        payload={"key": "w"},
    )
    manager._record_user_event(
        managed_session=managed,
        timestamp_s=0.8,
        event_type="key_down",
        payload={"key": "w"},
    )

    manager._catch_up_input_clock(
        managed_session=managed,
        now=1.0,
        chunk_duration=_NUM_FRAMES / _FPS,
    )
    await manager._step_inference_session(
        managed_session=managed,
        window=TimeWindow(start_s=0.75, end_s=1.0),
    )

    assert [(event.timestamp_s, event.event_type) for event in managed.user_events] == [
        (pytest.approx(0.8), "key_down")
    ]
    poses = runtime.session.steps[0].step[FIELD_CAMERA_TRAJECTORY]
    assert not torch.allclose(poses[:, :3, 3], torch.zeros_like(poses[:, :3, 3]))


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param(
            {"event_id": "unknown", "state": "trigger"},
            "Unknown Lingbot text event_id='unknown'",
            id="unknown_event",
        ),
        pytest.param(
            {"event_id": "storm", "state": "explode"},
            "Unsupported text event state 'explode'",
            id="bad_state",
        ),
    ],
)
async def test_text_event_rejects_invalid_payload_before_ack(
    payload: dict[str, str],
    message: str,
) -> None:
    runtime = _FakeRuntime(text_event_prompts={"storm": "a violent storm"})
    manager = _manager(runtime)
    managed = _managed_session(runtime)
    channel = _FakeControlChannel()
    managed.control_channel = channel

    handled = await manager._handle_event_message(
        managed_session=managed,
        payload=payload,
    )

    assert handled is False
    assert list(managed.user_events) == []
    assert channel.messages[0]["type"] == "error"
    assert message in str(channel.messages[0]["message"])
    assert len(channel.messages) == 1


def test_real_lingbot_runtime_selects_the_session_branch() -> None:
    """The shipped runtime must be session-capable while retaining segment stepping."""
    from lingbot.webrtc.session import LingbotInferenceRuntime, LingbotRuntimeConfig

    runtime = LingbotInferenceRuntime(config=LingbotRuntimeConfig(device="cpu"))

    assert BaseWebRTCSessionManager._drives_inference_session(runtime) is True
    assert callable(runtime.step)
    assert callable(runtime.start_inference_session)


def test_real_lingbot_inference_session_steps_on_runtime_worker() -> None:
    import asyncio

    from lingbot.model_session import LingbotModelSessionCore
    from lingbot.webrtc.session import LingbotInferenceRuntime, LingbotRuntimeConfig

    from flashdreams.infra.video_output import VideoOutputStream

    class _FakeTransformer:
        pass

    class _FakeDiffusionModel:
        def __init__(self) -> None:
            self.transformer = _FakeTransformer()

    class _FakePipeline:
        def __init__(self) -> None:
            self.diffusion_model = _FakeDiffusionModel()
            self.generated_inputs: list[object] = []

        def get_num_output_frames(self, autoregressive_index: int) -> int:
            del autoregressive_index
            return _NUM_FRAMES

        def generate(
            self,
            *,
            autoregressive_index: int,
            cache: object,
            input: object,
        ) -> torch.Tensor:
            del autoregressive_index, cache
            self.generated_inputs.append(input)
            return torch.zeros((_NUM_FRAMES, 3, 4, 4), dtype=torch.uint8)

        def finalize(
            self, *, autoregressive_index: int, cache: object
        ) -> dict[str, float]:
            del autoregressive_index, cache
            return {"decode_s": 0.1}

    runtime = LingbotInferenceRuntime(
        config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0, text_events=())
    )
    pipeline = _FakePipeline()
    runtime._pipeline = pipeline
    runtime._model_session = LingbotModelSessionCore(
        pipeline=pipeline,
        output_stream_factory=lambda: VideoOutputStream(
            postprocess_stream=None,
            output_layout="tchw",
        ),
    )
    runtime._model_session._cache = object()

    try:
        inference_session = asyncio.run(runtime.start_inference_session())
        request = inference_session.next_step_request()
        result = inference_session.step(
            InferenceInput(
                step={
                    FIELD_CAMERA_TRAJECTORY: torch.eye(4).repeat(_NUM_FRAMES, 1, 1),
                    FIELD_CAMERA_INTRINSICS: _BASE_INTRINSICS.repeat(_NUM_FRAMES, 1),
                },
            )
        )
    finally:
        asyncio.run(runtime.close())

    assert request is not None
    assert request.step_index == 0
    assert request.metadata["input_frame_count"] == _NUM_FRAMES
    assert result.step_index == 0
    assert result.frame_count == _NUM_FRAMES
    assert result.metrics["decode_s"] == pytest.approx(0.1)
    assert len(pipeline.generated_inputs) == 1


def test_session_start_requires_an_initialized_rollout() -> None:
    """Starting a session before reset must fail loudly, not silently no-op."""
    import asyncio

    from lingbot.webrtc.session import (
        LingbotInferenceRuntime,
        LingbotRuntimeConfig,
        LingbotRuntimeError,
    )

    runtime = LingbotInferenceRuntime(config=LingbotRuntimeConfig(device="cpu"))

    with pytest.raises(LingbotRuntimeError, match="Runtime is not initialized"):
        asyncio.run(runtime.start_inference_session())
