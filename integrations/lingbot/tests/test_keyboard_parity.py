# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live camera control must match the WebRTC path it will eventually replace.

The WebRTC runtime drives the camera with ``KeyboardResampler.sample_chunk``
feeding ``CameraPoseIntegrator``. The runtime-API path instead windows events
with ``StepRequest.user_input_window``, canonicalizes them into camera intent,
and integrates that. Both should produce the same trajectory for the same key
stream; these tests pin that, so a divergence shows up here rather than as
different handling in a live session.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from lingbot.controls import CameraPoseIntegrator, KeyboardResampler
from lingbot.input_mapping import (
    FIELD_CAMERA_TRAJECTORY,
    KeyboardToCameraCommand,
    LingbotInputMapping,
)

from flashdreams.runtime import (
    InferenceInput,
    InputCanonicalizer,
    StepRequest,
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)

pytestmark = pytest.mark.ci_cpu

_FPS = 16
_NUM_FRAMES = 4

_SOURCE = UserInputSchema(
    capabilities=(
        UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"})),
        UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"})),
    )
)

# (timestamp_s, event_type, key)
Edge = tuple[float, str, str]

_STREAMS: dict[str, list[Edge]] = {
    "hold_forward": [(0.0, "key_down", "w")],
    "forward_then_release": [(0.0, "key_down", "w"), (0.1, "key_up", "w")],
    "mid_chunk_turn": [(0.0, "key_down", "w"), (0.13, "key_down", "a")],
    "strafe_and_pitch": [(0.02, "key_down", "e"), (0.09, "key_down", "i")],
    "alternate_yaw_keys": [(0.0, "key_down", "w"), (0.05, "key_down", "j")],
    "conflicting_yaw": [(0.0, "key_down", "a"), (0.07, "key_down", "d")],
    "rapid_toggle": [
        (0.01, "key_down", "w"),
        (0.04, "key_up", "w"),
        (0.08, "key_down", "w"),
        (0.2, "key_up", "w"),
    ],
    "idle": [],
    # KeyboardResampler drains events with `event_t <= chunk_end` while
    # TimeWindow is half-open, so an edge landing exactly on a chunk boundary
    # is the most likely place for the two paths to disagree.
    "exact_chunk_boundary": [(0.0, "key_down", "w"), (0.25, "key_down", "a")],
    "boundary_release": [(0.0, "key_down", "w"), (0.25, "key_up", "w")],
    "second_boundary": [(0.0, "key_down", "w"), (0.5, "key_down", "d")],
}


def _legacy_poses(edges: list[Edge], *, chunks: int) -> np.ndarray:
    """Integrate a key stream the way the WebRTC session does."""
    resampler = KeyboardResampler(fps=_FPS, start_v=0.0)
    integrator = CameraPoseIntegrator()
    for timestamp_s, event_type, key in edges:
        resampler.on_edge(
            arrival_t=timestamp_s,
            event="keydown" if event_type == "key_down" else "keyup",
            key=key,
        )
    poses = []
    for _ in range(chunks):
        segments, frame_times = resampler.sample_chunk(_NUM_FRAMES)
        poses.append(
            integrator.integrate_chunk(segments=segments, frame_times=frame_times)
        )
    return np.concatenate(poses)


def _runtime_api_poses(edges: list[Edge], *, chunks: int) -> np.ndarray:
    """Integrate the same key stream through the runtime API input path."""
    canonicalizer = InputCanonicalizer([KeyboardToCameraCommand()])
    mapping = LingbotInputMapping(
        fps=_FPS,
        base_intrinsics=torch.tensor([416.0, 416.0, 416.0, 240.0]),
        world_scale=1.0,
    )
    user_inputs = UserInputs(
        events=tuple(
            UserInputEvent(
                timestamp_s=timestamp_s,
                event_type=event_type,
                payload={"key": key},
            )
            for timestamp_s, event_type, key in edges
        )
    )
    poses = []
    for chunk_index in range(chunks):
        frame_start = chunk_index * _NUM_FRAMES
        request = StepRequest(
            step_index=chunk_index,
            user_input_window=TimeWindow(
                start_s=frame_start / _FPS,
                end_s=(frame_start + _NUM_FRAMES) / _FPS,
            ),
            metadata={"num_frames": _NUM_FRAMES, "frame_start": frame_start},
        )
        assert request.user_input_window is not None
        step_inputs = mapping.map_step_inputs(
            canonical_inputs=canonicalizer.canonicalize(
                user_inputs,
                window=request.user_input_window,
                source_schema=_SOURCE,
            ),
            inference_input=InferenceInput(),
            request=request,
        )
        poses.append(step_inputs.step[FIELD_CAMERA_TRAJECTORY].numpy())
    return np.concatenate(poses)


@pytest.mark.parametrize("name", sorted(_STREAMS))
def test_single_chunk_matches_the_webrtc_path(name: str) -> None:
    edges = _STREAMS[name]

    legacy = _legacy_poses(edges, chunks=1)
    runtime_api = _runtime_api_poses(edges, chunks=1)

    assert legacy.shape == runtime_api.shape
    np.testing.assert_allclose(runtime_api, legacy, atol=1e-5)


@pytest.mark.parametrize("name", sorted(_STREAMS))
def test_multi_chunk_matches_the_webrtc_path(name: str) -> None:
    """Carried key state across chunk boundaries must agree too."""
    edges = _STREAMS[name]

    legacy = _legacy_poses(edges, chunks=3)
    runtime_api = _runtime_api_poses(edges, chunks=3)

    assert legacy.shape == runtime_api.shape
    np.testing.assert_allclose(runtime_api, legacy, atol=1e-5)
