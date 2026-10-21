# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned action anchors follow diffusion acceptance and rollout resets."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_bimanual.apps.handdreams.actions import (
    ControllerActionState,
    encode_controller_actions,
)
from cosmos3_nano_sim_bimanual.apps.handdreams.adapter import (
    CosmosHandDreamsRolloutBuilder,
)
from cosmos3_nano_sim_bimanual.encoder import HandActionInput

from .test_controller_actions import _targets
from .test_stationary_actions import _stationary_targets

pytestmark = pytest.mark.ci_cpu


class _Pipeline:
    actions_per_step = 8

    def __init__(self):
        self.reject = False
        self.fail_decode = False
        self.received = []

    def initialize_cache(self, **_kwargs):
        return object()

    def generate(self, _index, _cache, *, input, on_diffusion_complete):
        self.received.append(input)
        if self.reject:
            raise RuntimeError("diffusion rejected")
        on_diffusion_complete()
        if self.fail_decode:
            raise RuntimeError("decode failed")
        return input

    def release_session(self, _cache):
        pass

    def prepare_scene(self, _scene):
        pass


def _builder():
    return CosmosHandDreamsRolloutBuilder(
        pipeline=cast(Any, _Pipeline()),
        scene=cast(
            Any,
            SimpleNamespace(
                wrist_rotations={side: np.eye(3) for side in ("left", "right")},
                fps=30.0,
            ),
        ),
    )


def test_rejection_retries_anchors_and_decode_failure_preserves_acceptance() -> None:
    builder = _builder()
    cache = builder.build_cache()
    frames = _targets(25)
    actions = builder.encode_actions(frames[:9])
    builder.pipeline.reject = True
    accepted = []
    with pytest.raises(RuntimeError, match="diffusion rejected"):
        builder.generate(0, cache, actions, lambda: accepted.append(True))
    assert builder._action_state.rows == 0
    assert builder._pending_action_state is None
    assert builder._pending_domain_ids is None
    assert not accepted

    builder.pipeline.reject = False
    repeated = builder.encode_actions(frames[:9])
    torch.testing.assert_close(repeated, actions, rtol=0, atol=0)
    builder.generate(0, cache, repeated, lambda: accepted.append(True))
    assert builder._action_state.rows == 8
    assert accepted == [True]

    builder.pipeline.fail_decode = True
    actions = builder.encode_actions(frames[8:17])
    with pytest.raises(RuntimeError, match="decode failed"):
        builder.generate(1, cache, actions, lambda: accepted.append(True))
    assert builder._action_state.rows == 16
    assert builder._pending_action_state is None
    assert accepted == [True, True]


def test_reset_and_scene_replacement_do_not_share_action_anchors() -> None:
    builder = _builder()
    cache = builder.build_cache()
    frames = _stationary_targets(25)
    initial = builder.encode_actions(frames[:9])
    builder.generate(0, cache, initial, lambda: None)
    other = _builder()
    assert other._action_state.rows == 0
    builder.prepare_scene()
    assert builder._action_state.rows == 8

    builder.encode_actions(frames[8:17])
    builder.release_cache(cache)
    assert builder._action_state.rows == 0
    assert builder._action_state.stationary_rows == 0
    assert builder._pending_action_state is None
    assert builder._pending_domain_ids is None
    builder.build_cache()
    torch.testing.assert_close(
        builder.encode_actions(frames[:9]), initial, rtol=0, atol=0
    )


def test_generation_requires_a_prepared_action_window() -> None:
    builder = _builder()
    with pytest.raises(RuntimeError, match="encode controller actions"):
        builder.generate(0, builder.build_cache(), torch.empty(8, 64), lambda: None)


def test_invalid_replacement_window_cannot_accept_a_stale_proposal() -> None:
    builder = _builder()
    cache = builder.build_cache()
    frames = _targets(9)
    actions = builder.encode_actions(frames)
    del frames[4]["left"]
    with pytest.raises(ValueError, match="requires the left controller"):
        builder.encode_actions(frames)
    assert builder._action_state.rows == 0
    assert builder._pending_action_state is None
    assert builder._pending_domain_ids is None
    with pytest.raises(RuntimeError, match="encode controller actions"):
        builder.generate(0, cache, actions, lambda: None)


def test_moving_targets_preserve_codec_values_and_use_explicit_agibot_domains() -> None:
    builder = _builder()
    cache = builder.build_cache()
    frames = _targets(25)
    expected, _ = encode_controller_actions(
        frames, builder.scene, ControllerActionState()
    )
    actual = builder.encode_actions(frames)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert builder._pending_domain_ids.tolist() == [15] * 24
    builder.generate(0, cache, actual, lambda: None)
    payload = builder.pipeline.received[-1]
    assert isinstance(payload, HandActionInput)
    torch.testing.assert_close(payload.actions, expected, rtol=0, atol=0)
    assert payload.domain_ids.dtype == torch.long
    assert payload.domain_ids.tolist() == [15] * 24


def test_camera_rows_keep_normalized_head_and_mask_every_hand_channel() -> None:
    builder = _builder()
    frames = _stationary_targets(25)
    baseline, _ = encode_controller_actions(
        frames, builder.scene, ControllerActionState()
    )
    actual = builder.encode_actions(frames)
    torch.testing.assert_close(actual[:10], baseline[:10], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, :9], baseline[:, :9], rtol=0, atol=0)
    assert torch.count_nonzero(actual[10:, 9:]) == 0
    assert builder._pending_domain_ids.tolist() == [15] * 10 + [2] * 14
    # The raw identity camera is normalized, not sent as raw rotation coefficients.
    assert not torch.equal(actual[10, :9], torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0]))


def test_camera_routing_spans_chunks_and_preserves_manipulation_anchors() -> None:
    frames = _stationary_targets(49)
    # Resume both motion and gripper commands after an idle stretch.
    for i, frame in enumerate(frames[29:]):
        frame["left"]["position"][0] += 0.02 * (i + 1)
        frame["left"]["gripper_open"] = 0.5
    reference = _builder()
    expected = reference.encode_actions(frames)
    expected_domains = reference._pending_domain_ids.clone()
    builder = _builder()
    cache = builder.build_cache()
    chunks = []
    domains = []
    for frame in frames:
        frame["client_capture_ns"] += 100_000_000_000
    for step, start in enumerate(range(0, 48, 8)):
        actions = builder.encode_actions(frames[start : start + 9])
        chunks.append(actions)
        domains.append(builder._pending_domain_ids.clone())
        builder.generate(step, cache, actions, lambda: None)
        payload = builder.pipeline.received[-1]
        assert isinstance(payload, HandActionInput)
        torch.testing.assert_close(payload.actions, actions, rtol=0, atol=0)
        torch.testing.assert_close(payload.domain_ids, domains[-1], rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(chunks), expected, rtol=0, atol=0)
    assert builder._action_state.rows == 48
    torch.testing.assert_close(torch.cat(domains), expected_domains, rtol=0, atol=0)
    assert expected_domains[28:].tolist() == [15] * 20


def test_camera_classifier_advances_only_when_diffusion_accepts() -> None:
    builder = _builder()
    cache = builder.build_cache()
    frames = _stationary_targets(25)
    first = builder.encode_actions(frames[:9])
    builder.generate(0, cache, first, lambda: None)
    assert builder._action_state.stationary_rows == 8

    second = builder.encode_actions(frames[8:17])
    expected_domains = builder._pending_domain_ids.clone()
    builder.pipeline.reject = True
    with pytest.raises(RuntimeError, match="diffusion rejected"):
        builder.generate(1, cache, second, lambda: None)
    assert builder._action_state.stationary_rows == 8
    assert builder._pending_domain_ids is None
    repeated = builder.encode_actions(frames[8:17])
    torch.testing.assert_close(repeated, second, rtol=0, atol=0)
    torch.testing.assert_close(builder._pending_domain_ids, expected_domains)
    builder.pipeline.reject = False
    builder.pipeline.fail_decode = True
    with pytest.raises(RuntimeError, match="decode failed"):
        builder.generate(1, cache, repeated, lambda: None)
    assert builder._action_state.stationary_rows == 11
    assert builder._action_state.rows == 16
