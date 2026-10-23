# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for recorded dataset action conversion without controller input."""

from pathlib import Path

import numpy as np
import pytest
from cosmos3_nano_sim_bimanual.action_format import denormalize_action
from cosmos3_nano_sim_bimanual.apps.handdreams import agibot_replay

pytestmark = pytest.mark.ci_cpu


def _tracks(rows=64):
    head = np.tile(np.eye(4, dtype=np.float32), (rows + 1, 1, 1))
    wrists = {side: head.copy() for side in ("left", "right")}
    grips = {side: np.ones(rows + 1, dtype=np.float32) for side in wrists}
    return head, wrists, grips


def test_chunk_anchors_next_frame_grippers_and_padding():
    head, wrists, grips = _tracks()
    positions = np.arange(len(head), dtype=np.float32) * 0.01
    head[:, 0, 3] = positions
    wrists["right"][:, 1, 3] = positions * 2
    wrists["left"][:, 2, 3] = positions * 3
    grips["right"][:] = np.linspace(1, 0, len(head))
    grips["left"][:] = np.linspace(0, 1, len(head))
    actions, domains = agibot_replay.encode_recorded_actions(head, wrists, grips)
    assert actions.shape == (64, 64) and actions.dtype == np.float32
    np.testing.assert_array_equal(domains, np.full(64, 15, dtype=np.int64))
    raw = denormalize_action(actions[:, :59])
    steps = np.tile(np.arange(1, 17), 4) * 0.01
    for start, axis, gain in ((0, 0, 1), (9, 1, 2), (34, 2, 3)):
        np.testing.assert_allclose(raw[:, start + axis], steps * gain, atol=1e-6)
        np.testing.assert_allclose(
            raw[:, start + 3 : start + 9],
            np.tile([1, 0, 0, 0, 1, 0], (64, 1)),
            atol=1e-6,
        )
    np.testing.assert_allclose(raw[:, 33], grips["right"][1:] * 0.12, atol=1e-6)
    np.testing.assert_allclose(raw[:, 58], grips["left"][1:] * 0.12, atol=1e-6)
    for start, stop in ((18, 33), (43, 58), (59, 64)):
        assert not actions[:, start:stop].any()


@pytest.mark.parametrize("rows,domain", [(1, 15), (10, 15), (11, 2), (64, 2)])
def test_offline_mask_marks_entire_stationary_run(rows, domain):
    actions, domains = agibot_replay.encode_recorded_actions(*_tracks(rows))
    np.testing.assert_array_equal(domains, np.full(rows, domain))
    if domain == 2:
        assert not actions[:, 9:].any()


@pytest.mark.parametrize("change", [0, 30, 69])
def test_gripper_changes_guard_past_and_future_including_initial_transition(change):
    head, wrists, grips = _tracks(70)
    grips["left"][change + 1 :] = 0.5
    _, domains = agibot_replay.encode_recorded_actions(head, wrists, grips)
    begin, end = max(0, change - 15), min(70, change + 16)
    expected = np.full(70, 15)
    if begin >= 11:
        expected[:begin] = 2
    if 70 - end >= 11:
        expected[end:] = 2
    np.testing.assert_array_equal(domains, expected)


def test_camera_motion_is_retained_and_mask_scored_relative_to_head():
    head, wrists, grips = _tracks(32)
    head[:, 0, 3] = np.arange(len(head)) * 0.02
    wrists = {side: head.copy() for side in wrists}
    actions, domains = agibot_replay.encode_recorded_actions(head, wrists, grips)
    assert (domains == 2).all()
    assert not actions[:, 9:].any()
    raw = denormalize_action(actions[:, :59])
    np.testing.assert_allclose(
        raw[:, 0], np.tile(np.arange(1, 17), 2) * 0.02, atol=1e-6
    )
    assert raw[:, 0].max() > 0.3


def test_rotation_and_slow_motion_thresholds():
    head, wrists, grips = _tracks(32)
    angle = np.arange(len(head)) * 0.01
    wrists["left"][:, 0, 0] = np.cos(angle)
    wrists["left"][:, 1, 1] = np.cos(angle)
    wrists["left"][:, 0, 1] = -np.sin(angle)
    wrists["left"][:, 1, 0] = np.sin(angle)
    actions, domains = agibot_replay.encode_recorded_actions(head, wrists, grips)
    assert (domains == 15).all()
    expected_angle = np.tile(np.arange(1, 17), 2) * 0.01
    raw = denormalize_action(actions[:, :59])
    np.testing.assert_allclose(raw[:, 37], np.cos(expected_angle), atol=1e-6)
    np.testing.assert_allclose(raw[:, 38], np.sin(expected_angle), atol=1e-6)
    head, wrists, grips = _tracks(32)
    wrists["left"][:, 0, 3] = np.arange(len(head)) * 0.0005
    _, domains = agibot_replay.encode_recorded_actions(head, wrists, grips)
    assert (domains == 2).all()


@pytest.mark.parametrize("bad", ["pose", "gripper_length", "nan", "range", "short"])
def test_malformed_tracks_rejected(bad):
    head, wrists, grips = _tracks()
    if bad == "pose":
        wrists["left"] = wrists["left"][:-1]
    elif bad == "gripper_length":
        grips["left"] = grips["left"][:-1]
    elif bad == "nan":
        grips["right"][10] = np.nan
    elif bad == "range":
        grips["right"][10] = 2
    else:
        head = head[:1]
    with pytest.raises(ValueError):
        agibot_replay.encode_recorded_actions(head, wrists, grips)


def test_preparation_only_replays_tshirt_with_matching_trajectory(
    tmp_path, monkeypatch
):
    entry = {
        "name": "fold-tshirt",
        "frame": 39,
        "gripper_hardware": "robotiq",
        **{kind: {"member_name": kind} for kind in ("video", "state", "camera")},
    }
    manifest = {
        "repo_id": "test/dataset",
        "revision": "a" * 40,
        "scenes": [entry, {"name": "fold-shorts"}, {"name": "open-tissue-box"}],
    }
    calls = []
    monkeypatch.setattr(agibot_replay, "_manifest", lambda: manifest)
    monkeypatch.setattr(
        agibot_replay,
        "prepare_default_scenes",
        lambda **kw: calls.append(kw) or tmp_path,
    )

    def download(**kw):
        assert kw["repo_id"] == "test/dataset" and kw["revision"] == "a" * 40
        return Path(kw["member_name"])

    def frame(path, index):
        assert path == Path("video") and index == 39
        return None, 72

    def poses(state, camera, **kwargs):
        assert state == Path("state") and camera == Path("camera")
        assert kwargs == {
            "frame_index": 39,
            "frame_count": 72,
            "gripper_hardware": "robotiq",
        }
        return _tracks(32)

    monkeypatch.setattr(agibot_replay, "download_archive_member", download)
    monkeypatch.setattr(agibot_replay, "_frame", frame)
    monkeypatch.setattr(agibot_replay, "load_pose_sequence", poses)
    recordings = agibot_replay.prepare_dataset_replays()
    assert tuple(recordings) == ("fold-tshirt",)
    for name, (root, actions, domains) in recordings.items():
        assert root == tmp_path / name
        assert actions.shape == (32, 64) and (domains == 2).all()
    assert calls == [{}]
