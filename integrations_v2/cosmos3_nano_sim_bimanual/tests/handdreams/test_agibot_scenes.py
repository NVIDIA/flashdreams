# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline checks for authenticated scene preparation and immutable cache identity."""

import errno
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from cosmos3_nano_sim_bimanual.apps.handdreams import agibot_poses, agibot_scenes
from cosmos3_nano_sim_bimanual.apps.handdreams.agibot_poses import (
    load_initial_pose,
)
from cosmos3_nano_sim_bimanual.apps.handdreams.scenes import load_scene_data
from huggingface_hub import _CACHED_NO_EXIST
from PIL import Image

from flashdreams.core.io import disk

pytestmark = pytest.mark.ci_cpu


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    root = tmp_path / "prepared"
    root.mkdir()
    scene = {
        "name": "synthetic",
        "title": "Synthetic scene",
        "task": 1,
        "episode": 2,
        "frame": 39,
        "gripper_hardware": "robotiq",
        **{kind: {"member_name": kind} for kind in ("video", "state", "camera")},
    }
    manifest = {"repo_id": "test/dataset", "revision": "a" * 40, "scenes": [scene]}
    task = tmp_path / "task_1.json"
    # Public task files identify their task in the filename, not each episode.
    task.write_text(json.dumps([{"episode_id": 2, "task_name": "Fold it"}]))
    calls = []

    def hub(**kwargs):
        calls.append(("hub", kwargs))
        return str(task)

    def member(**kwargs):
        calls.append(("member", kwargs))
        return tmp_path / kwargs["member_name"]

    def frame(path, index):
        assert path == tmp_path / "video" and index == 39
        return Image.new("RGB", (64, 48), (30, 40, 50)), 100

    def pose(state, camera, **kwargs):
        assert state == tmp_path / "state" and camera == tmp_path / "camera"
        assert kwargs == {
            "frame_index": 39,
            "frame_count": 100,
            "gripper_hardware": "robotiq",
        }
        return {side: np.eye(4) for side in ("left", "right")}, {
            "left": 0.7,
            "right": 0.6,
        }

    monkeypatch.setattr(agibot_scenes, "_manifest", lambda: manifest)
    monkeypatch.setattr(agibot_scenes, "default_scene_directory", lambda: root)
    monkeypatch.setattr(agibot_scenes, "hf_hub_download", hub)
    monkeypatch.setattr(agibot_scenes, "try_to_load_from_cache", lambda **_: None)
    monkeypatch.setattr(agibot_scenes, "download_archive_member", member)
    monkeypatch.setattr(agibot_scenes, "_frame", frame)
    monkeypatch.setattr(agibot_poses, "load_initial_pose", pose)
    monkeypatch.setenv(disk.CACHE_MIN_FREE_ENV, "0")
    return root, manifest, calls


def test_preparation_uses_caller_hf_auth_and_preserves_conditioning(preparation):
    root, manifest, calls = preparation
    assert agibot_scenes.prepare_default_scenes() == root
    assert calls[0] == (
        "hub",
        {
            "repo_id": "test/dataset",
            "repo_type": "dataset",
            "revision": "a" * 40,
            "filename": "task_info/task_1.json",
            "token": True,
        },
    )
    assert [kwargs["member_name"] for kind, kwargs in calls if kind == "member"] == [
        "video",
        "state",
        "camera",
    ]
    data = json.loads((root / "synthetic/scene.json").read_text())
    assert data["prompt"] == "Fold it"
    assert data["image_size"] == [48, 64, 48, 64]
    assert data["fps"] == 30.0
    assert data["gripper_open"] == {"left": 0.7, "right": 0.6}
    for pose in data["wrist_poses"].values():
        np.testing.assert_array_equal(pose, np.eye(4))
    assert set(p.name for p in (root / "synthetic").iterdir()) == {
        "scene.json",
        "first_frame.jpg",
    }
    record = json.loads((root / "provenance.json").read_text())
    assert record["source"] == manifest
    assert record["conversion_version"] == 2


def test_complete_cache_needs_no_network_or_credentials(preparation, monkeypatch):
    root, _, _ = preparation
    agibot_scenes.prepare_default_scenes()
    for function in ("hf_hub_download", "download_archive_member", "_frame"):
        monkeypatch.setattr(
            agibot_scenes, function, lambda *a, **k: pytest.fail("cache fetched")
        )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert agibot_scenes.prepare_default_scenes() == root


@pytest.mark.parametrize("raw,opened", [([35, 120], [1, 0]), ([77.5, 35], [0.5, 1])])
def test_preparation_calibrates_raw_state_once(preparation, monkeypatch, raw, opened):
    root, _, calls = preparation
    with h5py.File(root.parent / "state", "w") as state:
        state["state/end/position"] = np.zeros((100, 2, 3))
        state["state/end/orientation"] = np.tile([0, 0, 0, 1], (100, 2, 1))
        state["state/effector/position"] = np.tile(raw, (100, 1))
    (root.parent / "camera").write_text(
        json.dumps(
            [
                {
                    "extrinsic": {
                        "rotation_matrix": np.eye(3).tolist(),
                        "translation_vector": [0, 0, 0],
                    }
                }
            ]
            * 100
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(agibot_poses, "load_initial_pose", load_initial_pose)
    agibot_scenes.prepare_default_scenes()
    expected = dict(zip(("left", "right"), opened, strict=True))
    assert load_scene_data(root / "synthetic")["gripper_open"] == expected

    def unexpected(*args, **kwargs):
        pytest.fail("prepared scene must not be downloaded or converted again")

    monkeypatch.setattr(agibot_poses, "load_initial_pose", unexpected)
    monkeypatch.setattr(agibot_scenes, "download_archive_member", unexpected)
    calls.clear()
    assert agibot_scenes.prepare_default_scenes() == root
    assert not calls
    assert load_scene_data(root / "synthetic")["gripper_open"] == expected


@pytest.mark.parametrize("filename", ["first_frame.jpg", "scene.json"])
def test_modified_output_is_not_a_completed_cache(preparation, filename):
    root, _, calls = preparation
    agibot_scenes.prepare_default_scenes()
    (root / "synthetic" / filename).write_bytes(b"incomplete")
    calls.clear()
    agibot_scenes.prepare_default_scenes()
    assert len(calls) == 4
    assert agibot_scenes._prepared(root, agibot_scenes._manifest())


def test_legacy_gripper_conversion_cache_is_rebuilt(preparation):
    root, _, calls = preparation
    agibot_scenes.prepare_default_scenes()
    provenance = root / "provenance.json"
    record = json.loads(provenance.read_text())
    record["conversion_version"] = 1
    provenance.write_text(json.dumps(record))
    assert not agibot_scenes._prepared(root, agibot_scenes._manifest())
    calls.clear()
    agibot_scenes.prepare_default_scenes()
    assert len(calls) == 4
    assert json.loads(provenance.read_text())["conversion_version"] == 2


def test_failed_conversion_does_not_publish_a_ready_scene(preparation, monkeypatch):
    root, _, _ = preparation

    def fail(*args, **kwargs):
        raise ValueError("mismatched source frame counts")

    monkeypatch.setattr(agibot_poses, "load_initial_pose", fail)
    with pytest.raises(ValueError, match="frame counts"):
        agibot_scenes.prepare_default_scenes()
    assert not (root / "provenance.json").exists()
    assert not (root / "synthetic").exists()


def test_conversion_disk_preflight_prevents_fetch(preparation, monkeypatch):
    root, _, calls = preparation
    monkeypatch.setenv(disk.CACHE_MIN_FREE_ENV, "1")
    monkeypatch.setattr(disk, "available_bytes", lambda _: 0)
    with pytest.raises(disk.DiskSpaceError, match="HF_ASSETS_CACHE"):
        agibot_scenes.prepare_default_scenes()
    assert not calls
    assert not (root / "provenance.json").exists()


@pytest.mark.parametrize("filename", ["scene.json", "provenance.json.part"])
def test_conversion_disk_exhaustion_cleans_staging(preparation, monkeypatch, filename):
    root, _, _ = preparation
    original = Path.write_text

    def no_space(path, *args, **kwargs):
        if path.name == filename:
            path.write_bytes(b"partial")
            raise OSError(errno.ENOSPC, "No space left on device")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", no_space)
    with pytest.raises(disk.DiskSpaceError, match="HF_ASSETS_CACHE"):
        agibot_scenes.prepare_default_scenes()
    assert not (root / "provenance.json").exists()
    assert not (root / "provenance.json.part").exists()
    assert not list(root.parent.glob(".handdreams-*"))


def test_cache_identity_changes_with_source_selection(tmp_path, monkeypatch):
    manifest = tmp_path / "sources.json"
    manifest.write_text("first")
    monkeypatch.setattr(agibot_scenes, "_MANIFEST_PATH", manifest)
    monkeypatch.setattr(
        agibot_scenes,
        "cached_assets_path",
        lambda library, namespace, subfolder: tmp_path / subfolder,
    )
    original = agibot_scenes.default_scene_directory()
    monkeypatch.setattr(agibot_scenes, "_CONVERSION_VERSION", 1)
    assert agibot_scenes.default_scene_directory() != original
    monkeypatch.setattr(agibot_scenes, "_CONVERSION_VERSION", 2)
    manifest.write_text("second")
    assert agibot_scenes.default_scene_directory() != original


def test_fetch_list_pins_the_three_selected_beta_episodes():
    manifest = agibot_scenes._manifest()
    assert manifest["repo_id"] == "agibot-world/AgiBotWorld-Beta"
    assert manifest["revision"] == "2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278"
    assert [
        (s["name"], s["task"], s["episode"], s["frame"]) for s in manifest["scenes"]
    ] == [
        ("fold-tshirt", 570, 816412, 0),
        ("fold-shorts", 599, 822065, 0),
        ("open-tissue-box", 522, 754102, 0),
    ]
    for scene in manifest["scenes"]:
        assert scene["gripper_hardware"] == "robotiq"
        assert (
            scene["video"]["member_name"] == f"{scene['episode']}/videos/head_color.mp4"
        )
        assert scene["video"]["archive"].startswith(f"observations/{scene['task']}/")
        for kind in ("video", "state", "camera"):
            assert len(scene[kind]["sha256"]) == 64
            assert scene[kind]["offset"] % 512 == 0
            assert 0 < scene[kind]["size"] < 512 * 1024 * 1024


@pytest.mark.parametrize("key", ["label_info", "lable_info"])
def test_prompt_matches_selected_action_and_falls_back_before_it(key):
    episode = {
        "task_name": "General task",
        key: {
            "action_config": [
                {"start_frame": 39, "end_frame": 60, "action_text": "Fold upwards"}
            ]
        },
    }
    assert agibot_scenes._prompt(episode, 0) == "General task"
    assert agibot_scenes._prompt(episode, 39) == "Fold upwards"
    assert agibot_scenes._prompt(episode, 59) == "Fold upwards"
    assert agibot_scenes._prompt(episode, 60) == "General task"


@pytest.mark.parametrize("task_id,episode_id", [(1, 3), (3, 2)])
def test_episode_lookup_rejects_wrong_public_episode(
    preparation, monkeypatch, tmp_path, task_id, episode_id
):
    wrong = tmp_path / "wrong.json"
    wrong.write_text(
        json.dumps(
            [{"task_id": task_id, "episode_id": episode_id, "task_name": "Wrong"}]
        )
    )
    monkeypatch.setattr(agibot_scenes, "hf_hub_download", lambda **_: wrong)
    with pytest.raises(ValueError, match="exactly one"):
        agibot_scenes.prepare_default_scenes()


def test_cached_sources_rebuild_scenes_without_credentials(preparation, monkeypatch):
    root, _, calls = preparation

    def cached(**kwargs):
        assert kwargs == {
            "repo_id": "test/dataset",
            "repo_type": "dataset",
            "revision": "a" * 40,
            "filename": "task_info/task_1.json",
        }
        return str(root.parent / "task_1.json")

    monkeypatch.setattr(agibot_scenes, "try_to_load_from_cache", cached)
    monkeypatch.setattr(
        agibot_scenes,
        "hf_hub_download",
        lambda **_: pytest.fail("requested credentials"),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert agibot_scenes.prepare_default_scenes() == root
    assert agibot_scenes._prepared(root, agibot_scenes._manifest())
    assert all(kind == "member" for kind, _ in calls)


@pytest.mark.parametrize("cached", [None, _CACHED_NO_EXIST])
def test_annotation_cache_miss_requires_authentication(
    preparation, monkeypatch, cached
):
    _, manifest, calls = preparation
    monkeypatch.setattr(agibot_scenes, "try_to_load_from_cache", lambda **_: cached)
    agibot_scenes._episode(manifest, manifest["scenes"][0])
    assert calls == [
        (
            "hub",
            {
                "repo_id": "test/dataset",
                "repo_type": "dataset",
                "revision": "a" * 40,
                "filename": "task_info/task_1.json",
                "token": True,
            },
        )
    ]
