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

"""User-cached HandDreams scenes prepared from the original AgiBotWorld-Beta dataset."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import av
from filelock import FileLock
from huggingface_hub import cached_assets_path, hf_hub_download, try_to_load_from_cache

from flashdreams.core.io.disk import (
    CACHE_MIN_FREE_ENV,
    DiskSpaceContext,
    cache_min_free_bytes,
    ensure_free_disk,
    raise_if_disk_space_error,
)

from .agibot_download import download_archive_member

_MANIFEST_PATH = Path(__file__).with_name("agibot_scenes.json")
_CONVERSION_VERSION = 2
"""Invalidate prepared scenes when the image or calibration conversion changes."""


def _manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def default_scene_directory() -> Path:
    """Return the versioned HF assets-cache directory without fetching data."""
    digest = hashlib.sha256(_MANIFEST_PATH.read_bytes()).hexdigest()[:16]
    return cached_assets_path(
        "flashdreams",
        namespace="handdreams",
        subfolder=f"agibot-beta-{digest}-v{_CONVERSION_VERSION}",
    )


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _prepared(root: Path, manifest: dict) -> bool:
    try:
        record = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        if (
            record["source"] != manifest
            or record["conversion_version"] != _CONVERSION_VERSION
        ):
            return False
        return all(
            _digest(root / scene["name"] / filename)
            == record["files"][scene["name"]][filename]
            for scene in manifest["scenes"]
            for filename in ("first_frame.jpg", "scene.json")
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _frame(video: Path, index: int):
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != 30 or stream.frames <= index:
            raise ValueError(
                "Expected a 30 FPS AgiBot video containing the selected frame"
            )
        for number, frame in enumerate(container.decode(stream)):
            if number == index:
                return frame.to_image(), stream.frames
    raise ValueError("AgiBot video ended before the selected frame")


def _episode(manifest: dict, scene: dict) -> dict:
    source = dict(
        repo_id=manifest["repo_id"],
        revision=manifest["revision"],
        filename=f"task_info/task_{scene['task']}.json",
    )
    path = try_to_load_from_cache(**source, repo_type="dataset")
    if not isinstance(path, str):
        path = hf_hub_download(
            repo_id=source["repo_id"],
            filename=source["filename"],
            revision=source["revision"],
            repo_type="dataset",
            token=True,
        )
    matches = [
        episode
        for episode in json.loads(Path(path).read_text(encoding="utf-8"))
        if episode.get("episode_id") == scene["episode"]
        and episode.get("task_id", scene["task"]) == scene["task"]
    ]
    if len(matches) != 1:
        raise ValueError(
            "AgiBot task annotations do not identify exactly one selected episode"
        )
    return matches[0]


def _prompt(episode: dict, frame: int) -> str:
    """Use the selected action annotation, or the task before annotations start."""
    labels = episode.get("label_info", episode.get("lable_info", {}))
    for action in labels.get("action_config", []):
        if action["start_frame"] <= frame < action["end_frame"]:
            return action["action_text"]
    return episode["task_name"]


def prepare_default_scenes() -> Path:
    """Prepare the selected scenes during application initialization.

    Fetch source files using the user's HF login and retain only local cache
    outputs. The repository supplies source coordinates, not dataset assets.
    Complete prepared caches work without network access or credentials.
    """
    from .agibot_poses import load_initial_pose
    from .scenes import load_scene_data

    manifest = _manifest()
    root = default_scene_directory()
    with FileLock(str(root / ".prepare.lock")):
        if _prepared(root, manifest):
            return root
        disk_context: DiskSpaceContext = dict(
            label="prepared HandDreams scene cache",
            required_bytes=cache_min_free_bytes(),
            env_vars=("HF_HOME", "HF_ASSETS_CACHE", CACHE_MIN_FREE_ENV),
            settings={"repo": manifest["repo_id"]},
        )
        ensure_free_disk(root, **disk_context)
        print(
            "[handdreams] preparing AgiBotWorld-Beta scenes with your Hugging Face login; "
            "dataset access and its license terms apply",
            flush=True,
        )
        try:
            files = {}
            for scene in manifest["scenes"]:
                print(f"[handdreams] fetching scene: {scene['title']}", flush=True)
                episode = _episode(manifest, scene)
                members = {
                    kind: download_archive_member(
                        repo_id=manifest["repo_id"],
                        revision=manifest["revision"],
                        **scene[kind],
                    )
                    for kind in ("video", "state", "camera")
                }
                image, frame_count = _frame(members["video"], scene["frame"])
                poses, grippers = load_initial_pose(
                    members["state"],
                    members["camera"],
                    frame_index=scene["frame"],
                    frame_count=frame_count,
                    gripper_hardware=scene["gripper_hardware"],
                )
                metadata = {
                    "schema_version": 1,
                    "title": scene["title"],
                    "prompt": _prompt(episode, scene["frame"]),
                    "fps": 30.0,
                    "image_size": [
                        image.height,
                        image.width,
                        image.height,
                        image.width,
                    ],
                    "wrist_poses": {
                        side: pose.tolist() for side, pose in poses.items()
                    },
                    "gripper_open": grippers,
                }
                # Publish both files only after the existing scene validator accepts them.
                with tempfile.TemporaryDirectory(
                    dir=root.parent, prefix=".handdreams-"
                ) as temporary:
                    stage = Path(temporary)
                    image.save(stage / "first_frame.jpg", quality=95)
                    (stage / "scene.json").write_text(
                        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                    )
                    load_scene_data(stage)
                    target = root / scene["name"]
                    target.mkdir(exist_ok=True)
                    files[scene["name"]] = {}
                    for filename in ("first_frame.jpg", "scene.json"):
                        files[scene["name"]][filename] = _digest(stage / filename)
                        (stage / filename).replace(target / filename)
            provenance = {
                "source": manifest,
                "conversion_version": _CONVERSION_VERSION,
                "files": files,
            }
            pending = root / "provenance.json.part"
            pending.write_text(
                json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
            )
            pending.replace(root / "provenance.json")
        except OSError as error:
            raise_if_disk_space_error(error, path=root, **disk_context)
            raise
        finally:
            (root / "provenance.json.part").unlink(missing_ok=True)
    return root
