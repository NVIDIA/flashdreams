# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scenes: the road a drawn run drives along, and what it starts from."""

import zipfile
from pathlib import Path

from omnidreams.interactive_drive.math3d import normalize_camera_name
from omnidreams.scenes import (
    SCENE_FRAME_SUFFIXES,
    SCENE_FRAMES_DIRNAME,
    hf_hub_download_scene,
    hf_scenes_repo_id,
    scenes_cache_root,
)

DEFAULT_SCENE = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
"""Scene a run drives when it asks for one without naming it."""

DEFAULT_SCENE_CAMERA = "camera:front:wide:120fov"
"""Camera to draw from, being the one the shipped single-view model reads."""

_PROMPT_ENTRIES = ("prompt.txt", "prompt1.txt", "prompt_1.txt")
"""Where a scene keeps its own description of the drive, best first."""

_SEED_FRAME_DIRNAME = "v2-seed-frames"
"""Where frames unpacked from scene archives are kept, under the scene cache."""


def fetch_scene(scene: str) -> Path:
    """Return the local archive for a scene, downloading it if needed.

    Args:
        scene: A path to a scene archive, or the id of one to download.

    Returns:
        The archive on disk. A downloaded scene lands in the Hugging Face
        cache, so asking for the same one again costs nothing.

    Raises:
        FileNotFoundError: The scene is neither a path that exists nor an id
            the scenes dataset has.
    """
    named = Path(scene)
    if named.exists():
        return named
    try:
        return hf_hub_download_scene(scene)
    except Exception as exc:
        raise FileNotFoundError(
            f"No scene {scene!r}: there is no such path, and "
            f"{hf_scenes_repo_id()} has no such scene ({exc})."
        ) from exc


def read_seed_frame(scene_path: Path, *, camera: str) -> tuple[Path, int]:
    """Unpack the earliest recorded frame for one camera of a scene.

    This is the frame a drawn run continues from. Its timestamp comes back with
    it because that is also where the road layout has to start being drawn: a
    run that continued from one moment while being shown the road at another
    would be asked to drive a corner it cannot see.

    Args:
        scene_path: Scene archive to read.
        camera: Camera whose recording to take, in either spelling.

    Returns:
        The unpacked image, and when it was captured, in microseconds.

    Raises:
        FileNotFoundError: The scene has no timestamped frames for that camera,
            so there is nothing to continue from.
    """
    _, logical_name = normalize_camera_name(camera)
    with zipfile.ZipFile(scene_path) as archive:
        entry, captured_us = _earliest_frame(archive, scene_path, camera)
        filename = f"{scene_path.stem}-{logical_name}-{Path(entry).name}"
        unpacked = scenes_cache_root() / _SEED_FRAME_DIRNAME / filename
        if not unpacked.exists():
            unpacked.parent.mkdir(parents=True, exist_ok=True)
            # Written beside the cached scene rather than to a temporary
            # directory, so a second run of the same command reuses it.
            unpacked.write_bytes(archive.read(entry))
    return unpacked, captured_us


def read_prompt(scene_path: Path) -> str | None:
    """Return a scene's own description of the drive, or ``None`` if it has none.

    Preferred over the model's generic prompt, since a description of this road
    in this weather is what the drive being drawn actually looks like.
    """
    with zipfile.ZipFile(scene_path) as archive:
        entries = set(archive.namelist())
        for candidate in _PROMPT_ENTRIES:
            if candidate in entries:
                return archive.read(candidate).decode("utf-8").strip() or None
    return None


def _earliest_frame(
    archive: zipfile.ZipFile, scene_path: Path, camera: str
) -> tuple[str, int]:
    """Return the first recorded frame for a camera, and its timestamp.

    Frames are kept as ``frames/<camera>/<microseconds>.jpeg``, so the earliest
    is the smallest of those names. Both spellings of a camera are looked for,
    since scenes have been staged with each.

    Raises:
        FileNotFoundError: No frame of that camera is named after a timestamp.
    """
    clipgt_name, logical_name = normalize_camera_name(camera)
    prefixes = tuple(
        f"{SCENE_FRAMES_DIRNAME}/{name}/"
        for name in {camera, clipgt_name, logical_name}
    )
    captured: list[tuple[int, str]] = []
    for entry in archive.namelist():
        name = Path(entry)
        if not entry.startswith(prefixes):
            continue
        if name.suffix.lower() not in SCENE_FRAME_SUFFIXES:
            continue
        if name.stem.isdigit():
            captured.append((int(name.stem), entry))
    if not captured:
        raise FileNotFoundError(
            f"Scene {scene_path} has no timestamped frames for camera "
            f"{camera!r} under {SCENE_FRAMES_DIRNAME}/, so there is no "
            "recorded frame for a run to continue from."
        )
    captured_us, entry = min(captured)
    return entry, captured_us
