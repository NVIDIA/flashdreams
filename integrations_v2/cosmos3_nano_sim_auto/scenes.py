# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached prepared residential driving scenes."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from filelock import FileLock
from huggingface_hub import hf_hub_download

from flashdreams.core.io.disk import default_flashdreams_cache_dir

SCENE_REPO_ID = "nvidia/omni-dreams-scenes"
SCENE_NAMES = (
    "clipgt-residential-iter1500",
    "clipgt-residential-rain-iter1500",
    "clipgt-residential-snow-iter1500",
)


def prepare_scene(scene: Path) -> Path:
    """Resolve a local directory or download and extract a built-in scene.

    Call during application initialization, before creating a session. Local
    directories take precedence over built-in names. Archives follow the Hugging
    Face cache settings; extracted bundles follow ``FLASHDREAMS_CACHE_DIR``.

    Args:
        scene: Built-in scene name or local prepared-scene directory.

    Returns:
        Directory containing the scene, matching opening images and prompt.

    Raises:
        ValueError: The downloaded ZIP has an unsafe layout or lacks manifests.
    """
    if scene.is_dir() or str(scene) not in SCENE_NAMES:
        return scene
    archive = Path(
        hf_hub_download(
            repo_id=SCENE_REPO_ID,
            repo_type="dataset",
            filename=f"scenes/synthetic-first-frames/{scene.name}.zip",
        )
    )
    cache = default_flashdreams_cache_dir() / "cosmos3-nano-sim-auto" / "scenes"
    cache.mkdir(parents=True, exist_ok=True)
    # HF blob names identify archive contents, including updated dataset revisions.
    destination = cache / f"{scene.name}-{archive.resolve().name}"
    with FileLock(str(destination) + ".lock"):
        if not destination.is_dir():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                staging = Path(temporary)
                with zipfile.ZipFile(archive) as source:
                    for member in source.infolist():
                        path = PurePosixPath(member.filename)
                        if (
                            path.is_absolute()
                            or ".." in path.parts
                            or "\\" in member.filename
                            or not path.parts
                            or path.parts[0] != scene.name
                        ):
                            raise ValueError(
                                f"Unsafe scene archive member: {member.filename}"
                            )
                    source.extractall(staging)
                bundle = staging / scene.name
                if not all(
                    (bundle / name).is_file() for name in ("scene.json", "sample.json")
                ):
                    raise ValueError(
                        f"Scene archive {archive} lacks scene.json or sample.json"
                    )
                bundle.rename(destination)
    return destination
