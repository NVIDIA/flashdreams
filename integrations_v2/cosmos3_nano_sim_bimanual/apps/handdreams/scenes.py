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

"""Local JPEG/JSON scenes and device-neutral conditioning for HandDreams V2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from handdreams.rotations import validate_pose_track
from handdreams.scenes import ScenePreview
from PIL import Image
from torch import Tensor

from cosmos3_nano_sim_bimanual import action_format
from cosmos3_nano_sim_bimanual.tokenizer import tokenize_prompt

_RUNTIME_CANVAS_HEIGHT = 480
_RUNTIME_CANVAS_WIDTH = 832
"""GA training canvas; the 640x360 content remains in its top-left corner."""


@dataclass(frozen=True, slots=True)
class Scene:
    """One scene's initial conditioning and controller calibration."""

    name: str
    """Stable scene identifier."""
    init_latent: Tensor
    """Patchified frame-zero latent that seeds the rollout."""
    init_rgb: np.ndarray
    """Initial image cropped to the decoded output region, as HWC uint8 pixels."""
    text_ids: Tensor
    """Caption tokens including the generation suffix."""
    fps: float
    """Conditioning cadence in frames per second."""
    content_height: int
    """Height of the image before training-canvas padding."""
    content_width: int
    """Width of the image before training-canvas padding."""
    wrist_rotations: dict[str, np.ndarray]
    """Initial wrist orientations in camera coordinates."""
    gripper_open: dict[str, float]
    """Initial gripper openness for each hand."""
    wrist_poses: dict[str, np.ndarray]
    """Initial camera-coordinate wrist transforms for the setup guide."""


def _resize_rgb_content(rgb: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Match training's antialiased bicubic tensor resize and uint8 rounding."""
    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"AgiBot RGB frame must be [H,W,3], got {array.shape}")
    if array.shape[:2] == (height, width):
        return np.ascontiguousarray(array)
    image = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    resized = torch.nn.functional.interpolate(
        image[None].float(),
        size=(int(height), int(width)),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return (
        resized.clamp(0, 255)
        .round()
        .to(torch.uint8)[0]
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )


def _prepare_runtime_canvas(
    rgb: np.ndarray,
    *,
    source_content_height: int,
    source_content_width: int,
) -> np.ndarray:
    """Correct AgiBot's stretched 4:3 content and recreate training padding."""
    source = np.asarray(rgb, dtype=np.uint8)[
        : int(source_content_height), : int(source_content_width)
    ]
    content = _resize_rgb_content(
        source,
        height=action_format.CONTENT_HEIGHT,
        width=action_format.CONTENT_WIDTH,
    )
    padding = (
        (0, _RUNTIME_CANVAS_HEIGHT - action_format.CONTENT_HEIGHT),
        (0, _RUNTIME_CANVAS_WIDTH - action_format.CONTENT_WIDTH),
        (0, 0),
    )
    return np.ascontiguousarray(np.pad(content, padding, mode="reflect"))


def resolve_scene_path(scene_path: str | Path) -> Path:
    """Resolve a local scene directory containing its JPEG and JSON files."""
    path = Path(scene_path).expanduser().resolve()
    for filename in ("scene.json", "first_frame.jpg"):
        if not (path / filename).is_file():
            raise FileNotFoundError(
                f"HandDreams V2 scene is missing: {path / filename}"
            )
    return path


def validate_scene_sample(
    data: dict[str, Any], *, init_rgb: np.ndarray
) -> dict[str, Any]:
    """Validate JSON metadata and its decoded first frame before model encoding."""
    required = {
        "schema_version",
        "title",
        "prompt",
        "fps",
        "image_size",
        "wrist_poses",
        "gripper_open",
    }
    if not isinstance(data, dict):
        raise ValueError("HandDreams scene.json must contain an object")
    if set(data) != required:
        raise ValueError(
            f"HandDreams scene.json fields: missing {sorted(required - data.keys())}, "
            f"unexpected {sorted(data.keys() - required)}"
        )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("HandDreams schema_version must be 1")
    for field in ("title", "prompt"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise ValueError(f"HandDreams {field} must be a nonempty string")
    fps = data["fps"]
    if type(fps) not in (int, float) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("HandDreams fps must be a finite positive number")

    image_size = data["image_size"]
    if (
        not isinstance(image_size, list)
        or len(image_size) != 4
        or any(type(value) is not int or value <= 0 for value in image_size)
    ):
        raise ValueError(
            "HandDreams image_size must contain four positive integers "
            "[target_h,target_w,content_h,content_w]"
        )
    target_h, target_w, content_h, content_w = image_size
    if init_rgb.shape != (target_h, target_w, 3):
        raise ValueError(
            f"HandDreams image_size target {target_h}x{target_w} "
            f"does not match RGB image {init_rgb.shape}"
        )
    if content_h > target_h or content_w > target_w:
        raise ValueError("HandDreams image_size content must fit its target")

    for field in ("wrist_poses", "gripper_open"):
        if not isinstance(data[field], dict) or set(data[field]) != {"left", "right"}:
            raise ValueError(f"HandDreams {field} requires both left and right hands")
    poses = {}
    grippers = {}
    for side in ("left", "right"):
        pose = np.asarray(data["wrist_poses"][side])
        if pose.dtype.kind not in "fiu":
            raise ValueError(f"HandDreams wrist_poses.{side} must contain numbers")
        poses[side] = validate_pose_track(
            pose[None], f"wrist_poses.{side}", minimum_frames=1
        )[0].copy()
        opened = data["gripper_open"][side]
        if (
            type(opened) not in (int, float)
            or not np.isfinite(opened)
            or not 0 <= opened <= 1
        ):
            raise ValueError(f"HandDreams gripper_open.{side} must be in [0,1]")
        grippers[side] = float(opened)
    return {
        **data,
        "init_rgb": init_rgb,
        "fps": float(fps),
        "content_height": content_h,
        "content_width": content_w,
        "wrist_poses": poses,
        "gripper_open": grippers,
    }


def load_scene_data(scene_path: str | Path) -> dict[str, Any]:
    """Read and validate one local ``scene.json`` / ``first_frame.jpg`` pair."""
    path = resolve_scene_path(scene_path)
    with (path / "scene.json").open(encoding="utf-8") as stream:
        data = json.load(stream)
    try:
        with Image.open(path / "first_frame.jpg") as image:
            if image.format != "JPEG":
                raise ValueError("first_frame.jpg must contain a JPEG image")
            init_rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid HandDreams first_frame.jpg in {path}") from error
    return validate_scene_sample(data, init_rgb=np.ascontiguousarray(init_rgb))


def discover_scenes(
    *,
    sample: str | None = None,
    samples: tuple[str, ...] | None = None,
    scene_path: str | Path | None = None,
    sample_dir: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Select exact scene names in the given order, or discover the full catalog."""
    if scene_path is not None and sample_dir is not None:
        raise ValueError("scene_path and sample_dir are mutually exclusive")
    if scene_path is not None:
        path = resolve_scene_path(scene_path)
        return [(sample or path.name, str(path))]
    if sample_dir is not None:
        root = Path(sample_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"sample directory does not exist: {root}")
        found: dict[str, Path] = {}
        for metadata in sorted(root.rglob("scene.json")):
            path = resolve_scene_path(metadata.parent)
            name = path.name
            if name in found:
                raise ValueError(
                    f"duplicate sample name {name!r}: {found[name]} and {path}"
                )
            found[name] = path
        if not found:
            raise ValueError(
                f"no scene.json / first_frame.jpg scenes found below {root}"
            )
        names = list(found) if samples is None else samples
        missing = [name for name in names if name not in found]
        if missing:
            raise ValueError(
                f"requested HandDreams samples are absent from {root}: {', '.join(missing)}"
            )
        return [(name, str(found[name])) for name in names]
    raise ValueError(
        "No scenarios are packaged with HandDreams. "
        "Supply --sample-dir /path/to/scenarios or --scene-path /path/to/scene."
    )


def load_scene(
    name: str,
    scene_path: str | Path,
    *,
    encode: Any,
    patchify: Any,
    device: str | torch.device = "cuda",
) -> Scene:
    """Encode a local scene's first frame and prompt for a fresh rollout."""
    sample = load_scene_data(scene_path)
    canvas = _prepare_runtime_canvas(
        sample["init_rgb"],
        source_content_height=sample["content_height"],
        source_content_width=sample["content_width"],
    )
    content_h, content_w = action_format.CONTENT_HEIGHT, action_format.CONTENT_WIDTH
    wrist_poses = sample["wrist_poses"]
    text_ids = tokenize_prompt(sample["prompt"])
    return Scene(
        name=name,
        init_latent=patchify(
            encode(torch.from_numpy(canvas)),
            content_height=content_h,
            content_width=content_w,
        ),
        init_rgb=np.ascontiguousarray(
            canvas[: action_format.OUTPUT_HEIGHT, : action_format.OUTPUT_WIDTH]
        ),
        text_ids=text_ids.to(device),
        fps=sample["fps"],
        content_height=content_h,
        content_width=content_w,
        wrist_rotations={
            side: pose[:3, :3].copy() for side, pose in wrist_poses.items()
        },
        gripper_open=sample["gripper_open"],
        wrist_poses=wrist_poses,
    )


def scene_previews(sources: list[tuple[str, str]]) -> tuple[ScenePreview, ...]:
    """Load titles and first-frame previews from the resolved scene directories."""
    previews = []
    for name, source in sources:
        sample = load_scene_data(source)
        canvas = _prepare_runtime_canvas(
            sample["init_rgb"],
            source_content_height=sample["content_height"],
            source_content_width=sample["content_width"],
        )
        thumbnail = np.ascontiguousarray(
            canvas[: action_format.OUTPUT_HEIGHT, : action_format.OUTPUT_WIDTH]
        )
        thumbnail.setflags(write=False)
        previews.append(ScenePreview(name, sample["title"], thumbnail))
    return tuple(previews)
