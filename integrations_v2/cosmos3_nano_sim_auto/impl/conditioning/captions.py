# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the structured camera prompts expected by the checkpoint.

The model uses one JSON caption object per camera, with camera direction,
resolution and timing metadata. :func:`view_caption_payloads` builds separate
prompts; :func:`multiview_caption` supports checkpoints using a joint views list.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from cosmos_interactive._tokenizer import DEFAULT_TOKENIZER, tokenize_prompt
from torch import Tensor

CAMERA_CAPTION_PREFIXES: dict[str, str] = {
    "camera_front_wide_120fov": (
        "The video is captured from a camera mounted on a car. The camera is facing forward."
    ),
    "camera_cross_right_120fov": (
        "The video is captured from a camera mounted on a car. The camera is facing to the right."
    ),
    "camera_rear_right_70fov": (
        "The video is captured from a camera mounted on a car. The camera is facing the rear right side."
    ),
    "camera_rear_tele_30fov": (
        "The video is captured from a camera mounted on a car. The camera is facing backwards."
    ),
    "camera_rear_left_70fov": (
        "The video is captured from a camera mounted on a car. The camera is facing the rear left side."
    ),
    "camera_cross_left_120fov": (
        "The video is captured from a camera mounted on a car. The camera is facing to the left."
    ),
    "camera_front_tele_30fov": (
        "The video is captured from a telephoto camera mounted on a car. The camera is facing forward."
    ),
    "camera_front_fisheye_200fov": (
        "The video is captured from a fisheye camera mounted on a car. The camera is facing forward."
    ),
    "camera_left_fisheye_200fov": (
        "The video is captured from a fisheye camera mounted on a car. The camera is facing to the left."
    ),
    "camera_right_fisheye_200fov": (
        "The video is captured from a fisheye camera mounted on a car. The camera is facing to the right."
    ),
    "camera_rear_fisheye_200fov": (
        "The video is captured from a fisheye camera mounted on a car. The camera is facing backwards."
    ),
}
"""Camera-direction prefixes; preserve the exact wording used during training."""

_AV_WSM_CONTROL_INSTRUCTION = (
    "Follow WSM controls for vehicles (including trucks), cyclists, pedestrians, "
    "traffic lights, traffic signs, road markings, lane boundaries, and road boundaries. "
    "Do not add objects or road features in these categories that are absent from WSM. "
    "Use captions for appearance and unconstrained background details; WSM takes precedence in any conflict."
)

AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates temporally synchronized, geometrically consistent autonomous-driving "
    "videos from per-camera scene descriptions and World Scenario Map (WSM) control videos depicting the controlled "
    "objects and road layout. Treat all camera views as simultaneous observations of the same driving scene, "
    "preserving each camera's viewpoint, shared ego motion, road layout, object identity and motion, weather, "
    f"lighting, and cross-view consistency.\n\n{_AV_WSM_CONTROL_INSTRUCTION}"
)
"""Exact ``av_multiview_transfer`` system message for RGB generation."""

AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT = (
    "You are a helpful assistant that jointly generates temporally synchronized, geometrically consistent "
    "autonomous-driving camera videos and LiDAR range-view sequences from per-camera scene descriptions and provided "
    "control signals: per-camera World Scenario Map (WSM) control videos depicting the controlled objects and road "
    "layout, and an HD-map control for LiDAR. Treat all camera views and LiDAR sweeps as synchronized observations "
    "of the same driving scene, preserving each camera's viewpoint, shared ego motion, road layout, object identity "
    f"and motion, weather, lighting, cross-view consistency, and camera-LiDAR alignment.\n\n{_AV_WSM_CONTROL_INSTRUCTION}"
)
"""Exact ``av_joint_camera_lidar_transfer`` system message for joint generation."""

DEFAULT_ASPECT_RATIO = "16,9"
"""Aspect ratio serialized verbatim into the structured prompt."""


def format_view_caption(caption: Any, camera: str) -> Any:
    """One camera's caption, tagged with which way that camera points.

    A dict gains a ``camera_view`` key; a plain string gets the same sentence
    glued to its front. Real recorded-scene captions take the dict branch, and the string
    branch is what a hand-written prompt hits.

    Raises:
        KeyError: ``camera`` is not one of the eleven recorded-scene cameras. Left to
            raise rather than given a default, since a default would point a
            camera the wrong way and the output would only look a bit off.
    """
    prefix = CAMERA_CAPTION_PREFIXES[camera]
    if isinstance(caption, Mapping):
        return {**caption, "camera_view": prefix}
    return f"{prefix} {caption}"


def multiview_caption(
    captions: Sequence[Any],
    cameras: Sequence[str],
    *,
    num_frames: int,
    fps: float,
    pixel_height: int,
    pixel_width: int,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
) -> dict[str, Any]:
    """The whole prompt payload, in the key order training serialized it in.

    Args:
        captions: one caption per camera, positionally aligned with ``cameras``.
            Already-parsed objects, not JSON text.
        cameras: Rig order. A caption's view is its index here, and the model
            numbers views the same way, so reordering points every caption at
            the wrong camera.
        num_frames: Pixel frames per view. A multiview video tensor joins its
            views along time, so its own length would overstate the clip
            elevenfold.
        fps: Source frames per second before any temporal subsampling.
        pixel_height, pixel_width: One view's size, not the mosaic's.

    Raises:
        ValueError: the counts disagree, or nothing was passed.
    """
    if len(captions) != len(cameras):
        raise ValueError(
            f"{len(captions)} caption(s) for {len(cameras)} camera(s); they pair by position."
        )
    if not captions:
        raise ValueError("a multiview caption needs at least one view.")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")

    views = [
        format_view_caption(caption, camera)
        for caption, camera in zip(captions, cameras, strict=True)
    ]
    # A single view is passed through unwrapped rather than made a one-element
    # list. Wrapping it would build a prompt shape the model never
    # saw at that view count.
    solo = len(views) == 1 and isinstance(views[0], Mapping)
    payload: dict[str, Any] = dict(views[0]) if solo else {"views": views}

    # int() rather than round(): the augmentor truncates, so 200 frames at 30
    # fps is "6s" and not "7s". Short clips therefore report "0s", which is
    # honest and out of distribution -- training never showed the model a
    # sub-second clip. The fix for that is more frames, not a rounder number.
    payload["duration"] = f"{int(num_frames / fps)}s"
    payload["fps"] = fps
    payload["resolution"] = {"H": int(pixel_height), "W": int(pixel_width)}
    payload["aspect_ratio"] = aspect_ratio
    return payload


def view_caption_payloads(
    captions: Sequence[Any],
    cameras: Sequence[str],
    *,
    num_frames: int,
    fps: float,
    pixel_height: int,
    pixel_width: int,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
) -> list[dict[str, Any]]:
    """One whole payload per camera, for a checkpoint that tokenizes each
    camera's caption on its own.

    Each is what :func:`multiview_caption` builds for a single camera, which is
    that camera's caption, the sentence saying which way it points, and the
    clip's duration, fps, resolution and aspect ratio. The metadata is stamped
    onto every caption rather than once onto the rig because training's
    augmentors walk an ordered caption list item by item, so that is the shape
    the model was fit to. The arguments are :func:`multiview_caption`'s and
    mean the same things.

    Raises:
        ValueError: the counts disagree, or nothing was passed.
    """
    if len(captions) != len(cameras):
        raise ValueError(
            f"{len(captions)} caption(s) for {len(cameras)} camera(s); they pair by position."
        )
    if not captions:
        raise ValueError("per-view captions need at least one camera.")
    return [
        multiview_caption(
            [caption],
            [camera],
            num_frames=num_frames,
            fps=fps,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            aspect_ratio=aspect_ratio,
        )
        for caption, camera in zip(captions, cameras, strict=True)
    ]


def caption_json(payload: Mapping[str, Any]) -> str:
    """The payload as the tokenizer receives it.

    ``json.dumps`` with stock settings, matching the augmentor: separators and
    spacing are part of the token sequence, and ``sort_keys`` would reorder the
    very fields the insertion order above was careful about.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"the prompt payload must be a mapping, got {type(payload).__name__}. "
            "A list tokenizes to an empty prompt rather than failing."
        )
    return json.dumps(dict(payload))


def caption_token_ids(
    payload: Mapping[str, Any],
    *,
    tokenizer_path: str = DEFAULT_TOKENIZER,
    system_prompt: str = AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT,
) -> Tensor:
    """Tokenize a multiview payload the way training did.

    Supply the RGB or joint camera/LiDAR system message explicitly. No
    resolution suffix is added: the payload already carries that field.
    """
    resolution = payload["resolution"]
    return tokenize_prompt(
        caption_json(payload),
        height=int(resolution["H"]),
        width=int(resolution["W"]),
        tokenizer_path=tokenizer_path,
        system_prompt=system_prompt,
        append_resolution=False,
    )


def structured_caption(
    raw: str | bytes | Mapping[str, Any],
    *,
    chunk: int = 0,
    chunk_key: str | None = None,
) -> dict[str, Any]:
    """Pull one caption object out of a recorded-scene structured-caption record.

    The record is a chunk map: keys are chunk names, values carry a ``caption``
    (itself JSON text) and the ``start_frame`` / ``end_frame`` it describes.
    Training cuts the clip to that range and so does this, so the frames the
    model generates are the frames the words are about.

    Args:
        raw: The tar member's bytes, its decoded text, or an already-parsed
            record. ``caption_structured`` may itself be a JSON string, as in
            the model's ``_json_dict_or_none``.
        chunk: Which chunk, by position, among those carrying both a caption
            and a frame range.
        chunk_key: Select by name instead, which is how the other ten cameras
            follow the first one's choice.

    Returns:
        A dict holding the parsed ``caption`` payload plus the ``chunk_key``,
        ``start_frame`` and ``end_frame`` it came with.

    Raises:
        ValueError: the record does not parse, holds no usable chunk, holds
            fewer than ``chunk + 1`` of them, or has none named ``chunk_key``.
    """
    record = _as_mapping(raw)
    if record is None:
        raise ValueError("the structured caption record is not a JSON object.")

    chunks = _as_mapping(record.get("caption_structured", record))
    if chunks is None:
        raise ValueError("caption_structured is not a JSON object.")

    usable: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in chunks.items():
        if not isinstance(value, Mapping):
            continue
        if "caption" not in value and "caption_json" not in value:
            continue
        if "start_frame" not in value or "end_frame" not in value:
            continue
        usable.append((key, value))

    if not usable:
        raise ValueError(
            "no chunk in the record carries both a caption and a frame range; "
            f"it holds {sorted(chunks)[:8]}"
        )

    if chunk_key is not None:
        matched = [pair for pair in usable if pair[0] == chunk_key]
        if not matched:
            raise ValueError(
                f"no chunk named {chunk_key!r}; the record holds {[key for key, _ in usable][:8]}"
            )
        key, selected = matched[0]
    else:
        if not 0 <= chunk < len(usable):
            raise ValueError(
                f"the record holds {len(usable)} usable chunk(s); asked for index {chunk}."
            )
        key, selected = usable[chunk]

    payload = selected.get("caption_json", selected.get("caption"))
    parsed = _as_mapping(payload)
    return {
        "caption": payload if parsed is None else dict(parsed),
        "chunk_key": key,
        "start_frame": int(selected["start_frame"]),
        "end_frame": int(selected["end_frame"]),
    }


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """``value`` as a mapping, decoding bytes and parsing JSON text on the way."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None
