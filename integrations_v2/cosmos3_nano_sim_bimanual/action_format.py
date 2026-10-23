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

"""AgiBot checkpoint action format and normalization.

Pure NumPy: this module owns the action layout, the checkpoint's normalization,
and the raw-action builders. Tensor construction belongs to the model session so
this contract stays trivially testable on CPU.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

AGIBOT_DOMAIN_ID = 15
"""Domain ID selecting the AgiBot row of the model's domain-aware action embed."""

CAMERA_DOMAIN_ID = 2
"""Domain ID selecting camera-only conditioning with masked hand channels."""

RAW_ACTION_DIM = 59
"""GA unified_v1 action width before padding."""

MAX_ACTION_DIM = 64
"""Width the model consumes; dimensions 59:64 are reserved and must stay zero."""

POSE_ANCHOR_FRAMES = 16
"""Pixel-frame interval for GA backward chunk-anchored wrist poses."""

GRIPPER_OPEN_WIDTH_METERS = 0.12
"""AgiBot's fully-open width; distinct from the normalizer's 0.082 m reference."""

FPS = 30.0
"""Fallback AgiBot frame rate; live rollouts use the scene's explicit FPS."""

TEMPORAL_COMPRESSION_FACTOR = 4
"""VAE temporal compression: four actions drive one latent frame."""

SPATIAL_COMPRESSION_FACTOR = 16
"""VAE spatial compression from pixels to latents."""

CONTENT_HEIGHT = 360
CONTENT_WIDTH = 640
"""Corrected 16:9 AgiBot head-camera content size before VAE encoding."""

LATENT_HEIGHT = CONTENT_HEIGHT // SPATIAL_COMPRESSION_FACTOR
LATENT_WIDTH = CONTENT_WIDTH // SPATIAL_COMPRESSION_FACTOR
"""Training keeps only complete content cells: a 22x40 logical latent."""

TOKEN_GRID = ((LATENT_HEIGHT + 1) // 2, (LATENT_WIDTH + 1) // 2)
"""Transformer grid after the 2x2 patchifier pads odd latent edges."""

OUTPUT_HEIGHT = LATENT_HEIGHT * SPATIAL_COMPRESSION_FACTOR
OUTPUT_WIDTH = LATENT_WIDTH * SPATIAL_COMPRESSION_FACTOR
"""Decoded region: 640x352 pixels, without stretching the source content."""

_POSE_STARTS = (0, 9, 34)
"""Start index of each 9D pose block: head, right wrist, left wrist."""

GRIPPER_INDICES = (33, 58)
"""Index of the right and left gripper widths in meters."""

_IDENTITY_POSE_RAW = np.array(
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    dtype=np.float32,
)
"""A 9D pose delta meaning "no motion": zero translation, identity rot6d."""

# Exact post-rotations applied after AgiBot forward kinematics. The model's
# wrist deltas live in these side-specific gripper/OpenCV frames, not in
# participant XYZ.
WRIST_TO_OPENCV: dict[str, np.ndarray] = {
    "right": np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    ),
    "left": np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    ),
}

NORMALIZER_PATH = (
    Path(__file__).resolve().parent
    / "normalizers"
    / "global_asinh_v1_backward_anchored_gripper_metric.json"
)
"""GA export artifact ae695e2c…: global asinh statistics for unified_v1."""

_MIN_SCALE = 1e-8


def load_agibot_normalizer(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the vendored GA unified action normalizer.

    Raises:
        ValueError: If the declared action width or any statistics row is wrong.
    """
    normalizer_path = Path(path).expanduser() if path is not None else NORMALIZER_PATH
    payload = json.loads(normalizer_path.read_text())
    metadata = payload.get("metadata", {})
    if (
        metadata.get("dim") != RAW_ACTION_DIM
        or metadata.get("method") != "global_asinh"
        or metadata.get("action_schema") != "unified_v1"
        or metadata.get("gripper_representation") != "metric_width_m"
    ):
        raise ValueError(
            "AgiBot normalizer must use the GA 59D global_asinh metric contract"
        )
    values = payload.get("global")
    if not isinstance(values, dict):
        raise ValueError("AgiBot normalizer is missing 'global'")
    for key in ("q01", "q99"):
        row = np.asarray(values.get(key), dtype=np.float64)
        if row.shape != (RAW_ACTION_DIM,) or not np.isfinite(row).all():
            raise ValueError(
                f"AgiBot normalizer global.{key} must contain "
                f"{RAW_ACTION_DIM} finite values"
            )
    if np.any(np.asarray(values["q99"]) <= np.asarray(values["q01"])):
        raise ValueError("AgiBot normalizer q99 must exceed q01")
    return payload


@lru_cache(maxsize=4)
def _norm_stats(path: str | None) -> tuple[np.ndarray, np.ndarray]:
    raw = load_agibot_normalizer(path)["global"]
    q01 = np.asarray(raw["q01"], dtype=np.float32)
    q99 = np.asarray(raw["q99"], dtype=np.float32)
    offset = ((q99 + q01) / 2.0).astype(np.float32)
    scale = (np.maximum(q99 - q01, _MIN_SCALE) / 2.0).astype(np.float32)
    offset.flags.writeable = False
    scale.flags.writeable = False
    return offset, scale


def load_agibot_norm_stats(
    path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the GA global-asinh ``(offset, scale)`` from its pinned artifact."""
    return _norm_stats(str(path) if path is not None else None)


def _validate_action_array(value: Any, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] != RAW_ACTION_DIM:
        raise ValueError(
            f"AgiBot {field} must end in {RAW_ACTION_DIM} values, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"AgiBot {field} contains non-finite values")
    return array


def _resolve_stats(
    offset: np.ndarray | None,
    scale: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if offset is None or scale is None:
        return load_agibot_norm_stats()
    offset = np.asarray(offset, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if offset.shape != (RAW_ACTION_DIM,) or scale.shape != (RAW_ACTION_DIM,):
        raise ValueError("AgiBot normalizer offset and scale must both be 59D")
    if not np.isfinite(offset).all() or not np.isfinite(scale).all():
        raise ValueError("AgiBot normalizer offset and scale must be finite")
    if np.any(scale < _MIN_SCALE / 2):
        raise ValueError(f"AgiBot normalizer scale must be at least {_MIN_SCALE / 2:g}")
    return offset, scale


def normalize_action(
    raw: Any,
    offset: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """Apply unclamped GA global-asinh normalization before source masking."""
    action = _validate_action_array(raw, "raw action")
    offset, scale = _resolve_stats(offset, scale)
    return (np.arcsinh((action - offset) / scale) / np.arcsinh(1.0)).astype(np.float32)


def denormalize_action(
    normalized: Any,
    offset: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """Invert :func:`normalize_action`."""
    action = _validate_action_array(normalized, "normalized action")
    offset, scale = _resolve_stats(offset, scale)
    return (np.sinh(action * np.arcsinh(1.0)) * scale + offset).astype(np.float32)


def pad_action(action: Any) -> np.ndarray:
    """Validate and zero-pad normalized actions from 59D to the model's 64D.

    Accepts already-padded 64D input so callers can pass either width, but
    rejects nonzero reserved dimensions rather than silently forwarding them.

    Raises:
        ValueError: If the width is neither 59 nor 64, values are non-finite,
            or reserved dimensions 59:64 are nonzero.
    """
    values = np.asarray(action, dtype=np.float32)
    if values.ndim < 1:
        raise ValueError(
            f"AgiBot action needs at least one dimension, got {values.shape}"
        )
    width = int(values.shape[-1])
    if width not in (RAW_ACTION_DIM, MAX_ACTION_DIM):
        raise ValueError(
            f"AgiBot action width must be {RAW_ACTION_DIM} or {MAX_ACTION_DIM}, "
            f"got {width}"
        )
    if not np.isfinite(values).all():
        raise ValueError("AgiBot action contains non-finite values")
    if width == MAX_ACTION_DIM:
        if np.count_nonzero(values[..., RAW_ACTION_DIM:]):
            raise ValueError(
                f"AgiBot padded dimensions {RAW_ACTION_DIM}:{MAX_ACTION_DIM} "
                "must be zero"
            )
        return values
    padding = np.zeros(
        (*values.shape[:-1], MAX_ACTION_DIM - RAW_ACTION_DIM), dtype=np.float32
    )
    return np.concatenate((values, padding), axis=-1)


def identity_raw_action(rows: int | None = None) -> np.ndarray:
    """Build one stationary raw 59D action, or ``rows`` copies of it."""
    action = np.zeros(RAW_ACTION_DIM, dtype=np.float32)
    for start in _POSE_STARTS:
        action[start : start + 9] = _IDENTITY_POSE_RAW
    if rows is None:
        return action
    if int(rows) < 0:
        raise ValueError("AgiBot identity action row count must be non-negative")
    return np.repeat(action[None], int(rows), axis=0)


def normalize_and_pad(
    raw: Any,
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Normalize, mask unavailable AgiBot fingertips, then pad to 64D.

    Mask after normalization: raw zeros in missing slots are not normalized
    zeros. Ego and wrist poses remain valid, including stationary ego rotation.
    """
    normalized = normalize_action(raw, *(stats or (None, None)))
    normalized[..., 18:33] = 0
    normalized[..., 43:58] = 0
    return pad_action(normalized)


def logical_latent_geometry(
    content_height: int,
    content_width: int,
    *,
    expected: tuple[int, int] | None = (LATENT_HEIGHT, LATENT_WIDTH),
) -> tuple[int, int]:
    """Map a scene's top-left content region to the checkpoint's logical latent.

    Training crops the padded-canvas latent using floor division, retaining at
    least one cell per axis. The 360-pixel AgiBot content therefore keeps 22
    latent rows, decoded to 352 pixels.

    Raises:
        ValueError: If the content size is non-positive or does not match
            ``expected`` after floor division by the VAE compression factor.
    """
    content_height, content_width = int(content_height), int(content_width)
    factor = SPATIAL_COMPRESSION_FACTOR
    if content_height <= 0 or content_width <= 0:
        raise ValueError(
            f"AgiBot content size must be positive, got {content_height}x{content_width}"
        )
    logical = (max(content_height // factor, 1), max(content_width // factor, 1))
    if expected is not None and logical != tuple(expected):
        raise ValueError(
            "AgiBot scene is incompatible with this checkpoint's logical model "
            f"geometry: content {content_height}x{content_width} gives latent "
            f"{logical[0]}x{logical[1]}, expected {expected[0]}x{expected[1]}"
        )
    return logical
