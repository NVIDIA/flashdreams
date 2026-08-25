# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame-aligned RGBA HUD generation for V2 result composition."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from crazy_robotaxi.rules import (
    TaxiGameSnapshot,
    project_taxi_markers_to_camera,
)
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.types import CameraCalibration

_PANEL = (8, 12, 20, 214)
_PICKUP = (255, 210, 48, 255)
_DROPOFF = (48, 232, 132, 255)


def render_hud(
    snapshots: Sequence[object],
    *,
    rig_poses_world: np.ndarray,
    calibration: CameraCalibration,
    bev_tchw: Tensor | None,
    width: int,
    height: int,
    device: torch.device | str,
    dtype: torch.dtype,
    player_name: str = "",
) -> Tensor:
    """Return synchronized floating-point RGBA layers in ``[T,4,H,W]``."""
    if len(snapshots) != len(rig_poses_world):
        raise ValueError("HUD snapshots and camera poses must align")
    bev_images = _bev_images(bev_tchw, len(snapshots))
    camera = FThetaCameraModel(
        calibration,
        output_width=width,
        output_height=height,
    )
    frames = [
        _render_frame(
            snapshot if isinstance(snapshot, TaxiGameSnapshot) else None,
            rig_pose=rig_poses_world[index],
            camera=camera,
            bev=bev_images[index],
            width=width,
            height=height,
            player_name=player_name,
        )
        for index, snapshot in enumerate(snapshots)
    ]
    rgba = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    color = rgba[:, :3].float() / 127.5 - 1.0
    alpha = rgba[:, 3:4].float() / 255.0
    return torch.cat((color, alpha), dim=1).to(device=device, dtype=dtype)


def _bev_images(bev: Tensor | None, count: int) -> list[Image.Image | None]:
    if bev is None:
        return [None] * count
    if int(bev.shape[0]) != count:
        raise ValueError("BEV and HUD frame counts must align")
    pixels = (
        ((bev.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return [Image.fromarray(frame, mode="RGB") for frame in pixels]


def _render_frame(
    snapshot: TaxiGameSnapshot | None,
    *,
    rig_pose: np.ndarray,
    camera: FThetaCameraModel,
    bev: Image.Image | None,
    width: int,
    height: int,
    player_name: str,
) -> np.ndarray:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if snapshot is None:
        return np.asarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(14, height // 34))
    small = ImageFont.load_default(size=max(12, height // 48))
    margin = max(12, width // 80)
    color = _PICKUP if snapshot.phase == "seeking_pickup" else _DROPOFF
    label = "PICKUP" if snapshot.phase == "seeking_pickup" else "DROPOFF"
    panel_width = max(310, width // 3)
    draw.rounded_rectangle(
        (margin, margin, margin + panel_width, margin + 94),
        radius=12,
        fill=_PANEL,
    )
    draw.text(
        (margin + 14, margin + 12),
        f"SCORE {snapshot.score:06d}",
        font=font,
        fill="white",
    )
    draw.text(
        (margin + 14, margin + 52),
        f"TIME {snapshot.global_remaining_time_s:05.1f}s",
        font=small,
        fill="white",
    )
    draw.text(
        (margin + panel_width // 2, margin + 52),
        f"{label} {snapshot.distance_m:04.0f}m",
        font=small,
        fill=color,
    )
    _draw_direction(draw, snapshot.relative_bearing_rad, width, margin, color)
    _draw_world_markers(draw, snapshot, rig_pose, camera, width, height, color)
    if bev is not None:
        _draw_bev(image, draw, bev, snapshot, width, height, color)
    if snapshot.event is not None:
        event = snapshot.event.replace("_", " ").upper()
        draw.text((width // 2, height // 6), event, font=font, anchor="mm", fill=color)
    if snapshot.session_state == "awaiting_name":
        _draw_center_panel(
            draw,
            width,
            height,
            "NEW HIGH SCORE",
            (f"ENTER NAME: {player_name or '_'}", "ENTER TO SUBMIT"),
            font,
            small,
        )
    elif snapshot.session_state == "leaderboard":
        lines = tuple(
            f"{index:>2}. {entry.name:<12} {entry.score:>7}"
            for index, entry in enumerate(snapshot.leaderboard, start=1)
        ) or ("NO SCORES YET",)
        _draw_center_panel(draw, width, height, "LEADERBOARD", lines, font, small)
    return np.asarray(image, dtype=np.uint8)


def _draw_direction(
    draw: ImageDraw.ImageDraw,
    bearing: float,
    width: int,
    margin: int,
    color: tuple[int, int, int, int],
) -> None:
    angle = bearing - math.pi / 2.0
    center = (width / 2.0, margin + 44.0)
    radius = 26.0
    points = [
        (
            center[0] + math.cos(angle + offset) * radius * scale,
            center[1] + math.sin(angle + offset) * radius * scale,
        )
        for offset, scale in ((0.0, 1.0), (2.45, 0.55), (-2.45, 0.55))
    ]
    draw.polygon(points, fill=color)


def _draw_world_markers(
    draw: ImageDraw.ImageDraw,
    snapshot: TaxiGameSnapshot,
    rig_pose: np.ndarray,
    camera: FThetaCameraModel,
    width: int,
    height: int,
    color: tuple[int, int, int, int],
) -> None:
    for projection in project_taxi_markers_to_camera(
        snapshot,
        rig_pose,
        camera,
        image_width=width,
        image_height=height,
    ):
        for edge in projection.ring_edges_uv:
            draw.line(edge, fill=color, width=3)
        if projection.beacon_top_uv is not None:
            draw.line(
                (projection.anchor_uv, projection.beacon_top_uv),
                fill=color,
                width=4,
            )


def _draw_bev(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    bev: Image.Image,
    snapshot: TaxiGameSnapshot,
    width: int,
    height: int,
    color: tuple[int, int, int, int],
) -> None:
    size = min(width // 4, height // 3)
    left = width - size - 18
    top = height - size - 18
    panel = bev.resize((size, size), Image.Resampling.BILINEAR).convert("RGBA")
    panel.putalpha(220)
    image.alpha_composite(panel, (left, top))
    draw.rectangle((left, top, left + size, top + size), outline="white", width=2)
    cx, cy = left + size // 2, top + size // 2
    draw.polygon(((cx, cy - 9), (cx - 6, cy + 7), (cx + 6, cy + 7)), fill="white")
    angle = snapshot.relative_bearing_rad - math.pi / 2.0
    radius = size * 0.38
    target = (cx + math.cos(angle) * radius, cy + math.sin(angle) * radius)
    draw.ellipse((target[0] - 6, target[1] - 6, target[0] + 6, target[1] + 6), fill=color)


def _draw_center_panel(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    title: str,
    lines: Sequence[str],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    small: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    panel_width = min(width - 40, max(440, width // 2))
    panel_height = min(height - 40, 100 + len(lines) * 28)
    left = (width - panel_width) // 2
    top = (height - panel_height) // 2
    draw.rounded_rectangle(
        (left, top, left + panel_width, top + panel_height),
        radius=18,
        fill=(5, 8, 15, 238),
        outline=_PICKUP,
        width=3,
    )
    draw.text((left + 24, top + 18), title, font=font, fill=_PICKUP)
    for index, line in enumerate(lines):
        draw.text((left + 24, top + 58 + index * 28), line, font=small, fill="white")
