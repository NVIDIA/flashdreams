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

"""Pseudo-3D road-plane compositing of kart-game sprites onto a drive video.

Unlike composite_assets.py (which recovers placement from hdmap render
diffs), this script places sprites with no conditioning inputs at all: a
hand-calibrated pinhole camera over a flat road plane. World items are
(lane_offset_m, height_above_road_m, depth_m, class); each frame the depth
shrinks by ego_speed/fps (traffic items carry their own forward speed), the
item is projected, scaled to its metric height, given a soft elliptical
ground shadow, mildly luminance-harmonized, and pasted far-to-near.

Ground-plane projection facts used for calibration: for a point on the road
at lateral X and depth Z, screen_y - horizon = fx*cam_h/Z and
screen_x - cx = X*(screen_y - horizon)/cam_h, so lane-anchored marker
tracks validate (cx, horizon, cam_h) independently of fx; fx then sets the
depth scale and is tuned jointly with ego speed against lane-dash motion.

Asset prep: several sprites ship as RGB/P on white or checkerboard
backgrounds. Alpha is recovered by flagging light low-chroma pixels,
flood-selecting only the components that touch the image border (interior
whites such as gloves and mushroom spots survive), eroding 1 px to kill the
white fringe, and feathering. Cutouts are cached beside the output and a
contact sheet is written for eyeballing.

Example:
    python composite_track_items.py \
        --rgb outputs/pr_videos/arcade_notrees_cas.mp4 \
        --output outputs/pr_videos/mario_kart_demo.mp4
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(SCRIPT_DIR, "assets", "mario_game")

FPS = 30.0
EGO_SPEED_MPS = 22.0

# Pinhole calibration for arcade_notrees_cas.mp4 (1280x704), tuned by
# overlaying lane-anchored debug markers (--debug-markers) and looking.
FX = 950.0
CX = 634.0
HORIZON_Y = 352.0
CAM_HEIGHT_M = 1.39
# The vanishing point drifts slightly right late in the clip (gentle right
# curve): linear cx correction, pixels per frame after DRIFT_START_FRAME.
CX_DRIFT_START = 110
CX_DRIFT_PX_PER_FRAME = 0.15

LANE_WIDTH_M = 3.7
EGO_LANE_CENTER_M = 0.6  # ego lane center sits slightly right of cx

Z_NEAR_M = 4.5
Z_FAR_M = 130.0
Z_FADE_M = 110.0  # alpha ramps 0->1 between Z_FAR and Z_FADE

SHADOW_ALPHA = 0.38
HARMONIZE_STRENGTH = 0.35  # mild: the world is stylized, keep sprites crisp
EDGE_BLUR_RADIUS = 0.4

LUMA = np.array([0.299, 0.587, 0.114])

# ---------------------------------------------------------------------------
# Asset preparation
# ---------------------------------------------------------------------------

ASSET_FILES = {
    "coin": "objects/coin.png",
    "qbox": "objects/?box.png",
    "star": "objects/star.png",
    "mushroom": "objects/mushroom.png",
    "flower": "objects/cannibal_flower.png",
    "shell": "objects/tutleshell.png",
    "ghost": "people/ghost.png",
    "car1": "cars/c1.png",
    "car2": "cars/c2.png",
    "mario": "people/mario.png",
    "luigi": "people/luigi.png",
}


def _key_out_background(rgb: np.ndarray) -> np.ndarray:
    """Alpha for an RGB sprite on a white/checkerboard background.

    Candidate background pixels are light and low-chroma; only candidate
    components touching the image border are removed, so interior whites
    (gloves, mushroom spots, coin highlights) are preserved.
    """
    arr = rgb.astype(np.int16)
    chroma = arr.max(axis=-1) - arr.min(axis=-1)
    candidate = (chroma < 30) & (arr.min(axis=-1) > 160)
    labels, n = ndimage.label(candidate)
    if n == 0:
        return np.full(rgb.shape[:2], 255, dtype=np.uint8)
    border = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    border = border[border != 0]
    background = np.isin(labels, border)
    fg = ~background
    # Drop small disconnected islands (checkerboard remnants, dust specks).
    fg_labels, fg_n = ndimage.label(fg)
    if fg_n > 1:
        sizes = ndimage.sum_labels(fg, fg_labels, index=np.arange(1, fg_n + 1))
        keep = 1 + np.flatnonzero(sizes >= 0.02 * sizes.max())
        fg = np.isin(fg_labels, keep)
    # Kill the white anti-aliasing fringe, then feather the edge.
    fg = ndimage.binary_erosion(fg, iterations=1)
    alpha = ndimage.gaussian_filter(fg.astype(np.float32), sigma=0.8)
    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def _crop_to_alpha(img: Image.Image) -> Image.Image:
    bbox = img.getchannel("A").getbbox()
    return img.crop(bbox) if bbox else img


def prepare_cutouts(cache_dir: str) -> dict[str, Image.Image]:
    """Load all sprites as tightly-cropped RGBA, extracting alpha if absent."""
    os.makedirs(cache_dir, exist_ok=True)
    cutouts: dict[str, Image.Image] = {}
    for name, rel in ASSET_FILES.items():
        img = Image.open(os.path.join(ASSET_DIR, rel))
        if img.mode in ("RGBA", "P"):
            rgba = img.convert("RGBA")
            arr = np.asarray(rgba)
            # Some RGBA assets (?box) ship an all-opaque alpha with the
            # checkerboard baked into the pixels: key those like RGB. Assets
            # with a real alpha are left untouched.
            if (arr[..., 3] < 128).mean() < 0.02:
                alpha = np.minimum(arr[..., 3], _key_out_background(arr[..., :3]))
                rgba = Image.fromarray(np.dstack([arr[..., :3], alpha]))
        else:
            rgb = np.asarray(img.convert("RGB"))
            alpha = _key_out_background(rgb)
            rgba = Image.fromarray(np.dstack([rgb, alpha]))
        rgba = _crop_to_alpha(rgba)
        rgba.save(os.path.join(cache_dir, f"{name}.png"))
        cutouts[name] = rgba
    _write_contact_sheet(cutouts, os.path.join(cache_dir, "cutout_sheet.png"))
    return cutouts


def _write_contact_sheet(cutouts: dict[str, Image.Image], path: str) -> None:
    tiles = []
    for name, img in sorted(cutouts.items()):
        t = img.copy()
        t.thumbnail((220, 220))
        tile = Image.new("RGB", (224, 244), (150, 60, 150))
        tile.paste(t, ((224 - t.width) // 2, (220 - t.height) // 2), t)
        ImageDraw.Draw(tile).text((6, 226), name, fill=(255, 255, 255))
        tiles.append(tile)
    sheet = Image.new("RGB", (224 * 5, 244 * math.ceil(len(tiles) / 5)), (30, 30, 30))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (224 * (i % 5), 244 * (i // 5)))
    sheet.save(path)


# ---------------------------------------------------------------------------
# Camera / projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    fx: float = FX
    cx: float = CX
    horizon_y: float = HORIZON_Y
    cam_h: float = CAM_HEIGHT_M

    def cx_at(self, frame: int) -> float:
        drift = max(0, frame - CX_DRIFT_START) * CX_DRIFT_PX_PER_FRAME
        return self.cx + drift

    def project(
        self, frame: int, x_m: float, up_m: float, z_m: float
    ) -> tuple[float, float, float]:
        """(screen_x, screen_y, px_per_m) for a world point z_m ahead."""
        sx = self.cx_at(frame) + self.fx * x_m / z_m
        sy = self.horizon_y + self.fx * (self.cam_h - up_m) / z_m
        return sx, sy, self.fx / z_m


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackItem:
    cls: str
    lane_x_m: float  # lateral offset, + right of camera axis
    up_m: float  # bottom of sprite above the road plane
    z0_m: float  # depth ahead of camera at frame 0
    height_m: float
    speed_mps: float = 0.0  # item's own forward speed (traffic)
    floating: bool = False  # smaller detached shadow

    def z_at(self, t_s: float) -> float:
        return self.z0_m - (EGO_SPEED_MPS - self.speed_mps) * t_s


LANE_C = EGO_LANE_CENTER_M  # ego (right) lane center
LANE_L = EGO_LANE_CENTER_M - LANE_WIDTH_M  # left lane center
ROAD_EDGE_R = EGO_LANE_CENTER_M + LANE_WIDTH_M / 2 + 0.7  # just past the line
SHOULDER_R = EGO_LANE_CENTER_M + LANE_WIDTH_M / 2 + 2.2  # right shoulder


def build_course(duration_s: float) -> list[TrackItem]:
    """Deterministic kart course covering the whole drive plus lead-in."""
    course_len = EGO_SPEED_MPS * duration_s + Z_FAR_M + 20.0
    items: list[TrackItem] = []

    z = 25.0
    while z < course_len:  # lateral rows of 3 coins across the ego lane
        for dx in (-1.1, 0.0, 1.1):
            items.append(TrackItem("coin", LANE_C + dx, 0.8, z, 0.62, floating=True))
        z += 40.0

    z = 45.0
    while z < course_len:  # ?boxes every ~35 m, alternating lane offsets
        items.append(TrackItem("qbox", LANE_C, 1.2, z, 0.85, floating=True))
        items.append(
            TrackItem("qbox", LANE_C - 1.2, 1.2, z + 12.0, 0.85, floating=True)
        )
        items.append(
            TrackItem("qbox", LANE_C + 1.2, 1.2, z + 24.0, 0.85, floating=True)
        )
        z += 35.0

    z = 55.0
    while z < course_len:  # ghosts hovering near the right shoulder
        items.append(TrackItem("ghost", SHOULDER_R, 1.0, z, 1.1, floating=True))
        z += 110.0

    # Pickups on the road surface.
    items.append(TrackItem("star", LANE_C - 1.0, 0.0, 130.0, 0.7))
    items.append(TrackItem("mushroom", LANE_C + 1.0, 0.0, 235.0, 0.65))
    items.append(TrackItem("star", LANE_C + 0.9, 0.0, 320.0, 0.7))
    items.append(TrackItem("mushroom", LANE_C - 0.9, 0.0, 420.0, 0.65))

    # Slower kart traffic in the left lane (the lane right of the ego lane
    # is the shoulder in this clip, and the ego lane already has a lead
    # car), receding at -8 m/s relative so the ego overtakes both.
    items.append(TrackItem("car1", LANE_L, 0.0, 30.0, 1.4, speed_mps=14.0))
    items.append(TrackItem("car2", LANE_L, 0.0, 62.0, 1.4, speed_mps=14.0))
    return items


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _harmonize(
    sprite: np.ndarray, frame: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    """Mildly pull sprite luminance toward the local background median."""
    x0, y0, x1, y1 = box
    h, w = frame.shape[:2]
    pad = max(8, (x1 - x0) // 4)
    ring = frame[
        max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)
    ]
    if ring.size == 0:
        return sprite
    bg_luma = float(np.median(ring.astype(np.float32) @ LUMA))
    rgb = sprite[..., :3].astype(np.float32)
    vis = sprite[..., 3] > 32
    if not vis.any():
        return sprite
    sprite_luma = float(np.mean(rgb[vis] @ LUMA))
    if sprite_luma < 1.0:
        return sprite
    gain = (bg_luma / sprite_luma) ** HARMONIZE_STRENGTH
    gain = float(np.clip(gain, 0.55, 1.25))
    out = sprite.copy()
    out[..., :3] = np.clip(rgb * gain, 0, 255).astype(np.uint8)
    return out


def _paste_shadow(
    canvas: Image.Image, cx_px: float, ground_y: float, width_px: float, alpha: float
) -> None:
    w = max(6.0, width_px)
    hgt = max(3.0, w * 0.22)
    shadow = Image.new("L", (int(w * 1.6) + 8, int(hgt * 1.6) + 8), 0)
    d = ImageDraw.Draw(shadow)
    d.ellipse(
        [
            shadow.width / 2 - w / 2,
            shadow.height / 2 - hgt / 2,
            shadow.width / 2 + w / 2,
            shadow.height / 2 + hgt / 2,
        ],
        fill=int(255 * alpha),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(1.5, w * 0.08)))
    black = Image.new("RGBA", shadow.size, (10, 10, 15, 0))
    black.putalpha(shadow)
    canvas.alpha_composite(
        black, (int(cx_px - shadow.width / 2), int(ground_y - shadow.height / 2))
    )


def _sprite_wobble(item: TrackItem, frame: int) -> tuple[float, float]:
    """(x_squash, up_bob_m): coin spin + gentle bob for floating items."""
    phase = item.z0_m * 0.61 + item.lane_x_m * 2.7  # de-sync neighbours
    if item.cls == "coin":
        squash = abs(math.cos(frame * 2 * math.pi / 36.0 + phase))
        return max(0.3, squash), 0.05 * math.sin(frame * 2 * math.pi / 52.0 + phase)
    if item.cls == "qbox":
        return 1.0, 0.06 * math.sin(frame * 2 * math.pi / 48.0 + phase)
    return 1.0, 0.0


def render(
    frames: np.ndarray,
    cutouts: dict[str, Image.Image],
    camera: Camera,
    items: list[TrackItem],
) -> np.ndarray:
    n, h, w = frames.shape[:3]
    out = np.empty_like(frames)
    for fi in range(n):
        t = fi / FPS
        base = frames[fi]
        canvas = Image.fromarray(base).convert("RGBA")
        visible = [(it.z_at(t), it) for it in items]
        visible = [(z, it) for z, it in visible if Z_NEAR_M <= z <= Z_FAR_M]
        for z, it in sorted(visible, key=lambda p: -p[0]):
            squash, bob = _sprite_wobble(it, fi)
            sx, sy_bottom, px_per_m = camera.project(fi, it.lane_x_m, it.up_m + bob, z)
            sprite_h = it.height_m * px_per_m
            if sprite_h < 3:
                continue
            src = cutouts[it.cls]
            sprite_w = sprite_h * src.width / src.height * squash
            if sprite_w < 2:
                continue
            fade = 1.0
            if z > Z_FADE_M:
                fade = max(0.0, (Z_FAR_M - z) / (Z_FAR_M - Z_FADE_M))
            resized = src.resize(
                (max(2, int(round(sprite_w))), max(3, int(round(sprite_h)))),
                Image.LANCZOS,
            )
            x0 = int(round(sx - resized.width / 2))
            y0 = int(round(sy_bottom - resized.height))
            arr = np.asarray(resized).copy()
            arr = _harmonize(
                arr, base, (x0, y0, x0 + resized.width, y0 + resized.height)
            )
            if fade < 1.0:
                arr[..., 3] = (arr[..., 3].astype(np.float32) * fade).astype(np.uint8)
            sprite = Image.fromarray(arr)
            if EDGE_BLUR_RADIUS > 0:
                a = sprite.getchannel("A").filter(
                    ImageFilter.GaussianBlur(EDGE_BLUR_RADIUS)
                )
                sprite.putalpha(a)
            _, ground_y, _ = camera.project(fi, it.lane_x_m, 0.0, z)
            shadow_w = resized.width * (0.55 if it.floating else 0.95)
            _paste_shadow(canvas, sx, ground_y, shadow_w, SHADOW_ALPHA * fade)
            canvas.alpha_composite(sprite, (x0, y0))
        out[fi] = np.asarray(canvas.convert("RGB"))
    return out


# ---------------------------------------------------------------------------
# Calibration debug overlay
# ---------------------------------------------------------------------------


def render_debug_markers(
    frames: np.ndarray, camera: Camera, out_dir: str, debug_frames: list[int]
) -> None:
    """Markers anchored to lane lines: discs every 10 m on both ego-lane
    boundaries plus dash-pitch ticks (12.19 m) advected at EGO_SPEED_MPS."""
    os.makedirs(out_dir, exist_ok=True)
    lane_lines = {
        (255, 80, 80): LANE_C - LANE_WIDTH_M / 2,
        (80, 160, 255): LANE_C + LANE_WIDTH_M / 2,
    }
    for fi in debug_frames:
        img = Image.fromarray(frames[fi]).convert("RGB")
        d = ImageDraw.Draw(img)
        for color, lx in lane_lines.items():
            for z in range(6, 90, 10):
                sx, sy, _ = camera.project(fi, lx, 0.0, float(z))
                r = max(2, int(30 / z * 6))
                d.ellipse([sx - r, sy - r, sx + r, sy + r], outline=color, width=2)
        # dash-pitch ticks advected with ego speed (yellow)
        offset = (fi / FPS * EGO_SPEED_MPS) % 12.19
        z = 6.0 - offset + 12.19
        while z < 100:
            sx, sy, _ = camera.project(fi, LANE_C - LANE_WIDTH_M / 2, 0.0, z)
            d.line([sx - 10, sy, sx + 10, sy], fill=(255, 230, 40), width=2)
            z += 12.19
        hx = camera.cx_at(fi)
        d.line([hx - 15, camera.horizon_y, hx + 15, camera.horizon_y], fill=(0, 255, 0))
        d.line([hx, camera.horizon_y - 15, hx, camera.horizon_y + 15], fill=(0, 255, 0))
        img.save(os.path.join(out_dir, f"debug_{fi:03d}.png"))


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument(
        "--rgb",
        default=os.path.join(SCRIPT_DIR, "outputs/pr_videos/arcade_notrees_cas.mp4"),
    )
    ap.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "outputs/pr_videos/mario_kart_demo.mp4"),
    )
    ap.add_argument(
        "--cache-dir", default=None, help="cutout cache (default: <output>_assets)"
    )
    ap.add_argument(
        "--debug-markers",
        nargs="*",
        type=int,
        default=None,
        help="write calibration overlay for these frame indices and exit",
    )
    ap.add_argument(
        "--contact-sheet",
        default=None,
        help="3-frame contact sheet path (default: <output>_sheet.png)",
    )
    args = ap.parse_args()

    cache_dir = args.cache_dir or os.path.splitext(args.output)[0] + "_assets"
    frames = iio.imread(args.rgb)
    camera = Camera()

    if args.debug_markers is not None:
        fr = args.debug_markers or [60, 110, 165, 215]
        render_debug_markers(frames, camera, cache_dir, fr)
        print(f"debug overlays -> {cache_dir}")
        return

    cutouts = prepare_cutouts(cache_dir)
    items = build_course(len(frames) / FPS)
    out = render(frames, cutouts, camera, items)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    iio.imwrite(
        args.output, out, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p"
    )

    sheet_path = args.contact_sheet or os.path.splitext(args.output)[0] + "_sheet.png"
    picks = [len(out) // 10, len(out) // 2, len(out) - 5]
    tiles = [Image.fromarray(out[i]) for i in picks]
    sheet = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height))
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width
    sheet.save(sheet_path)
    print(f"wrote {args.output} ({len(out)} frames) and {sheet_path}")


if __name__ == "__main__":
    main()
