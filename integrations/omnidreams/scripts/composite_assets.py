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

"""Geometric compositing + harmonization MVP for aggressive object spawning.

Composites sprites onto an omnidreams RGB rollout at the EXACT screen-space
locations of injected 3D boxes. Two sprite sources: procedural cartoon sprites
(pedestrians, cones, item boxes, bananas), or — with ``--sprite-dir`` —
photoreal RGBA cutouts harvested from omnidreams footage (assets/cutouts),
picked randomly-but-stably per track with optional horizontal flip. The
special class ``walker`` plays an animated walk-cycle sequence
(``<sprite-dir>/walkers/<name>/frame_*.png`` + ``meta.json``) per track:
the cycle advances with the track's ground-relative speed (frozen when
standing, natural cadence at walking pace, ``--walk-speed-mps`` when the
boxes' world speed is known), loops ping-pong (no wrap jump), and the
cutout is flipped so its stride direction roughly matches the track's
screen-space motion.
Placement comes from the conditioning renderer itself: per frame,
``|boxed_hdmap - baseline_hdmap|.max(channel) > threshold`` recovers the pixels
the Ludus renderer drew for each injected box (fisheye projection included), so
the generative model's placement-manifold limits do not apply.

Pipeline per frame:
  1. Diff the boxed vs. box-free hdmap renders, threshold, connected components.
  2. Track components across frames (greedy IoU, centroid fallback) so each box
     keeps a stable sprite identity; EMA-smooth anchor and height (alpha 0.5).
  3. Paste the sprite bottom-center anchored on the component bbox, scaled to
     the component height, with a soft elliptical ground shadow and luminance /
     color-temperature harmonization against a background ring.

Known limitation (MVP): no occlusion handling. Sprites always paste on top of
the RGB frame, so scene geometry that should occlude a spawned object (poles,
passing cars) will be drawn behind it.

Example:
    python composite_assets.py \
        --rgb outputs/ped20/drive.mp4 \
        --boxed-hdmap outputs/ped20_hdmap/hdmap.mp4 \
        --baseline-hdmap outputs/baseline_hdmap/hdmap.mp4 \
        --sprites pedestrian \
        --output outputs/pr_videos/composite_crowd_midroad.mp4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import zlib
from dataclasses import dataclass, field

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

DIFF_THRESHOLD = 40
MIN_COMPONENT_AREA = 40
EMA_ALPHA = 0.5
IOU_MATCH_THRESHOLD = 0.1
CENTROID_MATCH_FRACTION = 0.75  # of component diagonal
TRACK_MAX_MISSES = 3
SHADOW_ALPHA = 0.35
HARMONIZE_TARGET = 0.9  # sprite luminance -> 0.9x background ring median
EDGE_BLUR_RADIUS = 0.5
FACING_MIN_VEL = 0.4  # px/frame of EMA x-velocity before a walker (re)faces
MAX_WALKER_UPSCALE = 1.5  # avoid blowing low-res walk cycles up beyond this
# Walk-cycle cadence is coupled to the track's GROUND-relative screen speed,
# normalized by on-screen height: a 1.7 m person walking 1.4 m/s covers
# ~0.82 heights/s. At that speed the sequence plays at its natural (source)
# cadence; speed ~0 freezes the cycle (a standing person); the rate is
# capped so cadence never exceeds ~2 steps/s. Ego-induced screen motion is
# removed by subtracting the looming-expected radial flow (a world-static
# object expands radially from the FOE at its own height-growth rate), so
# static boxes driven past do NOT treadmill. Radially approaching walkers
# are screen-indistinguishable from static ones, so runs whose boxes are
# KNOWN to translate take an explicit --walk-speed-mps override instead.
FULL_CADENCE_SPEED = 0.82 / 30.0  # heights per frame at natural cadence
NOMINAL_WALK_SPEED_MPS = 1.4
MAX_PLAYBACK_RATE = 1.2
FOE_Y_FRACTION = 0.47  # nominal focus-of-expansion height (approx horizon)

LUMA = np.array([0.299, 0.587, 0.114])


# --------------------------------------------------------------------------
# Procedural sprites (RGBA uint8, drawn at high resolution, downscaled later)
# --------------------------------------------------------------------------


def _canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def make_pedestrian(rng: np.random.Generator) -> np.ndarray:
    """Dark walking figure: head, torso, arms, legs; per-instance clothing tint."""
    w, h = 220, 512
    img, draw = _canvas(w, h)
    cx = w // 2

    shirt = tuple(rng.integers(30, 110, size=3).tolist())
    pants = tuple((np.array(shirt) * rng.uniform(0.4, 0.8)).astype(int).tolist())
    base = float(rng.integers(120, 200))
    skin = (int(base), int(base * 0.78), int(base * 0.62))

    head_r = 34
    head_cy = 60
    draw.ellipse(
        (cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r),
        fill=(*skin, 255),
    )
    # Neck + torso.
    draw.rectangle(
        (cx - 10, head_cy + head_r - 6, cx + 10, head_cy + head_r + 14),
        fill=(*skin, 255),
    )
    torso_top, torso_bot = head_cy + head_r + 8, 290
    draw.rounded_rectangle(
        (cx - 52, torso_top, cx + 52, torso_bot), radius=26, fill=(*shirt, 255)
    )
    # Arms, slightly asymmetric for a mid-stride look.
    arm_sway = int(rng.integers(6, 18))
    draw.rounded_rectangle(
        (cx - 74, torso_top + 12, cx - 46, torso_bot - 20 + arm_sway),
        radius=14,
        fill=(*shirt, 255),
    )
    draw.rounded_rectangle(
        (cx + 46, torso_top + 12, cx + 74, torso_bot - 20 - arm_sway),
        radius=14,
        fill=(*shirt, 255),
    )
    # Legs in a slight stride.
    stride = int(rng.integers(4, 22))
    draw.polygon(
        [
            (cx - 44, torso_bot - 10),
            (cx - 6, torso_bot - 10),
            (cx - 14 - stride, h - 16),
            (cx - 46 - stride, h - 16),
        ],
        fill=(*pants, 255),
    )
    draw.polygon(
        [
            (cx + 6, torso_bot - 10),
            (cx + 44, torso_bot - 10),
            (cx + 46 + stride, h - 16),
            (cx + 14 + stride, h - 16),
        ],
        fill=(*pants, 255),
    )
    # Shoes.
    shoe = (25, 22, 20, 255)
    draw.ellipse((cx - 56 - stride, h - 28, cx - 4 - stride, h - 4), fill=shoe)
    draw.ellipse((cx + 4 + stride, h - 28, cx + 56 + stride, h - 4), fill=shoe)
    return np.asarray(img)


def make_cone(rng: np.random.Generator) -> np.ndarray:
    """Orange traffic cone with white reflective stripes on a dark base."""
    del rng  # cones are uniform
    w, h = 300, 400
    img, draw = _canvas(w, h)
    cx = w // 2
    base_y = h - 20
    orange = (235, 110, 20, 255)
    draw.polygon(
        [
            (cx - 34, 30),
            (cx + 34, 30),
            (cx + 108, base_y - 24),
            (cx - 108, base_y - 24),
        ],
        fill=orange,
    )
    draw.ellipse((cx - 36, 12, cx + 36, 48), fill=orange)
    # White stripes: horizontal bands clipped to the cone silhouette.
    stripe_mask = Image.new("L", (w, h), 0)
    smd = ImageDraw.Draw(stripe_mask)
    smd.polygon(
        [
            (cx - 34, 30),
            (cx + 34, 30),
            (cx + 108, base_y - 24),
            (cx - 108, base_y - 24),
        ],
        fill=255,
    )
    stripes = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    std = ImageDraw.Draw(stripes)
    for y0, y1 in ((120, 165), (210, 260)):
        std.rectangle((0, y0, w, y1), fill=(245, 245, 245, 255))
    img.paste(
        stripes,
        (0, 0),
        Image.composite(stripes.split()[3], Image.new("L", (w, h), 0), stripe_mask),
    )
    # Base plate.
    draw.rounded_rectangle(
        (cx - 130, base_y - 30, cx + 130, base_y), radius=12, fill=(190, 85, 15, 255)
    )
    return np.asarray(img)


def make_item_box(rng: np.random.Generator) -> np.ndarray:
    """Yellow pseudo-3D cube with a question mark (arcade item box)."""
    del rng
    w, h = 360, 360
    img, draw = _canvas(w, h)
    face = (250, 200, 40, 255)
    top = (255, 230, 110, 255)
    side = (200, 150, 20, 255)
    fx0, fy0, fx1, fy1 = 40, 90, 280, 330
    depth = 50
    draw.polygon(
        [
            (fx0, fy0),
            (fx0 + depth, fy0 - depth),
            (fx1 + depth, fy0 - depth),
            (fx1, fy0),
        ],
        fill=top,
    )
    draw.polygon(
        [
            (fx1, fy0),
            (fx1 + depth, fy0 - depth),
            (fx1 + depth, fy1 - depth),
            (fx1, fy1),
        ],
        fill=side,
    )
    draw.rectangle((fx0, fy0, fx1, fy1), fill=face)
    draw.rectangle((fx0, fy0, fx1, fy1), outline=(150, 100, 10, 255), width=8)
    # Question mark drawn with primitives (no font dependency).
    qm = (150, 60, 160, 255)
    qcx, qcy = (fx0 + fx1) // 2, fy0 + 85
    draw.arc(
        (qcx - 45, qcy - 45, qcx + 45, qcy + 45), start=140, end=90, fill=qm, width=22
    )
    draw.line((qcx, qcy + 40, qcx, qcy + 80), fill=qm, width=22)
    draw.ellipse((qcx - 14, qcy + 100, qcx + 14, qcy + 128), fill=qm)
    return np.asarray(img)


def make_banana(rng: np.random.Generator) -> np.ndarray:
    """Yellow banana crescent with brown tips."""
    del rng
    w, h = 400, 300
    img, draw = _canvas(w, h)
    # Crescent = big yellow disc minus an offset disc.
    body = Image.new("L", (w, h), 0)
    bd = ImageDraw.Draw(body)
    bd.ellipse((30, 20, 370, 360), fill=255)
    bd.ellipse((60, -80, 400, 260), fill=0)
    yellow = Image.new("RGBA", (w, h), (240, 205, 50, 255))
    img.paste(yellow, (0, 0), body)
    # Ridge highlight.
    draw.arc(
        (60, 60, 340, 340), start=200, end=340, fill=(255, 235, 130, 255), width=14
    )
    # Brown tips.
    draw.ellipse((22, 130, 66, 174), fill=(110, 70, 25, 255))
    draw.ellipse((334, 130, 378, 174), fill=(110, 70, 25, 255))
    return np.asarray(img)


SPRITE_MAKERS = {
    "pedestrian": make_pedestrian,
    "cone": make_cone,
    "itembox": make_item_box,
    "banana": make_banana,
}


# --------------------------------------------------------------------------
# Photoreal cutout library (RGBA PNGs harvested from omnidreams footage)
# --------------------------------------------------------------------------


def load_cutout_library(sprite_dir: str) -> dict[str, list[np.ndarray]]:
    """Load ``<class>_<n>.png`` RGBA cutouts, grouped by class.

    A ``manifest.json`` ({class: [{"file": ...}, ...]}) is honored when
    present; otherwise every ``*.png`` is grouped by its ``<class>_`` prefix.
    """
    manifest_path = os.path.join(sprite_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        groups = {
            cls: [os.path.join(sprite_dir, e["file"]) for e in entries]
            for cls, entries in manifest.items()
        }
    else:
        groups = {}
        for path in sorted(glob.glob(os.path.join(sprite_dir, "*.png"))):
            cls = os.path.basename(path).rsplit("_", 1)[0]
            groups.setdefault(cls, []).append(path)
    library: dict[str, list[np.ndarray]] = {}
    for cls, paths in groups.items():
        cutouts = []
        for path in paths:
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            if rgba[..., 3].max() == 0:
                raise ValueError(f"cutout {path} has an empty alpha channel")
            cutouts.append(rgba)
        if cutouts:
            library[cls] = cutouts
    if not library:
        raise ValueError(f"no RGBA cutouts found in {sprite_dir}")
    return library


@dataclass
class WalkerSequence:
    """An animated walk cycle: aligned RGBA frames + playback metadata."""

    frames: list[np.ndarray]
    fps: float
    direction: int  # +1 = strides screen-right, -1 = screen-left

    def sprite_at(self, index: int) -> np.ndarray:
        """Ping-pong frame lookup: ...0 1 2 3 2 1 0 1... (no wrap jump)."""
        period = max(1, 2 * len(self.frames) - 2)
        idx = index % period
        if idx >= len(self.frames):
            idx = period - idx
        return self.frames[idx]


def load_walker_library(sprite_dir: str) -> list[WalkerSequence]:
    """Load ``walkers/<name>/frame_*.png`` sequences with their meta.json."""
    sequences = []
    for seq_dir in sorted(glob.glob(os.path.join(sprite_dir, "walkers", "*"))):
        paths = sorted(glob.glob(os.path.join(seq_dir, "frame_*.png")))
        if not paths:
            continue
        with open(os.path.join(seq_dir, "meta.json")) as f:
            meta = json.load(f)
        frames = [np.asarray(Image.open(p).convert("RGBA")) for p in paths]
        sequences.append(
            WalkerSequence(
                frames=frames,
                fps=float(meta.get("fps", 30.0)),
                direction=1 if meta.get("direction", "right") == "right" else -1,
            )
        )
    return sequences


# --------------------------------------------------------------------------
# Component extraction and tracking
# --------------------------------------------------------------------------


@dataclass
class Component:
    bbox: tuple[int, int, int, int]  # y0, y1, x0, x1 (half-open)
    centroid: tuple[float, float]
    area: int


@dataclass
class Track:
    track_id: int
    sprite: np.ndarray
    anchor_x: float
    anchor_y: float
    height: float
    bbox: tuple[int, int, int, int]
    misses: int = 0
    matched: bool = field(default=True)
    # Walk-cycle playback (None for still sprites).
    sequence: WalkerSequence | None = None
    cycle_pos: float = 0.0  # fractional sequence index (random initial phase)
    vel_x: float = 0.0  # EMA of screen-space x velocity, px/frame
    vel_y: float = 0.0  # EMA of screen-space y velocity, px/frame
    last_h: float = 0.0  # last raw component height, for the looming rate
    growth: float = 0.0  # EMA of per-frame relative height growth (looming)
    facing: int = 0  # +1 right, -1 left, 0 undecided (use sequence default)

    def update(self, comp: Component) -> None:
        y0, y1, x0, x1 = comp.bbox
        new_x = (x0 + x1) / 2
        self.vel_x = 0.3 * (new_x - self.anchor_x) + 0.7 * self.vel_x
        self.vel_y = 0.3 * (y1 - self.anchor_y) + 0.7 * self.vel_y
        if self.last_h > 0:
            self.growth = 0.3 * ((y1 - y0) / self.last_h - 1.0) + 0.7 * self.growth
        self.last_h = float(y1 - y0)
        self.anchor_x = EMA_ALPHA * new_x + (1 - EMA_ALPHA) * self.anchor_x
        self.anchor_y = EMA_ALPHA * y1 + (1 - EMA_ALPHA) * self.anchor_y
        self.height = EMA_ALPHA * (y1 - y0) + (1 - EMA_ALPHA) * self.height
        self.bbox = comp.bbox
        self.misses = 0
        self.matched = True

    def _ground_velocity(self, foe: tuple[float, float]) -> tuple[float, float]:
        """Screen velocity minus the looming-expected radial flow (px/frame)."""
        exp_vx = (self.anchor_x - foe[0]) * self.growth
        exp_vy = (self.anchor_y - foe[1]) * self.growth
        return self.vel_x - exp_vx, self.vel_y - exp_vy

    def sprite_for_frame(
        self,
        video_fps: float,
        foe: tuple[float, float],
        rate_override: float | None = None,
    ) -> np.ndarray:
        """Current sprite: still image, or a walk-cycle frame.

        The cycle advances proportionally to the track's height-normalized
        ground-relative speed (natural cadence at walking pace, frozen when
        standing, capped near 2 steps/s) — or at ``rate_override`` when the
        boxes' world speed is known — and is flipped to match the motion
        direction (with hysteresis).
        """
        if self.sequence is None:
            return self.sprite
        gvx, gvy = self._ground_velocity(foe)
        if rate_override is not None:
            rate = rate_override
        else:
            speed = float(np.hypot(gvx, gvy)) / max(self.height, 1.0)
            rate = min(speed / FULL_CADENCE_SPEED, MAX_PLAYBACK_RATE)
        self.cycle_pos += rate * self.sequence.fps / video_fps
        sprite = self.sequence.sprite_at(int(self.cycle_pos))
        if abs(gvx) > FACING_MIN_VEL:
            self.facing = 1 if gvx > 0 else -1
        facing = self.facing or self.sequence.direction
        if facing != self.sequence.direction:
            sprite = sprite[:, ::-1]
        return sprite


def extract_components(
    boxed: np.ndarray, baseline: np.ndarray, threshold: int, min_area: int
) -> list[Component]:
    diff = np.abs(boxed.astype(np.int16) - baseline.astype(np.int16)).max(axis=2)
    mask = diff > threshold
    labels, n = ndimage.label(mask)
    comps: list[Component] = []
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        area = int((labels[sl] == idx).sum())
        if area < min_area:
            continue
        ys, xs = sl
        bbox = (ys.start, ys.stop, xs.start, xs.stop)
        cy, cx = ndimage.center_of_mass(labels[sl] == idx)
        comps.append(Component(bbox, (ys.start + cy, xs.start + cx), area))
    return comps


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    y0 = max(a[0], b[0])
    y1 = min(a[1], b[1])
    x0 = max(a[2], b[2])
    x1 = min(a[3], b[3])
    inter = max(0, y1 - y0) * max(0, x1 - x0)
    if inter == 0:
        return 0.0
    area_a = (a[1] - a[0]) * (a[3] - a[2])
    area_b = (b[1] - b[0]) * (b[3] - b[2])
    return inter / (area_a + area_b - inter)


class Tracker:
    """Greedy IoU tracker with centroid fallback and EMA smoothing."""

    def __init__(
        self,
        sprite_names: list[str],
        seed: int = 7,
        library: dict[str, list[np.ndarray]] | None = None,
        allow_flip: bool = True,
        walkers: list[WalkerSequence] | None = None,
    ) -> None:
        self.sprite_names = sprite_names
        self.rng = np.random.default_rng(seed)
        self.library = library
        self.allow_flip = allow_flip
        self.walkers = walkers or []
        self.tracks: list[Track] = []
        self.next_id = 0

    def _new_sprite(
        self, comp_height: float
    ) -> tuple[np.ndarray, WalkerSequence | None, int]:
        """Sprite for a fresh track: (still_sprite, sequence, phase)."""
        name = self.sprite_names[self.next_id % len(self.sprite_names)]
        # Stable per-track choice: tracks are created in a deterministic
        # order, so a seeded rng keeps the cutout/flip fixed per track id.
        name_salt = zlib.crc32(name.encode()) % 4096
        rng = np.random.default_rng((self.next_id + 1) * 7919 + name_salt)
        if name == "walker":
            # Big on-screen tracks need high-res cycles; low-res ones blur.
            eligible = [
                s
                for s in self.walkers
                if s.frames[0].shape[0] * MAX_WALKER_UPSCALE >= comp_height
            ] or [max(self.walkers, key=lambda s: s.frames[0].shape[0])]
            seq = eligible[int(rng.integers(len(eligible)))]
            phase = int(rng.integers(2 * len(seq.frames)))
            return seq.frames[0], seq, phase  # phase seeds cycle_pos
        if self.library is None:
            return SPRITE_MAKERS[name](self.rng), None, 0
        cutouts = self.library[name]
        sprite = cutouts[int(rng.integers(len(cutouts)))]
        if self.allow_flip and rng.random() < 0.5:
            sprite = sprite[:, ::-1]
        return sprite, None, 0

    def step(self, comps: list[Component]) -> list[Track]:
        for tr in self.tracks:
            tr.matched = False
        unassigned = list(range(len(comps)))

        pairs = [
            (_iou(tr.bbox, comps[ci].bbox), ti, ci)
            for ti, tr in enumerate(self.tracks)
            for ci in unassigned
        ]
        for iou, ti, ci in sorted(pairs, reverse=True):
            if iou < IOU_MATCH_THRESHOLD:
                break
            tr = self.tracks[ti]
            if tr.matched or ci not in unassigned:
                continue
            tr.update(comps[ci])
            unassigned.remove(ci)

        # Centroid fallback for fast-moving small components.
        for ci in list(unassigned):
            comp = comps[ci]
            y0, y1, x0, x1 = comp.bbox
            max_dist = CENTROID_MATCH_FRACTION * float(np.hypot(y1 - y0, x1 - x0))
            best: Track | None = None
            best_dist = max_dist
            for tr in self.tracks:
                if tr.matched:
                    continue
                dist = float(
                    np.hypot(
                        tr.anchor_y - comp.centroid[0], tr.anchor_x - comp.centroid[1]
                    )
                )
                if dist < best_dist:
                    best, best_dist = tr, dist
            if best is not None:
                best.update(comp)
                unassigned.remove(ci)

        for ci in unassigned:
            y0, y1, x0, x1 = comps[ci].bbox
            sprite, sequence, phase = self._new_sprite(float(y1 - y0))
            self.tracks.append(
                Track(
                    track_id=self.next_id,
                    sprite=sprite,
                    anchor_x=(x0 + x1) / 2,
                    anchor_y=float(y1),
                    height=float(y1 - y0),
                    bbox=comps[ci].bbox,
                    sequence=sequence,
                    cycle_pos=float(phase),
                    last_h=float(y1 - y0),
                )
            )
            self.next_id += 1

        survivors = []
        for tr in self.tracks:
            if not tr.matched:
                tr.misses += 1
            if tr.misses <= TRACK_MAX_MISSES:
                survivors.append(tr)
        self.tracks = survivors
        return [tr for tr in self.tracks if tr.matched]


# --------------------------------------------------------------------------
# Harmonization and compositing
# --------------------------------------------------------------------------


def _background_ring_median(
    frame: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    """Median RGB of a ring around the component bbox (expanded 30%)."""
    h, w = frame.shape[:2]
    y0, y1, x0, x1 = bbox
    my = max(8, int(0.3 * (y1 - y0)))
    mx = max(8, int(0.3 * (x1 - x0)))
    oy0, oy1 = max(0, y0 - my), min(h, y1 + my)
    ox0, ox1 = max(0, x0 - mx), min(w, x1 + mx)
    ring = np.ones((oy1 - oy0, ox1 - ox0), dtype=bool)
    ring[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0] = False
    pixels = frame[oy0:oy1, ox0:ox1][ring]
    if pixels.size == 0:
        return np.array([128.0, 128.0, 128.0])
    return np.median(pixels.reshape(-1, 3), axis=0)


def harmonize(sprite: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """Match sprite brightness and color temperature to local background."""
    rgba = sprite.astype(np.float64)
    alpha = rgba[..., 3] > 0
    if not alpha.any():
        return sprite
    sprite_lum = float((rgba[..., :3][alpha] @ LUMA).mean())
    bg_lum = float(bg_rgb @ LUMA)
    gain = np.clip(HARMONIZE_TARGET * bg_lum / max(sprite_lum, 1.0), 0.35, 1.6)
    # Gentle color-temperature pull: blend per-channel gains toward the
    # background's chromaticity (25% strength keeps sprites recognizable).
    bg_chroma = bg_rgb / max(bg_lum, 1.0)
    channel_gain = gain * (0.75 + 0.25 * bg_chroma)
    out = rgba.copy()
    out[..., :3] = np.clip(rgba[..., :3] * channel_gain, 0, 255)
    return out.astype(np.uint8)


def composite_track(
    frame: np.ndarray, track: Track, bg_rgb: np.ndarray, sprite_rgba: np.ndarray
) -> None:
    """Paste shadow + harmonized sprite onto ``frame`` in place."""
    fh, fw = frame.shape[:2]
    target_h = max(8, int(round(track.height)))
    sp_h, sp_w = sprite_rgba.shape[:2]
    target_w = max(4, int(round(sp_w * target_h / sp_h)))

    sprite = harmonize(sprite_rgba, bg_rgb)
    img = Image.fromarray(sprite).resize((target_w, target_h), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(EDGE_BLUR_RADIUS))
    sprite = np.asarray(img).astype(np.float64)

    ax = int(round(track.anchor_x))
    ay = int(round(track.anchor_y))

    _paste_shadow(frame, ax, ay, target_w)

    x0 = ax - target_w // 2
    y0 = ay - target_h
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1 = target_w - max(0, x0 + target_w - fw)
    sy1 = target_h - max(0, y0 + target_h - fh)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    region = frame[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1].astype(np.float64)
    patch = sprite[sy0:sy1, sx0:sx1]
    a = patch[..., 3:4] / 255.0
    blended = patch[..., :3] * a + region * (1 - a)
    frame[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1] = blended.astype(np.uint8)


def _paste_shadow(frame: np.ndarray, ax: int, ay: int, sprite_w: int) -> None:
    sw = int(sprite_w * 1.1)
    sh = max(6, int(sprite_w * 0.28))
    pad = max(4, sh // 2)
    shadow = Image.new("L", (sw + 2 * pad, sh + 2 * pad), 0)
    ImageDraw.Draw(shadow).ellipse((pad, pad, pad + sw, pad + sh), fill=255)
    shadow = shadow.filter(ImageFilter.GaussianBlur(pad // 2 + 2))
    smask = np.asarray(shadow).astype(np.float64) / 255.0 * SHADOW_ALPHA

    fh, fw = frame.shape[:2]
    x0 = ax - smask.shape[1] // 2
    y0 = ay - smask.shape[0] // 2
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1 = smask.shape[1] - max(0, x0 + smask.shape[1] - fw)
    sy1 = smask.shape[0] - max(0, y0 + smask.shape[0] - fh)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    region = frame[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1].astype(np.float64)
    m = smask[sy0:sy1, sx0:sx1][..., None]
    frame[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1] = (region * (1 - m)).astype(
        np.uint8
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    env = os.environ.get
    parser.add_argument(
        "--rgb", default=env("COMPOSITE_RGB"), required=env("COMPOSITE_RGB") is None
    )
    parser.add_argument(
        "--boxed-hdmap",
        default=env("COMPOSITE_BOXED_HDMAP"),
        required=env("COMPOSITE_BOXED_HDMAP") is None,
    )
    parser.add_argument(
        "--baseline-hdmap",
        default=env("COMPOSITE_BASELINE_HDMAP"),
        required=env("COMPOSITE_BASELINE_HDMAP") is None,
    )
    parser.add_argument(
        "--output",
        default=env("COMPOSITE_OUTPUT"),
        required=env("COMPOSITE_OUTPUT") is None,
    )
    parser.add_argument(
        "--sprites",
        default=env("COMPOSITE_SPRITES", "pedestrian"),
        help="Comma-separated sprite classes cycled per track. Procedural "
        "mode offers: " + ", ".join(SPRITE_MAKERS) + ". With --sprite-dir, "
        "names must be cutout classes from the library (e.g. pedestrian, car).",
    )
    parser.add_argument(
        "--sprite-dir",
        default=env("COMPOSITE_SPRITE_DIR"),
        help="Directory of photoreal RGBA cutouts (<class>_<n>.png + optional "
        "manifest.json). When set, cutouts replace procedural sprites; each "
        "track gets a stable random cutout of its class.",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable random horizontal flip of library cutouts.",
    )
    parser.add_argument(
        "--walk-speed-mps",
        type=float,
        default=(
            float(env("COMPOSITE_WALK_SPEED") or 0)
            if env("COMPOSITE_WALK_SPEED")
            else None
        ),
        help="Known world speed of the boxes (m/s). Overrides the screen-space "
        "cadence estimate, which cannot see radial (toward-ego) walking.",
    )
    parser.add_argument("--threshold", type=int, default=DIFF_THRESHOLD)
    parser.add_argument("--min-area", type=int, default=MIN_COMPONENT_AREA)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--no-side-by-side",
        action="store_true",
        help="Write only the composited video, not original|composited.",
    )
    args = parser.parse_args()
    args.sprite_list = [s.strip() for s in args.sprites.split(",") if s.strip()]
    if args.sprite_dir:
        args.library = load_cutout_library(args.sprite_dir)
        args.walkers = load_walker_library(args.sprite_dir)
        known = set(args.library) | ({"walker"} if args.walkers else set())
        unknown = [s for s in args.sprite_list if s not in known]
        if unknown:
            parser.error(
                f"unknown cutout classes {unknown}; "
                f"{args.sprite_dir} offers {sorted(known)}"
            )
    else:
        args.library = None
        args.walkers = []
        unknown = [s for s in args.sprite_list if s not in SPRITE_MAKERS]
        if unknown:
            parser.error(
                f"unknown sprites {unknown}; choose from {sorted(SPRITE_MAKERS)}"
            )
    return args


def main() -> None:
    args = parse_args()
    fps = iio.immeta(args.rgb, plugin="pyav").get("fps", 30.0)
    tracker = Tracker(
        args.sprite_list,
        seed=args.seed,
        library=args.library,
        allow_flip=not args.no_flip,
        walkers=args.walkers,
    )

    rate_override = (
        min(args.walk_speed_mps / NOMINAL_WALK_SPEED_MPS, MAX_PLAYBACK_RATE)
        if args.walk_speed_mps is not None
        else None
    )
    foe: tuple[float, float] | None = None
    frames_out: list[np.ndarray] = []
    heights_all: list[float] = []
    rgb_iter = iio.imiter(args.rgb, plugin="pyav")
    boxed_iter = iio.imiter(args.boxed_hdmap, plugin="pyav")
    base_iter = iio.imiter(args.baseline_hdmap, plugin="pyav")

    for fi, (rgb, boxed, base) in enumerate(zip(rgb_iter, boxed_iter, base_iter)):
        if foe is None:
            foe = (rgb.shape[1] / 2.0, rgb.shape[0] * FOE_Y_FRACTION)
        comps = extract_components(boxed, base, args.threshold, args.min_area)
        active = tracker.step(comps)
        # Paint far (small anchor_y) tracks first so nearer sprites overdraw.
        active.sort(key=lambda tr: tr.anchor_y)
        out = rgb.copy()
        heights = []
        for tr in active:
            bg = _background_ring_median(rgb, tr.bbox)
            composite_track(out, tr, bg, tr.sprite_for_frame(fps, foe, rate_override))
            heights.append(tr.height)
        mean_h = float(np.mean(heights)) if heights else 0.0
        heights_all.extend(heights)
        print(
            f"frame {fi:4d}: components={len(active):3d} mean_sprite_height={mean_h:7.1f}px"
        )
        frames_out.append(
            np.concatenate([rgb, out], axis=1) if not args.no_side_by_side else out
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    iio.imwrite(
        args.output, np.stack(frames_out), plugin="pyav", fps=fps, codec="libx264"
    )
    overall = float(np.mean(heights_all)) if heights_all else 0.0
    print(
        f"wrote {args.output}: {len(frames_out)} frames, "
        f"{tracker.next_id} tracks total, overall mean sprite height {overall:.1f}px"
    )


if __name__ == "__main__":
    main()
