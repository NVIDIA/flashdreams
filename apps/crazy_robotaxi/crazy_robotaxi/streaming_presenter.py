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

"""MJPEG-over-HTTP presenter for Crazy Robotaxi.

CPU-JPEG-encoded frames are served as a
``multipart/x-mixed-replace`` stream with keydown/keyup posted back.

Dependency-free fallback for headless / compute-only hosts with no
graphics GPU; prefer ``omnidreams.webrtc.server`` for a richer viewer.
"""

from __future__ import annotations

import hmac
import json
import math as _math
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from loguru import logger
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.input.keyboard import KeyboardState
from omnidreams_game_engine.loading_overlay import render_loading_overlay
from omnidreams_game_engine.physx_debug import select_presented_rgb
from omnidreams_game_engine.types import (
    CameraCalibration,
    DriverCommand,
    PresentedFrame,
)
from omnidreams_game_engine.visual_flare import (
    CollisionVisualFlare,
    darken_rgb,
)
from PIL import Image, ImageDraw

from crazy_robotaxi.game import (
    TaxiCameraMarkerProjection,
    project_segment_pose_to_bev,
    project_target_to_bev,
    project_taxi_markers_to_camera,
)
from flashdreams.serving.realtime.frame_bus import LatestFrameBus
from flashdreams.serving.realtime.media import (
    encode_rgb_frame_to_jpeg,
    rgb_frame_to_uint8,
)

# Boundary marker embedded in the multipart response. The exact string
# doesn't matter as long as it never appears inside a JPEG payload (they
# start with the JPEG SOI marker 0xFFD8 so ``--interactive_drive`` is always safe).
_MULTIPART_BOUNDARY = "interactive_drive"

# Per-connection kernel send-buffer cap for the MJPEG streams. The
# LatestFrameBus already drops to the newest frame at the application
# level, but with the default SO_SNDBUF (often several MB) a slow client
# (VPN / SSH tunnel) buffers seconds of already-encoded frames inside the
# kernel and watches stale pixels. Capping the buffer to roughly two
# frames' worth makes ``wfile.write`` block as soon as the link is
# saturated, so the next bus read skips straight to the newest frame.
_STREAM_SNDBUF_BYTES = 128 * 1024

# How often each /stream connection logs its sent/dropped/bandwidth line.
_STREAM_STATS_INTERVAL_S = 10.0


class _StreamClientStats:
    """Per-connection sent/dropped/bytes accounting for an MJPEG stream.

    Dropped frames are inferred from the bus's monotonically increasing
    publication counts: consuming count ``c`` after last seeing ``last``
    means ``c - last - 1`` intermediate frames were skipped (drop-to-latest
    behavior under backpressure). Pure bookkeeping so it is unit-testable
    without sockets.
    """

    def __init__(
        self,
        label: str,
        *,
        log_interval_s: float = _STREAM_STATS_INTERVAL_S,
    ) -> None:
        self._label = label
        self._log_interval_s = float(log_interval_s)
        self.sent_frames = 0
        self.dropped_frames = 0
        self.sent_bytes = 0
        self._started_at: float | None = None
        self._last_logged_at: float | None = None

    def record(
        self, *, frame_count: int, last_seen_count: int, nbytes: int, now: float
    ) -> str | None:
        """Account one delivered frame; return a log line when one is due."""
        if self._started_at is None:
            self._started_at = now
            self._last_logged_at = now
        if last_seen_count > 0:
            self.dropped_frames += max(0, frame_count - last_seen_count - 1)
        self.sent_frames += 1
        self.sent_bytes += int(nbytes)
        assert self._last_logged_at is not None
        if now - self._last_logged_at < self._log_interval_s:
            return None
        self._last_logged_at = now
        return self.summary(now=now)

    def summary(self, *, now: float) -> str:
        """One-line sent/dropped/bandwidth summary for this connection."""
        elapsed = max(1e-6, now - (self._started_at if self._started_at else now))
        fps = self.sent_frames / elapsed
        kbps = self.sent_bytes / elapsed / 1024.0
        return (
            f"[presenter] {self._label}: sent={self.sent_frames} "
            f"dropped={self.dropped_frames} "
            f"({fps:.1f} fps, {kbps:.0f} KiB/s over {elapsed:.1f}s)"
        )


# Browser ``event.key`` values to the keysym strings that
# :meth:`KeyboardDriveState.set_key` (in ``demo.py``) recognises. The
# slangpy HUD path uses the SDL/pygame-style ``"Up"``/``"Down"``/etc.
# keysyms locally; the browser sends ``ArrowUp``/``ArrowDown`` instead,
# so we re-map at the network boundary rather than extend
# :func:`_keyboard_drive_key` with browser-specific aliases.
_BROWSER_KEY_TO_DRIVE_KEYSYM: dict[str, str] = {
    "w": "w",
    "W": "w",
    "a": "a",
    "A": "a",
    "s": "s",
    "S": "s",
    "d": "d",
    "D": "d",
    "ArrowUp": "Up",
    "ArrowDown": "Down",
    "ArrowLeft": "Left",
    "ArrowRight": "Right",
    " ": "space",
    "Spacebar": "space",
}

_BROWSER_KEY_TO_VIEW_MODE: dict[str, str] = {
    # 1 = world-model RGB (the generated drive view, the main demo output).
    # 2 = HDMap with traffic (the rasterizer's conditioning input).
    # 3 = active PhysX colliders and invisible walls.
    "1": "model_rgb",
    "2": "rgb",
    "3": "physx",
}

_RESERVED_BROWSER_KEYS = frozenset(
    set(_BROWSER_KEY_TO_DRIVE_KEYSYM) | set(_BROWSER_KEY_TO_VIEW_MODE) | {"r", "R"}
)
"""Browser keys ``_apply_control`` dispatches itself; extras may not shadow them."""


def _validate_extra_key_handlers(
    handlers: dict[str, Callable[[], None]] | None,
) -> dict[str, Callable[[], None]]:
    """Reject handler keys that collide with the presenter's own bindings.

    A shadowed registration would never fire (``_apply_control`` returns
    before the extra-handler lookup), so failing loudly at construction
    beats a silently dead key. Single-character keys are normalized to
    lowercase because the browser posts ``e.key`` verbatim (``K`` when
    Shift is held) while handlers are looked up case-insensitively.
    """
    validated: dict[str, Callable[[], None]] = {}
    for key, handler in (handlers or {}).items():
        validated[key.lower() if len(key) == 1 else key] = handler
    conflicts = sorted(key for key in validated if key in _RESERVED_BROWSER_KEYS)
    if conflicts:
        raise ValueError(
            f"extra_key_handlers may not rebind reserved keys: {conflicts}"
        )
    return validated


class _KeyboardDriveSink:
    """In-process duck-typed ``ControlClient`` writing to ``KeyboardState``.

    Duplicates the HUD's ``KeyboardStateDriveSink`` rather than importing it,
    to keep SlangPy / Vulkan out of the streaming presenter's import graph.
    """

    def __init__(self, keyboard: KeyboardState) -> None:
        self._keyboard = keyboard

    def set_drive(
        self,
        *,
        steer: float,
        throttle: float,
        brake: float,
        handbrake: bool = False,
        reverse: bool = False,
    ) -> None:
        self._keyboard.set_drive_command(
            DriverCommand(
                throttle=max(0.0, min(1.0, throttle)),
                brake=max(0.0, min(1.0, brake)),
                steer=max(-1.0, min(1.0, steer)),
                handbrake=bool(handbrake),
                reverse=bool(reverse),
                steer_is_direct=True,
                manual_control=True,
            ),
            source="browser",
        )

    def release_all(self) -> None:
        self._keyboard.set_drive_command(None, source="browser")

    # No-ops in-process: the streaming presenter writes directly via
    # ``KeyboardState`` from its HTTP handler thread.
    def set_key(self, key: str, down: bool) -> None:  # noqa: ARG002
        return

    def pulse(self, key: str) -> None:  # noqa: ARG002
        return


def _print_port_conflict_help(host: str, port: int, exc: OSError) -> None:
    """Print a helpful message when the HTTP server can't bind to the port."""
    logger.error(
        f"\n[presenter] MJPEG server failed to start: port {port} is already in use.\n"
        f"            ({exc})\n",
    )
    # Try to show which process is using the port (Linux: ss, macOS/BSD: lsof).
    shown = False
    if shutil.which("ss"):
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            logger.warning(
                f"[presenter] The following process is blocking port {port}:\n",
            )
            logger.warning(result.stdout)
            shown = True
    if not shown and shutil.which("lsof"):
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            logger.warning(
                f"[presenter] The following process is blocking port {port}:\n",
            )
            logger.warning(result.stdout)
            shown = True
    if not shown:
        logger.warning(
            f"[presenter] Could not determine which process is using port {port}.\n",
        )
    logger.warning(
        f"[presenter] To fix this, either:\n"
        f"  1. Stop the process above, or\n"
        f"  2. Choose a different port: --stream-mjpeg :{port + 1}\n",
    )


# Single HTML page served at ``/``. Shows the MJPEG stream, an HTML/CSS
# HUD with a speed readout and WASD-indicator chiclets keyed off the
# locally-tracked DOWN_KEYS set, and JS that forwards keydown/keyup to
# ``/control``. The HUD is intentionally inline (no template loading)
# because the presenter is meant to be a single-file drop-in for hosts
# without local windowing. The browser reads the speed snapshot from
# ``/state`` once every 100 ms; that's cheap (a ~80-byte JSON blob) and
# keeps the readout responsive without flooding the server.
_INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>interactive_drive (MJPEG)</title>
<style>
  html, body { margin: 0; padding: 0; background: #111; height: 100%; font-family: sans-serif; }
  body { display: flex; align-items: center; justify-content: center; }
  img#stream { max-width: 100%; max-height: 100%; object-fit: contain; image-rendering: pixelated; }
  .hint { position: fixed; top: 8px; left: 8px; color: #aaa; font-size: 12px; }
  .hud {
    position: fixed; bottom: 24px; left: 24px;
    display: flex; gap: 28px; align-items: flex-end;
    pointer-events: none;
    color: white;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.85), 0 0 4px rgba(0, 0, 0, 0.85);
  }
  .speed {
    display: flex; align-items: baseline; gap: 6px;
    line-height: 1;
  }
  .speed-value { font-size: 56px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .speed-unit { font-size: 18px; opacity: 0.75; letter-spacing: 0.04em; text-transform: uppercase; }
  .speed.disconnected .speed-value { color: #888; }
  .taxi-hud {
    position: fixed; top: 18px; left: 50%; transform: translateX(-50%);
    min-width: 360px; padding: 10px 18px;
    border-radius: 12px; background: rgba(0, 0, 0, 0.72);
    border: 2px solid #76b900; color: #76b900;
    text-align: center; pointer-events: none;
    text-shadow: 0 2px 5px rgba(0, 0, 0, 0.9);
  }
  .taxi-hud.hidden, .taxi-map.hidden { display: none; }
  .taxi-arrow {
    width: 58px; height: 64px; margin: 0 auto 2px;
    transform-origin: center; color: inherit;
  }
  .taxi-arrow svg { display: block; width: 100%; height: 100%; overflow: visible; }
  .taxi-arrow path {
    fill: currentColor; stroke: black; stroke-width: 3px;
    stroke-linejoin: round; paint-order: stroke fill;
  }
  .taxi-status { font-size: 18px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .taxi-event { min-height: 18px; color: white; font-weight: 700; margin-top: 3px; }
  .taxi-hud.dropoff { color: #c89632; border-color: #c89632; }
  .taxi-map {
    position: fixed; top: 18px; right: 18px; width: 240px;
    border: 3px solid rgba(255, 255, 255, 0.85); border-radius: 10px;
    overflow: hidden; background: #222; pointer-events: none;
  }
  .taxi-map img { display: block; width: 100%; height: auto; }
  .taxi-boundaries, #taxi-pins { position: absolute; inset: 0; width: 100%; height: 100%; }
  .taxi-boundaries { overflow: hidden; }
  .taxi-boundaries line { stroke: rgb(235, 50, 50); stroke-width: 1.8; vector-effect: non-scaling-stroke; }
  .taxi-pin {
    position: absolute; width: 18px; height: 18px; border-radius: 50%;
    border: 3px solid white; background: #76b900;
    transform: translate(-50%, -50%); box-shadow: 0 2px 6px black;
  }
  .taxi-pin.dropoff { background: #c89632; }
  .game-over {
    position: fixed; inset: 0; display: flex; align-items: center;
    justify-content: center; background: rgba(0, 0, 0, 0.72);
    color: white; z-index: 20;
  }
  .game-over.hidden { display: none; }
  .game-over-card {
    width: min(520px, calc(100vw - 48px)); max-height: calc(100vh - 48px);
    overflow-y: auto; box-sizing: border-box; padding: 28px;
    border: 3px solid #76b900; border-radius: 16px;
    background: rgba(12, 12, 18, 0.97); text-align: center;
  }
  .game-over-card h1 { margin: 0 0 8px; color: #76b900; }
  .final-score { font-size: 24px; font-weight: 700; margin-bottom: 20px; }
  .name-entry.hidden, .leaderboard.hidden { display: none; }
  .name-entry input {
    width: 100%; box-sizing: border-box; padding: 10px 12px; margin: 10px 0;
    border: 2px solid white; border-radius: 7px; background: #242432;
    color: white; font-size: 20px; text-align: center;
  }
  .game-button {
    padding: 10px 18px; border: none; border-radius: 7px;
    background: #76b900; color: #111; font-weight: 700; cursor: pointer;
  }
  .name-error { min-height: 18px; margin-top: 8px; color: #ff8585; }
  .score-table { width: 100%; border-collapse: collapse; margin: 14px 0 22px; }
  .score-table td { padding: 5px 8px; text-align: left; }
  .score-table td:first-child { width: 36px; text-align: right; color: #aaa; }
  .score-table td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  .score-table tr.current { color: #c89632; font-weight: 700; }
  .keys { display: flex; flex-direction: column; gap: 6px; align-items: center; }
  .key-row { display: flex; gap: 6px; }
  .key {
    width: 38px; height: 38px;
    border-radius: 7px;
    background: rgba(0, 0, 0, 0.45);
    border: 2px solid rgba(255, 255, 255, 0.35);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 700;
    transition: background-color 0.05s ease-out, border-color 0.05s ease-out, color 0.05s ease-out;
  }
  .key.down {
    background: rgba(255, 255, 255, 0.92);
    border-color: white;
    color: #111;
  }
  .scene-picker {
    position: fixed; bottom: 16px; right: 16px;
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 10px;
    color: white;
    font-size: 12px;
    display: flex; flex-direction: column;
    max-height: 60vh;
    backdrop-filter: blur(6px);
    overflow: hidden;
    /* Animate the collapse so the toggle feels physical rather than a
       hard show/hide. ``max-height`` is the lever rather than ``display``
       because ``display: none`` short-circuits transitions. */
    transition: max-height 0.18s ease-out;
  }
  .scene-picker.hidden { display: none; }
  .scene-picker.collapsed { max-height: 38px; }
  .scene-picker-toggle {
    background: none; border: none; color: white;
    padding: 9px 12px;
    display: flex; align-items: center; gap: 8px;
    cursor: pointer;
    font-size: 11px; font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    width: 100%;
    text-align: left;
    flex-shrink: 0;
    pointer-events: auto;
    user-select: none;
  }
  .scene-picker-toggle:hover { background: rgba(255, 255, 255, 0.06); }
  .scene-picker-count {
    opacity: 0.55;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    font-size: 11px;
  }
  .scene-picker-chevron {
    margin-left: auto;
    font-size: 10px;
    transition: transform 0.18s ease-out;
  }
  .scene-picker.collapsed .scene-picker-chevron {
    transform: rotate(-90deg);
  }
  .scene-picker-list {
    display: flex; flex-direction: column; gap: 6px;
    padding: 0 10px 10px 10px;
    overflow-y: auto;
  }
  /* Hide the list's scroll viewport entirely while the panel is
     collapsed so no scrollbar artifacts leak through the parent's
     ``overflow: hidden`` clipping. */
  .scene-picker.collapsed .scene-picker-list { overflow: hidden; }
  /* Replace Chromium's default scrollbar (which carries the up/down
     arrow buttons that were poking out the bottom-right of the
     collapsed panel) with a slim button-less rail. Firefox's
     standards-track ``scrollbar-width`` covers the same ground. */
  .scene-picker-list { scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.22) transparent; }
  .scene-picker-list::-webkit-scrollbar { width: 6px; }
  .scene-picker-list::-webkit-scrollbar-track { background: transparent; }
  .scene-picker-list::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.22);
    border-radius: 3px;
  }
  .scene-picker-list::-webkit-scrollbar-button { display: none; }
  .scene-picker-list::-webkit-scrollbar-corner { background: transparent; }
  .scene-card {
    width: 160px;
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
    border: 2px solid transparent;
    transition: border-color 0.1s, transform 0.05s;
    background: rgba(255, 255, 255, 0.05);
    pointer-events: auto;
    user-select: none;
  }
  .scene-card:hover { border-color: rgba(120, 200, 255, 0.7); }
  .scene-card.loading {
    border-color: rgba(120, 200, 255, 1.0);
    pointer-events: none;
    opacity: 0.7;
  }
  .scene-card img {
    width: 100%; height: 72px;
    object-fit: cover;
    display: block;
    background: #222;
  }
  .scene-card .scene-label {
    padding: 6px 8px;
    font-size: 11px; line-height: 1.3;
  }
  /* Weather-variant pills, shown only for multi-variant scenes. */
  .scene-variants {
    display: flex; flex-wrap: wrap; gap: 4px;
    padding: 0 8px 8px 8px;
  }
  .variant-pill {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 999px;
    color: white;
    font-size: 10px;
    padding: 2px 8px;
    cursor: pointer;
    pointer-events: auto;
    user-select: none;
    transition: background-color 0.1s, border-color 0.1s;
  }
  .variant-pill:hover { background: rgba(120, 200, 255, 0.3); border-color: rgba(120, 200, 255, 0.7); }
  .variant-pill.loading { border-color: rgba(120, 200, 255, 1.0); opacity: 0.7; pointer-events: none; }
</style>
</head>
<body>
<img id="stream">
<div class="taxi-hud hidden" id="taxi-hud">
  <div class="taxi-arrow" id="taxi-arrow" aria-hidden="true">
    <svg viewBox="0 0 64 72">
      <path d="M32 3 L57 36 L42 36 L42 68 L22 68 L22 36 L7 36 Z"></path>
    </svg>
  </div>
  <div class="taxi-status" id="taxi-status"></div>
  <div class="taxi-event" id="taxi-event"></div>
</div>
<div class="taxi-map hidden" id="taxi-map">
  <img id="taxi-bev">
  <svg class="taxi-boundaries" id="taxi-boundaries" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
  <div id="taxi-pins"></div>
</div>
<div class="game-over hidden" id="game-over">
  <div class="game-over-card">
    <h1 id="game-over-title">HIGH SCORES</h1>
    <div class="final-score" id="final-score"></div>
    <form class="name-entry hidden" id="name-entry">
      <div id="high-score-rank"></div>
      <input id="player-name" maxlength="12" autocomplete="nickname"
             pattern="[A-Za-z0-9 _-]{1,12}" placeholder="Your name" required>
      <button class="game-button" type="submit">Save score</button>
      <div class="name-error" id="name-error"></div>
    </form>
    <div class="leaderboard hidden" id="leaderboard">
      <table class="score-table"><tbody id="score-rows"></tbody></table>
      <button class="game-button" id="new-game" type="button">New Game</button>
    </div>
  </div>
</div>
<div class="hint" id="drive-hint">WASD / Arrows = Drive &middot; 1 = World-Model RGB &middot; 2 = HDMap &middot; 3 = PhysX &middot; R = Reset Rollout</div>
<div class="scene-picker hidden" id="scene-picker">
  <button class="scene-picker-toggle" id="scene-picker-toggle" type="button">
    <span>Scenes</span>
    <span class="scene-picker-count" id="scene-picker-count"></span>
    <span class="scene-picker-chevron">&#9662;</span>
  </button>
  <div class="scene-picker-list" id="scene-picker-list"></div>
</div>
<div class="hud">
  <div class="speed disconnected" id="speed">
    <span class="speed-value" id="speed-value">--</span>
    <span class="speed-unit">mph</span>
  </div>
  <div class="keys">
    <div class="key-row"><div class="key" id="key-w">W</div></div>
    <div class="key-row">
      <div class="key" id="key-a">A</div>
      <div class="key" id="key-s">S</div>
      <div class="key" id="key-d">D</div>
    </div>
  </div>
</div>
<script>
// Shared stream token: the page embeds it from its own URL (?token=...)
// and appends it to every request it makes (streams, fetches, thumbnails).
// Empty when the server runs without --stream-token; withToken() is then
// the identity function and the page behaves exactly as before.
const STREAM_TOKEN = new URLSearchParams(window.location.search).get('token') || '';
function withToken(url) {
  if (!STREAM_TOKEN) return url;
  return url + (url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(STREAM_TOKEN);
}
document.getElementById('stream').src = withToken('/stream');
document.getElementById('taxi-bev').src = withToken('/bev_stream');
const DOWN_KEYS = new Set();
const INDICATOR_FOR_KEY = {
  "w":"w","W":"w","ArrowUp":"w",
  "a":"a","A":"a","ArrowLeft":"a",
  "s":"s","S":"s","ArrowDown":"s",
  "d":"d","D":"d","ArrowRight":"d",
};
const PRESSED_INDICATORS = new Map();   // indicator-name -> count of held keys

function paintIndicator(name) {
  const el = document.getElementById("key-" + name);
  if (!el) return;
  const held = (PRESSED_INDICATORS.get(name) || 0) > 0;
  el.classList.toggle("down", held);
}
function bumpIndicator(name, delta) {
  const next = Math.max(0, (PRESSED_INDICATORS.get(name) || 0) + delta);
  PRESSED_INDICATORS.set(name, next);
  paintIndicator(name);
}
function send(key, down) {
  if (down && DOWN_KEYS.has(key)) return;   // debounce: browsers send keydown repeatedly while held
  if (!down) DOWN_KEYS.delete(key); else DOWN_KEYS.add(key);
  // Update the local indicator UI immediately (no round-trip latency).
  const indicator = INDICATOR_FOR_KEY[key];
  if (indicator) bumpIndicator(indicator, down ? 1 : -1);
  fetch(withToken('/control?key=' + encodeURIComponent(key) + '&down=' + (down ? 1 : 0)))
    .catch(() => {});                       // ignore network hiccups, next event will resync
}
// Skip key handling when focus is on a form input (e.g. a future
// settings panel). The scene picker is now click-driven so the
// keyboard never lands on a button there.
function shouldIgnoreKey(e) {
  const t = e.target;
  if (!t) return false;
  const tag = t.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA';
}
document.addEventListener('keydown', e => { if (!shouldIgnoreKey(e)) send(e.key, true); });
document.addEventListener('keyup', e => { if (!shouldIgnoreKey(e)) send(e.key, false); });
function releaseAllKeys() {
  DOWN_KEYS.forEach(k => send(k, false));
  PRESSED_INDICATORS.clear();
  ["w","a","s","d"].forEach(paintIndicator);
}
// When the page loses focus we must release all keys so the car doesn't keep steering.
window.addEventListener('blur', releaseAllKeys);
// Speed readout polling. 100 ms keeps the digit lively without
// generating meaningful HTTP load (~10 req/s of <100 B each).
const speedEl = document.getElementById('speed');
const speedValueEl = document.getElementById('speed-value');
const driveHintEl = document.getElementById('drive-hint');
const taxiHudEl = document.getElementById('taxi-hud');
const taxiArrowEl = document.getElementById('taxi-arrow');
const taxiStatusEl = document.getElementById('taxi-status');
const taxiEventEl = document.getElementById('taxi-event');
const taxiMapEl = document.getElementById('taxi-map');
const taxiBoundariesEl = document.getElementById('taxi-boundaries');
const taxiPinsEl = document.getElementById('taxi-pins');
const gameOverEl = document.getElementById('game-over');
const gameOverTitleEl = document.getElementById('game-over-title');
const finalScoreEl = document.getElementById('final-score');
const nameEntryEl = document.getElementById('name-entry');
const highScoreRankEl = document.getElementById('high-score-rank');
const playerNameEl = document.getElementById('player-name');
const nameErrorEl = document.getElementById('name-error');
const leaderboardEl = document.getElementById('leaderboard');
const scoreRowsEl = document.getElementById('score-rows');
const newGameEl = document.getElementById('new-game');
const MPS_TO_MPH = 2.236936;
let previousTaxiSession = null;
function paintGameOver(taxi) {
  const state = taxi && taxi.session_state;
  const gameOver = state === 'awaiting_name' || state === 'leaderboard';
  gameOverEl.classList.toggle('hidden', !gameOver);
  if (!gameOver) {
    previousTaxiSession = state || null;
    return;
  }
  finalScoreEl.textContent = `FINAL SCORE  ${taxi.score}`;
  if (previousTaxiSession !== state) releaseAllKeys();
  const enteringName = state === 'awaiting_name';
  gameOverTitleEl.textContent = enteringName ? 'NEW HIGH SCORE!' : 'HIGH SCORES';
  nameEntryEl.classList.toggle('hidden', !enteringName);
  leaderboardEl.classList.toggle('hidden', enteringName);
  if (enteringName) {
    highScoreRankEl.textContent = `Congratulations — you reached #${taxi.high_score_rank}`;
    if (previousTaxiSession !== state) {
      playerNameEl.value = '';
      nameErrorEl.textContent = '';
      setTimeout(() => playerNameEl.focus(), 0);
    }
  } else {
    scoreRowsEl.replaceChildren();
    (taxi.leaderboard || []).forEach((entry, index) => {
      const row = document.createElement('tr');
      if (index + 1 === taxi.high_score_rank) row.classList.add('current');
      [String(index + 1), entry.name, String(entry.score)].forEach(value => {
        const cell = document.createElement('td');
        cell.textContent = value;
        row.appendChild(cell);
      });
      scoreRowsEl.appendChild(row);
    });
  }
  previousTaxiSession = state;
}
function paintTaxi(taxi) {
  driveHintEl.textContent = taxi
    ? 'WASD / Arrows = Drive · Space = Handbrake · K = Skin · C = Coins · 1 = World-Model RGB · 2 = HDMap · 3 = PhysX · R = Reset Rollout'
    : 'WASD / Arrows = Drive · 1 = World-Model RGB · 2 = HDMap · 3 = PhysX · R = Reset Rollout';
  if (!taxi) {
    taxiHudEl.classList.add('hidden');
    taxiMapEl.classList.add('hidden');
    paintGameOver(null);
    return;
  }
  paintGameOver(taxi);
  if (taxi.session_state !== 'playing') {
    taxiHudEl.classList.add('hidden');
    taxiMapEl.classList.add('hidden');
    return;
  }
  const dropoff = taxi.phase === 'to_dropoff';
  taxiHudEl.classList.remove('hidden');
  taxiHudEl.classList.toggle('dropoff', dropoff);
  taxiArrowEl.style.transform = `rotate(${-taxi.relative_bearing_rad * 180 / Math.PI}deg)`;
  const phase = dropoff ? 'DROPOFF' : 'PICKUP';
  const timer = typeof taxi.remaining_time_s === 'number'
    ? `  ${taxi.remaining_time_s.toFixed(1)}s` : '';
  const highScore = typeof taxi.high_score === 'number'
    ? `  HIGH ${taxi.high_score}` : '';
  taxiStatusEl.textContent = `GAME ${taxi.global_remaining_time_s.toFixed(1)}s  ${phase}  ${Math.round(taxi.distance_m)}m${timer}  SCORE ${taxi.score}${highScore}`;
  taxiEventEl.textContent = taxi.event === 'pickup_complete'
    ? 'PASSENGER PICKED UP'
    : (taxi.event === 'fare_complete'
      ? `FARE COMPLETE  +${taxi.awarded_points}  +${taxi.awarded_global_time_s}s`
      : (taxi.event === 'time_expired' ? 'TIME EXPIRED' : ''));
  const markers = taxi.bev_targets || [];
  const showMap = taxi.bev_enabled;
  taxiMapEl.classList.toggle('hidden', !showMap);
  taxiBoundariesEl.replaceChildren();
  (taxi.bev_enclosure_segments || []).forEach(segment => {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', `${segment.u0 * 100}`);
    line.setAttribute('y1', `${segment.v0 * 100}`);
    line.setAttribute('x2', `${segment.u1 * 100}`);
    line.setAttribute('y2', `${segment.v1 * 100}`);
    taxiBoundariesEl.appendChild(line);
  });
  taxiPinsEl.replaceChildren();
  markers.filter(marker => marker.visible).forEach(marker => {
    const pin = document.createElement('div');
    pin.className = `taxi-pin${dropoff ? ' dropoff' : ''}`;
    pin.style.left = `${marker.u * 100}%`;
    pin.style.top = `${marker.v * 100}%`;
    taxiPinsEl.appendChild(pin);
  });
}
nameEntryEl.addEventListener('submit', async event => {
  event.preventDefault();
  nameErrorEl.textContent = '';
  try {
    const response = await fetch(withToken('/taxi/name'), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: playerNameEl.value}),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || 'Could not save that name.');
    }
  } catch (error) {
    nameErrorEl.textContent = error.message || 'Could not save that name.';
  }
});
newGameEl.addEventListener('click', async () => {
  await fetch(withToken('/control?key=r&down=1')).catch(() => {});
  await fetch(withToken('/control?key=r&down=0')).catch(() => {});
});
async function pollState() {
  try {
    const r = await fetch(withToken('/state'), { cache: 'no-store' });
    if (!r.ok) throw new Error('http ' + r.status);
    const s = await r.json();
    if (typeof s.speed_mps === 'number') {
      speedValueEl.textContent = Math.round(s.speed_mps * MPS_TO_MPH).toString();
      speedEl.classList.remove('disconnected');
    } else {
      speedValueEl.textContent = '--';
      speedEl.classList.add('disconnected');
    }
    paintTaxi(s.taxi || null);
  } catch {
    speedValueEl.textContent = '--';
    speedEl.classList.add('disconnected');
    paintTaxi(null);
  }
}
setInterval(pollState, 100);
pollState();

// Scene picker. Hidden until /scenes returns at least one entry,
// then renders as a panel in the bottom-right. Auto-expanded on first
// load because nothing happens until the user picks a scene -- the
// server is blocked on ``wait_for_scene_selection`` and the MJPEG
// stream shows the "Select a scene to begin driving" overlay frame.
// After the first pick it auto-collapses (and click-outside collapses
// thereafter), so the panel stays out of the way during driving.
const scenePicker = document.getElementById('scene-picker');
const scenePickerList = document.getElementById('scene-picker-list');
const scenePickerToggle = document.getElementById('scene-picker-toggle');
const scenePickerCount = document.getElementById('scene-picker-count');
let SCENES = [];
let firstSceneLoaded = false;
function setScenePickerCollapsed(collapsed) {
  scenePicker.classList.toggle('collapsed', collapsed);
}
scenePickerToggle.addEventListener('click', () => {
  setScenePickerCollapsed(!scenePicker.classList.contains('collapsed'));
});
// Click outside the panel collapses it -- but only after the user has
// actually picked their first scene. Pre-selection clicks (e.g. the
// user clicking on the camera area to dismiss something) don't tuck
// the picker away, since the panel is the only way to start driving.
document.addEventListener('mousedown', e => {
  if (!firstSceneLoaded) return;
  if (scenePicker.classList.contains('hidden')) return;
  if (scenePicker.contains(e.target)) return;
  setScenePickerCollapsed(true);
});
async function fetchScenes() {
  try {
    const r = await fetch(withToken('/scenes'), { cache: 'no-store' });
    if (!r.ok) return;
    const data = await r.json();
    SCENES = Array.isArray(data.scenes) ? data.scenes : [];
    scenePickerCount.textContent = SCENES.length ? `(${SCENES.length})` : '';
    if (!SCENES.length) {
      scenePicker.classList.add('hidden');
      return;
    }
    scenePickerList.innerHTML = '';
    SCENES.forEach((s, i) => {
      const card = document.createElement('div');
      card.className = 'scene-card';
      card.dataset.idx = String(i);
      if (s.has_thumbnail) {
        const img = document.createElement('img');
        img.src = withToken('/thumbnail?scene=' + encodeURIComponent(s.path));
        img.alt = '';
        img.onerror = () => { img.style.display = 'none'; };
        card.appendChild(img);
      }
      const label = document.createElement('div');
      label.className = 'scene-label';
      label.textContent = s.label || ('Scene ' + (i + 1));
      card.appendChild(label);
      // Clicking the card (outside a pill) loads the default variant.
      card.addEventListener('click', () => loadScene(i, card));
      const variants = Array.isArray(s.variants) ? s.variants : [];
      if (variants.length > 1) {
        const row = document.createElement('div');
        row.className = 'scene-variants';
        variants.forEach(v => {
          const pill = document.createElement('button');
          pill.className = 'variant-pill';
          pill.type = 'button';
          pill.textContent = variantLabel(v);
          pill.addEventListener('click', e => {
            e.stopPropagation();   // don't also trigger the card's default-variant load
            loadScene(i, card, v, pill);
          });
          row.appendChild(pill);
        });
        card.appendChild(row);
      }
      scenePickerList.appendChild(card);
    });
    scenePicker.classList.remove('hidden');
  } catch {}
}
function variantLabel(v) {
  const labels = {
    default: 'Default', clear: 'Clear', snow: 'Snow', rain: 'Rain',
  };
  return labels[v] || (v.charAt(0).toUpperCase() + v.slice(1));
}
async function loadScene(idx, card, variant, pill) {
  const scene = SCENES[idx];
  if (!scene) return;
  (pill || card).classList.add('loading');
  try {
    let url = '/scene/select?scene=' + encodeURIComponent(scene.path);
    // No variant -> server uses the scene's default; a pill selects one.
    if (variant) url += '&variant=' + encodeURIComponent(variant);
    await fetch(withToken(url), { method: 'GET', cache: 'no-store' });
  } catch {}
  // Tuck the panel away so the user gets the camera view back; the
  // scene transition itself is driven by the server-side loop. From
  // this point on, click-outside dismissal is enabled too.
  firstSceneLoaded = true;
  setScenePickerCollapsed(true);
  setTimeout(() => { (pill || card).classList.remove('loading'); }, 1500);
}
fetchScenes();
</script>
</body>
</html>
"""


class MJPEGStreamingPresenter:
    """Drop-in replacement for :class:`SlangPyPresenter` that streams frames
    over HTTP instead of opening a Vulkan swapchain window.

    Exposes the same duck-typed interface consumed by
    :class:`omnidreams_game_engine.app.InteractiveDriveApp`:
    ``should_close`` / ``process_events`` / ``present_frame`` / ``close``.
    The simulation thread doesn't know the presenter changed.
    """

    def __init__(
        self,
        raster: RasterConfig,
        keyboard: KeyboardState,
        bind_host: str,
        bind_port: int,
        *,
        jpeg_quality: int = 85,
        stream_scale: float = 1.0,
        scenes: tuple[dict[str, object], ...] = (),
        thumbnails: dict[str, bytes] | None = None,
        stream_token: str | None = None,
        extra_key_handlers: dict[str, Callable[[], None]] | None = None,
    ) -> None:
        self._raster = raster
        # Shared-secret gate for every HTTP endpoint. ``None`` / empty keeps
        # the historical open behavior; a non-empty token means requests must
        # carry it (``?token=`` or ``X-Stream-Token`` header) or get a 403.
        self._stream_token = (stream_token or "").strip() or None
        self._keyboard = keyboard
        # Composition-root key extensions (e.g. live-edit abilities): browser
        # keysym (lowercase for letters) -> zero-arg callback, fired on
        # keydown. Mirrors ``SlangPyHudPresenter``'s ``extra_key_handlers``
        # so both transports expose the same hook.
        self._extra_key_handlers = _validate_extra_key_handlers(extra_key_handlers)
        self._visual_flare = CollisionVisualFlare()
        self._taxi_enabled = False
        self._bev_config: BevConfig | None = None
        self._taxi_camera_calibration: CameraCalibration | None = None
        self._taxi_camera_models: dict[tuple[int, int], FThetaCameraModel] = {}
        self._taxi_enclosure_segments_world = np.empty((0, 2, 3), dtype=np.float32)
        self._jpeg_quality = int(jpeg_quality)
        if not 1 <= self._jpeg_quality <= 100:
            raise ValueError(
                f"stream JPEG quality must be in [1, 100], got {jpeg_quality}"
            )
        self._stream_scale = float(stream_scale)
        if not 0.1 <= self._stream_scale <= 1.0:
            raise ValueError(f"stream scale must be in [0.1, 1.0], got {stream_scale}")
        self._stop_event = threading.Event()
        self._frame_bus = LatestFrameBus[bytes]()
        # BEV minimap stream lives on its own JPEG buffer so connected
        # clients of /bev_stream can paginate at a different rate than
        # /stream (e.g. if the HUD process throttles).
        self._bev_frame_bus = LatestFrameBus[bytes]()
        self._latest_presented_frame: PresentedFrame | None = None
        # Scene options surfaced to the browser dropdown via /scenes.
        # Each entry is a dict with ``label``, ``path``, ``variants``;
        # the demo wrapper builds these from its scene-discovery layer
        # and passes them in. Empty tuple = no dropdown.
        self._scenes: tuple[dict[str, object], ...] = tuple(scenes)
        # Pre-encoded JPEG thumbnails keyed by scene path. The demo
        # wrapper takes :class:`SceneOption.thumbnail` (a PIL ``Image``)
        # and JPEG-encodes once at startup -- per-tile encoding under
        # the HTTP handler thread would compete with the main camera's
        # encode budget for no good reason. Keys must match the
        # ``path`` strings posted in :attr:`_scenes` so the
        # ``/thumbnail`` endpoint can resolve them by an exact string
        # compare instead of building a separate id->path map.
        self._thumbnails: dict[str, bytes] = dict(thumbnails or {})
        # Scene-change request channel mirroring the slangpy HUD's
        # ``pending_scene_change`` flag. ``should_close`` returns True
        # when this is non-None so the runtime loop unwinds; the demo
        # wrapper then calls ``acknowledge_scene_change`` and re-enters
        # the long-lived engine with the new scene (model stays resident).
        self._pending_scene_change: tuple[Path, str] | None = None
        # Pre-cached idle overlay frames keyed by message. Lazily filled on
        # the first call to :meth:`_publish_idle_frame`. Cached so the
        # heartbeat republish in ``wait_for_scene_selection`` doesn't redo
        # the PIL text render every 2 s; keyed by message so the "Loading
        # world model..." (warmup) and "Select a scene to begin driving"
        # (ready) variants are each rendered at most once.
        self._idle_frame_cache_by_message: dict[str, np.ndarray] = {}
        # Model-warmup status, wired by the demo via :meth:`set_model_status`
        # (mirrors the slangpy HUD). Defaults inert so the idle overlay
        # reads "Select a scene to begin driving" if never wired.
        self._model_can_prewarm = False
        self._model_ready_probe: Callable[[], bool] = lambda: True
        # Scene-selection lock (wired by the demo with --preload-scenes).
        # While the probe returns True, /scene/select is rejected and the
        # idle frame reads "Preloading scenes..." so the browser can't pick
        # a scene until every scene is cached.
        self._scene_selection_locked_probe: Callable[[], bool] = lambda: False
        # Keyboard drive integrator. Late-imported because ``demo``
        # imports the streaming presenter via the CLI's presenter
        # factory; a top-level import would be circular. The integrator
        # owns the same ``set_drive`` -> ``KeyboardState`` plumbing the
        # slangpy HUD uses, keeping browser and desktop controls aligned.
        from crazy_robotaxi.cli import KeyboardDriveState

        self._keyboard_drive_factory = KeyboardDriveState
        self._keyboard_drive = KeyboardDriveState(_KeyboardDriveSink(keyboard))

        try:
            self._server = ThreadingHTTPServer(
                (bind_host, bind_port), _make_handler(self)
            )
        except OSError as exc:
            _print_port_conflict_help(bind_host, bind_port, exc)
            raise
        # ``daemon=True`` means the server thread won't block interpreter
        # exit if the main thread raises; ``close()`` still shuts it down
        # cleanly on the normal path.
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="interactive_drive-mjpeg",
            daemon=True,
        )
        self._server_thread.start()
        # ThreadingHTTPServer.server_address is typed as
        # ``_AfInetAddress | _AfInet6Address`` in stdlib stubs -- a 2-tuple
        # for IPv4 and a 4-tuple for IPv6. Index into it instead of
        # unpacking so pyright is happy on both variants.
        actual_host = self._server.server_address[0]
        actual_port = self._server.server_address[1]
        gate_note = (
            " -- token gate ACTIVE, append ?token=... to the URL"
            if self._stream_token is not None
            else ""
        )
        logger.info(
            f"[presenter] MJPEG stream listening on http://{actual_host}:{actual_port}/ "
            f"(open that URL in a browser on the same network){gate_note}",
        )

    @property
    def should_close(self) -> bool:
        # There's no window to close. The app loop runs until the
        # simulation thread finishes, the user Ctrl-C's the process,
        # or a /scene/select request flips the pending-change channel
        # so the demo wrapper can switch the long-lived engine to a new scene.
        return self._stop_event.is_set() or self._pending_scene_change is not None

    @property
    def pending_scene_change(self) -> tuple[Path, str] | None:
        """Scene the browser asked to load next (via ``/scene/select``), or ``None``."""
        return self._pending_scene_change

    def acknowledge_scene_change(self, scene_path: Path, variant: str) -> None:
        """Clear the pending scene change after the demo wrapper has applied it."""
        del scene_path, variant  # accepted for symmetry with the slangpy HUD API
        self._pending_scene_change = None

    def set_model_status(
        self, *, can_prewarm: bool, ready_probe: Callable[[], bool]
    ) -> None:
        """Wire the idle overlay text to model-warmup progress (mirrors the HUD).

        While ``can_prewarm`` and not ``ready_probe()``, the idle frame reads
        "Loading world model..." instead of the "select a scene" prompt.
        """
        self._model_can_prewarm = bool(can_prewarm)
        self._model_ready_probe = ready_probe

    def set_scene_selection_locked(self, probe: Callable[[], bool]) -> None:
        """Reject ``/scene/select`` while ``probe()`` returns True (--preload-scenes).

        Locks scene picking until every scene is cached; idle overlay then
        reads "Preloading scenes...".
        """
        self._scene_selection_locked_probe = probe

    def wait_for_scene_selection(self) -> tuple[Path, str] | None:
        """Block until the browser POSTs a scene selection (or the presenter closes).

        Publishes an idle overlay frame, re-published on a 2 s heartbeat so a
        late-connecting browser still gets the placeholder promptly. Returns
        ``(scene_path, variant)`` on selection, or ``None`` if closed first.
        """
        idle_heartbeat_s = 2.0
        last_publish = 0.0
        while True:
            now = time.monotonic()
            if now - last_publish >= idle_heartbeat_s:
                self._publish_idle_frame()
                last_publish = now
            if self._stop_event.wait(timeout=0.1):
                return None
            if self._pending_scene_change is not None:
                return self._pending_scene_change

    def _publish_idle_frame(self) -> None:
        """Stream the cached idle placeholder frame.

        Overlay text follows warmup / lock state; each variant's PIL render is
        memoised so the heartbeat doesn't re-pay the text-overlay cost.
        """
        if self._model_can_prewarm and not self._model_ready_probe():
            message = "Loading world model..."
        elif self._scene_selection_locked_probe():
            message = "Preloading scenes..."
        else:
            message = "Select a scene to begin driving"
        cached = self._idle_frame_cache_by_message.get(message)
        if cached is None:
            base = np.zeros(
                (self._raster.height, self._raster.width, 3), dtype=np.uint8
            )
            cached = render_loading_overlay(base, message=message)
            self._idle_frame_cache_by_message[message] = cached
        self._publish(cached)

    def bind_keyboard(self, keyboard: KeyboardState) -> None:
        """Re-target the presenter (and rebuild the keyboard-drive integrator) at ``keyboard``."""
        self._keyboard = keyboard
        self._latest_presented_frame = None
        self._keyboard_drive = self._keyboard_drive_factory(
            _KeyboardDriveSink(keyboard)
        )

    def configure_taxi_hud(self, bev: BevConfig) -> None:
        """Configure BEV projection used by browser taxi overlays."""
        from crazy_robotaxi.driving import (
            TaxiKeyboardDriveState,
        )

        self._taxi_enabled = True
        self._bev_config = bev
        self._keyboard_drive_factory = TaxiKeyboardDriveState
        self._keyboard_drive = TaxiKeyboardDriveState(
            _KeyboardDriveSink(self._keyboard)
        )

    def configure_taxi_camera(self, calibration: CameraCalibration) -> None:
        """Configure camera projection for world-anchored taxi markers."""
        self._taxi_camera_calibration = calibration
        self._taxi_camera_models.clear()

    def configure_taxi_enclosure(self, segments_world: np.ndarray) -> None:
        """Configure static Taxi-only closure lines drawn over the browser BEV."""
        segments = np.asarray(segments_world, dtype=np.float32)
        if segments.ndim != 3 or segments.shape[1:] != (2, 3):
            raise ValueError("Taxi enclosure segments must have shape (N, 2, 3).")
        self._taxi_enclosure_segments_world = segments.copy()

    def process_events(self) -> None:
        # Update the integrator at simulation cadence regardless of how often
        # the browser posts /control events.
        self._keyboard_drive.update()

    def trigger_visual_flare(self) -> None:
        """Start the collision-feedback fade."""
        self._visual_flare.trigger()

    def prepare_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        if view_mode == "physx":
            if frame.physx_debug is None:
                return
            _prefetch_to_numpy(
                select_presented_rgb(
                    frame,
                    view_mode,
                    width=self._raster.width,
                    height=self._raster.height,
                )
            )
        elif view_mode == "model_rgb" and frame.model_rgb_host_uint8 is not None:
            _prefetch_to_numpy(frame.model_rgb_host_uint8)
        else:
            _prefetch_to_numpy(frame.rgb_host_uint8)
        if frame.bev_host_uint8 is not None:
            _prefetch_to_numpy(frame.bev_host_uint8)

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        if view_mode == "physx" and frame.physx_debug is None:
            # Preserve the last published JPEG until a PhysX-enabled chunk is
            # ready; publishing frame.rgb_host_uint8 here flashes the HDMap.
            return
        visual_flare = getattr(self, "_visual_flare", None)
        flare_opacity = visual_flare.opacity() if visual_flare is not None else 0.0
        self._latest_presented_frame = frame

        def with_flare(rgb: object) -> np.ndarray:
            return darken_rgb(_as_rgb_host_uint8(rgb), flare_opacity)

        # Mirror SlangPyPresenter.present_frame's view-mode branching so
        # the user's `1`/`2` toggles behave identically.
        if view_mode == "physx":
            image = _with_status_overlay(
                select_presented_rgb(
                    frame,
                    view_mode,
                    width=self._raster.width,
                    height=self._raster.height,
                ),
                frame.status_message,
            )
        elif view_mode == "model_rgb" and frame.model_rgb_host_uint8 is not None:
            image = _with_status_overlay(
                frame.model_rgb_host_uint8, frame.status_message
            )
        else:
            image = _with_status_overlay(frame.rgb_host_uint8, frame.status_message)
        image = self._with_taxi_world_marker(image, frame)
        self._publish(with_flare(image))
        if frame.bev_host_uint8 is not None:
            self._publish_bev(frame.bev_host_uint8)

    def close(self) -> None:
        self._stop_event.set()
        self._frame_bus.close()
        self._bev_frame_bus.close()
        self._server.shutdown()
        self._server.server_close()
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)

    # -- Internals --------------------------------------------------

    def _publish(self, rgb_host_uint8: object) -> None:
        jpeg = encode_rgb_frame_to_jpeg(
            _downscale_rgb(_as_rgb_host_uint8(rgb_host_uint8), self._stream_scale),
            quality=self._jpeg_quality,
            value_range="uint8",
        )
        _publish_if_open(self._frame_bus, jpeg, stop_event=self._stop_event)

    def _with_taxi_world_marker(
        self, rgb_host_uint8: np.ndarray, frame: PresentedFrame
    ) -> np.ndarray:
        """Overlay the in-view world target while leaving off-screen targets alone."""
        snapshot = frame.application_state
        if (
            snapshot is None
            or snapshot.session_state != "playing"
            or frame.rig_to_world is None
            or self._taxi_camera_calibration is None
        ):
            return rgb_host_uint8
        image_height, image_width = rgb_host_uint8.shape[:2]
        model_key = (image_width, image_height)
        camera_model = self._taxi_camera_models.get(model_key)
        if camera_model is None:
            camera_model = FThetaCameraModel(
                self._taxi_camera_calibration,
                output_width=image_width,
                output_height=image_height,
            )
            self._taxi_camera_models[model_key] = camera_model
        markers = project_taxi_markers_to_camera(
            snapshot,
            frame.rig_to_world,
            camera_model,
            image_width=image_width,
            image_height=image_height,
        )
        if not markers:
            return rgb_host_uint8

        image = Image.fromarray(rgb_host_uint8, mode="RGB")
        draw = ImageDraw.Draw(image)
        color = (118, 185, 0) if snapshot.phase == "seeking_pickup" else (200, 150, 50)
        label = "PICKUP" if snapshot.phase == "seeking_pickup" else "DROPOFF"
        for marker in markers:
            self._draw_taxi_marker(draw, marker, color=color, label=label)
        return np.asarray(image)

    @staticmethod
    def _draw_taxi_marker(
        draw: ImageDraw.ImageDraw,
        marker: TaxiCameraMarkerProjection,
        *,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        """Draw one camera-projected pickup or dropoff marker."""
        for edge in marker.ring_edges_uv:
            draw.line(edge, fill=(0, 0, 0), width=7)
            draw.line(edge, fill=color, width=4)

        anchor = marker.anchor_uv
        if marker.beacon_top_uv is None:
            top = (anchor[0], anchor[1] - 64.0)
        else:
            vector_x = marker.beacon_top_uv[0] - anchor[0]
            vector_y = marker.beacon_top_uv[1] - anchor[1]
            length = max(1.0, _math.hypot(vector_x, vector_y))
            display_length = min(170.0, max(52.0, length))
            top = (
                anchor[0] + vector_x * display_length / length,
                anchor[1] + vector_y * display_length / length,
            )
        draw.line((anchor, top), fill=(0, 0, 0), width=9)
        draw.line((anchor, top), fill=color, width=5)
        draw.ellipse(
            (
                anchor[0] - 9,
                anchor[1] - 9,
                anchor[0] + 9,
                anchor[1] + 9,
            ),
            fill=color,
            outline=(255, 255, 255),
            width=3,
        )
        label_box = draw.textbbox((0, 0), label)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        draw.rounded_rectangle(
            (
                top[0] - label_width / 2 - 8,
                top[1] - label_height - 15,
                top[0] + label_width / 2 + 8,
                top[1] + 5,
            ),
            radius=6,
            fill=(8, 8, 12),
            outline=color,
            width=2,
        )
        draw.text(
            (top[0] - label_width / 2, top[1] - label_height - 10),
            label,
            fill=color,
        )

    def _publish_bev(self, bev_rgb_host_uint8: object) -> None:
        """Encode the BEV minimap (quality 95, not 85) and stash it for ``/bev_stream``.

        Higher quality avoids JPEG ringing the HUD's Google-Maps filter would
        otherwise surface as grey halos around lane / vehicle edges.
        """
        jpeg = encode_rgb_frame_to_jpeg(
            _as_rgb_host_uint8(bev_rgb_host_uint8),
            quality=95,
            value_range="uint8",
        )
        _publish_if_open(self._bev_frame_bus, jpeg, stop_event=self._stop_event)

    def _wait_for_new_frame(self, last_seen_count: int) -> tuple[bytes, int] | None:
        """Block until a frame newer than ``last_seen_count`` is ready or
        the server is shutting down. Returns ``(jpeg_bytes, frame_count)``
        on success, ``None`` when closing.
        """
        return _wait_for_bus_frame(
            self._frame_bus,
            last_seen_count=last_seen_count,
            stop_event=self._stop_event,
        )

    def _wait_for_new_bev_frame(self, last_seen_count: int) -> tuple[bytes, int] | None:
        """Same as :meth:`_wait_for_new_frame` but for the BEV stream.

        Returns ``None`` when the server is closing.
        """
        return _wait_for_bus_frame(
            self._bev_frame_bus,
            last_seen_count=last_seen_count,
            stop_event=self._stop_event,
        )

    def _apply_control(self, key: str, down: bool) -> None:
        # Direction keys (W/A/S/D + arrows + Space) flow through the
        # ``KeyboardDriveState`` integrator so the MJPEG path posts the
        # exact same ``DriverCommand(manual_control=True, ...)`` shape
        # the slangpy HUD does.
        drive_keysym = _BROWSER_KEY_TO_DRIVE_KEYSYM.get(key)
        if drive_keysym is not None and self._keyboard_drive.set_key(
            drive_keysym, down
        ):
            return
        if down:
            view_mode = _BROWSER_KEY_TO_VIEW_MODE.get(key)
            if view_mode is not None:
                self._keyboard.set_view_mode(view_mode)
                return
            # ``r`` / ``R`` restarts the rollout. Only fire on keydown so
            # holding the key doesn't trigger a cascade of resets.
            if key in ("r", "R"):
                self._keyboard.request_reset()
                return
            # Composition-root key extensions; single characters were
            # normalized to lowercase at registration, so fold case here to
            # keep Shift-modified presses bound to the same handler.
            handler = self._extra_key_handlers.get(
                key.lower() if len(key) == 1 else key
            )
            if handler is not None:
                handler()

    def _state_snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable vehicle and taxi telemetry snapshot.

        Returns ``None`` fields before the first chunk so the browser shows
        ``--`` instead of a stale zero.
        """
        if not self._taxi_enabled:
            snapshot = self._keyboard.vehicle_state
            if snapshot is None:
                return {
                    "speed_mps": None,
                    "steer_rad": None,
                    "yaw_rad": None,
                }
            return {
                "speed_mps": float(snapshot.speed_mps),
                "steer_rad": float(snapshot.steer_rad),
                "yaw_rad": float(snapshot.yaw_rad),
            }
        frame = self._latest_presented_frame
        vehicle_state = None if frame is None else frame.vehicle_state
        taxi_state = None if frame is None else frame.application_state
        live_taxi_state = self._keyboard.taxi_game_state
        if live_taxi_state is not None and live_taxi_state.session_state != "playing":
            taxi_state = live_taxi_state
        result: dict[str, object] = {
            "speed_mps": None,
            "steer_rad": None,
            "yaw_rad": None,
            "taxi": None,
        }
        if vehicle_state is not None:
            result.update(
                speed_mps=float(vehicle_state.speed_mps),
                steer_rad=float(vehicle_state.steer_rad),
                yaw_rad=float(vehicle_state.yaw_rad),
            )
        if taxi_state is not None:
            taxi_payload = taxi_state.as_dict()
            taxi_payload["bev_enabled"] = bool(
                self._bev_config is not None and self._bev_config.enabled
            )
            if vehicle_state is not None and self._bev_config is not None:
                targets = (
                    taxi_state.pickup_targets_xyz_m
                    if taxi_state.phase == "seeking_pickup"
                    and taxi_state.pickup_targets_xyz_m
                    else (taxi_state.target_xyz_m,)
                )
                bev_targets = []
                for target in targets:
                    u, v, visible = project_target_to_bev(
                        target, vehicle_state, self._bev_config
                    )
                    bev_targets.append({"u": u, "v": v, "visible": visible})
                taxi_payload["bev_targets"] = bev_targets
            else:
                taxi_payload["bev_targets"] = []
            bev_enclosure_segments = []
            if (
                frame is not None
                and frame.bev_rig_to_world is not None
                and self._bev_config is not None
            ):
                for segment in getattr(
                    self,
                    "_taxi_enclosure_segments_world",
                    np.empty((0, 2, 3), dtype=np.float32),
                ):
                    projected = project_segment_pose_to_bev(
                        segment, frame.bev_rig_to_world, self._bev_config
                    )
                    if projected is None:
                        continue
                    start, end = projected
                    bev_enclosure_segments.append(
                        {
                            "u0": start[0],
                            "v0": start[1],
                            "u1": end[0],
                            "v1": end[1],
                        }
                    )
            taxi_payload["bev_enclosure_segments"] = bev_enclosure_segments
            result["taxi"] = taxi_payload
        return result

    def _is_request_authorized(
        self, query: dict[str, list[str]], header_token: str | None
    ) -> bool:
        """Check the shared stream token; always True when no token is set.

        Accepts the token either as a ``?token=`` query parameter (what the
        served page appends to its own fetches) or as an ``X-Stream-Token``
        header (curl-friendly, keeps the token out of proxy access logs).
        """
        if self._stream_token is None:
            return True
        candidate = query.get("token", [""])[0] or (header_token or "")
        return hmac.compare_digest(candidate, self._stream_token)

    def _request_scene_change(self, scene_path_str: str, variant: str) -> bool:
        """Validate a ``/scene/select`` request against registered ``_scenes`` and stash it.

        Only registered paths load; an arbitrary ``scene=`` from a stale tab
        is rejected rather than latching a filesystem path into the engine.
        """
        if not scene_path_str:
            return False
        if self._scene_selection_locked_probe():
            # Scenes are still preloading; reject selection so the browser
            # waits for the instant (cached) switch instead of triggering a
            # mid-preload parse.
            return False
        # Match against the registered scenes by string-comparing the
        # path; ``Path("a") == "a"`` is False so we normalize first.
        for entry in self._scenes:
            entry_path = str(entry.get("path", ""))
            if entry_path == scene_path_str:
                entry_variants = entry.get("variants", ()) or ("default",)
                if not isinstance(entry_variants, (list, tuple)):
                    entry_variants = ("default",)
                resolved_variant = (
                    variant if variant in entry_variants else entry_variants[0]
                )
                self._pending_scene_change = (Path(entry_path), str(resolved_variant))
                return True
        return False


# Type alias for ``_serve_mjpeg``'s blocking getter parameter. ``None``
# means the server is shutting down; ``(jpeg, count)`` is a fresh frame.
_WaitForFrame = Callable[[int], tuple[bytes, int] | None]


def _make_handler(presenter: MJPEGStreamingPresenter) -> type[BaseHTTPRequestHandler]:
    """Build a BaseHTTPRequestHandler subclass closed over ``presenter``.

    http.server instantiates handlers per-request with a fixed signature,
    so this factory is the standard way to inject shared state.
    """

    class Handler(BaseHTTPRequestHandler):
        # Keep log lines off stderr during normal operation; they'd
        # interleave badly with the backend's per-chunk timing logs.
        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _check_stream_token(self, parsed_query: str) -> bool:
            """Reject the request with 403 unless the stream token matches."""
            authorized = presenter._is_request_authorized(
                parse_qs(parsed_query), self.headers.get("X-Stream-Token")
            )
            if not authorized:
                self.send_error(HTTPStatus.FORBIDDEN, "Missing or invalid stream token")
            return authorized

        def do_GET(self) -> None:  # noqa: N802 (http.server mandated name)
            parsed = urlparse(self.path)
            if not self._check_stream_token(parsed.query):
                return
            if parsed.path in ("/", "/index.html"):
                self._serve_index()
            elif parsed.path == "/stream":
                self._serve_stream()
            elif parsed.path == "/bev_stream":
                self._serve_bev_stream()
            elif parsed.path == "/state":
                self._serve_state()
            elif parsed.path == "/scenes":
                self._serve_scenes()
            elif parsed.path == "/scene/select":
                self._serve_scene_select(parse_qs(parsed.query))
            elif parsed.path == "/thumbnail":
                self._serve_thumbnail(parse_qs(parsed.query))
            elif parsed.path == "/control":
                self._serve_control(parse_qs(parsed.query))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 (http.server mandated name)
            parsed = urlparse(self.path)
            if not self._check_stream_token(parsed.query):
                return
            if parsed.path == "/taxi/name":
                self._serve_taxi_name()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _serve_index(self) -> None:
            body = _INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # Aggressive no-cache so a browser that still has a
            # pre-scene-picker tab open doesn't keep rendering the old
            # HTML after a server upgrade. The page is tiny (~10 KB) so
            # bypassing the cache on every reload costs nothing.
            self.send_header(
                "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
            )
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(body)

        def _serve_state(self) -> None:
            """Latest telemetry as JSON; polled ~10 Hz by the browser speed readout."""
            body = json.dumps(presenter._state_snapshot()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_scenes(self) -> None:
            """Discovered scenes as JSON ``{label, path, variants, has_thumbnail}`` for the picker.

            ``has_thumbnail`` lets the client skip ``/thumbnail`` for scenes
            with no image instead of relying on ``<img onerror>``.
            """
            scenes_with_thumbs = [
                {
                    **entry,
                    "has_thumbnail": str(entry.get("path", ""))
                    in presenter._thumbnails,
                }
                for entry in presenter._scenes
            ]
            body = json.dumps({"scenes": scenes_with_thumbs}).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_scene_select(self, query: dict[str, list[str]]) -> None:
            """Mark ``?scene=PATH&variant=NAME`` as the next scene to load.

            Validated against ``_scenes``; an omitted/unknown ``variant``
            falls back to the scene's first registered variant.
            """
            scene = query.get("scene", [""])[0]
            variant = query.get("variant", ["default"])[0]
            ok = presenter._request_scene_change(scene, variant)
            if not ok:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _serve_thumbnail(self, query: dict[str, list[str]]) -> None:
            """Return the pre-encoded JPEG thumbnail for ``?scene=PATH`` (404 if none)."""
            scene = query.get("scene", [""])[0]
            data = presenter._thumbnails.get(scene)
            if not data:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            # Long cache: the thumbnail never changes for a given
            # session, and the path string already keys the cache
            # bucket per-scene.
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(data)

        def _serve_stream(self) -> None:
            self._serve_mjpeg(presenter._wait_for_new_frame, endpoint="/stream")

        def _serve_bev_stream(self) -> None:
            self._serve_mjpeg(presenter._wait_for_new_bev_frame, endpoint="/bev_stream")

        def _serve_mjpeg(self, wait_fn: _WaitForFrame, *, endpoint: str) -> None:
            """Generic ``multipart/x-mixed-replace`` writer used by /stream and
            /bev_stream. ``wait_fn(last_seen)`` is the per-stream blocking
            getter that returns ``(jpeg, frame_count)`` or ``None`` on
            shutdown.
            """
            # Cap the kernel send buffer so a slow client (VPN / tunnel)
            # can't queue seconds of frames inside TCP: once the buffer is
            # full ``wfile.write`` blocks, and the next ``wait_fn`` call
            # jumps to the newest bus frame (dropping the stale ones).
            try:
                self.connection.setsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF, _STREAM_SNDBUF_BYTES
                )
            except OSError:
                pass  # non-fatal: stream still works, just with more buffering
            stats = _StreamClientStats(f"{endpoint} {self.client_address[0]}")
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
            )
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={_MULTIPART_BOUNDARY}",
            )
            self.end_headers()
            last_seen = 0
            try:
                # Loop until shutdown (``wait_fn`` returns None only on
                # ``_stop_event``). NOT gated on ``should_close``: that also
                # flips True on a pending scene/variant change, and closing the
                # connection there would freeze the browser's multipart <img>
                # (it never auto-reconnects) mid-switch.
                while True:
                    result = wait_fn(last_seen)
                    if result is None:
                        break
                    jpeg, frame_count = result
                    part = (
                        (
                            f"--{_MULTIPART_BOUNDARY}\r\n"
                            f"Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n"
                        ).encode("ascii")
                        + jpeg
                        + b"\r\n"
                    )
                    self.wfile.write(part)
                    self.wfile.flush()
                    line = stats.record(
                        frame_count=frame_count,
                        last_seen_count=last_seen,
                        nbytes=len(part),
                        now=time.monotonic(),
                    )
                    last_seen = frame_count
                    if line is not None:
                        logger.info(line)
            except (BrokenPipeError, ConnectionResetError):
                # Client disconnected; that's normal, not an error.
                pass
            if stats.sent_frames:
                logger.info(stats.summary(now=time.monotonic()) + " (disconnected)")

        def _serve_control(self, query: dict[str, list[str]]) -> None:
            key = query.get("key", [""])[0]
            down_raw = query.get("down", ["0"])[0]
            try:
                down = bool(int(down_raw))
            except ValueError:
                down = False
            if key:
                presenter._apply_control(key, down)
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _serve_taxi_name(self) -> None:
            """Validate and queue a high-score name from the browser modal."""
            taxi_state = presenter._keyboard.taxi_game_state
            if taxi_state is None or taxi_state.session_state != "awaiting_name":
                self.send_error(HTTPStatus.CONFLICT)
                return
            try:
                content_length = max(
                    0, min(int(self.headers.get("Content-Length", "0")), 1024)
                )
                payload = json.loads(self.rfile.read(content_length))
                name = payload.get("name", "") if isinstance(payload, dict) else ""
            except (TypeError, ValueError, json.JSONDecodeError):
                name = ""
            if not isinstance(name, str) or not presenter._keyboard.submit_taxi_name(
                name
            ):
                body = json.dumps(
                    {
                        "error": (
                            "Name must be 1-12 characters using letters, numbers, "
                            "spaces, hyphens, or underscores."
                        )
                    }
                ).encode("utf-8")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def _as_rgb_host_uint8(frame: object) -> np.ndarray:
    """Materialize a frame to ``(H, W, 3)`` uint8.

    World-model frames are lazy GPU handles (``_LazyRGBFrame``) with
    ``to_numpy()`` but no ``__array_interface__``, so shared media helpers
    need an explicit materialization step. Mirrors the slangpy presenter.
    """
    to_numpy = getattr(frame, "to_numpy", None)
    if callable(to_numpy):
        frame = to_numpy()
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]
    return rgb_frame_to_uint8(array, value_range="uint8")


def _scaled_dims(width: int, height: int, scale: float) -> tuple[int, int]:
    """Target (width, height) for a stream scale factor.

    Rounded to even numbers so downstream video encoders (yuv420p) accept
    captured frames without a resample; clamped to at least 2x2.
    """
    scaled_width = max(2, round(width * scale / 2.0) * 2)
    scaled_height = max(2, round(height * scale / 2.0) * 2)
    return scaled_width, scaled_height


def _downscale_rgb(rgb_host_uint8: np.ndarray, scale: float) -> np.ndarray:
    """Bilinear-downscale an ``(H, W, 3)`` uint8 frame; identity at scale 1.0."""
    if scale >= 1.0:
        return rgb_host_uint8
    height, width = rgb_host_uint8.shape[:2]
    target = _scaled_dims(width, height, scale)
    if target == (width, height):
        return rgb_host_uint8
    image = Image.fromarray(rgb_host_uint8, mode="RGB")
    return np.asarray(image.resize(target, Image.BILINEAR))


def _publish_if_open(
    bus: LatestFrameBus[bytes], jpeg: bytes, *, stop_event: threading.Event
) -> None:
    if stop_event.is_set():
        return
    try:
        bus.publish(jpeg)
    except RuntimeError:
        if not stop_event.is_set():
            raise


def _wait_for_bus_frame(
    bus: LatestFrameBus[bytes],
    *,
    last_seen_count: int,
    stop_event: threading.Event,
) -> tuple[bytes, int] | None:
    while not stop_event.is_set():
        frame = bus.wait_for_frame(last_seen_count=last_seen_count, timeout_s=1.0)
        if frame is not None:
            return frame.payload, frame.count
        if bus.closed:
            return None
    return None


def _prefetch_to_numpy(frame: object) -> None:
    prefetch = getattr(frame, "prefetch_to_numpy", None)
    if callable(prefetch):
        prefetch()


def _with_status_overlay(rgb_host_uint8: object, message: str | None) -> np.ndarray:
    rgb_host_uint8 = _as_rgb_host_uint8(rgb_host_uint8)
    if message is None:
        return rgb_host_uint8
    return render_loading_overlay(rgb_host_uint8, message=message)


def parse_bind(value: str) -> tuple[str, int]:
    """Accept ``HOST:PORT``, bare ``:PORT``, or a bare port (all default to 0.0.0.0).

    Pass an explicit host (e.g. ``127.0.0.1:8080``) to restrict the listener
    to one interface, e.g. behind an SSH tunnel.
    """
    if ":" not in value:
        # Bare port number form (``--stream-mjpeg 8080``); equivalent to ``:8080``.
        host = "0.0.0.0"
        port_str = value
    else:
        host, port_str = value.rsplit(":", 1)
        if not host:
            host = "0.0.0.0"
    try:
        port = int(port_str)
    except ValueError as exc:
        raise ValueError(
            f"--stream-mjpeg port must be an integer, got {port_str!r}"
        ) from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"--stream-mjpeg port out of range: {port}")
    return host, port
