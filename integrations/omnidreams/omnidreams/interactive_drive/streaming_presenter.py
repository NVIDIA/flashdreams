# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MJPEG-over-HTTP presenter.

An alternative to :class:`omnidreams.interactive_drive.presenter.SlangPyPresenter`
for deployments where no graphics-capable GPU is available (e.g. a DGX
Station with only a GB300 compute card). Frames produced by the backend
are JPEG-encoded on the CPU and served to connected HTTP clients as a
``multipart/x-mixed-replace`` stream. The user's browser posts keydown/
keyup events back to the server so the demo stays interactive.

For a full WebRTC viewer with a polished frontend and lower latency,
use ``omnidreams.webrtc.server`` instead. This presenter is the
single-process, dependency-free fallback for headless / compute-only
hosts where the WebRTC server isn't a fit.

Expected end-to-end latency on the same LAN:

  * JPEG encode (PIL / libjpeg-turbo, 704x1280 @ quality 85): 10-15 ms
  * TCP transmit: <5 ms on 1 Gbps LAN
  * Browser decode + <img> swap: 15-30 ms

so the *streaming* latency is roughly 50 ms. Keypress-to-visible-effect
latency is dominated by the backend's per-chunk wall-clock time (~900 ms
steady-state), which is the same problem we have with the local Vulkan
presenter; streaming doesn't add more than ~50 ms on top of it.
"""

from __future__ import annotations

import io
import json
import shutil
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
from omnidreams.interactive_drive.config import RasterConfig
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.loading_overlay import render_loading_overlay
from omnidreams.interactive_drive.types import DriverCommand, PresentedFrame
from PIL import Image

# Boundary marker embedded in the multipart response. The exact string
# doesn't matter as long as it never appears inside a JPEG payload (they
# start with the JPEG SOI marker 0xFFD8 so ``--interactive_drive`` is always safe).
_MULTIPART_BOUNDARY = "interactive_drive"

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
    "1": "model_rgb",
    "2": "rgb",
}


class _KeyboardDriveSink:
    """In-process duck-typed ``ControlClient`` that writes to ``KeyboardState``.

    The slangpy HUD ships its own ``KeyboardStateDriveSink`` in
    :mod:`slangpy_hud_presenter`, but importing that module would pull
    SlangPy / Vulkan into the streaming-presenter import graph -- the
    very thing the streaming presenter exists to avoid (it's the
    fallback for compute-only hosts where SlangPy can't initialise a
    Vulkan device). We replicate the same minimal surface here so the
    MJPEG keyboard path produces the byte-identical
    ``DriverCommand(manual_control=True, steer_is_direct=True, ...)``
    the HUD does, without dragging the graphics stack in.
    """

    def __init__(self, keyboard: KeyboardState) -> None:
        self._keyboard = keyboard

    def set_drive(self, *, steer: float, throttle: float, brake: float) -> None:
        self._keyboard.set_drive_command(
            DriverCommand(
                throttle=max(0.0, min(1.0, throttle)),
                brake=max(0.0, min(1.0, brake)),
                steer=max(-1.0, min(1.0, steer)),
                steer_is_direct=True,
                manual_control=True,
            )
        )

    def release_all(self) -> None:
        self._keyboard.set_drive_command(None)

    # The methods below exist so anything wired against the legacy
    # supervisor-era ``ControlClient`` surface fails silently rather
    # than raising ``AttributeError`` -- they're no-ops in-process
    # because the streaming presenter writes directly via
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


# Single HTML page served at ``/``. Shows the MJPEG stream and forwards
# keydown/keyup to ``/control``. Kept inline (not a separate file) so the
# presenter is a single-file drop-in with no template loading to configure.
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
<img id="stream" src="/stream">
<div class="hint">WASD / Arrows = Drive &middot; 1 = World-Model RGB &middot; 2 = HDMap &middot; R = Reset Rollout</div>
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
  fetch('/control?key=' + encodeURIComponent(key) + '&down=' + (down ? 1 : 0))
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
// When the page loses focus we must release all keys so the car doesn't keep steering.
window.addEventListener('blur', () => {
  DOWN_KEYS.forEach(k => send(k, false));
  PRESSED_INDICATORS.clear();
  ["w","a","s","d"].forEach(paintIndicator);
});
// Speed readout polling. 100 ms keeps the digit lively without
// generating meaningful HTTP load (~10 req/s of <100 B each).
const speedEl = document.getElementById('speed');
const speedValueEl = document.getElementById('speed-value');
const MPS_TO_MPH = 2.236936;
async function pollState() {
  try {
    const r = await fetch('/state', { cache: 'no-store' });
    if (!r.ok) throw new Error('http ' + r.status);
    const s = await r.json();
    if (typeof s.speed_mps === 'number') {
      speedValueEl.textContent = Math.round(s.speed_mps * MPS_TO_MPH).toString();
      speedEl.classList.remove('disconnected');
    } else {
      speedValueEl.textContent = '--';
      speedEl.classList.add('disconnected');
    }
  } catch {
    speedValueEl.textContent = '--';
    speedEl.classList.add('disconnected');
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
    const r = await fetch('/scenes', { cache: 'no-store' });
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
        img.src = '/thumbnail?scene=' + encodeURIComponent(s.path);
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
    await fetch(url, { method: 'GET', cache: 'no-store' });
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
    :class:`omnidreams.interactive_drive.app.InteractiveDriveApp`:
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
        scenes: tuple[dict[str, object], ...] = (),
        thumbnails: dict[str, bytes] | None = None,
    ) -> None:
        self._raster = raster
        self._keyboard = keyboard
        self._jpeg_quality = int(jpeg_quality)
        self._stop_event = threading.Event()
        # Guarded by ``_frame_cond`` so a sending thread can ``wait()``
        # for the next frame rather than spinning.
        self._latest_jpeg: bytes | None = None
        self._frame_count = 0
        self._frame_cond = threading.Condition()
        # BEV minimap stream lives on its own JPEG buffer so connected
        # clients of /bev_stream can paginate at a different rate than
        # /stream (e.g. if the HUD process throttles). We reuse the same
        # condition variable as the main stream because frames are only
        # published when ``present_frame`` runs anyway, so notifications
        # to either waiter are always safe.
        self._latest_bev_jpeg: bytes | None = None
        self._bev_frame_count = 0
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
        # slangpy HUD uses, which is what gives us the alpasim-style
        # ~10 mph auto-crawl on key release: the integrator posts
        # ``DriverCommand(manual_control=True, throttle=0, brake=0)``
        # which routes through ``integrate_vehicle``'s manual branch
        # where the creep-toward-4.47-m/s logic lives.
        from omnidreams.interactive_drive.demo import KeyboardDriveState

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
        logger.info(
            f"[presenter] MJPEG stream listening on http://{actual_host}:{actual_port}/ "
            f"(open that URL in a browser on the same network)",
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
        """Scene the browser asked to load next, or ``None`` if no change is pending.

        Set by the ``/scene/select`` endpoint and cleared by
        :meth:`acknowledge_scene_change`. The demo wrapper polls this
        between scenes to drive the scene-change loop without tearing
        down the HTTP server / browser session.
        """
        return self._pending_scene_change

    def acknowledge_scene_change(self, scene_path: Path, variant: str) -> None:
        """Clear the pending scene change after the demo wrapper has applied it."""
        del scene_path, variant  # accepted for symmetry with the slangpy HUD API
        self._pending_scene_change = None

    def set_model_status(
        self, *, can_prewarm: bool, ready_probe: Callable[[], bool]
    ) -> None:
        """Wire the idle overlay text to model-warmup progress.

        Mirrors :meth:`SlangPyHudPresenter.set_model_status`. When
        ``can_prewarm`` is True the idle "select a scene" frame published
        during :meth:`wait_for_scene_selection` reads "Loading world
        model..." until ``ready_probe`` returns True, then falls back to
        the normal "Select a scene to begin driving" prompt.
        """
        self._model_can_prewarm = bool(can_prewarm)
        self._model_ready_probe = ready_probe

    def set_scene_selection_locked(self, probe: Callable[[], bool]) -> None:
        """Gate ``/scene/select`` while ``probe()`` returns True.

        Mirrors :meth:`SlangPyHudPresenter.set_scene_selection_locked`: used
        with --preload-scenes so the browser can't pick a scene until every
        scene has preloaded. While locked the idle overlay reads "Preloading
        scenes..." and select requests are rejected.
        """
        self._scene_selection_locked_probe = probe

    def wait_for_scene_selection(self) -> tuple[Path, str] | None:
        """Block until the browser POSTs a scene selection.

        Used by the demo wrapper at startup so we don't burn world-model
        warmup on whatever ``args.scene`` defaulted to before the user
        has actually picked something. Publishes an idle "Select a
        scene to begin" overlay frame so connected browsers have
        something to display while the wait spins; then polls
        :attr:`_pending_scene_change` until either the browser triggers
        ``/scene/select`` or :meth:`close` flips the stop event.

        Re-publishes the idle frame on a slow heartbeat (every 2 s) so
        a browser that connects after ``wait_for_scene_selection``
        first fired -- e.g. the user navigates to the demo URL after
        the server has already started waiting -- gets the placeholder
        promptly via the standard MJPEG ``frame_count`` increment path
        instead of having to wait for an unrelated frame.

        Returns ``(scene_path, variant)`` once the user picks, or
        ``None`` when the presenter is closed first (Ctrl-C in the
        terminal where the demo is running). Mirrors
        :meth:`SlangPyHudPresenter.wait_for_scene_selection`'s contract
        so the demo wrapper's flow is identical across HUD / MJPEG.
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
        """Stream the cached black placeholder frame.

        While the model is still pre-warming (see :meth:`set_model_status`)
        the overlay reads "Loading world model..."; otherwise it reads
        "Select a scene to begin driving". Each variant's PIL render is
        memoised so the heartbeat in :meth:`wait_for_scene_selection`
        doesn't pay the text-overlay cost on every tick. Each publish call
        still bumps :attr:`_frame_count` so connected MJPEG handlers wake
        up and push the frame out to the browser, which is what actually
        matters for late-arriving clients.
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
        """Re-target the presenter at ``keyboard``.

        :class:`InteractiveDriveApp` owns one long-lived ``KeyboardState``
        and binds the injected presenter to it at construction. The
        keyboard-drive integrator captures the keyboard reference
        internally, so it gets rebuilt here.
        """
        self._keyboard = keyboard
        self._keyboard_drive = self._keyboard_drive_factory(
            _KeyboardDriveSink(keyboard)
        )

    def process_events(self) -> None:
        # Input arrives asynchronously via ``/control`` HTTP requests, but
        # the keyboard-drive integrator owes a per-tick update so its
        # auto-crawl smoothing (steer rate, throttle taper, creep target)
        # advances at sim cadence regardless of how often the browser
        # sends events. Mirrors the slangpy HUD's per-frame
        # ``_keyboard_drive.update()`` call.
        self._keyboard_drive.update()

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        # Mirror SlangPyPresenter.present_frame's view-mode branching so
        # the user's `1`/`2` toggles behave identically.
        if view_mode == "model_rgb" and frame.model_rgb_host_uint8 is not None:
            self._publish(
                _with_status_overlay(frame.model_rgb_host_uint8, frame.status_message)
            )
        else:
            self._publish(
                _with_status_overlay(frame.rgb_host_uint8, frame.status_message)
            )
        if frame.bev_host_uint8 is not None:
            self._publish_bev(frame.bev_host_uint8)

    def close(self) -> None:
        self._stop_event.set()
        # Wake any /stream handlers blocked in ``_frame_cond.wait`` so
        # they observe ``should_close`` and exit their per-connection loop.
        with self._frame_cond:
            self._frame_cond.notify_all()
        self._server.shutdown()
        self._server.server_close()
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)

    # -- Internals --------------------------------------------------

    def _publish(self, rgb_host_uint8: object) -> None:
        buf = io.BytesIO()
        Image.fromarray(_as_rgb_host_uint8(rgb_host_uint8)).save(
            buf, format="JPEG", quality=self._jpeg_quality
        )
        jpeg = buf.getvalue()
        with self._frame_cond:
            self._latest_jpeg = jpeg
            self._frame_count += 1
            self._frame_cond.notify_all()

    def _publish_bev(self, bev_rgb_host_uint8: object) -> None:
        """Encode the BEV minimap and stash it for ``/bev_stream`` waiters.

        BEV frames are tiny (<= 384x384) so JPEG encode is sub-millisecond
        and we boost quality to 95 vs 85 for the main stream. The HUD's
        Google-Maps post-process is sensitive to JPEG ringing around the
        high-contrast lane / vehicle edges (dim ringing pixels survive as
        dirty grey halos), so paying ~12 KB / frame of bandwidth to keep
        edges clean is a good trade.
        """
        buf = io.BytesIO()
        Image.fromarray(_as_rgb_host_uint8(bev_rgb_host_uint8)).save(
            buf, format="JPEG", quality=95
        )
        jpeg = buf.getvalue()
        with self._frame_cond:
            self._latest_bev_jpeg = jpeg
            self._bev_frame_count += 1
            self._frame_cond.notify_all()

    def _wait_for_new_frame(self, last_seen_count: int) -> tuple[bytes, int] | None:
        """Block until a frame newer than ``last_seen_count`` is ready or
        the server is shutting down. Returns ``(jpeg_bytes, frame_count)``
        on success, ``None`` when closing.
        """
        with self._frame_cond:
            while self._latest_jpeg is None or self._frame_count <= last_seen_count:
                if self._stop_event.is_set():
                    return None
                self._frame_cond.wait(timeout=1.0)
            return self._latest_jpeg, self._frame_count

    def _wait_for_new_bev_frame(self, last_seen_count: int) -> tuple[bytes, int] | None:
        """Same as :meth:`_wait_for_new_frame` but for the BEV stream.

        Returns ``None`` when the server is closing. Sharing the condition
        variable means the waiter wakes immediately on every published
        frame; the loop body then re-checks the BEV-specific counter.
        """
        with self._frame_cond:
            while (
                self._latest_bev_jpeg is None
                or self._bev_frame_count <= last_seen_count
            ):
                if self._stop_event.is_set():
                    return None
                self._frame_cond.wait(timeout=1.0)
            return self._latest_bev_jpeg, self._bev_frame_count

    def _apply_control(self, key: str, down: bool) -> None:
        # Direction keys (W/A/S/D + arrows + Space) flow through the
        # ``KeyboardDriveState`` integrator so the MJPEG path posts the
        # exact same ``DriverCommand(manual_control=True, ...)`` shape
        # the slangpy HUD does -- which is what unlocks the integrator's
        # ~10 mph creep-toward-target on key release.
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

    def _apply_drive_control(
        self,
        *,
        throttle: float,
        brake: float,
        steer: float,
        reverse: bool = False,
    ) -> None:
        self._keyboard.set_drive_command(
            DriverCommand(
                throttle=max(0.0, min(1.0, throttle)),
                brake=max(0.0, min(1.0, brake)),
                steer=max(-1.0, min(1.0, steer)),
                reverse=reverse,
                steer_is_direct=True,
                manual_control=True,
            )
        )

    def _state_snapshot(self) -> dict[str, float | None]:
        """Return a JSON-serialisable snapshot of the current sim telemetry.

        Reads from :attr:`KeyboardState.vehicle_state`, which the runtime
        loop refreshes once per chunk via
        :func:`omnidreams.interactive_drive.runtime.loop.push_telemetry`.
        Before the simulation has produced its first chunk (warmup
        window), ``vehicle_state`` is ``None`` and we return ``None``s
        so the browser shows ``--`` instead of a stale zero.
        """
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

    def _request_scene_change(self, scene_path_str: str, variant: str) -> bool:
        """Validate and stash a ``/scene/select`` request.

        The validation gate is paranoid on purpose: the only paths that
        can be loaded are ones the demo wrapper explicitly registered
        in :attr:`_scenes`. A stale browser tab that POSTs an arbitrary
        ``scene=`` path gets a 400 instead of having its filesystem
        path latch into the next ``app.run`` iteration.
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
                # Wake any handlers waiting on the frame condition so
                # they observe ``should_close`` flipping and exit their
                # per-connection loop promptly. Not strictly required
                # for correctness (the existing 1 s timeout would
                # eventually retry) but it makes the scene transition
                # feel snappier.
                self._pending_scene_change = (Path(entry_path), str(resolved_variant))
                with self._frame_cond:
                    self._frame_cond.notify_all()
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

        def do_GET(self) -> None:  # noqa: N802 (http.server mandated name)
            parsed = urlparse(self.path)
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
            elif parsed.path == "/drive":
                self._serve_drive(parse_qs(parsed.query))
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
            """Return the latest simulation telemetry as JSON.

            Polled by the browser ~10 Hz to drive the speed readout. The
            payload is intentionally tiny so the polling cost is negligible
            even on slow links; if a richer dashboard wants more state the
            ``KeyboardState`` snapshot is the authoritative source and we
            can extend this without touching the producer side.
            """
            body = json.dumps(presenter._state_snapshot()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_scenes(self) -> None:
            """Return the discovered scene options as JSON for the browser dropdown.

            Each entry is ``{label, path, variants, has_thumbnail}``;
            the browser renders them as clickable cards in the
            bottom-right scene picker. ``has_thumbnail`` lets the
            client decide whether to issue a ``/thumbnail`` request --
            cleaner than relying on ``<img onerror>`` and avoids the
            broken-image flash for scenes that ship no first-image
            asset.
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

            Sets the presenter's ``pending_scene_change`` channel; the
            demo wrapper picks it up between scenes and switches the
            long-lived engine to the new scene. Validates the
            requested scene against the presenter's ``_scenes`` list so a
            stale browser tab can't smuggle in an arbitrary path. The
            ``variant`` query parameter is optional: when omitted (or
            unrecognised) the scene's first registered variant is used,
            which is what the streaming UI relies on now that it has
            no variant selector.
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
            """Return the JPEG-encoded thumbnail for ``?scene=PATH``.

            The thumbnails were JPEG-encoded once at startup in the
            demo wrapper (no per-request encode cost). Sends a 404
            when the scene has no thumbnail or wasn't registered, so
            the browser's ``onerror`` hook can hide the ``<img>``
            element cleanly.
            """
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
            self._serve_mjpeg(presenter._wait_for_new_frame)

        def _serve_bev_stream(self) -> None:
            self._serve_mjpeg(presenter._wait_for_new_bev_frame)

        def _serve_mjpeg(self, wait_fn: _WaitForFrame) -> None:
            """Generic ``multipart/x-mixed-replace`` writer used by /stream and
            /bev_stream. ``wait_fn(last_seen)`` is the per-stream blocking
            getter that returns ``(jpeg, frame_count)`` or ``None`` on
            shutdown.
            """
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
                    jpeg, last_seen = result
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
            except (BrokenPipeError, ConnectionResetError):
                # Client disconnected; that's normal, not an error.
                return

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

        def _serve_drive(self, query: dict[str, list[str]]) -> None:
            presenter._apply_drive_control(
                throttle=_query_float(query, "throttle"),
                brake=_query_float(query, "brake"),
                steer=_query_float(query, "steer"),
                reverse=bool(int(query.get("reverse", ["0"])[0])),
            )
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def _query_float(query: dict[str, list[str]], name: str) -> float:
    try:
        return float(query.get(name, ["0"])[0])
    except ValueError:
        return 0.0


def _as_rgb_host_uint8(frame: object) -> np.ndarray:
    """Materialize a frame to ``(H, W, 3)`` uint8.

    World-model frames are lazy GPU handles (``_LazyRGBFrame``) with
    ``to_numpy()`` but no ``__array_interface__``, so ``Image.fromarray`` can't
    take them directly. Mirrors the slangpy presenter.
    """
    to_numpy = getattr(frame, "to_numpy", None)
    if callable(to_numpy):
        frame = to_numpy()
    return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[..., :3])


def _with_status_overlay(rgb_host_uint8: object, message: str | None) -> np.ndarray:
    rgb_host_uint8 = _as_rgb_host_uint8(rgb_host_uint8)
    if message is None:
        return rgb_host_uint8
    return render_loading_overlay(rgb_host_uint8, message=message)


def parse_bind(value: str) -> tuple[str, int]:
    """Accept ``HOST:PORT``, bare ``:PORT``, or a bare port number.

    All three forms bind on ``0.0.0.0`` (all interfaces) by default;
    pass an explicit host (e.g. ``127.0.0.1:8080``) to restrict the
    listener to a single interface, which is the right choice when
    you're terminating the connection through an SSH tunnel and don't
    want the port reachable directly from the network.
    """
    if ":" not in value:
        # Bare port number form (``--stream-mjpeg 8080``). Friendly
        # shortcut for the common all-interfaces case; equivalent to
        # ``:8080``.
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
