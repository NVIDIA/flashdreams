# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MJPEG-over-HTTP presenter.

An alternative to :class:`interactive_drive.presenter.SlangPyPresenter` for
deployments where no graphics-capable GPU is available (e.g. a DGX
Station with only a GB300 compute card). Frames produced by the backend
are JPEG-encoded on the CPU and served to connected HTTP clients as a
``multipart/x-mixed-replace`` stream. The user's browser posts keydown/
keyup events back to the server so the demo stays interactive.

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
import shutil
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from interactive_drive.config import RasterConfig
from interactive_drive.input.keyboard import KeyboardState
from interactive_drive.loading_overlay import render_loading_overlay
from interactive_drive.types import DriverCommand, PresentedFrame

# Boundary marker embedded in the multipart response. The exact string
# doesn't matter as long as it never appears inside a JPEG payload (they
# start with the JPEG SOI marker 0xFFD8 so ``--interactive_drive`` is always safe).
_MULTIPART_BOUNDARY = "interactive_drive"

# Browser-side keys (``event.key``) to KeyboardState names. Matches the
# set handled by SlangPyPresenter so the URL-triggered keyboard input
# looks exactly like a local window keypress to the simulation thread.
_BROWSER_KEY_TO_NAME: dict[str, str] = {
    "w": "w",
    "W": "w",
    "a": "a",
    "A": "a",
    "s": "s",
    "S": "s",
    "d": "d",
    "D": "d",
    "ArrowUp": "up",
    "ArrowLeft": "left",
    "ArrowDown": "down",
    "ArrowRight": "right",
}

_BROWSER_KEY_TO_VIEW_MODE: dict[str, str] = {
    # 1 = world-model RGB (the generated drive view, the main demo output).
    # 2 = HDMap with traffic (the rasterizer's conditioning input).
    "1": "model_rgb",
    "2": "rgb",
}


# Single HTML page served at ``/``. Shows the MJPEG stream and forwards
# keydown/keyup to ``/control``. Kept inline (not a separate file) so the
# presenter is a single-file drop-in with no template loading to configure.
def _print_port_conflict_help(host: str, port: int, exc: OSError) -> None:
    """Print a helpful message when the HTTP server can't bind to the port."""
    print(
        f"\n[presenter] MJPEG server failed to start: port {port} is already in use.\n"
        f"            ({exc})\n",
        file=sys.stderr,
        flush=True,
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
            print(
                f"[presenter] The following process is blocking port {port}:\n",
                file=sys.stderr,
                flush=True,
            )
            print(result.stdout, file=sys.stderr, flush=True)
            shown = True
    if not shown and shutil.which("lsof"):
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(
                f"[presenter] The following process is blocking port {port}:\n",
                file=sys.stderr,
                flush=True,
            )
            print(result.stdout, file=sys.stderr, flush=True)
            shown = True
    if not shown:
        print(
            f"[presenter] Could not determine which process is using port {port}.\n",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"[presenter] To fix this, either:\n"
        f"  1. Stop the process above, or\n"
        f"  2. Choose a different port: --stream-mjpeg :{port + 1}\n",
        file=sys.stderr,
        flush=True,
    )


_INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>interactive_drive (MJPEG)</title>
<style>
  html, body { margin: 0; padding: 0; background: #111; height: 100%; }
  body { display: flex; align-items: center; justify-content: center; }
  img { max-width: 100%; max-height: 100%; object-fit: contain; image-rendering: pixelated; }
  .hint { position: fixed; top: 8px; left: 8px; color: #aaa; font: 12px sans-serif; }
</style>
</head>
<body>
<img id="stream" src="/stream">
<div class="hint">WASD / Arrows = Drive &middot; 1 = World-Model RGB &middot; 2 = HDMap &middot; R = Reset Rollout</div>
<script>
const DOWN_KEYS = new Set();
function send(key, down) {
  if (down && DOWN_KEYS.has(key)) return;   // debounce: browsers send keydown repeatedly while held
  if (!down) DOWN_KEYS.delete(key); else DOWN_KEYS.add(key);
  fetch('/control?key=' + encodeURIComponent(key) + '&down=' + (down ? 1 : 0))
    .catch(() => {});                       // ignore network hiccups, next event will resync
}
document.addEventListener('keydown', e => send(e.key, true));
document.addEventListener('keyup', e => send(e.key, false));
// When the page loses focus we must release all keys so the car doesn't keep steering.
window.addEventListener('blur', () => { DOWN_KEYS.forEach(k => send(k, false)); });
</script>
</body>
</html>
"""


class MJPEGStreamingPresenter:
    """Drop-in replacement for :class:`SlangPyPresenter` that streams frames
    over HTTP instead of opening a Vulkan swapchain window.

    Exposes the same duck-typed interface consumed by
    :class:`interactive_drive.app.InteractiveDriveApp`: ``should_close`` /
    ``process_events`` / ``present_frame`` / ``present_loading`` /
    ``close``. The simulation thread doesn't know the presenter changed.
    """

    def __init__(
        self,
        raster: RasterConfig,
        keyboard: KeyboardState,
        bind_host: str,
        bind_port: int,
        *,
        jpeg_quality: int = 85,
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
        # Cache for the rendered "Loading world model..." overlay. The
        # scene's initial_rgb is a fixed numpy buffer for the session so
        # a single id() check is enough to avoid redoing the PIL draw
        # on every warmup tick (~30x/s for ~60s of warmup).
        self._loading_overlay_cache: np.ndarray | None = None
        self._loading_overlay_source_id: int | None = None

        try:
            self._server = ThreadingHTTPServer((bind_host, bind_port), _make_handler(self))
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
        print(
            f"[presenter] MJPEG stream listening on http://{actual_host}:{actual_port}/ "
            f"(open that URL in a browser on the same network)",
            flush=True,
        )

    @property
    def should_close(self) -> bool:
        # There's no window to close. The app loop runs until the
        # simulation thread finishes or the user Ctrl-C's the process.
        return self._stop_event.is_set()

    def process_events(self) -> None:
        # Input arrives asynchronously via ``/control`` HTTP requests, so
        # the main loop has nothing to poll here.
        return

    def present_loading(self, rgb_host_uint8: np.ndarray) -> None:
        cached = self._loading_overlay_cache
        if cached is None or id(rgb_host_uint8) != self._loading_overlay_source_id:
            cached = render_loading_overlay(rgb_host_uint8)
            self._loading_overlay_cache = cached
            self._loading_overlay_source_id = id(rgb_host_uint8)
        self._publish(cached)

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        # Mirror SlangPyPresenter.present_frame's view-mode branching so
        # the user's `1`/`2` toggles behave identically.
        if view_mode == "model_rgb" and frame.model_rgb_host_uint8 is not None:
            self._publish(_with_status_overlay(frame.model_rgb_host_uint8, frame.status_message))
        else:
            self._publish(_with_status_overlay(frame.rgb_host_uint8, frame.status_message))
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

    def _publish(self, rgb_host_uint8: np.ndarray) -> None:
        buf = io.BytesIO()
        Image.fromarray(rgb_host_uint8).save(buf, format="JPEG", quality=self._jpeg_quality)
        jpeg = buf.getvalue()
        with self._frame_cond:
            self._latest_jpeg = jpeg
            self._frame_count += 1
            self._frame_cond.notify_all()

    def _publish_bev(self, bev_rgb_host_uint8: np.ndarray) -> None:
        """Encode the BEV minimap and stash it for ``/bev_stream`` waiters.

        BEV frames are tiny (<= 384x384) so JPEG encode is sub-millisecond
        and we boost quality to 95 vs 85 for the main stream. The HUD's
        Google-Maps post-process is sensitive to JPEG ringing around the
        high-contrast lane / vehicle edges (dim ringing pixels survive as
        dirty grey halos), so paying ~12 KB / frame of bandwidth to keep
        edges clean is a good trade.
        """
        buf = io.BytesIO()
        Image.fromarray(bev_rgb_host_uint8).save(buf, format="JPEG", quality=95)
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
            while self._latest_bev_jpeg is None or self._bev_frame_count <= last_seen_count:
                if self._stop_event.is_set():
                    return None
                self._frame_cond.wait(timeout=1.0)
            return self._latest_bev_jpeg, self._bev_frame_count

    def _apply_control(self, key: str, down: bool) -> None:
        name = _BROWSER_KEY_TO_NAME.get(key)
        if name is not None:
            self._keyboard.set_key(name, down)
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


def _make_handler(presenter: MJPEGStreamingPresenter):
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
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_stream(self) -> None:
            self._serve_mjpeg(presenter._wait_for_new_frame)

        def _serve_bev_stream(self) -> None:
            self._serve_mjpeg(presenter._wait_for_new_bev_frame)

        def _serve_mjpeg(self, wait_fn) -> None:
            """Generic ``multipart/x-mixed-replace`` writer used by /stream and
            /bev_stream. ``wait_fn(last_seen)`` is the per-stream blocking
            getter that returns ``(jpeg, frame_count)`` or ``None`` on
            shutdown.
            """
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={_MULTIPART_BOUNDARY}",
            )
            self.end_headers()
            last_seen = 0
            try:
                while not presenter.should_close:
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


def _with_status_overlay(rgb_host_uint8: np.ndarray, message: str | None) -> np.ndarray:
    if message is None:
        return rgb_host_uint8
    return render_loading_overlay(rgb_host_uint8, message=message)
