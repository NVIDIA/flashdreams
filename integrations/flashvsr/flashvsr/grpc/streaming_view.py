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

"""HTTP MJPEG viewer support for the FlashVSR gRPC server."""

import io
import queue
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from loguru import logger

from flashdreams.serving.realtime.media import encode_rgb_frame_to_jpeg

DEFAULT_VIEWER_CHUNK_QUEUE_DEPTH = 8
DEFAULT_VIEWER_JPEG_QUALITY = 90
DEFAULT_VIEWER_JPEG_BACKEND = "auto"
DEFAULT_VIEWER_MAX_FPS = 60.0
DEFAULT_VIEWER_FRAME_STRIDE = 1

_VIEWER_STOP = object()
_TORCHVISION_ENCODE_JPEG = None


@dataclass
class _ViewerPlaybackChunk:
    elapsed_ms: float
    frames: np.ndarray | None = None
    jpegs: list[bytes] | None = None


def _load_pillow_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "JPEG support requires Pillow. Install the flashvsr integration "
            "dependencies before using JPEG input or the browser viewer."
        ) from exc

    return Image


def _encode_jpeg_rgb(frame: np.ndarray, quality: int) -> bytes:
    """Encode one uint8 RGB frame to JPEG bytes."""
    try:
        return encode_rgb_frame_to_jpeg(frame, quality=quality, value_range="uint8")
    except ModuleNotFoundError as exc:
        if exc.name not in ("PIL", "PIL.Image"):
            raise
        raise RuntimeError(
            "JPEG support requires Pillow. Install the flashvsr integration "
            "dependencies before using JPEG input or the browser viewer."
        ) from exc


def _load_torchvision_encode_jpeg():
    global _TORCHVISION_ENCODE_JPEG
    if _TORCHVISION_ENCODE_JPEG is None:
        from torchvision.io import encode_jpeg

        _TORCHVISION_ENCODE_JPEG = encode_jpeg
    return _TORCHVISION_ENCODE_JPEG


def encode_jpeg_cuda_tensor(
    out: torch.Tensor, *, quality: int, frame_stride: int = 1
) -> list[bytes]:
    """Encode [1, 3, T, H, W] CUDA RGB tensor in [-1, 1] to JPEG bytes."""
    if not out.is_cuda:
        raise ValueError("CUDA JPEG encode requires a CUDA tensor")
    encode_jpeg = _load_torchvision_encode_jpeg()
    if frame_stride > 1:
        out = out[:, :, ::frame_stride]

    frames = ((out[0].float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    frames = frames.permute(1, 0, 2, 3).contiguous()  # [T, 3, H, W]
    encoded = encode_jpeg(list(frames), quality=quality)
    return [frame.detach().cpu().numpy().tobytes() for frame in encoded]


def decode_jpeg_rgb(data: bytes) -> np.ndarray:
    """Decode one JPEG image to a contiguous uint8 RGB array."""
    Image = _load_pillow_image()
    with Image.open(io.BytesIO(data)) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(arr)


class _MjpegFrameHub:
    """Fan out JPEG frames to all connected browser clients."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._subscribers: set[queue.Queue[bytes]] = set()
        self._latest: bytes | None = None
        self._frames_published = 0

    @property
    def frames_published(self) -> int:
        with self._cond:
            return self._frames_published

    def subscribe(self) -> queue.Queue[bytes]:
        q: queue.Queue[bytes] = queue.Queue(maxsize=8)
        with self._cond:
            if self._latest is not None:
                q.put_nowait(self._latest)
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[bytes]) -> None:
        with self._cond:
            self._subscribers.discard(q)

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._latest = jpeg
            self._frames_published += 1
            subscribers = list(self._subscribers)

        for q in subscribers:
            try:
                q.put_nowait(jpeg)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(jpeg)
                except queue.Full:
                    pass


class StreamingViewer:
    """Paced HTTP MJPEG viewer for generated upsampler frames."""

    def __init__(
        self,
        host: str,
        port: int,
        jpeg_quality: int,
        jpeg_backend: str,
        chunk_queue_depth: int,
        max_fps: float,
        frame_stride: int,
    ) -> None:
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.jpeg_backend = jpeg_backend
        self.chunk_queue_depth = max(1, int(chunk_queue_depth))
        self.max_fps = float(max_fps)
        self.frame_stride = max(1, int(frame_stride))
        self.original_hub = _MjpegFrameHub()
        self.upscaled_hub = _MjpegFrameHub()
        self.hub = self.upscaled_hub
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._playback_queues: dict[str, queue.Queue] = {
            "original": queue.Queue(maxsize=self.chunk_queue_depth),
            "upscaled": queue.Queue(maxsize=self.chunk_queue_depth),
        }
        self._playback_threads: list[threading.Thread] = []

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("", "0.0.0.0", "::") else self.host
        port = self.port
        if self._server is not None:
            port = int(self._server.server_address[1])
        return f"http://{host}:{port}/"

    def start(self) -> None:
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                logger.debug("viewer: {}", format % args)

            def do_GET(self) -> None:
                if self.path in ("", "/"):
                    self._send_index()
                elif self.path in ("/stream.mjpg", "/stream/upscaled.mjpg"):
                    self._send_stream(viewer.upscaled_hub)
                elif self.path == "/stream/original.mjpg":
                    self._send_stream(viewer.original_hub)
                elif self.path == "/healthz":
                    self._send_health()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def _send_index(self) -> None:
                body = b"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FlashVSR Stream</title>
  <style>
    html, body { margin: 0; min-height: 100%; background: #111; color: #eee; font-family: system-ui, sans-serif; }
    body { display: grid; place-items: stretch; }
    main { display: grid; grid-template-rows: auto 1fr; min-height: 100vh; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 10px 14px; color: #bbb; }
    h1 { margin: 0; font-size: 14px; font-weight: 600; color: #ddd; }
    button { appearance: none; border: 1px solid #555; background: #222; color: #eee; border-radius: 6px; padding: 7px 12px; font: inherit; cursor: pointer; }
    button:hover { background: #2b2b2b; border-color: #777; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; background: #333; min-height: 0; }
    .pane { display: grid; grid-template-rows: auto 1fr; min-width: 0; min-height: 0; background: #080808; }
    .label { padding: 8px 10px; font-size: 12px; color: #aaa; background: #181818; border-bottom: 1px solid #333; }
    .frame { position: relative; display: grid; place-items: center; min-height: 0; overflow: auto; }
    img, canvas { display: block; grid-area: 1 / 1; width: 100%; height: auto; background: #000; }
    .live { z-index: 1; }
    canvas { display: none; }
    .screenshot { visibility: hidden; pointer-events: none; z-index: 2; }
    .pane.screenshot-active .screenshot { visibility: visible; pointer-events: auto; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>FlashVSR stream viewer</h1>
      <button id="screenshot" type="button">Screenshot</button>
    </header>
    <section id="viewer" class="grid">
      <article class="pane">
        <div class="label">Received</div>
        <div class="frame">
          <img class="live" src="/stream/original.mjpg" alt="Received frame stream">
          <img class="screenshot" alt="Received screenshot">
          <canvas></canvas>
        </div>
      </article>
      <article class="pane">
        <div class="label">Upscaled</div>
        <div class="frame">
          <img class="live" src="/stream/upscaled.mjpg" alt="Upscaled frame stream">
          <img class="screenshot" alt="Upscaled screenshot">
          <canvas></canvas>
        </div>
      </article>
    </section>
  </main>
  <script>
    const button = document.getElementById('screenshot');
    const panes = Array.from(document.querySelectorAll('.pane'));
    let showing_screenshot = false;

    function capture_pane(pane) {
      const img = pane.querySelector('.live');
      const overlay = pane.querySelector('.screenshot');
      const canvas = pane.querySelector('canvas');
      const width = img.naturalWidth || img.clientWidth;
      const height = img.naturalHeight || img.clientHeight;
      if (!width || !height) return;
      canvas.width = width;
      canvas.height = height;
      canvas.getContext('2d').drawImage(img, 0, 0, width, height);
      overlay.src = canvas.toDataURL('image/png');
      pane.classList.add('screenshot-active');
    }

    function discard_screenshot(pane) {
      const overlay = pane.querySelector('.screenshot');
      pane.classList.remove('screenshot-active');
      overlay.removeAttribute('src');
    }

    button.addEventListener('click', () => {
      showing_screenshot = !showing_screenshot;
      button.textContent = showing_screenshot ? 'Resume' : 'Screenshot';
      panes.forEach(showing_screenshot ? capture_pane : discard_screenshot);
    });
  </script>
</body>
</html>
"""
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_health(self) -> None:
                body = (
                    "ok "
                    f"original_frames_published={viewer.original_hub.frames_published} "
                    f"upscaled_frames_published={viewer.upscaled_hub.frames_published}\n"
                ).encode("ascii")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=ascii")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_stream(self, hub: _MjpegFrameHub) -> None:
                q = hub.subscribe()
                boundary = b"frame"
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                try:
                    while True:
                        try:
                            jpeg = q.get(timeout=30.0)
                        except queue.Empty:
                            continue
                        self.wfile.write(b"--" + boundary + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
                finally:
                    hub.unsubscribe(q)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="UpsamplerViewerHTTP",
            daemon=True,
        )
        self._thread.start()
        self._playback_threads = [
            threading.Thread(
                target=self._playback_loop,
                args=("original", self.original_hub),
                name="UpsamplerViewerOriginalPlayback",
                daemon=True,
            ),
            threading.Thread(
                target=self._playback_loop,
                args=("upscaled", self.upscaled_hub),
                name="UpsamplerViewerUpscaledPlayback",
                daemon=True,
            ),
        ]
        for thread in self._playback_threads:
            thread.start()
        logger.info("Viewer listening on {}", self.url)

    def _playback_loop(self, channel: str, hub: _MjpegFrameHub) -> None:
        playback_queue = self._playback_queues[channel]
        next_frame_at = time.perf_counter()
        while True:
            item = playback_queue.get()
            if item is _VIEWER_STOP:
                return
            assert isinstance(item, _ViewerPlaybackChunk)
            frames = item.frames
            jpegs = item.jpegs
            frame_count = (
                len(jpegs)
                if jpegs is not None
                else (int(frames.shape[0]) if frames is not None else 0)
            )
            if frame_count == 0:
                continue

            elapsed_s = max(item.elapsed_ms / 1000.0, 1e-3)
            frame_interval_s = elapsed_s / max(1, frame_count)
            if self.max_fps > 0:
                frame_interval_s = max(frame_interval_s, 1.0 / self.max_fps)

            now = time.perf_counter()
            if next_frame_at < now:
                next_frame_at = now

            for idx in range(frame_count):
                if jpegs is not None:
                    jpeg = jpegs[idx]
                else:
                    assert frames is not None
                    jpeg = _encode_jpeg_rgb(frames[idx], self.jpeg_quality)
                delay = next_frame_at - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                hub.publish(jpeg)
                next_frame_at += frame_interval_s

    def enqueue_original_chunk(self, frames: np.ndarray, elapsed_ms: float) -> None:
        item = _ViewerPlaybackChunk(
            elapsed_ms=max(1.0, float(elapsed_ms)),
            frames=np.ascontiguousarray(frames),
        )
        self._enqueue_playback_item("original", item)

    def enqueue_upscaled_chunk(self, frames: np.ndarray, elapsed_ms: float) -> None:
        item = _ViewerPlaybackChunk(
            elapsed_ms=max(1.0, float(elapsed_ms)),
            frames=np.ascontiguousarray(frames),
        )
        self._enqueue_playback_item("upscaled", item)

    def enqueue_chunk(self, frames: np.ndarray, elapsed_ms: float) -> None:
        """Compatibility shim for older upscaled-frame call sites."""
        self.enqueue_upscaled_chunk(frames, elapsed_ms)

    def enqueue_upscaled_jpeg_chunk(
        self, jpegs: list[bytes], elapsed_ms: float
    ) -> None:
        item = _ViewerPlaybackChunk(
            elapsed_ms=max(1.0, float(elapsed_ms)),
            jpegs=jpegs,
        )
        self._enqueue_playback_item("upscaled", item)

    def enqueue_jpeg_chunk(self, jpegs: list[bytes], elapsed_ms: float) -> None:
        """Compatibility shim for older upscaled-JPEG call sites."""
        self.enqueue_upscaled_jpeg_chunk(jpegs, elapsed_ms)

    def _enqueue_playback_item(self, channel: str, item: _ViewerPlaybackChunk) -> None:
        playback_queue = self._playback_queues[channel]
        while True:
            try:
                playback_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    playback_queue.get_nowait()
                    logger.warning(
                        "Viewer {} playback queue full; dropped oldest completed chunk",
                        channel,
                    )
                except queue.Empty:
                    pass

    def publish_chunk(self, frames: np.ndarray) -> None:
        """Compatibility shim for older call sites."""
        for frame in frames:
            self.upscaled_hub.publish(_encode_jpeg_rgb(frame, self.jpeg_quality))

    def stop(self) -> None:
        for playback_queue in self._playback_queues.values():
            try:
                playback_queue.put_nowait(_VIEWER_STOP)
            except queue.Full:
                try:
                    playback_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    playback_queue.put_nowait(_VIEWER_STOP)
                except queue.Full:
                    pass
        for thread in self._playback_threads:
            thread.join(timeout=5.0)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
