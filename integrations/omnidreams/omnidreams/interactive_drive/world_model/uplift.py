# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import io
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Literal

import grpc
import numpy as np
from loguru import logger

FrameEncoding = Literal["jpeg", "raw"]


@dataclass(frozen=True)
class UpliftStreamConfig:
    server: str = "127.0.0.1:8090"
    scale: int = 4
    sparse_ratio: float = 0.0
    input_format: FrameEncoding = "jpeg"
    input_jpeg_quality: int = 90
    return_frames: bool = False
    max_queue_chunks: int = 4
    max_message_mb: int = 512
    connect_timeout_s: float = 5.0


@dataclass(frozen=True)
class _UpliftChunk:
    chunk_index: int
    frames: tuple[object, ...]


class UpliftStreamClient:
    """Best-effort async sender from interactive-drive chunks to FlashVSR uplift.

    The demo does not consume upsampled frames yet. By default requests are
    ``display_only`` so the FlashVSR server's HTTP viewer owns presentation.
    The response loop still runs so gRPC backpressure and server errors are
    observed, and ``return_frames=True`` is kept as the future integration hook.
    """

    def __init__(self, config: UpliftStreamConfig) -> None:
        if config.scale not in (2, 4):
            raise ValueError(f"Uplift scale must be 2 or 4, got {config.scale}")
        if config.input_format not in ("jpeg", "raw"):
            raise ValueError(
                f"Uplift input format must be 'jpeg' or 'raw', got {config.input_format!r}"
            )
        if not 1 <= config.input_jpeg_quality <= 100:
            raise ValueError("Uplift JPEG quality must be between 1 and 100")
        self._config = config
        self._queue: queue.Queue[_UpliftChunk | None] = queue.Queue(
            maxsize=max(1, int(config.max_queue_chunks))
        )
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._next_chunk_index = 0
        self._queued = 0
        self._sent = 0
        self._received = 0
        self._dropped = 0
        self._last_error: str | None = None
        self._last_elapsed_ms: float | None = None
        self._last_response_chunk: int | None = None
        self._connected = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="interactive_drive-uplift-stream",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frames: list[object]) -> str:
        """Queue one generated world-model chunk without blocking rendering."""
        if self._thread is None:
            self.start()
        chunk = _UpliftChunk(
            chunk_index=self._next_chunk_index,
            frames=tuple(frames),
        )
        self._next_chunk_index += 1
        try:
            self._queue.put_nowait(chunk)
            with self._lock:
                self._queued += 1
        except queue.Full:
            with self._lock:
                self._dropped += 1
            logger.warning(
                "[uplift] drop chunk due to backpressure chunk_index={} "
                "server={} queue_depth={}",
                chunk.chunk_index,
                self._config.server,
                self._queue.maxsize,
            )
        return self.status_message()

    def status_message(self) -> str:
        with self._lock:
            state = "connected" if self._connected else "connecting"
            parts = [
                f"Upsampling {state}",
                f"server={self._config.server}",
                f"scale={self._config.scale}x",
                f"queue={self._queue.qsize()}/{self._queue.maxsize}",
                f"sent={self._sent}",
                f"ack={self._received}",
            ]
            if self._dropped:
                parts.append(f"dropped={self._dropped}")
            if self._last_elapsed_ms is not None:
                parts.append(f"last={self._last_elapsed_ms:.0f}ms")
            if self._last_error:
                parts.append(f"error={self._last_error}")
            return " | ".join(parts)

    def close(self) -> None:
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        try:
            from flashvsr.grpc.protos import flashvsr_pb2 as pb2
            from flashvsr.grpc.protos import flashvsr_pb2_grpc as pb2_grpc
        except ModuleNotFoundError as exc:
            self._set_error(
                "flashdreams-flashvsr is required when upsampling_enabled=true; "
                "install flashdreams-omnidreams with the uplift extra"
            )
            logger.exception("[uplift] unable to import FlashVSR gRPC protos")
            return

        max_bytes = self._config.max_message_mb * 1024 * 1024
        channel = grpc.insecure_channel(
            self._config.server,
            options=[
                ("grpc.max_send_message_length", max_bytes),
                ("grpc.max_receive_message_length", max_bytes),
            ],
        )
        stub = pb2_grpc.FlashVSRStub(channel)
        try:
            status = stub.get_status(
                pb2.StatusRequest(), timeout=self._config.connect_timeout_s
            )
            with self._lock:
                self._connected = bool(status.ready)
                self._last_error = None if status.ready else "server not ready"
            if not status.ready:
                return
            logger.info(
                "[uplift] connected server={} device={} model={}",
                self._config.server,
                status.device,
                status.model_name,
            )

            for response in stub.upscale_video(self._request_iter(pb2)):
                if response.error:
                    self._set_error(response.error)
                    logger.error("[uplift] server error: {}", response.error)
                    continue
                with self._lock:
                    self._received += 1
                    self._last_elapsed_ms = float(response.elapsed_ms)
                    self._last_response_chunk = int(response.chunk_index)
                if self._config.return_frames and response.frames_rgb:
                    # Future hook: this is where upsampled frames can be routed
                    # back into the demo. For now the FlashVSR HTTP viewer owns
                    # display, so we only drain responses to keep flow healthy.
                    pass
        except grpc.RpcError as exc:
            self._set_error(_grpc_error_details(exc))
            logger.exception("[uplift] gRPC stream failed")
        except Exception as exc:
            self._set_error(str(exc))
            logger.exception("[uplift] stream failed")
        finally:
            with self._lock:
                self._connected = False
            channel.close()

    def _request_iter(self, pb2: object):
        session_id = str(uuid.uuid4())
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return
            try:
                request = self._build_request(pb2, item, session_id=session_id)
            except Exception as exc:
                self._set_error(str(exc))
                logger.exception(
                    "[uplift] failed to build request chunk_index={}",
                    item.chunk_index,
                )
                continue
            with self._lock:
                self._sent += 1
                self._last_error = None
            yield request

    def _build_request(
        self, pb2: object, item: _UpliftChunk, *, session_id: str
    ) -> object:
        frame_data = _frames_to_numpy(item.frames)
        num_frames, height, width, _channels = frame_data.shape
        req = pb2.UpscaleChunkRequest(
            session_id=session_id,
            chunk_index=item.chunk_index,
            num_frames=num_frames,
            height=height,
            width=width,
            display_only=not self._config.return_frames,
        )
        if self._config.input_format == "jpeg":
            req.frame_encoding = pb2.FRAME_ENCODING_JPEG
            req.frames_jpeg.extend(
                _encode_jpeg_frames(frame_data, self._config.input_jpeg_quality)
            )
        else:
            req.frame_encoding = pb2.FRAME_ENCODING_RAW_RGB
            req.frames_rgb = frame_data.tobytes()

        if item.chunk_index == 0:
            req.input_height = height
            req.input_width = width
            req.scale = self._config.scale
            req.sparse_ratio = self._config.sparse_ratio
        return req

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message


def _frames_to_numpy(frames: tuple[object, ...]) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot send an empty chunk to uplift")
    arrays = [
        np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[..., :3])
        for frame in frames
    ]
    return np.ascontiguousarray(np.stack(arrays, axis=0))


def _encode_jpeg_frames(frames: np.ndarray, quality: int) -> list[bytes]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Uplift JPEG input requires Pillow") from exc

    encoded: list[bytes] = []
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(frame)).save(
            buf, format="JPEG", quality=quality
        )
        encoded.append(buf.getvalue())
    return encoded


def _grpc_error_details(exc: grpc.RpcError) -> str:
    details = getattr(exc, "details", None)
    if callable(details):
        try:
            return str(details())
        except Exception:
            pass
    return str(exc)
