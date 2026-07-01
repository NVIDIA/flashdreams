# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video encoding backends for WebRTC.

This module exposes two encoder implementations that share the
``VideoEncoder`` interface:

- ``PyNvHardwareEncoder``: GPU-accelerated NVENC H.264 (via PyNvVideoCodec).
- ``DefaultRTCVideoEncoder``: aiortc/PyAV built-in software encoder.

The session picks one or the other at initialization; downstream code
uses the encoder solely through :meth:`VideoEncoder.create_track` and
:meth:`VideoEncoder.deliver_chunk`, so the branch on backend lives in
exactly one place.

Also exposed:
- ``tensor_chunk_to_abgr_cuda_frames``: RGB→ABGR conversion on GPU.
- ``EncodedVideoPacket`` / ``ChunkEncodingResult`` / ``ChunkDeliveryResult``: output dataclasses.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from aiortc import MediaStreamTrack
from loguru import logger

if TYPE_CHECKING:
    from flashdreams.serving.webrtc.media import (
        BufferedVideoTrack,
        EncodedPacketVideoTrack,
    )

# PyNvVideoCodec is an optional dependency: it is required only when the
# NVENC hardware backend (``PyNvHardwareEncoder``) is actually selected.
# End-user environments without NVENC hardware or without the library
# installed should still be able to import this module and use the
# software backend. Attempted use of the hardware encoder without the
# library installed raises at instantiation, not at import.
try:
    import PyNvVideoCodec as nvc  # type: ignore[import-untyped]
    _PYNVVIDEOCODEC_AVAILABLE = True
except ImportError:
    nvc = None  # type: ignore[assignment]
    _PYNVVIDEOCODEC_AVAILABLE = False


def is_nvenc_available() -> bool:
    """Return True if the PyNvVideoCodec library is importable.

    This is a *library availability* check, not a hardware check: the
    GPU/driver may still refuse to create an NVENC session at runtime.
    A definitive check requires instantiating :class:`PyNvHardwareEncoder`
    and observing whether the constructor raises.
    """
    return _PYNVVIDEOCODEC_AVAILABLE


def tensor_chunk_to_abgr_cuda_frames(video_chunk: torch.Tensor) -> list[torch.Tensor]:
    """Convert a ``[B, V, T, C, H, W]`` video tensor to a list of ABGR uint8 CUDA tensors.

    Accepts either:
    - **uint8** ``[0, 255]`` input (already quantized, e.g. OmniDreams)
    - **float** ``[-1, 1]`` input (requires GPU quantization)

    Returns:
        List of ``T`` tensors, each ``[H, W, 4]`` uint8 ABGR on CUDA.
    """
    if video_chunk.ndim != 6:
        raise ValueError(
            f"Expected video chunk with 6 dimensions [B, V, T, C, H, W], "
            f"got {tuple(video_chunk.shape)}"
        )
    if video_chunk.shape[0] < 1 or video_chunk.shape[1] < 1:
        raise ValueError("Video chunk must contain at least one batch and one view.")

    frames_rgb = video_chunk[0, 0]  # [T, 3, H, W]
    if not frames_rgb.is_cuda:
        raise ValueError("video_chunk must be on a CUDA device for PyNvVideoCodec encode.")

    if frames_rgb.dtype == torch.uint8:
        # Already quantized — just reorder to [T, H, W, 3]
        frames_rgb = frames_rgb.permute(0, 2, 3, 1).contiguous()
    else:
        # Float [-1, 1] — quantize on GPU then reorder
        frames_rgb = (
            frames_rgb.detach()
            .to(dtype=torch.float32)
            .clamp_(-1.0, 1.0)
            .add_(1.0)
            .mul_(127.5)
            .round_()
            .to(dtype=torch.uint8)
        )
        frames_rgb = frames_rgb.permute(0, 2, 3, 1).contiguous()

    # PyNvVideoCodec "ABGR" surface format = 32-bit packed pixel with A in
    # MSB and R in LSB.  On little-endian the memory byte order is R,G,B,A.
    alpha = torch.full(
        (frames_rgb.shape[0], frames_rgb.shape[1], frames_rgb.shape[2], 1),
        255,
        dtype=torch.uint8,
        device=frames_rgb.device,
    )
    frames_abgr = torch.cat(
        (
            frames_rgb,  # R, G, B
            alpha,       # A
        ),
        dim=-1,
    ).contiguous()
    return list(frames_abgr.unbind(dim=0))


@dataclass(slots=True, frozen=True)
class EncodedVideoPacket:
    """A single encoded video packet (H.264 NAL unit bitstream)."""

    payload: bytes
    keyframe: bool = False


@dataclass(slots=True, frozen=True)
class ChunkEncodingResult:
    """Result of encoding a video chunk (batch, per-encoder-internal)."""

    packets: list[EncodedVideoPacket]
    backend: str
    encode_ms: float
    num_input_frames: int
    num_keyframes: int


@dataclass(slots=True, frozen=True)
class ChunkDeliveryResult:
    """Result of encoding + delivering a chunk to a MediaStreamTrack.

    Uniform shape across encoder backends; downstream code uses this
    (rather than ``ChunkEncodingResult``) so it does not need to know
    which encoder produced the chunk.
    """

    backend: str
    num_frames: int
    num_keyframes: int
    encode_ms: float


@runtime_checkable
class VideoEncoder(Protocol):
    """Encoder backend paired with a compatible :class:`MediaStreamTrack`.

    Each backend owns two responsibilities:

    - Creating a fresh media track sized for one session (``create_track``).
    - Encoding and enqueueing one chunk of frames onto that track
      (``deliver_chunk``).

    The concrete pairing (raw frames + aiortc SW encoder vs. pre-encoded
    H.264 packets + NVENC) is entirely encapsulated inside the
    implementation, so callers pick one backend at startup and branch
    nowhere else.
    """

    fps: int
    backend: str
    prefers_codec: str | None
    """SDP codec preference hint. ``"h264"`` for NVENC (must be honored
    or the pre-encoded bitstream will not decode); ``None`` to use
    aiortc's default codec selection."""

    def create_track(self, *, maxsize: int) -> MediaStreamTrack:
        """Create a fresh session track compatible with this encoder."""
        ...

    async def deliver_chunk(
        self,
        chunk: torch.Tensor,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        """Encode ``chunk`` and enqueue the resulting frames/packets to ``track``."""
        ...

    def close(self) -> None:
        """Release any encoder-owned resources (HW handles, etc.)."""
        ...


_H264_NAL_TYPE_IDR = 5
# Filler data NAL (type 12) — decoders are required to ignore it.
# Used to maintain 1:1 packet-to-frame pacing when NVENC buffers a frame.
_H264_FILLER_NAL = b"\x00\x00\x00\x01\x0c\x80"


def _payload_contains_idr(payload: bytes) -> bool:
    """Check if an H.264 bitstream contains an IDR NAL unit (type 5)."""
    i = 0
    while True:
        i = payload.find(b"\x00\x00\x01", i)
        if i == -1:
            return False
        # Skip past start code; handle 3-byte (00 00 01) and 4-byte (00 00 00 01)
        i += 3
        if i >= len(payload):
            return False
        nal_type = payload[i] & 0x1F
        if nal_type == _H264_NAL_TYPE_IDR:
            return True


class PyNvHardwareEncoder:
    """NVENC H.264 encoder for video chunks using PyNvVideoCodec.

    Implements :class:`VideoEncoder`. Accepts ``[B, V, T, C, H, W]`` CUDA
    tensors (uint8 or float) and produces H.264 bitstream packets that
    are streamed into an :class:`EncodedPacketVideoTrack` as they are
    encoded (no per-chunk batch wait).
    """

    prefers_codec: str | None = "h264"

    @classmethod
    def is_supported(
        cls,
        *,
        gpu_id: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> tuple[bool, str]:
        """Check whether NVENC H.264 is usable on this machine.

        Uses PyNvVideoCodec's ``GetEncoderCaps`` to query the driver
        without allocating an encoder session, so it is safe/cheap to
        call at startup for capability detection.

        Args:
            gpu_id: GPU device index to probe (matches the value that
                would be passed to :class:`PyNvHardwareEncoder`).
            width, height: Optional target resolution. If > 0, the
                probe additionally verifies the requested resolution
                falls within the driver-reported min/max limits.

        Returns:
            ``(True, "")`` if the environment can create an NVENC H.264
            session at the requested resolution; ``(False, reason)``
            otherwise. ``reason`` is a human-readable diagnostic
            suitable for logging.
        """
        if not _PYNVVIDEOCODEC_AVAILABLE:
            return False, "PyNvVideoCodec library is not installed"
        try:
            # PyNvVideoCodec 2.1.0 signature is ``GetEncoderCaps(gpuid=0, codec='h264')``
            # -- note the keyword is ``gpuid`` (no underscore), and the arg
            # order is (gpuid, codec) despite what some docs pages suggest.
            caps = nvc.GetEncoderCaps(gpuid=gpu_id, codec="h264")
        except Exception as exc:
            return False, (
                f"PyNvVideoCodec encoder is not supported "
                f"(GetEncoderCaps gpuid={gpu_id} raised "
                f"{type(exc).__name__}: {exc})"
            )
        if not caps:
            return False, (
                f"PyNvVideoCodec encoder is not supported "
                f"(GetEncoderCaps gpuid={gpu_id} returned no capabilities)"
            )
        max_w = int(caps.get("width_max", 0) or 0)
        max_h = int(caps.get("height_max", 0) or 0)
        min_w = int(caps.get("width_min", 0) or 0)
        min_h = int(caps.get("height_min", 0) or 0)
        if width > 0 and height > 0:
            if max_w > 0 and max_h > 0 and (width > max_w or height > max_h):
                return False, (
                    f"Insufficient capabilities of PyNvVideoCodec encoder "
                    f"(gpuid={gpu_id} max resolution {max_w}x{max_h}, "
                    f"requested {width}x{height})"
                )
            if (min_w > 0 or min_h > 0) and (width < min_w or height < min_h):
                return False, (
                    f"Insufficient capabilities of PyNvVideoCodec encoder "
                    f"(gpuid={gpu_id} min resolution {min_w}x{min_h}, "
                    f"requested {width}x{height})"
                )
        return True, ""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        gpu_id: int,
    ) -> None:
        if not _PYNVVIDEOCODEC_AVAILABLE:
            raise RuntimeError(
                "PyNvVideoCodec is not installed; PyNvHardwareEncoder cannot be used. "
                "Install it with `pip install pynvvideocodec` for GPU H.264 encoding, "
                "or select DefaultRTCVideoEncoder instead."
            )
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {width}x{height}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if bitrate <= 0:
            raise ValueError(f"bitrate must be > 0, got {bitrate}")

        # Query hardware caps before session allocation so unsupported
        # environments (no NVENC, wrong GPU, resolution out of range)
        # surface a clear diagnostic instead of a generic CreateEncoder
        # failure. Not redundant with the CreateEncoder call below:
        # GetEncoderCaps validates driver/device capability without
        # touching the session pool, and lets us include capability info
        # in the success log.
        supported, reason = self.is_supported(
            gpu_id=gpu_id, width=width, height=height
        )
        if not supported:
            raise RuntimeError(
                f"NVENC H.264 encoder not supported on this system: {reason}"
            )

        self.backend = "pynvvideocodec"
        self.fps = fps
        self._width = width
        self._height = height
        self._bitrate = bitrate
        self._gpu_id = gpu_id
        self._force_idr_flag = int(nvc.FORCEIDR)  # type: ignore[attr-defined]
        self._encoder: Any = self._create_encoder()
        logger.info(
            "PyNvVideoCodec H.264 encoder created: {}x{} fps={} bitrate={} "
            "gpu_id={} preset=P4 tuning=ultra_low_latency rc=cbr",
            width, height, fps, bitrate, gpu_id,
        )

    def _create_encoder(self) -> Any:
        return nvc.CreateEncoder(
            self._width,
            self._height,
            "ABGR",
            False,
            gpu_id=self._gpu_id,
            codec="h264",
            preset="P4",
            tuning_info="ultra_low_latency",
            rc="cbr",
            fps=self.fps,
            bitrate=self._bitrate,
            bf=0,
            lookahead=0,
            repeatspspps=1,
        )

    def create_track(self, *, maxsize: int) -> EncodedPacketVideoTrack:
        # Import lazily to sidestep the encode ↔ media module cycle.
        from flashdreams.serving.webrtc.media import EncodedPacketVideoTrack
        return EncodedPacketVideoTrack(fps=self.fps, maxsize=maxsize)

    async def deliver_chunk(
        self,
        chunk: torch.Tensor,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        from flashdreams.serving.webrtc.media import EncodedPacketVideoTrack
        if not isinstance(track, EncodedPacketVideoTrack):
            raise TypeError(
                "PyNvHardwareEncoder requires an EncodedPacketVideoTrack; "
                f"got {type(track).__name__}. Create the track via encoder.create_track()."
            )
        loop = asyncio.get_running_loop()

        def _stream_packet(packet: EncodedVideoPacket) -> None:
            # NVENC encode runs on the to_thread worker; marshal each
            # packet back onto the asyncio loop so recv() sees it as
            # soon as it is produced, without waiting for the whole
            # chunk's batch to finish encoding.
            try:
                loop.call_soon_threadsafe(track.enqueue_encoded_packet_nowait, packet)
            except RuntimeError:
                return

        result = await asyncio.to_thread(
            self.encode_chunk,
            chunk,
            force_keyframe=force_keyframe,
            on_packet=_stream_packet,
        )
        return ChunkDeliveryResult(
            backend=self.backend,
            num_frames=result.num_input_frames,
            num_keyframes=result.num_keyframes,
            encode_ms=result.encode_ms,
        )

    def encode_chunk(
        self,
        video_chunk: torch.Tensor,
        *,
        force_keyframe: bool = False,
        on_packet: Callable[[EncodedVideoPacket], None] | None = None,
    ) -> ChunkEncodingResult:
        """Encode a video chunk to H.264 packets.

        Args:
            video_chunk: ``[B, V, T, C, H, W]`` tensor on CUDA (uint8 or float).
            force_keyframe: If True, the first frame of the chunk is encoded as an IDR frame.
            on_packet: If provided, called synchronously after each frame is
                encoded so the caller can stream packets downstream without
                waiting for the whole chunk. Invoked from the encode worker
                thread; the callback is responsible for any thread marshaling.

        Returns:
            ``ChunkEncodingResult`` with the encoded H.264 packets.
        """
        frames_abgr = tensor_chunk_to_abgr_cuda_frames(video_chunk)
        packets: list[EncodedVideoPacket] = []
        num_keyframes = 0
        start_s = time.perf_counter()
        for frame_index, frame_abgr in enumerate(frames_abgr):
            if force_keyframe and frame_index == 0:
                bitstream = self._encoder.Encode(frame_abgr, self._force_idr_flag)
            else:
                bitstream = self._encoder.Encode(frame_abgr)
            payload = bytes(bitstream)
            if not payload:
                payload = _H264_FILLER_NAL
            is_keyframe = _payload_contains_idr(payload)
            if is_keyframe:
                num_keyframes += 1
            packet = EncodedVideoPacket(payload=payload, keyframe=is_keyframe)
            packets.append(packet)
            if on_packet is not None:
                on_packet(packet)

        encode_ms = (time.perf_counter() - start_s) * 1000.0
        total_bytes = sum(len(p.payload) for p in packets)
        logger.debug(
            "PyNvVideoCodec encode: frames={} packets={} keyframes={} "
            "bytes={} encode_ms={:.2f}",
            len(frames_abgr), len(packets), num_keyframes,
            total_bytes, encode_ms,
        )
        return ChunkEncodingResult(
            packets=packets,
            backend=self.backend,
            encode_ms=encode_ms,
            num_input_frames=len(frames_abgr),
            num_keyframes=num_keyframes,
        )

    def close(self) -> None:
        """Flush the encoder and release resources."""
        logger.debug("PyNvVideoCodec encoder closing ({}x{}).", self._width, self._height)
        with contextlib.suppress(Exception):
            self._encoder.EndEncode()


class DefaultRTCVideoEncoder:
    """Software encoder path via aiortc's built-in PyAV/FFmpeg encoder.

    Implements :class:`VideoEncoder`. This class does not encode itself
    -- it wraps a :class:`BufferedVideoTrack` that carries raw RGB
    frames and lets aiortc's own RTP sender loop drive encoding
    frame-by-frame. The concrete codec is picked internally by aiortc.
    ``deliver_chunk`` therefore reduces to enqueueing the chunk's
    frames onto the track.
    """

    prefers_codec: str | None = None
    backend = "aiortc"

    def __init__(self, *, fps: int) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.fps = fps
        logger.info("FFmpeg software encoder (aiortc) selected: fps={}", fps)

    def create_track(self, *, maxsize: int) -> BufferedVideoTrack:
        from flashdreams.serving.webrtc.media import BufferedVideoTrack
        return BufferedVideoTrack(fps=self.fps, maxsize=maxsize)

    async def deliver_chunk(
        self,
        chunk: torch.Tensor,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        # force_keyframe is a no-op on this path: aiortc's SW encoder
        # decides keyframe cadence internally and responds to receiver
        # PLI/FIR feedback, so an out-of-band request from us is neither
        # possible nor needed.
        del force_keyframe
        from flashdreams.serving.webrtc.media import BufferedVideoTrack
        if not isinstance(track, BufferedVideoTrack):
            raise TypeError(
                "DefaultRTCVideoEncoder requires a BufferedVideoTrack; "
                f"got {type(track).__name__}. Create the track via encoder.create_track()."
            )
        enqueued = await track.enqueue_chunk(chunk)
        return ChunkDeliveryResult(
            backend=self.backend,
            num_frames=enqueued,
            num_keyframes=0,
            encode_ms=0.0,
        )

    def close(self) -> None:
        # No encoder-owned resources; the track is owned by aiortc's PC.
        return
