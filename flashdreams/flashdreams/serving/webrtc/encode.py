# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated H.264 video encoding for WebRTC via PyNvVideoCodec (NVENC).

This module provides:
- ``tensor_chunk_to_abgr_cuda_frames``: dtype-aware RGB→ABGR conversion on GPU
- ``PyNvVideoCodecH264ChunkEncoder``: NVENC H.264 chunk encoder
- ``EncodedVideoPacket`` / ``ChunkEncodingResult``: encoded output dataclasses
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

try:
    import PyNvVideoCodec as nvc  # type: ignore[import-untyped]
except ImportError as _exc:
    raise ImportError(
        "PyNvVideoCodec is required for GPU video encoding. "
        "Install it with: pip install pynvvideocodec"
    ) from _exc


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
    """Result of encoding a video chunk."""

    packets: list[EncodedVideoPacket]
    backend: str
    encode_ms: float
    num_input_frames: int
    num_keyframes: int


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


class PyNvVideoCodecH264ChunkEncoder:
    """NVENC H.264 encoder for video chunks using PyNvVideoCodec.

    Accepts ``[B, V, T, C, H, W]`` CUDA tensors (uint8 or float) and
    produces H.264 bitstream packets suitable for WebRTC transmission.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        gpu_id: int,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {width}x{height}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if bitrate <= 0:
            raise ValueError(f"bitrate must be > 0, got {bitrate}")

        self.backend = "pynvvideocodec"
        self._width = width
        self._height = height
        self._fps = fps
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
            fps=self._fps,
            bitrate=self._bitrate,
            bf=0,
            lookahead=0,
            repeatspspps=1,
        )

    def encode_chunk(
        self,
        video_chunk: torch.Tensor,
        *,
        force_keyframe: bool = False,
    ) -> ChunkEncodingResult:
        """Encode a video chunk to H.264 packets.

        Args:
            video_chunk: ``[B, V, T, C, H, W]`` tensor on CUDA (uint8 or float).
            force_keyframe: If True, the first frame of the chunk is encoded as an IDR frame.

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
            packets.append(
                EncodedVideoPacket(payload=payload, keyframe=is_keyframe)
            )

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
