# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVENC hardware H.264 encoder implementation.

Isolated from :mod:`flashdreams.serving.webrtc.encoders` so that importing
the shared encoder surface does not drag ``PyNvVideoCodec`` in with it —
that library's import has global side effects (CUDA driver init,
shared-library loading, potential ``atexit`` hooks) that processes not
intending to allocate an NVENC session should not pay. Processes on the
software path — tests, integrations that do not opt in to hardware
encoding — never trigger this module's import and never pay those side
effects.

:func:`~flashdreams.serving.webrtc.encoders.select_encoder` probes
availability without importing via ``importlib.util.find_spec`` and only
imports this module when a hardware encoder is about to be constructed.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import torch
from aiortc import MediaStreamTrack
from av.packet import Packet
from loguru import logger
from torch import Tensor

from flashdreams.runtime import StepResult
from flashdreams.serving.webrtc.encoders import ChunkDeliveryResult

# Runtime imports ``PyNvVideoCodec`` unconditionally (the isolation
# rationale is in the module docstring). The ``TYPE_CHECKING`` branch
# hides the untyped module from ty, which otherwise flags every
# ``nvc.<attr>`` access as unresolved.
if TYPE_CHECKING:
    nvc: Any = None
    from flashdreams.serving.webrtc.media import NVENCVideoTrack
else:
    import PyNvVideoCodec as nvc


# H.264 NAL type identifiers used when inspecting Annex-B bitstreams.
_H264_NAL_TYPE_IDR = 5
_H264_NAL_TYPE_SPS = 7
_H264_NAL_TYPE_PPS = 8

# RTP video clock, per RFC 6184. aiortc's ``H264Encoder.pack()`` rescales
# from whatever ``time_base`` the packet carries into this base, so setting
# ``time_base = 1 / _RTP_VIDEO_CLOCK`` on emitted packets is the
# lowest-conversion choice.
_RTP_VIDEO_CLOCK = 90_000


@dataclass(frozen=True, slots=True)
class NVENCChunkPayload:
    """Encoder-owned CUDA frames prepared before async delivery is scheduled."""

    frames: Tensor


def _payload_contains_nal_type(payload: bytes, nal_type: int) -> bool:
    """Scan an Annex-B H.264 payload for the presence of a specific NAL type."""
    i = 0
    while True:
        idx = payload.find(b"\x00\x00\x01", i)
        if idx < 0:
            return False
        nal_start = idx + 3
        if nal_start >= len(payload):
            return False
        if (payload[nal_start] & 0x1F) == nal_type:
            return True
        i = nal_start + 1


def _result_to_abgr_frames(result: StepResult) -> torch.Tensor:
    """Convert a declared video result to NVENC-``ABGR``-formatted frames.

    The result layout selects the time, channel, batch, and view axes; tensor
    rank is never used to guess the model's output contract. The returned
    contiguous ``[T, H, W, 4]`` uint8 tensor stays on the source device.

    **NVENC ``NV_ENC_BUFFER_FORMAT_ABGR`` is a word-ordered token, not
    memory-ordered.** From ``nvEncodeAPI.h``: "a pixel is represented by
    a 32-bit word with R in the lowest 8 bits, G in the next 8 bits, B
    in the 8 bits after that and A in the highest 8 bits" (word
    ``0xAABBGGRR``). In little-endian memory that is the byte sequence
    ``[R, G, B, A]`` — so the channel-last tensor we hand to the encoder
    must have channel 0 = R, 1 = G, 2 = B, 3 = A. Writing ``[A, B, G, R]``
    (the naive memory-order reading of the name) makes NVENC interpret
    the alpha byte as R, producing a visible RGB↔BGR swap on the wire.

    ABGR (rather than NV12) is chosen so NVENC's driver-side RGB→YUV
    conversion handles the colour transform, sparing us a bespoke NV12
    kernel.
    """
    rgb = result.video_hwc_uint8()
    t, h, w, _ = rgb.shape
    a = torch.full((t, h, w, 1), 255, dtype=torch.uint8, device=rgb.device)
    # Channel-last [R, G, B, A] → little-endian bytes [R, G, B, A] →
    # NVENC word 0xAABBGGRR = NV_ENC_BUFFER_FORMAT_ABGR.
    rgba = torch.cat([rgb, a], dim=-1)
    return rgba.contiguous()


class PyNvHardwareEncoder:
    """NVENC H.264 encoder backed by ``PyNvVideoCodec`` 2.1.

    Accepts CUDA tensors and emits Annex-B H.264 packets streamed onto an
    :class:`~flashdreams.serving.webrtc.media.NVENCVideoTrack` as they are
    encoded. Packets carry ``pts`` on the RTP 90 kHz video clock so
    aiortc's ``H264Encoder.pack()`` can rescale without loss.
    """

    prefers_codec: str | None = "h264"
    backend = "pynvvideocodec"

    @classmethod
    def is_supported(
        cls,
        *,
        gpu_id: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> tuple[bool, str]:
        """Query the driver for NVENC H.264 support at the target resolution.

        Uses ``PyNvVideoCodec.GetEncoderCaps`` (public API) so that no
        NVENC session is allocated for the probe itself. Returns
        ``(True, "")`` when the environment supports NVENC H.264 at the
        requested resolution, else ``(False, human_reason)`` where
        ``human_reason`` is a diagnostic suitable for logging.
        """
        try:
            # PyNvVideoCodec 2.1 signature: GetEncoderCaps(gpuid=<int>, codec=<str>).
            # Keyword is ``gpuid`` (no underscore); positional order is
            # (gpuid, codec).
            caps = nvc.GetEncoderCaps(gpuid=gpu_id, codec="h264")
        except Exception as exc:
            return False, (
                f"GetEncoderCaps gpuid={gpu_id} raised {type(exc).__name__}: {exc}"
            )
        if not caps:
            return False, (f"GetEncoderCaps gpuid={gpu_id} returned no capabilities")
        max_w = int(caps.get("width_max", 0) or 0)
        max_h = int(caps.get("height_max", 0) or 0)
        min_w = int(caps.get("width_min", 0) or 0)
        min_h = int(caps.get("height_min", 0) or 0)
        if width > 0 and height > 0:
            if max_w > 0 and max_h > 0 and (width > max_w or height > max_h):
                return False, (
                    f"requested resolution {width}x{height} exceeds driver "
                    f"maximum {max_w}x{max_h} (gpuid={gpu_id})"
                )
            if (min_w > 0 or min_h > 0) and (width < min_w or height < min_h):
                return False, (
                    f"requested resolution {width}x{height} below driver "
                    f"minimum {min_w}x{min_h} (gpuid={gpu_id})"
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
        gop: int = 30,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {width}x{height}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if bitrate <= 0:
            raise ValueError(f"bitrate must be > 0, got {bitrate}")
        if gop <= 0:
            raise ValueError(f"gop must be > 0, got {gop}")

        # Fail fast with a driver-specific diagnostic before allocating a
        # session slot. When called via ``select_encoder`` this is
        # redundant with the factory's own probe, but a direct instantiation
        # (e.g. from a test) still gets a clear reason.
        supported, reason = self.is_supported(
            gpu_id=gpu_id,
            width=width,
            height=height,
        )
        if not supported:
            raise RuntimeError(f"NVENC H.264 not supported: {reason}")

        self.fps = fps
        self._width = width
        self._height = height
        self._bitrate = bitrate
        self._gpu_id = gpu_id
        self._gop = gop
        self._pts_counter = 0
        self._time_base = Fraction(1, _RTP_VIDEO_CLOCK)
        # ``FORCEIDR`` is exposed by PyNvVideoCodec as an integer flag
        # bitmask. Cache it so we don't hit ``getattr`` on every frame.
        self._force_idr_flag = int(nvc.FORCEIDR)

        # ``repeatspspps=1`` prepends SPS+PPS to every IDR: aiortc's
        # ``H264Encoder.pack()`` does not synthesize parameter sets, so the
        # RTP stream must carry them in-band or the receiver cannot lock on.
        # ``bf=0`` and ``lookahead=0`` keep the output strictly 1:1 with
        # input frames, which is what interactive streaming needs.
        self._encoder = nvc.CreateEncoder(
            width=width,
            height=height,
            fmt="ABGR",
            usecpuinputbuffer=False,
            codec="h264",
            preset="P4",
            tuning_info="ultra_low_latency",
            rc="cbr",
            fps=fps,
            bitrate=bitrate,
            bf=0,
            lookahead=0,
            repeatspspps=1,
            idrperiod=gop,
        )
        logger.info(
            "Video encoder ready: backend={} codec=h264 {}x{}@{}fps "
            "bitrate={}bps gop={} gpu={}",
            self.backend,
            width,
            height,
            fps,
            bitrate,
            gop,
            gpu_id,
        )

    def create_track(self, *, maxsize: int) -> NVENCVideoTrack:
        # Lazy import to avoid an ``nvenc`` ↔ ``media`` cycle.
        from flashdreams.serving.webrtc.media import NVENCVideoTrack

        return NVENCVideoTrack(fps=self.fps, maxsize=maxsize)

    def prepare_chunk_payload(
        self,
        result: StepResult,
        track: MediaStreamTrack,
    ) -> NVENCChunkPayload:
        from flashdreams.serving.webrtc.media import NVENCVideoTrack

        if not isinstance(track, NVENCVideoTrack):
            raise TypeError(
                "PyNvHardwareEncoder requires an NVENCVideoTrack; got "
                f"{type(track).__name__}. Create it via encoder.create_track()."
            )
        return NVENCChunkPayload(frames=_result_to_abgr_frames(result))

    async def deliver_prepared_chunk(
        self,
        payload: object,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        if not isinstance(payload, NVENCChunkPayload):
            raise TypeError("PyNvHardwareEncoder payload must be an NVENCChunkPayload.")
        return await self._deliver_prepared_frames(
            payload.frames,
            track,
            force_keyframe=force_keyframe,
        )

    async def deliver_chunk(
        self,
        result: StepResult,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        from flashdreams.serving.webrtc.media import NVENCVideoTrack

        if not isinstance(track, NVENCVideoTrack):
            raise TypeError(
                "PyNvHardwareEncoder requires an NVENCVideoTrack; got "
                f"{type(track).__name__}. Create it via encoder.create_track()."
            )
        loop = asyncio.get_running_loop()
        emitted = 0
        enqueued = 0

        def _stream(packet: Packet) -> None:
            nonlocal emitted, enqueued
            emitted += 1
            enqueue = track.enqueue_encoded_packet(packet)
            try:
                future = asyncio.run_coroutine_threadsafe(
                    enqueue,
                    loop,
                )
            except RuntimeError:
                enqueue.close()
                return
            try:
                accepted = future.result()
            except Exception:
                return
            if accepted:
                enqueued += 1

        _num_frames, num_keyframes, encode_ms = await asyncio.to_thread(
            self.encode_chunk_sync,
            result,
            force_keyframe=force_keyframe,
            on_packet=_stream,
        )
        if enqueued < emitted:
            logger.debug(
                "NVENC track closed while enqueueing encoded chunk; "
                "enqueued {} of {} packet(s).",
                enqueued,
                emitted,
            )
        return ChunkDeliveryResult(
            backend=self.backend,
            num_frames=enqueued,
            num_keyframes=num_keyframes,
            encode_ms=encode_ms,
        )

    async def _deliver_prepared_frames(
        self,
        frames: Tensor,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        from flashdreams.serving.webrtc.media import NVENCVideoTrack

        if not isinstance(track, NVENCVideoTrack):
            raise TypeError(
                "PyNvHardwareEncoder requires an NVENCVideoTrack; got "
                f"{type(track).__name__}. Create it via encoder.create_track()."
            )
        loop = asyncio.get_running_loop()
        emitted = 0
        enqueued = 0

        def _stream(packet: Packet) -> None:
            nonlocal emitted, enqueued
            emitted += 1
            enqueue = track.enqueue_encoded_packet(packet)
            try:
                future = asyncio.run_coroutine_threadsafe(
                    enqueue,
                    loop,
                )
            except RuntimeError:
                enqueue.close()
                return
            try:
                accepted = future.result()
            except Exception:
                return
            if accepted:
                enqueued += 1

        _num_frames, num_keyframes, encode_ms = await asyncio.to_thread(
            self.encode_frames_sync,
            frames,
            force_keyframe=force_keyframe,
            on_packet=_stream,
        )
        if enqueued < emitted:
            logger.debug(
                "NVENC track closed while enqueueing encoded chunk; "
                "enqueued {} of {} packet(s).",
                enqueued,
                emitted,
            )
        return ChunkDeliveryResult(
            backend=self.backend,
            num_frames=enqueued,
            num_keyframes=num_keyframes,
            encode_ms=encode_ms,
        )

    def encode_chunk_sync(
        self,
        result: StepResult,
        *,
        force_keyframe: bool = False,
        on_packet: Callable[[Packet], None] | None = None,
    ) -> tuple[int, int, float]:
        """Encode a result and return frame, keyframe, and timing counts.

        Kept public because callers (e.g. tests) that already run on a
        worker thread should not have to route through :meth:`deliver_chunk`
        just to get access to the emitted packets.
        """
        frames = _result_to_abgr_frames(result)
        return self.encode_frames_sync(
            frames,
            force_keyframe=force_keyframe,
            on_packet=on_packet,
        )

    def encode_frames_sync(
        self,
        frames: Tensor,
        *,
        force_keyframe: bool = False,
        on_packet: Callable[[Packet], None] | None = None,
    ) -> tuple[int, int, float]:
        """Encode preconverted ``ABGR`` frames for prepared async delivery."""
        if not frames.is_cuda:
            raise ValueError("expected CUDA tensor for hardware encode path")
        num_frames = frames.shape[0]
        num_keyframes = 0
        start_s = time.perf_counter()
        for i in range(num_frames):
            frame = frames[i].contiguous()
            if force_keyframe and i == 0:
                bs = self._encoder.Encode(frame, self._force_idr_flag)
            else:
                bs = self._encoder.Encode(frame)
            if not bs:
                continue
            payload = bytes(bs)
            packet = Packet(payload)
            packet.pts = (self._pts_counter * _RTP_VIDEO_CLOCK) // self.fps
            packet.time_base = self._time_base
            self._pts_counter += 1
            if _payload_contains_nal_type(payload, _H264_NAL_TYPE_IDR):
                num_keyframes += 1
            if on_packet is not None:
                on_packet(packet)
        encode_ms = (time.perf_counter() - start_s) * 1000.0
        return num_frames, num_keyframes, encode_ms

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._encoder.EndEncode()
