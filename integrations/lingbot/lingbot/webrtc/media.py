from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import PyNvVideoCodec as nvc  # type: ignore[import-untyped]
import torch
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av.packet import Packet

def tensor_chunk_to_abgr_cuda_frames(video_chunk: torch.Tensor) -> list[torch.Tensor]:
    """Convert Lingbot tensor [B, V, T, C, H, W] to ABGR uint8 CUDA frames."""
    if video_chunk.ndim != 6:
        raise ValueError(
            f"Expected video chunk with 6 dimensions [B, V, T, C, H, W], got {video_chunk.shape}"
        )
    if video_chunk.shape[0] < 1 or video_chunk.shape[1] < 1:
        raise ValueError("Video chunk must contain at least one batch and one view.")

    frames_rgb = video_chunk[0, 0]
    if not frames_rgb.is_cuda:
        raise ValueError("video_chunk must be on CUDA device for PyNvVideoCodec encode.")

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

    alpha = torch.full(
        (
            frames_rgb.shape[0],
            frames_rgb.shape[1],
            frames_rgb.shape[2],
            1,
        ),
        255,
        dtype=torch.uint8,
        device=frames_rgb.device,
    )
    frames_abgr = torch.cat(
        (
            alpha,
            frames_rgb[..., 2:3],
            frames_rgb[..., 1:2],
            frames_rgb[..., 0:1],
        ),
        dim=-1,
    ).contiguous()
    return list(frames_abgr.unbind(dim=0))


@dataclass(slots=True, frozen=True)
class EncodedVideoPacket:
    payload: bytes
    keyframe: bool = False


@dataclass(slots=True, frozen=True)
class ChunkEncodingResult:
    packets: list[EncodedVideoPacket]
    backend: str
    encode_ms: float
    num_input_frames: int
    num_keyframes: int


class PyNvVideoCodecH264ChunkEncoder:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        gpu_id: int,
    ) -> None:
        self.backend = "pynvvideocodec"
        self._width = width
        self._height = height
        self._fps = fps
        self._bitrate = bitrate
        self._gpu_id = gpu_id
        self._force_idr_flag = int(nvc.FORCEIDR) # type: ignore[attr-defined]
        self._encoder = self._create_encoder()

    def _create_encoder(self) -> Any:
        return nvc.CreateEncoder(
            self._width,
            self._height,
            "ABGR",
            False,
            gpu_id=self._gpu_id,
            codec="h264",
            preset="P1",
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
        frames_abgr = tensor_chunk_to_abgr_cuda_frames(video_chunk)
        packets: list[EncodedVideoPacket] = []
        start_s = time.perf_counter()
        for frame_index, frame_abgr in enumerate(frames_abgr):
            if force_keyframe and frame_index == 0:
                bitstream = self._encoder.Encode(frame_abgr, self._force_idr_flag)
            else:
                bitstream = self._encoder.Encode(frame_abgr)
            payload = bytes(bitstream)
            if payload:
                packets.append(
                    EncodedVideoPacket(
                        payload=payload,
                        keyframe=bool(force_keyframe and frame_index == 0),
                    )
                )

        return ChunkEncodingResult(
            packets=packets,
            backend=self.backend,
            encode_ms=(time.perf_counter() - start_s) * 1000.0,
            num_input_frames=len(frames_abgr),
            num_keyframes=1 if (packets and force_keyframe) else 0,
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._encoder.EndEncode()


class LingbotVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, fps: int = 16, queue_max_size: int = 0) -> None:
        super().__init__()
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if queue_max_size < 0:
            raise ValueError("queue_max_size must be >= 0")
        self._fps = fps
        self._time_base = Fraction(1, fps)
        self._frame_interval_s = 1.0 / fps
        self._next_deadline_s: float | None = None
        self._pts = 0
        self._queue_max_size = queue_max_size
        self._frames: asyncio.Queue[EncodedVideoPacket | None]
        self._frames = (
            asyncio.Queue(maxsize=queue_max_size)
            if queue_max_size > 0
            else asyncio.Queue()
        )
        self._dropped_units = 0
        self._closed = False

    @property
    def queue_depth(self) -> int:
        return self._frames.qsize()

    @property
    def dropped_units(self) -> int:
        return self._dropped_units

    async def _enqueue_item(self, item: EncodedVideoPacket) -> None:
        if self._queue_max_size > 0 and self._frames.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._frames.get_nowait()
                self._dropped_units += 1
        await self._frames.put(item)

    async def enqueue_encoded_packets(self, packets: list[EncodedVideoPacket]) -> int:
        for packet in packets:
            await self._enqueue_item(packet)
        return len(packets)

    async def recv(self) -> Packet:
        if self._closed:
            raise MediaStreamError

        item = await self._frames.get()
        if item is None:
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        now_s = loop.time()
        if self._next_deadline_s is None:
            self._next_deadline_s = now_s
        else:
            self._next_deadline_s += self._frame_interval_s
            wait_s = self._next_deadline_s - now_s
            if wait_s > 0:
                await asyncio.sleep(wait_s)

        packet = Packet(item.payload)
        packet.pts = self._pts
        packet.time_base = self._time_base
        self._pts += 1
        return packet

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._frames.put(None)
        self.stop()
